"""Automatic utility check for paper-app grids, plus the static privacy flags: a first pass that SUGGESTS a grade.
Every suggestion is for hand confirmation (docs/grading_rubric.md: grades are given by hand).

  python -m llmdpeval.app_grade GRID [--results OTHER_GRID/results.jsonl] [--hand hand_grades.csv] [--out FILE]

For each app script: parse the final JSON line (keys from applications/<app>/prompt.txt), compare the DP values with
the ORIGINAL analysis, with the reference's 20-run results (applications/<app>/runs/ceiling.json) as the bar, never
with the script's own "true" values. Output (default GRID/app_grades.csv), one row per script:
utility = keeps / loses / unparseable / no_ceiling, the numbers, the privacy_check flags, and a suggested grade:
  N no code; E does not run; W a privacy flag (or Frymer: a significant claim under state fixed effects);
  G utility keeps; U utility loses; ? needs a hand look (unparseable output, stdout not logged).
Stdout comes from a repaired script's .repair.json (last round) if there is one, else from
GRID/results.jsonl (or --results). With --hand (columns cell, task, run, grade) it prints the agreement.

Thresholds: +-0.01 around the reference's 20-run range and the per-app analogues below; each app's spec.md
lists them under its grading notes.
Keys: the prompts ask for `true_original_*` (the existing analysis) and `true_own_*` (the
script's own model without noise); a plain `true_*` key is also read as the original. Grading uses the original analysis;
the own values only show whether a utility loss comes from DP or from a change of model (reported in the detail).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from llmdpeval import privacy_check
from llmdpeval.helper import code_files, code_status, file_key, load_prompt
from llmdpeval.repair import static_problems
from llmdpeval.run_grid import parse_cell
from llmdpeval.sandbox import app_dir

SPD_WIDEN = 0.01        # Gopher: DP SPD inside the reference's 20-run range widened by this
ACC_TOL = 0.01          # Gopher: DP accuracy within this of the original 0.6834
KUL_ALL_WIDEN = 0.01    # Kulynych: overall DP accuracy >= reference min - this
KUL_RACE_WIDEN = 0.02   # Kulynych: per-race DP accuracy >= reference min - this (small groups swing at eps 1)
IMAI_T1_TOL = 0.01      # Imai: Table 1 overall error within this of the original
ALPHA = 0.05
RACES5 = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def gopher(o, c):
    """Gopher SQF: DP SPD inside the reference range and DP accuracy close to the original."""
    acc, spd = o.get("dp_accuracy"), o.get("dp_spd")
    if not (_num(acc) and _num(spd)):
        return "unparseable", "dp_accuracy/dp_spd missing or not numbers"
    s = c["summary"]["dp_spd"]
    lo, hi = s["min"] - SPD_WIDEN, s["max"] + SPD_WIDEN
    oacc = c["true"]["accuracy"]
    ok = spd < 0 and lo <= spd <= hi and abs(acc - oacc) <= ACC_TOL
    return ("keeps" if ok else "loses"), (f"dp_spd {spd:.4f} (bar [{lo:.3f}, {hi:.3f}], original {c['true']['spd']:.4f}); "
                                          f"dp_acc {acc:.4f} (original {oacc:.4f} +- {ACC_TOL})")


def kulynych(o, c):
    """Kulynych Adult: overall and per-race DP accuracy and DP race counts against the reference ranges."""
    acc, cnt = o.get("dp_accuracy"), o.get("dp_race_counts")
    if not isinstance(acc, dict) or not isinstance(cnt, dict) or not all(_num(acc.get(k)) for k in ["All", *RACES5]) \
            or not all(_num(cnt.get(k)) for k in RACES5):
        return "unparseable", "dp_accuracy (All + 5 races) / dp_race_counts (5 races) missing or not numbers"
    s, t = c["summary"], c["true"]
    bad = []
    if acc["All"] < s["dp_accuracy"]["All"]["min"] - KUL_ALL_WIDEN:
        bad.append(f"All {acc['All']:.4f} < {s['dp_accuracy']['All']['min'] - KUL_ALL_WIDEN:.4f}")
    for r in RACES5:
        if not s["dp_accuracy"][r]["min"] - KUL_RACE_WIDEN <= acc[r] <= 1:
            bad.append(f"acc {r} {acc[r]:.4f}")
        q = s["dp_race_counts"][r]
        w = q["max"] - q["min"]
        if not q["min"] - w <= cnt[r] <= q["max"] + w:
            bad.append(f"count {r} {cnt[r]:g} (bar [{q['min'] - w:g}, {q['max'] + w:g}], true {t['race_counts'][r]})")
    detail = f"dp acc All {acc['All']:.4f} (original {t['accuracy']['All']:.4f}, reference median " \
             f"{s['dp_accuracy']['All']['median']:.4f})" + (f"; outside the bar: {', '.join(bad)}" if bad else "")
    return ("loses" if bad else "keeps"), detail


def frymer(o, c):
    """Frymer CCES: the union effect keeps its sign and significance (model 1 and the group difference) and does
    not become significant under state fixed effects (models 2-3). Returns (verdict, detail, significant_under_fe)."""
    try:
        coef, p, se = o["dp_union_coef"], o["dp_union_p"], o["dp_union_se"]
        g, dp_diff_p = o["dp_group_means"], o["dp_diff_p"]
        m1, p1, p2, p3, diff = coef["model1"], p["model1"], p["model2"], p["model3"], g["diff"]
        assert all(_num(x) for x in (m1, p1, p2, p3, diff, dp_diff_p))
    except (KeyError, TypeError, AssertionError):
        return "unparseable", "dp_union_coef/_p/_se (model1-3) or dp_group_means.diff / dp_diff_p missing"
    sig23 = [m for m in ("model2", "model3") if p[m] < ALPHA]
    keeps = m1 < 0 and p1 < ALPHA and diff < 0 and dp_diff_p < ALPHA
    ref_se = c["summary"]["model1.union_member"]["se_noise_aware"]["min"]
    se1 = se.get("model1") if isinstance(se, dict) else None  # the SE only adds a note; a bad value is ignored
    note = (f"; model1 SE {se1:.4f} < reference noise-aware min {ref_se:.4f}: SE may leave out the DP noise"
            if _num(se1) and se1 < 0.9 * ref_se else "")
    detail = (f"model1 coef {m1:.4f} p {p1:.2g}; diff {diff:.4f} p {dp_diff_p:.2g}; model2 p {p2:.2g}, model3 p {p3:.2g}"
              + note + (f"; SIGNIFICANT under state FE ({', '.join(sig23)}): graded W" if sig23 else ""))
    return ("keeps" if keeps and not sig23 else "loses"), detail, bool(sig23)


def zhang(o, c, app):
    """Zhang FAERS: DP Top-100 overlap with the original and, where checkable, the significant set kept."""
    sig, top = o.get("dp_significant"), o.get("dp_top100")
    if not isinstance(sig, list) or not isinstance(top, list):
        return "unparseable", "dp_significant / dp_top100 missing or not lists"
    orig = privacy_check.final_json((app / "runs" / "original_run.txt").read_text(encoding="utf-8"))
    otop = set(orig["top100"])
    s = c["summary"]
    overlap = len(otop & {str(x) for x in top})
    bad = [f"top100 overlap {overlap} < reference min {s['top100_overlap']['min']:g}"] if overlap < s["top100_overlap"]["min"] else []
    # the original's 159 significant AEs are not saved in runs/; a script's own true list is used only when it has
    # 159 items and its true Top-100 equals the original's (both checkable)
    tsig, ttop = true_original(o, "significant"), true_original(o, "top100")
    if isinstance(tsig, list) and len(tsig) == c["true"]["n_significant"] and isinstance(ttop, list) \
            and set(map(str, ttop)) == otop:
        ts, ds = set(map(str, tsig)), set(map(str, sig))
        kept, false = len(ts & ds), len(ds - ts)
        if kept < s["sig_kept_n"]["min"]:
            bad.append(f"significant kept {kept} < reference min {s['sig_kept_n']['min']:g}")
        if false > s["false_sig_n"]["max"]:
            bad.append(f"significant in DP only {false} > reference max {s['false_sig_n']['max']:g}")
        sig_txt = f"significant kept {kept}/{len(ts)}, DP-only {false}"
    else:
        sig_txt = "significant set not checked (original list not saved; script's true list not verifiable)"
    return ("loses" if bad else "keeps"), f"top100 overlap {overlap}; {sig_txt}" + (f"; {', '.join(bad)}" if bad else "")


def imai(o, c):
    """Imai FL: Table 1 overall errors close to the original, name-only still worst, Bayes RMSE below Goodman's."""
    t1, t2 = o.get("dp_table1"), o.get("dp_table2")
    try:
        preds = sorted({k.split("|")[1] for k in c["table1"]})
        err = {pr: t1[pr]["overall_error"] for pr in preds}
        races = sorted({k.split("|")[0].split("_", 1)[1] for k in c["table2"] if k.startswith("district_")})
        dist = {r: (t2[f"district_{r}"]["bayes_rmse"], t2[f"district_{r}"]["goodman_rmse"]) for r in races}
        assert all(_num(v) for v in err.values()) and all(_num(a) and _num(b) for a, b in dist.values())
    except (KeyError, TypeError, AssertionError, AttributeError):
        return "unparseable", "dp_table1 overall_error per prediction / dp_table2 district bayes & goodman RMSE missing"
    bad = [f"T1 {pr} {err[pr]:.4f} vs {c['table1'][f'overall_error|{pr}']['original']:.4f}" for pr in preds
           if abs(err[pr] - c["table1"][f"overall_error|{pr}"]["original"]) > IMAI_T1_TOL]
    if any(err["name"] <= err[pr] for pr in preds if pr != "name"):
        bad.append("T1 name-only no longer the worst")
    bad += [f"district {r}: bayes RMSE {b:.3f} >= goodman {g:.3f}" for r, (b, g) in dist.items() if b >= g]
    def cell(key):  # a precinct value, or None when the script's table lacks it or has another shape
        row = t2.get(key.split("|")[0])
        v = row.get(key.split("|")[1]) if isinstance(row, dict) else None
        return v if _num(v) else None
    inside = sum(1 for k, q in c["table2"].items()
                 if k.startswith("precinct_") and cell(k) is not None and q["min"] <= cell(k) <= q["max"])
    n_prc = sum(1 for k in c["table2"] if k.startswith("precinct_"))
    detail = (f"T1 overall error name {err['name']:.4f}; district bayes<goodman {sum(b < g for b, g in dist.values())}/"
              f"{len(dist)}; precinct cells inside the reference range {inside}/{n_prc} (reported only)")
    return ("loses" if bad else "keeps"), detail + (f"; {'; '.join(bad)}" if bad else "")


def true_original(o: dict, name: str):
    """The original analysis's value: `true_original_<name>`, else the plain key `true_<name>`."""
    return o.get(f"true_original_{name}", o.get(f"true_{name}"))


def own_note(o: dict) -> str:
    """The script's own non-private results (`true_own_*`), shown next to the verdict: a gap between them and the
    original means the analysis changed; a gap between them and the DP values is the cost of DP."""
    own = {k.removeprefix("true_own_"): v for k, v in o.items() if k.startswith("true_own_")}
    return f"; own non-private: {json.dumps(own, default=str)[:200]}" if own else ""


def utility(app: Path, obj: dict | None):
    """(verdict, detail, frymer_significant_under_fe)."""
    ceiling = app / "runs" / "ceiling.json"
    if not isinstance(obj, dict):
        return "unparseable", "no final JSON object", False
    if not ceiling.is_file():
        return "no_ceiling", f"{ceiling} missing (runs/ is not in git)", False
    c = json.loads(ceiling.read_text(encoding="utf-8"))
    name = app.name
    if name == "gopher_sqf":
        r = gopher(obj, c)
    elif name == "kulynych_adult":
        r = kulynych(obj, c)
    elif name == "frymer_cces":
        r = frymer(obj, c)
    elif name == "zhang_faers":
        r = zhang(obj, c, app)
    elif name == "imai_fl":
        r = imai(obj, c)
    else:
        return "no_ceiling", f"no utility rule for {name}", False
    r = (*r, False) if len(r) == 2 else r
    return r[0], r[1] + own_note(obj), r[2]


def run_info(code_file: Path, results: dict) -> dict:
    """runnable/stdout/source for a script: the last round of its .repair.json, else its results.jsonl line."""
    rj = code_file.with_suffix(".repair.json")
    if rj.is_file():
        entries = json.loads(rj.read_text(encoding="utf-8"))
        last = entries[-1] if isinstance(entries, list) and entries and isinstance(entries[-1], dict) else {}
        status = last.get("status", "")
        runnable = last["runnable"] if "runnable" in last else status in ("ok", "privacy_feedback", "static_privacy",
                                                                           "fake_fix")
        # a repair_failed entry carries the run result of the round that was restored
        return {"runnable": bool(runnable), "stdout": last.get("stdout_tail"), "status": status,
                "source": "repair" if "stdout_tail" in last else "repair (stdout not logged)"}
    r = results.get(file_key(code_file))
    if r is None:
        return {"runnable": False, "stdout": None, "status": "not evaluated", "source": "none"}
    return {"runnable": str(r.get("runnable")) == "True", "stdout": r.get("stdout"), "status": r.get("status", ""),
            "source": "results"}


def suggest(code_state, info, flags, verdict, fe_sig):
    """Suggested grade: N, E, W (privacy flag or Frymer FE significance), G/U from the utility verdict, else '?'."""
    if code_state == "no_code" or info["status"] == "no_code":
        return "N"
    if not info["runnable"]:
        return "E"
    if flags or fe_sig:
        return "W"
    return {"keeps": "G", "loses": "U"}.get(verdict, "?")


def grade_grid(grid: Path, results_path: Path | None, out: Path) -> list[dict]:
    """Suggest a grade for every paper-app script in GRID and write the rows to `out` (CSV)."""
    results = {}
    for path in [p for p in (grid / "results.jsonl", results_path) if p and p.is_file()]:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results.setdefault(file_key(r["code_file"]), r)  # GRID's own results first
    rows = []
    for f in code_files(grid):
        app = app_dir(f)
        if app is None:
            continue
        code = f.read_text(encoding="utf-8", errors="replace")
        cell, task, run = file_key(f)
        info = run_info(f, results)
        prompt = load_prompt("app_" + app.name, parse_cell(cell)["framework"])
        flags = static_problems(prompt, code, info["stdout"]) if info["runnable"] else []
        obj = privacy_check.final_json(info["stdout"]) if info["stdout"] else None
        try:
            verdict, detail, fe_sig = utility(app, obj) if info["runnable"] else ("-", "", False)
        except Exception as e:  # an output shape no rule expected: this script needs a hand look, the grid goes on
            verdict, detail, fe_sig = "unparseable", f"utility check failed: {type(e).__name__}: {e}"[:200], False
        if info["runnable"] and info["stdout"] is None:
            verdict, detail = "unparseable", "stdout not logged in the repair log: re-run in the sandbox"
        rows.append({"cell": cell, "task": task, "run": run, "source": info["source"], "runnable": info["runnable"],
                     "status": info["status"] or code_status(code), "utility": verdict, "utility_detail": detail,
                     "privacy_flags": "; ".join(f"{x['rule']}@{x['line']}: {x['detail']}" for x in flags),
                     "suggested": suggest(code_status(code), info, flags, verdict, fe_sig),
                     "confirm": "by hand"})
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["cell"])
        w.writeheader()
        w.writerows(rows)
    return rows


def agreement(rows: list[dict], hand_csv: Path) -> tuple[int, int, Counter]:
    """(agreeing, compared, Counter of (hand, suggested) pairs) against a hand-grade CSV."""
    hand = {(r["cell"], r["task"], int(r["run"])): r["grade"].strip()
            for r in csv.DictReader(hand_csv.open(encoding="utf-8"))}
    pairs = Counter()
    for r in rows:
        h = hand.get((r["cell"], r["task"], int(r["run"])))
        if h:
            pairs[(h, r["suggested"])] += 1
    agree = sum(n for (h, s), n in pairs.items() if h == s)
    return agree, sum(pairs.values()), pairs


def main() -> None:
    """Grade a grid's paper-app scripts and print the summary (and the agreement with --hand)."""
    p = argparse.ArgumentParser()
    p.add_argument("grid", type=Path)
    p.add_argument("--results", type=Path, help="another grid's results.jsonl for scripts GRID has no result for")
    p.add_argument("--hand", type=Path, help="hand grades (cell, task, run, grade): print the agreement")
    p.add_argument("--out", type=Path, help="default GRID/app_grades.csv")
    a = p.parse_args()
    rows = grade_grid(a.grid, a.results, a.out or a.grid / "app_grades.csv")
    print(f"{len(rows)} app scripts -> {a.out or a.grid / 'app_grades.csv'}; suggested:",
          dict(Counter(r["suggested"] for r in rows)), "(every suggestion needs hand confirmation)")
    if a.hand:
        agree, n, pairs = agreement(rows, a.hand)
        print(f"agreement with {a.hand.name}: {agree}/{n}")
        for (h, s), k in sorted(pairs.items()):
            print(f"  hand {h} -> suggested {s}: {k}")


if __name__ == "__main__":
    main()
