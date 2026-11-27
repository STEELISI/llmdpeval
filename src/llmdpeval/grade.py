"""Grading helpers for a run_grid grid. The grades themselves are given by hand (docs/grading_rubric.md).

  python -m llmdpeval.grade triage GRID            # first-pass flags per script -> GRID/triage.csv (decides nothing;
                                                   # with GRID/audit.jsonl, an audit VIOLATION is shown as a suggested grade)
  python -m llmdpeval.grade table GRID [--runs 6]  # hand_grades.csv + results.jsonl -> GRID/grades.csv and a summary
  python -m llmdpeval.grade median GRID [GRID ...] [--n 20]   # usefulness (G vs U) on the median of N re-runs

Marks: G good, P partial (wrong or missing reported epsilon), L not the requested framework, U valid but too noisy,
W wrong answer or wrong DP, E execution error, N no code. Hand grades live next to the grid outputs
(GRID/hand_grades.csv, columns cell, task, run, grade, reason; runnable scripts only), never in the repo.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdpeval.audit import load_audit, suggestion as audit_suggestion
from llmdpeval.helper import TASKS, code_files, file_key, safe_path_part
from llmdpeval.run_grid import parse_cell
from llmdpeval.sandbox import check_exec_allowed, is_unsafe, kill_sandbox_user, run_sandboxed

GRADES = "GPLUWEN"

# Ground truth from data/adult_clean.csv and data/auto_mpg_clean.csv; also in docs/grading_rubric.md.
TRUE_COUNT = 7508
TRUE_TOP5 = ["Exec-managerial", "Prof-specialty", "Protective-serv", "Tech-support", "Sales"]
OCCUPATIONS = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
               "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty",
               "Protective-serv", "Sales", "Tech-support", "Transport-moving"]  # the 14 values in adult_clean.csv
TRUE_PRED = [44.69, 45.45, 11.80, 46.14]  # sklearn OLS mpg ~ weight at the test inputs
INPUTS = {200.0, 100.0, 4500.0, 10.0}
# Usefulness thresholds (docs/grading_rubric.md). The LinReg cut depends only on the true predictions; whether a
# valid DP fit can reach it depends on the public bounds the prompt states.
COUNT_CUT = 3
TOPK_CUT = 3
MAE_CUT = 0.10 * sum(TRUE_PRED) / 4  # 3.70
# Triage: regexes for the true answers in stdout (rounding tolerated).
TRUE_CHECKS = {
    "stat-count": [r"\b7,?508\b"],
    "selection-topk": [re.escape(o) for o in TRUE_TOP5],
    "ml-regression": [r"44\.6[89]", r"45\.4[45]", r"11\.80", r"46\.1[34]"],
}
FRAMEWORK_PATTERNS = {
    "OpenDP": r"^\s*(import opendp|from opendp)",
    "Diffprivlib": r"^\s*(import diffprivlib|from diffprivlib)",
}
HANDROLLED = r"np\.random\.(laplace|exponential|normal|choice|gumbel)|random\.(random|choice|expovariate)|scipy\.stats"


def load_results(grid: Path) -> dict:
    """GRID/results.jsonl keyed by (cell, task, run); empty if the grid was not evaluated."""
    results = {}
    path = grid / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            results[file_key(r["code_file"])] = r
    return results


# ---- usefulness: a metric from the printed output -------------------------------------------------------------
# Lines are cut into "label: value" segments; a segment is private when its label (or the last section header)
# says private/DP and not true/exact. Bare output without labels falls back to order: the prompt asks for the
# true answer first, then the private one.
TRUE_RE = re.compile(r"\b(true|exact|non[- ]?private|non[- ]?dp|actual|ground|baseline|ols|sklearn|standard|original)\b", re.I)
DP_RE = re.compile(r"\b(private|privacy[- ]preserving|dp|noisy|noised|differentially|differential)\b", re.I)
NOT_RESULT = re.compile(r"epsilon|ε|budget|scale|sensitiv|delta|bound|accountant|spent|seed|noise\b|noise (added|level|std)|error|\bmae\b|\bdiff(erence)?\b|coef|slope|intercept|weight\b|theta|beta|time|elapsed", re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)
SEG = re.compile(r",\s+(?=[A-Za-z])|;\s*|\|")


def segments(stdout: str):
    """(label, value_text, kind) per printed segment; kind is 'true', 'dp' or None (unlabelled)."""
    mode, header, out = None, "", []
    for line in stdout.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*#=>]+)\s+", "", line)  # "1. True ...", "- DP ..."
        line = re.sub(r"\([^()]*[A-Za-z][^()]*\)", "", line)  # "(Laplace, scale=1.0)" in a label
        line = re.sub(r"^\s*\[([A-Za-z ]+)\]\s*$", r"\1", line)  # "[True Answer]" header
        for seg in SEG.split(line):
            if ":" in seg:
                label, value = seg.rsplit(":", 1)
            elif re.match(r"\s*([\d\-]|\[\s*[\d'\"\-]|\(\s*[\d'\"])", seg) or any(o in seg for o in TRUE_TOP5):
                label, value = "", seg
            else:
                label, value = seg, ""
            value = re.sub(r"\[(200|100|4500|10)\]", " ", value)  # "Test Input [200]: 43.2"
            has_value = bool(NUM.search(value) or re.search(r"'[A-Za-z-]+'|\b[A-Z][a-z]+-[a-z]", value)) or any(o in value for o in TRUE_TOP5)
            t, d = bool(TRUE_RE.search(label)), bool(DP_RE.search(label))
            if (t or d) and not has_value:
                mode, header = ("true" if t else "dp"), label
                continue
            if not has_value:
                if label.strip():
                    header = label  # a sub-header such as "Results:" keeps the mode
                continue
            kind = "true" if t else "dp" if d else mode
            if NOT_RESULT.search(label) or (kind == mode and not (t or d) and NOT_RESULT.search(header) and not re.search(r"pred|count|top|occup|result|answer", header, re.I)):
                continue
            out.append((label + " " + header, value, kind))
    return out


def metric(task: str, stdout: str):
    """stat-count: |dp - 7508|; selection-topk: how many of the true top 5 are in the private top 5;
    ml-regression: mean absolute error of the 4 DP predictions. None if the output cannot be parsed."""
    if task not in USEFUL:
        return None  # no automatic usefulness metric for this task (e.g. a paper application)
    segs = segments(stdout)
    dp = [s for s in segs if s[2] == "dp"]
    bare = [s for s in segs if s[2] is None]
    if task == "stat-count":
        pick = lambda ss: [abs(x - TRUE_COUNT) for _, v, _ in ss for x in map(float, NUM.findall(v)) if 5000 <= x <= 10000]
        got = pick(dp)
        if got:
            return got[0]
        got = pick(bare)
        return got[1] if len(got) >= 2 else None
    if task == "selection-topk":
        lists = [v for _, v, _ in dp if any(o in v for o in TRUE_TOP5) or "'" in v or re.search(r"[A-Z][a-z]+-", v)]
        if not lists:
            named = [v for _, v, _ in bare if re.search(r"[A-Z][a-z]+-[a-z]", v)]
            lists = named[1:2] if len(named) >= 2 else []
        if not lists and not dp:
            return None
        return topk_score(lists if lists else [v for _, v, _ in dp])
    if task == "ml-regression":
        def preds_from(ss, need_pred):
            out = []
            for lab, v, _ in ss:
                if need_pred and not re.search(r"pred|input|output|estimate|result|mpg", lab, re.I):
                    continue
                out += [x for x in map(float, NUM.findall(v)) if x not in INPUTS]
            return out
        p = preds_from(dp, True) or preds_from(dp, False)
        if len(p) < 4:
            groups = [g for g in ([float(x) for x in NUM.findall(v)] for _, v, _ in bare) if len(g) >= 4]
            p = groups[1] if len(groups) >= 2 else p
        if len(p) < 4:
            return None
        return sum(abs(a - b) for a, b in zip(p[:4], TRUE_PRED)) / 4


def topk_score(texts: list[str]) -> int:
    """How many of the true top 5 are in the printed private top 5. Only the top-5 list counts: the first
    bracketed list of exactly 5 names; without one, the first 5 occupations named, in printed order. A script
    that also prints its full noisy table therefore does not score 5/5 by naming every occupation."""
    for text in texts:
        for inner in re.findall(r"\[([^\[\]]*)\]", text):
            items = [x.strip(" '\"") for x in inner.split(",") if x.strip(" '\"")]
            if len(items) == 5 and not all(NUM.fullmatch(x) for x in items):  # names, not 5 noisy rates
                return sum(o in items for o in TRUE_TOP5)
    named = sorted((m.start() + 100000 * i, o) for i, text in enumerate(texts) for o in OCCUPATIONS
                   for m in re.finditer(rf"(?<![\w-]){re.escape(o)}(?![\w-])", text))
    first5 = list(dict.fromkeys(o for _, o in named))[:5]
    return sum(o in first5 for o in TRUE_TOP5)


USEFUL = {"stat-count": lambda v: v <= COUNT_CUT, "selection-topk": lambda v: v >= TOPK_CUT,
          "ml-regression": lambda v: v <= MAE_CUT}  # the tasks with an automatic usefulness metric


def useful(task: str, m) -> bool | None:
    """Whether metric value m passes the task's usefulness cut; None for a task without a metric."""
    return USEFUL[task](m) if task in USEFUL else None


# ---- triage ---------------------------------------------------------------------------------------------------
def static_flags(code_file: Path) -> dict:
    """Triage flags read from the code: framework imports, hand-rolled noise, fixed seeds, unsafe calls, data writes."""
    code = code_file.read_text(encoding="utf-8", errors="replace")
    cell, task, run = file_key(code_file)
    return {
        "cell": cell, "task": task, "run": run,
        "framework_asked": parse_cell(cell)["framework"],
        "imports_opendp": bool(re.search(FRAMEWORK_PATTERNS["OpenDP"], code, re.M)),
        "imports_diffprivlib": bool(re.search(FRAMEWORK_PATTERNS["Diffprivlib"], code, re.M)),
        "handrolled_noise": bool(re.search(HANDROLLED, code)),
        "fixed_seed": bool(re.search(r"seed\s*\(|random_state\s*=\s*\d|seed\s*=\s*\d", code)),
        "unsafe": is_unsafe(code),
        # Repairs can add "if the data file is missing, write a stand-in" code.
        "writes_data": bool(re.search(r"to_csv\(|open\([^)]*['\"][wa]b?['\"]", code)),
        "mock_data_fallback": bool(re.search(r"\b(mock|dummy|placeholder|synthetic|fake)\b.*\b(data|dataframe|df)\b", code, re.I)),
        "code_file": str(code_file),
    }


def output_flags(run: dict, task: str) -> dict:
    """Triage flags read from one run: runnable, how many true answers appear in stdout, the last stderr line."""
    stdout = run["stdout"] or ""
    hits = sum(bool(re.search(p, stdout)) for p in TRUE_CHECKS.get(task, []))
    return {
        "runnable": run["runnable"],
        "true_answer_hits": f"{hits}/{len(TRUE_CHECKS.get(task, []))}",
        "stderr_tail": "".join((run["stderr"] or "").strip().splitlines()[-1:])[:160],
    }


def triage(grid: Path) -> None:
    """Write GRID/triage.csv: static and output flags per script, to decide what to read first."""
    runs = load_results(grid)
    audits = load_audit(grid)
    rows = []
    for code_file in code_files(grid):
        row = static_flags(code_file)
        key = (row["cell"], row["task"], row["run"])
        if key in runs:
            row |= output_flags(runs[key], row["task"])
        if audits:  # GRID/audit.jsonl: a VIOLATION is a suggested grade with its evidence (rubric rule 18)
            r = audits.get(key)
            row["audit_verdict"] = r["verdict"] if r else ""
            row["audit_suggested"], row["audit_evidence"] = audit_suggestion(r)
        rows.append(row)
    out = grid / "triage.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} files -> {out}")
    print("unsafe:", [r["code_file"] for r in rows if r["unsafe"]])


# ---- table ----------------------------------------------------------------------------------------------------
def table(grid: Path, runs: int, grades_file: str, conditions: list[str] | None, tasks: list[str]) -> None:
    """One row per cell x task x run for every cell folder in the grid (a model that never started has no folder)."""
    with (grid / grades_file).open(newline="", encoding="utf-8") as f:
        hand = {(r["cell"], r["task"], int(r["run"])): (r["grade"], r["reason"]) for r in csv.DictReader(f)}
    results = load_results(grid)
    cells = sorted(d.name for d in grid.iterdir() if d.is_dir() and d.name.count("_") >= 2)
    cells = [c for c in cells if conditions is None or parse_cell(c)["condition"] in conditions]
    rows, used = [], set()
    for cell in cells:
        c = parse_cell(cell)
        for t in tasks:
            for n in range(1, runs + 1):
                key = (cell, t, n)
                r = results.get(key)
                if r is None:
                    g, why = "N", "no code generated"
                elif r.get("status") == "no_code":
                    g, why = "N", "the answer contained no code"
                elif r["runnable"] != "True" and r.get("status") != "unsafe_not_run":  # unsafe: graded by hand
                    g, why = "E", ((r["stderr"] or "").strip().splitlines() or ["timeout"])[-1][:140]
                elif key in hand:
                    g, why = hand[key]
                    used.add(key)
                else:
                    sys.exit(f"runnable but ungraded: {key}")
                rows.append({"model": safe_path_part(c["model"]), "condition": c["condition"], "task": t, "run": n,
                             "grade": g, "reason": why})
    # only the slice this table covers: --conditions, --tasks and --runs may be narrower than hand_grades.csv
    stray = [k for k in hand if k[0] in cells and k[1] in tasks and k[2] <= runs and k not in used]
    if stray:
        sys.exit(f"hand grades for scripts that are missing or not runnable: {stray[:5]}")
    with (grid / "grades.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    models = list(dict.fromkeys(r["model"] for r in rows))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    print(f"{'':40}" + "".join(f"{m:>17}" for m in models))
    for t in tasks:
        for cond in conds:
            cnt = [Counter(r["grade"] for r in rows if (r["model"], r["condition"], r["task"]) == (m, cond, t))
                   for m in models]
            print(f"{t:15}{cond:25}" + "".join(f"{' '.join(f'{c[g]}{g}' for g in GRADES if c[g]):>17}" for c in cnt))
    print(Counter(r["grade"] for r in rows))


# ---- median ---------------------------------------------------------------------------------------------------
def median(grid: Path, n: int, workers: int) -> None:
    """Re-run every script graded G, U or P n times (same sandbox as the evaluator) and grade on the median
    metric: useful -> G, not useful -> U. P stays P (the reported epsilon is still wrong); when its median is not
    useful, the reason says so. W/E/N/L are about validity and are copied unchanged. Fewer than n/2 parsable runs:
    the hand grade is kept and the reason says so. Not re-run, hand grade kept with a note: tasks with no automatic
    usefulness metric (paper applications) and code that sandbox.is_unsafe flags. A script that raises in the
    harness is recorded with its error, not fatal.
    Writes GRID/rerun_median.jsonl (one line per run, resumable) and GRID/hand_grades_median.csv."""
    check_exec_allowed()
    rows = list(csv.DictReader(open(grid / "hand_grades.csv", encoding="utf-8")))
    path = lambda r: grid / r["cell"] / f"{r['task']}_code_{r['run']}.py"
    skip = {}
    for r in rows:
        if r["grade"] in ("G", "U", "P"):
            if r["task"] not in USEFUL:
                skip[r["cell"], r["task"], r["run"]] = "no automatic usefulness metric"
            elif not path(r).exists():
                skip[r["cell"], r["task"], r["run"]] = "script missing"
            elif is_unsafe(path(r).read_text(encoding="utf-8", errors="replace")):
                skip[r["cell"], r["task"], r["run"]] = "unsafe_not_run (matches sandbox.UNSAFE)"
    pending = [r for r in rows if r["grade"] in ("G", "U", "P") and (r["cell"], r["task"], r["run"]) not in skip]
    log = grid / "rerun_median.jsonl"
    done = {}
    if log.exists():
        for line in open(log, encoding="utf-8"):
            x = json.loads(line)
            done.setdefault((x["cell"], x["task"], x["run"]), []).append(x)
    jobs = [(r, i) for r in pending for i in range(len(done.get((r["cell"], r["task"], r["run"]), [])), n)]
    print(f"{grid.name}: {len(pending)} scripts, {len(jobs)} runs to do, {len(skip)} not re-run", flush=True)

    def job(arg):
        r, i = arg
        x = {"cell": r["cell"], "task": r["task"], "run": r["run"], "i": i}
        try:
            res = run_sandboxed(path(r))
            m = metric(r["task"], res.get("stdout") or "") if str(res.get("returncode")) == "0" else None
            return x | {"returncode": res.get("returncode"), "metric": m}
        except Exception as e:  # one script's failure is a result, not a reason to stop the pool
            return x | {"returncode": None, "metric": None, "error": f"{type(e).__name__}: {e}"[:300]}

    try:
        with ThreadPoolExecutor(workers) as pool, open(log, "a", encoding="utf-8") as out:
            for k, x in enumerate(pool.map(job, jobs), 1):
                out.write(json.dumps(x) + "\n")
                out.flush()
                done.setdefault((x["cell"], x["task"], x["run"]), []).append(x)
                if k % 100 == 0:
                    print(f"  {k}/{len(jobs)}", flush=True)
    finally:
        kill_sandbox_user()  # end of the pass: no run is in progress, so anything left of the sandbox user goes

    with open(grid / "hand_grades_median.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "task", "run", "grade", "reason"])
        w.writeheader()
        for r in rows:
            r = dict(r)
            key = (r["cell"], r["task"], r["run"])
            if key in skip:
                r["reason"] += f" [median: not re-run, {skip[key]}; hand grade kept]"
            elif r["grade"] in ("G", "U", "P"):
                ms = [x["metric"] for x in done.get(key, []) if x["metric"] is not None]
                if len(ms) < n / 2:
                    r["reason"] += f" [median: only {len(ms)}/{n} runs parsable, hand grade kept]"
                else:
                    med = statistics.median(ms)
                    is_useful = useful(r["task"], med)
                    # P stays P either way: its reported epsilon is still wrong; the note says if it is too noisy
                    new = "P" if r["grade"] == "P" else "G" if is_useful else "U"
                    r["reason"] += (f" [median of {len(ms)} runs: {med:.2f}; single run gave {r['grade']}"
                                    + ("" if is_useful else "; not useful on the median") + "]")
                    r["grade"] = new
            w.writerow(r)


def main() -> None:
    """Dispatch the triage / table / median subcommands."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="first-pass flags -> GRID/triage.csv")
    t.add_argument("grid", type=Path)
    g = sub.add_parser("table", help="hand grades + results -> GRID/grades.csv and a summary")
    g.add_argument("grid", type=Path)
    g.add_argument("--runs", type=int, default=6)
    g.add_argument("--grades", default="hand_grades.csv", help="e.g. hand_grades_median.csv")
    g.add_argument("--conditions", nargs="+", help="only these conditions (labels as printed, e.g. OpenDP-rag-k3_th_code)")
    g.add_argument("--tasks", nargs="+", default=TASKS)
    m = sub.add_parser("median", help="usefulness on the median of N re-runs")
    m.add_argument("grids", nargs="+", type=Path)
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.cmd == "triage":
        triage(a.grid)
    elif a.cmd == "table":
        table(a.grid, a.runs, a.grades, a.conditions, [safe_path_part(x) for x in a.tasks])
    else:
        for grid in a.grids:
            median(grid.resolve(), a.n, a.workers)


if __name__ == "__main__":
    main()
