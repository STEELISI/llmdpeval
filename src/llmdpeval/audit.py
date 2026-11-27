"""Empirical DP audit: run a script on neighbouring datasets and test whether its releases leak more than the
budget allows.

  python -m llmdpeval.audit GRID [SCRIPT ...] [--deterministic-only] [--n 1000] [--workers 6]  # -> GRID/audit.jsonl
  python -m llmdpeval.audit SCRIPT [SCRIPT ...] [--app APP | --task TASK] [--n 1000]
                            [--records typical,extreme,group,worst] [--rows M] [--workers 6] [--budget 1.0]
                            [--confidence 0.95] [--extra FILE ...] [--out FILE]

A first-pass filter before hand grading: a VIOLATION removes a script, with numbers as evidence (suggested grade
W, or P for UNDERREPORTED; see suggestion()); a PASS only means that the script goes on to hand grading. Zero false
violations comes first (every bound holds at the stated confidence), then catching as many flawed scripts as
possible. With a grid folder, the deterministic test runs on every script that ran, and the statistical test
only on the scripts it did not remove (GRID/audit.jsonl, resumable per mode). --deterministic-only stops after
the first test (DETERMINISTIC_VIOLATION or NO_DETERMINISTIC_FINDING); scripts listed after GRID restrict the
audit to them (e.g. only the scripts suggested G).

Neighbours follow the task's relation: paper applications are replace-one (one record's values are replaced by
another record's), the toy tasks are add/remove (one record is removed). The records:
  typical  the middle row of the private file; replace-one: replaced by the next row that differs from it in the
           group and label columns.
  extreme  the row with the largest contribution (the sum of its min-max scaled numeric columns, or the app's own
           score, e.g. the number of AEs of a report); replace-one: it replaces the row with the smallest score.
  group    replace-one: a row of the smallest group (the sensitive or grouping attribute) is replaced by the
           highest-scoring row of the largest group with another label, so one record moves between groups;
           add/remove: a row of the smallest group is removed.
  worst    replace-one only, one per private file: a typical record of the largest group is replaced by a crafted
           record whose every value is in the public domain of spec.md (worst_tabular, worst_zhang, worst_imai):
           the smallest group, the rarer label, each bounded column at its far public bound; for Zhang a report at
           the cap K on the AEs closest to the significance boundary; for Imai a voter of the smallest race group
           moved to the smallest precinct.
The base dataset D and every neighbour are written by the same code into a scratch folder, and the script runs on
each N times through sandbox.run_sandboxed (the same sandbox as every evaluation).

Releases: the final JSON line (privacy_check.final_json), keys starting with dp_, flattened to named scalars
(dict keys and list positions join the name; a list of strings becomes one 0/1 indicator per item seen). Toy
scripts without a JSON line: the numbers and occupation names of the output segments labelled private
(grade.segments).

Tests per neighbour, on the N runs of D and of D':
  deterministic  a release with one value in every run on D and one, different, value on D': no noise or seeded
                 noise; a violation for certain (the privacy loss is unbounded). Every release constant across
                 runs on D (seeded or noise-free output) is also a violation. Checked on a first batch of runs.
                 A heuristic outside the Bonferroni count: confirm its findings in the code (rubric rule 18).
  statistical    (Ding et al., CCS 2018) the first half of the runs chooses events: threshold events {x > t} and
                 {x <= t} on each release and conjunctions of the best events of up to 4 releases; the second half
                 estimates P(S | D) and P(S | D') with Clopper-Pearson intervals. eps_lb = log(p_low(D) / p_high(D'))
                 in both directions is a lower confidence bound on the privacy loss. Bonferroni over every tested
                 event of every neighbour, so the chance of a false violation for one script is at most
                 1 - confidence (per script, not per grid).
                 VIOLATION when eps_lb > budget.
Verdict per script: PASS (no violation found; not a proof of DP), VIOLATION (release, record, eps_lb, test),
UNDERREPORTED (eps_lb above the printed epsilon_total but within the budget) or ERROR (the script does not run
or prints no release). Only what reaches the output can be tested: a leak that is
hidden by later post-processing, or one record's effect spread thinly over many releases, can pass.
"""
import argparse
import json
import math
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from llmdpeval.helper import APP_DIR, DATA_DIR, file_key
from llmdpeval.privacy_check import final_json
from llmdpeval.sandbox import app_dir, check_exec_allowed, eval_settings, kill_sandbox_user, run_sandboxed

RECORDS = ("typical", "extreme", "group", "worst")
N_DEFAULT = 1000      # runs per dataset; see the power note in main()'s help and the validation report
N_FIRST = 10          # first batch per dataset: errors and the deterministic test
TOP_EVENTS = 5       # single-release events tested per neighbour and direction (plus one joint event)
JOINT_MAX = 4         # releases in a conjunction event
MAX_ERROR_SHARE = 0.1  # more failed runs than this: ERROR
PRINTED = "__epsilon_total__"  # key of the printed epsilon in a run's release dict (never a release name)


def _n_aes(df):
    """Number of AEs per report (Zhang)."""
    return df["aes"].map(lambda s: len([a for a in s.split(";") if a]))


def _imai_race(df):
    """Self-reported race of a voter (argmax of the SR.* one-hot columns; 'none' if all 0)."""
    sr = df[[c for c in df.columns if c.startswith("SR.")]].apply(pd.to_numeric)
    return sr.idxmax(axis=1).where(sr.sum(axis=1) > 0, "none")


# Per task: the private files (others are copied unchanged), the group and label columns (a column name or a
# function of the frame), an optional score for the extreme record, the relation, a default row subsample, and
# "complete": records are chosen among rows without missing values (the apps whose complete-case counts are
# public, so a neighbour must keep them).
_GOPHER_BIN = ["perobs", "cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_drgtr", "cs_vcrim", "ac_proxm"]
SPECS = {
    # Public domains for the worst-case records, from each app's spec.md (codebook codes and ranges).
    "gopher_sqf": {"files": ["sqf_train.csv", "sqf_test.csv"], "group": "race", "label": "frisked",
                   "bounds": {"build": (0, 2), "age": (0, 2), "ht_feet": (0, 2), "weight": (0, 2),
                              **{c: (0, 1) for c in _GOPHER_BIN}},
                   "onehot": [["inout_I", "inout_O"], ["sex_F", "sex_M"]]},
    "kulynych_adult": {"files": ["train.csv", "test.csv"], "group": "race", "label": "income", "complete": True,
                       "bounds": {"capital-gain": (0, 99999), "capital-loss": (0, 9999), "hours-per-week": (1, 99),
                                  "education-num": (1, 16)},
                       "categorical": ["work-class", "marital-status", "occupation", "relationship", "sex",
                                       "native-country"]},
    "frymer_cces": {"files": ["cces_white.csv"], "group": "union_member", "label": "racial_resentment",
                    "complete": True,
                    "bounds": {"racial_resentment": (0, 1), "gender": (1, 2), "famincome": (1, 18), "age": (18, 100),
                               "educ": (1, 6)},
                    "categorical": ["state_pre", "year"], "fixed": {"past_union_member": "0.0"}},
    "imai_fl": {"files": ["voters.parquet"], "group": _imai_race, "label": "vote08", "rows": 100_000,
                "worst": "imai"},
    "zhang_faers": {"files": ["reports.csv"], "group": "window", "label": None, "score": _n_aes, "rows": 50_000,
                    "worst": "zhang", "k": 6},
    "stat-count": {"files": ["adult_clean.csv"], "group": "occupation", "label": "income", "relation": "add_remove"},
    "selection-topk": {"files": ["adult_clean.csv"], "group": "occupation", "label": "income",
                       "relation": "add_remove"},
    "ml-regression": {"files": ["auto_mpg_clean.csv"], "group": "cylinders", "label": "mpg",
                      "relation": "add_remove"},
}


def task_of(code_file: Path) -> str:
    """The SPECS key for a generated script: its app (app-<app>_code_<n>.py) or its toy task."""
    app = app_dir(code_file)
    if app:
        return app.name
    return Path(code_file).name.split("_code_")[0]


# ---- neighbours -----------------------------------------------------------------------------------------------
def read_frame(path: Path) -> pd.DataFrame:
    """CSV as text (written back byte-compatible), parquet as typed columns."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_frame(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _col(df, spec_value):
    if spec_value is None:
        return pd.Series("", index=df.index)
    return spec_value(df) if callable(spec_value) else df[spec_value].astype(str)


def score(df: pd.DataFrame, spec: dict) -> pd.Series:
    """A record's contribution: the app's score, else the sum of its min-max scaled numeric columns."""
    if spec.get("score"):
        return spec["score"](df).astype(float)
    total = pd.Series(0.0, index=df.index)
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() > 0.9 and x.max() > x.min():
            total += ((x - x.min()) / (x.max() - x.min())).fillna(0)
    return total


def pick_records(df: pd.DataFrame, spec: dict, kinds=RECORDS[:3]) -> list[dict]:
    """[{kind, victim, donor}] as row positions of df; donor is None for add/remove (the victim is removed)."""
    replace = spec.get("relation", "replace_one") == "replace_one"
    group, label = _col(df, spec["group"]).to_numpy(), _col(df, spec.get("label")).to_numpy()
    ok = (df.notna() & (df.astype(str) != "")).all(axis=1).to_numpy() if spec.get("complete") \
        else np.ones(len(df), bool)
    sc = np.where(ok, score(df, spec).to_numpy(), np.nan)
    out = []
    for kind in kinds:
        if kind == "typical":
            v = int(np.flatnonzero(ok)[ok.sum() // 2])
            later = [j for j in np.flatnonzero(ok) if j > v and (group[j], label[j]) != (group[v], label[v])]
            d = next((j for j in later if group[j] != group[v] and label[j] != label[v]), later[0])
        elif kind == "extreme":
            v, d = int(np.nanargmin(sc)), int(np.nanargmax(sc))
            if not replace:
                v = d
        elif kind == "group":
            counts = pd.Series(group[ok]).value_counts()
            small, large = counts.index[-1], counts.index[0]
            in_large = np.flatnonzero((group == large) & ok)
            d = int(in_large[np.nanargmax(sc[in_large])])
            in_small = np.flatnonzero((group == small) & ok)
            other = in_small[label[in_small] != label[d]]
            v = int((other if len(other) else in_small)[0])
        else:
            raise ValueError(f"unknown record kind {kind}")
        out.append({"kind": kind, "victim": int(v), "donor": int(d) if replace else None})
    return out


# ---- worst-case records ---------------------------------------------------------------------------------------
# The auditor may search the whole dataset, because DP must hold for every neighbour. A crafted record need not be
# a real row, but every value it gets is in the task's public domain (spec.md): a public bound, a category seen in
# the data, or a value copied from a real row (e.g. a consistent set of geography columns).
def _fmt(value, column: pd.Series):
    """A number in the text form of a CSV column ('2' or '2.0')."""
    return str(float(value)) if column.astype(str).str.contains(r"\.").any() else str(int(value))


def _far_bound(column: pd.Series, lo, hi):
    """The public bound farther from the column's mean in the data."""
    m = pd.to_numeric(column, errors="coerce").mean()
    return hi if abs(hi - m) >= abs(m - lo) else lo


def _complete(df, spec):
    return (df.notna() & (df.astype(str) != "")).all(axis=1).to_numpy() if spec.get("complete") \
        else np.ones(len(df), bool)


def worst_tabular(df: pd.DataFrame, spec: dict) -> tuple[int, pd.Series]:
    """A record of the largest group with the majority label is replaced by a crafted record: the smallest group,
    the rarer label, every bounded column at the public bound farther from its mean, every categorical column at
    its rarest value, the rarer member of each one-hot set. (victim position, crafted row)."""
    ok = _complete(df, spec)
    group, label = _col(df, spec["group"]), spec.get("label")
    counts = group[ok].value_counts()
    small, large = counts.index[-1], counts.index[0]
    bounds = spec["bounds"]
    lab = df[label].astype(str)
    if label in bounds:
        rare_label = _fmt(_far_bound(df[label][ok], *bounds[label]), df[label])
        major = (pd.to_numeric(df[label], errors="coerce") - float(rare_label)).abs() > 0.5 * (bounds[label][1] - bounds[label][0])
    else:
        lc = lab[ok].value_counts()
        rare_label, major = lc.index[-1], (lab == lc.index[0]).to_numpy()
    sc = np.where(ok, score(df, spec).to_numpy(), np.nan)
    pool = np.flatnonzero(ok & (group == small).to_numpy())
    base = df.iloc[int(pool[np.nanargmax(sc[pool])])].copy()
    base[spec["group"]] = small
    base[label] = rare_label
    for c, (lo, hi) in bounds.items():
        if c != label:
            base[c] = _fmt(_far_bound(df[c][ok], lo, hi), df[c])
    for c in spec.get("categorical", []):
        vals = df[c][ok].astype(str)
        base[c] = vals.value_counts().index[-1]
    for cols in spec.get("onehot", []):
        rarest = min(cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").mean())
        for c in cols:
            base[c] = _fmt(1 if c == rarest else 0, df[c])
    for c, v in spec.get("fixed", {}).items():
        base[c] = v
    cand = np.flatnonzero(ok & (group == large).to_numpy() & np.asarray(major))
    victim = int(cand[np.nanargmin(np.abs(sc[cand] - np.nanmedian(sc[cand])))])
    return victim, base


def worst_zhang(df: pd.DataFrame, spec: dict, terms: pd.DataFrame) -> tuple[int, pd.Series]:
    """A report with no AE is replaced by a report at the public cap K whose AEs are the tested table AEs closest
    to the Bonferroni significance boundary (Pearson z of the window x AE table), filed in the window that pushes
    most of them across it."""
    k = spec["k"]
    w = df["window"].astype(str).to_numpy()
    n19, n = float((w == "2019").sum()), float(len(w))
    pi = (n - n19) / n
    a19, a20 = {}, {}
    for win, aes in zip(w, df["aes"]):
        for a in set(x for x in aes.split(";") if x):
            (a19 if win == "2019" else a20)[a] = (a19 if win == "2019" else a20).get(a, 0) + 1
    tested = terms[terms["drop_before_test"].astype(int) == 0].index.astype(str)
    t = pd.DataFrame({"A19": pd.Series(a19), "A20": pd.Series(a20)}).reindex(tested).fillna(0)
    t = t[(t.A19 + t.A20) > 0]
    T = t.A19 + t.A20
    z = (t.A20 - pi * T) / np.sqrt(np.maximum(T * pi * (1 - pi) * (n - T) / (n - 1), 1e-12))
    from scipy.stats import norm
    zstar = norm.isf(0.05 / len(t) / 2)
    shown = terms["drop_from_table"].astype(int) == 0
    zt = z[[a for a in z.index if shown.get(a, False)]]
    pick = (zt.abs() - zstar).abs().nsmallest(k).index
    # outward when below the boundary, inward when above; 2020 raises z
    want = [("2020" if zt[a] > 0 else "2019") if abs(zt[a]) < zstar else ("2019" if zt[a] > 0 else "2020")
            for a in pick]
    win = max(set(want), key=want.count)
    n_aes = _n_aes(df).to_numpy()
    same = np.flatnonzero(w == win)
    victim = int(same[np.argmin(n_aes[same])])
    row = df.iloc[victim].copy()
    row["aes"] = ";".join(pick)
    return victim, row


def worst_imai(df: pd.DataFrame, spec: dict) -> tuple[int, pd.Series]:
    """A voter of the largest race group, majority turnout, in the largest precinct is replaced by a voter of the
    smallest race group with the rarer turnout, moved to the smallest precinct: the geography columns (district,
    precinct, block and their Census tables) are copied together from a real voter of that precinct."""
    race = _imai_race(df).to_numpy()
    part = df["District"].astype(str) + "|" + df["Precinct"].astype(str)
    pc = part.value_counts()
    rc = pd.Series(race[race != "none"]).value_counts()
    small_r, large_r = rc.index[-1], rc.index[0]
    vote = df[spec["label"]].to_numpy()
    rare_v = pd.Series(vote).value_counts().index[-1]
    pool = np.flatnonzero((race == small_r) & (vote == rare_v))
    pool = pool if len(pool) else np.flatnonzero(race == small_r)
    row = df.iloc[int(pool[0])].copy()
    geo = [c for c in df.columns if c in ("District", "Precinct", "Block") or c.startswith(("r_prc_", "r_blk_"))]
    src = df.iloc[int(np.flatnonzero((part == pc.index[-1]).to_numpy())[0])]
    row[geo] = src[geo]
    cand = np.flatnonzero((part == pc.index[0]).to_numpy() & (race == large_r) & (vote != rare_v))
    cand = cand if len(cand) else np.flatnonzero((part == pc.index[0]).to_numpy())
    return int(cand[0]), row


def worst_records(frames: dict, spec: dict) -> list[dict]:
    """One worst-case record per private file (Zhang: the reports file; the AE term list is public)."""
    out = []
    for f in spec["files"]:
        df = frames[f]
        if spec.get("worst") == "zhang":
            v, row = worst_zhang(df, spec, frames["_terms"])
        elif spec.get("worst") == "imai":
            v, row = worst_imai(df, spec)
        elif spec.get("bounds"):
            v, row = worst_tabular(df, spec)
        else:
            continue
        out.append({"kind": "worst" if len(spec["files"]) == 1 else f"worst:{f}", "file": f, "victim": v,
                    "donor": "crafted", "row": row})
    return out


def neighbour(df: pd.DataFrame, rec: dict) -> pd.DataFrame:
    """D' for one record: the victim row's values replaced by the donor's or by a crafted row (replace-one), or
    the victim removed (add/remove)."""
    if rec["donor"] is None:
        return df.drop(index=df.index[rec["victim"]]).reset_index(drop=True)
    out = df.copy()
    new = rec["row"][df.columns] if rec["donor"] == "crafted" else df.iloc[rec["donor"]]
    out.iloc[rec["victim"]] = new.to_numpy()
    return out


def build_datasets(task: str, data_dir: Path, root: Path, kinds=RECORDS, rows: int | None = None) -> dict:
    """Write D and one neighbour per record (and per private file for 'typical'/'extreme' on the first file,
    'group' on the last) under root/<name>/. Returns {name: {"files": [...], "record": {...}}}."""
    spec = SPECS[task]
    rows = rows if rows is not None else spec.get("rows")
    frames = {}
    for f in spec["files"]:
        df = read_frame(data_dir / f)
        if rows and len(df) > rows:  # any subset of the data is a dataset too; DP must hold on it
            keep = np.sort(np.random.default_rng(0).choice(len(df), rows, replace=False))
            df = df.iloc[keep].reset_index(drop=True)
        frames[f] = df
    others = [p for p in sorted(data_dir.glob("*")) if p.is_file() and p.name not in frames]

    def write(name, fr, record):
        d = root / name
        d.mkdir(parents=True)
        for f, df in fr.items():
            write_frame(df, d / f)
        for p in others:
            shutil.copy(p, d / p.name)
        return {"files": sorted(d.iterdir()), "record": record}

    out = {"D": write("D", frames, None)}
    recs = []
    for kind in kinds:
        if kind == "worst":
            if spec.get("relation", "replace_one") == "replace_one":
                terms = {"_terms": read_frame(data_dir / "ae_terms.csv").set_index("meddra_id")} \
                    if spec.get("worst") == "zhang" else {}
                recs += worst_records(frames | terms, spec)
            continue
        f = spec["files"][-1] if kind == "group" else spec["files"][0]
        recs.append(pick_records(frames[f], spec, [kind])[0] | {"file": f})
    for rec in recs:
        f = rec["file"]
        nb = neighbour(frames[f], rec)
        # only positions and group labels go into the report, never a record's values (the data is private)
        g = _col(frames[f], spec["group"]).to_numpy()
        donor_g = [] if rec["donor"] is None else [str(_col(nb, spec["group"]).to_numpy()[rec["victim"]])]
        rec = {k: v for k, v in rec.items() if k != "row"} | {"groups": [str(g[rec["victim"]])] + donor_g}
        out[rec["kind"]] = write(rec["kind"].replace(":", "_"), frames | {f: nb}, rec)
    return out


# ---- releases -------------------------------------------------------------------------------------------------
def flatten(obj, name="") -> dict:
    """Named scalars of a JSON value: numbers and bools as floats, a string s as {name=s: 1}, a list of strings
    as one indicator per item, other lists by position, dicts by key. None becomes NaN."""
    if isinstance(obj, dict):
        return {k: v for key, val in obj.items() for k, v in flatten(val, f"{name}.{key}" if name else str(key)).items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            return {f"{name}∋{x}": 1.0 for x in obj}
        return {k: v for i, val in enumerate(obj) for k, v in flatten(val, f"{name}[{i}]").items()}
    if isinstance(obj, str):
        return {f"{name}={obj}": 1.0}
    if obj is None:
        return {name: math.nan}
    return {name: float(obj)}


def releases(stdout: str) -> dict | None:
    """The released scalars of one run: dp_* keys of the final JSON line, else the private segments of a toy
    output. None when the run released nothing parsable."""
    obj = final_json(stdout)
    if obj is not None:
        rel = {k: v for k, v in obj.items() if k.startswith("dp_")}
        return flatten(rel) if rel else None
    from llmdpeval.grade import NUM, OCCUPATIONS, segments  # toy outputs: no JSON line
    out = {}
    for i, (label, value, kind) in enumerate(segments(stdout or "")):
        if kind != "dp":
            continue
        for j, x in enumerate(NUM.findall(value)):
            out[f"seg{i}[{j}] {label.strip()[:40]}"] = float(x)
        for o in OCCUPATIONS:
            if re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])", value):
                out[f"seg{i}∋{o}"] = 1.0
    return out or None


def matrix(runs: list[dict], names: list[str]) -> np.ndarray:
    """Runs x releases; an indicator absent from a run is 0, another missing value NaN."""
    m = np.full((len(runs), len(names)), np.nan)
    for i, r in enumerate(runs):
        for j, k in enumerate(names):
            m[i, j] = r.get(k, 0.0 if "∋" in k or "=" in k else math.nan)
    return m


# ---- statistics -----------------------------------------------------------------------------------------------
def cp_low(k, n, a):
    """Clopper-Pearson one-sided lower bound at level a (arrays ok)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.where(k > 0, beta.ppf(a, np.maximum(k, 1e-12), n - k + 1), 0.0)


def cp_high(k, n, a):
    """Clopper-Pearson one-sided upper bound at level a (arrays ok)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.where(k < n, beta.ppf(1 - a, k + 1, np.maximum(n - k, 1e-12)), 1.0)


def eps_lb(ka, na, kb, nb, a):
    """Lower confidence bound (level a, split over both intervals) on log P_A(S) / P_B(S)."""
    lo, hi = cp_low(ka, na, a / 2), cp_high(kb, nb, a / 2)
    with np.errstate(divide="ignore"):
        return np.where(lo > 0, np.log(lo) - np.log(hi), -np.inf)


def _thresholds(x):
    u = np.unique(x[~np.isnan(x)])
    if len(u) <= 64:
        return (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    return np.unique(np.quantile(u, np.linspace(0.01, 0.99, 49)))


def _mask(x, ev):
    """Indicator of a single event (j, t, '>' or '<=') on a runs x releases matrix (NaN is in neither)."""
    j, t, side = ev
    with np.errstate(invalid="ignore"):
        return x[:, j] > t if side == ">" else x[:, j] <= t


def best_events(A, B, a) -> list[tuple]:
    """Candidate events chosen on the selection samples: per release, the threshold event with the largest
    eps_lb of A over B (score at level a). Sorted by score, best first: [(score, (j, t, side))]."""
    out = []
    for j in range(A.shape[1]):
        ts = _thresholds(np.concatenate([A[:, j], B[:, j]]))
        if not len(ts):
            continue
        with np.errstate(invalid="ignore"):
            ka = (A[:, j, None] > ts).sum(0), (A[:, j, None] <= ts).sum(0)
            kb = (B[:, j, None] > ts).sum(0), (B[:, j, None] <= ts).sum(0)
        for side, ka_s, kb_s in ((">", ka[0], kb[0]), ("<=", ka[1], kb[1])):
            s = eps_lb(ka_s, len(A), kb_s, len(B), a)
            i = int(np.argmax(s))
            if np.isfinite(s[i]):
                out.append((float(s[i]), (j, float(ts[i]), side)))
    return sorted(out, key=lambda e: -e[0])


def joint_event(A, B, singles, a):
    """The best conjunction of the top 2..JOINT_MAX single events (distinct releases): (score, [events]) or None."""
    evs = [e for _, e in singles[:JOINT_MAX]]
    best = None
    for k in range(2, len(evs) + 1):
        ma = np.logical_and.reduce([_mask(A, e) for e in evs[:k]])
        mb = np.logical_and.reduce([_mask(B, e) for e in evs[:k]])
        s = float(eps_lb(ma.sum(), len(A), mb.sum(), len(B), a))
        if np.isfinite(s) and (best is None or s > best[0]):
            best = (s, evs[:k])
    return best


def dp_test(A: np.ndarray, B: np.ndarray, alpha: float, n_hyp: int) -> list[dict]:
    """Statistical test of one neighbour pair. A, B: runs x releases on D and D'. The first half of each chooses
    the events, the second half tests them, each at level alpha / n_hyp (Bonferroni; n_hyp counts every event the
    whole audit tests). Returns one row per tested event: direction, events, counts, eps_lb."""
    ha, hb = len(A) // 2, len(B) // 2
    a = alpha / n_hyp
    rows = []
    for direction, (X, Y, hx, hy) in (("D>D'", (A, B, ha, hb)), ("D'>D", (B, A, hb, ha))):
        singles = best_events(X[:hx], Y[:hy], a)
        cands = [[e] for _, e in singles[:TOP_EVENTS]]
        joint = joint_event(X[:hx], Y[:hy], singles, a)
        if joint:
            cands.append(joint[1])
        for evs in cands:
            mx = np.logical_and.reduce([_mask(X[hx:], e) for e in evs])
            my = np.logical_and.reduce([_mask(Y[hy:], e) for e in evs])
            kx, ky, nx, ny = int(mx.sum()), int(my.sum()), len(X) - hx, len(Y) - hy
            rows.append({"direction": direction, "events": evs, "k": [kx, ky], "n": [nx, ny],
                         "eps_lb": float(eps_lb(kx, nx, ky, ny, a))})
    return rows


def hypotheses_per_neighbour() -> int:
    return 2 * (TOP_EVENTS + 1)


def deterministic(A: np.ndarray, B: np.ndarray) -> list[int]:
    """Releases constant within D's runs and within D''s runs, with different values: no or seeded noise."""
    same = lambda x: np.all((x == x[0]) | (np.isnan(x) & np.isnan(x[0])), axis=0)
    differ = ~((A[0] == B[0]) | (np.isnan(A[0]) & np.isnan(B[0])))
    return list(np.flatnonzero(same(A) & same(B) & differ))


# ---- running --------------------------------------------------------------------------------------------------
def run_batch(script: Path, files: list[Path], n: int, extra, settings, workers: int) -> tuple[list, list]:
    """n sandboxed runs on one dataset: (release dicts of the successful runs, errors)."""
    def one(_):
        try:
            r = run_sandboxed(script, data=files, extra=extra, settings=settings)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"[:300]
        if r.get("status") in ("no_code", "unsafe_not_run"):
            return None, r["status"]
        if str(r.get("returncode")) != "0":
            return None, ("".join((r.get("stderr") or "").strip().splitlines()[-1:]) or "timeout")[:300]
        rel = releases(r.get("stdout") or "")
        if not rel:
            return None, "no release in the output (no dp_* keys in a final JSON line)"
        eps = (final_json(r.get("stdout")) or {}).get("epsilon_total")
        return rel | ({PRINTED: float(eps)} if isinstance(eps, (int, float)) and not isinstance(eps, bool) else {}), None
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, range(n)))
    return [r for r, _ in res if r is not None], [e for _, e in res if e is not None]


def printed_epsilon(runs: list[dict]):
    """The epsilon_total the script prints (the median over its runs on D), or None."""
    eps = [r[PRINTED] for r in runs if PRINTED in r]
    return float(np.median(eps)) if eps else None


def audit(script: Path, task: str, n: int = N_DEFAULT, kinds=RECORDS, rows: int | None = None,
          workers: int = 6, budget: float = 1.0, confidence: float = 0.95, extra=(), data_dir: Path | None = None,
          timeout_s: int | None = None, deterministic_only: bool = False) -> dict:
    """Audit one script; returns the report (verdict PASS / VIOLATION / UNDERREPORTED / ERROR, per-neighbour
    results). deterministic_only: stop after the first batch (verdict VIOLATION, ERROR or PASS_DETERMINISTIC)."""
    t0 = time.time()
    script = Path(script)
    spec = SPECS[task]
    relation = spec.get("relation", "replace_one")
    data_dir = data_dir or ((APP_DIR / task / "data") if (APP_DIR / task).is_dir() else DATA_DIR)
    settings = None
    if timeout_s or app_dir(script) is None:  # a reference or a copy: the app's own eval settings by name
        settings = eval_settings(Path(f"app-{task.replace('_', '-')}_code_1.py") if (APP_DIR / task).is_dir()
                                 else script)
        if timeout_s:
            settings["timeout_s"] = timeout_s
    report = {"script": str(script), "task": task, "relation": relation, "budget": budget,
              "confidence": confidence, "n_requested": n, "records": list(kinds),
              "note": "PASS means no violation was found with this many runs; it is not a proof of DP."}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        sets = build_datasets(task, data_dir, Path(tmp), kinds, rows)
        report["neighbours"] = {k: v["record"] for k, v in sets.items() if k != "D"}
        report["rows"] = {f: len(read_frame(Path(tmp) / "D" / f)) for f in spec["files"]}
        try:
            runs = {k: run_batch(script, v["files"], min(N_FIRST, n), extra, settings, workers)
                    for k, v in sets.items()}
            errors = {k: e for k, (_, e) in runs.items() if e}
            if any(len(e) > MAX_ERROR_SHARE * min(N_FIRST, n) for e in errors.values()):
                return report | {"verdict": "ERROR", "errors": {k: e[:3] for k, e in errors.items()},
                                 "runtime_s": round(time.time() - t0, 1)}
            names = sorted({k for rs, _ in runs.values() for r in rs for k in r if k != PRINTED})
            mats = {k: matrix(rs, names) for k, (rs, _) in runs.items()}
            det = {k: deterministic(mats["D"], mats[k]) for k in sets if k != "D"}
            varies = lambda x: ~np.all((x == x[0]) | (np.isnan(x) & np.isnan(x[0])), axis=0)
            if not varies(mats["D"]).any():
                v = {"test": "deterministic", "detail": "every release is identical in every run on D: no noise or "
                     "seeded noise", "eps_lb": math.inf}
                found = [{"record": k, "release": names[j], "values": [mats["D"][0, j], mats[k][0, j]]}
                         for k, js in det.items() for j in js[:5]]
                return report | {"verdict": "VIOLATION", "violation": v | {"examples": found},
                                 "n_runs": {k: len(rs) for k, (rs, _) in runs.items()},
                                 "runtime_s": round(time.time() - t0, 1)}
            if any(det.values()):
                k, js = next((k, js) for k, js in det.items() if js)
                return report | {"verdict": "VIOLATION", "violation": {
                    "test": "deterministic", "record": k, "release": names[js[0]], "eps_lb": math.inf,
                    "detail": f"{len(js)} releases constant on D and on D' with different values: no or seeded noise",
                    "values": [mats["D"][0, js[0]], mats[k][0, js[0]]]},
                    "n_runs": {k: len(rs) for k, (rs, _) in runs.items()}, "runtime_s": round(time.time() - t0, 1)}
            if deterministic_only:
                return report | {"verdict": "PASS_DETERMINISTIC", "n_runs": {k: len(rs) for k, (rs, _) in runs.items()},
                                 "runtime_s": round(time.time() - t0, 1)}
            if n > N_FIRST:
                for k, v in sets.items():
                    rs, es = run_batch(script, v["files"], n - N_FIRST, extra, settings, workers)
                    runs[k] = (runs[k][0] + rs, runs[k][1] + es)
        finally:
            kill_sandbox_user()
    errors = {k: e for k, (_, e) in runs.items() if e}
    report["n_runs"] = {k: len(rs) for k, (rs, _) in runs.items()}
    report["errors"] = {k: {"count": len(e), "first": e[0]} for k, e in errors.items()}
    if any(len(e) > MAX_ERROR_SHARE * n for e in errors.values()):
        return report | {"verdict": "ERROR", "runtime_s": round(time.time() - t0, 1)}
    report["epsilon_printed"] = printed_epsilon(runs["D"][0])
    names = sorted({k for rs, _ in runs.values() for r in rs for k in r if k != PRINTED})
    mats = {k: matrix(rs, names) for k, (rs, _) in runs.items()}
    n_hyp = hypotheses_per_neighbour() * (len(sets) - 1)
    alpha = 1 - confidence
    results, worst = {}, None
    for k in sets:
        if k == "D":
            continue
        rows_ = dp_test(mats["D"], mats[k], alpha, n_hyp)
        for r in rows_:
            r["releases"] = [names[e[0]] for e in r["events"]]
            r["events"] = [f"{names[j]} {side} {t:.6g}" for j, t, side in r["events"]]
        top = max(rows_, key=lambda r: r["eps_lb"]) if rows_ else None
        per_release = {}
        for r in rows_:
            if len(r["releases"]) == 1:
                per_release[r["releases"][0]] = max(per_release.get(r["releases"][0], -math.inf), r["eps_lb"])
        results[k] = {"eps_lb": top["eps_lb"] if top else None, "best": top,
                      "per_release": dict(sorted(per_release.items(), key=lambda x: -x[1])[:TOP_EVENTS])}
        if top and (worst is None or top["eps_lb"] > worst[1]["eps_lb"]):
            worst = (k, top)
    report["tests"] = results
    report["n_hypotheses"] = n_hyp
    report["eps_lb"] = worst[1]["eps_lb"] if worst else None
    if worst and worst[1]["eps_lb"] > budget:
        report["verdict"] = "VIOLATION"
        report["violation"] = {"test": "statistical", "record": worst[0], "release": worst[1]["releases"],
                               "event": worst[1]["events"], "direction": worst[1]["direction"],
                               "eps_lb": worst[1]["eps_lb"], "counts": worst[1]["k"], "n": worst[1]["n"]}
    elif worst and report["epsilon_printed"] is not None and worst[1]["eps_lb"] > report["epsilon_printed"] * (1 + 1e-3):
        report["verdict"] = "UNDERREPORTED"  # within the budget, but more than the script says it spends
        report["violation"] = {"test": "statistical", "record": worst[0], "release": worst[1]["releases"],
                               "event": worst[1]["events"], "eps_lb": worst[1]["eps_lb"],
                               "epsilon_printed": report["epsilon_printed"]}
    else:
        report["verdict"] = "PASS"
    report["runtime_s"] = round(time.time() - t0, 1)
    return report


def summary(r: dict) -> str:
    """One line per script."""
    head = f"{r['verdict']:9} {Path(r['script']).name} [{r['task']}, {r['relation']}]"
    if r["verdict"] == "ERROR":
        return f"{head} errors: {json.dumps(r.get('errors'), default=str)[:300]}"
    if r["verdict"] in ("PASS_DETERMINISTIC", "NO_DETERMINISTIC_FINDING"):
        return f"{head} deterministic test passed on {r['n_runs']} ({r['runtime_s']} s); statistical test pending"
    if r["verdict"] == "UNDERREPORTED":
        v = r["violation"]
        return (f"{head} record {v['record']}, release {v['release']}: eps_lb {v['eps_lb']:.3g} > printed epsilon "
                f"{v['epsilon_printed']:g} (within the budget {r['budget']}) ({r['runtime_s']} s)")
    if r["verdict"] in ("VIOLATION", "DETERMINISTIC_VIOLATION"):
        v = r["violation"]
        ex = (v.get("examples") or [{}])[0]
        return (f"{head} {v['test']}: record {v.get('record', ex.get('record', '-'))}, "
                f"release {v.get('release', ex.get('release', 'all (constant output)'))}, "
                f"eps_lb {v['eps_lb']:.3g} > budget {r['budget']} ({r['runtime_s']} s)")
    lbs = ", ".join(f"{k} {t['eps_lb']:.3f}" for k, t in r["tests"].items() if t["eps_lb"] is not None)
    return (f"{head} eps_lb per record: {lbs}; N {r['n_runs']}; {r['runtime_s']} s "
            "(no violation found; not a proof of DP)")


# ---- grids ----------------------------------------------------------------------------------------------------
def grid_scripts(grid: Path) -> list[tuple[tuple, Path, str]]:
    """((cell, task, run), script, audit task) for every script of GRID/results.jsonl that ran (exit code 0) and
    whose task has an audit spec."""
    out = []
    path = grid / "results.jsonl"
    for line in (path.read_text(encoding="utf-8").splitlines() if path.is_file() else []):
        r = json.loads(line)
        if str(r.get("runnable")) != "True" or r.get("status") in ("unsafe_not_run", "no_code"):
            continue
        key = file_key(r["code_file"])
        script = grid / key[0] / f"{key[1]}_code_{key[2]}.py"
        task = task_of(script)
        if script.is_file() and task in SPECS:
            out.append((key, script, task))
    return sorted(set(out))


def load_audit(grid: Path) -> dict:
    """GRID/audit.jsonl as {(cell, task, run): the report that decides}: the full audit if there is one, else the
    deterministic-only entry. Empty if there is no audit."""
    return {k: m.get("full") or m.get("deterministic") for k, m in entries_by_mode(grid).items()}


def suggestion(r: dict | None) -> tuple[str | None, str]:
    """The grade an audit report suggests (docs/grading_rubric.md, rule 18) and its evidence. VIOLATION suggests
    W (no or seeded noise, or a measured loss above the budget); UNDERREPORTED suggests P (a measured loss above
    the printed epsilon, within the budget: rule 14). Any other verdict suggests nothing: the script goes on to
    hand grading."""
    if not r or r.get("verdict") not in ("VIOLATION", "DETERMINISTIC_VIOLATION", "UNDERREPORTED"):
        return None, (f"audit {r['verdict']}" if r else "")
    v = r["violation"]
    where = f"record {v.get('record') or (v.get('examples') or [{}])[0].get('record', '-')}"
    if v["test"] == "deterministic":
        return "W", f"audit VIOLATION: deterministic release ({where}): no or seeded noise"
    if r["verdict"] in ("VIOLATION", "DETERMINISTIC_VIOLATION"):
        return "W", f"audit VIOLATION: eps_lb {v['eps_lb']:.3g} > budget {r['budget']} ({where}, {v['release']})"
    return "P", (f"audit UNDERREPORTED: eps_lb {v['eps_lb']:.3g} > printed epsilon {v['epsilon_printed']:g} "
                 f"({where}, {v['release']})")


def entries_by_mode(grid: Path) -> dict:
    """GRID/audit.jsonl as {(cell, task, run): {mode: latest entry}}, mode "deterministic" or "full"."""
    path = Path(grid) / "audit.jsonl"
    out = {}
    for line in (path.read_text(encoding="utf-8").splitlines() if path.is_file() else []):
        r = json.loads(line)
        out.setdefault((r["cell"], r["task"], int(r["run"])), {})[r.get("mode", "full")] = r
    return out


DETERMINISTIC_VERDICT = {"VIOLATION": "DETERMINISTIC_VIOLATION", "PASS_DETERMINISTIC": "NO_DETERMINISTIC_FINDING"}


def audit_grid(grid: Path, n: int, workers: int, budget: float, confidence: float,
               deterministic_only: bool = False, only: list[Path] | None = None) -> None:
    """Audit the scripts of GRID that ran (or only the listed ones) -> GRID/audit.jsonl, resumable per mode.
    The deterministic test runs first on every script without a deterministic entry (verdict
    DETERMINISTIC_VIOLATION, NO_DETERMINISTIC_FINDING or ERROR, mode "deterministic"). Unless deterministic_only,
    the full audit (statistical test, mode "full") then runs on every script with NO_DETERMINISTIC_FINDING and no
    full entry yet."""
    log = grid / "audit.jsonl"
    todo = grid_scripts(grid)
    if only is not None:
        wanted = {Path(x).resolve() for x in only}
        todo = [t for t in todo if t[1].resolve() in wanted]
        missing = wanted - {t[1].resolve() for t in todo}
        if missing:
            print(f"not audited (not in {grid.name}/results.jsonl as run, or no audit spec): "
                  + ", ".join(sorted(str(m) for m in missing)), flush=True)

    def write(key, r, mode):
        with log.open("a", encoding="utf-8") as f:
            entry = r | {"audit_task": r["task"], "cell": key[0], "task": key[1], "run": key[2], "mode": mode}
            f.write(json.dumps(entry, default=str) + "\n")
        print(f"{key[0]} {key[1]} {key[2]}: {summary(r)}", flush=True)

    done = entries_by_mode(grid)
    first = [t for t in todo if "deterministic" not in done.get(t[0], {})]
    print(f"{grid.name}: {len(todo)} scripts; deterministic test on {len(first)}", flush=True)
    def safe_audit(script, task, **kw):
        try:
            return audit(script, task, n, workers=workers, budget=budget, confidence=confidence, **kw)
        except Exception as e:  # one failing script is an ERROR entry, not the end of the grid run
            return {"verdict": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}

    for key, script, task in first:
        r = safe_audit(script, task, deterministic_only=True)
        write(key, r | {"verdict": DETERMINISTIC_VERDICT.get(r["verdict"], r["verdict"])}, "deterministic")
    if deterministic_only:
        return
    done = entries_by_mode(grid)
    second = [t for t in todo if done.get(t[0], {}).get("deterministic", {}).get("verdict") == "NO_DETERMINISTIC_FINDING"
              and "full" not in done[t[0]]]
    print(f"statistical test on {len(second)} scripts (N = {n})", flush=True)
    for key, script, task in second:
        write(key, safe_audit(script, task), "full")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scripts", nargs="+", type=Path,
                   help="scripts; or a grid folder (audits GRID/results.jsonl), optionally followed by some of its "
                        "scripts to audit only those (results still go to GRID/audit.jsonl)")
    p.add_argument("--deterministic-only", action="store_true",
                   help="grid mode: only the deterministic test (10 runs per neighbour); a later full run on the "
                        "same grid adds the statistical test")
    p.add_argument("--app", help="paper app (applications/<app>) for scripts whose name does not say it")
    p.add_argument("--task", help="toy task (stat-count, selection-topk, ml-regression) for such scripts")
    p.add_argument("--n", type=int, default=N_DEFAULT,
                   help="runs per dataset (default 1000: detects a 2x under-charged count at budget 1 with "
                        "probability above 0.99; see the validation report)")
    p.add_argument("--records", default=",".join(RECORDS), help="comma-separated subset of " + ",".join(RECORDS))
    p.add_argument("--rows", type=int, help="random subset of this many rows per private file (default per app)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--extra", nargs="*", type=Path, default=[], help="files copied next to the script (its modules)")
    p.add_argument("--data-dir", type=Path, help="the task's data folder (default applications/<app>/data or data/)")
    p.add_argument("--timeout", type=int, help="seconds per run (default: the app's eval settings)")
    p.add_argument("--out", type=Path, help="append one JSON report per script to this file")
    a = p.parse_args()
    check_exec_allowed()
    if a.scripts[0].is_dir():
        audit_grid(a.scripts[0].resolve(), a.n, a.workers, a.budget, a.confidence, a.deterministic_only,
                   a.scripts[1:] or None)
        return
    kinds = [k for k in a.records.split(",") if k]
    for s in a.scripts:
        task = a.app or a.task or task_of(s)
        if task not in SPECS:
            raise SystemExit(f"{s}: unknown task {task}; pass --app or --task")
        r = audit(s, task, a.n, kinds, a.rows, a.workers, a.budget, a.confidence, tuple(a.extra), a.data_dir,
                  a.timeout)
        print(summary(r), flush=True)
        if a.out:
            with a.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
