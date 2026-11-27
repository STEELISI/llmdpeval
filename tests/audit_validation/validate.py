"""Validation of the DP audit (llmdpeval.audit) before use: it must pass the reference DP implementations and fail
known-bad scripts. Every script runs in the sandbox, so this runs only on the SPHERE evaluation node.

  PYTHONPATH=src python tests/audit_validation/validate.py [--only NAME ...] [--n 1000] [--gen-root DIR] [--out FILE]

Cases: the five references through small wrappers that print the prompt's dp_* JSON line (the references are not
edited); a seeded copy of the Gopher reference (random_state=0 on the DP model); the Zhang reference without noise;
an OpenDP race histogram calibrated for add/remove under replace-one, with its correct twin; a toy count at half and
at the right Laplace scale; three held-out known-bad variants; and real generated scripts graded W (paths
under --gen-root, see GENERATED).
"""
import argparse
import json
import tempfile
from pathlib import Path

from llmdpeval import audit
from llmdpeval.helper import APP_DIR

HERE = Path(__file__).parent
APPS = ("gopher_sqf", "kulynych_adult", "frymer_cces", "zhang_faers", "imai_fl")


def modules(app):
    return [APP_DIR / app / m for m in ("reference_dp.py", "original.py")]


def variant(tmp: Path, app: str, name: str, edits: list[tuple[str, str]]) -> Path:
    """A copy of an app's reference_dp.py with text edits (each must apply), in tmp/<name>/reference_dp.py."""
    src = (APP_DIR / app / "reference_dp.py").read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, f"{name}: edit point not found: {old}"
        src = src.replace(old, new)
    (tmp / name).mkdir()
    (tmp / name / "reference_dp.py").write_text(src, encoding="utf-8")
    return tmp / name / "reference_dp.py"


def cases(tmp: Path) -> list[dict]:
    """The constructed cases: name, script, task, files copied next to it, expected verdict."""
    seeded = variant(tmp, "gopher_sqf", "seeded", [("accountant=acc).fit(", "accountant=acc, random_state=0).fit(")])
    # held-out cases, written after the worst-case records and before any run of them
    zhang_k = variant(tmp, "zhang_faers", "zhang_sens2", [("                             2 * k, BUDGET",
                                                           "                             2, BUDGET"),
                                                          ('m_ae.map(2 * k)', 'm_ae.map(2)')])
    gopher_half = variant(tmp, "gopher_sqf", "gopher_din1", [("D_IN = 2  #", "D_IN = 1  #")])
    c = [{"name": f"reference {app}", "script": HERE / f"ref_{app}.py", "task": app, "extra": modules(app),
          "expected": "PASS"} for app in APPS]
    c += [
        {"name": "seeded Gopher reference", "script": HERE / "ref_gopher_sqf.py", "task": "gopher_sqf",
         "extra": [seeded, APP_DIR / "gopher_sqf" / "original.py"], "expected": "VIOLATION"},
        {"name": "Zhang reference without noise", "script": HERE / "zhang_nonoise.py", "task": "zhang_faers",
         "extra": modules("zhang_faers"), "expected": "VIOLATION"},
        {"name": "race histogram, add/remove calibration", "script": HERE / "hist_race_add_remove.py",
         "task": "kulynych_adult", "extra": [], "expected": "VIOLATION"},
        {"name": "race histogram, replace-one calibration", "script": HERE / "hist_race_replace_one.py",
         "task": "kulynych_adult", "extra": [], "expected": "PASS"},
        {"name": "toy count, half Laplace scale", "script": HERE / "count_half_scale.py", "task": "stat-count",
         "extra": [], "expected": "VIOLATION"},
        {"name": "toy count, right Laplace scale", "script": HERE / "count_right_scale.py", "task": "stat-count",
         "extra": [], "expected": "PASS"},
        {"name": "held-out: Kulynych per-race rate at sensitivity 1/n", "script": HERE / "heldout_rate_1overN.py",
         "task": "kulynych_adult", "extra": [], "expected": "VIOLATION"},
        {"name": "held-out: Zhang reference, AE histogram at sensitivity 2 instead of 2K",
         "script": HERE / "ref_zhang_faers.py", "task": "zhang_faers",
         "extra": [zhang_k, APP_DIR / "zhang_faers" / "original.py"], "expected": "VIOLATION"},
        {"name": "held-out: Gopher reference, accuracy and SPD at d_in 1 (2x under-charge, 0.4 of the budget)",
         "script": HERE / "ref_gopher_sqf.py", "task": "gopher_sqf",
         "extra": [gopher_half, APP_DIR / "gopher_sqf" / "original.py"], "expected": "VIOLATION"},
    ]
    return c


# Generated scripts graded W by hand, relative to --gen-root (the folder holding the grids).
GENERATED = [
    ("gopher qwen3.6 None 1: SPD sensitivity 1/N_test", "gopher_sqf",
     "grid_gopher_v4final_2026-11-10/qwen3.6-27b_ragv2gk3_None_th_code_thinkon/app-gopher-sqf_code_1.py"),
    ("gopher qwen3.6 Diffprivlib 2: group sizes public (tools.mean per race)", "gopher_sqf",
     "grid_gopher_v4final_2026-11-10/qwen3.6-27b_ragv2gk3_Diffprivlib_th_code_thinkon/app-gopher-sqf_code_2.py"),
    ("frymer qwen3.6 None 3 (repaired): default_rng(42) noise, SE from exact group sizes", "frymer_cces",
     "grid_frymer_r1_2026-11-12_fix/qwen3.6-27b_ragk3_None_th_code_thinkon/app-frymer-cces_code_3.py"),
    ("zhang qwen3.6 None 3: joint histogram at K (add/remove), true cost 1.75", "zhang_faers",
     "grid_zhang_r1_2026-11-12/qwen3.6-27b_ragk3_None_th_code_thinkon/app-zhang-faers_code_3.py"),
]
# Out of scope for an output test: the seeded model is never released, only unseeded noisy metrics of it
# (the static seed rule covers these). Run with --out-of-scope.
OUT_OF_SCOPE = [
    ("commercial gpt-5-mini gopher: seeded DP model (random_state=SEED)", "gopher_sqf",
     "grid_commercial_test_2026-11-18/gpt-5-mini_ragk3_Diffprivlib_th_code_thinkon/app-gopher-sqf_code_1.py"),
    ("commercial gpt-5-mini kulynych: seeded DP model (random_state=SEED)", "kulynych_adult",
     "grid_commercial_test_2026-11-18/gpt-5-mini_ragk3_Diffprivlib_th_code_thinkon/app-kulynych-adult_code_1.py"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="substrings of case names")
    p.add_argument("--n", type=int, default=audit.N_DEFAULT)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--gen-root", type=Path, help="folder holding the generated grids of GENERATED")
    p.add_argument("--out-of-scope", action="store_true", help="also run the OUT_OF_SCOPE scripts")
    p.add_argument("--out", type=Path, default=Path("audit_validation.jsonl"))
    a = p.parse_args()
    audit.check_exec_allowed()
    with tempfile.TemporaryDirectory() as tmp:
        todo = cases(Path(tmp))
        if a.gen_root:
            todo += [{"name": name, "script": a.gen_root / path, "task": task, "extra": [], "expected": "VIOLATION"}
                     for name, task, path in GENERATED + (OUT_OF_SCOPE if a.out_of_scope else [])]
        if a.only:
            todo = [c for c in todo if any(s in c["name"] for s in a.only)]
        for c in todo:
            r = audit.audit(c["script"], c["task"], a.n, workers=a.workers, extra=tuple(c["extra"]))
            r |= {"case": c["name"], "expected": c["expected"]}
            print(f"{'ok ' if r['verdict'] == c['expected'] else 'BAD'} {c['name']}: expected {c['expected']}; "
                  + audit.summary(r), flush=True)
            with a.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
