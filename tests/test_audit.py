"""Checks of the pure parts of the DP audit: neighbour construction, release flattening and the test statistics on
synthetic samples. No script is run.

  PYTHONPATH=src python -m pytest tests/test_audit.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from llmdpeval import audit

N_HYP = audit.hypotheses_per_neighbour() * 3  # three neighbours, as in a default audit


def frame():
    return pd.DataFrame({"race": ["0", "1", "1", "0", "1", "1", "1", "0"],
                         "frisked": ["1", "0", "1", "0", "0", "1", "0", "1"],
                         "age": ["0", "2", "1", "1", "0", "2", "1", "0"],
                         "weight": ["1", "2", "0", "1", "0", "2", "2", "0"]})


def test_replace_one_changes_one_row_to_another_records_values():
    df = frame()
    spec = audit.SPECS["gopher_sqf"]
    for rec in audit.pick_records(df, spec):
        assert rec["donor"] is not None and rec["victim"] != rec["donor"]
        nb = audit.neighbour(df, rec)
        assert len(nb) == len(df)
        changed = [i for i in range(len(df)) if not (nb.iloc[i] == df.iloc[i]).all()]
        assert changed == [rec["victim"]]
        assert (nb.iloc[rec["victim"]].to_numpy() == df.iloc[rec["donor"]].to_numpy()).all()


def test_record_kinds():
    df = frame()
    recs = {r["kind"]: r for r in audit.pick_records(df, audit.SPECS["gopher_sqf"])}
    assert recs["typical"]["victim"] == len(df) // 2
    t = recs["typical"]
    assert df.race[t["donor"]] != df.race[t["victim"]] and df.frisked[t["donor"]] != df.frisked[t["victim"]]
    sc = audit.score(df, audit.SPECS["gopher_sqf"])
    assert sc[recs["extreme"]["donor"]] == sc.max() and sc[recs["extreme"]["victim"]] == sc.min()
    g = recs["group"]  # a record of the smaller group (race 0) takes the values of one of the larger group
    assert df.race[g["victim"]] == "0" and df.race[g["donor"]] == "1"
    assert df.frisked[g["victim"]] != df.frisked[g["donor"]]


def test_add_remove_drops_one_row():
    df = pd.DataFrame({"occupation": ["A", "A", "B", "A", "C"], "income": ["<=50K", ">50K", "<=50K", ">50K", ">50K"],
                       "age": ["30", "40", "90", "20", "50"]})
    recs = {r["kind"]: r for r in audit.pick_records(df, audit.SPECS["stat-count"])}
    assert all(r["donor"] is None for r in recs.values())
    assert recs["extreme"]["victim"] == 2  # the largest contribution (age 90)
    assert df.occupation[recs["group"]["victim"]] in ("B", "C")  # a smallest group
    nb = audit.neighbour(df, recs["typical"])
    assert len(nb) == len(df) - 1 and list(nb.age) == ["30", "40", "20", "50"]


def test_zhang_score_is_number_of_aes():
    df = pd.DataFrame({"window": ["2019", "2020", "2019"], "aes": ["1;2", "1;2;3;4", ""]})
    assert list(audit.SPECS["zhang_faers"]["score"](df)) == [2, 4, 0]
    recs = {r["kind"]: r for r in audit.pick_records(df, audit.SPECS["zhang_faers"])}
    assert recs["extreme"] == {"kind": "extreme", "victim": 2, "donor": 1}


def test_flatten_and_releases():
    flat = audit.flatten({"dp_acc": {"All": 0.7, "Black": 0.6}, "dp_top": ["a", "b"], "dp_coef": [1, 2]})
    assert flat == {"dp_acc.All": 0.7, "dp_acc.Black": 0.6, "dp_top∋a": 1.0, "dp_top∋b": 1.0,
                    "dp_coef[0]": 1.0, "dp_coef[1]": 2.0}
    out = 'text\n{"true_original_spd": -0.07, "dp_spd": -0.08, "epsilon_total": 1.0}\n'
    assert audit.releases(out) == {"dp_spd": -0.08}
    m = audit.matrix([{"dp_top∋a": 1.0}, {"dp_top∋b": 1.0}], ["dp_top∋a", "dp_top∋b"])
    assert m.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_deterministic_release_is_found():
    A = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 4.0]])
    B = np.array([[2.0, 5.5], [2.0, 4.5], [2.0, 6.5]])
    assert audit.deterministic(A, B) == [0]
    assert audit.deterministic(A, A.copy()) == []  # constant and unchanged: not a finding


def test_clopper_pearson_bounds():
    assert audit.cp_low(0, 100, 0.025) == 0 and audit.cp_high(100, 100, 0.025) == 1
    lo, hi = audit.cp_low(50, 100, 0.025), audit.cp_high(50, 100, 0.025)
    assert 0.39 < lo < 0.5 < hi < 0.61


def laplace_pair(scale, n, rng, sensitivity=1.0):
    return 100 + rng.laplace(0, scale, (n, 1)), 100 + sensitivity + rng.laplace(0, scale, (n, 1))


def max_lb(A, B):
    return max(r["eps_lb"] for r in audit.dp_test(A, B, 0.05, N_HYP))


def test_laplace_at_the_right_scale_passes():
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert max_lb(*laplace_pair(1.0, 1000, rng)) <= 1.0  # eps = 1 exactly


def test_laplace_at_half_the_scale_fails():
    rng = np.random.default_rng(1)
    hits = sum(max_lb(*laplace_pair(0.5, 1000, rng)) > 1.0 for _ in range(5))  # true eps = 2
    assert hits >= 4


def test_histogram_calibrated_for_add_remove_fails_under_replace_one():
    """One record moves between two bins; per-bin scale 1 gives eps 2, which needs a joint event to show."""
    rng = np.random.default_rng(2)
    base = np.array([50.0, 80.0, 20.0])
    lbs = []
    for _ in range(5):
        A = base + rng.laplace(0, 1.0, (1000, 3))
        B = base + np.array([-1.0, 1.0, 0.0]) + rng.laplace(0, 1.0, (1000, 3))
        lbs.append(max_lb(A, B))
    assert sum(x > 1.0 for x in lbs) >= 3, lbs
    A = base + rng.laplace(0, 2.0, (1000, 3))  # the right calibration (scale 2) passes
    B = base + np.array([-1.0, 1.0, 0.0]) + rng.laplace(0, 2.0, (1000, 3))
    assert max_lb(A, B) <= 1.0


def test_no_change_gives_no_finite_large_bound():
    rng = np.random.default_rng(3)
    A, B = rng.laplace(0, 1, (1000, 4)), rng.laplace(0, 1, (1000, 4))
    assert max_lb(A, B) < 0.5 and not math.isnan(max_lb(A, B))


def test_complete_rows_only_when_counts_are_public():
    df = frame()
    df.loc[[0, 4], "age"] = ""  # incomplete rows must be neither victim nor donor
    for rec in audit.pick_records(df, {**audit.SPECS["gopher_sqf"], "complete": True}):
        assert {rec["victim"], rec["donor"]}.isdisjoint({0, 4}), rec


def test_app_median_object():
    from llmdpeval.app_grade import median_object
    runs = [{"dp_spd": -0.08, "dp_top": ["a", "b"], "dp_acc": {"All": 0.7}},
            {"dp_spd": -0.06, "dp_top": ["a", "c"], "dp_acc": {"All": 0.8}},
            {"dp_spd": -0.07, "dp_top": ["a", "b"], "dp_acc": {"All": 0.9}}]
    m = median_object(runs)
    assert m["dp_spd"] == -0.07 and m["dp_acc"] == {"All": 0.8} and m["dp_top"] == ["a", "b"]


# ---- worst-case records, grid mode and grade suggestions -------------------------------------------------------
def gopher_frame():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({c: rng.integers(0, 2, n).astype(str) for c in audit._GOPHER_BIN})
    for c in ("build", "age", "ht_feet", "weight"):
        df[c] = rng.integers(0, 3, n).astype(str)
    df["race"] = np.where(np.arange(n) < 15, "0", "1")
    df["frisked"] = np.where(np.arange(n) % 4 == 0, "1", "0")
    io = rng.integers(0, 2, n)
    df["inout_I"], df["inout_O"] = io.astype(str), (1 - io).astype(str)
    sx = (rng.random(n) < 0.1).astype(int)
    df["sex_F"], df["sex_M"] = sx.astype(str), (1 - sx).astype(str)
    return df


def test_worst_tabular_is_in_the_public_domain():
    df, spec = gopher_frame(), audit.SPECS["gopher_sqf"]
    v, row = audit.worst_tabular(df, spec)
    assert df.race[v] == "1" and df.frisked[v] == "0"  # a typical record of the larger group, majority label
    assert row["race"] == "0" and row["frisked"] == "1"  # moved to the smaller group with the rarer label
    for c, (lo, hi) in spec["bounds"].items():
        assert float(row[c]) in (lo, hi), c
    assert row["sex_F"] == "1" and row["sex_M"] == "0"  # the rarer member of the one-hot pair
    assert {row["inout_I"], row["inout_O"]} == {"0", "1"}
    nb = audit.neighbour(df, {"victim": v, "donor": "crafted", "row": row})
    assert len(nb) == len(df) and (nb.iloc[v] == row[df.columns]).all()
    assert sum(not (nb.iloc[i] == df.iloc[i]).all() for i in range(len(df))) == 1


def test_worst_zhang_report_at_the_cap_near_the_boundary():
    rng = np.random.default_rng(1)
    terms = pd.DataFrame({"meddra_id": [str(i) for i in range(20)], "drop_before_test": "0",
                          "drop_from_table": "0"}).set_index("meddra_id")
    reports = pd.DataFrame({"window": rng.choice(["2019", "2020"], 400),
                            "aes": [";".join(map(str, rng.choice(20, rng.integers(0, 4), replace=False)))
                                    for _ in range(400)]})
    v, row = audit.worst_zhang(reports, audit.SPECS["zhang_faers"], terms)
    aes = row["aes"].split(";")
    assert len(aes) == audit.SPECS["zhang_faers"]["k"] and set(aes) <= set(terms.index)
    assert row["window"] == reports.window[v]  # same window: only the AE counts move
    assert audit._n_aes(reports)[v] == audit._n_aes(reports[reports.window == row["window"]]).min()


def test_grid_scripts_and_suggestions(tmp_path):
    cell = tmp_path / "m_ragk3_None_th_code_thinkon"
    cell.mkdir()
    (cell / "app-gopher-sqf_code_1.py").write_text("print(1)\n")
    (cell / "app-gopher-sqf_code_2.py").write_text("print(1)\n")
    with (tmp_path / "results.jsonl").open("w") as f:
        f.write('{"code_file": "x/m_ragk3_None_th_code_thinkon/app-gopher-sqf_code_1.py", "runnable": "True"}\n')
        f.write('{"code_file": "x/m_ragk3_None_th_code_thinkon/app-gopher-sqf_code_2.py", "runnable": "False"}\n')
    got = audit.grid_scripts(tmp_path)
    assert [(k, t) for k, _, t in got] == [(("m_ragk3_None_th_code_thinkon", "app-gopher-sqf", 1), "gopher_sqf")]
    det = {"verdict": "VIOLATION", "budget": 1.0, "violation": {"test": "deterministic", "record": "typical",
                                                                 "eps_lb": math.inf}}
    stat = {"verdict": "VIOLATION", "budget": 1.0, "violation": {"test": "statistical", "record": "worst",
                                                                  "release": ["dp_spd"], "eps_lb": 1.7}}
    under = {"verdict": "UNDERREPORTED", "budget": 1.0, "violation": {
        "test": "statistical", "record": "group", "release": ["dp_spd"], "eps_lb": 0.6, "epsilon_printed": 0.3}}
    assert audit.suggestion(det)[0] == "W" and "seeded" in audit.suggestion(det)[1]
    assert audit.suggestion(stat)[0] == "W" and "1.7" in audit.suggestion(stat)[1]
    assert audit.suggestion(under)[0] == "P"
    assert audit.suggestion({"verdict": "PASS"})[0] is None and audit.suggestion(None) == (None, "")
    with (tmp_path / "audit.jsonl").open("w") as f:
        f.write(json.dumps({"cell": "c", "task": "t", "run": 1, "verdict": "PASS_DETERMINISTIC"}) + "\n")
        f.write(json.dumps({"cell": "c", "task": "t", "run": 1} | stat) + "\n")
    assert audit.load_audit(tmp_path)[("c", "t", 1)]["verdict"] == "VIOLATION"  # the latest report wins


def test_printed_epsilon_and_underreported_verdict():
    assert audit.printed_epsilon([{audit.PRINTED: 0.5}, {audit.PRINTED: 0.5}, {}]) == 0.5
    assert audit.printed_epsilon([{}]) is None


def test_app_grade_and_triage_use_the_audit(tmp_path):
    from llmdpeval import app_grade, grade
    info = {"status": "", "runnable": True}
    assert app_grade.suggest("ok", info, [], "keeps", False, "W") == "W"
    assert app_grade.suggest("ok", info, [], "keeps", False, "P") == "P"
    assert app_grade.suggest("ok", info, [], "loses", False, "P") == "U"  # rule 14: P, or U if not useful
    assert app_grade.suggest("ok", info, [], "keeps", False, None) == "G"
    cell = tmp_path / "m_ragk3_None_th_code_thinkon"
    cell.mkdir()
    (cell / "stat-count_code_1.py").write_text("import numpy as np\nnp.random.seed(0)\n")
    (tmp_path / "audit.jsonl").write_text(json.dumps(
        {"cell": cell.name, "task": "stat-count", "run": 1, "verdict": "VIOLATION", "budget": 1.0,
         "violation": {"test": "deterministic", "record": "typical", "eps_lb": "inf"}}) + "\n")
    grade.triage(tmp_path)
    row = next(iter(__import__("csv").DictReader((tmp_path / "triage.csv").open(encoding="utf-8"))))
    assert row["audit_verdict"] == "VIOLATION" and row["audit_suggested"] == "W"


def fake_grid(tmp_path, n=3):
    cell = tmp_path / "m_ragk3_None_th_code_thinkon"
    cell.mkdir()
    with (tmp_path / "results.jsonl").open("w") as f:
        for i in range(1, n + 1):
            (cell / f"app-gopher-sqf_code_{i}.py").write_text("print(1)\n")
            f.write(json.dumps({"code_file": f"x/{cell.name}/app-gopher-sqf_code_{i}.py", "runnable": "True"}) + "\n")
    return cell


def test_grid_deterministic_only_then_full_resumes_per_mode(tmp_path, monkeypatch):
    cell = fake_grid(tmp_path)
    calls = []

    def fake_audit(script, task, n, workers=6, budget=1.0, confidence=0.95, deterministic_only=False, **kw):
        run = int(Path(script).stem.rsplit("_", 1)[1])
        calls.append((run, deterministic_only))
        if deterministic_only:  # run 1 is seeded; runs 2 and 3 show no deterministic finding
            return {"script": str(script), "task": task, "relation": "replace_one", "budget": budget,
                    "verdict": "VIOLATION" if run == 1 else "PASS_DETERMINISTIC", "n_runs": {}, "runtime_s": 0,
                    "violation": {"test": "deterministic", "record": "typical", "eps_lb": math.inf}}
        return {"script": str(script), "task": task, "relation": "replace_one", "budget": budget,
                "verdict": "PASS", "tests": {}, "n_runs": {}, "runtime_s": 0}
    monkeypatch.setattr(audit, "audit", fake_audit)
    audit.audit_grid(tmp_path, 1000, 6, 1.0, 0.95, deterministic_only=True)
    assert calls == [(1, True), (2, True), (3, True)]
    got = audit.load_audit(tmp_path)
    assert got[(cell.name, "app-gopher-sqf", 1)]["verdict"] == "DETERMINISTIC_VIOLATION"
    assert got[(cell.name, "app-gopher-sqf", 2)]["verdict"] == "NO_DETERMINISTIC_FINDING"
    assert audit.suggestion(got[(cell.name, "app-gopher-sqf", 1)])[0] == "W"
    assert audit.suggestion(got[(cell.name, "app-gopher-sqf", 2)])[0] is None
    calls.clear()  # a later full run: no deterministic re-run, the statistical test only where nothing was found
    audit.audit_grid(tmp_path, 1000, 6, 1.0, 0.95)
    assert calls == [(2, False), (3, False)]
    assert audit.load_audit(tmp_path)[(cell.name, "app-gopher-sqf", 2)]["verdict"] == "PASS"
    calls.clear()
    audit.audit_grid(tmp_path, 1000, 6, 1.0, 0.95)  # everything done: nothing runs
    assert calls == []


def test_grid_audits_only_the_listed_scripts(tmp_path, monkeypatch):
    cell = fake_grid(tmp_path)
    seen = []
    monkeypatch.setattr(audit, "audit", lambda script, task, n, deterministic_only=False, **kw: seen.append(
        (Path(script).name, deterministic_only)) or {"script": str(script), "task": task, "relation": "r",
                                                     "budget": 1.0, "verdict": "PASS_DETERMINISTIC" if
                                                     deterministic_only else "PASS", "tests": {}, "n_runs": {},
                                                     "runtime_s": 0})
    audit.audit_grid(tmp_path, 1000, 6, 1.0, 0.95, only=[cell / "app-gopher-sqf_code_2.py"])
    assert seen == [("app-gopher-sqf_code_2.py", True), ("app-gopher-sqf_code_2.py", False)]
    assert set(audit.load_audit(tmp_path)) == {(cell.name, "app-gopher-sqf", 2)}  # written to GRID/audit.jsonl


def test_triage_reads_deterministic_only_verdicts(tmp_path):
    from llmdpeval import grade
    cell = tmp_path / "m_ragk3_None_th_code_thinkon"
    cell.mkdir()
    for i in (1, 2):
        (cell / f"stat-count_code_{i}.py").write_text("print(1)\n")
    with (tmp_path / "audit.jsonl").open("w") as f:
        f.write(json.dumps({"cell": cell.name, "task": "stat-count", "run": 1, "mode": "deterministic",
                            "verdict": "DETERMINISTIC_VIOLATION", "budget": 1.0,
                            "violation": {"test": "deterministic", "record": "group", "eps_lb": "inf"}}) + "\n")
        f.write(json.dumps({"cell": cell.name, "task": "stat-count", "run": 2, "mode": "deterministic",
                            "verdict": "NO_DETERMINISTIC_FINDING", "budget": 1.0}) + "\n")
    grade.triage(tmp_path)
    rows = {r["run"]: r for r in __import__("csv").DictReader((tmp_path / "triage.csv").open(encoding="utf-8"))}
    assert rows["1"]["audit_suggested"] == "W" and "seeded" in rows["1"]["audit_evidence"]
    assert rows["2"]["audit_verdict"] == "NO_DETERMINISTIC_FINDING" and rows["2"]["audit_suggested"] == ""
