"""Regression checks: grading metrics, sandbox, prompts, retrieval helpers, grid bookkeeping. Plain asserts, no LLM,
no network.

  PYTHONPATH=src python -m pytest tests/test_grid_and_grading.py

Module self-checks cover the rest: python -m llmdpeval.helper | llmdpeval.sandbox | llmdpeval.repair --selftest.
"""
import csv
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from llmdpeval import check_corpus, grade, prompts, retrieval, run_grid, sandbox

CELL = "gemma4-latest_ragv2gk3_OpenDP_th_code"


def make_grid(d: Path, scripts: dict, grades: list, results: list) -> Path:
    (d / CELL).mkdir(parents=True)
    for name, code in scripts.items():
        (d / CELL / name).write_text(code, encoding="utf-8")
    with (d / "hand_grades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "task", "run", "grade", "reason"])
        w.writerows([CELL, *g] for g in grades)
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    return d


def test_table_narrower_than_hand_grades():
    with tempfile.TemporaryDirectory() as d:
        ok = {"runnable": "True", "stderr": ""}
        grid = make_grid(Path(d), {}, [("stat-count", 1, "G", "x"), ("stat-count", 2, "U", "x"),
                                       ("selection-topk", 1, "G", "x")],
                         [ok | {"code_file": f"g/{CELL}/{t}_code_{n}.py"}
                          for t, n in (("stat-count", 1), ("stat-count", 2), ("selection-topk", 1))])
        with redirect_stdout(io.StringIO()):
            grade.table(grid, 1, "hand_grades.csv", None, ["stat-count"])  # no sys.exit for grades outside the slice
        assert (grid / "grades.csv").read_text().count("\n") == 2


def test_topk_scores_only_the_top5_list():
    full = ("{'Adm-clerical': 0.13, 'Armed-Forces': 0.0, 'Craft-repair': 0.22, 'Exec-managerial': 0.48, "
            "'Farming-fishing': 0.11, 'Prof-specialty': 0.44, 'Protective-serv': 0.32, 'Sales': 0.27, "
            "'Tech-support': 0.30, 'Transport-moving': 0.20}")
    out = ("True top 5: ['Exec-managerial', 'Prof-specialty', 'Protective-serv', 'Tech-support', 'Sales']\n"
           "Private top 5: ['Exec-managerial', 'Adm-clerical', 'Sales', 'Craft-repair', 'Armed-Forces']\n"
           f"Private noisy rates: {full}\n")
    assert grade.metric("selection-topk", out) == 2, grade.metric("selection-topk", out)
    # all 14 candidates printed in a DP section: must not score 5/5
    cands = "['" + "', '".join(grade.OCCUPATIONS) + "']"
    assert grade.metric("selection-topk", f"--- DP Top-K ---\nCandidates: {cands}\nerror\n") == 1
    # unchanged for a plain correct list
    assert grade.metric("selection-topk", out.replace("'Adm-clerical', 'Sales', 'Craft-repair', 'Armed-Forces'",
                        "'Prof-specialty', 'Protective-serv', 'Tech-support', 'Sales'")) == 5


def test_unknown_task_has_no_metric():
    assert grade.metric("app-gopher-sqf", "DP accuracy: 0.8\n") is None
    assert grade.useful("app-gopher-sqf", 0.8) is None


def test_median_skips_apps_and_survives_a_failing_script():
    with tempfile.TemporaryDirectory() as d:
        grid = make_grid(Path(d), {"app-gopher-sqf_code_1.py": "print(1)\n",
                                   "stat-count_code_1.py": "print('DP count: 7509')\n",
                                   "stat-count_code_2.py": "print('DP count: 7510')\n",
                                   "stat-count_code_3.py": "import shutil\nprint('DP count: 7508')\n"},
                         [("app-gopher-sqf", 1, "G", "a"), ("stat-count", 1, "G", "b"), ("stat-count", 2, "G", "c"),
                          ("stat-count", 3, "G", "d")], [])
        real = grade.run_sandboxed

        def flaky(path):
            if path.name == "stat-count_code_2.py":
                raise OSError("harness failure")
            return real(path)
        grade.run_sandboxed = flaky
        try:
            with redirect_stdout(io.StringIO()):
                grade.median(grid, 2, 2)
        finally:
            grade.run_sandboxed = real
        rows = {r["task"] + r["run"]: r for r in csv.DictReader((grid / "hand_grades_median.csv").open())}
        assert "no automatic usefulness metric" in rows["app-gopher-sqf1"]["reason"]
        assert "median of 2 runs: 1.00" in rows["stat-count1"]["reason"], rows["stat-count1"]
        assert "hand grade kept" in rows["stat-count2"]["reason"]
        assert "unsafe_not_run" in rows["stat-count3"]["reason"]
        log = [json.loads(l) for l in (grid / "rerun_median.jsonl").open()]
        assert any("harness failure" in x.get("error", "") for x in log)


def test_output_flags_whitespace_stderr():
    assert grade.output_flags({"runnable": "True", "stdout": "", "stderr": " \n"}, "stat-count")["stderr_tail"] == ""


def test_resume_needs_repair_log():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stat-count_code_1.py"
        f.write_text("print(1)\n")
        assert run_grid.done(f, 0) and not run_grid.done(f, 2)
        f.with_suffix(".repair.json").write_text("[]")
        assert run_grid.done(f, 2)


def test_evaluate_writes_each_result():
    with tempfile.TemporaryDirectory() as d:
        grid = make_grid(Path(d), {"stat-count_code_1.py": "print(1)\n", "stat-count_code_2.py": "print(2)\n"},
                         [], [])
        real = run_grid.run_sandboxed

        def crash_second(path):
            if path.name.endswith("_2.py"):
                raise PermissionError("cleanup failed")
            return real(path)
        run_grid.run_sandboxed = crash_second
        try:
            with redirect_stdout(io.StringIO()):
                run_grid.evaluate(grid)
            raise AssertionError("expected the PermissionError")
        except PermissionError:
            pass
        finally:
            run_grid.run_sandboxed = real
        lines = (grid / "results.jsonl").read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["stdout"] == "1\n", lines


def test_rag_fallback_uses_whole_prompt():
    prompt = "You are given two csv files.\n\nHere is code that trains a logistic regression on `frisked`.\n"
    query = retrieval.make_query(retrieval.release_prompt(retrieval.fallback_release(prompt), prompt), "fields")
    assert "logistic regression" in query, query


def test_index_stamp_tracks_config():
    before = retrieval.index_stamp("nomic-embed-text:latest")
    assert retrieval.index_stamp("other-embedder") != before
    keys = retrieval.EMBED_KEYS
    retrieval.EMBED_KEYS = keys + ["api_calls"]
    try:
        assert retrieval.index_stamp("nomic-embed-text:latest") != before
    finally:
        retrieval.EMBED_KEYS = keys


def test_skeleton_tests_get_no_keys():
    os.environ["SPHERE_LLM_API_KEY"] = "dummy-for-test"
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, '{"epsilon": 1.0}\n', ""

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        return Done()
    with tempfile.TemporaryDirectory() as d:
        sk = Path(d) / "x.py"
        sk.write_text("def dp_x():\n    return 1\n\ndef _test():\n    return {'epsilon': 1.0}\n")
        sk.with_suffix(".md").write_text("outline")
        real = check_corpus.subprocess.run
        check_corpus.subprocess.run = fake_run
        try:
            assert check_corpus.check_skeleton(sk) == []
        finally:
            check_corpus.subprocess.run = real
    assert seen and not [k for k in seen if "KEY" in k], sorted(seen)


def test_xdc_guard():
    real_host, real_platform = sandbox.socket.gethostname, sys.platform
    sandbox.socket.gethostname = lambda: "myxdc-someproject"
    sys.platform = "linux"
    try:
        assert "XDC" in sandbox.exec_refused_reason()
    finally:
        sandbox.socket.gethostname, sys.platform = real_host, real_platform
    import subprocess
    env = {**os.environ, "LLMDPEVAL_NO_EXEC": "1", "PYTHONPATH": "src"}
    for args in (["llmdpeval.run_grid", "--grid-dir", "nonexistent_grid", "--repair", "1"],
                 ["llmdpeval.run_grid", "--grid-dir", "nonexistent_grid", "--evaluate"],
                 ["llmdpeval.grade", "median", "nonexistent_grid"]):
        p = subprocess.run([sys.executable, "-m", *args], env=env, capture_output=True, text=True)
        assert p.returncode != 0 and "Refusing to run generated code" in p.stderr, (args, p.stderr[-300:])
    assert not Path("nonexistent_grid").exists()


def test_task_type_boost_and_composition_slot():
    """Boost moves matching task types first (stable); the budget item is an extra slot, pure DP only;
    analyze_code ends its prompt with the JSON reminder and keeps only vocabulary task types."""
    from types import SimpleNamespace as NS
    from llama_index.core import Settings

    def node(i, tt, notion="pure"):
        return NS(node=NS(node_id=i, metadata={"task_type": tt, "dp_notion": notion}))
    run = [node("a", "mean"), node("b", "histogram"), node("c", "count"), node("d", "histogram")]
    assert [n.node.node_id for n in retrieval.boost_by_task_type(run, ["histogram"], 3)] == ["b", "d", "a"]
    assert retrieval.boost_by_task_type(run, [], 2) == run[:2]

    comp = [node("z", "composition", "zCDP"), node("y", "composition")]
    real = retrieval.retrieve_nodes
    retrieval.retrieve_nodes = lambda index, prompt, *a: comp if prompt == retrieval.COMPOSITION_QUERY else [
        node(f"{prompt[:14]}{j}", "count") for j in range(retrieval.BOOST_DEPTH)]
    try:
        rels = [{"query_type": q, "definition": "d", "target_columns": [], "conditions": [], "task_types": ["count"]}
                for q in ("Q1", "Q2", "Q3")]
        out = [n.node.node_id for n in retrieval.plan_nodes(None, rels, "", "None", 3, "fields", "framework")]
        assert len(out) == retrieval.MAX_PLAN_CONTEXTS + 1 and out[-1] == "y", out  # zCDP item skipped, cap kept
        assert len(retrieval.plan_nodes(None, rels[:1], "", "None", 3, "fields", "framework")) == 3  # 1 release: no slot
    finally:
        retrieval.retrieve_nodes = real

    class FakeLLM:
        def complete(self, prompt):
            assert prompt.rstrip().endswith("fenced block.") and "TASK-TEXT" in prompt
            return NS(text='```json\n[{"query_type": "Count", "definition": "d", "target_columns": [], '
                           '"conditions": [], "task_types": ["count", "bogus", "mean", "sum"]}]\n```')
    old = Settings._llm
    Settings._llm = FakeLLM()
    try:
        releases, _ = prompts.analyze_code("TASK-TEXT")
    finally:
        Settings._llm = old
    assert releases[0]["task_types"] == ["count", "mean"], releases


def test_pipeline_freeze():
    """pipeline.toml feeds the defaults; a grid keeps the settings it was frozen with."""
    from argparse import Namespace
    from llmdpeval.helper import LOCK_FILE, PIPELINE_CONFIG
    assert retrieval.PIPELINE == {"query": PIPELINE_CONFIG["query"], "filter": PIPELINE_CONFIG["filter"]}
    o = Namespace(models=["gemma4:latest"], conditions=["OpenDP-rag-k3"], runs=4, template="guided", theory=True,
                  code=True, holdout=False, think="off", repair=0, reask=False, hard_cap=1800, lock=LOCK_FILE)
    with tempfile.TemporaryDirectory() as tmp:
        grid = Path(tmp)
        run_grid.freeze_pipeline(grid, o, "abc123")
        frozen = json.loads((grid / "pipeline_frozen.json").read_text())
        assert frozen["config"]["theory"] is True and frozen["config"]["embedder"] == PIPELINE_CONFIG["embedder"]
        assert frozen["corpus"] == retrieval.corpus_hash() and frozen["lock"].startswith(LOCK_FILE.name)
        o.models, o.runs = ["qwen3:32b"], 6  # cell selection may change between runs of one grid
        with redirect_stdout(io.StringIO()):
            run_grid.freeze_pipeline(grid, o, "def456")  # a new commit is only a note
        o.theory = False
        try:
            run_grid.freeze_pipeline(grid, o, "abc123")
            raise AssertionError("expected a refusal")
        except SystemExit as e:
            assert "theory: frozen True, now False" in str(e), e
        assert json.loads((grid / "pipeline_frozen.json").read_text()) == frozen  # never overwritten


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print("ok", name)
    print(f"{len(tests)} checks ok")
