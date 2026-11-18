"""Checks of repair (task prompt, failure restore), no-code grading, per-run sandbox cleanup, killed generations on
resume, robust app grading, code-fence choice, the AST unsafe check, the fields query and the hash helper.
Plain asserts, no LLM, no network.

  PYTHONPATH=src python -m pytest tests/test_repair_and_sandbox.py
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from llmdpeval import app_grade, grade, privacy_check, repair, retrieval, run_grid, sandbox
from llmdpeval.helper import APP_DIR, NO_CODE, PIPELINE_CONFIG, load_prompt, response_to_code
from llmdpeval.prompts import PROMPT_TEMPLATES, task_prompt_of

# A context that mentions what the outlines mention: replace-one, delta = 0, another epsilon.
CONTEXT = ("[Context 1] Count\nUnder replace-one (the number of rows is public) the sensitivity is 2.\n"
           "    - epsilon = 0.5\nUse delta = 0 throughout.\n")


# ---- repair reads the budget, delta and relation from the task prompt ----------------------------------------
def test_task_prompt_of_guided_and_plain():
    for task in ("stat_count", "app_gopher_sqf"):
        tp = load_prompt(task, "OpenDP")
        for template in PROMPT_TEMPLATES.values():
            assert task_prompt_of(template.format(prompt=tp, context=CONTEXT)) == tp
        assert task_prompt_of(tp) == tp  # no template: the prompt is the task prompt


def test_repair_uses_task_prompt_relation(monkeypatch):
    seen = []
    real = privacy_check.check

    def spy(code, stdout=None, **kw):
        seen.append(kw)
        return real(code, stdout, **kw)
    monkeypatch.setattr(privacy_check, "check", spy)
    toy = load_prompt("stat_count", "OpenDP")
    app = load_prompt("app_gopher_sqf", "OpenDP")
    for tp, want_replace_one, want_delta_zero in ((toy, False, False), (app, True, True)):
        final = PROMPT_TEMPLATES["guided"].format(prompt=tp, context=CONTEXT)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "stat-count_code_1.py"
            f.write_text("print('Epsilon spent: 1.0')\n")
            seen.clear()
            repair.repair_loop(None, f, final, rounds=0)  # extracted from the final prompt
            assert seen[-1] == {"budget": 1.0, "delta_zero": want_delta_zero, "replace_one": want_replace_one}
            seen.clear()
            repair.repair_loop(None, f, final, rounds=0, task_prompt=tp)  # passed explicitly
            assert seen[-1]["replace_one"] is want_replace_one
    # the guided template's "(the number of rows is public)" must not count as replace-one
    assert not repair.REPLACE_ONE.search(PROMPT_TEMPLATES["guided"].format(prompt=toy, context=CONTEXT))


def test_privacy_problem_budget_from_task_prompt():
    toy = load_prompt("stat_count", "OpenDP")
    final = PROMPT_TEMPLATES["guided"].format(prompt=toy, context="    - epsilon = 0.5\n")
    assert repair.privacy_problem(toy, "Epsilon spent: 1.0\n", final) is None
    # the guided template asks for epsilon to be printed; the toy task prompt alone does not
    assert "no value labelled as epsilon" in repair.privacy_problem(toy, "count 7510\n", final)
    assert repair.privacy_problem(toy, "count 7510\n") is None


def test_select_jobs_toy_task_is_add_remove():
    # add/remove calibration (d_in=1) is a flaw under replace-one only; the toy prompts are add/remove
    code = ("import opendp.prelude as dp\n"
            "scale = dp.binary_search_param(lambda s: hist >> dp.m.then_laplace(s), d_in=1, d_out=0.5)\n")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        cell = src / "gemma4-latest_ragk3_None_th_code_thinkon"
        cell.mkdir()
        (cell / "selection-topk_code_1.py").write_text(code)
        (src / "results.jsonl").write_text(json.dumps(
            {"code_file": str(cell / "selection-topk_code_1.py"), "runnable": "True", "stdout": "x\n"}) + "\n")
        task_prompt = load_prompt("selection_topk", "None")
        rules = lambda flags: {f["rule"] for f in flags}
        assert "add_remove_calibration" in rules(privacy_check.check(code, "x\n", replace_one=True))
        assert "add_remove_calibration" not in rules(repair.static_problems(task_prompt, code, "x\n"))
        assert bool(repair.select_jobs(src)[1]) == bool(repair.static_problems(task_prompt, code, "x\n"))


def test_repair_failure_restores_last_executed_round():
    class Boom:
        def complete(self, prompt):
            raise TimeoutError("api timeout")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stat-count_code_1.py"
        f.write_text("print(1/0)\n")
        log = repair.repair_loop(Boom(), f, "PROMPT", rounds=2)
        assert [x["status"] for x in log] == ["error", "repair_failed"], log
        assert log[-1]["restored"] == "stat-count_code_1.round0.py" and log[-1]["runnable"] is False
        assert "TimeoutError" in log[-1]["error"] and f.read_text() == "print(1/0)\n"
        assert json.loads(f.with_suffix(".repair.json").read_text())[-1]["status"] == "repair_failed"


# ---- no-code answers grade N ------------------------------------------------------------------------------------
def test_table_grades_no_code_as_n():
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d)
        cell = "gemma4-latest_ragk3_OpenDP_th_code"
        (grid / cell).mkdir()
        (grid / cell / "stat-count_code_1.py").write_text(NO_CODE + "\n")
        (grid / "hand_grades.csv").write_text("cell,task,run,grade,reason\n")
        (grid / "results.jsonl").write_text(json.dumps(
            {"code_file": f"{cell}/stat-count_code_1.py", "runnable": "False", "returncode": None, "stdout": "",
             "stderr": "no_code: the answer contained no code", "status": "no_code"}) + "\n")
        grade.table(grid, 1, "hand_grades.csv", None, ["stat-count"])
        rows = list(csv.DictReader((grid / "grades.csv").open(encoding="utf-8")))
        assert [r["grade"] for r in rows] == ["N"], rows


def test_median_keeps_p(monkeypatch):
    monkeypatch.setattr(grade, "check_exec_allowed", lambda: None)
    monkeypatch.setattr(grade, "run_sandboxed", lambda p: {"returncode": 0, "stdout": "DP count: 7600\n"})
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d)
        cell = "gemma4-latest_ragk3_OpenDP_th_code"
        (grid / cell).mkdir()
        (grid / cell / "stat-count_code_1.py").write_text("print('DP count: 7600')\n")
        (grid / "hand_grades.csv").write_text(f"cell,task,run,grade,reason\n{cell},stat-count,1,P,eps printed 0\n")
        grade.median(grid, 4, 1)
        row = next(csv.DictReader((grid / "hand_grades_median.csv").open(encoding="utf-8")))
        assert row["grade"] == "P" and "not useful on the median" in row["reason"], row


# ---- sandbox: per-run process group, private scratch, end-of-pass kill -------------------------------------------
def test_sandbox_command_construction():
    cmd, env = sandbox.command(Path("s.py"), Path("/tmp/x"), 4, user=None)
    assert cmd[0] == sys.executable and cmd[-1] == "." and "sudo" not in cmd  # local: unchanged
    cmd, env = sandbox.command(Path("s.py"), Path("/tmp/x"), 4, user="dpsandbox")
    assert cmd[:4] == ["sudo", "-n", "-u", "dpsandbox"] and f"--as={4 * 2**30}" in cmd and env["HOME"] == str(Path("/tmp/x"))
    assert sandbox.kill_group_command("dpsandbox", 4242) == ["sudo", "-n", "-u", "dpsandbox", "kill", "-KILL", "--",
                                                              "-4242"]
    assert sandbox.kill_user_command("dpsandbox") == ["sudo", "-n", "pkill", "-KILL", "-u", "dpsandbox"]
    assert sandbox.scratch_owner_commands("dpsandbox", Path("/tmp/s")) == [
        ["sudo", "-n", "chown", "-R", "dpsandbox:", str(Path("/tmp/s"))], ["sudo", "-n", "chmod", "700", str(Path("/tmp/s"))]]


def test_run_in_group_kills_group_after_exit_and_timeout(monkeypatch):
    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(sandbox, "SANDBOX_USER", "dpsandbox")
    monkeypatch.setattr(sandbox.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) if cmd[0] == "sudo" else real_run(cmd, **kw))
    with tempfile.TemporaryDirectory() as d:
        rc, out, err = sandbox._run_in_group([sys.executable, "-c", "print('hi')"], Path(d), None, 60)
        assert (rc, out.strip()) == (0, "hi") and calls[-1][:6] == ["sudo", "-n", "-u", "dpsandbox", "kill", "-KILL"]
        rc, out, err = sandbox._run_in_group([sys.executable, "-c", "import time; time.sleep(30)"], Path(d), None, 1)
        assert rc is None and calls[-1][4:6] == ["kill", "-KILL"] and len(calls) == 2


def test_kill_sandbox_user_only_with_sandbox_user(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(sandbox, "SANDBOX_USER", None)
    sandbox.kill_sandbox_user()
    assert calls == []
    monkeypatch.setattr(sandbox, "SANDBOX_USER", "dpsandbox")
    sandbox.kill_sandbox_user()
    assert calls == [["sudo", "-n", "pkill", "-KILL", "-u", "dpsandbox"]]


# ---- a generation killed by the hard cap is marked and not regenerated ---------------------------------------
def test_killed_generation_marked_and_not_regenerated(monkeypatch):
    started = []

    class KilledPopen:  # saved code (e.g. mid-repair), then held past the hard cap
        def __init__(self, command, **kw):
            started.append(command)
            out_dir = Path(command[command.index("--output-dir") + 1])
            (out_dir / "stat-count_code_1.py").write_text("print(1)\n")
            self.pid, self.calls = 1, 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("gen", timeout)
            return -9
    monkeypatch.setattr(run_grid.subprocess, "Popen", KilledPopen)
    monkeypatch.setattr(run_grid.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(run_grid.os, "killpg", lambda *a: None, raising=False)
    o = argparse.Namespace(template="guided", theory=True, code=True, holdout=False, repair=2, think="on",
                           reask=False, hard_cap=1)
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d)
        cell = run_grid.cell_dir(grid, "gemma4:latest", "OpenDP", 3, o)
        cell.mkdir(parents=True)
        (cell / "stat-count_code_1.round1.py").write_text("stale\n")  # left by an earlier unfinished attempt
        run_grid.generate(grid, "gemma4:latest", "OpenDP", 3, "stat_count", 1, o)
        assert (cell / "stat-count_code_1.timeout").exists() and (cell / "stat-count_code_1.py").exists()
        assert not (cell / "stat-count_code_1.round1.py").exists()
        run_grid.generate(grid, "gemma4:latest", "OpenDP", 3, "stat_count", 1, o)  # resume: nothing to do
        assert len(started) == 1


def test_frozen_defaults_generate_without_exec():
    assert PIPELINE_CONFIG["repair"] == 0  # run_grid only calls check_exec_allowed for --evaluate or --repair
    assert PIPELINE_CONFIG["conditions"] == ["None", "Diffprivlib", "OpenDP", "None-rag-k3", "Diffprivlib-rag-k3",
                                             "OpenDP-rag-k3"]
    assert all(c in run_grid.CONDITIONS for c in PIPELINE_CONFIG["conditions"])


# ---- app grading never aborts on an unexpected output shape --------------------------------------------------
def ceiling(app):
    return json.loads((APP_DIR / app / "runs" / "ceiling.json").read_text(encoding="utf-8"))


def test_app_grade_odd_shapes():
    f = ceiling("frymer_cces")
    obj = {"dp_union_coef": {"model1": -0.062}, "dp_union_se": 0.005,  # SE not a dict
           "dp_union_p": {"model1": 1e-29, "model2": 0.9, "model3": 0.99},
           "dp_group_means": {"diff": -0.057}, "dp_diff_p": 1e-20}
    assert app_grade.frymer(obj, f)[0] == "keeps"
    z = ceiling("zhang_faers")
    otop = privacy_check.final_json((APP_DIR / "zhang_faers" / "runs" / "original_run.txt").read_text())["top100"]
    odd = {"dp_significant": [], "dp_top100": otop[:65],
           "true_original_significant": list(range(z["true"]["n_significant"])), "true_original_top100": 5}
    assert app_grade.zhang(odd, z, APP_DIR / "zhang_faers")[0] == "keeps"
    im = ceiling("imai_fl")
    assert app_grade.imai({"dp_table1": [1], "dp_table2": "x"}, im)[0] == "unparseable"
    assert app_grade.utility(APP_DIR / "gopher_sqf", [1, 2])[0] == "unparseable"


def test_app_grade_grid_survives_a_bad_row(monkeypatch):
    def boom(app, obj):
        raise TypeError("unexpected shape")
    monkeypatch.setattr(app_grade, "utility", boom)
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d)
        cell = grid / "qwen3.6-27b_ragk3_Diffprivlib_th_code_thinkon"
        cell.mkdir()
        (cell / "app-gopher-sqf_code_1.py").write_text("print(1)\n")
        (grid / "results.jsonl").write_text(json.dumps(
            {"code_file": str(cell / "app-gopher-sqf_code_1.py"), "runnable": "True", "stdout": "{}\n"}) + "\n")
        rows = app_grade.grade_grid(grid, None, grid / "app_grades.csv")
        assert rows[0]["utility"] == "unparseable" and "TypeError" in rows[0]["utility_detail"]


# ---- fences, unsafe check, fields query, hashing ------------------------------------------------------------------------------------
def test_response_to_code_prefers_python_fence():
    assert response_to_code("```bash\npip install opendp\n```\nthen\n```python\nprint(1)\n```") == "print(1)\n"
    assert response_to_code("```text\nnotes\n```\n```\nx = 1\n```") == "x = 1\n"
    assert response_to_code("```sh\nls\n```") == "ls\n"


def test_is_unsafe_on_ast():
    unsafe = ["import subprocess", "from socket import socket", "import os\nos.system('x')", "import os as o\no.popen('x')",
              "from os import remove", "import os\nos.rename('a', 'b')", "from pathlib import Path\nPath('a').unlink()",
              "import shutil", "import requests", "import urllib.request", "import http.client",
              "__import__('subprocess')", "import importlib\nimportlib.import_module('socket')"]
    safe = ["# do not use subprocess or os.remove here\nprint(1)", "x = 'socket requests urllib'\nprint(x)",
            "import os\nprint(os.path.join('a', 'b'))", "xs = [1, 2]\nxs.remove(1)"]
    assert [c for c in unsafe if not sandbox.is_unsafe(c)] == []
    assert [c for c in safe if sandbox.is_unsafe(c)] == []
    assert sandbox.is_unsafe("import subprocess\ndef (")  # does not parse: regex fallback


def test_fields_query_keeps_bullets():
    q = retrieval.make_query(load_prompt("ml_regression", "OpenDP"), "fields")
    assert "- epsilon = 1" in q and "- bounds_x = (1500.0, 5500.0)" in q and "DP framework: OpenDP" in q
    q = retrieval.make_query(load_prompt("stat_count", "OpenDP"), "fields")
    assert "epsilon = 1" in q and "- " not in q


def test_hash_helper_keeps_values():
    from llmdpeval.helper import lf_hash
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.txt"
        p.write_bytes(b"x\r\ny\n")
        import hashlib
        assert run_grid.file_hash(p) == hashlib.sha256(b"x\ny\n").hexdigest()[:16]
        assert lf_hash([p], Path(d)) == hashlib.sha256(b"a.txtx\ny\n").hexdigest()[:16]
