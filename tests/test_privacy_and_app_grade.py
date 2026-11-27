"""Checks of the static privacy check, the repair log, truncated / no-code answers, per-app eval settings, the app
utility checker and the app prompt facts. Plain asserts, no LLM, no network; every script run here is written by
the test itself.

  PYTHONPATH=src python -m pytest tests/test_privacy_and_app_grade.py
"""
import json
import tempfile
from pathlib import Path

from llmdpeval import app_grade, privacy_check, repair, sandbox
from llmdpeval.helper import APP_DIR, NO_CODE, TRUNCATED, code_status, load_prompt, response_to_code


def rules(code, stdout=None, **kw):
    return {f["rule"] for f in privacy_check.check(code, stdout, **kw)}


# ---- privacy_check: one positive per rule (patterns from hand-graded failures) and valid negatives -------------
NONPRIVATE_SKLEARN = """
from sklearn.linear_model import LogisticRegression
import numpy as np
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
correct = clf.predict(X_test) == y_test
acc = correct.mean()
dp_acc = acc + np.random.laplace(0, 1 / (len(y_test) * 0.5))
"""
NONPRIVATE_TORCH_ONLY = """
import torch
def train(X, y):
    model = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad(); loss = ((model(X) - y) ** 2).mean(); loss.backward(); opt.step()
    return model
model = train(X_train, y_train)
"""
PRIVATE_BOUNDS = """
from diffprivlib import models
lo = X_train.min(axis=0)
hi = X_train.max(axis=0)
clf = models.LogisticRegression(epsilon=0.5, data_norm=1.0, bounds=(lo, hi))
clf.fit(X_train, y_train)
"""
SCALED_INTO_DP_MODEL = """
from diffprivlib import models
mu = train_df.mean()
sd = train_df.std()
X = ((train_df - mu) / sd).to_numpy()
clf = models.LogisticRegression(epsilon=0.5, data_norm=1.0)
clf.fit(X, y)
"""
MISSING_NORM = """
from diffprivlib.models import LogisticRegression
clf = LogisticRegression(epsilon=0.5, bounds_X=(0, 1))
clf.fit(X, y)
"""
GROUP_SIZE = """
from diffprivlib import tools
p0 = p[race == 0]
mean0 = tools.mean(p0, bounds=(0, 1), epsilon=0.25)
correct_r = correct[race == 1]
noisy = correct_r.sum() + rng.laplace(0, 2 / 0.25)
acc_r = noisy / len(correct_r)
"""
ADD_REMOVE = """
import opendp.prelude as dp
scale = dp.binary_search_param(lambda s: hist >> dp.m.then_laplace(s), d_in=1, d_out=0.5)
ctx = dp.Context.compositor(data=df, privacy_unit=dp.unit_of(contributions=1), privacy_loss=dp.loss_of(epsilon=1.0))
total = max(epsilon_counts)
"""
UNUSED_NOISE = """
def dp_fit(X, y, eps):
    w = fit_logreg(X, y)
    z = rng.normal(size=len(w))
    b = rng.gamma(shape=len(w), scale=1 / eps) * z / np.linalg.norm(z)
    return w, eps
"""
DELTA = """
delta_used = 1e-9
acc = BudgetAccountant(epsilon=1.0, delta=1e-6)
"""
VALID_DIFFPRIVLIB = """
import numpy as np
from diffprivlib import models, tools
X = np.clip(train[cols].to_numpy() / MAXCODE, 0, 1) / np.sqrt(18)
clf = models.LogisticRegression(epsilon=0.6, data_norm=1.0, fit_intercept=False, C=0.1)
clf.fit(X, y)
correct = clf.predict(Xt) == yt
dp_correct = tools.count_nonzero(correct, epsilon=0.1)
dp_acc = dp_correct / len(yt)
p = clf.predict_proba(Xt)[:, 1]
s0 = tools.sum(p[race == 0], bounds=(0, 1), epsilon=0.1)
n0 = tools.count_nonzero(race == 0, epsilon=0.1)
dp_mean0 = s0 / max(n0, 1)
"""
VALID_OUTPUT_PERTURBATION = """
import numpy as np
def _fit_logreg(X, y, lam, iters=100):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        grad = X.T @ (1 / (1 + np.exp(-X @ w)) - y) / len(y) + lam * w
        w -= 0.5 * grad
    return w
w_true = _fit_logreg(X / np.sqrt(18), y, 0.01)
z = rng.normal(size=len(w_true))
b = rng.gamma(shape=len(w_true), scale=2 / (len(y) * 0.01 * 0.6)) * z / np.linalg.norm(z)
w_dp = w_true + b
p_dp = 1 / (1 + np.exp(-(Xt @ w_dp)))
correct = (p_dp > 0.5) == yt
dp_acc = (correct.sum() + rng.laplace(0, 2 / 0.1)) / len(yt)
"""


def test_static_rules_fire():
    assert "nonprivate_model" in rules(NONPRIVATE_SKLEARN)
    assert "nonprivate_model" in rules(NONPRIVATE_TORCH_ONLY)
    assert "private_bounds" in rules(PRIVATE_BOUNDS)
    assert "private_bounds" in rules(SCALED_INTO_DP_MODEL)
    assert "missing_bounds" in rules(MISSING_NORM)
    assert rules(GROUP_SIZE) >= {"group_size_public"}
    assert len([f for f in privacy_check.check(GROUP_SIZE) if f["rule"] == "group_size_public"]) == 2
    assert {f["detail"] for f in privacy_check.check(ADD_REMOVE)} >= {"d_in=1", "contributions=1"}
    assert "add_remove_calibration" not in rules(ADD_REMOVE, replace_one=False)
    assert "unused_noise" in rules(UNUSED_NOISE)
    assert {f["detail"] for f in privacy_check.check(DELTA)} == {"delta_used = 1e-09", "delta=1e-06"}
    assert not rules(DELTA, delta_zero=False)


def test_static_rules_quiet_on_valid_designs():
    assert rules(VALID_DIFFPRIVLIB) == set(), privacy_check.check(VALID_DIFFPRIVLIB)
    assert rules(VALID_OUTPUT_PERTURBATION) == set(), privacy_check.check(VALID_OUTPUT_PERTURBATION)
    # d_in=1 next to a vector metric is a sensitivity, not a row count
    assert "add_remove_calibration" not in rules(
        "space = (dp.vector_domain(dp.atom_domain(T=float)), dp.l1_distance(T=float))\n"
        "m = dp.binary_search_param(make, d_in=1, d_out=0.5)\n")
    assert rules("this is not python (") == set()


def test_output_flags_and_feedback():
    out = 'x\n{"true_accuracy": 0.84, "dp_accuracy": 0.84, "true_spd": -0.07, "dp_spd": -0.08, "epsilon_total": 1.2}\n'
    got = {f["rule"]: f for f in privacy_check.check("print(1)\n", out, budget=1.0)}
    assert set(got) == {"dp_equals_true", "over_budget"} and "`dp_accuracy`" in got["dp_equals_true"]["detail"]
    text = privacy_check.feedback(privacy_check.check(NONPRIVATE_SKLEARN))
    assert text.startswith("- A model is trained on private rows without DP") and "(line" in text


# ---- truncated and no-code answers ------------------------------------------------------------------------------
def test_truncated_and_no_code():
    cut = response_to_code("```python\nimport numpy as np\nx = np.ones(")
    assert cut.splitlines()[0] == TRUNCATED and code_status(cut) == "truncated"
    assert code_status(response_to_code("Here is my plan: first add noise, then it's done.")) == "no_code"
    assert code_status(response_to_code("```python\nprint(1)\n```")) == "ok"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stat-count_code_1.py"
        f.write_text(NO_CODE + "\n# prose\n")
        r = sandbox.run_sandboxed(f)
        assert r["status"] == "no_code" and r["runnable"] == "False"
        f.write_text(TRUNCATED + "\nprint(1/0)\n")
        assert sandbox.run_sandboxed(f)["status"] == "truncated"
        f.write_text(TRUNCATED + "\nprint(1)\n")  # complete code with a missing closing fence still runs
        r = sandbox.run_sandboxed(f)
        assert r["runnable"] == "True" and "status" not in r


# ---- per-app eval settings --------------------------------------------------------------------------------------
def test_eval_settings_per_app():
    import os
    assert json.loads((APP_DIR / "imai_fl" / "eval.json").read_text()) == {"timeout_s": 600, "memory_gb": 32}
    assert json.loads((APP_DIR / "zhang_faers" / "eval.json").read_text()) == {"timeout_s": 600}
    saved = {k: os.environ.pop(k, None) for k in ("LLMDPEVAL_RUN_TIMEOUT", "LLMDPEVAL_SANDBOX_MEMORY_GB")}
    try:
        assert sandbox.eval_settings(Path("g/c/app-imai-fl_code_1.py")) == {"timeout_s": 600, "memory_gb": 32}
        assert sandbox.eval_settings(Path("g/c/app-zhang-faers_code_1.py")) == {"timeout_s": 600, "memory_gb": 16}
        assert sandbox.eval_settings(Path("g/c/app-gopher-sqf_code_1.py")) == {"timeout_s": 120, "memory_gb": 16}
        os.environ["LLMDPEVAL_SANDBOX_MEMORY_GB"] = "8"  # an explicit env var wins
        assert sandbox.eval_settings(Path("g/c/app-imai-fl_code_1.py"))["memory_gb"] == 8
        os.environ["LLMDPEVAL_RUN_TIMEOUT"] = "5"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "stat-count_code_1.py"
            f.write_text("import time\ntime.sleep(30)\n")
            assert sandbox.run_sandboxed(f)["stderr"] == "Timed out after 5 seconds"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# ---- repair: static feedback order, stdout in the log, job selection -------------------------------------------
APP_PROMPT = load_prompt("app_gopher_sqf", "Diffprivlib")


def test_repair_static_before_epsilon_feedback():
    seen = []

    class LLM:
        def complete(self, prompt):
            seen.append(prompt)
            return "```python\nprint('Epsilon total: 1.0')\nprint('{\"dp_spd\": -0.08, \"epsilon_total\": 1.0}')\n```"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "app-gopher-sqf_code_1.py"
        # runs, non-private model AND an overspend: static feedback comes first (one kind per round)
        f.write_text(NONPRIVATE_SKLEARN.replace("clf.fit(X_train, y_train)", "pass").replace(
            "correct = clf.predict(X_test) == y_test\nacc = correct.mean()\n", "").replace(
            "dp_acc = acc + np.random.laplace(0, 1 / (len(y_test) * 0.5))", "print('Epsilon total: 2.0')"))
        log = repair.repair_loop(LLM(), f, APP_PROMPT, rounds=2)
        assert [x["status"] for x in log] == ["static_privacy", "ok"], log
        assert "static privacy check" in seen[0] and "privacy-accounting" not in seen[0]
        assert log[-1]["runnable"] is True and '"epsilon_total": 1.0' in log[-1]["stdout_tail"]
        assert json.loads(f.with_suffix(".repair.json").read_text())[0]["static"][0]["rule"] == "nonprivate_model"


def test_select_jobs_includes_flagged_runnable():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d)
        cell = src / "qwen3.6-27b_ragk3_Diffprivlib_th_code_thinkon"
        cell.mkdir()
        (cell / "app-gopher-sqf_code_1.py").write_text(NONPRIVATE_SKLEARN)
        (cell / "app-gopher-sqf_code_2.py").write_text(VALID_DIFFPRIVLIB)
        (cell / "app-gopher-sqf_code_3.py").write_text("print(1/0)\n")
        (cell / "app-gopher-sqf_code_4.py").write_text(NO_CODE + "\n")
        res = [{"code_file": str(cell / f"app-gopher-sqf_code_{i}.py"), "runnable": rn, "stdout": "", **extra}
               for i, rn, extra in ((1, "True", {}), (2, "True", {}), (3, "False", {}),
                                    (4, "False", {"status": "no_code"}))]
        (src / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
        crashed, flagged = repair.select_jobs(src)
        assert crashed == {(cell.name, "app-gopher-sqf_code_3.py")}
        assert flagged == {(cell.name, "app-gopher-sqf_code_1.py")}


# ---- app_grade ---------------------------------------------------------------------------------------------------
def ceiling(app):
    return json.loads((APP_DIR / app / "runs" / "ceiling.json").read_text(encoding="utf-8"))


def test_utility_gopher_kulynych():
    c = ceiling("gopher_sqf")
    assert app_grade.gopher({"dp_accuracy": 0.6829, "dp_spd": -0.0798}, c)[0] == "keeps"
    assert app_grade.gopher({"dp_accuracy": 0.6716, "dp_spd": -0.0110}, c)[0] == "loses"   # SPD near 0: the effect is lost
    assert app_grade.gopher({"dp_accuracy": 0.6717, "dp_spd": 7.2331}, c)[0] == "loses"    # SPD far outside the range
    assert app_grade.gopher({"dp_accuracy": "x"}, c)[0] == "unparseable"
    k = ceiling("kulynych_adult")
    med = {"dp_accuracy": {g: v["median"] for g, v in k["summary"]["dp_accuracy"].items()},
           "dp_race_counts": {g: v["median"] for g, v in k["summary"]["dp_race_counts"].items()}}
    assert app_grade.kulynych(med, k)[0] == "keeps"
    half = dict(med, dp_race_counts={g: v / 2 for g, v in med["dp_race_counts"].items()})  # counted the test half
    assert app_grade.kulynych(half, k)[0] == "loses"


def test_utility_frymer_zhang_imai():
    f = ceiling("frymer_cces")
    good = {"dp_union_coef": {"model1": -0.062, "model2": 0.02, "model3": -0.3},
            "dp_union_se": {"model1": 0.0055, "model2": 18.0, "model3": 20.0},
            "dp_union_p": {"model1": 1e-29, "model2": 0.9, "model3": 0.99},
            "dp_group_means": {"union": 0.626, "nonunion": 0.683, "diff": -0.057}, "dp_diff_p": 1e-20}
    assert app_grade.frymer(good, f)[0::2] == ("keeps", False)
    fe = json.loads(json.dumps(good))
    fe["dp_union_p"]["model2"] = 0.001  # significant under state fixed effects -> W
    assert app_grade.frymer(fe, f)[0::2] == ("loses", True)
    z = ceiling("zhang_faers")
    otop = privacy_check.final_json((APP_DIR / "zhang_faers" / "runs" / "original_run.txt").read_text())["top100"]
    assert app_grade.zhang({"dp_significant": [], "dp_top100": otop[:65]}, z, APP_DIR / "zhang_faers")[0] == "keeps"
    assert app_grade.zhang({"dp_significant": [], "dp_top100": otop[:40]}, z, APP_DIR / "zhang_faers")[0] == "loses"
    im = ceiling("imai_fl")
    t1, t2 = {}, {}
    for key, v in im["table1"].items():
        row, pred = key.split("|")
        t1.setdefault(pred, {})[row] = v["original"]
    for key, v in im["table2"].items():
        row, col = key.split("|")
        t2.setdefault(row, {})[col] = v["median"]
    assert app_grade.imai({"dp_table1": t1, "dp_table2": t2}, im)[0] == "keeps"
    t2["district_black"]["bayes_rmse"] = 1.0
    assert app_grade.imai({"dp_table1": t1, "dp_table2": t2}, im)[0] == "loses"


def test_app_grade_grid_and_agreement():
    with tempfile.TemporaryDirectory() as d:
        grid = Path(d)
        cell = grid / "qwen3.6-27b_ragk3_Diffprivlib_th_code_thinkon"
        cell.mkdir()
        (cell / "app-gopher-sqf_code_1.py").write_text(VALID_DIFFPRIVLIB)
        (cell / "app-gopher-sqf_code_2.py").write_text(NONPRIVATE_SKLEARN)
        (cell / "app-gopher-sqf_code_3.py").write_text(VALID_DIFFPRIVLIB)
        (cell / "app-gopher-sqf_code_4.py").write_text(NO_CODE + "\n")
        js = lambda spd: json.dumps({"true_accuracy": 0.68, "dp_accuracy": 0.683, "true_spd": -0.075, "dp_spd": spd,
                                     "epsilon_total": 1.0})
        res = [{"code_file": str(cell / "app-gopher-sqf_code_1.py"), "runnable": "True", "stdout": js(-0.08) + "\n"},
               {"code_file": str(cell / "app-gopher-sqf_code_2.py"), "runnable": "True", "stdout": js(-0.08) + "\n"},
               {"code_file": str(cell / "app-gopher-sqf_code_3.py"), "runnable": "False", "stdout": ""},
               {"code_file": str(cell / "app-gopher-sqf_code_4.py"), "runnable": "False", "stdout": "",
                "status": "no_code"}]
        (grid / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
        # script 3 was repaired: the stdout logged in its .repair.json wins over the old crash
        (cell / "app-gopher-sqf_code_3.repair.json").write_text(json.dumps(
            [{"round": 0, "status": "error"}, {"round": 1, "status": "ok", "runnable": True,
                                               "stdout_tail": js(-0.01) + "\n"}]))
        rows = app_grade.grade_grid(grid, None, grid / "app_grades.csv")
        got = {r["run"]: (r["suggested"], r["utility"], r["source"]) for r in rows}
        assert got == {1: ("G", "keeps", "results"), 2: ("W", "keeps", "results"), 3: ("U", "loses", "repair"),
                       4: ("N", "-", "results")}, got
        assert all(r["confirm"] == "by hand" for r in rows)
        (grid / "hand.csv").write_text("cell,task,run,grade\n" + "".join(
            f"{cell.name},app-gopher-sqf,{i},{g}\n" for i, g in ((1, "G"), (2, "W"), (3, "W"), (4, "N"))))
        agree, n, _ = app_grade.agreement(rows, grid / "hand.csv")
        assert (agree, n) == (3, 4)


# ---- app prompt facts --------------------------------------------------------------------------------------------
def test_app_prompt_facts():
    g = load_prompt("app_gopher_sqf", "OpenDP")
    assert "shape (rows, 17)" in g and "size of every subgroup" in g and "weights are released" in g
    assert "mean predicted probability of the rows with `race` = 0" in g
    k = load_prompt("app_kulynych_adult", "None")
    assert "over all 45,222 rows, not the test half" in k and "size of every subgroup" in k
    assert "one-hot encoded over the public domains" in k
    fr = load_prompt("app_frymer_cces", "Diffprivlib")
    assert "model 2 has 57" in fr and "must not use statistics computed from the data" in fr
    # app prompts: original and own non-private values under separate keys, next to the dp_* keys
    for p, keys in ((g, "`true_original_accuracy`, `true_own_accuracy`, `dp_accuracy`, `true_original_spd`, "
                        "`true_own_spd`, `dp_spd` and `epsilon_total`"),
                    (k, "`true_original_accuracy`, `true_own_accuracy`, `dp_accuracy`, `true_original_race_counts`, "
                        "`true_own_race_counts`, `dp_race_counts` and `epsilon_total`"),
                    (fr, "`true_original_union_coef`, `true_own_union_coef`, `dp_union_coef`, `dp_union_se`")):
        assert keys in p and "the true value from your own analysis without noise" in p
    for app, key in (("zhang_faers", "true_significant"), ("imai_fl", "true_table1")):
        p = load_prompt(f"app_{app}", "OpenDP")
        assert f"`true_original_{key[5:]}`, `true_own_{key[5:]}`" in p and "the true value from your own analysis" in p
    # every app prompt states replace-one and keeps the given helpers; every toy prompt states add/remove
    for app in ("gopher_sqf", "kulynych_adult", "frymer_cces", "zhang_faers", "imai_fl"):
        p = load_prompt(f"app_{app}", "OpenDP")
        assert "(replace-one)" in p and "helper functions as given" in p and "add/remove" not in p
    for task in ("stat_count", "selection_topk", "ml_regression", "ml_pca"):
        assert "(add/remove)" in load_prompt(task, "OpenDP")


def test_true_keys_in_grading_and_output_flags():
    otop = privacy_check.final_json((APP_DIR / "zhang_faers" / "runs" / "original_run.txt").read_text())["top100"]
    assert app_grade.true_original({"true_original_top100": [1], "true_top100": [2]}, "top100") == [1]
    assert app_grade.true_original({"true_top100": [2]}, "top100") == [2]  # a plain true_* key
    z = ceiling("zhang_faers")
    sig = [str(i) for i in range(159)]
    r = app_grade.zhang({"dp_significant": sig[:40], "dp_top100": otop[:65], "true_original_significant": sig,
                         "true_original_top100": otop}, z, APP_DIR / "zhang_faers")
    assert "significant kept 40/159" in r[1], r
    v, detail, _ = app_grade.utility(APP_DIR / "gopher_sqf", {"dp_accuracy": 0.6829, "dp_spd": -0.0798,
                                                               "true_own_spd": -0.015, "true_own_accuracy": 0.67})
    assert v == "keeps" and "own non-private" in detail and "-0.015" in detail
    out = '{"true_original_spd": -0.075, "true_own_spd": -0.02, "dp_spd": -0.02, "epsilon_total": 1.0}\n'
    assert {f["rule"] for f in privacy_check.check("print(1)\n", out)} == {"dp_equals_true"}


def test_fifth_sphere_model():
    from llmdpeval.helper import LLM_MODELS, SPHERE_MODELS
    from llmdpeval import run_grid
    assert LLM_MODELS["qwen3.6:35b"] == "sphere" and len(SPHERE_MODELS) == 5
    assert run_grid.parse_cell("qwen3.6-35b_ragk3_OpenDP_th_code_thinkon")["model"] == "qwen3.6:35b"
    assert run_grid.parse_cell("qwen3.6-27b_ragk3_OpenDP_th_code_thinkon")["model"] == "qwen3.6:27b"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print("ok", name)
    print(f"{len(tests)} checks ok")
