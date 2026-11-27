"""Repair loop: run generated code in a sandbox; send crashes and privacy-accounting problems back to the same LLM.

Feedback, one kind per round, all within the same --rounds budget:
- crash (exit code != 0): the tail of stderr;
- fake fix: a "fix" that exits 0 with no output, or adds a catch-all except;
- static privacy: the code exits 0, but privacy_check finds a design flaw in the code or its final JSON line (a model
  trained without DP, bounds/scaling from the private data, a private group size used as public, add/remove-one
  calibration under replace-one, noise never added, delta > 0, a DP value equal to its true value, ...);
- privacy: the code exits 0, but its own output shows an accounting problem (privacy_problem; no LLM judge, only
  the printed epsilon values and the budget stated in the prompt).
Code that sandbox.is_unsafe flags is never executed: it is left for hand review. Every round is kept as
<task>_code_<n>.round<r>.py plus a .repair.json log (status per round: error, fake_fix, static_privacy,
privacy_feedback, ok, unsafe_not_run, no_code; with runnable, returncode, stderr_tail and stdout_tail, so the last
round can be graded without a re-run). An answer with no code (helper.NO_CODE) is not run or repaired: graded N.

  python -m llmdpeval.repair SRC_GRID OUT_GRID [--rounds 2]   # repair a finished, evaluated grid
  python -m llmdpeval.repair --selftest                       # no LLM needed

For a finished grid: copies every <cell>/<task>_code_<n>.py from SRC to OUT, picks the scripts that crashed and the
runnable ones that privacy_check flags (code + stdout from results.jsonl), rebuilds each one's exact
prompt (same retrieval, template, theory, skeleton and holdout flags as the cell; paper apps reuse the logged final
prompt, since it holds the model's own code analysis), then runs repair_loop. Use the corpus the grid used:
repair_config.json records the corpus hash.
"""
import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from llmdpeval import api_lookup, privacy_check
from llmdpeval.helper import code_status, file_key, load_prompt, response_to_code
from llmdpeval.prompts import task_prompt_of
from llmdpeval.sandbox import check_exec_allowed, is_unsafe, kill_sandbox_user, run_sandboxed

API_LOOKUP = False  # --api-lookup: add the exact pinned API of a misnamed framework call to crash feedback
STDERR_LINES = 40  # tail of stderr sent back; tracebacks end with the actual error
STDOUT_TAIL = 20000  # characters of stdout kept per round in .repair.json (the final JSON line is at the end)
# A "fix" that wraps the failure in a catch-all handler exits 0 without a DP answer.
CATCH_ALL = re.compile(r"^\s*except\s*(Exception|BaseException)?\s*(as\s+\w+)?\s*:", re.M)
FAKE_FIX_FEEDBACK = ("Your previous fix hid the failure: it added a catch-all except (or printed nothing) instead "
                     "of fixing the cause. Remove the handler and fix the actual error.")


def fake_fix(original: str, code: str, stdout: str) -> bool:
    """Exit 0 but no output, or more catch-all handlers than the original code had."""
    return not stdout.strip() or len(CATCH_ALL.findall(code)) > len(CATCH_ALL.findall(original))


# ---- privacy feedback: checks on the script's own output ------------------------------------------------------
NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)"
# The budget: the first requirement line ("3. Use ... epsilon = 1", "    - epsilon = 1") of the task prompt.
# Requirement lines come before any retrieved context, so an outline's "epsilon = 0.5" is never read as the budget.
BUDGET = re.compile(r"^\s*(?:\d+\.|-)\s[^\n]*?\bepsilon\s*=\s*(\d+(?:\.\d+)?)", re.I | re.M)
# The prompt asks for the epsilon to be printed (paper apps: requirement 5; guided RAG template).
NEEDS_EPSILON = re.compile(r"\bprint\b[^.\n]*\bepsilon\b", re.I)
EPS_WORD = re.compile(r"epsilon|ε", re.I)
# The value of an epsilon line: the first number after the first ':' or '=' that follows the word epsilon.
EPS_VALUE = re.compile(r"(?:epsilon|ε)[^:=\n]*[:=]\s*\(?\s*" + NUM, re.I)
NOTE = re.compile(r"\((?![^()]*(?:epsilon|ε))[^()]*\)", re.I)  # "(b=2)", "(1)": parentheses without epsilon
TOTAL = re.compile(r"total|overall", re.I)
PER_RELEASE = re.compile(r"\bepsilon\b\s*(?:spent|used)?\s*(?:for|of|on)\b", re.I)  # Epsilon for count: 0.5
TOLERANCE = 1e-3  # relative; printed values are rounded
DELTA_ZERO = re.compile(r"\bdelta\s*=\s*0(?![.\d])", re.I)
# The neighbouring relation as the task prompt states it: paper-app prompts say "(replace-one)", toy prompts say
# "(add/remove)". Under replace-one a histogram changes by 2 in L1.
REPLACE_ONE = re.compile(r"\(replace-one\)", re.I)


def static_problems(task_prompt: str, code: str, stdout: str) -> list[dict]:
    """privacy_check flags for a script that ran, with the budget, delta and neighbour relation of its TASK prompt
    (before any RAG wrapping: the template and the retrieved outlines mention replace-one, delta = 0 and other
    epsilons, which are not the task's)."""
    m = BUDGET.search(task_prompt)
    return privacy_check.check(code, stdout, budget=float(m.group(1)) if m else None,
                               delta_zero=bool(DELTA_ZERO.search(task_prompt)),
                               replace_one=bool(REPLACE_ONE.search(task_prompt)))


def printed_epsilons(stdout: str) -> list[tuple[str, float]]:
    """(line, value) for each output line that names epsilon and gives a value (EPS_VALUE, after dropping
    parentheses that do not mention epsilon): "Epsilon spent (accountant): 1.0", "(epsilon=0.5, scale=2)",
    '"epsilon_total": 1.0', "epsilon spent: 1.0, delta spent: 0.0" -> 1.0."""
    out = []
    for line in stdout.splitlines():
        if not EPS_WORD.search(line):
            continue
        text = line
        while NOTE.search(text):
            text = NOTE.sub("", text)
        m = EPS_VALUE.search(text)
        if m:
            out.append((line.strip(), float(m.group(1))))
    return out


def privacy_problem(task_prompt: str, stdout: str, final_prompt: str | None = None) -> str | None:
    """A plain description of the first failed check, or None. The budget comes from the task prompt; whether
    epsilon must be printed comes from the prompt the model got (final_prompt, default the task prompt: the guided
    RAG template asks for it). Checks, in order:
    (i) the prompt asks to print epsilon but no epsilon value is printed;
    (ii) a printed epsilon is 0, negative or NaN;
    (iii) the printed epsilon exceeds the prompt's budget: a printed total (a line with "total"/"overall") if any,
    else the sum of per-release lines ("epsilon for X: ..."), else the largest printed value."""
    eps = printed_epsilons(stdout)
    if not eps:
        return ("The task asks you to print the epsilon spent, but the output prints no value labelled as epsilon."
                if NEEDS_EPSILON.search(final_prompt or task_prompt) else None)
    bad = [(l, v) for l, v in eps if math.isnan(v) or v <= 0]
    if bad:
        return f"The output prints an epsilon that is zero, negative or NaN: {bad[0][0]!r}."
    m = BUDGET.search(task_prompt)
    if not m:
        return None
    budget = float(m.group(1))
    totals = [(l, v) for l, v in eps if TOTAL.search(l)]
    per_release = [(l, v) for l, v in eps if PER_RELEASE.search(l)]
    if totals:
        line, spent = max(totals, key=lambda x: x[1])
        what = f"the printed total ({line!r})"
    elif len(per_release) > 1:
        line, spent = "; ".join(l for l, _ in per_release), sum(v for _, v in per_release)
        what = f"the sum of the per-release epsilons ({line!r})"
    else:
        line, spent = max(eps, key=lambda x: x[1])
        what = f"the printed epsilon ({line!r})"
    if spent > budget * (1 + TOLERANCE):
        return f"The privacy budget is epsilon = {budget:g}, but {what} is {spent:g}."
    return None


def repair_prompt(final_prompt: str, code: str, stderr: str, privacy: str | None = None,
                  static: str | None = None) -> str:
    """The repair request: the original prompt, the previous code and one kind of feedback (static privacy flags,
    else the printed-epsilon problem, else the tail of stderr)."""
    if static:
        return (f"{final_prompt}\n\nYour previous code:\n```python\n{code}```\n\n"
                f"It ran, but a static privacy check of the code found these problems:\n{static}\n"
                "Fix them without changing the parts that are correct. Return only executable Python code.")
    if privacy:
        return (f"{final_prompt}\n\nYour previous code:\n```python\n{code}```\n\n"
                f"It ran, but its output shows a privacy-accounting problem:\n{privacy}\n"
                "Fix the privacy accounting (the epsilon each release spends, and what you print) without changing "
                "the parts that are correct. Return only executable Python code.")
    tail = "\n".join(stderr.strip().splitlines()[-STDERR_LINES:])
    api = api_lookup.lookup(stderr, code) if API_LOOKUP else None
    return (f"{final_prompt}\n\nYour previous code:\n```python\n{code}```\n\n"
            f"It failed with this error:\n```\n{tail}\n```\n" + (f"{api}\n" if api else "") +
            "Fix the cause of the error. Do not hide it with a catch-all except, and do not change the privacy "
            "mechanisms, sensitivities or budget split unless the error is in them. Return only executable Python code.")


def repair_loop(llm, code_file: Path, final_prompt: str, rounds: int, task_prompt: str | None = None) -> list[dict]:
    """Up to `rounds` repairs. Returns the per-round log; the final code stays at code_file.
    final_prompt is what the model got (and gets again with the feedback); task_prompt is the task before RAG
    wrapping, which states the budget, delta and neighbouring relation (default: extracted from final_prompt).
    An exception mid-repair (an API timeout, ...) restores the last executed round to code_file and ends the log
    with a repair_failed entry that carries that round's run result."""
    task_prompt = task_prompt or task_prompt_of(final_prompt)
    log = []
    original = code_file.read_text(encoding="utf-8", errors="replace")
    try:
        _repair_rounds(llm, code_file, final_prompt, task_prompt, rounds, original, log)
    except Exception as e:
        last = log[-1] if log else {}
        restored = code_file.with_suffix(f".round{last.get('round', 0)}.py")
        if restored.exists():
            shutil.copy(restored, code_file)
        log.append({**{k: last[k] for k in ("runnable", "returncode", "stderr_tail", "stdout_tail") if k in last},
                    "round": len(log), "status": "repair_failed", "error": f"{type(e).__name__}: {e}"[:300],
                    "restored": restored.name if restored.exists() else None})
    code_file.with_suffix(".repair.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def _repair_rounds(llm, code_file: Path, final_prompt: str, task_prompt: str, rounds: int, original: str,
                   log: list) -> None:
    for r in range(rounds + 1):
        code = code_file.read_text(encoding="utf-8", errors="replace")
        shutil.copy(code_file, code_file.with_suffix(f".round{r}.py"))
        if code_status(code) == "no_code":
            log.append({"round": r, "status": "no_code", "runnable": False})
            break
        if is_unsafe(code):
            log.append({"round": r, "status": "unsafe_not_run", "runnable": False})
            break
        run = run_sandboxed(code_file)
        ok = str(run.get("runnable")) == "True"
        stdout = run.get("stdout") or ""
        fake = ok and r > 0 and fake_fix(original, code, stdout)
        # one kind of feedback per round: crash, fake fix, static privacy, then printed-epsilon accounting
        static = static_problems(task_prompt, code, stdout) if ok and not fake else []
        privacy = privacy_problem(task_prompt, stdout, final_prompt) if ok and not fake and not static else None
        status = ("fake_fix" if fake else "static_privacy" if static else "privacy_feedback" if privacy
                  else "ok" if ok else "error")
        log.append({"round": r, "status": status, "runnable": ok, "returncode": run.get("returncode"),
                    "stderr_tail": (run.get("stderr") or "")[-2000:], "stdout_tail": stdout[-STDOUT_TAIL:],
                    **({"privacy": privacy} if privacy else {}),
                    **({"static": [{k: f[k] for k in ("rule", "line", "detail")} for f in static]} if static else {})})
        if status == "ok" or r == rounds:
            break
        feedback = FAKE_FIX_FEEDBACK if fake else run.get("stderr") or ""
        response = llm.complete(repair_prompt(final_prompt, code, feedback, privacy,
                                              privacy_check.feedback(static) if static else None))
        code_file.write_text(response_to_code(response), encoding="utf-8")


BANNER = "Filled prompt\n" + "=" * 128 + "\n"


def logged_prompt(code_file: Path) -> str:
    """The exact final prompt from the generation log (the last banner; ends at the usage line or the file end)."""
    log = (code_file.parent / code_file.name.replace("_code_", "_gen_").replace(".py", ".log")).read_text(encoding="utf-8")
    return log.rsplit(BANNER, 1)[1].split("\nusage: ", 1)[0].split("\n\nGeneration run ", 1)[0].rstrip("\n")


def cached_llm(llms: dict, model: str, temp: float, think, build):
    """One client per (model, temperature, think), built only the first time it is needed."""
    if (model, temp, think) not in llms:
        llms[model, temp, think] = build(model, temp)
    return llms[model, temp, think]


def select_jobs(src: Path) -> tuple[set, set]:
    """(crashed, flagged) (cell, file name) pairs from SRC/results.jsonl: scripts that crashed, and runnable scripts
    that privacy_check flags (code + stdout, with the budget, delta and neighbouring relation of the task prompt,
    as in the repair feedback). Unsafe and no-code scripts are never picked."""
    from llmdpeval.run_grid import parse_cell
    crashed, flagged = set(), set()
    for line in (src / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") in ("unsafe_not_run", "no_code"):  # never run / nothing to repair (graded N)
            continue
        cell, task, run = file_key(r["code_file"])
        key = (cell, f"{task}_code_{run}.py")
        if r["runnable"] != "True":
            crashed.add(key)
        elif (src / cell / key[1]).exists() and static_problems(
                load_prompt(task.replace("-", "_"), parse_cell(cell)["framework"]),
                (src / cell / key[1]).read_text(encoding="utf-8", errors="replace"), r.get("stdout") or ""):
            flagged.add(key)  # runs, but has a static privacy flaw: gets static-privacy feedback
    return crashed, flagged


def repair_grid(src: Path, out: Path, rounds: int, workers: int = 1) -> None:
    """Copy the scripts of SRC to OUT and repair the crashed and privacy-flagged ones in OUT (see the module
    docstring). Resumable: scripts that already have a .repair.json are skipped. workers > 1 repairs several scripts
    of one model at once and finishes each model before the next, so a shared SPHERE GPU never serves two models."""
    from llmdpeval import rag
    from concurrent.futures import ThreadPoolExecutor
    from llmdpeval.helper import DEFAULT_EMBED_MODEL, LLM_MODELS, build_llm, keep_awake, load_prompt
    from llmdpeval.run_grid import SAMPLED_TEMPERATURE, parse_cell
    check_exec_allowed()  # repair runs the generated code: never on the XDC
    keep_awake()
    crashed, flagged = select_jobs(src)
    for f in sorted(src.glob("*/*_code_*.py")):
        dst = out / f.parent.name / f.name
        if ".round" not in f.name and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)

    index = rag.load_index(DEFAULT_EMBED_MODEL, build=False)
    theory = rag.load_index(DEFAULT_EMBED_MODEL, "theory", build=False)
    (out / "repair_config.json").write_text(json.dumps(
        {"src": str(src), "rounds": rounds, "corpus": rag.corpus_hash(), "theory": rag.corpus_hash("theory"),
         "crashed": len(crashed), "runnable_flagged": len(flagged), "api_lookup": API_LOOKUP}), encoding="utf-8")
    llms = {}
    # One model at a time so the shared GPU does not swap models between requests.
    jobs = sorted(crashed | flagged, key=lambda cf: (parse_cell(cf[0])["model"], cf))

    def repair_one(i: int, cell: str, name: str) -> None:
        code_file = out / cell / name
        if code_file.with_suffix(".repair.json").exists():
            return
        c = parse_cell(cell)
        _, task, run = file_key(code_file)
        task = task.replace("-", "_")
        temp = 0.0 if run == 1 else SAMPLED_TEMPERATURE
        if c["think"]:  # same endpoint and think flag as the generation
            os.environ["SPHERE_THINK"] = c["think"]
        else:
            os.environ.pop("SPHERE_THINK", None)
        llm = cached_llm(llms, c["model"], temp, c["think"], build_llm)
        task_prompt = load_prompt(task, framework=c["framework"])
        try:
            if task.startswith("app_"):
                prompt = logged_prompt(src / cell / name)
            elif c["k"]:
                prompt, _ = rag.build_rag_prompt(index, task_prompt, c["framework"], c["k"], c["template"],
                                                 theory if c["theory"] else None, c["code"], c["holdout"])
            else:
                prompt = task_prompt
        except Exception as e:  # a missing log or a retrieval error is a result for this script
            status = f"repair_failed: {type(e).__name__}"
            code_file.with_suffix(".repair.json").write_text(json.dumps([{"status": status}]), encoding="utf-8")
        else:
            # repair_loop records an API timeout etc. in the log and restores the last executed round
            status = repair_loop(llm, code_file, prompt, rounds, task_prompt)[-1]["status"]
        if workers == 1:  # one script at a time, so no sibling run can be hit: kill anything this one left behind
            kill_sandbox_user()
        print(f"[{i}/{len(jobs)}] {status:14} {cell}/{name}", flush=True)

    if workers == 1:
        for i, (cell, name) in enumerate(jobs, 1):
            repair_one(i, cell, name)
    else:
        # One model at a time, its scripts in parallel: the shared SPHERE GPU never serves two models at once.
        numbered = list(enumerate(jobs, 1))
        for model in dict.fromkeys(parse_cell(cell)["model"] for _, (cell, _) in numbered):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(repair_one, i, cell, name) for i, (cell, name) in numbered
                           if parse_cell(cell)["model"] == model]
                try:
                    for f in futures:
                        f.result()
                except BaseException:  # Ctrl-C or a failing job: drop the queued scripts instead of running them all
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
    kill_sandbox_user()


def selftest() -> None:
    """A crashing script and a fake LLM that fixes it; unsafe code is not run; the fake-fix check."""
    class FakeLLM:
        def complete(self, prompt):
            assert "ZeroDivisionError" in prompt
            return "```python\nprint('fixed')\n```"

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t_code_1.py"
        f.write_text("print(1/0)\n")
        log = repair_loop(FakeLLM(), f, "PROMPT", rounds=2)
        assert [x["status"] for x in log] == ["error", "ok"], log
        assert f.read_text().strip() == "print('fixed')"
        f.write_text("import subprocess\n")
        assert repair_loop(FakeLLM(), f, "PROMPT", 2)[0]["status"] == "unsafe_not_run"
        assert fake_fix("x = 1/0\n", "try:\n    x = 1/0\nexcept Exception as e:\n    pass\nprint(1)\n", "1")
        assert fake_fix("print(1/0)\n", "print(1)\n", "")
        assert not fake_fix("print(1/0)\n", "print(1)\n", "1")

        # privacy feedback: a script that exits 0 but overspends gets one more round, then the fix is accepted
        app = ("Code...\n3. Use a total privacy budget of epsilon = 1 and delta = 0.\n"
               "5. Print the private values and the epsilon used by each output.\n")
        class PrivacyLLM:
            def complete(self, prompt):
                assert "privacy-accounting problem" in prompt and "is 2" in prompt, prompt[-400:]
                return "```python\nprint('Total epsilon spent: 1.0')\n```"
        f.write_text("print('Total epsilon spent: 2.0')\n")
        log = repair_loop(PrivacyLLM(), f, app, rounds=2)
        assert [x["status"] for x in log] == ["privacy_feedback", "ok"], log
        f.write_text("print('Total epsilon spent: 2.0')\n")
        assert [x["status"] for x in repair_loop(PrivacyLLM(), f, app, rounds=0)] == ["privacy_feedback"]
    # static privacy: a runnable script with a non-private model gets static feedback before epsilon feedback,
    # and every round logs runnable + stdout, so the final run can be graded without a re-run
    class StaticLLM:
        def complete(self, prompt):
            assert "static privacy check" in prompt and "trained on private rows without DP" in prompt, prompt[-600:]
            return "```python\nprint('Total epsilon spent: 1.0')\n```"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t_code_1.py"
        f.write_text("from sklearn.linear_model import LogisticRegression\nclf = LogisticRegression()\n"
                     "print('Total epsilon spent: 2.0')\n")
        log = repair_loop(StaticLLM(), f, app, rounds=2)
        assert [x["status"] for x in log] == ["static_privacy", "ok"], log
        assert log[-1]["runnable"] is True and "Total epsilon spent: 1.0" in log[-1]["stdout_tail"], log
        saved = json.loads(f.with_suffix(".repair.json").read_text())
        assert saved[-1]["stdout_tail"] == log[-1]["stdout_tail"]
        f.write_text("# llmdpeval: no_code\n# prose\n")
        assert [x["status"] for x in repair_loop(StaticLLM(), f, app, 2)] == ["no_code"]
    assert REPLACE_ONE.search("Two datasets are neighbours if one row is replaced by another (replace-one).") \
        and not REPLACE_ONE.search("Neighbouring datasets differ by adding or removing one row (add/remove)")
    # (i) no epsilon printed although the prompt asks for it; not required -> no problem
    assert "no value labelled as epsilon" in privacy_problem(app, "accuracy 0.8\n")
    assert privacy_problem("3. Use the following privacy parameters: epsilon = 1\n", "accuracy 0.8\n") is None
    # (iii) total over budget; per-release sum over budget; the budget is read from requirement lines only
    toy = "Query type: Count\n3. Use the following privacy parameters: mechanism = laplace, epsilon = 1\n"
    assert "is 1.5" in privacy_problem(toy, "Epsilon for count: 0.75\nEpsilon for mean: 0.75\n")
    assert privacy_problem(toy, "Epsilon for count: 0.5\nEpsilon for mean: 0.5\nTotal epsilon spent: 1.0\n") is None
    assert privacy_problem(toy, "Epsilon spent (framework accounting): 1.0000\nEpsilon implied by sensitivity "
                                "(1) and noise scale (b=2): 1.0000\n") is None
    assert privacy_problem(toy + "\nOutlines:\n    epsilon = 0.5\n", "Epsilon spent: 1.0\n") is None
    assert "is 3" in privacy_problem("    - epsilon = 1\n", '{"dp": 7510, "epsilon_total": 3}\n')
    assert printed_epsilons("Framework reported epsilon spent: 1.0, delta spent: 0.0\n"
                            "Laplace (epsilon=0.5, scale=2): 7510\nTop-5 (framework epsilon spent: 0.9):\n") == [
        ("Framework reported epsilon spent: 1.0, delta spent: 0.0", 1.0),
        ("Laplace (epsilon=0.5, scale=2): 7510", 0.5), ("Top-5 (framework epsilon spent: 0.9):", 0.9)]
    # (ii) zero, negative or NaN
    for bad in ("0", "-1", "nan"):
        assert "zero, negative or NaN" in privacy_problem(toy, f"Epsilon spent: {bad}\n"), bad
    # one client per (model, temperature, think)
    built = []
    llms = {}
    for _ in range(3):
        cached_llm(llms, "m", 0.0, None, lambda m, t: built.append(m) or object())
    assert len(built) == 1, built
    print("repair self-check ok")


def main() -> None:
    """Repair a grid (SRC OUT) or run the self-test."""
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", type=Path)
    p.add_argument("out", nargs="?", type=Path)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--workers", type=int, default=1, help="scripts of one model repaired at once (models run one after another)")
    p.add_argument("--api-lookup", action="store_true", help="add exact pinned-API entries to crash feedback")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    global API_LOOKUP
    API_LOOKUP = a.api_lookup
    if a.selftest:
        selftest()
    elif a.src and a.out:
        repair_grid(a.src, a.out, a.rounds, a.workers)
    else:
        p.error("give SRC_GRID OUT_GRID, or --selftest")


if __name__ == "__main__":
    main()
