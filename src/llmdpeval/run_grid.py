"""Run a grid: models x conditions x tasks x runs, then evaluate the generated code.

  python -m llmdpeval.run_grid --grid-dir generated_code/grid_X --conditions OpenDP-rag-k3 --theory --code
  python -m llmdpeval.run_grid --grid-dir generated_code/grid_X --evaluate

Run 1 of each cell uses temperature 0; runs 2..N use SAMPLED_TEMPERATURE.
Defaults come from pipeline.toml (flags override). The first generation run of a grid saves the resolved settings
in <grid_dir>/pipeline_frozen.json; a later run with different settings is refused (freeze_pipeline).
Existing files are skipped, so the script can be re-run to resume. With --repair, a script counts as done only
once its .repair.json exists. A generation killed by the hard cap (during the model call or during repair) is
marked <task>_code_<n>.timeout and is not generated again on resume; whatever code it saved is kept for evaluation.
Repair normally runs as a separate step on the evaluation node (pipeline.toml repair = 0; python -m
llmdpeval.repair SRC OUT --rounds 2).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdpeval.helper import (FRAMEWORKS, LLM_MODELS, LOCK_FILE, PIPELINE_CONFIG, REPO_ROOT, TASKS, code_files,
                              keep_awake, lf_hash, safe_path_part)
from llmdpeval.sandbox import check_exec_allowed, kill_sandbox_user, run_sandboxed
from llmdpeval.verify_env import mismatches

CONDITIONS = {  # name: (framework, K = outlines retrieved into the prompt; 0 = the task prompt alone)
    **{fw: (fw, 0) for fw in FRAMEWORKS},
    **{f"{fw}-rag-k{k}": (fw, k) for fw in FRAMEWORKS for k in PIPELINE_CONFIG["rag_k"]},
}
SAMPLED_TEMPERATURE = PIPELINE_CONFIG["sampled_temperature"]
# Which cells a run fills, not how each one is generated: a grid can be filled over several runs.
SELECTION_KEYS = ("models", "conditions", "runs")


def cell_dir(grid_dir: Path, model: str, framework: str, k: int, o: argparse.Namespace) -> Path:
    """<model>_<rag part>_<framework><flags>, e.g. gemma4-latest_ragk3_OpenDP_th_code_thinkoff.
    Prompt flags only mark RAG cells (they change nothing without retrieval); repair and think mark every cell."""
    rag_part = "" if not k else f"rag{'plain' if o.template == 'plain' else ''}k{k}"
    suffix = (("_th" if o.theory else "") + (("_codeHO" if o.holdout else "_code") if o.code else "")) if k else ""
    suffix += (f"_fix{o.repair}" if o.repair else "") + (f"_think{o.think}" if o.think else "")
    return grid_dir / ("_".join([safe_path_part(model), rag_part, framework]) + suffix)


def parse_cell(name: str) -> dict:
    """Inverse of cell_dir. Also accepts the legacy rag parts "ragv2gk<K>" (guided) and "ragv2k<K>" (plain)."""
    model_part, rag_part, framework, *flags = name.split("_")
    k = int(rag_part.rsplit("k", 1)[1]) if rag_part else 0
    plain = rag_part.startswith(("ragplaink", "ragv2k"))
    models = {safe_path_part(m): m for m in LLM_MODELS}
    return {"model": models.get(model_part, model_part), "framework": framework, "k": k,
            "template": "plain" if plain else "guided",
            "theory": "th" in flags, "code": any(f.startswith("code") for f in flags), "holdout": "codeHO" in flags,
            "repair": next((int(f[3:]) for f in flags if f.startswith("fix")), 0),
            "think": next((f.removeprefix("think") for f in flags if f in ("thinkon", "thinkoff")), None),
            "condition": framework + (f"-rag{'plain' if plain else ''}-k{k}" if k else "")
                         + "".join(f"_{f}" for f in flags)}


def done(code_file: Path, repair: int) -> bool:
    """A generation is final when its code exists and, with repair rounds, its repair log too: a hard-cap kill
    during repair leaves a half-repaired file without .repair.json."""
    return code_file.exists() and (not repair or code_file.with_suffix(".repair.json").exists())


def generate(grid_dir: Path, model: str, framework: str, k: int, task: str, run: int,
             o: argparse.Namespace) -> None:
    """One generation in a subprocess (llmdpeval.generate), logged to <task>_gen_<run>.log and capped at
    o.hard_cap seconds. Skips work that is already done or marked as timed out."""
    out_dir = cell_dir(grid_dir, model, framework, k, o)
    code_file = out_dir / f"{safe_path_part(task)}_code_{run}.py"
    timeout_file = code_file.with_suffix(".timeout")
    if done(code_file, o.repair) or timeout_file.exists():
        return
    # Repair rounds left by an earlier attempt that did not finish belong to code that is about to be replaced.
    for stale in out_dir.glob(f"{code_file.stem}.round*.py"):
        stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    temperature = 0.0 if run == 1 else SAMPLED_TEMPERATURE
    command = [sys.executable, "-m", "llmdpeval.generate", "--llm", model, "--task", task, "--framework", framework,
               "--temperature", str(temperature), "--run-index", str(run), "--output-dir", str(out_dir.resolve())]
    if k:
        command += ["--k", str(k), "--template", o.template]
        command += (["--theory"] if o.theory else []) + (["--code"] if o.code else []) + (["--holdout"] if o.holdout else [])
    if o.repair:
        command += ["--repair", str(o.repair)]
    if o.think:
        command += ["--think", o.think]
    if o.reask:
        command += ["--reask"]
    log_file = out_dir / f"{safe_path_part(task)}_gen_{run}.log"
    with log_file.open("w", encoding="utf-8") as log:
        # RAG prompts contain non-cp1252 characters; Windows would crash printing them to the log.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env, cwd=REPO_ROOT,
                                start_new_session=sys.platform != "win32")
        try:
            returncode = proc.wait(timeout=o.hard_cap)
        except subprocess.TimeoutExpired:
            # The client timeout does not fire while the server holds the connection open,
            # so cap wall-clock time here. Kill the whole tree (on Windows the venv launcher has a child python).
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode = "killed"
            log.write(f"\nAPITimeoutError: killed by run_grid after {o.hard_cap} s\n")
    # A timed-out run is a result ("no answer in time"); mark it so resumes don't retry it. A run killed by the
    # hard cap is always marked, also when it had saved code and was killed during repair: the saved code (the
    # round being run when it was killed) is what gets evaluated.
    if returncode == "killed":
        timeout_file.write_text(f"killed by run_grid after {o.hard_cap} s\n", encoding="utf-8")
    elif not code_file.exists() and "APITimeoutError" in log_file.read_text(encoding="utf-8", errors="replace"):
        timeout_file.write_text("no answer within the client timeout\n", encoding="utf-8")
    print(f"{'ok ' if code_file.exists() else 'ERR'} {model} {framework} k={k} {task} run {run}"
          f" (exit {returncode})", flush=True)


def file_hash(path: Path) -> str:
    """sha256 of a text file with LF line endings (the same on Windows and Linux checkouts)."""
    return lf_hash([path])


def freeze_pipeline(grid_dir: Path, o: argparse.Namespace, commit: str) -> None:
    """Save the resolved pipeline (pipeline.toml + flags), the corpus and theory hashes, the lock hash and the commit
    in <grid_dir>/pipeline_frozen.json. If the file exists, exit unless everything but the cell selection
    (SELECTION_KEYS) and the commit matches it."""
    from llmdpeval.retrieval import corpus_hash, skeleton_hash
    config = {**PIPELINE_CONFIG, "models": o.models, "conditions": o.conditions, "runs": o.runs,
              "template": o.template, "theory": o.theory, "code": o.code, "holdout": o.holdout,
              "think": o.think or "", "repair": o.repair, "reask": o.reask, "hard_cap": o.hard_cap}
    frozen = {"config": config, "corpus": corpus_hash(), "theory": corpus_hash("theory"), "skeletons": skeleton_hash(),
              "lock": f"{o.lock.name} {file_hash(o.lock)}", "commit": commit}
    path = grid_dir / "pipeline_frozen.json"
    if not path.exists():
        path.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    diff = [f"  {k}: frozen {old['config'].get(k)!r}, now {config.get(k)!r}"
            for k in sorted(set(old["config"]) | set(config))
            if k not in SELECTION_KEYS and old["config"].get(k) != config.get(k)]
    diff += [f"  {k}: frozen {old.get(k)}, now {frozen[k]}" for k in ("corpus", "theory", "skeletons", "lock")
             if old.get(k) != frozen[k] and (k != "skeletons" or k in old)]  # older frozen files lack a skeleton hash
    if diff:
        sys.exit(f"{path} does not match this run:\n" + "\n".join(diff)
                 + "\nThis grid was frozen with other settings. Use a new --grid-dir, or restore the frozen settings.")
    if old.get("commit") != commit:
        print(f"note: grid frozen at commit {old.get('commit')}, now at {commit}", flush=True)


def display_path(path: Path) -> str:
    """`path` relative to the repo root when it is inside it, else as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def evaluate(grid_dir: Path, workers: int = 1) -> Path:
    """Run every generated script in its own sandbox (sandbox.run_sandboxed); workers > 1 only changes speed.
    Each result is appended to results.jsonl as soon as it is in, so a crash keeps the results so far. Code that
    sandbox.is_unsafe flags is not run (status unsafe_not_run)."""
    check_exec_allowed()
    results_file = grid_dir.resolve() / "results.jsonl"

    def run_one(code_file: Path) -> dict:
        run = run_sandboxed(code_file)
        run["code_file"] = display_path(code_file)
        run["cell"] = code_file.parent.name
        print(f"{run.get('status', run['runnable']):5} {run['code_file']}", flush=True)
        return run

    try:
        with ThreadPoolExecutor(workers) as pool, results_file.open("w", encoding="utf-8") as out:
            for run in pool.map(run_one, code_files(grid_dir.resolve())):
                out.write(json.dumps(run) + "\n")
                out.flush()
    finally:
        kill_sandbox_user()  # end of the pass: no run is in progress, so anything left of the sandbox user goes
    return results_file


def main() -> None:
    """Generate the grid, or with --evaluate run its scripts."""
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", required=True, type=Path, help="e.g. generated_code/grid_A")
    p.add_argument("--runs", type=int, default=PIPELINE_CONFIG["runs"])
    p.add_argument("--workers", type=int, default=3, help="concurrent requests within one model")
    p.add_argument("--models", nargs="+", default=PIPELINE_CONFIG["models"], choices=list(LLM_MODELS))
    p.add_argument("--tasks", nargs="+", default=TASKS, help="prompt names, or app_<name> for a paper application")
    p.add_argument("--conditions", nargs="+", default=PIPELINE_CONFIG["conditions"], choices=list(CONDITIONS))
    p.add_argument("--template", choices=["guided", "plain"], default=PIPELINE_CONFIG["template"],
                   help="RAG prompt template")
    flag = argparse.BooleanOptionalAction  # --theory / --no-theory: a pipeline.toml "true" can be turned off
    p.add_argument("--theory", action=flag, default=PIPELINE_CONFIG["theory"],
                   help="add one DP theory card to RAG prompts")
    p.add_argument("--code", action=flag, default=PIPELINE_CONFIG["code"],
                   help="follow each outline with its code skeleton")
    p.add_argument("--holdout", action=flag, default=PIPELINE_CONFIG["holdout"],
                   help="with --code: hide the grid tasks' own skeletons")
    p.add_argument("--repair", type=int, default=PIPELINE_CONFIG["repair"],
                   help="repair rounds inline (runs the code here; default 0: repair is a separate step on the "
                        "evaluation node, python -m llmdpeval.repair SRC OUT --rounds 2)")
    p.add_argument("--think", choices=["on", "off"], default=PIPELINE_CONFIG["think"] or None,
                   help="explicit think flag via the Ollama-native endpoint")
    p.add_argument("--reask", action=flag, default=PIPELINE_CONFIG["reask"],
                   help="re-ask once when an answer has no code")
    p.add_argument("--hard-cap", type=int, default=PIPELINE_CONFIG["hard_cap"],
                   help="wall-clock cap per generation in s (paper apps make two calls: analysis + code)")
    p.add_argument("--evaluate", action="store_true", help="only run the generated code")
    p.add_argument("--eval-workers", type=int, default=1, help="scripts run in parallel during --evaluate")
    p.add_argument("--lock", type=Path, default=LOCK_FILE, help="lock file the environment must match")
    o = p.parse_args()
    if o.evaluate or o.repair:
        check_exec_allowed()  # generated code never runs on the XDC

    # Refuse to run in an environment that differs from the lock: library versions change the generated code's behaviour.
    bad = mismatches(o.lock)
    if bad:
        sys.exit("\n".join(bad) + f"\nEnvironment does not match {o.lock}; use the venv built from it.")
    o.grid_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=REPO_ROOT).stdout.strip()
    with (o.grid_dir / "run_config.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "lock": o.lock.name, "python": sys.version.split()[0],
                            "platform": sys.platform, "commit": commit}) + "\n")

    if o.evaluate:
        print(f"Results: {evaluate(o.grid_dir, o.eval_workers)}")
        return
    freeze_pipeline(o.grid_dir, o, commit)
    keep_awake()
    # One model at a time so the shared GPU does not swap models between requests.
    for model in o.models:
        jobs = [(o.grid_dir, model, *CONDITIONS[c], t, n)
                for c in o.conditions for t in o.tasks for n in range(1, o.runs + 1)]
        with ThreadPoolExecutor(o.workers) as pool:
            list(pool.map(lambda job: generate(*job, o), jobs))


if __name__ == "__main__":
    main()
