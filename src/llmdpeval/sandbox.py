"""Run one generated script the way every evaluation does: a fresh scratch dir with a copy of the script and
copies of the data its task needs.

Some generated scripts overwrite or delete the CSVs, so they never run in the repo root. Used by
run_grid --evaluate, repair and grade median, so all three run code the same way.

What this does: cwd = the scratch dir, the script runs from a copy there (its __file__ does not point into the
repo), an allowlisted environment (ENV_ALLOW: the API keys that generation and repair load are not passed on),
a timeout, and code that is_unsafe flags is not run at all (a triage filter).
On a local machine (e.g. Windows) this does not isolate: the script runs with the user's rights and has network
access, so keep the *_key.txt files outside the repo (LLMDPEVAL_KEY_DIR, see helper.key_dir).
On Linux, generated code runs only as a separate sandbox user (LLMDPEVAL_SANDBOX_USER, set up by
sphere/setup_node.sh on a SPHERE experiment node): no network, no access to your home, memory limit via prlimit.
Generated code never runs on a SPHERE XDC: check_exec_allowed() refuses there.
"""
import ast
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from llmdpeval.helper import APP_DIR, DATA_DIR, code_status, safe_path_part

# Defaults; an app can set its own in applications/<app>/eval.json ({"timeout_s": ..., "memory_gb": ...}).
# An env var that is set wins over both (eval_settings).
RUN_TIMEOUT_SECONDS = 120
SANDBOX_MEMORY_GB = 16  # address-space limit per script
SANDBOX_USER = os.environ.get("LLMDPEVAL_SANDBOX_USER")  # Linux: run generated code as this user (setup_node.sh)
# Code that uses any of these is never executed (repair, --evaluate, grade median); it is left for hand review.
# A triage filter on the code's AST (imports and calls), not isolation: comments and strings do not count.
UNSAFE_MODULES = {"subprocess", "shutil", "socket", "requests", "urllib", "urllib3", "http", "httpx", "ftplib",
                  "smtplib", "telnetlib"}  # network, processes, file trees ("http" covers http.client)
UNSAFE_OS = {"system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace"}
UNSAFE_METHODS = {"unlink", "rmdir", "rmtree"}  # pathlib.Path.unlink / rmdir, shutil.rmtree on any object
# Fallback for code that does not parse (it cannot run either, but is still flagged for review).
UNSAFE = r"\b(subprocess|shutil|socket|requests|urllib|http\.client|os\.(remove|system|unlink|rmdir|popen|rename))\b"
# Generated code gets only these variables: generation and repair put *_API_KEY in os.environ, and an
# LLM-written script must not be able to read them.
ENV_ALLOW = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
             "LANG", "LC_ALL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def is_unsafe(code: str) -> bool:
    """True if the code imports an UNSAFE_MODULES module (also via __import__ / importlib.import_module with a
    literal name), calls an UNSAFE_OS function of os (`os.system(...)`, `from os import remove`), or calls an
    UNSAFE_METHODS method on anything. Code that does not parse falls back to the UNSAFE regex."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return bool(re.search(UNSAFE, code))
    top = lambda name: name.split(".")[0]
    os_names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if top(a.name) in UNSAFE_MODULES:
                    return True
                if a.name == "os":
                    os_names.add(a.asname or "os")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if top(node.module) in UNSAFE_MODULES:
                return True
            if node.module == "os" and any(a.name in UNSAFE_OS or a.name == "*" for a in node.names):
                return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            if f.attr in UNSAFE_METHODS:
                return True
            if f.attr in UNSAFE_OS and isinstance(f.value, ast.Name) and f.value.id in os_names:
                return True
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name in ("__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and top(node.args[0].value) in UNSAFE_MODULES | {"os"}:
            return True
    return False


def exec_refused_reason() -> str | None:
    """Why generated code must not run on this machine, or None. A SPHERE XDC is a shared Linux container: its
    hostname contains "xdc" (<xdc>-<project>, FQDN *.xdc.*). An XDC named otherwise needs LLMDPEVAL_NO_EXEC=1."""
    if os.environ.get("LLMDPEVAL_NO_EXEC") == "1":
        return "LLMDPEVAL_NO_EXEC=1 is set"
    if sys.platform.startswith("linux") and "xdc" in (socket.gethostname() + " " + socket.getfqdn()).lower():
        return "this looks like a SPHERE XDC (hostname contains 'xdc')"
    if sys.platform.startswith("linux") and not SANDBOX_USER:
        return "on Linux generated code runs only as a sandbox user: run sphere/setup_node.sh, set LLMDPEVAL_SANDBOX_USER"
    return None


def check_exec_allowed() -> None:
    """Exit if generated code must not run here (exec_refused_reason)."""
    reason = exec_refused_reason()
    if reason:
        raise SystemExit(f"Refusing to run generated code here: {reason}. See README, 'Where code runs'.")


def child_env() -> dict:
    """The environment a generated script gets: only the ENV_ALLOW variables, plus UTF-8 stdio."""
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    # UTF-8 stdout on every platform: under Windows cp1252, printing e.g. an emoji raises UnicodeEncodeError,
    # an environment failure rather than a model error.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def app_dir(code_file: Path) -> Path | None:
    """applications/<app> for a script of task app_<app> (file name app-<app>_code_<n>.py), else None."""
    task = Path(code_file).name.split("_code_")[0]  # <task>_code_<run>.py
    if task.startswith("app"):
        apps = {safe_path_part("app_" + d.name): d for d in APP_DIR.iterdir() if d.is_dir()}
        return apps.get(task)
    return None


def data_files(code_file: Path) -> list[Path]:
    """The data a script's task needs: applications/<app>/data/* for app_<app>, else data/*.csv."""
    app = app_dir(code_file)
    if app:
        return sorted(p for p in (app / "data").glob("*") if p.is_file())
    return sorted(DATA_DIR.glob("*.csv"))


def eval_settings(code_file: Path) -> dict:
    """{"timeout_s", "memory_gb"} for this script: env var (LLMDPEVAL_RUN_TIMEOUT, LLMDPEVAL_SANDBOX_MEMORY_GB) if
    set, else applications/<app>/eval.json, else the defaults. Used by run_grid --evaluate, repair and grade median
    (all run code through run_sandboxed)."""
    app = app_dir(code_file)
    cfg = json.loads((app / "eval.json").read_text(encoding="utf-8")) if app and (app / "eval.json").is_file() else {}
    return {"timeout_s": int(os.environ.get("LLMDPEVAL_RUN_TIMEOUT") or cfg.get("timeout_s") or RUN_TIMEOUT_SECONDS),
            "memory_gb": int(os.environ.get("LLMDPEVAL_SANDBOX_MEMORY_GB") or cfg.get("memory_gb") or SANDBOX_MEMORY_GB)}


def command(code_file: Path, cwd: Path, memory_gb: int = SANDBOX_MEMORY_GB,
            user: str | None = SANDBOX_USER) -> tuple[list, dict]:
    """(argv, env) to run a script copy in `cwd`. With a sandbox user: via sudo as that user, inside `cwd`
    (env -C) and under a prlimit address-space limit."""
    # "." = the data directory: paper-app scripts take it as argv[1]; toy-task scripts ignore it.
    cmd, env = [sys.executable, str(Path(code_file).resolve()), "."], child_env()
    if user:
        env["HOME"] = str(cwd)
        # env -C: the sandbox user enters its private (0700) scratch dir itself; we cannot.
        cmd = ["sudo", "-n", "-u", user, "env", "-C", str(cwd), *[f"{k}={v}" for k, v in env.items()],
               "prlimit", f"--as={memory_gb * 2**30}", "--"] + cmd
    return cmd, env


# Per-run cleanup on Linux. Each sandboxed run starts in its own session (start_new_session), so the sudo process
# and everything the script starts share one process group; that group is killed as the sandbox user after the
# run, on timeout and after a normal exit, so a background child cannot outlive its run. Parallel runs have
# different groups and never kill each other. A process that leaves its group (setsid, daemonising) escapes this;
# kill_sandbox_user() at the end of an evaluation pass (run_grid --evaluate, grade median) and after each script
# in repair (one script at a time) removes those.
def kill_group_command(user: str, pgid: int) -> list:
    """argv that kills process group `pgid` as `user`."""
    return ["sudo", "-n", "-u", user, "kill", "-KILL", "--", f"-{pgid}"]


def kill_user_command(user: str) -> list:
    """argv that kills every process of `user`."""
    return ["sudo", "-n", "pkill", "-KILL", "-u", user]


def scratch_owner_commands(user: str, scratch: Path) -> list[list]:
    """Hand the scratch dir to the sandbox user, private (0700): other users of the node cannot read the data or
    the script, and the sandbox user writes its outputs there."""
    return [["sudo", "-n", "chown", "-R", f"{user}:", str(scratch)], ["sudo", "-n", "chmod", "700", str(scratch)]]


def kill_sandbox_user() -> None:
    """Kill every process of the sandbox user. Only when no other sandboxed run is in progress: at the end of an
    evaluation pass, or between scripts when they run one at a time. No-op without a sandbox user."""
    if SANDBOX_USER:
        subprocess.run(kill_user_command(SANDBOX_USER), capture_output=True)


def _run_in_group(cmd: list, cwd: Path, env: dict, timeout_s: int) -> tuple[int | None, str, str]:
    """Run cmd in a new session; kill its process group as the sandbox user after it ends or times out.
    Output goes to temporary files, not pipes, so a leftover process holding them open cannot block the read."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        # No cwd here: the scratch dir is private to the sandbox user; cmd enters it (env -C, see command()).
        proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            returncode = None
        finally:
            subprocess.run(kill_group_command(SANDBOX_USER, proc.pid), capture_output=True)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        out.seek(0)
        err.seek(0)
        decode = lambda f: f.read().decode("utf-8", errors="replace")
        return returncode, decode(out), decode(err)


def run_python_file(code_file: Path, cwd: Path, timeout_s: int = RUN_TIMEOUT_SECONDS,
                    memory_gb: int = SANDBOX_MEMORY_GB) -> dict:
    """Run a script already placed in `cwd`; returns code_file, runnable ("True"/"False"), returncode (None on
    timeout), stdout and stderr."""
    cmd, env = command(code_file, cwd, memory_gb)
    if SANDBOX_USER:
        returncode, stdout, stderr = _run_in_group(cmd, cwd, env, timeout_s)
        return {"code_file": str(code_file), "runnable": str(returncode == 0), "returncode": returncode,
                "stdout": stdout, "stderr": stderr if returncode is not None else f"Timed out after {timeout_s} seconds"}
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        return {
            "code_file": str(code_file),
            "runnable": str(result.returncode == 0),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        if isinstance(stdout, bytes):  # POSIX gives bytes here even with text=True
            stdout = stdout.decode("utf-8", errors="replace")
        return {
            "code_file": str(code_file),
            "runnable": "False",
            "returncode": None,
            "stdout": stdout,
            "stderr": f"Timed out after {timeout_s} seconds",
        }


def run_sandboxed(code_file: Path, data: list[Path] | None = None, extra: tuple = (),
                  settings: dict | None = None) -> dict:
    """Run a generated script from a copy in a fresh scratch dir with its task's data; returns the run_python_file
    result. Thread-safe (no chdir). An answer without code gets status "no_code", and code that is_unsafe flags
    gets status "unsafe_not_run"; neither is run. `data` replaces the task's data files (the audit passes
    neighbouring copies), `extra` files are copied next to the script (modules it imports), and `settings`
    replaces eval_settings(code_file)."""
    check_exec_allowed()
    code_file = Path(code_file)
    code = code_file.read_text(encoding="utf-8", errors="replace")
    status = code_status(code)
    if status == "no_code":  # the answer had no code: graded N, nothing to run
        return {"code_file": str(code_file), "runnable": "False", "returncode": None, "stdout": "",
                "stderr": "no_code: the answer contained no code", "status": "no_code"}
    if is_unsafe(code):
        return {"code_file": str(code_file), "runnable": "False", "returncode": None, "stdout": "",
                "stderr": "unsafe_not_run: matches sandbox.UNSAFE; review by hand", "status": "unsafe_not_run"}
    # ignore_cleanup_errors: on Windows a leftover child can hold a file; that must not lose the result.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        for src in [*(data_files(code_file) if data is None else data), *extra]:
            shutil.copy(src, scratch)
        copy = Path(scratch) / code_file.name
        shutil.copy(code_file, copy)
        try:
            if SANDBOX_USER:  # the sandbox user owns the private scratch dir: reads the copies, writes its outputs
                for cmd in scratch_owner_commands(SANDBOX_USER, Path(scratch)):
                    if subprocess.run(cmd, capture_output=True).returncode != 0:
                        raise RuntimeError(f"could not hand {scratch} to {SANDBOX_USER}: {' '.join(cmd)} failed")
            run = run_python_file(copy, cwd=Path(scratch), **(settings or eval_settings(code_file)))
        finally:
            if SANDBOX_USER:  # always take the dir back, so it can be removed even if the run failed to start
                subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", scratch],
                               capture_output=True)
                subprocess.run(["sudo", "-n", "chmod", "700", scratch], capture_output=True)
        run["code_file"] = str(code_file)
        if status == "truncated" and run["runnable"] != "True":
            run["status"] = "truncated"  # the answer was cut off (no closing fence): not an ordinary error
        return run


if __name__ == "__main__":  # self-check
    os.environ.setdefault("SPHERE_LLM_API_KEY", "dummy-for-selfcheck")
    assert not [k for k in child_env() if "KEY" in k.upper() or "TOKEN" in k.upper()], sorted(child_env())
    # per-task data: toy tasks get data/*.csv only, an app gets only its own data folder
    toy = {p.name for p in data_files(Path("g/c/stat-count_code_1.py"))}
    assert toy == {"adult_clean.csv", "auto_mpg_clean.csv"}, toy
    for app in (d for d in APP_DIR.iterdir() if (d / "data").is_dir()):
        got = data_files(Path(f"g/c/{safe_path_part('app_' + app.name)}_code_1.py"))
        assert got and all(p.parent == app / "data" for p in got), (app, got)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stat-count_code_1.py"
        # runs from a copy in scratch (not from d), cwd = scratch, toy data present
        f.write_text("import os, sys; print(os.path.dirname(os.path.abspath(__file__)) == os.getcwd(), "
                     "os.path.exists('adult_clean.csv'), sys.argv[1])\n")
        r = run_sandboxed(f)
        assert r["stdout"].split() == ["True", "True", "."] and r["code_file"] == str(f), r
        f.write_text("import shutil\nprint(1)\n")
        assert run_sandboxed(f)["status"] == "unsafe_not_run"
        os.environ["LLMDPEVAL_NO_EXEC"] = "1"
        try:
            run_sandboxed(f)
            raise AssertionError("XDC guard did not fire")
        except SystemExit as e:
            assert "LLMDPEVAL_NO_EXEC" in str(e)
        del os.environ["LLMDPEVAL_NO_EXEC"]
        if SANDBOX_USER:  # on a node: no network, no read access to our home
            f.write_text("import os, socket\n"
                         "try:\n    socket.create_connection(('1.1.1.1', 443), 5); print('net')\n"
                         "except OSError:\n    print('nonet')\n"
                         f"print(os.access({str(Path.home())!r}, os.R_OK))\n")
            os.chmod(d, 0o777); os.chmod(f, 0o644)
            out = run_python_file(f, cwd=Path(d))  # direct call: is_unsafe would skip a socket import
            assert out["stdout"].split() == ["nonet", "False"], out
            # a background child does not outlive its run (process-group kill after a normal exit)
            f.write_text("import os, time\nif os.fork() == 0:\n    time.sleep(300)\nprint('parent')\n")
            assert run_python_file(f, cwd=Path(d))["stdout"].split() == ["parent"]
            left = subprocess.run(["pgrep", "-u", SANDBOX_USER], capture_output=True, text=True).stdout.split()
            assert not left, f"processes of {SANDBOX_USER} left after the run: {left}"
    print("sandbox self-check ok" + (f" (as {SANDBOX_USER})" if SANDBOX_USER else ""))
