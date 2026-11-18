"""Validate the corpus against corpus/FORMAT.md: outlines (front matter, API names resolve, no leakage, size),
code skeletons (each _test runs and reports epsilon <= 1) and theory cards.

Usage: python -m llmdpeval.check_corpus [--root corpus] [--workers 4] [--no-skeletons]
Exit code 1 if anything fails.
"""
import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from llmdpeval.helper import CORPUS, FRAMEWORK_DIRS
from llmdpeval.sandbox import child_env

REQUIRED = ["id", "title", "framework", "framework_version", "mechanism", "task_type",
            "dp_notion", "needs_bounds", "api_calls", "source_url", "source_license"]
SECTIONS = ["## Goal", "## Steps", "## Privacy accounting", "## Pitfalls"]
# Test-task leakage (FORMAT.md rules). Word-boundary match, case-insensitive.
FORBIDDEN = re.compile(r"\b(adult|income|occupation|auto_mpg|mpg|7508)\b", re.I)
# Theory cards are stricter: substring match, and `weight` (the LinReg predictor) is banned too.
FORBIDDEN_THEORY = re.compile(r"adult|income|occupation|auto_mpg|mpg|weight|7508", re.I)
# Pseudo-code only: nothing pasteable such as imports or data loading.
RUNNABLE = re.compile(r"^\s*(import \w|from [\w.]+ import )|read_csv|read_parquet|pl\.scan_|open\(", re.M)
SKELETON_TIMEOUT = 180  # an OpenDP skeleton spends ~15 s importing opendp/polars; headroom for a busy machine

THEORY_TOPICS = {"epsilon", "sensitivity", "neighbours", "mechanism", "composition", "bounds",
                 "selection", "approx-dp", "post-processing", "implementation"}
THEORY_TASKS = {"count", "sum", "mean", "variance", "histogram", "quantile", "selection",
                "regression", "classification", "pca", "inference", "composition", "other"}  # = outline task_type vocabulary
THEORY_SECTIONS = ["## Idea", "## Why it matters when writing DP code", "## Common mistakes"]


def resolve(name: str) -> bool:
    """True if a dotted name like opendp.measurements.make_laplace exists (module prefix + attributes)."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return True
        except AttributeError:
            return False
    return False


def check(path: Path) -> list[str]:
    """Errors of one outline: front matter, sections, API names, leak words, pasteable code, length."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return ["no front matter"]
    meta, body = yaml.safe_load(m.group(1)) or {}, m.group(2)
    errors = [f"missing field {f}" for f in REQUIRED if f not in meta]
    if meta.get("id") != path.stem:
        errors.append(f"id {meta.get('id')!r} != filename")
    if meta.get("framework") != path.parent.name:
        errors.append(f"framework {meta.get('framework')!r} != folder {path.parent.name}")
    errors += [f"missing section {s}" for s in SECTIONS if s not in body]
    errors += [f"api not found: {a}" for a in meta.get("api_calls") or [] if not resolve(a)]
    errors += [f"leak word: {w}" for w in sorted({w.lower() for w in FORBIDDEN.findall(body + meta.get("title", ""))})]
    errors += [f"runnable code: {m.group(0).strip()!r}" for m in RUNNABLE.finditer(body)]
    words = len(body.split())
    if not 100 <= words <= 700:
        errors.append(f"{words} words (accepted 100-700; FORMAT.md target 150-500)")
    return errors


SEEDS = re.compile(r"random_state\s*=\s*\d|seed\s*\(|default_rng\s*\(\s*\d|RandomState\s*\(\s*\d")
# Runs one skeleton's _test() in a fresh interpreter, in a scratch dir; prints the returned epsilon.
TEST_RUNNER = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('sk', sys.argv[1]); "
               "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m._test()))")


def skeleton_without_test(path: Path) -> str:
    """The skeleton as shown in prompts: everything before `def _test`."""
    return path.read_text(encoding="utf-8").split("\ndef _test", 1)[0].rstrip() + "\n"


def check_skeleton(path: Path) -> list[str]:
    """Errors of one skeleton: static rules, then its _test() run in a fresh interpreter must report 0 < epsilon <= 1."""
    code = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    errors = []
    if not path.with_suffix(".md").exists():
        errors.append("no matching outline .md")
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "_test" not in funcs or not any(f.startswith("dp_") for f in funcs):
        errors.append(f"needs a dp_* function and _test (has {funcs})")
    shown = skeleton_without_test(path)
    errors += [f"seed: {m.group(0)}" for m in SEEDS.finditer(code)]
    errors += [f"leak word: {w}" for w in sorted({w.lower() for w in FORBIDDEN.findall(shown)})]
    if re.search(r"read_csv|read_parquet|open\(|to_csv", shown):
        errors.append("file I/O in skeleton")
    if len(shown.splitlines()) > 60:
        errors.append(f"{len(shown.splitlines())} lines before _test (want <= ~45)")
    if errors:
        return errors
    with tempfile.TemporaryDirectory() as scratch:
        try:
            run = subprocess.run([sys.executable, "-c", TEST_RUNNER, str(path.resolve())], cwd=scratch,
                                 capture_output=True, text=True, timeout=SKELETON_TIMEOUT,
                                 env={**child_env(), "PYTHONDONTWRITEBYTECODE": "1"})  # no API keys, no __pycache__
        except subprocess.TimeoutExpired:
            return [f"_test timed out ({SKELETON_TIMEOUT} s)"]
    if run.returncode != 0:
        return ["_test failed: " + (run.stderr.strip().splitlines() or ["?"])[-1][:160]]
    try:
        eps = float(json.loads(run.stdout.strip().splitlines()[-1])["epsilon"])
    except Exception:
        return [f"_test must return {{'epsilon': x}}, got {run.stdout.strip()[-120:]!r}"]
    return [] if 0 < eps <= 1 + 1e-9 else [f"_test epsilon {eps} not in (0, 1]"]


def check_card(p: Path) -> list[str]:
    """Errors of one theory card: front matter, topic and task_types vocabularies, sections, banned words, length."""
    errs, text = [], p.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not m:
        return ["no front matter"]
    fm, body = m.groups()
    get = lambda k: (re.search(rf"^{k}:\s*(.*)$", fm, re.M) or [None, None])[1]
    for k in ("id", "title", "topic", "task_types", "sources"):
        if get(k) is None:
            errs.append(f"missing {k}")
    if get("id") != p.stem:
        errs.append(f"id {get('id')!r} != filename")
    if get("topic") not in THEORY_TOPICS:
        errs.append(f"bad topic {get('topic')!r}")
    tt = re.fullmatch(r"\[(.*)\]", get("task_types") or "")
    bad = {t.strip() for t in tt[1].split(",")} - THEORY_TASKS if tt else {"<not a list>"}
    if bad:
        errs.append(f"bad task_types {bad}")
    if not re.findall(r"^  - \".+\"$", fm, re.M):
        errs.append("no sources")
    heads = re.findall(r"^#+ .*$", body, re.M)
    if heads != THEORY_SECTIONS:
        errs.append(f"sections {heads}")
    if "`" in text:
        errs.append("backtick/code fence")
    for w in FORBIDDEN_THEORY.findall(text):
        errs.append(f"banned word {w!r}")
    n = len(re.sub(r"^#+ .*$", "", body, flags=re.M).split())
    if not 150 <= n <= 250:
        errs.append(f"{n} words")
    return errs


def report(name: str, files: list[Path], checker, root: Path, workers: int = 1) -> int:
    """Run `checker` on every file, print the failures and a summary line; returns the number of failing files."""
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(checker, files))
    bad = 0
    for p, errs in zip(files, results):
        if errs:
            bad += 1
            print(f"FAIL {p.relative_to(root).as_posix()}: " + "; ".join(errs))
    print(f"{len(files) - bad}/{len(files)} {name} ok")
    return bad


def main() -> int:
    """Check outlines, theory cards and (unless --no-skeletons) skeletons; returns the exit code."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=CORPUS)
    ap.add_argument("--workers", type=int, default=1, help="skeleton tests run in parallel")
    ap.add_argument("--no-skeletons", action="store_true", help="skip running the skeleton tests (slow)")
    a = ap.parse_args()
    root = a.root
    import opendp.prelude as dp  # some OpenDP constructors need contrib enabled to import cleanly
    dp.enable_features("contrib")
    outlines = sorted(p for fw in FRAMEWORK_DIRS for p in (root / fw).glob("*.md"))
    bad = report("outlines", outlines, check, root)
    print("  per framework:", {fw: len(list((root / fw).glob("*.md"))) for fw in FRAMEWORK_DIRS})
    cards = sorted((root / "theory").glob("*.md"))
    bad += report("theory cards", cards, check_card, root)
    if not 15 <= len(cards) <= 30:  # expected size of the theory deck
        print(f"FAIL theory: {len(cards)} cards (want 15-30)")
        bad += 1
    if not a.no_skeletons:
        skeletons = sorted(p for fw in FRAMEWORK_DIRS for p in (root / fw).glob("*.py"))
        bad += report(f"skeletons (run in {sys.executable})", skeletons, check_skeleton, root, a.workers)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
