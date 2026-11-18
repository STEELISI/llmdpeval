"""Exit non-zero if the installed packages differ from requirements-lock.txt. Stdlib only.

Usage: python -m llmdpeval.verify_env [path/to/requirements-lock.txt]   (run_grid calls it before every grid)
"""
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

DEFAULT_LOCK = Path(__file__).resolve().parents[2] / "requirements-lock.txt"


def norm(name: str) -> str:
    """PEP 503 normalised package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def mismatches(lock: Path = DEFAULT_LOCK) -> list[str]:
    """One line per locked package whose installed version differs from the lock (or is missing)."""
    want = {}  # name -> (allowed versions, has_marker)
    for line in open(lock, encoding="utf-8"):
        m = re.match(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;]+)\s*(;.*)?$", line.strip())
        if m:
            vs, marked = want.get(norm(m[1]), (set(), False))
            want[norm(m[1])] = (vs | {m[2]}, marked or bool(m[3]))
    have = {norm(d.metadata["Name"]): d.version for d in distributions()}
    # A package with a platform marker is checked only if installed: markers are not evaluated (stdlib only).
    return [f"{n}: locked {'/'.join(sorted(vs))}, installed {have.get(n, 'MISSING')}"
            for n, (vs, marked) in sorted(want.items())
            if have.get(n) not in vs and not (marked and n not in have)]


if __name__ == "__main__":
    lock = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOCK
    bad = mismatches(lock)
    print("\n".join(bad) or f"environment matches {lock.name}")
    sys.exit(1 if bad else 0)
