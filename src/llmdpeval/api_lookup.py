"""Exact API lookup for repair: when a crash names a framework function, class or module that does not exist (or is
called wrongly), return the closest real entries of the pinned API index (corpus/api_index), with signatures.

  python -m llmdpeval.api_lookup            # self-check, no LLM
  python -m llmdpeval.api_lookup FILE.txt   # lookup on a saved stderr

Handled errors (last matching line of the traceback):
- AttributeError: module 'opendp.prelude' has no attribute 'make_base_laplace'
- AttributeError: type object 'X' / 'X' object has no attribute 'y'   (X a framework class: its methods)
- ImportError: cannot import name 'y' from 'diffprivlib.tools'
- ModuleNotFoundError: No module named 'opendp.polars'
- NameError: name 'PrivacyLeakWarning' is not defined                 (exact framework names only)
- TypeError: f() got an unexpected keyword argument / missing ... required / takes ... positional  (exact names)
Bare names (NameError, TypeError, class attributes) are looked up only in the frameworks the code imports, so a
sklearn LinearRegression error never gets diffprivlib's LinearRegression.
"""
import difflib
import re
import sys
from functools import lru_cache

from llmdpeval.helper import REPO_ROOT

INDEX = {"opendp": ("OpenDP", "0.16.0"), "diffprivlib": ("diffprivlib", "0.6.6")}
MAX_HITS = 3
# opendp.prelude (imported as dp) re-exports these modules flat, and t/m/c as submodules
DP_FLAT = ("opendp.mod", "opendp.context", "opendp.domains", "opendp.metrics", "opendp.measures", "opendp.accuracy")
DP_SUB = {"opendp.transformations": "dp.t", "opendp.measurements": "dp.m", "opendp.combinators": "dp.c",
          "opendp.extras.polars": "dp.polars"}

MODULE_ATTR = re.compile(r"AttributeError: module '([\w.]+)' has no attribute '(\w+)'")
CLASS_ATTR = re.compile(r"AttributeError: (?:type object '(\w+)'|'(\w+)' object) has no attribute '(\w+)'")
IMPORT = re.compile(r"ImportError: cannot import name '(\w+)' from '([\w.]+)'")
NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
NAME = re.compile(r"NameError: name '(\w+)' is not defined")
CALL = re.compile(r"TypeError: (?:[\w.]*\.)?(\w+)\(\) (?:got an unexpected keyword|missing \d+ required|takes )")
CALL_INIT = re.compile(r"TypeError: (\w+)\.__init__\(\) ")


def short_signature(line: str) -> str:
    """'name(a: T = 1, *, b: U) -> R    [defined in m]' -> 'name(a=1, *, b)': annotations and the return type
    dropped (OpenDP's RuntimeType unions run to 300 characters), defaults kept."""
    line = line.split("    [defined in", 1)[0].strip().removeprefix("class ").removeprefix("def ")
    if "(" not in line:
        return line
    name, rest = line.split("(", 1)
    params, depth, cur = [], 0, ""
    for ch in rest:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            params.append(cur)
            cur = ""
        else:
            cur += ch
    params.append(cur)
    out = []
    for p in (p.strip() for p in params if p.strip()):
        head, _, default = p.partition("=")
        head = head.split(":", 1)[0].strip()
        if head == "self":
            continue
        out.append(f"{head}={default.strip()}" if default else head)
    return f"{name}({', '.join(out)})"


@lru_cache(maxsize=None)
def load(framework: str) -> tuple[dict, dict]:
    """(entries, methods): entries {full dotted name: signature line}; methods {class full name: [method lines]}."""
    name, version = INDEX[framework]
    path = REPO_ROOT / "corpus" / "api_index" / f"API_INDEX_{framework}_{version}.txt"
    entries, methods, last = {}, {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        if raw.startswith(" "):  # a method of the class above: "    .fit(...)" or "    opendp.x.C.fit(...)"
            if last:
                meth = raw.strip().lstrip(".").removeprefix(last + ".")
                methods.setdefault(last, []).append(short_signature(meth))
            continue
        sig = short_signature(raw)
        full = sig.split("(", 1)[0]
        entries[full] = sig
        last = full
    return entries, methods


def display(full: str, sig: str) -> str:
    """OpenDP names as code writes them (dp.m.make_laplace); others unchanged."""
    module, short = full.rsplit(".", 1)
    prefix = "dp" if module in DP_FLAT else DP_SUB.get(module)
    return f"{prefix}.{short}{sig[len(full):]}" if prefix else sig


def namespace(framework: str, module: str) -> dict:
    """{full name: signature} of the names reachable as module.<name>; opendp.prelude reaches all of DP_FLAT and,
    one level down, dp.t / dp.m / dp.c / dp.polars (dp.make_laplace -> dp.m.make_laplace)."""
    entries, _ = load(framework)
    if module == "opendp.prelude":
        mods = (*DP_FLAT, *DP_SUB)
    else:
        mods = (module,)
    return {f: s for f, s in entries.items() if f.rsplit(".", 1)[0] in mods}


def closest(name: str, pool: dict) -> list[str]:
    """Full names in pool whose last part is closest to name (case-insensitive; containment counts as close)."""
    def score(full):
        short, n = full.rsplit(".", 1)[-1].lower(), name.lower()
        tokens = [t for t in n.split("_") if len(t) >= 4]  # laplace_mechanism -> make_laplace
        bonus = 0.3 if n in short or short in n else 0.2 if any(t in short for t in tokens) else 0
        return difflib.SequenceMatcher(None, n, short).ratio() + bonus
    out, seen = [], set()
    for f in sorted(pool, key=score, reverse=True):
        if score(f) >= 0.6 and f.rsplit(".", 1)[-1] not in seen:  # dp.t.x and dp.polars.x are one function
            seen.add(f.rsplit(".", 1)[-1])
            out.append(f)
    return out[:MAX_HITS]


def lines_for(framework: str, fulls: list[str]) -> list[str]:
    entries, _ = load(framework)
    return [display(f, entries[f]) for f in fulls]


def imported(code: str) -> list[str]:
    return [fw for fw in INDEX if re.search(rf"^\s*(?:import|from)\s+{fw}\b", code, re.M)]


def exact(name: str, frameworks: list[str]) -> list[tuple[str, str]]:
    return [(fw, f) for fw in frameworks for f in load(fw)[0] if f.rsplit(".", 1)[-1] == name][:MAX_HITS]


def lookup(stderr: str, code: str = "") -> str | None:
    """A short block with the exact pinned API for the failing call, or None if the error is not an API error."""
    hits: list[str] = []
    frameworks = imported(code) or [fw for fw in INDEX if fw in stderr]
    fw = None
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if (m := MODULE_ATTR.search(line) or IMPORT.search(line)):
            name, module = (m.group(2), m.group(1)) if m.re is MODULE_ATTR else (m.group(1), m.group(2))
            fw = module.split(".")[0]
            if fw not in INDEX:
                return None
            found = closest(name, namespace(fw, module)) or closest(name, load(fw)[0])
            hits = [f"{module} has no '{name}'. Closest real names:"] + lines_for(fw, found) if found else []
            break
        if (m := NO_MODULE.search(line)):
            module = m.group(1)
            fw = module.split(".")[0]
            if fw not in INDEX:
                return None
            modules = {f.rsplit(".", 1)[0]: None for f in load(fw)[0]}
            found = closest(module.rsplit(".", 1)[-1], modules)
            hits = [f"No module {module}. Real modules:"] + found if found else []
            break
        if (m := CLASS_ATTR.search(line)):
            cls, attr = m.group(1) or m.group(2), m.group(3)
            for fw, full in exact(cls, frameworks):
                meths = load(fw)[1].get(full, [])
                near = closest(attr, {mm.split("(")[0]: None for mm in meths})
                listed = [mm for mm in meths if mm.split("(")[0] in near] or meths
                if listed:  # a class with no listed methods (a warning class) has nothing to suggest
                    hits = [f"{display(full, load(fw)[0][full])} has no attribute '{attr}'. Its public methods: "
                            + "; ".join(listed)]
                break
            break
        if (m := NAME.search(line) or CALL_INIT.search(line) or CALL.search(line)):
            found = exact(m.group(1), frameworks)
            fw = found[0][0] if found else None
            hits = [f"Real signature{'s' if len(found) > 1 else ''}:"] + [
                display(f, load(fw)[0][f]) + (f"   # from {f.rsplit('.', 1)[0]} import {m.group(1)}"
                                              if m.re is NAME else "") for fw, f in found] if found else []
            break
    if not hits:
        return None
    name, version = INDEX[fw]
    return f"Exact API of {name} {version} for the failing call (from the pinned API index; use only these names):\n" + "\n".join(hits)


def selftest() -> None:
    assert short_signature("opendp.metrics.l1_distance(T: Union[ForwardRef('R'), str], x: int = 1) -> M") == \
        "opendp.metrics.l1_distance(T, x=1)"
    b = lookup("AttributeError: module 'opendp.prelude' has no attribute 'make_base_laplace'", "import opendp.prelude as dp")
    assert "OpenDP 0.16.0" in b and "dp.m.make_laplace(input_domain, input_metric, scale" in b, b
    b = lookup("ImportError: cannot import name 'laplace' from 'diffprivlib.tools' (/x/__init__.py)")
    assert "diffprivlib.mechanisms.Laplace(*, epsilon" in b, b
    b = lookup("ImportError: cannot import name 'clip_to_bounds' from 'diffprivlib.mechanisms.transforms' (/x)")
    assert "diffprivlib.validation.clip_to_bounds(array, bounds)" in b, b
    b = lookup("ModuleNotFoundError: No module named 'opendp.polars'")
    assert "opendp.extras.polars" in b, b
    b = lookup("TypeError: l1_distance() missing 1 required positional argument: 'T'", "import opendp.prelude as dp")
    assert "dp.l1_distance(T)" in b, b
    b = lookup("AttributeError: 'BudgetAccountant' object has no attribute 'add'", "from diffprivlib import BudgetAccountant")
    assert "spend(epsilon, delta)" in b, b
    b = lookup("NameError: name 'PrivacyLeakWarning' is not defined", "import diffprivlib")
    assert "from diffprivlib.utils import PrivacyLeakWarning" in b, b
    # not API errors, or not a framework name: no block
    assert lookup("NameError: name 'pd' is not defined", "import diffprivlib") is None
    assert lookup("TypeError: gamma() got an unexpected keyword argument 'a'", "import diffprivlib") is None
    assert lookup("AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'", "import diffprivlib") is None
    assert lookup("TypeError: LinearRegression.__init__() got an unexpected keyword argument 'x'",
                  "from sklearn.linear_model import LinearRegression") is None
    assert lookup("KeyError: 'female'") is None
    assert lookup("AttributeError: type object 'PrivacyLeakWarning' has no attribute 'simplefilter'", "import diffprivlib") is None
    assert "dp.m.make_laplace(" in lookup("AttributeError: module 'opendp.prelude' has no attribute 'laplace_mechanism'")
    print("api_lookup self-check ok")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(lookup(open(sys.argv[1], encoding="utf-8").read()))
    else:
        selftest()
