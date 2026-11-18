"""List every public callable of OpenDP (pinned venv) with its signature.

Usage (from the repo root, in the pinned venv): python corpus/api_index/make_api_index_opendp.py > corpus/api_index/API_INDEX_opendp_0.16.0.txt
"""
import importlib
import inspect
import re

import opendp.prelude as dp

dp.enable_features("contrib")
MODULES = ["opendp.prelude", "opendp.mod", "opendp.context", "opendp.measurements", "opendp.transformations",
           "opendp.combinators", "opendp.domains", "opendp.metrics", "opendp.measures", "opendp.accuracy",
           "opendp.extras.numpy", "opendp.extras.polars", "opendp.extras.sklearn",
           "opendp.extras.sklearn.decomposition", "opendp.extras.sklearn.linear_model", "opendp.extras.mbi",
           "opendp.extras.examples"]


def sig(obj):
    try:
        return re.sub(r" at 0x[0-9A-Fa-f]+", "", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "(...)"


print(f"# OpenDP {dp.__version__} public callables: module.name(signature). Classes list public methods.")
print("# Import as `import opendp.prelude as dp`; dp.t = transformations, dp.m = measurements, dp.c = combinators.")
for modname in MODULES:
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        print(f"\n## {modname}: not importable ({e})")
        continue
    print(f"\n## {modname}")
    for name in sorted(getattr(mod, "__all__", None) or dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if inspect.ismodule(obj) or not callable(obj):
            continue
        print(f"{modname}.{name}{sig(obj)}")
        if inspect.isclass(obj) and obj.__module__.startswith("opendp"):
            for m, f in sorted(vars(obj).items()):
                if not m.startswith("_") and callable(f):
                    print(f"    {modname}.{name}.{m}{sig(f)}")
