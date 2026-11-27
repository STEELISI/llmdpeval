"""Turn the authors' poetry.lock into exact pip pins (name==version), main + dev, versions unchanged.
Used because poetry 1.1.8 (the README's version) no longer imports against current virtualenv/packaging.
poetry 1.1 locks keep platform markers on the parent's dependency entry, so a package is skipped only when
every reference to it carries a marker that is false for Linux / CPython 3.8 (e.g. pywin32, pywinpty)."""
import sys, tomllib
from pip._vendor.packaging.markers import Marker
ENV = {"python_version": "3.8", "python_full_version": "3.8.0"}
lock = tomllib.load(open(sys.argv[1], "rb"))
refs = {}
for p in lock["package"]:
    for dep, spec in p.get("dependencies", {}).items():
        for s in spec if isinstance(spec, list) else [spec]:
            m = s.get("markers") if isinstance(s, dict) else None
            refs.setdefault(dep.lower(), []).append(Marker(m).evaluate(ENV) if m else True)
for p in sorted(lock["package"], key=lambda p: p["name"].lower()):
    r = refs.get(p["name"].lower())
    if r and not any(r):
        print(f'# skipped (platform marker): {p["name"]}=={p["version"]}'); continue
    print(f'{p["name"]}=={p["version"]}')
