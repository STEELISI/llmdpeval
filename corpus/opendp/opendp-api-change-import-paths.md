---
id: opendp-api-change-import-paths
title: OpenDP 0.16.0 import paths - names live on opendp.prelude, not on the top-level package
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.prelude.enable_features
  - opendp.prelude.vector_domain
  - opendp.prelude.atom_domain
  - opendp.measurements.make_noisy_top_k
  - opendp.mod.binary_search_param
  - opendp.extras.polars.Margin
  - opendp.extras.sklearn.decomposition.PCA
  - opendp.extras.sklearn.linear_model.LinearRegression
  - opendp.measurements.make_laplace
  - opendp.measurements.then_laplace
  - opendp.transformations.make_count_by
  - opendp.transformations.make_count_by_categories
  - opendp.context.unit_of
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/src/opendp/prelude.py
source_license: MIT
---

## Goal
Derived from observed model errors. Fix ImportError / AttributeError / ModuleNotFoundError caused by importing OpenDP names from places where they do not exist in 0.16.0. Import and name errors are among the most common OpenDP crashes in generated code.

## Steps
1. Always start with the prelude: `import opendp.prelude as dp`, then `dp.enable_features("contrib")`. The bare package `opendp` exposes almost nothing.
2. Observed wrong form -> 0.16.0 form:
   - `opendp.enable_features(...)`, `from opendp import enable_features`, bare `enable_features(...)` -> `dp.enable_features("contrib")`.
   - `opendp.vector_domain`, `opendp.atom_domain`, `from opendp import vector_domain` -> `dp.vector_domain`, `dp.atom_domain`.
   - `from opendp import make_noisy_top_k`, `from opendp.combinators import make_noisy_top_k`, `from opendp.mod import make_noisy_top_k` -> `dp.m.make_noisy_top_k` (it is a measurement).
   - `opendp.contrib` (module) -> no such module; "contrib" is a feature flag string for `dp.enable_features`.
   - `opendp.polars` -> `dp.polars` (= `opendp.extras.polars`), e.g. `dp.polars.Margin`.
   - `opendp.sklearn` -> `dp.sklearn.decomposition.PCA`, `dp.sklearn.linear_model.LinearRegression` (= `opendp.extras.sklearn...`).
   - `opendp.binary_search` (module) -> functions `dp.binary_search`, `dp.binary_search_param`, `dp.binary_search_chain`.
   - `opendp.prelude.metrics` (submodule import) -> `prelude` is a module, not a package; use `dp.symmetric_distance()` etc.
   - `opendp.whitenoise`, `import dp` -> do not exist (whitenoise is the pre-2021 project name).
3. Short module aliases on the prelude: `dp.t` = transformations, `dp.m` = measurements, `dp.c` = combinators.
4. Names that do not exist in 0.16.0 (each raises AttributeError or ModuleNotFoundError), with what to use:
   - `dp.laplace`, `dp.laplace_mechanism`, `dp.make_base_laplace` -> `dp.m.make_laplace` / `dp.m.then_laplace` (and `then_lapace` is a typo of the latter).
   - `dp.make_select`, `dp.col` -> polars: `pl.col(X)` inside a Context query (`import polars as pl`).
   - `dp.unit_data` -> `dp.unit_of(contributions=K)` or `dp.unit_of(changes=K)`.
   - `dp.l_sensitivity` -> no such helper: the sensitivity is the `d_in` you pass to `map` / `binary_search_param`, or what `transformation.map(d_in)` returns.
   - `dp.t.then_map`, `dp.t.then_user_transform` -> do per-row work in pandas/polars before the chain.
   - `dp.t.make_count_by_key` -> `dp.t.make_count_by(input_domain, input_metric, TV=int)` (keys are the values seen, so it needs a threshold mechanism); it has no `key_type=` argument. For a public key list use `make_count_by_categories`.
   - `dp.polars.from_pandas` -> `pl.from_pandas(DF).lazy()` (a polars function), then `Context.compositor(data=LF, ...)`.
   - Corpus item ids such as `opendp-polars-context-basic` are document names, not modules: `opendp.extras.polars_context_basic` does not exist.

## Privacy accounting
Not applicable: renames do not change privacy. After fixing imports, still read epsilon from `meas.map(d_in)`.

## Pitfalls
- `opendp.core` exists but holds low-level internals; user code should not import from it.
- `make_laplace` lives in `dp.m` (measurements), not `opendp.measures`; measures are the privacy definitions (`dp.max_divergence()`).
- Check `dp.__version__`; plain `opendp` has no `__version__`.
