---
id: opendp-legacy-api-names
title: Old OpenDP names found in public code and their 0.16.0 replacements
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.measurements.make_laplace
  - opendp.measurements.then_laplace
  - opendp.measurements.make_gaussian
  - opendp.transformations.make_sum
  - opendp.transformations.then_sum
  - opendp.transformations.make_mean
  - opendp.transformations.then_mean
  - opendp.transformations.then_clamp
  - opendp.transformations.then_resize
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.extras.polars.Margin
  - opendp.mod.enable_features
  - opendp.mod.binary_search_param
source_url: https://github.com/camelop/dp_lab/blob/dec9b95380e34ea7c486e91bbaf85d66d5e75fbf/src/dplab/library_workload/opendp.py
source_license: Apache-2.0
---

## Goal
Translate a pattern copied from older OpenDP code (tutorials, benchmarks, papers from 2021-2024) into names that exist in 0.16.0. Most public GitHub code that calls OpenDP was written for an older release, so a model that learned from it will produce names that fail at import. Neighbouring relation for the translated code: replace-one (n public) by default, so maps are read at `d_in=2`.

## Steps
1. Measurements:
   - `make_base_laplace(...)`, `make_base_discrete_laplace(...)` → `make_laplace(input_domain, input_metric, scale=?)` or `then_laplace(scale=?)`. One constructor now covers integer (discrete) and float (continuous) inputs, picked from the domain's type.
   - `make_base_gaussian(...)` → `make_gaussian(input_domain, input_metric, scale=?)`.
   - Old string type arguments such as `D="VectorDomain<AllDomain<float>>"` → explicit domains: `vector_domain(atom_domain(T=float))`.
2. Aggregates:
   - `make_bounded_sum(bounds)`, `make_sized_bounded_sum(size, bounds)` → declare bounds and size on the domain, then `then_sum()`: e.g. `vector_domain(atom_domain(T=float, nan=False), size=N) >> then_clamp((L, U)) >> then_sum()`. Declaring `atom_domain(bounds=(L, U))` instead is safe only if the data are already clamped; OpenDP does not check it at call time.
   - `make_sized_bounded_mean(size, bounds)` → replace-one (n public): `vector_domain(atom_domain(T=float, nan=False), size=N) >> then_clamp((L, U)) >> then_mean()`, `d_in=2`, charged (U - L) / N. Alternative, add/remove-one (n private): unsized domain, `then_clamp((L, U)) >> then_resize(size=DP_N, constant=C) >> then_mean()` with a DP count DP_N, `d_in=1`.
3. Feature flags: `enable_features("floating-point")` is now a deprecated alias of `"idealized-numerics"`; most pipelines need only `"contrib"`.
4. Polars margins: `Margin(max_partition_length=?, public_info="keys")` → `Margin(by=[...], max_length=?, invariant="keys")`. The old keywords raise TypeError.
5. After translating, rebuild the scale with `binary_search_param(make_chain, d_in=?, d_out=EPS)` and print `meas.map(d_in)`; do not reuse a hand-computed scale from the old code.

## Privacy accounting
Renaming does not change the math, but domains now carry bounds and size, so the sensitivity the map uses can differ from what the old code assumed. Re-derive epsilon from `meas.map(d_in)` after every translation: `map(2)` under replace-one, `map(1)` under add/remove-one. Old code often used `d_in=1`; check which relation the task states.

## Pitfalls
- Seen in the wild (camelop/dp_lab@dec9b95): `make_bounded_sum`, `make_sized_bounded_mean`, `make_base_laplace`, `make_base_discrete_laplace`; none exist in 0.16.0. The same file passes `size=len(data)`: fine when n is public (replace-one), a leak when n is private (add/remove-one), where the size must come from a DP count.
- Seen in the wild (eth-sri/Delta-Siege@358377f): `make_base_laplace(scale=?, D="VectorDomain<AllDomain<float>>")`, written for OpenDP 0.2-era docs.
- Seen in the wild (MacroAcon/Tavren@e32e4c5): `make_base_laplace().eval(scale=?, location=0.0)`; this is not an OpenDP call form (measurements have no `.eval`). Noise must come from calling a measurement on data.
- Seen in the wild (InseeFrLab/diff-privacy-OpenDP@992407b, different7/DSL-rewriting-to-inject-DP-mechanisms@a6a6845): `Margin(max_partition_length=..., public_info=...)`.
- Version pins matter: check `dp.__version__` (after `import opendp.prelude as dp`; plain `opendp` has no `__version__`) before trusting a snippet.
