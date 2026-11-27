---
id: opendp-api-change-domains-measures
title: OpenDP 0.16.0 domain, metric and measure constructors (AllDomain, max_divergence, bounds=)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.metrics.l1_distance
  - opendp.metrics.absolute_distance
  - opendp.transformations.then_impute_constant
  - opendp.transformations.then_resize
  - opendp.domains.atom_domain
  - opendp.domains.vector_domain
  - opendp.measures.max_divergence
  - opendp.measures.zero_concentrated_divergence
  - opendp.measures.approximate
  - opendp.metrics.symmetric_distance
  - opendp.measurements.make_laplace
  - opendp.transformations.then_clamp
  - opendp.transformations.then_mean
source_url: https://github.com/opendp/opendp/blob/v0.16.0/CHANGELOG.md
source_license: MIT
---

## Goal
Derived from observed model errors. Replace domain/measure spellings from older OpenDP releases or other libraries with the 0.16.0 constructors.

## Steps
1. Domains are built by lowercase functions:
   - `AllDomain`, `AtomicDomain`, `AtomDomain(...)` as constructors -> `dp.atom_domain(T=float)`; `AtomDomain` is only the returned class.
   - `atom_domain(lower_bound=L, upper_bound=U)` -> `dp.atom_domain(bounds=(L, U), nan=False)`. Signature: `atom_domain(bounds=None, nan=None, T=None)`.
   - `VectorDomain<AllDomain<float>>` strings -> `dp.vector_domain(dp.atom_domain(T=float))`; add `size=N` for a public size.
2. Measures are functions on the prelude and take no type argument since 0.12:
   - `max_divergence` used without the prelude (NameError), `max_divergence(T=float)` -> `dp.max_divergence()`.
   - `from opendp.core import max_divergence` -> `dp.max_divergence()`.
   - zCDP: `dp.zero_concentrated_divergence()`; (epsilon, delta): `dp.approximate(dp.max_divergence())`.
   - `SMDCurve` -> `PrivacyProfile` (0.12); read epsilon with `profile.epsilon(delta=...)`.
3. Aggregators from older code: `make_bounded_mean`, `make_sized_bounded_mean`, `make_bounded_selector`, `dp.sum(...)` -> put bounds/size on the data with `then_clamp((L, U))` / `then_resize(size=N, constant=C)` and call `then_mean()` / `then_sum()`; in polars use `pl.col(X).dp.sum((L, U))`.
4. `make_laplace` is imported from `dp.m` (measurements), never from `opendp.measures`.
5. Metric constructors need the element type: `dp.absolute_distance(T=float)`, `dp.linf_distance(T=int)`, `dp.l1_distance(T=float)`, `dp.l2_distance(T=float)`. Without it: "missing 1 required positional argument: 'T'".
6. Domain errors in 0.16.0 and their fix:
   - `MetricSpace("AbsoluteDistance requires non-nan elements")` from Laplace on a scalar, and `MakeTransformation("input_domain may not contain NaN elements")` from Laplace on a float vector (same cause as "Domain contains nan" from `then_clamp`): `atom_domain(T=float)` allows NaN. Use `atom_domain(T=float, nan=False)` after dropping or filling NaN by a public rule, or chain `then_impute_constant(C)` before `then_clamp`.
   - "dataset size must be known. Either specify size in the input domain or use make_resize": a mean or variance without `size=N` (n public) or `then_resize(size=N, constant=C)`.

## Privacy accounting
Not applicable to renames; once the pipeline builds, read epsilon from `meas.map(d_in)`.

## Pitfalls
- `dp.atom_domain(bounds=(L, U))` for floats defaults to allowing NaN; downstream sized aggregates may then reject the domain. Pass `nan=False`.
- Bounds must match each other and the data type: on float data `then_clamp((0, 10.0))` fails ("inferred type is f64, expected i32"; models also hit "bounds must share the same type"). Write `(0.0, 10.0)`.
- Mixed measures cannot be composed without casting (see `opendp-measure-casting`).
