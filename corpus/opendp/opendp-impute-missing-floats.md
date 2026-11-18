---
id: opendp-impute-missing-floats
title: Handling missing float values before an aggregate (impute constant, impute uniform, drop)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.make_impute_constant
  - opendp.transformations.then_impute_constant
  - opendp.transformations.make_impute_uniform_float
  - opendp.transformations.then_impute_uniform_float
  - opendp.transformations.then_drop_null
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/index.rst
source_license: MIT
---

## Goal
Compute a DP aggregate on a float column that contains NaN. Aggregators require non-null data, so a null-handling step must come first. Neighbouring relation: replace-one (n public): sized domain, `d_in=2`.

## Steps
1. `enable_features("contrib")`.
2. Declare that the data may contain NaN: `space <- vector_domain(atom_domain(T=float, nan=True), size=N), symmetric_distance()` (N = public row count).
3. Choose one null step:
   - `then_impute_constant(C)`: replace NaN by a public constant C.
   - `then_impute_uniform_float((L, U))`: replace NaN by a uniform draw from public bounds.
   - `then_drop_null()`: remove NaN rows (the length becomes data-dependent, so the domain is no longer sized and the sum is charged 2·max(|L|, |U|, U - L) at `d_in=2`).
4. `pre <- space >> NULL_STEP >> then_clamp((L, U)) >> then_sum()`
5. `meas <- binary_search_chain(lambda s: pre >> then_laplace(s), d_in=2, d_out=EPS)`
6. `dp_sum <- meas(VALUES)`.

Signature for reference: `make_impute_uniform_float(input_domain, input_metric, bounds)`.

## Privacy accounting
Replace-one: the two imputers are row-by-row and keep the size, so the sized sum is charged U - L; `meas.map(2)` gives epsilon. A sized mean is `then_mean()` after the clamp. Alternative, add/remove-one (n private): unsized domain, `meas.map(1)` (sum charged max(|L|, |U|, U - L)); a mean then needs `then_resize` to a DP count (see `opendp-mean-unknown-size-resize`).

## Pitfalls
- Leaving `nan=False` on data that has NaN violates the domain. Do not count on an error: OpenDP does not check every domain property when you call the measurement (a bounded `nan=False` domain fed NaN or out-of-range values can release a result silently). Declare `nan=True` and impute.
- After `then_cast(...)` the nulls are `None` in an `OptionDomain`; `then_impute_constant` handles both forms, `then_impute_uniform_float` only float NaN.
- C, L and U must be public, not the data mean or min/max.
- Imputing biases the statistic toward C; dropping makes the count uncertain.
