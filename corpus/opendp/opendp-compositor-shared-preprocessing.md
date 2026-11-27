---
id: opendp-compositor-shared-preprocessing
title: Chaining preprocessing in front of an adaptive compositor (d_in from the transformation)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.combinators.then_adaptive_composition
  - opendp.transformations.then_cast_default
  - opendp.transformations.then_impute_constant
  - opendp.transformations.then_clamp
  - opendp.transformations.then_count
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.Transformation.map
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/compositor-chaining-and-nesting.rst
source_license: MIT
---

## Goal
Write the preprocessing once (parse, impute, clamp) and put an adaptive compositor after it, so several DP queries share it. Neighbouring relation: replace-one on the raw rows (n public): sized raw domain, raw `d_in = 2`.

## Steps
1. `enable_features("contrib")`.
2. `raw <- vector_domain(atom_domain(T=str), size=N), symmetric_distance()` (N = public row count).
3. `prep <- raw >> then_cast_default(float) >> then_impute_constant(C) >> then_clamp((L, U))`
4. `D_MID <- prep.map(2)`: the distance after preprocessing for one changed raw row (2; row-wise steps keep it).
5. `comp <- prep >> then_adaptive_composition(output_measure=max_divergence(), d_in=D_MID, d_mids=[EPS_1, EPS_2])`
6. `qbl <- comp(RAW_STRINGS)`.
7. Queries start from `prep.output_space` (clamped floats), not from the raw space:
   `m1 <- binary_search_chain(lambda s: prep.output_space >> then_count() >> then_laplace(s), d_in=D_MID, d_out=EPS_1)`
   `m2 <- ... >> then_sum() >> then_laplace(s) ...` with `d_out=EPS_2`.
8. `answers <- qbl(m1), qbl(m2)`.

Signature for reference: `then_adaptive_composition(output_measure, d_in, d_mids)`.

## Privacy accounting
Replace-one: `comp.map(2)` is the sum of `d_mids`. Alternative, add/remove-one (n private): unsized raw domain, `D_MID <- prep.map(1)` and `comp.map(1)`; the sum is then charged max(|L|, |U|, U - L) instead of the sized U - L. Each query must satisfy its own `d_mid` at the compositor's `d_in` (here `D_MID`), so search each query's scale with `d_in=D_MID`.

## Pitfalls
- The compositor's `d_in` is NOT the raw `d_in` when the preprocessing is not 1-stable; always use `prep.map(raw_d_in)`.
- `then_cast_default` silently maps unparseable text to the type's default (0.0); use `then_cast` + impute if that default would bias results.
- Queries submitted to `qbl` must accept exactly `prep.output_domain`; a domain mismatch error means a query rebuilt the clamp or used a different type.
- On the sized domain a plain `then_count` is still charged 2 although n is public; prefer a 0/1 sum for counts of matching rows.
- `d_mids` are spent in order; queries beyond the list are refused ("out of queries").
