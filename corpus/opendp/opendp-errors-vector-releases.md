---
id: opendp-errors-vector-releases
title: Common errors when releasing vectors of counts or sums with OpenDP 0.16 (error message -> cause -> fix)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.metrics.l1_distance
  - opendp.transformations.make_count_by_categories
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - opendp.context.Context.compositor
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/additive-noise-mechanisms.rst
source_license: MIT
---

## Goal
Map the errors that OpenDP 0.16.0 code raises when it releases histograms, many sums, or a precomputed vector, to their cause and fix. All messages were reproduced in the pinned version.

## Steps
1. `Type("No match for concrete type i8")` (or `i16`, "unrecognized dtype: UInt8") when building `Context.compositor` -> the polars frame has Int8/Int16/UInt8 columns, which the Context does not support. Casting inside the query is too late. Fix: cast those columns to Int32, Int64 or Float64 on the LazyFrame BEFORE building the Context.
2. `Type("No match for concrete type SymmetricDistance")` from `then_laplace` -> noise put straight on a dataset space. Fix: aggregate first (`then_count_by_categories`, `then_sum`), or, for a vector you aggregated yourself, use `(vector_domain(atom_domain(T=float, nan=False)), l1_distance(T=float))` with `d_in` = your L1 bound.
3. `l1_distance() missing 1 required positional argument: 'T'` -> pass `T=float` (or `T=int`).
4. `module 'opendp.transformations' has no attribute 'then_map'` -> no such constructor. Do per-row work in pandas/numpy, then release.
5. `FailedFunction("Domain contains nan")` from `then_clamp` -> `atom_domain(T=float)` allows NaN. Fix: fill or drop NaN by a public rule, then `atom_domain(T=float, nan=False)`.
6. `input_domain must be bounded. Use make_clamp` -> a sum on an unbounded domain: `then_clamp(bounds=(L, U))` first, or `atom_domain(bounds=(L, U))`.
7. `Predicate in binary search always raises an exception` -> the chain fails for every scale; the real cause is the FIRST error in the traceback (a typo such as `then_lapace`, items 2-6). Build the chain once with `scale=1.0` to see it.
8. `make_count_by_categories() got an unexpected keyword argument 'domain'` -> its arguments are `input_domain, input_metric, categories, null_category`; or chain `space >> then_count_by_categories(categories=...)`.
9. `unsupported operand type(s) for *: 'int' and 'Metric'` -> sensitivity is not set by multiplying a metric. State the distance as `d_in` in `map`/`binary_search_param`.
10. `NameError: name 'pl' is not defined` -> polars queries need `import polars as pl`.

## Privacy accounting
Fixes do not change the accounting: read epsilon from `meas.map(d_in)` or the Context accountant.

## Pitfalls
- Catching these errors and printing a message releases nothing.
- When you aggregate outside OpenDP, `d_in` is your own proof: write it next to the code (e.g. one row adds at most C, replace-one: 2C).
