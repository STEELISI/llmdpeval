---
id: opendp-api-change-chaining-and-search
title: OpenDP 0.16.0 chaining with >> and binary search arguments (common TypeErrors)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.mod.binary_search
  - opendp.mod.binary_search_param
  - opendp.mod.binary_search_chain
  - opendp.domains.vector_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count
  - opendp.transformations.make_count
  - opendp.measurements.then_laplace
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/trouble-shooting.rst
source_license: MIT
---

## Goal
Derived from observed model errors. Fix TypeErrors raised while building a pipeline: wrong chaining operands and wrong binary-search keywords.

## Steps
1. Chain from a (domain, metric) TUPLE, then `then_*` constructors:
   `space <- (dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance())`; `meas <- space >> dp.t.then_count() >> dp.m.then_laplace(scale=S)`.
   - Error "unsupported operand type(s) for >>: 'VectorDomain' and 'Metric'": you wrote `domain >> metric`; make them a tuple.
   - Error "Cannot chain _PartialConstructor with VectorDomain": a `then_*` came first; start the chain with the space tuple.
   - `make_*` constructors take `(input_domain, input_metric, ...)` explicitly; `then_*` take only the remaining arguments.
2. Binary search signatures:
   - `binary_search_param(make_chain, d_in, d_out, bounds=(None, None), T=None)` returns the parameter; `binary_search_chain(...)` returns the built measurement. The first argument is positional (no `make=` keyword).
   - `binary_search(predicate, bounds=None, T=None, return_sign=False)` takes a predicate only; `binary_search(..., d_in=...)` fails. Use it for searches that are not one d_in/d_out check (e.g. thresholds).
3. "inferred type is X, expected Y": an argument's Python type does not match the domain. Match the types (`T=float` with float data and float bounds). Messages and causes (0.16.0):
   - "inferred type is f64, expected i32": float bounds such as `(0, 10.0)` on an int domain.
   - "inferred type is i128, expected i32" (or "expected u32" for a `d_in`): NumPy integer scalars (`np.int64`) as bounds, categories or `d_in`. Convert with `int(x)` or `.tolist()`.
   - "Expected dtype int32, got int64": a NumPy int64 array passed to a measurement on `atom_domain(T=int)`. Pass `ARR.tolist()` or `ARR.astype(np.int32)`.
   - "inferred type is f32, expected f64": float32 values; cast to float.
   - "inferred type is i32, expected String" / "Vec<String>, expected String": integer or nested-list categories on a string domain; categories must be a flat list of the data's type.
   - "inferred type is String, expected f64": a whole pandas DataFrame passed to the measurement; pass one column, `DF[COL].to_numpy(dtype=float)`.
4. Building and searching:
   - `'float' object has no attribute 'map'`: `binary_search_param` returns the scale (a float); build the measurement with it, or use `binary_search_chain` to get the measurement.
   - `'_PartialConstructor' object has no attribute 'check'` (or `map`): a `then_*` constructor was used alone; it becomes a measurement only after `space >> ...`.
   - "the decision boundary of the predicate is outside the bounds": the `bounds=` given to the search do not contain the answer; widen them or drop them.
   - "unable to infer bounds": `binary_search` got a predicate that is never true (or never false); check the predicate and pass `T=float` and `bounds`.

## Privacy accounting
After the chain builds, `meas.map(d_in)` gives the privacy loss; `binary_search_chain(..., d_out=EPS)` guarantees `map(d_in) <= EPS`.

## Pitfalls
- A missing feature flag raises "Attempted to use function that requires "contrib"": call `dp.enable_features("contrib")` once at the top.
- Tuples of (epsilon, delta) cannot be searched with `binary_search_chain` when both parts move ("unknown ordering"); search each part with `binary_search`.
- `Budget` objects and `.accuracy` on queries come from other libraries / older OpenDP; polars queries use `.summarize(alpha=...)`.
- Since 0.16 search bounds may be one-sided, e.g. `bounds=(0.0, None)`.
