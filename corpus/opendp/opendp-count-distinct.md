---
id: opendp-count-distinct
title: Number of distinct values in a column (make_count_distinct + Laplace)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.transformations.make_count_distinct
  - opendp.transformations.then_count_distinct
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/test_trans.py
source_license: MIT
---

## Goal
Release how many different values a column takes (e.g. distinct customers, distinct codes). Neighbouring relation: replace-one (n public): changing one row moves the distinct count by at most 1, so the sensitivity is 1.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()` (use the column's own type, e.g. `T=int`).
3. `meas <- binary_search_chain(lambda s: space >> then_count_distinct() >> then_laplace(s), d_in=1, d_out=EPS)`. Here `d_in=1` is deliberate: OpenDP's d_in=1 map of this transformation is 1, which equals the replace-one sensitivity; `d_in=2` would charge 2 and double the noise.
4. `dp_value <- meas(VALUES)`: an integer (discrete Laplace noise, since the count is an integer).

Signature for reference: `make_count_distinct(input_domain, input_metric, TO='int')`.

## Privacy accounting
Replace-one: one changed row moves the distinct count by -1, 0 or +1 (sensitivity 1), and `meas.map(1)` = 1/scale is exactly that charge; report `meas.map(1)`. `meas.map(2)` = 2/scale is a valid but conservative replace-one bound. Add/remove-one (n private) gives the same number: sensitivity 1, `map(1)`.

## Pitfalls
- If one person can contribute k rows, use `d_in=k`: the distinct count can then change by up to k.
- Under replace-one neighbours the distinct count can still change by 1, unlike the plain row count (which is public). This `d_in=1` shortcut holds only for this scalar: histograms and group-bys need `d_in=2`.
- The noisy result can be negative or exceed the number of possible values; clip only as post-processing.
- In the polars Context the same statistic is `col(X).dp.n_unique(signed=True)` (see `opendp-polars-null-and-distinct-counts`).
