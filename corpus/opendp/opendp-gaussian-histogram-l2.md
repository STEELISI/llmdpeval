---
id: opendp-gaussian-histogram-l2
title: Histogram (counts per public category) with Gaussian noise (L2 sensitivity)
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: histogram
dp_notion: approx
needs_bounds: false
api_calls:
  - opendp.transformations.then_count_by_categories
  - opendp.metrics.l2_distance
  - opendp.measurements.then_gaussian
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/integration/test_histogram.py
source_license: MIT
---

## Goal
Release noisy counts for a fixed public list of categories with Gaussian noise, under (epsilon, delta)-DP. Gaussian noise is calibrated to L2 sensitivity, which helps when one person can touch several bins. Neighbours: replace-one (bounded DP, n public), which is symmetric distance 2.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`
3. `hist <- space >> then_count_by_categories(categories=CATS, MO=L2Distance[int])`. The `MO` argument makes the histogram report its sensitivity in L2 (default is L1, which only Laplace accepts).
4. `make <- lambda s: make_fix_delta(make_zCDP_to_approxDP(hist >> then_gaussian(s)), delta=DELTA)`
5. `meas <- binary_search_chain(make, d_in=2, d_out=(EPS, DELTA))` (replace-one = one removal + one addition)
6. `counts <- meas(VALUES_AS_STR_LIST)`: one count per category, plus one final count for values outside CATS.

Signature for reference: `make_count_by_categories(input_domain, input_metric, categories, null_category=True, MO='L1Distance<int>', TOA='int')`.

## Privacy accounting
`then_gaussian` measures rho (zCDP). Convert with `make_zCDP_to_approxDP`, fix delta with `make_fix_delta`, and read `(epsilon, delta)` from `meas.map(2)`: replace-one (bounded DP), so the map is evaluated at symmetric distance 2. Use `D_IN = 1` only when the task says add/remove-one. For a pure rho budget, stop after `then_gaussian` and search with `d_out=RHO`.

## Pitfalls
- `d_in=1` / `map(1)` under replace-one understates the guarantee.
- Without `MO=L2Distance[int]`, `then_gaussian` fails to chain (metric mismatch).
- The category list must be public (codebook), never `unique()` of the data.
- The extra last bin exists because `null_category=True`; set it to False only if values outside CATS are impossible.
- Gaussian cannot give delta = 0; use the Laplace histogram outline for pure DP.
