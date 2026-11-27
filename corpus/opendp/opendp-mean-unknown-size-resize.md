---
id: opendp-mean-unknown-size-resize
title: Mean when the dataset size is private (DP count, resize, then mean)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.then_count
  - opendp.transformations.then_impute_constant
  - opendp.transformations.then_clamp
  - opendp.transformations.then_resize
  - opendp.transformations.make_resize
  - opendp.transformations.then_mean
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.combinators.make_composition
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-mean.rst
source_license: MIT
---

## Goal
Release a mean when n itself is private. Neighbouring relation: add/remove-one (n private); this is the labelled alternative route. When n is public (replace-one, the default for most tasks) use a sized domain and `d_in=2` instead (see `opendp-mean-sized-bounded`): no count is spent and the mean is charged (U - L) / N. Two documented routes: (a) DP count, then resize to that count, then sized mean; (b) DP sum and DP count, divided afterwards.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=float)), symmetric_distance()`.
2. Split the budget: EPS_COUNT + EPS_MEAN = EPS (or use a compositor; see `opendp-context-compositor-framework`).
3. Route (a):
   - `dp_n <- binary_search_chain(lambda s: space >> then_count() >> then_laplace(s), d_in=1, d_out=EPS_COUNT)(DATA)`
   - `mean_pre <- space >> then_impute_constant(C) >> then_clamp((L, U)) >> then_resize(size=dp_n, constant=C) >> then_mean()`
   - `dp_mean <- binary_search_chain(lambda s: mean_pre >> then_laplace(s), d_in=1, d_out=EPS_MEAN)(DATA)`
4. Route (b): `make_composition([sum_meas, count_meas])` with `sum_meas <- space >> impute >> clamp >> then_sum() >> then_laplace(?)`, then `dp_sum / dp_n` as post-processing.

Signature for reference: `make_resize(input_domain, input_metric, size, constant, MO='SymmetricDistance')`.

## Privacy accounting
Each release is its own measurement; the total is their sum by sequential composition. Add/remove-one: check it with `make_composition([count_meas, mean_meas]).map(1) <= EPS`, or let a compositor enforce it. `then_resize` turns the unsized input into a sized one, and OpenDP accounts for the padding/truncation in the map.

## Pitfalls
- `then_resize` needs `size` as an int: use the released DP count, never `len(DATA)`.
- If the task states replace-one or a public n, this route wastes EPS_COUNT; use the sized-domain item.
- The imputation constant C must be public and inside [L, U].
- Resizing to a noisy n biases the mean slightly (padding with C or dropping rows). That is expected.
- Forgetting to count the count's epsilon: the total is EPS_COUNT + EPS_MEAN, not EPS_MEAN.
