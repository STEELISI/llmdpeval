---
id: opendp-quantile-exponential
title: Quantiles with the exponential mechanism over public candidates (make_private_quantile)
framework: opendp
framework_version: "0.16.0"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.measures.max_divergence
  - opendp.transformations.make_drop_null
  - opendp.measurements.make_private_quantile
  - opendp.measurements.then_private_quantile
  - opendp.combinators.make_composition
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-quantile.ipynb
source_license: MIT
---

## Goal
Release one or more quantiles (median, quartiles) of a numeric column. Additive noise works badly here because one record can move a quantile a lot; the exponential mechanism picks the best of a public candidate grid instead. Neighbours: replace-one (bounded DP, n public), evaluated at symmetric distance `d_in = 2`.

## Steps
1. `enable_features("contrib")`.
2. `CANDS`: an evenly spaced public grid between public bounds L and U (finer grid = more precision, more time). Same numeric type as the data.
3. `space <- vector_domain(atom_domain(T=float, nan=False)), symmetric_distance()`, or start from a nullable space and chain `make_drop_null` first.
4. One quantile:
   `make <- lambda scale: make_private_quantile(*space, max_divergence(), candidates=CANDS, alpha=A, scale=scale)`
5. Several quantiles under one budget, as the docs do:
   `make <- lambda scale: make_composition([make_private_quantile(*space, max_divergence(), CANDS, A_i, scale) for A_i in ALPHAS])`
6. `meas <- binary_search_chain(make, d_in=2, d_out=EPS)`; `dp_q <- meas(VALUES_AS_LIST)` (a list when composed).

Signature for reference: `make_private_quantile(input_domain, input_metric, output_measure, candidates, alpha, scale)`.

## Privacy accounting
Replace-one is one removal plus one addition, so the symmetric distance between neighbours is 2. The search finds one shared scale so that the whole composition maps to at most EPS at `d_in = 2`; with m quantiles each gets roughly EPS / m. Assert `epsilon_spent <- meas.map(2) <= EPS`. The quantile measurement accepts only symmetric (or insert-delete) distance, not `change_one_distance`.

## Pitfalls
- `d_in=1` / `map(1)` is add/remove-one (n private); under replace-one it reports half the real ε.
- Candidates from the data (sorted unique values) leak the data. Use a public grid.
- Each quantile is a separate release; m quantiles at EPS each costs m * EPS.
- The output is always one of the candidates, so the grid sets the resolution.
- Released quartiles may come out out of order; sort them afterwards if the task needs monotone output.
- For many quantiles, a DP histogram (or b-ary tree) plus `make_quantiles_from_counts` can do better (see `opendp-quantiles-from-histogram`).
