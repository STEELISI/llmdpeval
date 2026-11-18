---
id: opendp-survival-curve-exit-histogram
title: Survival curves and life tables per group from one OpenDP exit histogram
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
Release a survival curve (Kaplan-Meier style, one point per public time bin), at-risk counts, event counts and discrete-time hazards for each group of a public group list. Also covers a life table and year-by-year hazards. One noisy count vector over (group, exit bin, event flag) cells, then post-processing. Neighbours: replace-one (bounded DP, n public), so maps are evaluated at `d_in = 2`.

## Steps
1. `enable_features("contrib")`.
2. Public inputs: time bin edges E from 0 to the public maximum follow-up (study design); the group list; event coding. Rows with time ≥ E_last are censored at the last bin.
3. Integer cell code per row (pandas/NumPy, before OpenDP): `cell <- (g*B + exit_bin)*2 + event`. `CATS <- list(range(K*B*2))`.
4. `space <- vector_domain(atom_domain(T=int)), symmetric_distance()`
   `make <- lambda s: space >> then_count_by_categories(categories=CATS, null_category=False) >> then_laplace(s)`
5. `meas <- binary_search_chain(make, d_in=2, d_out=EPS)`; `noisy <- max(meas(cell_list), 0)` reshaped to (K, B, 2).
6. Post-processing per group: `exits = events + censored`; `at_risk[0] = sum(exits)`; `at_risk[b] = at_risk[0] - cumsum(exits)[b-1]`; `hazard = clip(events / max(at_risk, FLOOR), 0, 1)`; `S = cumprod(1 - hazard)`.
7. Life table variant: denominator `at_risk - censored/2`. Hazard ratios / Cox model: not available in OpenDP; a Poisson model fitted to the released table is post-processing.

## Privacy accounting
One changed person leaves one cell and enters another: the count vector moves by 2 in L1, which the transformation's map gives from `d_in = 2`. Assert `epsilon_spent <- meas.map(2) <= EPS`. Integer counts get discrete Laplace noise. Everything after the release is free.

## Pitfalls
- Noising at-risk counts directly: a person is at risk in every bin up to exit, so that vector's sensitivity grows with the number of bins. Release exits and rebuild at-risk.
- `map(1)` under replace-one reports half the real ε.
- A separate measurement per bin or group with ε split: the cells are one release.
- Bin edges, the follow-up maximum or the category list taken from the data.
- Group sizes or at-risk start from exact data; use the noisy cell sums.
- Survey weights: a weighted count has sensitivity equal to the largest weight. Drop or cap them and say the estimand changed.
- Late bins with few at risk: report the noisy at-risk counts next to the curve.
