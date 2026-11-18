---
id: opendp-topk-rate-two-histograms
title: Top-k groups by PROPORTION (rate, share of rows meeting a condition) from two OpenDP noisy histograms
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
DP selection of the top K candidates (from a public list) by the proportion of rows that satisfy a condition, with pure ε-DP under replace-one (bounded DP, n public). Two count-by-category histograms with Laplace noise at ε/2 each (L1 sensitivity 2 per histogram under replace-one), then proportion = noisy numerator / noisy denominator and the top K are post-processing. Preferred selection route for proportion scores; no separate selection mechanism is needed.

## Steps
1. `space <- (vector_domain(atom_domain(T=str)), symmetric_distance())`: the input is the group column as a list of str.
2. `make(scale) <- space >> then_count_by_categories(categories=PUBLIC_GROUPS) >> then_laplace(scale=scale)`. The output has one extra last entry (the null bin for values outside the list); drop it.
3. `scale <- binary_search_param(make, d_in=2, d_out=ε/2)` (replace-one = symmetric distance 2); build two measurements `make(scale)`.
4. Numerator: apply one to the group values of rows meeting the condition; denominator: apply the other to all group values.
5. Post-processing: drop groups whose noisy size is below `10 * scale`, `rate <- noisy_num / noisy_den`, sort, keep k.

## Privacy accounting
Replace-one (bounded DP, n public): a changed row can leave one bin and enter another, so each histogram has L1 sensitivity 2 (filtering rows first keeps it at 2); the map reports it from `D_IN = 2`. A row is in both inputs, so the two releases compose: `epsilon_spent <- m_num.map(2) + m_den.map(2)`; assert it is <= ε. The ratio, cut-off and sort are free.

## Pitfalls
- `then_noisy_top_k` over rate scores with L-inf sensitivity 1 at k picks: valid but the noise swamps rate differences of a few percent; picks are near random.
- `d_in=1` / `map(1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Forgetting the null bin at the end of the count vector, or building the category list from the data.
- Ranking by the count of matching rows instead of the rate: a different query.
- The noisy-size cut-off must be a fixed rule (here 10 noise scales), not tuned on the data.
