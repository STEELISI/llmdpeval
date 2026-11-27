---
id: opendp-topk-count-from-histogram
title: Top-k categories by COUNT from one noisy histogram (preferred over peeling for a short public list)
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
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-quantile.ipynb
source_license: MIT
---

## Goal
Release the k categories with the most rows (or the most rows meeting a condition) from a short public category list, with pure epsilon-DP under replace-one (bounded DP, n public) by default. Release one Laplace histogram over every listed category, then sort it. The sort is post-processing (Dwork & Roth 2014, Prop. 2.1), so the cost does not grow with k.

## Steps
1. `enable_features("contrib")`.
2. `CATS`: the public category list, fixed before looking at the data, in a public order (e.g. alphabetical). Include categories whose true count may be zero.
3. If counting only rows that meet a condition, filter the rows first; one row still touches one bin.
4. Neighbouring relation: replace-one (n public) gives `D_IN = 2`, because one changed row lowers one bin and raises another; only a task that says add/remove-one gives `D_IN = 1`.
5. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`
6. `make <- lambda s: space >> then_count_by_categories(categories=CATS, null_category=False) >> then_laplace(s)`
7. `meas <- binary_search_chain(make, d_in=D_IN, d_out=EPS)`; `noisy <- meas(VALUES_AS_STR_LIST)`.
8. Sort indices by `(-noisy[i], i)`: noisy count first, ties broken by position in the public list. Keep the first k.

## Privacy accounting
Replace-one (bounded DP): one vector release priced on its L1 sensitivity (2); read it with `meas.map(2)` and assert it is at most EPS. Sorting, slicing and tie-breaking use only the noisy counts and the public list, so they are free. Peeling (k noisy-max rounds) must split EPS over k rounds instead.

## Pitfalls
- Peeling or `make_noisy_top_k` on a short public list: each round runs at EPS/k, so its noise grows with k. When each row adds to one candidate, Laplace noise on one histogram is the better choice (Gillenwater et al., ICML 2022, Conclusion): it pays epsilon once for any k, while each peeling round gets only EPS/k.
- If you do use `then_noisy_top_k(max_divergence(), k=K, scale=?)`: its privacy map grows linearly in K at a fixed scale (checked on 0.16.0: map(1) = K/scale for monotone counts), so search the scale with the real K and assert `meas.map(1) <= EPS`.
- Category list or order taken from the data (`unique()`, `value_counts()` order): a tie in the noisy counts then resolves toward the true ranking, a leak through the sort order.
- `D_IN = 1` under replace-one: the histogram's L1 sensitivity is 2 there.
- This route is for COUNT scores. For proportions (rates) use `opendp-topk-rate-two-histograms`.
- Printing the exact counts next to the private top-k releases them.
