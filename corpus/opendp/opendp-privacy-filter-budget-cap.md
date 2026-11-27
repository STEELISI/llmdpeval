---
id: opendp-privacy-filter-budget-cap
title: Privacy filter - answer queries until a total epsilon cap is reached
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.combinators.make_fully_adaptive_composition
  - opendp.combinators.make_privacy_filter
  - opendp.measures.max_divergence
  - opendp.transformations.then_count
  - opendp.transformations.then_clamp
  - opendp.transformations.then_resize
  - opendp.transformations.then_mean
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
  - opendp.mod.OpenDPException
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/privacy-filters.rst
source_license: MIT
---

## Goal
Run an open-ended list of DP queries against one dataset under a hard total epsilon. Each query picks its own epsilon; the filter refuses any query that would push the total over the cap. Neighbouring relation: replace-one (n public): sized domain, `d_in=2` for the filter and every query.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count).
2. `odometer <- make_fully_adaptive_composition(*space, output_measure=max_divergence())`
3. `filter <- make_privacy_filter(odometer, d_in=2, d_out=EPS_TOTAL)`. This is an ordinary `Measurement` (it can itself be composed).
4. `qbl <- filter(DATA)`.
5. For each query: build `meas <- binary_search_chain(lambda s: PRE >> then_laplace(s), d_in=2, d_out=EPS_Q)` where `PRE` is e.g. `space >> then_count()`, or `space >> then_clamp((L, U)) >> then_mean()` (sized: charged (U - L) / N).
6. `try: answer <- qbl(meas)` / `except OpenDPException: stop` (message "filter is now exhausted").
7. After each accepted query, read `qbl.privacy_loss(2)`.

Signature for reference: `make_privacy_filter(odometer, d_in, d_out)`.

## Privacy accounting
`make_privacy_filter(...).map(2)` is the cap `d_out`. The spent amount is `qbl.privacy_loss(2)`, read after each successful query: once the filter is exhausted, the queryable also refuses further calls, including `privacy_loss`.

## Pitfalls
- A rejected query is not answered and costs nothing, but the filter stays closed afterwards.
- Do not catch the exception and retry with a smaller epsilon in a loop driven by the data; choose epsilons from public needs.
- `d_in` passed to the filter must match the neighbour definition: 2 for replace-one (one changed row). Alternative, add/remove-one (n private): unsized domain and `d_in=1` throughout, the cap is then `map(1)`, and a mean needs `then_resize` to a DP count.
- N, C, L, U are public values, never computed from the data.
