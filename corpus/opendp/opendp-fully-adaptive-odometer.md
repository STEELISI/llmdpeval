---
id: opendp-fully-adaptive-odometer
title: Fully adaptive composition (odometer) - pick each query and its epsilon as you go
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.combinators.make_fully_adaptive_composition
  - opendp.measures.max_divergence
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/fully-adaptive-composition.rst
source_license: MIT
---

## Goal
Answer several DP queries one after another when neither the queries nor their privacy losses are fixed in advance (a later query may depend on an earlier answer). An odometer tracks the running privacy loss. Neighbouring relation: replace-one (n public): sized domain, `d_in=2` everywhere.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count).
3. `odometer <- make_fully_adaptive_composition(*space, output_measure=max_divergence())`. Unlike adaptive composition there is no `d_in` / `d_mids` argument.
4. `qbl <- odometer(DATA)`: an odometer queryable.
5. For each query, build a measurement on the same `space` with its own epsilon, e.g.
   `m_count <- binary_search_chain(lambda s: space >> then_count() >> then_laplace(s), d_in=2, d_out=EPS_1)` (OpenDP charges a plain count 2; for matching rows use a 0/1 sum, charged 1)
   `m_sum <- binary_search_chain(lambda s: space >> then_clamp((L, U)) >> then_sum() >> then_laplace(s), d_in=2, d_out=EPS_2)` (charged U - L)
6. `answer <- qbl(m_count)`, then decide the next query (and its EPS) from the answer.
7. Read the total so far with `qbl.privacy_loss(2)`.

Signature for reference: `make_fully_adaptive_composition(input_domain, input_metric, output_measure)`.

## Privacy accounting
The odometer adds up the actual losses of the queries it answered; `qbl.privacy_loss(2)` returns the total under replace-one. Alternative, add/remove-one (n private): unsized domain, every search at `d_in=1`, total `qbl.privacy_loss(1)`; the clamped sum is then charged max(|L|, |U|, U - L). It does not stop you: if you need a hard cap, wrap it in a privacy filter (see `opendp-privacy-filter-budget-cap`). Report the value from `privacy_loss`, not a hand sum.

## Pitfalls
- An odometer alone has no budget limit. Use it only when the analyst enforces the cap, or use `make_privacy_filter`.
- Every query must start from the odometer's input space (same domain and metric).
- (L, U) must be public, not min/max of the data.
- Searching queries at `d_in=1` while n is public under-charges histograms and group-bys by 2x.
- The odometer is not a `Measurement`; it cannot go into `make_composition` until wrapped by a filter.
