---
id: opendp-zcdp-composition-gaussian
title: Composing several Gaussian releases in zCDP, converting to (epsilon, delta) once
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: composition
dp_notion: zcdp
needs_bounds: true
api_calls:
  - opendp.combinators.make_composition
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.measurements.then_gaussian
  - opendp.transformations.then_count
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/measure-casting.rst
source_license: MIT
---

## Goal
Release several statistics with Gaussian noise under one (epsilon, delta) budget. Composing in zCDP (rho adds up) and converting once at the end gives less noise than converting each query and adding epsilons. Neighbouring relation: replace-one (n public): sized domain, `d_in=2`.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count).
2. Aggregates: `pre_count <- space >> then_count()`; `pre_sum <- space >> then_clamp((L, U)) >> then_sum()`.
3. One noise knob shared by all queries, scaled per query by public weights and ranges:
   `make <- lambda s: make_fix_delta(make_zCDP_to_approxDP(make_composition([pre_count >> then_gaussian(s * W1), pre_sum >> then_gaussian(s * W2 * (U - L))])), delta=DELTA)`
4. `meas <- binary_search_chain(make, d_in=2, d_out=(EPS, DELTA))`
5. `count, total <- meas(VALUES)`.

## Privacy accounting
`make_composition` of zCDP measurements adds rho values; `make_zCDP_to_approxDP` turns the total rho into a privacy profile; `make_fix_delta` fixes delta. `meas.map(2)` returns the replace-one `(epsilon, delta)`. Alternative, add/remove-one (n private): unsized domain and `d_in=1` throughout (`meas.map(1)`). If the task gives a rho budget, skip the conversion and search with `d_out=RHO`.

## Pitfalls
- Do not convert each Gaussian query separately and add the epsilons; that throws away the zCDP advantage.
- All measurements in `make_composition` must share input space and output measure; mix with pure-DP queries only after `make_pureDP_to_zCDP` (see `opendp-measure-casting`).
- Weights decide how the budget is shared; choose them from public accuracy needs.
- delta should be far smaller than 1/n.
