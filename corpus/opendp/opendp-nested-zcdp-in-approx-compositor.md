---
id: opendp-nested-zcdp-in-approx-compositor
title: Nesting a zCDP (Gaussian) compositor inside an (epsilon, delta) compositor
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: composition
dp_notion: approx
needs_bounds: true
api_calls:
  - opendp.combinators.make_adaptive_composition
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.combinators.make_approximate
  - opendp.measures.approximate
  - opendp.measures.max_divergence
  - opendp.measures.zero_concentrated_divergence
  - opendp.measurements.then_gaussian
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/compositor-chaining-and-nesting.rst
source_license: MIT
---

## Goal
Spend an overall (epsilon, delta) budget on a mix of queries: some with Gaussian noise (composed in zCDP, which is tighter) and some with Laplace noise (pure DP). Neighbouring relation: replace-one (n public): sized domain, `d_in=2` for the root, the child and every query.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count).
2. Root: `root <- make_adaptive_composition(*space, output_measure=approximate(max_divergence()), d_in=2, d_mids=[(EPS_A, DELTA), (EPS_B, 0.0)])`; `qbl_root <- root(DATA)`.
3. Child zCDP compositor, sized to fit slot A:
   `make_child <- lambda r: make_fix_delta(make_zCDP_to_approxDP(make_adaptive_composition(*space, output_measure=zero_concentrated_divergence(), d_in=2, d_mids=[r, r])), delta=DELTA)`
   `R <- binary_search_param(make_child, d_in=2, d_out=(EPS_A, DELTA), T=float)`
   `qbl_child <- qbl_root(make_child(R))`
4. Gaussian queries into the child, each with rho `R`: `binary_search_chain(lambda s: space >> then_clamp((L, U)) >> then_sum() >> then_gaussian(s), d_in=2, d_out=R)`.
5. Laplace query into slot B: build it with `d_out=EPS_B`, then wrap with `make_approximate(meas)` (delta = 0) before `qbl_root(...)`.

## Privacy accounting
`root.map(2)` returns the total replace-one `(epsilon, delta)` = sums of `d_mids`. Alternative, add/remove-one (n private): unsized domain and `d_in=1` throughout (`root.map(1)`). The child's rho per query comes from `binary_search_param` on the converted compositor, never from a hand rho-to-epsilon formula.

## Pitfalls
- A pure-DP measurement cannot enter an approximate-DP compositor unless wrapped in `make_approximate`.
- `make_zCDP_to_approxDP` alone gives a privacy profile; `make_fix_delta` turns it into a pair the root accepts.
- Measures must match exactly: `approximate(max_divergence())` is the (epsilon, delta) measure.
- Each compositor is spent in `d_mids` order; extra queries fail with "out of queries".
