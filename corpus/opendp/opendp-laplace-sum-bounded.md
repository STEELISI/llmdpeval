---
id: opendp-laplace-sum-bounded
title: Bounded sum with clamping and the Laplace mechanism
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_impute_constant
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.transformations.make_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-sum.ipynb
source_license: MIT
---

## Goal
Release the sum of a numeric column with epsilon-DP. Each value's influence is limited by clamping to public bounds (L, U). Neighbouring relation: replace-one (n public), the default: sized domain, `D_IN = 2`, charged U - L.

## Steps
1. `enable_features("contrib")`.
2. Choose L, U from public knowledge (documentation, units, legal ranges), never from the data.
3. Input space (replace-one, N = public row count):
   - no NaN possible: `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()`;
   - NaN possible: `space <- vector_domain(atom_domain(T=float, nan=True), size=N), symmetric_distance()`.
4. Preprocess inside OpenDP so the stability maps see it:
   - no NaN: `pre <- space >> then_clamp((L, U))`;
   - NaN possible: `pre <- space >> then_impute_constant(C) >> then_clamp((L, U))` (C is a public constant in [L, U]). `then_clamp` on a `nan=True` domain without the imputer raises.
5. `D_IN = 2` (one changed row = symmetric distance 2). Alternative, add/remove-one (n private): drop `size=N` and use `D_IN = 1`; the sum is then charged max(|L|, |U|, U - L).
6. `make <- lambda scale: pre >> then_sum() >> then_laplace(scale)`
7. `meas <- binary_search_chain(make, d_in=D_IN, d_out=EPS)`; `dp_sum <- meas(VALUES_AS_LIST)`.

Signature for reference: `make_sum(input_domain, input_metric)`; the bounds come from the input domain, which `then_clamp` sets.

## Privacy accounting
Do not compute sensitivity yourself. Check `(pre >> then_sum()).map(D_IN)` to see what OpenDP charges (replace-one: `map(2)` = U - L on the sized domain; add/remove: `map(1)` = max(|L|, |U|, U - L) unsized), and assert `meas.map(D_IN) <= EPS` before release. Float sums include a small extra term for rounding error, which a hand formula would miss.

## Pitfalls
- On an unsized domain (add/remove) the charge is max(|L|, |U|, U - L). For same-sign bounds away from 0, e.g. (2, 10), that is 10 against a width of 8: shift by a public offset or, with n public, use the sized domain (charged U - L). For mixed-sign bounds the charge already equals U - L, so shifting gains nothing.
- Integer columns: keep them as int (atom T=int) where possible; float sums over very large n carry a rounding penalty.
- Clamping outside OpenDP (e.g. numpy clip) and then passing an unbounded domain does not tell OpenDP the bounds. Clamp with `then_clamp`. Declaring `atom_domain(bounds=...)` is only safe on data already clamped to those bounds: OpenDP does not check it at call time, so an out-of-range value is released unprotected.
- Clamping biases the sum if the bounds are too tight; the reported accuracy covers noise only, not that bias.
