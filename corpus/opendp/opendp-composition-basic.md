---
id: opendp-composition-basic
title: Non-adaptive sequential composition of several queries (make_composition)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.combinators.make_composition
  - opendp.combinators.make_basic_composition
  - opendp.mod.binary_search_chain
  - opendp.measurements.then_laplace
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/non-adaptive-composition.rst
source_license: MIT
---

## Goal
Answer several queries on the same data under one total epsilon, decided up front, and have OpenDP report the total. Neighbouring relation: replace-one (n public): sized domain `vector_domain(atom_domain(T=float, nan=False), size=N)` and `D_IN = 2`.

## Steps
1. `enable_features("contrib")`.
2. Choose a public split of the total: EPS_1 + ... + EPS_m = EPS_TOTAL (even, or weighted toward the queries that need accuracy).
3. Build each query as its own measurement, searching its scale against its share:
   `meas_i <- binary_search_chain(lambda s: PRE_i >> then_laplace(s), d_in=D_IN, d_out=EPS_i)`
   All must share the same input domain, input metric and output measure.
4. Compose: `comp <- make_composition([meas_1, ..., meas_m])`.
5. Check: `assert comp.map(D_IN) <= EPS_TOTAL`.
6. Release all at once: `answers <- comp(DATA)`; a list in the same order.

Signature for reference: `make_composition(measurements)`.

## Privacy accounting
The composition's privacy map sums the parts (basic sequential composition), so `comp.map(2)` is the total actually spent under replace-one. Alternative, add/remove-one (n private): unsized domain and `D_IN = 1` everywhere; the count is then charged 1 and a clamped sum max(|L|, |U|, U - L). For pure DP with delta = 0, the plain sum is the right bound.

## Pitfalls
- `make_basic_composition` still exists but is deprecated since 0.14 (renamed to `make_composition`).
- `D_IN = 1` with n public under-charges histograms and group-bys by 2x (for a single clamped sum OpenDP's unsized `map(1)` is already max(|L|, |U|, U - L), the replace-one change). On the sized domain a clamped sum costs U - L at `D_IN = 2`; a plain `then_count` is charged 2 although n is public, so count rows that match a condition as a 0/1 sum (charged 1).
- Giving each query the full EPS_TOTAL spends m times the budget.
- Queries on different domains (e.g. one on str, one on float columns) cannot share one composition. Start every query from the same row-level domain and select columns inside the pipeline, or use the Context API.
- Mixing pure-DP (Laplace) and zCDP (Gaussian) measurements fails with "All output measures must be the same"; cast first (see `opendp-measure-casting`).
- For queries that depend on earlier answers, use an adaptive compositor or a Context instead.
