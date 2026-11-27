---
id: opendp-adaptive-composition-queryable
title: Adaptive composition with a budget-enforcing queryable
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.combinators.make_adaptive_composition
  - opendp.combinators.make_sequential_composition
  - opendp.measures.max_divergence
  - opendp.metrics.symmetric_distance
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/adaptive-composition.rst
source_license: MIT
---

## Goal
Run several queries one after another, possibly choosing later queries from earlier answers, while OpenDP enforces a per-query budget and the total. Neighbouring relation: replace-one (n public), so every map and search uses `D_IN = 2` (one changed row = symmetric distance 2).

## Steps
1. `enable_features("contrib")`.
2. Fix the shared space and the budget plan (sized domain, n public: `DOMAIN <- vector_domain(atom_domain(T=float, nan=False), size=N)`; `D_IN = 2`):
   `comp <- make_adaptive_composition(input_domain=DOMAIN, input_metric=symmetric_distance(), output_measure=max_divergence(), d_in=D_IN, d_mids=[EPS_1, ..., EPS_m])`
3. Check the total before touching data: `comp.map(D_IN)` equals the sum of `d_mids`.
4. Start the session: `qbl <- comp(DATA)`. This returns a queryable, not an answer.
5. For each query i, build a measurement whose map at `D_IN` is at most EPS_i, e.g. `binary_search_chain(lambda s: PRE_i >> then_laplace(s), d_in=D_IN, d_out=EPS_i)`, then `answer_i <- qbl(meas_i)`.
6. Queries must be submitted in the order of `d_mids`.

Signature for reference: `make_adaptive_composition(input_domain, input_metric, output_measure, d_in, d_mids)`.

## Privacy accounting
The compositor rejects a query whose map exceeds its slot ("insufficient budget for query") and refuses more queries than `d_mids` entries. The total guarantee is `comp.map(2)` under replace-one. Report that, not a sum you typed.

## Pitfalls
- `D_IN` given to the compositor must equal the privacy unit used to search each query's scale: 2 for replace-one.
- Alternative, add/remove-one (n private): unsized domain, `D_IN = 1` for the compositor and every query; then the count is charged 1 and a clamped sum max(|L|, |U|, U - L). Do not use `D_IN = 1` when n is public: for histograms and group-bys it reports half the replace-one epsilon.
- With n public the row count is known; OpenDP still charges a `then_count` 2 at `D_IN = 2`. For a filtered count, sum a 0/1 indicator on the sized domain (charged 1).
- Every query must start from the compositor's input domain and metric.
- A query that asks for slightly more than its slot (e.g. a scale found for a larger epsilon) is refused. Search against the exact slot.
- `make_sequential_composition` is the older name, deprecated since 0.14; use `make_adaptive_composition`. The Context API wraps the same compositor with less boilerplate.
