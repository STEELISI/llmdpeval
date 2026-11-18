---
id: opendp-polars-conditional-sum-filter
title: Polars Context - sum of one column over rows meeting a public condition (expression filter)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.DPExpr.sum
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/manipulation.rst
source_license: MIT
---

## Goal
Release the sum of a numeric column restricted to rows where another numeric column passes a public threshold (e.g. total hours of people above some level), using a filter inside the aggregate expression. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)`, `polars.Margin(max_length=N_MAX)`.
2. `q <- context.query().select(col(X).cast(float).filter(col(C) > T).dp.sum((L, U)))`
   The filter's predicate must be row-by-row; both sides come from the same rows.
3. `q.summarize()`, then `q.release().collect().item()`.
To get both sides of the condition at once with the same budget: `group_by(col(C) > T).agg(...)` with keys `[True, False]`.

## Privacy accounting
The filter is a public row-by-row rule, so the sensitivity is that of the unfiltered sum. Replace-one: a changed row moves the sum by up to U - L (OpenDP charges U - L when L < 0 < U and max(|L|, |U|) otherwise; e.g. bounds (-5, 10): scale 15 per unit epsilon). Alternative, add/remove-one (n private): `unit_of(contributions=1)`, charged max(|L|, |U|) per row (scale 10 for (-5, 10)). One `select` uses one context slot; total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` with n public under-charges sums whose bounds straddle 0 (1.5x for (-5, 10)).
- T must be public (from the task or domain knowledge), not chosen after looking at the data.
- OpenDP does not infer non-null from the predicate; `.dp.sum` still inserts its own imputation.
- A filter whose inputs are not row-by-row (e.g. comparing to a column mean) is rejected.
