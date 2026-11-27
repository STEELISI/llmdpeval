---
id: opendp-polars-preprocess-then-noise
title: Polars Context - clean with cast/fill_null/clip, then plain .sum() with .dp.noise()
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.with_columns
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.DPExpr.noise
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/manipulation.rst
source_license: MIT
---

## Goal
Separate data cleaning from the DP release: prepare a column with ordinary polars expressions, then add noise to a plain aggregate. OpenDP tracks facts such as clip bounds and non-nullness through the plan and uses them for the sensitivity. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`; alternative, add/remove-one (n private): `unit_of(contributions=1)`.

## Steps
1. `enable_features("contrib")`; context with `loss_of(epsilon=EPS)` and `polars.Margin(max_length=N_MAX)`.
2. `q <- context.query().with_columns(col(X).cast(int).fill_null(C).clip(L, U)).select(col(X).sum().dp.noise())`
   - `cast(int)`: failed casts become null (no exception), so `fill_null` must follow.
   - `fill_null(C)` before `clip`: fill_null drops known bounds.
   - `clip(L, U)` sets the bounds that the sum's sensitivity uses.
3. `q.summarize()`, then `q.release().collect().item()`.

## Privacy accounting
`.dp.noise()` picks Laplace or Gaussian from the context's privacy loss and a scale from the context split. `context.accountant.map(context.d_in)` gives the total. Under `changes=1` the length is public (released without noise) and a sum is charged U - L; under `contributions=1` (add/remove) the length gets noise and a sum is charged max(|L|, |U|), which under-charges a replace-one task when L < 0 < U. The shortcut `col(X).dp.sum((L, U))` inserts the same fill and clip steps for you.

## Pitfalls
- A float sum carries an extra rounding term in its sensitivity that grows with n and the bounds; casting to int before summing removes it (the docs show a much smaller scale).
- Casting a grouping column voids margin facts for that column.
- `.sum()` without a prior `clip` is rejected: the sensitivity is unbounded.
- Fill values and bounds must be public.
