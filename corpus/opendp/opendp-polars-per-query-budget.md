---
id: opendp-polars-per-query-budget
title: Polars Context - each query sets its own epsilon (context.query(epsilon=...))
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.Context.query
  - opendp.context.Context.remaining_privacy_loss
  - opendp.context.Context.current_privacy_loss
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.sum
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/context/test_context.py
source_license: MIT
---

## Goal
Use a Context with only a total budget and let each query name its own epsilon when it is asked, instead of splitting the budget up front. The context refuses a query that would overspend. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`; alternative, add/remove-one (n private): `unit_of(contributions=1)`.

## Steps
1. `enable_features("contrib")`.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS_TOTAL), margins=[polars.Margin(max_length=N_MAX)])`, with no `split_evenly_over` / `split_by_weights`.
3. `count <- context.query(epsilon=E1).select(dp.len(signed=True)).release().collect()`
4. `context.remaining_privacy_loss()`: budget left (EPS_TOTAL - E1).
5. `total <- context.query(epsilon=E2).select(col(X).cast(float).dp.sum((L, U))).release().collect()`
6. `context.current_privacy_loss()`: spent so far.
Under zCDP the per-query keyword is `rho=...` (plus `delta=...` when keys are released).

## Privacy accounting
The context composes queries sequentially and tracks the running total; `current_privacy_loss()` is the value to report. Under `changes=1` (replace-one) the length is public and released without noise and a sum is charged U - L; under `contributions=1` (add/remove) the length gets noise and a sum is charged max(|L|, |U|), which under-charges a replace-one task when L < 0 < U. E1 + E2 + ... must stay at or below EPS_TOTAL or the next release fails.

## Pitfalls
- With a split already set in `Context.compositor`, do not also pass `epsilon=` to `query()`.
- Choosing E2 after seeing the first answer is allowed (adaptive), but choose it from accuracy needs, not from the data value.
- `remaining_privacy_loss()` going negative is impossible; a refused query raises instead.
- Framework-API version: `opendp-context-compositor-framework`.
