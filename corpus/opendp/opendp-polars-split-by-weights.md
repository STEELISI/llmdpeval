---
id: opendp-polars-split-by-weights
title: Polars Context - uneven budget split (split_by_weights) and accuracy from summarize(alpha)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.Context.remaining_privacy_loss
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/index.rst
source_license: MIT
---

## Goal
Split one total epsilon unevenly over a fixed sequence of queries, giving more budget to the query that needs more accuracy, and check each query's error bound before releasing it. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`; alternative, add/remove-one (n private): `unit_of(contributions=1)`.

## Steps
1. `enable_features("contrib")`.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_by_weights=[W1, W2], margins=[polars.Margin(max_length=N_MAX)])`. Query i gets EPS * Wi / sum(W).
3. Query 1: `q1 <- context.query().select(dp.len(signed=True))`; `q1.summarize(alpha=A)` has an "accuracy" column: with probability 1 - A the error is below it. Then `q1.release().collect()`.
4. Query 2: `q2 <- context.query().select(col(X).cast(float).dp.mean((L, U)))`; summarize, then release.
5. `context.remaining_privacy_loss()` is 0 after the last weighted query.

## Privacy accounting
`context.d_mids` lists the per-query budgets (e.g. weights [1, 3] give [0.25, 0.75] of EPS). Queries take budgets in order. `context.accountant.map(context.d_in)` is the total. Under `changes=1` the length is public (released without noise) and a sum is charged U - L; under `contributions=1` (add/remove) the length gets noise and a sum is charged max(|L|, |U|), which under-charges a replace-one task when L < 0 < U.

## Pitfalls
- Use either `split_evenly_over` or `split_by_weights`, not both.
- Weights are fixed before seeing data; do not reweight after looking at results.
- A mean has no single accuracy number (it is a ratio of noisy sum and noisy length); summarize shows the two parts.
- An extra query beyond the list of weights is refused.
