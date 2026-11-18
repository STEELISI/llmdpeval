---
id: opendp-polars-quantile-median
title: Polars Context API - median and quantiles from public candidates
framework: opendp
framework_version: "0.16.0"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.DPExpr.quantile
  - opendp.extras.polars.DPExpr.median
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Release a median or several quantiles of a column with the polars Context. OpenDP scores each public candidate and picks one with noise (an exponential-mechanism style selection), instead of adding noise to the value. Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; LazyFrame data; context as in `opendp-polars-context-basic`, with `privacy_unit=unit_of(changes=1)`, `privacy_loss=loss_of(epsilon=EPS)` and `polars.Margin(max_length=N_MAX)`.
2. `CANDS`: a public grid of possible answers (e.g. whole numbers between public bounds L and U), matching the column's type after casting.
3. Median: `q <- context.query().select(col(X).cast(?).dp.median(CANDS))`.
4. Several quantiles in one release:
   `q <- context.query().select(col(X).cast(?).dp.quantile(A, CANDS).alias(NAME_A) for A in ALPHAS)`
5. `q.summarize()` lists one row per quantile with distribution "ExponentialMin" and its scale; accuracy is not available for this mechanism.
6. `df <- q.release().collect()`.

Signature for reference: `DPExpr.quantile(self, alpha, candidates, scale=None)`.

## Privacy accounting
One `select` is one release and uses one slot of the context's split; the quantiles inside it share that slot evenly (the docs note that the shown scales differ by quantile for internal reasons, not because the budget differs). With `unit_of(changes=1)` OpenDP charges a replaced row, so the shown scales are larger than under add/remove. Read the total with `context.accountant.map(context.d_in)` and assert it is ≤ EPS.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one (n private); under replace-one it under-states the real ε.
- Candidates built from the data (unique values, min/max) leak it.
- The answer is always one of the candidates; a coarse grid limits precision.
- Filtering inside the query is allowed (e.g. dropping a "not applicable" code) but must be a public rule.
- Released quantiles are not forced to be monotone; sort afterwards if needed.
- See `opendp-polars-context-basic` for the polars version pin.
