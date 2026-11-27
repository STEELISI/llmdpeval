---
id: opendp-polars-null-and-distinct-counts
title: Polars Context - non-null count, null count and distinct count (signed counting queries)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.DPExpr.count
  - opendp.extras.polars.DPExpr.null_count
  - opendp.extras.polars.DPExpr.n_unique
  - opendp.extras.polars.DPExpr.len
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/aggregation.rst
source_license: MIT
---

## Goal
Profile a column's data quality under DP: how many values are present, how many are missing, and how many distinct values there are. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`; alternative, add/remove-one (n private): `unit_of(contributions=1)`.

## Steps
1. `enable_features("contrib")`; context with `loss_of(epsilon=EPS)`, `split_evenly_over=K`, `polars.Margin(max_length=N_MAX)`.
2. `q <- context.query().select(col(X).dp.count(signed=True).alias("count"), col(X).dp.null_count(signed=True).alias("null_count"), col(X).dp.n_unique(signed=True).alias("n_unique"))`
3. `q.summarize()` lists one scale per statistic; `row <- q.release().collect()`.
Related: `col(X).dp.len(signed=True)` counts all rows of the column including nulls; `dp.len(signed=True)` counts frame rows.

## Privacy accounting
All expressions in one `select` share one slot of the context split, so each gets a share of it. Read the total from `context.accountant.map(context.d_in)`. Each of the three counts moves by at most 1 under either relation; OpenDP's `changes=1` charge is conservative (twice the `contributions=1` noise scale), so `contributions=1` is also exact here, but keep `changes=1` when the task states replace-one.
Tip from the docs: count and null_count are complementary. Grouping by `col(X).is_null()` with keys `[True, False]` and one `dp.len(signed=True)` answers both with half the noise.

## Pitfalls
- `signed=True` (new in 0.16) returns Int64 that may be negative. Omitting it warns and clamps negatives to 0; the default will change.
- `n_unique` stays sensitive even under replace-one: one change can add or remove a distinct value.
- Filtering the column first changes what is counted; the filter must be a public rule.
- The Framework API equivalent of `n_unique` is `then_count_distinct` (see `opendp-count-distinct`).
