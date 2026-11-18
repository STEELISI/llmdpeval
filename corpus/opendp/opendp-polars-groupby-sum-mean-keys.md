---
id: opendp-polars-groupby-sum-mean-keys
title: Polars Context - per-group count, sum and mean with public keys
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyGroupByQuery.agg
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.sum
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/grouping.ipynb
source_license: MIT
---

## Goal
Release, for each group in a public list of group labels, the number of rows and the sum and mean of a numeric column. Public keys make the release pure DP: no keys are learned from the data. Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context as in `opendp-polars-context-basic` with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)` and `margins=[polars.Margin(max_length=N_MAX)]`.
2. `q <- context.query().group_by(G).agg(dp.len(signed=True), col(X).cast(float).dp.sum((L, U)).alias("sum"), col(X).cast(float).dp.mean((L, U)).alias("mean")).with_keys(LazyFrame({G: KEYS}))`
3. `q.summarize()`: one row per noisy quantity (the mean is reported as a noisy sum and a noisy length).
4. `out <- q.release().collect()`: one row per key in KEYS, also for keys absent from the data.

## Privacy accounting
The whole `group_by ... agg` is one query and uses one slot of the context split; the aggregates inside share that slot. Replace-one (bounded DP): a changed row can leave one group and join another; `unit_of(changes=1)` makes OpenDP charge both groups (it doubles the add/remove noise). Group sizes stay private: they are the noisy `len`. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Without `.with_keys(...)` the keys must be released privately, which needs a delta in `loss_of` (see `opendp-polars-groupby-public-keys`).
- KEYS must come from a codebook, not from `unique()` on the data.
- The dtype of KEYS must match the grouping column (e.g. Int32 vs Int64 fails).
- Small groups give very noisy means; consider reporting sum and count separately.
