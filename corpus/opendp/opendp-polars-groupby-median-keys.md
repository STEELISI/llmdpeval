---
id: opendp-polars-groupby-median-keys
title: Polars Context - median per group from public candidates and public keys
framework: opendp
framework_version: "0.16.0"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.DPExpr.median
  - opendp.extras.polars.DPExpr.quantile
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/test_polars.py
source_license: MIT
---

## Goal
Release the median (or another quantile) of a numeric column within each group of a public group list. Each group's answer is picked from a public candidate grid by a noisy score (exponential-mechanism style). Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)` and `polars.Margin(max_length=N_MAX)`.
2. `CANDS`: public sorted grid of possible answers, same dtype as the column after casting.
3. `q <- context.query().group_by(G).agg(col(X).cast(float).dp.median(CANDS)).with_keys(LazyFrame({G: KEYS}))`
   Other quantiles: `col(X).cast(float).dp.quantile(ALPHA, CANDS)`.
4. `q.summarize()` shows distribution "ExponentialMin"; `q.release().collect()`.

## Privacy accounting
Replace-one (bounded DP): a changed row can leave one group and join another, so two groups' selections can change; `unit_of(changes=1)` makes OpenDP charge that. Together they cost one slot of the context split. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Candidates from the data (unique values, min/max) leak; use a public grid.
- Answers are always grid points; a coarse grid caps precision.
- Small groups give near-random medians; the noise does not shrink with the group.
- Without `with_keys`, key release needs a delta.
- Ungrouped version: `opendp-polars-quantile-median`.
