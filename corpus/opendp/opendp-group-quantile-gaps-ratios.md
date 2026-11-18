---
id: opendp-group-quantile-gaps-ratios
title: Polars Context - several quantiles per group, quantile gaps between groups and percentile ratios (p90/p10)
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
  - opendp.extras.polars.DPExpr.quantile
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Release several quantiles (e.g. 10th, 50th, 90th percentiles) of a numeric column in each group of a public group list, then derive the gap between two groups at each quantile and each group's percentile ratio (p90/p10, p50/p10) or spread (p90 - p10). Neighbours: replace-one (bounded DP, n public), `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`. Context: `Context.compositor(data=LazyFrame, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=1, margins=[polars.Margin(max_length=N_MAX)])`.
2. `CANDS`: a public sorted grid between public bounds (topcode, design limits). For ratios, a log-spaced grid gives the same relative precision at the low and high end.
3. `q <- context.query().group_by(G).agg(col(X).cast(float).dp.quantile(A, CANDS).alias(NAME_A) for A in ALPHAS).with_keys(LazyFrame({G: GROUPS}))`
4. `out <- q.release().collect()`.
5. Post-processing: sort each group's quantiles (independent draws can cross); `gap[A] = q[g1][A] - q[g2][A]`; `ratio[g] = q[g][hi] / q[g][lo]` if `q[g][lo] > 0`; for log values, the gap in logs is a log ratio.

Signature for reference: `DPExpr.quantile(self, alpha, candidates, scale=None)`.

## Privacy accounting
All groups and quantiles are one query, one slot of the split. Under `unit_of(changes=1)` a changed row can leave one group and join another, and OpenDP charges both. Read the total from `context.accountant.map(context.d_in)` and assert it is ≤ EPS.

## Pitfalls
- `unit_of(contributions=1)` is add/remove; under replace-one the shown ε is too low.
- Candidates from the data (unique values, min, max) leak them; use public bounds.
- A gap or ratio computed from exact quantiles of one group and DP quantiles of the other.
- Survey or sampling weights: a weighted quantile lets one row move the weighted rank by its weight; the framework's quantile has no weights. Drop them (unweighted estimand) or say the method is not available.
- Small groups give near-random quantiles; tail quantiles (p5, p10) of small groups are the least stable, so ratios with them inherit the noise.
- One query per group or per quantile at the full EPS each: the total is the sum.
- Grouped median only: `opendp-polars-groupby-median-keys`.
