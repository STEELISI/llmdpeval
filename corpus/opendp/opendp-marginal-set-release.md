---
id: opendp-marginal-set-release
title: Set of 1-way and 2-way marginals under one budget with the polars Context (input for synthetic data)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.context.loss_of
  - opendp.extras.polars.Margin
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/grouping.ipynb
source_license: MIT
---

## Goal
Release several marginal tables (one-way counts per category, two-way cross-tabulations of column pairs) over public category lists, all under one epsilon, for replace-one neighbours (bounded DP, n public). The noisy tables are the only input to later steps: consistency post-processing, a synthetic data generator, or utility checks.

## Steps
1. Public inputs: the list of marginals as column tuples, e.g. `[(A,), (B,), (A, B), (B, C)]`, and a public category list per column (codebook values, zeros included).
2. `context <- Context.compositor(data=LAZYFRAME_OF_STRING_COLUMNS, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=M, margins=[Margin(max_length=PUBLIC_UPPER_N)])`, with M = number of tables.
3. For each table: `keys <- LazyFrame of the public cross product of its category lists`, then `context.query().group_by([cols]).agg(dp.len(signed=True)).with_keys(keys).release().collect()`.
4. Re-index the result by the key columns (row order is not key order) and reshape into a K_a x K_b array.
5. `epsilon_spent <- context.accountant.map(context.d_in)`; assert it is at most EPS.
6. Post-processing, free: clip at 0, combine every noisy estimate of a shared one-way margin, rescale to the public n, fit each two-way table to its margins (iterative proportional fitting).
7. Synthetic rows: sample a root column from its noisy one-way table, then each child column from the noisy conditional table of its parent (a tree chosen from public knowledge, or the maximum mutual-information tree over noisy pair tables).

## Privacy accounting
A changed row moves between two cells in EVERY table, so the budget is split over the M tables, and `unit_of(changes=1)` makes OpenDP charge the replace-one cost of each group-by count. Read epsilon from the Context accountant and assert it. Steps 6-7 use only the noisy tables, so they cost nothing.

## Pitfalls
- Keys from the observed data (`group_by` without `with_keys`): absent categories then leak.
- One query per cell, or a full EPS per table: a row is in every table, so the per-table costs add up.
- Polars Int8/Int16 columns fail in the Context; cast category codes to String.
- MST, AIM, PrivBayes and private-pgm style generators (graphical-model fitting, adaptive marginal selection) are not in OpenDP; they need other libraries (smartnoise-synth, private-pgm). Keep to independent or tree models here.
- More tables means less budget each: release only the marginals the analysis needs.
