---
id: opendp-polars-identifier-truncation
title: Polars Context - privacy unit by person id with truncate_per_group
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.extras.polars.LazyFrameQuery.truncate_per_group
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.SortBy
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/identifier-truncation.rst
source_license: MIT
---

## Goal
Protect people, not rows, when one person can own any number of rows that share an id column. The privacy unit is "all rows with one id"; the query keeps at most k rows per id, which bounds the sensitivity. Neighbours: replace-one person (all rows of one id change; the number of ids is public), set with `unit_of(changes=1, identifier=...)`.

## Steps
1. `enable_features("contrib")`.
2. `unit <- unit_of(changes=1, identifier=col(ID))`: replace-one at the person level (one id's rows swapped for another person's). `contributions=1` would be add/remove one person.
3. `context <- Context.compositor(data=LF, privacy_unit=unit, privacy_loss=loss_of(epsilon=EPS), split_evenly_over=K, margins=[polars.Margin(max_length=N_MAX)])`
4. `q <- context.query().truncate_per_group(KMAX).select(col(X).cast(float).dp.mean((L, U)))`. The truncation keeps a random sample of at most KMAX rows per id. To choose which rows: `truncate_per_group(KMAX, keep="first" | "last" | polars.SortBy(col(S)))`.
5. `q.summarize()`, then `q.release().collect()`.

## Privacy accounting
After truncation one person changes at most KMAX rows, and OpenDP sets the noise scale from that. Larger KMAX keeps more data (less bias) but adds proportionally more noise. Replace-one person: OpenDP charges one removed and one added id, so the noise is twice the add/remove noise. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1, identifier=...)` under replace-one reports half the real epsilon.
- With an identifier in the unit, a query without truncation is rejected (sensitivity unbounded).
- KMAX is a public modelling choice, not the observed maximum rows per id.
- `keep="first"` is fastest but biased if row order means something.
- For a mean over people, collapsing each id to one row first is often better (see `opendp-polars-identifier-aggregate-per-person`).
- In the pinned 0.16.0 + polars 1.36.1 environment, `truncate_num_groups(...)` failed at release with a polars "multiple fields with name" Duplicate error; prefer public keys (see `opendp-polars-identifier-grouped-counts`) or `polars.Bound` in the unit.
