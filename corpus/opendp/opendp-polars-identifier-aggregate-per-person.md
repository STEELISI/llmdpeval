---
id: opendp-polars-identifier-aggregate-per-person
title: Polars Context - collapse each person id to one row, then a DP mean over people
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyGroupByQuery.agg
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/bounds.rst
source_license: MIT
---

## Goal
Compute a mean over people (not rows) when each person owns many rows identified by an id. First reduce each id to one row with an ordinary aggregation, then release the DP mean. Neighbouring relation: replace-one person (the number of people is public), the default: `unit_of(changes=1, identifier=col(ID))`. Alternative, add/remove-one person (number of people private): `unit_of(contributions=1, identifier=col(ID))`, half the noise scale.

## Steps
1. `enable_features("contrib")`.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1, identifier=col(ID)), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=K, margins=[polars.Margin(max_length=N_MAX)])`
3. `q <- context.query().group_by(col(ID)).agg(col(X).mean()).select(col(X).dp.mean((L, U)))`
   - The `group_by(ID).agg(...)` is the truncation step: afterwards each person is one row.
   - Any per-person summary works inside `agg` (mean, max, sum, count).
4. `q.summarize()`, then `q.release().collect().item()`.

## Privacy accounting
After collapsing, one person changes one row, so the noise scale is the same as for one-row-per-person data, however many rows a person had. Total from `context.accountant.map(context.d_in)`. Replace-one person charges twice the add/remove scale for the same epsilon (e.g. scales 40 and 4 vs 20 and 2 for a mean with bounds (0, 10)).

## Pitfalls
- A strict `cast` inside the per-person `agg` is rejected ("Strict casting may cause data-dependent errors"); aggregate first (the mean is already float) or cast before grouping.
- The answer is a mean of per-person means, not the row-level mean; say which one the task wants.
- (L, U) bound the per-person summary, which may have a different range than raw values (e.g. a per-person sum).
- `contributions=1` with a public number of people reports half the real epsilon; use it only when the task states add/remove of a person.
- Alternative that keeps rows: `truncate_per_group` (see `opendp-polars-identifier-truncation`).
