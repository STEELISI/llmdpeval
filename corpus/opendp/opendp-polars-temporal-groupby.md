---
id: opendp-polars-temporal-groupby
title: Polars Context - parse dates and count per year (str.to_date, dt.year, public keys)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.with_columns
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/temporal.rst
source_license: MIT
---

## Goal
Count rows per calendar year (or month, quarter, weekday) from a date stored as text. Parsing and component extraction happen inside the DP query; the list of periods is public. Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)` and `polars.Margin(max_length=N_MAX)`.
2. `q <- context.query().with_columns(col(D).str.to_date(format=FMT)).with_columns(YEAR=col(D).dt.year()).group_by("YEAR").agg(dp.len(signed=True)).with_keys(LazyFrame({"YEAR": YEARS}, schema={"YEAR": Int32}))`
3. `out <- q.release().collect()`.
Supported components (docs): date `dt.year, iso_year, quarter, month, week, weekday, day, ordinal_day`; time `dt.hour, minute, second, millisecond, microsecond, nanosecond`.

## Privacy accounting
Each row belongs to one year. Replace-one (bounded DP): a changed row can move between two years (L1 sensitivity 2), charged by `unit_of(changes=1)`; one context slot. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- `dt.year()` returns Int32; keys given as Int64 do not match. Pass a schema for the keys.
- Text that fails to parse becomes null and lands in no listed year; it is silently dropped from the keyed counts.
- YEARS must be a public range, not the years seen in the data.
- If one person has rows in several years, use an identifier unit and truncation (see `opendp-polars-identifier-grouped-counts`).
