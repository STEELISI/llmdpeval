---
id: opendp-polars-context-basic
title: Polars Context API - setup, select, summarize and release (row count, sum, mean)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.context.loss_of
  - opendp.extras.polars.Margin
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.sum
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.filter
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Release counts, sums and means of a table's columns with the polars-based Context API, under one total budget. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`. For a mean with public n, see `opendp-polars-mean-bounded-dp`.

## Steps
1. `enable_features("contrib")`; data is a polars LazyFrame (`scan_csv(...)` or `LazyFrame(...)`), not an eager DataFrame.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=M, margins=[Margin(max_length=MAXLEN)])`
   MAXLEN is a public upper bound on the number of rows. Sums need it (overflow and float error). Replace-one person with ids: `unit_of(changes=1, identifier=col(ID))`. Alternative, add/remove (n private): `unit_of(contributions=R)`.
3. Count rows: `q <- context.query().select(dp.len(signed=True))`. Under `changes=1` n is public and OpenDP releases it without noise; under `contributions=1` it gets noise.
4. Filtered count: `context.query().filter(COL_COND).select(dp.len(signed=True))`.
5. Sum: `context.query().select(col(X).cast(int).fill_null(C).dp.sum(bounds=(L, U)))`.
6. Mean: `select(col(X).dp.sum(bounds=(L, U)), dp.len(signed=True))` in one query, then divide after collecting. Under `changes=1` the length is exact, so this is the sized mean; under add/remove (n private) both parts are noisy.
7. Inspect first: `q.summarize(alpha=ALPHA)` shows column, aggregate, distribution, scale and accuracy.
8. Release: `df <- q.release().collect()`; read cells from `df`.

Signature for reference: `DPExpr.sum(self, bounds, scale=None)`.

## Privacy accounting
Report `context.accountant.map(context.d_in)`. Each `release()` spends one slot (EPS / M). Replace-one charges a sum U - L when L < 0 < U (add/remove charges max(|L|, |U|), half for symmetric bounds) and group-by counts at twice the add/remove noise scale; single filtered counts are charged 2 under `changes=1` (conservative). Several expressions inside one `select` share that slot; OpenDP splits it and shows the resulting scales in `summarize`. Use `summarize` to confirm the scale before releasing.

## Pitfalls
- `unit_of(contributions=1)` with n public reports half the real epsilon for group-bys and for sums whose bounds straddle 0.
- Version pin: opendp 0.16.0 serializes polars plans for one polars version (1.36.x at release time). A different polars (e.g. 2.x) fails at `Context.compositor` with a DSL schema error. `summarize` also needs pyarrow.
- Filtering inside the query is fine for counts and sums, but drops length invariants (see the bounded-DP mean outline).
- Without `fill_null`, OpenDP imputes a midpoint for sums; choose a public value explicitly.
- Bounds must be public, and `cast(int)` before summing integer-valued columns gives better accuracy.
- `dp.len(signed=True)` is the documented form for frame length.
