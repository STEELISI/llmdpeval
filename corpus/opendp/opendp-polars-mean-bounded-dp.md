---
id: opendp-polars-mean-bounded-dp
title: Polars Context API - mean under bounded DP with invariant lengths
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.extras.polars.Margin
  - opendp.extras.polars.DPExpr.mean
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.summarize
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Release the mean of a column when the number of rows is public (bounded DP). Only a noisy sum is released and divided by the exact, public length.

## Steps
1. `enable_features("contrib")`.
2. Do any row filtering on the LazyFrame before building the context; a filter inside the query makes the length unknown again.
3. Declare the length invariant in the margin:
   `margins=[Margin(max_length=MAXLEN, invariant="lengths")]`
4. Privacy unit for replace-one neighbours: `unit_of(changes=1)` (or equivalently `unit_of(contributions=2)`).
5. `context <- Context.compositor(data=LF, privacy_unit=UNIT, privacy_loss=loss_of(epsilon=EPS), split_evenly_over=M, margins=...)`
6. `q <- context.query().select(col(X).cast(?).dp.mean(bounds=(L, U)))`
7. `q.summarize()` must show a nonzero scale on the Sum row and scale 0 on the Length row (length is public). Then `q.release().collect()`.

Signature for reference: `DPExpr.mean(self, bounds, scale=None)`.

## Privacy accounting
Check `summarize()` before release. The Sum row's scale is the noise actually added. Setting `invariant="lengths"` means the row count is not protected; that is the bounded-DP assumption and must be stated.

## Pitfalls
- **Zero-noise trap:** with `invariant="lengths"` and `unit_of(contributions=1)`, adding or removing one row is impossible when the length is fixed. The context then charges no sensitivity and `summarize` shows a Sum scale of about 4e-9 (a float residue, not exactly 0): the mean is released with no real noise. Always pair the lengths invariant with `unit_of(changes=1)` or `contributions=2`, and check that the scale is well above 0 (a `> 0` check never fails).
- Without the invariant (unbounded DP), `dp.mean` releases a noisy sum and a noisy length; that is fine too, but is a different privacy model.
- Claiming a public n that is really private (e.g. the size after a data-dependent filter) breaks the guarantee.
- See `opendp-polars-context-basic` for the polars version pin.
