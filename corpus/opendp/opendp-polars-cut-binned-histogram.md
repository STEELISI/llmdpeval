---
id: opendp-polars-cut-binned-histogram
title: Polars Context - histogram of a numeric column over public bin edges (cut + public keys)
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
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/manipulation.rst
source_license: MIT
---

## Goal
Release counts of a numeric column in bins defined by public edges (age bands, hour ranges). Because the full list of bin labels is known in advance, it is passed as public keys and the release is pure DP with no delta. Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)` and a `polars.Margin(max_length=N_MAX)`.
2. `BREAKS`: public sorted edges b0 < b1 < ... < bk.
3. Labels exactly as polars `cut` writes them (right-closed by default): `"(-inf, b0]", "(b0, b1]", ..., "(bk, inf]"`, as a `pl.Categorical` Series named like the column. Check the text with polars on a public toy series (e.g. `2.0` prints as `2`).
4. `q <- context.query().with_columns(col(X).cast(float).cut(breaks=BREAKS)).group_by(X).agg(dp.len(signed=True)).with_keys(LazyFrame([LABELS]))`
5. `out <- q.release().collect()`: one row per bin, empty bins included.
Alternative: `.cut(breaks=BREAKS).to_physical()` gives bin indices 0..k; then keys are `range(len(BREAKS) + 1)` as UInt32.

## Privacy accounting
Replace-one (bounded DP): a changed row leaves one bin and enters another, so the counts vector has L1 sensitivity 2; `unit_of(changes=1)` makes OpenDP charge that. The query uses one slot of the context split. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Without `with_keys` the bin labels would be treated as data-derived keys and need a delta.
- A label string that differs from polars' own (e.g. "2.0" vs "2") produces a key that never matches: that bin shows only noise.
- Edges from data quantiles leak; take them from public knowledge.
- For numeric binning in the Framework API use `then_find_bin` (see `opendp-histogram-count-by-categories`).
