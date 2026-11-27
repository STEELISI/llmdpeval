---
id: opendp-api-change-polars-context
title: OpenDP polars/Context API changes up to 0.16.0 (Margin kwargs, summarize, signed counts)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.extras.polars.Margin
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.count
  - opendp.extras.polars.DPExpr.n_unique
  - opendp.context.Context.compositor
  - opendp.context.unit_of
source_url: https://github.com/opendp/opendp/blob/v0.16.0/CHANGELOG.md
source_license: MIT
---

## Goal
Update polars Context code written for OpenDP 0.10-0.15 so it runs on 0.16.0 without TypeErrors or warnings.

## Steps
1. Margins:
   - `dp.Margin(...)` -> `dp.polars.Margin(...)` (moved in 0.11).
   - Keyword renames (0.14): `public_info=` -> `invariant=` ("keys" or "lengths"); `max_partition_length=` -> `max_length=`; `max_num_partitions=` -> `max_groups=`.
   - Margins given as a dict `{(cols,): Margin(...)}` are deprecated (0.13); pass a list and name the grouping columns with `by=[...]`.
2. Query inspection: `query.accuracy(...)` -> `query.summarize(alpha=...)` (renamed in 0.11.1).
3. Signed counts (0.16): pass `signed=True` to every counting query, `dp.len(signed=True)`, `pl.col(X).dp.len/count/null_count/n_unique(signed=True)`. Omitting it emits a warning and returns unsigned counts with negatives clamped to 0; the default will become True.
4. Privacy units: replace-one (n public, the default) is `dp.unit_of(changes=1)`, with person ids `dp.unit_of(changes=1, identifier=pl.col(ID))`; add/remove (n private) is `dp.unit_of(contributions=K)` / `dp.unit_of(contributions=1, identifier=pl.col(ID))`; read epsilon with `context.accountant.map(context.d_in)`; per-group limits via `dp.polars.Bound(...)`.
5. Feature flag: `"floating-point"` was renamed `"idealized-numerics"` (0.15); most Context code needs only `"contrib"`.

## Privacy accounting
Unchanged by renames: read totals from `context.accountant.map(d_in)` or `context.current_privacy_loss()`.

## Pitfalls
- `Margin(max_partition_length=...)` raises TypeError in 0.16.0; it is not silently ignored.
- Clamping negative counts to 0 (unsigned) biases small counts upward.
- `dp.len()` counts frame rows; `pl.col(X).dp.len()` counts the column after any column-level filtering.
- Numpy and sklearn helpers moved too: `dp.np_array2_domain` -> `dp.numpy.array2_domain`; `dp.sklearn.PCA` -> `dp.sklearn.decomposition.PCA` (0.11).
