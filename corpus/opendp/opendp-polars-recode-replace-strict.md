---
id: opendp-polars-recode-replace-strict
title: Polars Context - recode raw codes with a public codebook (replace_strict) and count per label
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/expressions/manipulation.rst
source_license: MIT
---

## Goal
Map raw numeric or text codes to labels from a public codebook, merging unknown codes into one default label, and release counts per label. The output key set is known in advance, so it is passed as keys and the release is pure DP. Neighbours: replace-one (bounded DP, n public), set with `privacy_unit=unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; context with `privacy_unit=unit_of(changes=1)`, `loss_of(epsilon=EPS)` and `polars.Margin(max_length=N_MAX)`.
2. `CODEBOOK`: public dict {code: label}; `DEFAULT`: label for any other code; `LABELS <- sorted(set(CODEBOOK.values()) | {DEFAULT})`.
3. `q <- context.query().group_by(col(C).replace_strict(CODEBOOK, default=DEFAULT)).agg(dp.len(signed=True)).with_keys(LazyFrame({C: LABELS}))`
4. `out <- q.release().collect()`.
To keep the dtype and only rewrite some values, use `replace(old=[...], new=...)` instead.

## Privacy accounting
Recoding is row-by-row; each row still lands in exactly one label. Replace-one (bounded DP): a changed row moves between two labels (L1 sensitivity 2), charged by `unit_of(changes=1)`; one context slot. Total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Without `default=`, OpenDP turns unmapped codes into null (polars itself would raise a data-dependent error); those rows then fall outside your keys.
- `replace` does not work on categorical columns (polars remapping warning); cast first.
- OpenDP rejects recodes that need fallible casting of old or new values; keep their types consistent with the column.
- The codebook must be public documentation, not built from `unique()`.
