---
id: opendp-polars-groupby-public-keys
title: Polars Context API - grouped counts with public keys (with_keys) or a delta threshold
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.loss_of
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
  - opendp.extras.polars.Margin
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/grouping.ipynb
source_license: MIT
---

## Goal
Release a count (or sum) per group of a categorical column. The set of group keys is itself private unless it comes from public knowledge, and OpenDP makes you choose how to handle that. Neighbouring relation: replace-one (bounded DP, n public), set with `unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; LazyFrame data; context as in `opendp-polars-context-basic` but with `privacy_unit=unit_of(changes=1)` (replace-one; OpenDP doubles the noise of add/remove).
2. Public key list (preferred, pure DP): build `KEYS <- LazyFrame({GROUP_COL: PUBLIC_CATEGORY_LIST})`, then
   `q <- context.query().group_by(GROUP_COL).agg(dp.len(signed=True)).with_keys(KEYS)`.
   Keys absent from the data get a noisy count of an empty group, so absence does not leak.
3. No public key list: give the context a delta, `loss_of(epsilon=EPS, delta=DELTA)`, and drop `with_keys`. OpenDP then releases only groups whose noisy count passes a threshold; `summarize` shows the threshold.
4. Keys as a declared invariant (`Margin(by=[GROUP_COL], invariant="keys")`) also works but removes protection from the key set; use only if the task says the keys are public.
5. `q.summarize(alpha=ALPHA)`, then `df <- q.release().collect()`.

Signature for reference: `LazyFrameQuery.with_keys(self, keys, on=None)`.

## Privacy accounting
Replace-one (bounded DP): a changed record can lower one group's count and raise another's, and `unit_of(changes=1)` makes OpenDP charge that. One grouped query is one release and costs one slot; read the total with `context.accountant.map(context.d_in)`. With no keys and no delta, OpenDP refuses to build the query ("key-set ... is private"). It does not leak silently.

## Pitfalls
- `unit_of(contributions=1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Keys from `df[col].unique()` on the private data are not public keys.
- The released frame's row order is not the key order; join or sort by the key column.
- Noisy counts of empty or small groups can be negative; clip after release if needed.
- With thresholding, small groups are missing from the output by design. Report that.
- See `opendp-polars-context-basic` for the polars version pin.
