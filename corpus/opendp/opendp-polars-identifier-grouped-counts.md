---
id: opendp-polars-identifier-grouped-counts
title: Polars Context - grouped counts when a person id has rows in many groups
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.extras.polars.LazyFrameQuery.truncate_per_group
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/identifier-truncation.rst
source_license: MIT
---

## Goal
Release counts per group when the privacy unit is a person id and one id can have several rows in several groups. Truncate each id to a public number of rows per group; with public keys, the number of groups one id can touch is bounded by the key count. Neighbours: replace-one person (all rows of one id change; the number of ids is public), set with `unit_of(changes=1, identifier=...)`.

## Steps
1. `enable_features("contrib")`.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1, identifier=col(ID)), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=K, margins=[polars.Margin(max_length=N_MAX)])`
3. `q <- context.query().truncate_per_group(KMAX, by=[G]).group_by(G).agg(dp.len(signed=True)).with_keys(LazyFrame({G: KEYS}))`
4. `q.summarize()`: the scale grows with KMAX times the number of keys a person could touch.
5. `q.release().collect()`.

## Privacy accounting
OpenDP derives the sensitivity from the truncation (at most KMAX rows per id per group) and the number of public keys. Replace-one person: one id's rows leave and another person's arrive, so OpenDP doubles the add/remove noise. Read the total from `context.accountant.map(context.d_in)`.

## Pitfalls
- `unit_of(contributions=1, identifier=...)` is add/remove one person; under replace-one it reports half the real epsilon.
- Without truncation, a grouped query on an identifier unit is rejected.
- Many keys and few groups per person: limiting groups with `truncate_num_groups(J, by=[G])` is the documented tool, but in the pinned 0.16.0 + polars 1.36.1 environment it failed at release (polars "Duplicate ... multiple fields" error). Declare `polars.Bound(by=[G], num_groups=J)` in the unit only if it holds by design.
- Group-by truncations must be the last truncations in the plan.
- KEYS must be public.
