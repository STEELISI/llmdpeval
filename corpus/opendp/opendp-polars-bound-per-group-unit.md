---
id: opendp-polars-bound-per-group-unit
title: Polars Context - privacy unit with per-group contribution bounds (dp.polars.Bound)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.context.unit_of
  - opendp.extras.polars.Bound
  - opendp.context.Context.compositor
  - opendp.extras.polars.LazyFrameQuery.group_by
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/gaussian-noise.rst
source_license: MIT
---

## Goal
Release counts per period (or any grouping) when one person has up to M rows in total but at most m rows in any one group, e.g. one survey row per quarter. Telling OpenDP both limits lowers the sensitivity of grouped queries. Neighbouring relation: replace-one person (the number of people is public), the default. `dp.polars.Bound` exists only as `contributions`, so replace-one is expressed by doubling both bounds (one person's rows removed, another's added): `per_group=2*M` and `per_group=2*m`. Alternative, add/remove-one person: the bounds as given, half the noise scale.

## Steps
1. `enable_features("contrib")`.
2. `unit <- unit_of(contributions=[polars.Bound(per_group=2*M), polars.Bound(by=[G], per_group=2*m)])` (replace-one person): M rows per person overall, m rows per person within one value of G. For add/remove-one person use `M` and `m`. A bound on how many groups one person touches: `polars.Bound(by=[G], num_groups=J)`.
3. `context <- Context.compositor(data=LF, privacy_unit=unit, privacy_loss=loss_of(epsilon=EPS) or loss_of(rho=RHO), split_evenly_over=K, margins=[polars.Margin(max_length=N_MAX)])`
4. `q <- context.query().group_by(G).agg(dp.len(signed=True)).with_keys(LazyFrame({G: PERIODS}))`
5. `q.summarize()`, then `q.release().collect()`.

## Privacy accounting
`context.d_in` is the list of Bounds; read the total with `context.accountant.map(context.d_in)`. Under replace-one the doubled bounds double the noise scale (Laplace counts: scale 8 vs 4 at M=4, m=1, epsilon 1). Under zCDP (rho) the per-group bound gives the largest gain: the OpenDP docs show the noise scale for quarterly counts dropping six-fold (sensitivity 36 vs sqrt(36)) when each person has at most one row per quarter.

## Pitfalls
- Bounds in the unit are promises about the data. If a person exceeds them, their privacy loss exceeds the guarantee. The docs recommend truncating in the query (`truncate_per_group`) unless the bound holds by design.
- `Bound(per_group=M)` with no `by` is the overall bound.
- `unit_of(changes=...)` does not accept Bound lists in OpenDP 0.16.0; double the bounds instead.
- With Laplace (epsilon) the L1 sensitivity of the grouped counts stays M (2M under replace-one); the gain is mainly for Gaussian.
- PERIODS must be public.
