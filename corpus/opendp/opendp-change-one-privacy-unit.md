---
id: opendp-change-one-privacy-unit
title: Replace-one (bounded DP) privacy unit with change_one_distance and metric conversion
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.metrics.change_one_distance
  - opendp.metrics.symmetric_distance
  - opendp.transformations.make_metric_unbounded
  - opendp.transformations.then_metric_unbounded
  - opendp.transformations.make_metric_bounded
  - opendp.transformations.then_mean
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/index.rst
source_license: MIT
---

## Goal
Release a statistic when the task defines neighbours as "same size n, one record changed" (bounded DP). State that unit directly as `change_one_distance` with d_in = 1, and let OpenDP convert it to the add/remove metric its aggregators use.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(bounds=(L, U), nan=False), size=N), change_one_distance()`. Size and bounds are declared on the domain, so the data must already be clamped with the public (L, U) and have exactly N rows.
3. `pre <- space >> then_metric_unbounded() >> then_mean()`. `then_metric_unbounded` maps change-one distance k to symmetric distance 2k.
4. `meas <- binary_search_chain(lambda s: pre >> then_laplace(s), d_in=1, d_out=EPS)`
5. `dp_mean <- meas(VALUES)`.

The reverse conversion, `then_metric_bounded()`, goes from a sized `symmetric_distance` space to `change_one_distance`.

## Privacy accounting
With change-one units, d_in = 1. The same release written with `symmetric_distance` needs d_in = 2 (one change = one removal + one addition); both give the same `meas.map(...)`. Always read epsilon from the map with the d_in that matches the unit you used.

## Pitfalls
- Aggregators such as `then_mean` do not accept `change_one_distance` directly ("No match for concrete type ChangeOneDistance"); convert first.
- `atom_domain(bounds=...)` defaults to `nan=True` for floats; set `nan=False` or the chain fails with a domain mismatch.
- N must be public. Under replace-one DP the size is public by definition.
- Mixing units (d_in = 1 with symmetric distance on a sized domain) halves the true sensitivity and overstates privacy.
