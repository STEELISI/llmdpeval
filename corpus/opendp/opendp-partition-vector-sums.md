---
id: opendp-partition-vector-sums
title: Several bounded sums for thousands of disjoint partitions as one OpenDP Laplace release on an L1 vector
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.l1_distance
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - numpy.add.at
  - numpy.polyfit
  - pandas.DataFrame.to_numpy
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/additive-noise-mechanisms.rst
source_license: MIT
---

## Goal
Release, for every partition in a public list (areas, sites, sub-areas; thousands), the sums of several non-negative columns: one-hot group flags, flag x outcome, soft membership probabilities and probability x outcome. Rates per group, differences between estimators, error metrics, per-region regressions on partition shares and coarser regions are all computed afterwards from this one release. Replace-one row (bounded DP, n public; all group and partition sizes private).

## Steps
1. Public: partition list, value columns, row L1 bound C (one per block of one-hot flags or probabilities), missing values -> 0.
2. Per row: writable float copy (`to_numpy(dtype=float, copy=True)`), NaN -> 0, clip at 0, scale rows with L1 > C down to C; sum per partition (`add.at`) into a partitions x columns matrix.
3. `space <- (vector_domain(atom_domain(T=float, nan=False)), l1_distance(T=float))`; `scale <- binary_search_param(lambda s: space >> then_laplace(scale=s), d_in=2C, d_out=EPS)`; `meas <- space >> then_laplace(scale=scale)`; `noisy <- meas(flattened matrix as a list of float)`, reshape.
4. Post-processing: ratios with a public floor, clip to valid ranges, regions = sums of partitions, estimators and errors from noisy columns only. A regression within a region on partition shares (ecological / Goodman regression: outcome rate on group share, group rate = intercept + slope) is fitted on noisy rates and shares; skip partitions whose noisy total is below a public floor.
5. The polars Context alternative (`group_by(PARTITION).agg(dp.sum(...))` with public keys) works too, but splits the budget over the columns and rejects Int8/Int16 columns ("No match for concrete type i8"): cast to Int32/Int64/Float64 on the LazyFrame before building the Context.

## Privacy accounting
The aggregation runs outside OpenDP, so the input distance is argued by hand: one row in one partition adds at most C; replace-one removes it and adds another (maybe in another partition): `d_in = 2C` in L1. OpenDP calibrates the noise: `eps_spent <- meas.map(2C)`, assert <= EPS.

## Pitfalls
- An exact group statistic merged back onto rows (a partition mean or share attached to every row, then noised as a row value): one person changes it for the whole group, so the per-row bound fails. Only the row's own values may enter its vector.
- `then_laplace` on `(vector_domain(atom_domain(T=int)), symmetric_distance())`: "No match for concrete type SymmetricDistance". Noise on an aggregated vector needs the L1 space above.
- `dp.l1_distance()` without `T`, `atom_domain(T=float)` with NaN allowed ("Domain contains nan" in `then_clamp`), a sum without bounds ("must be bounded"; inside a search it shows as "Predicate in binary search always raises"): read the FIRST error line.
- `dp.t.then_map` does not exist; do per-row work in pandas/numpy before the release.
- A measurement per cell with EPS/cells: huge noise and timeouts on large data.
- Noise on each final metric with a guessed sensitivity instead of releasing the sums.
