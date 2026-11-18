---
id: opendp-gaussian-threshold-unknown-keys
title: Counts per private key set with Gaussian noise and a threshold (make_gaussian_threshold)
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: histogram
dp_notion: approx
needs_bounds: false
api_calls:
  - opendp.transformations.then_count_by
  - opendp.measurements.make_gaussian_threshold
  - opendp.metrics.l02inf_distance
  - opendp.metrics.absolute_distance
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.mod.binary_search
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/thresholded-noise-mechanisms.rst
source_license: MIT
---

## Goal
Release a noisy count for each key that appears in the data when there is no public key list, using Gaussian noise. Keys whose noisy count falls below a threshold are dropped, so rare keys stay hidden. Result is (epsilon, delta)-DP. Neighbours: replace-one (bounded DP, n public) = symmetric distance 2.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`.
2. `counts <- space >> then_count_by()`; `(l0, l1, li) <- counts.map(2)` (replace-one: two keys change by 1 each).
3. The Gaussian threshold reads a (l0, l2, linf) triple under `l02inf_distance(absolute_distance(T=int))`, while `count_by` reports (l0, l1, linf). Derive `D_MID <- (l0, min(l1, sqrt(l0) * li), li)`.
4. `make_g <- lambda s, t: make_gaussian_threshold(counts.output_domain, l02inf_distance(absolute_distance(T=int)), scale=s, threshold=t)`; its map returns (rho, delta_threshold).
5. `make <- lambda s, t, d: make_fix_delta(make_zCDP_to_approxDP(make_g(s, t)), delta=d)`
6. Search one part at a time: `S <- binary_search(lambda s: make(s, BIG_T, DELTA/2).map(D_MID)[0] <= EPS, bounds=(LO, HI))`, then `T <- binary_search(lambda t: make_g(S, t).map(D_MID)[1] <= DELTA/2, bounds=(2, T_MAX))`.
7. `meas <- make(S, T, DELTA)`; `released <- meas(counts(VALUES))`.

## Privacy accounting
Replace-one (bounded DP): `D_MID` comes from `counts.map(2)`, so the L2 part is min(l1, sqrt(l0)·linf) = 2 as OpenDP reports it (the true replace-one L2 change is sqrt(2); OpenDP is conservative). Assert `meas.map(D_MID)` returns epsilon <= EPS and delta <= DELTA. Half of delta covers keys leaking past the threshold and half the Gaussian tail; the fixed total delta passed to `make_fix_delta` includes both.

## Pitfalls
- Chaining `counts >> then_gaussian_threshold(...)` fails: count_by outputs `L01InfDistance`, the Gaussian threshold needs `L02InfDistance`. Build it on `counts.output_domain` and apply it to `counts(data)` as shown.
- Searching for (eps, delta) tuples with `binary_search_chain` fails ("unknown ordering"); search each part.
- Very small scales overflow in the map ("inf_exp() is not finite"); keep the lower search bound reasonable.
- A Laplace version is in `opendp-count-by-unknown-keys-threshold`.
