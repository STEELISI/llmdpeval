---
id: opendp-count-by-unknown-keys-threshold
title: Counts per key when the key set is private (count_by + Laplace threshold)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: approx
needs_bounds: false
api_calls:
  - opendp.transformations.make_count_by
  - opendp.transformations.then_count_by
  - opendp.measurements.make_laplace_threshold
  - opendp.measurements.then_laplace_threshold
  - opendp.mod.binary_search
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/thresholded-noise-mechanisms.rst
source_license: MIT
---

## Goal
Release a noisy count for each key that appears in the data, when no public key list exists. Keys that only a few people have must not be revealed, so small noisy counts are suppressed and the guarantee becomes (epsilon, delta). Neighbours: replace-one (bounded DP, n public) = symmetric distance 2.

## Steps
1. `enable_features("contrib")`. Prefer a public category list if one exists (`opendp-histogram-count-by-categories` gives pure DP); use this only when keys are truly unknown.
2. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`
3. `counts <- space >> then_count_by()`: a map from key to count.
4. `make <- lambda scale, thresh: counts >> then_laplace_threshold(scale=scale, threshold=thresh)`
5. Search the two parameters one at a time, each against its own part of the map (epsilon depends on the scale, delta on the threshold):
   `S <- binary_search(lambda s: make(s, BIG_T).map(2)[0] <= EPS, bounds=(LO, HI))`
   `T <- binary_search(lambda t: make(S, t).map(2)[1] <= DELTA, bounds=(2, T_MAX))`
   `meas <- make(S, T)`
6. `released <- meas(VALUES_AS_LIST)`: a dict with only the keys whose noisy count passed T.

Signature for reference: `make_laplace_threshold(input_domain, input_metric, scale, threshold, k=None, MO='Approximate<MaxDivergence>')`.

## Privacy accounting
Replace-one (bounded DP): one changed row lowers one key and raises another, so evaluate at `D_IN = 2`. `meas.map(2)` returns an (epsilon, delta) pair; delta is the chance that a key held by one person is released. `D_IN = 1` only if the task says add/remove-one. Assert both parts against the target.

## Pitfalls
- `binary_search_chain(make, d_in=2, d_out=(EPS, DELTA))` fails here with "unknown ordering": both parts of the pair change with the scale, so tuples cannot be compared. Search each part separately as in step 5.
- Using `then_laplace` on `count_by` output without a threshold reveals the key set. It is not DP.
- Delta must be small relative to 1/n.
- Missing keys in the output may mean "small", not "absent".
- The polars Context does the same thing automatically when you `group_by` without `with_keys` and give a delta.
