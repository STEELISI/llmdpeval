---
id: opendp-alp-sparse-counts-queryable
title: Approximate Laplace Projection - pure-DP counts for an unknown, large key space
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.then_count_by
  - opendp.measurements.make_alp_queryable
  - opendp.measurements.then_alp_queryable
  - opendp.domains.map_domain
  - opendp.metrics.l01inf_distance
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/approximate-laplace-projection.rst
source_license: MIT
---

## Goal
Release a DP sketch of key counts when the keys are private and the key space is huge (names, IDs, free text). The sketch can then be asked about any key. Unlike thresholding, this is pure epsilon-DP with no delta. Neighbours: replace-one (bounded DP, n public) = symmetric distance 2.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`
3. `counts <- space >> then_count_by()`: a map key -> count under `l01inf_distance(absolute_distance(T=int))` (sensitivity triple l0, l1, linf).
4. `meas <- binary_search_chain(lambda s: counts >> then_alp_queryable(scale=s, total_limit=TOTAL, value_limit=VMAX), d_in=2, d_out=EPS)`
5. `sketch <- meas(VALUES)`: a queryable.
6. `sketch(KEY)` returns a noisy count for any KEY, including keys never seen (usually near 0).

Signature for reference: `make_alp_queryable(input_domain, input_metric, scale, total_limit, value_limit=None, size_factor=50, alpha=4)`.

## Privacy accounting
Replace-one (bounded DP): `meas.map(2)` gives epsilon for one changed row (one removal plus one addition); `map(1)` would be add/remove-one and half of that. Asking the sketch more keys is post-processing and costs nothing extra.

## Pitfalls
- `total_limit` (bound on the sum of counts) and `value_limit` (bound on any one count) must be public; they size the hash space.
- Answers are coarse: they move in steps of `alpha / scale` (default alpha 4). Here `scale` means epsilon/sensitivity, the inverse of a Laplace noise scale, so a bigger scale spends more epsilon; let the search pick it.
- Too small a projection (small `size_factor`) causes hash collisions between frequent keys.
- If a public key list exists, the plain histogram (`opendp-histogram-count-by-categories`) is simpler and more accurate.
- To list which keys exist, use a threshold mechanism instead; ALP answers only keys you ask about.
