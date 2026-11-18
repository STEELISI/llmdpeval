---
id: opendp-fraction-of-rows-public-n
title: Fraction of rows matching a predicate (support, classifier accuracy) as a noisy count over a public n in OpenDP
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
Release the share of rows that satisfy a row-level predicate: the support of an itemset or rule, the accuracy of a fixed classifier on private test rows (prediction equals label), a hit rate. Bounded DP: the number of rows N is public and one row may be replaced.

## Steps
1. Turn each row into a 0/1 indicator with a row-level function (predicate on that row only; for accuracy, `pred == label` where the model does not depend on these rows, or was itself trained with DP on other rows).
2. `space <- (vector_domain(atom_domain(T=int), size=N), symmetric_distance())` with the PUBLIC N.
3. `make(scale) <- space >> then_clamp(bounds=(0, 1)) >> then_sum() >> then_laplace(scale=scale)`. The clamp is what makes the bounds true: OpenDP does not check declared bounds against the data.
4. Replace-one = symmetric distance 2: `scale <- binary_search_param(make, d_in=2, d_out=EPS)`.
5. `noisy_count <- meas(INDICATORS)`; `fraction <- clip(noisy_count / N, 0, 1)` (post-processing).

## Privacy accounting
The sized, clamped sum has sensitivity 1 under replace-one; let the map show it: assert `meas.map(2) <= EPS`. Dividing by the public N is free.

## Pitfalls
- Using the length of the private data as N when N is not public: then use add/remove (`d_in=1`, unsized domain) and divide by a separately released noisy count, or by a public N.
- `d_in=1` with a sized domain: replace-one is distance 2 there; the map then reports half the true epsilon.
- An accuracy computed on the training rows of a non-private model also leaks through the model; the DP count covers only this release.
- Printing the exact fraction next to the private one.
