---
id: opendp-mean-sized-bounded
title: Mean with a public dataset size (bounded DP)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.make_mean
  - opendp.transformations.then_mean
  - opendp.transformations.then_clamp
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-mean.rst
source_license: MIT
---

## Goal
Release the mean of a numeric column when the number of records n is public. Neighbours differ by replacing one record (bounded DP).

## Steps
1. `enable_features("contrib")`.
2. Public inputs: bounds (L, U) from domain knowledge, and N, the public dataset size.
3. Space with known size and bounds:
   `space <- vector_domain(atom_domain(bounds=(L, U)), size=N), symmetric_distance()`
   If raw values may fall outside [L, U], start from `vector_domain(atom_domain(T=float, nan=False), size=N)` and chain `then_clamp((L, U))` first. OpenDP does not verify declared bounds when the measurement is called, so out-of-range raw values would be released unprotected.
4. `make <- lambda scale: space >> then_mean() >> then_laplace(scale)`
5. `meas <- binary_search_chain(make, d_in=2, d_out=EPS)`; `D_IN = 2` because a replacement is one removal plus one addition.
6. `dp_mean <- meas(VALUES_AS_LIST)`; the list must have exactly N elements.

Signature for reference: `make_mean(input_domain, input_metric)`.

## Privacy accounting
Check the transformation's sensitivity with `(space >> then_mean()).map(2)`; it scales with the bound width divided by N. Assert `meas.map(2) <= EPS` before release and print it.

## Pitfalls
- `then_mean` divides by the domain's `size`, not by `len(data)`. A list of the wrong length gives a wrong mean with no error. Check with `space[0].member(data)` if unsure.
- `D_IN = 1` on a sized domain is not a valid neighbour (it is rounded down, so the map is near zero). Use 2.
- N must be public. If it is not, use `opendp-mean-unknown-size-resize` or release sum and count separately.
- Do not take L, U from `min()` / `max()` of the private data.
