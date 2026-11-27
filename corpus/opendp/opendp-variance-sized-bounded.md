---
id: opendp-variance-sized-bounded
title: Variance with a public dataset size and public bounds
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: variance
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.make_variance
  - opendp.transformations.then_variance
  - opendp.transformations.then_clamp
  - opendp.transformations.then_resize
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-mean.rst
source_license: MIT
---

## Goal
Release the sample variance of a numeric column with epsilon-DP. `then_variance` needs a sized, bounded domain, like the sized mean. Neighbouring relation: replace-one (n public), so maps are read at `d_in=2`.

## Steps
1. `enable_features("contrib")`.
2. Public inputs: bounds (L, U) and size N. If n is private, first release a DP count and resize to it (as in `opendp-mean-unknown-size-resize`).
3. Space: `space <- vector_domain(atom_domain(bounds=(L, U)), size=N), symmetric_distance()`
   or, from raw values: `vector_domain(atom_domain(T=float)) >> then_clamp((L, U)) >> then_resize(size=N, constant=C)`.
4. `make <- lambda scale: space >> then_variance(ddof=1) >> then_laplace(scale)`
5. `meas <- binary_search_chain(make, d_in=D_IN, d_out=EPS)`, with `D_IN = 2` for bounded DP on a sized domain (or the `d_in` of the unsized input if you resized).
6. `dp_var <- meas(VALUES_AS_LIST)`. A standard deviation is `sqrt(max(dp_var, 0))`, as post-processing.

Signature for reference: `make_variance(input_domain, input_metric, ddof=1, S='Pairwise<T>')`.

## Privacy accounting
Read the sensitivity with `(space >> then_variance()).map(D_IN)`; it grows with (U - L)^2 / N, so wide bounds hurt a lot. Assert `meas.map(D_IN) <= EPS`. If you also release a mean on the same data, the costs add; use a compositor.

## Pitfalls
- A DP variance can come out negative. Clip at 0 after release.
- `ddof` changes the estimator, not the privacy; keep the one the task asks for.
- Data length must equal N, as for `then_mean`.
- Bounds taken from the private data's min/max break the guarantee.
