---
id: opendp-make-noise-by-measure
title: then_noise - one constructor that picks Laplace or Gaussian from the privacy measure
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.measurements.make_noise
  - opendp.measurements.then_noise
  - opendp.measures.max_divergence
  - opendp.measures.zero_concentrated_divergence
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/test_meas.py
source_license: MIT
---

## Goal
Write one pipeline that adds the right noise for the privacy notion the task asks for: Laplace for pure epsilon-DP, Gaussian for rho-zCDP. Neighbouring relation: replace-one (n public): sized domain, `d_in=2`.

## Steps
1. `enable_features("contrib")`.
2. `MEASURE <- max_divergence()` for pure DP, or `zero_concentrated_divergence()` for zCDP.
3. `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count); `pre <- space >> then_clamp((L, U)) >> then_sum()`
4. `meas <- binary_search_chain(lambda s: pre >> then_noise(output_measure=MEASURE, scale=s), d_in=2, d_out=BUDGET)`, where BUDGET is epsilon or rho.
5. `dp_value <- meas(VALUES)`; `meas.output_measure` confirms which notion was used.

Signature for reference: `make_noise(input_domain, input_metric, output_measure, scale, k=None)`.

## Privacy accounting
Under replace-one, `meas.map(2)` returns epsilon under `max_divergence` and rho under `zero_concentrated_divergence` (the sized sum is charged U - L). Alternative, add/remove-one (n private): unsized domain and `meas.map(1)` (sum charged max(|L|, |U|, U - L)). For (epsilon, delta) from a rho measurement, wrap with `make_zCDP_to_approxDP` and `make_fix_delta` (see `opendp-gaussian-zcdp-to-approx`).

## Pitfalls
- `then_laplace` and `then_gaussian` still exist; `then_noise` is the general form added in 0.14.
- Integer inputs get discrete noise, float inputs continuous noise; the type comes from the domain.
- Gaussian noise uses L2 sensitivity for vectors; scalar aggregates are the same in L1 and L2.
- Do not report rho as epsilon; they are different units.
