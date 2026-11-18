---
id: opendp-canonical-noise-hypothesis-test
title: Private hypothesis test and confidence interval for a proportion (canonical noise)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: inference
dp_notion: approx
needs_bounds: true
api_calls:
  - opendp.measurements.make_canonical_noise
  - opendp.extras.numpy.canonical.BinomialCND
  - opendp.extras.numpy.canonical.BinomialCND.p_value
  - opendp.extras.numpy.canonical.BinomialCND.confidence_interval
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/src/opendp/extras/numpy/canonical.py
source_license: MIT
---

## Goal
Test a hypothesis about a proportion (H0: p <= theta0) and give a confidence interval, using a DP release that supports exact private inference (Awan and Vadhan, Ann. Stat. 2023). The number of 1s in a 0/1 indicator column gets canonical (Tulap) noise. Neighbouring relation: replace-one (n public), so the privacy unit is symmetric distance 2 on a sized domain.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = the public row count).
3. `count <- space >> then_clamp((0.0, 1.0)) >> then_sum()`; `D_MID <- count.map(2)` (= 1: replacing one row moves the count of 1s by at most 1).
4. `noise <- make_canonical_noise(count.output_domain, count.output_metric, d_in=D_MID, d_out=(EPS, DELTA))`; `meas <- count >> noise`.
5. `z <- meas(INDICATOR_VALUES)`.
6. Inference is post-processing on z only:
   `cnd <- BinomialCND(estimate=z, d_in=D_MID, d_out=(EPS, DELTA), size=N)`
   `p <- cnd.p_value(THETA0, tail="right")`; `ci <- cnd.confidence_interval(ALPHA)`.

Signature for reference: `make_canonical_noise(input_domain, input_metric, d_in, d_out)`.

## Privacy accounting
The constructor takes `d_out=(EPS, DELTA)` directly and, under replace-one, `meas.map(2)` returns that pair. The p-value and interval use only the release and public parameters, so they cost nothing extra.

## Pitfalls
- `BinomialCND` assumes the true count is Binomial(N, p): each row contributes one Bernoulli trial.
- Do not run a classical binomial test on z: the noise invalidates its p-values.
- `tail="right"` tests H0 p <= theta0, `tail="left"` H0 p >= theta0, `None` two-sided.
- N is the public row count (replace-one) and must equal the number of rows, or the sized domain rejects the data. If n itself is private (add/remove-one), this item does not apply: the Binomial model needs a known N.
- Import with `from opendp.extras.numpy.canonical import BinomialCND` (needs numpy and scipy).
