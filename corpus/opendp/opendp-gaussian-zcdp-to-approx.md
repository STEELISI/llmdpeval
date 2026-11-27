---
id: opendp-gaussian-zcdp-to-approx
title: Mean or sum of a bounded column with Gaussian noise (zCDP converted to (epsilon, delta)-DP)
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: mean
dp_notion: zcdp
needs_bounds: true
api_calls:
  - opendp.measurements.make_gaussian
  - opendp.measurements.then_gaussian
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.measures.zero_concentrated_divergence
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/measure-casting.rst
source_license: MIT
---

## Goal
Release the sum and the mean of a bounded numeric column with Gaussian noise. Neighbouring relation: replace-one (n public): sized domain, `d_in = 2`; the mean is the noisy sum divided by the public n (post-processing). OpenDP's Gaussian measures privacy in rho (zero-concentrated DP). If the task asks for (epsilon, delta), convert the measurement and fix delta.

## Steps
1. `enable_features("contrib")`.
2. Build the aggregate as usual, e.g. `pre <- sized bounded space >> then_sum()` (see the sum outline).
3. zCDP release: `make_rho <- lambda scale: pre >> then_gaussian(scale)`; `meas <- binary_search_chain(make_rho, d_in=D_IN, d_out=RHO)`.
4. (epsilon, delta) release: wrap and fix delta inside the search:
   `make_ed <- lambda scale: make_fix_delta(make_zCDP_to_approxDP(pre >> then_gaussian(scale)), delta=DELTA)`
   `meas <- binary_search_chain(make_ed, d_in=D_IN, d_out=(EPS, DELTA))`
5. `dp_sum <- meas(DATA)`; `dp_mean <- dp_sum / N` (N public, no extra budget).

Signature for reference: `make_gaussian(input_domain, input_metric, scale, k=None, MO='ZeroConcentratedDivergence')`.

## Privacy accounting
- `meas.map(D_IN)` returns rho for step 3 and an `(epsilon, delta)` tuple for step 4. Assert it against your target.
- `make_zCDP_to_approxDP` alone returns a privacy profile; call `.map(D_IN).epsilon(delta=DELTA)` to read epsilon at a delta. `make_fix_delta` turns it into a plain (epsilon, delta) pair that composes with other approx-DP measurements.
- delta should be much smaller than 1/n. It is a task parameter, not something to tune on the data.

## Pitfalls
- Gaussian cannot give pure DP (delta = 0). If the task says pure epsilon-DP, use Laplace.
- Vector Gaussian uses L2 sensitivity (`l2_distance`), not L1.
- Do not convert rho to epsilon with a hand formula; use the combinators.
- In the Context API, `loss_of(rho=?)` gives Gaussian and `loss_of(epsilon=?)` gives Laplace automatically.
