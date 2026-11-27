---
id: numpy-topk-rate-two-histograms
title: Top-k groups by PROPORTION (rate, share of rows meeting a condition) from two noisy histograms in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.argsort
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release the k groups (from a public list) with the highest proportion (rate) of rows meeting a condition, with ε-DP under replace-one (bounded DP, n public). Two Laplace histograms (Dwork & Roth 2014, Thm. 3.6), sequential composition (Thm. 3.16), then post-processing (Prop. 2.1). This is the preferred route for rate scores.

## Steps
1. Public, fixed group list G (never the data's distinct values); `num[g]` = rows of g meeting the condition, `den[g]` = all rows of g, zeros included.
2. Under replace-one a changed row can leave one group and enter another (or flip the condition), so each histogram has L1 sensitivity Δ₁ = 2 (1 under add/remove). Split the budget: ε/2 per histogram, `scale <- Δ₁ / (ε/2)`.
3. `noisy_num <- num + laplace(0, scale)`, `noisy_den <- den + laplace(0, scale)` (one draw per group).
4. Post-processing: drop groups whose noisy size `noisy_den < 10 * scale`, then `rate <- noisy_num / noisy_den`, sort, keep the first k.

## Privacy accounting
Replace-one (bounded DP, n public): Δ₁ = 2 per histogram. Two releases at ε/2 each give ε in total; the ratio, the size cut-off and sorting use only released values, so they are free. The noise does not grow with k. epsilon_spent = 2 × (Δ₁/scale); assert it equals the budget.

## Pitfalls
- Scoring the RATE directly with sensitivity 1 (a rate lies in [0, 1]) and peeling at ε/k per pick: valid DP, but the noise (2k/ε) swamps differences between rates of a few percent, so the picks are close to random.
- Ranking by the count of rows meeting the condition instead of the rate: a different query.
- Tiny groups: their noisy rate can be anything; the noisy-size cut-off must be a fixed rule, not tuned on the data.
- Printing exact rates or exact sizes next to the private picks leaks them.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
