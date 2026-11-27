---
id: numpy-laplace-mean-bounded
title: Bounded mean with the Laplace mechanism in NumPy (two neighbouring relations)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.sum
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release the mean of a numeric column with epsilon-DP. Neighbouring relation: replace-one (n public) is the default (recipe A); add/remove with a private N is the alternative (recipe B). Built from the Laplace mechanism (Dwork & Roth Def. 3.3, Thm. 3.6), sequential composition (Cor. 3.15) and post-processing (Prop. 2.1).

## Steps
1. Public bounds [L, U], fixed without looking at the data; clamp: `x_c <- numpy.clip(x, L, U)`.
2. Pick the recipe that matches the relation.

   **A. Replace one record, N public (default).** Noise the mean directly.
   - Sensitivity of the clamped mean: (U − L) / N, since one value can move from L to U.
   - `dp_mean <- numpy.sum(x_c) / N + rng.laplace(loc=0, scale=?)`, scale = (U − L) / (N·ε).

   **B. Add/remove one record (N private), alternative.** Noisy sum and noisy count, then divide.
   - Split the budget: ε_sum + ε_cnt = ε.
   - `noisy_sum <- numpy.sum(x_c) + rng.laplace(loc=0, scale=?)`, scale = max(|L|, |U|) / ε_sum (sum sensitivity under add/remove).
   - `noisy_cnt <- len(x_c) + rng.laplace(loc=0, scale=?)`, scale = 1 / ε_cnt (count sensitivity 1).
   - `dp_mean <- noisy_sum / max(noisy_cnt, 1)` (post-processing).
3. Post-process: `dp_mean <- clip(dp_mean, L, U)`; the true clamped mean lies in [L, U].

## Privacy accounting
Recipe A (replace-one) spends ε once and is valid only if N is public by design, not merely observed. Recipe B spends ε_sum + ε_cnt (sequential composition); check in code that the sum is ≤ ε. Print the relation, bounds, sensitivities and epsilons.

## Pitfalls
- Using A's (U − L)/N under add/remove: N itself changes between neighbours.
- Dividing the noisy sum by the exact N under add/remove: N is released unprotected.
- Bounds from the data's min/max: a data-dependent sensitivity is not a DP guarantee.
- A noisy count near 0 or negative for small groups: guard the division and clip.
- Splitting ε and then spending the full ε on each part.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
