---
id: numpy-laplace-variance-bounded
title: Bounded variance from noisy moments with the Laplace mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: variance
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
Release the variance of a bounded numeric column with epsilon-DP by noising the first and second moments separately and combining them as post-processing. Laplace mechanism (Dwork & Roth Thm. 3.6), sequential composition (Cor. 3.15), post-processing (Prop. 2.1). This simple moment estimator is noisy for small N. Neighbouring relation: replace-one (n public) by default.

## Steps
1. Public bounds [L, U]; clamp: `x_c <- numpy.clip(x, L, U)`.
2. Centre on the public midpoint: c = (L + U)/2, r = (U − L)/2, `z <- x_c - c`. Every z lies in [−r, r] and every z² in [0, r²]. Variance is unchanged by a shift.
3. Exact statistics: `s1 <- numpy.sum(z)`, `s2 <- numpy.sum(z**2)`.
4. Sensitivities, per relation:
   - replace-one, N public (default): s1 → 2r (= U − L), s2 → r². Release two; ε1 + ε2 = ε; use the public N.
   - add/remove (alternative): s1 → r, s2 → r², count → 1. Release all three; split ε1 + ε2 + ε3 = ε.
5. Noise each: `ns1 <- s1 + rng.laplace(loc=0, scale=?)` with scale = sens(s1)/ε1; same pattern for s2 and (add/remove only) the count.
6. Combine (post-processing): m = ns1 / N̂, var = ns2 / N̂ − m², where N̂ is the noisy count (guarded ≥ 1) or the public N.
7. `var <- clip(var, 0, r²)`: any distribution on [L, U] has variance at most r². A standard deviation is √var (free).

## Privacy accounting
Total spend is the sum of the per-statistic epsilons; assert it is ≤ ε. Each statistic uses the sensitivity for the relation you state (replace-one by default); print relation, bounds and the split.

## Pitfalls
- Squaring uncentred values: the sum of squares then has sensitivity max(L², U²), far larger than r² when the range is far from 0.
- Noising `numpy.var` output with a mean's sensitivity: the variance has its own sensitivity; use the moment recipe.
- Dividing by the exact N under add/remove: it leaks N.
- Negative variance from noise: clip, don't take abs.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
