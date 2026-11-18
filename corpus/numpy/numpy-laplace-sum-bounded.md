---
id: numpy-laplace-sum-bounded
title: Bounded sum with clamping and the Laplace mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: sum
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
Release the sum of a numeric column with epsilon-DP. A raw sum has unbounded sensitivity, so each value is first clamped to public bounds [L, U]; then Laplace noise is added (Dwork & Roth Def. 3.1 l1-sensitivity, Def. 3.3, Thm. 3.6). Neighbouring relation: replace-one (n public).

## Steps
1. Choose public bounds L ≤ U from domain knowledge (units, legal ranges, survey top-codes) before looking at the data.
2. Clamp each value: `x_c <- numpy.clip(x, L, U)`. Clamping is per-record; dropping out-of-range rows is also per-record but silently changes the statistic.
3. Exact clamped sum: `s <- numpy.sum(x_c)`.
4. Sensitivity Δ of the clamped sum under replace-one (default, N public): Δ = U − L (one value moves from L to U at worst). Alternative, add/remove one record: Δ = max(|L|, |U|).
5. `dp_sum <- s + rng.laplace(loc=0, scale=?)` with scale = Δ / ε, `rng <- numpy.random.default_rng()`.

## Privacy accounting
Replace-one: Δ = U − L. One Laplace draw with scale Δ/ε gives (ε, 0)-DP only if Δ matches the relation you claim and the bounds are public. State the relation, Δ and ε in the output. If the sum feeds a mean or variance with other noisy parts, add their epsilons (Dwork & Roth Cor. 3.15).

## Pitfalls
- Bounds from the private data (min, max, percentiles) leak and make Δ data-dependent. Seen in real code: a helper that returns `data.max() - data.min()` as the sensitivity (github.com/tomedias/CSD_P2, `differential_privacy.py` @ 8f10ab5).
- Mixing up the relations; each formula is too small under the other relation in some cases:
  - replace-one with max(|L|, |U|): too small when L < 0 < U (bounds [−5, 5] need 10, not 5);
  - add/remove with U − L: too small when L > 0 or U < 0 (bounds [10, 20] need 20, not 10).
- Forgetting to clamp: noise is calibrated to [L, U] but an outlier moves the sum by far more.
- Bounds too wide add noise; too narrow bias the sum. That is a utility choice, not a privacy one.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
