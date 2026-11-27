---
id: numpy-two-by-two-table-odds-ratio
title: 2x2 contingency table counts and an odds ratio or relative risk from one Laplace histogram in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - math.erfc
  - numpy.random.default_rng
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release the four cells of a 2x2 contingency table (exposure yes/no by outcome yes/no) with pure ε-DP, and statistics computed from them: odds ratio, relative risk, difference in proportions. Default neighbours: replace-one (bounded DP, n public).

## Steps
1. Exact cells c11, c10, c01, c00 (exposure x outcome), from public levels of both variables.
2. L1 sensitivity of the 4-cell vector: 1 under add/remove, 2 under replace-one. `scale <- L1 / EPS`.
3. `noisy <- cells + rng.laplace(0, scale, size=4)` with `rng <- numpy.random.default_rng()` (no seed).
4. Post-processing: clamp at 0; odds ratio `((c11+0.5)(c00+0.5)) / ((c10+0.5)(c01+0.5))`, relative risk `((c11+0.5)/(c11+c10+1)) / ((c01+0.5)/(c01+c00+1))`, both from the clamped noisy cells with the same correction.
5. Noise-aware significance of the odds ratio: with v the variance of the noise on each cell (Laplace: 2·scale²) and corrected cells c+0.5, `SE(log OR) <- sqrt(Σ 1/c + Σ v/c²)` (delta method); `p <- erfc(|log OR| / SE / √2)`. Apply any multiple-testing correction (e.g. Bonferroni) to these p-values.

## Privacy accounting
Replace-one (bounded DP, n public) by default: L1 = 2 (1 only under add/remove). NumPy has no accountant: epsilon_spent = L1 / scale; state the relation next to it and assert it is <= EPS. The ratios use only the released cells, so they are free (Dwork & Roth 2014, Prop. 2.1).

## Pitfalls
- The classical Wald SE sqrt(Σ 1/c) alone: when cells are small the noise dominates and far too many odds ratios look significant.
- Under replace-one (n public) the table's L1 sensitivity is 2: one changed row leaves one cell and enters another.
- Four separate count queries at ε each spend 4ε; one histogram over the four cells spends ε.
- Noisy cells can be 0 or negative: clamp at 0 and add a continuity correction (0.5) before forming ratios, and say so.
- Printing the exact table or exact odds ratio next to the private one.
- The two levels of each variable must be public (e.g. value / not value), not read from the data.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
