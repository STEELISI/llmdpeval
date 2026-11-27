---
id: diffprivlib-two-by-two-table-odds-ratio
title: 2x2 contingency table counts and an odds ratio or relative risk from one diffprivlib histogram
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - math.erfc
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release the four cells of a 2x2 contingency table (exposure yes/no by outcome yes/no) with pure ε-DP, and statistics computed from them: odds ratio, relative risk, difference in proportions. Neighbours: replace-one (bounded DP, n public).

## Steps
1. Code each row as one cell index: `code <- 2*(1 - exposed) + (1 - outcome)`, giving 0 = (1,1), 1 = (1,0), 2 = (0,1), 3 = (0,0).
2. `acc <- BudgetAccountant()`; `cells, _ <- histogram(CODES, epsilon=EPS/2, bins=4, range=(-0.5, 3.5), accountant=acc)`: public range, one bin per cell.
3. Post-processing: clamp at 0; odds ratio `((c11+0.5)(c00+0.5)) / ((c10+0.5)(c01+0.5))`, relative risk `((c11+0.5)/(c11+c10+1)) / ((c01+0.5)/(c01+c00+1))`, both from the clamped noisy cells with the same correction.
4. Noise-aware significance of the odds ratio: with v the variance of the noise on each cell (diffprivlib's histogram adds geometric noise, variance about 2·(2/EPS)² at EPS/2; truncation at 0 biases small cells upward) and corrected cells c+0.5, `SE(log OR) <- sqrt(Σ 1/c + Σ v/c²)` (delta method); `p <- erfc(|log OR| / SE / √2)`. Apply any multiple-testing correction (e.g. Bonferroni) to these p-values.

## Privacy accounting
Replace-one (bounded DP, n public): one changed row leaves one cell and enters another, L1 sensitivity 2. diffprivlib calibrates for sensitivity 1, so pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`; assert <= EPS. Under add/remove (n private) pass EPS.

## Pitfalls
- The classical Wald SE sqrt(Σ 1/c) alone: when cells are small the noise dominates and far too many odds ratios look significant.
- Passing the full EPS under replace-one: the real guarantee is 2 x EPS.
- Four separate count queries at ε each spend 4ε; one histogram over the four cells spends ε.
- Noisy cells can be 0 or negative: clamp at 0 and add a continuity correction (0.5) before forming ratios, and say so.
- Printing the exact table or exact odds ratio next to the private one.
- The two levels of each variable must be public (e.g. value / not value), not read from the data.
