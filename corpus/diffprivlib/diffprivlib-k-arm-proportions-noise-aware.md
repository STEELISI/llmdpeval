---
id: diffprivlib-k-arm-proportions-noise-aware
title: Outcome rates in k arms, each arm vs control, with a noise-aware z-test, confidence interval and Holm correction (diffprivlib)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.Laplace.randomise
  - diffprivlib.mechanisms.Laplace.variance
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.accountant.BudgetAccountant.spend
  - statistics.NormalDist
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT; Holm 1979, Scand. J. Stat. 6(2) (step-down correction, textbook)
---

## Goal
Compare the rate of a binary outcome across k arms (or groups): rate per arm, every arm minus the control arm, a standard error and confidence interval that include the DP noise, and Holm-corrected p-values. Arm sizes are private. Replace-one neighbours (bounded DP, total n public).

## Steps
1. Public inputs: arm list with the control first (design, not data), outcome value, alpha.
2. Count one k x 2 table (arm x outcome yes/no); rows outside the list are dropped.
3. `mech <- Laplace(epsilon=EPS, sensitivity=2)`; `noisy <- vectorize(mech.randomise)(table)`; `acc.spend(EPS, 0)` once for the whole table (`acc <- BudgetAccountant(epsilon=EPS, delta=0)`, `acc.check(EPS, 0)` before the release). Do not clamp the counts used in the test: clamping at 0 shrinks the noise variance below `v` and biases small cells up.
4. Noise variance from the mechanism: `v <- mech.variance(0)` (= 2*scale^2).
5. Post-processing on the unclamped counts: `n_a <- max(yes_a + no_a, floor)`; `p_a <- yes_a/n_a` (clip to [0, 1] only for display and inside the sampling term); `var_a <- p_a(1-p_a)/n_a + v*(yes_a^2 + no_a^2)/n_a^4`.
6. Per arm: `diff <- p_a - p_0`, `se <- sqrt(var_a + var_0)`, `p <- 2*Phi(-|diff/se|)`, CI `diff +- z_(1-alpha/2)*se`.
7. Holm over the k-1 p-values.

## Privacy accounting
Replace-one: a changed row leaves one cell and enters another, L1 sensitivity 2 for the whole table. The Laplace mechanism at sensitivity 2 on every cell is one EPS-DP release: record one spend and report `acc.total()`, assert <= EPS. Tests are post-processing.

## Pitfalls
- `diffprivlib.tools.histogram` is calibrated for add/remove and truncates noise at 0; if used, pass EPS/2, report twice the accountant total, and note that its noise variance is no longer 2*scale^2 for small cells.
- Calling `randomise` per cell and spending EPS per cell overstates the cost; spending nothing understates it. One table = one spend.
- A z-test or `chi2_contingency` with sampling variance only: false positives far above alpha when arms are small.
- Exact arm sizes in the denominators.
- Several outcomes on the same rows need separate tables and a budget split.
- Add/remove alternative (n private): a row is added to or removed from its cells instead of moving between them, so the L1 sensitivity is half the replace-one value used here; anything divided by n then needs a noisy n.
