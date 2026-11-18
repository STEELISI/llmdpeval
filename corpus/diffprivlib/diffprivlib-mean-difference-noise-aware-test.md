---
id: diffprivlib-mean-difference-noise-aware-test
title: Difference of two group means with private group sizes and a noise-aware significance test
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.Laplace.randomise
  - diffprivlib.mechanisms.Laplace.variance
  - diffprivlib.accountant.BudgetAccountant
  - math.erfc
  - math.sqrt
source_url: https://arxiv.org/abs/1705.09561
source_license: arXiv preprint (Barrientos et al., JCGS 2019; ideas paraphrased)
---

## Goal
Release the mean of a bounded column in two groups, their difference, and a standard error and p-value for the difference that include the DP noise variance. The group sizes are private (only the total n is public). Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: bounds `(L, U)` of the value column (codebook) and the two group labels. Clamp the value and min-max it to [0, 1].
2. Per group g compute `n_g` (rows), `S_g` (sum), `Q_g` (sum of squares). Stack the six numbers into one vector.
3. Sensitivity: one row adds (1, x, x²) to one group, at most 3 in L1. Under replace-one it can leave one group and join the other, so `SENS <- 6`.
4. `mech <- Laplace(epsilon=EPS, sensitivity=SENS)`; `noisy <- [mech.randomise(t) for t in vector]` (an L1 vector release is this noise on every entry); `acc.spend(EPS, 0)` once. `NOISE_VAR <- mech.variance(0)`.
5. Post-processing on noisy values only: floor `n_g` at a public minimum; `mean_g <- clip(S_g/n_g, 0, 1)`; sampling variance of the mean `(Q_g/n_g − mean_g²)/n_g`; DP noise variance of the mean by the delta method `NOISE_VAR·(1 + mean_g²)/n_g²`.
6. `diff <- mean_a − mean_b`; `SE <- sqrt(sum of both groups' sampling + noise variances)`; `z <- diff/SE`; `p <- erfc(|z|/√2)`. Scale mean, diff and SE back by (U − L).

## Privacy accounting
One Laplace release of a 6-vector with L1 sensitivity 6 under replace-one (n public): recorded once: `epsilon_spent <- acc.total()[0]`, assert <= EPS. Means, SE and p-value are post-processing.

## Pitfalls
- Dividing a noisy sum by the exact group size, or calling `diffprivlib.tools.mean` inside each group: its sensitivity assumes the group size is public.
- A t-test on the noisy means with sampling variance only: when a group is small the DP noise dominates and false positives rise far above the nominal level.
- Sensitivity 3 (add/remove) under replace-one: the row can move between groups.
- Printing the exact means or exact p-value next to the DP ones.
- Report "not significant" when the noise-aware test says so, even if the original analysis was significant.
