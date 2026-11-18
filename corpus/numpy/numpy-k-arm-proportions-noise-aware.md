---
id: numpy-k-arm-proportions-noise-aware
title: Outcome rates in k arms, each arm vs control, with a noise-aware z-test, confidence interval and Holm correction (NumPy)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.add.at
  - numpy.maximum.accumulate
  - statistics.NormalDist
source_url: https://doi.org/10.1561/0400000042
source_license: Dwork & Roth 2014 (Laplace mechanism, ideas paraphrased); Holm 1979, Scand. J. Stat. 6(2) (step-down correction)
---

## Goal
Compare the rate of a binary outcome across k arms of an experiment (or k groups): release the rate per arm, the difference of every arm against the control arm, a standard error and confidence interval that include the DP noise, and p-values corrected for k-1 comparisons. Arm sizes are private. Replace-one neighbours (bounded DP, total n public).

## Steps
1. Public inputs: the arm list with the control first (from the design, never from the data), the outcome value, alpha.
2. Count one k x 2 table: arm x (outcome yes, outcome no). Rows outside the arm list are dropped.
3. One release: `noisy <- table + laplace(0, scale, shape)` with `scale <- 2/EPS`. Clamp cells at 0.
4. Post-processing on noisy cells only, with `v <- 2*scale^2` (Laplace variance): `n_a <- max(yes_a + no_a, floor)`; `p_a <- yes_a / n_a`; `var_a <- p_a(1-p_a)/n_a + v*(yes_a^2 + no_a^2)/n_a^4` (sampling + noise, delta method on the ratio).
5. For each arm a != control: `diff <- p_a - p_0`; `se <- sqrt(var_a + var_0)`; `z <- diff/se`; `p <- 2*Phi(-|z|)`; CI `diff +- z_(1-alpha/2) * se`.
6. Holm: sort the k-1 p-values, multiply the i-th smallest (i = 1..m, m = k-1) by (m-i+1), take the running maximum, cap at 1.

## Privacy accounting
Replace-one: a changed row leaves one cell and enters another, so the whole table moves by at most 2 in L1. One Laplace release at scale 2/EPS: `epsilon_spent <- 2/scale`, assert <= EPS. Rates, differences, SEs, CIs and corrected p-values are post-processing.

## Pitfalls
- A two-proportion z-test or chi-square with sampling variance only: with small arms or small EPS the noise dominates and false positives rise far above alpha.
- Dividing noisy counts by the exact arm sizes: arm sizes are private unless the design fixed them publicly.
- One count query per arm or per comparison, each at EPS: that spends k*EPS. Release the one table.
- Sensitivity 1 for the table: correct for add/remove only.
- Several outcomes (e.g. several later time points) on the same rows are separate tables: split EPS across them and correct over all tests.
- If the honest test is not significant, report that.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
