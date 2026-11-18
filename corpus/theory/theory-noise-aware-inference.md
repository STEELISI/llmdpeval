---
id: theory-noise-aware-inference
title: Significance tests on DP outputs must count the DP noise
topic: post-processing
task_types: [inference, regression, mean, histogram]
sources:
  - "Barrientos, Reiter, Machanavajjhala & Chen 2019, JCGS 28(2), DP significance tests for regression coefficients"
  - "Wang, Kifer & Lee 2019, JPC 9(1), DP confidence intervals for empirical risk minimization"
  - "Dwork & Roth 2014, Proposition 2.1 (post-processing)"
---
## Idea
A DP estimate carries sampling error and the noise the mechanism added. A test that uses only the sampling standard error treats the noise as signal. When the noise is large, as in small groups or many dummy columns, such a test reports far too many significant results.

The fix is a noise-aware standard error. Its variance is the sampling variance plus the variance the DP noise adds to the estimate. The noise scale is public, so this is free post-processing. For a ratio or a mean of noisy parts, the delta method gives the noise term. For a coefficient solved from noisy moments, redraw noise of the same scale many times, re-solve, and take the variance.

## Why it matters when writing DP code
- Keep every noise scale, or read it from the framework.
- Report the estimate, its noise-aware standard error and p-value.
- If the honest test is not significant, say so. Do not drop the noise term to recover the original conclusion.

## Common mistakes
- A t-test or Wald test on noisy values with the textbook standard error.
- Adding fresh noise to the p-value instead of deriving it from released values.
- Using exact counts, residuals or group variances in the variance formula: a standard error, p-value or R² is a release too.
- Hard-coded standard errors or p-values.
- A simulated null with M draws cannot give p below 1/(M+1): M must exceed m/α for m Bonferroni tests, or nothing can be significant. Never report p = 0.
