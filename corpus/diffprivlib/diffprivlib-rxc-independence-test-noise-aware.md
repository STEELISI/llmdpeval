---
id: diffprivlib-rxc-independence-test-noise-aware
title: r x c contingency table and a chi-square independence test whose null includes the DP noise (diffprivlib, Monte Carlo)
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
  - numpy.random.multinomial
  - numpy.random.laplace
source_url: https://arxiv.org/abs/1602.03090
source_license: arXiv preprint (Gaboardi, Lim, Rogers & Vadhan, ICML 2016; Monte Carlo idea paraphrased); diffprivlib API MIT
---

## Goal
Release a cross-tabulation of two categorical columns (public level lists), its margins, and a test of independence that stays valid after DP noise: simulated null tables get the same noise as the released one. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: both level lists, alpha, number of simulations M.
2. Count the r x c table.
3. `mech <- Laplace(epsilon=EPS, sensitivity=2)`; `noisy <- vectorize(mech.randomise)(table)`; one `acc.spend(EPS, 0)`. `scale <- sqrt(mech.variance(0)/2)`.
4. Statistic: clamp at 0, expected from margins, `T <- sum((t - e)^2 / max(e, 1))`.
5. Null from released values: `p0 <- outer(row_margin, col_margin)/total^2`, `n <- round(total)`; M tables `multinomial(n, p0)` + Laplace noise with the same scale; same clamp and statistic.
6. `p <- (1 + count(T* >= T)) / (M + 1)`.

## Privacy accounting
Replace-one: one changed row moves two cells by 1 each, L1 sensitivity 2. Laplace at sensitivity 2 on every cell is one EPS-DP release; report `acc.total()`, assert <= EPS. The simulation uses only released and public values: post-processing.

## Pitfalls
- `scipy.stats.chi2_contingency` on the noisy table: the noise inflates the statistic and the textbook null rejects far too often.
- `tools.histogramdd` / `histogram2d` assume add/remove (pass EPS/2 under replace-one) and truncate noise at 0, so a simulation with plain Laplace noise would not match them.
- With M simulations the smallest p-value is 1/(M+1); under Bonferroni at alpha/m pick M > m/alpha and never print p = 0.
- Level lists from the data.
- Add/remove alternative (n private): a row is added to or removed from its cells instead of moving between them, so the L1 sensitivity is half the replace-one value used here; anything divided by n then needs a noisy n.
