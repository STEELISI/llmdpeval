---
id: numpy-rxc-independence-test-noise-aware
title: r x c contingency table and a chi-square independence test whose null includes the DP noise (NumPy, Monte Carlo)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.random.Generator.multinomial
  - numpy.add.at
  - numpy.outer
source_url: https://arxiv.org/abs/1602.03090
source_license: arXiv preprint (Gaboardi, Lim, Rogers & Vadhan, ICML 2016; Monte Carlo idea paraphrased)
---

## Goal
Release a cross-tabulation of two categorical columns (r x c cells, public level lists), its margins, and a test of independence that stays valid after DP noise. The noise is added to the table; the test compares the observed statistic with statistics of simulated tables that get the same noise. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: both level lists (codebook; map unknown values to an explicit "other" level or drop them), alpha, the number of simulations M.
2. Count the r x c table. One release: `noisy <- table + laplace(0, 2/EPS, (r, c))`.
3. Statistic (post-processing): clamp at 0, expected `e <- row_sum * col_sum / total`, `T <- sum((t - e)^2 / max(e, 1))`.
4. Null model from released values only: `p0 <- outer(row_margin, col_margin) / total^2`, `n <- round(total)`.
5. Simulate M tables: `multinomial(n, p0)` + fresh Laplace noise of the SAME scale, same clamp, same statistic `T*`.
6. `p <- (1 + count(T* >= T)) / (M + 1)`.
7. Margins and any odds ratio or Cramér's V are computed from the same noisy table.

## Privacy accounting
Replace-one: one changed row moves one cell down and another up, L1 sensitivity 2 for the whole table. One release at scale 2/EPS: `epsilon_spent <- 2/scale`, assert <= EPS. The simulation uses only public values and released cells, so it is post-processing and costs nothing.

## Pitfalls
- `scipy.stats.chi2_contingency` or `chi2.sf(T, (r-1)(c-1))` on the noisy table: the noise adds to T, so the textbook null rejects far too often when cells are small.
- Monte Carlo floor: with M simulations the smallest p-value is 1/(M+1). Under a Bonferroni level alpha/m, M must exceed m/alpha, or no test can ever reject (and a p of 0 must never be reported).
- Simulating without the noise, or with a different scale, gives the wrong null.
- Separate one-way counts for the margins next to the table spend more budget; sum the noisy table instead.
- Level lists read from the data reveal which combinations occur.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
