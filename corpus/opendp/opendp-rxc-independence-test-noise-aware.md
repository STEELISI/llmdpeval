---
id: opendp-rxc-independence-test-noise-aware
title: r x c contingency table and a chi-square independence test whose null includes the DP noise (OpenDP, Monte Carlo)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - numpy.random.multinomial
  - numpy.random.geometric
source_url: https://arxiv.org/abs/1602.03090
source_license: arXiv preprint (Gaboardi, Lim, Rogers & Vadhan, ICML 2016; Monte Carlo idea paraphrased); OpenDP API MIT
---

## Goal
Release a cross-tabulation of two categorical columns (public level lists), its margins, and a test of independence that stays valid after DP noise: simulated tables under the null get the same noise as the released one. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: both level lists, alpha, number of simulations M.
2. `CELLS <- [a + "|" + b for a in ROWS for b in COLS]`; label each row the same way.
3. `make(scale) <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=CELLS) >> then_laplace(scale=scale)`; `scale <- binary_search_param(make, d_in=2, d_out=EPS)`.
4. `noisy <- meas(labels)[:-1]` reshaped to r x c (the null bin holds rows outside the lists).
5. Statistic: clamp at 0, expected from margins, `T <- sum((t - e)^2 / max(e, 1))`.
6. Null from released values: `p0 <- outer(row_margin, col_margin)/total^2`, `n <- round(total)`. For each of M sims: `multinomial(n, p0)` plus discrete Laplace noise of the same scale (`geometric(q) - geometric(q)` with `q = 1 - exp(-1/scale)`, the integer noise `then_laplace` adds to counts); same clamp and statistic.
7. `p <- (1 + count(T* >= T)) / (M + 1)`.

## Privacy accounting
Replace-one = symmetric distance 2; the histogram's L1 change is 2. `epsilon_spent <- meas.map(2)`, assert <= EPS. The simulation reads only released cells and public values: post-processing.

## Pitfalls
- The textbook chi-square null (`chi2.sf(T, (r-1)(c-1))`) on noisy cells: rejects far too often when cells are small.
- With M simulations the smallest p-value is 1/(M+1); under a Bonferroni level alpha/m choose M > m/alpha. Never report p = 0.
- Simulating noise with a different scale or without noise.
- `map(1)` is add/remove; replace-one needs `map(2)`.
- Separate count queries for each margin spend extra budget; sum the noisy table.
