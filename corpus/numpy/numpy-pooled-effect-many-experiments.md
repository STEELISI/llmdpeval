---
id: numpy-pooled-effect-many-experiments
title: Per-experiment mean differences and a pooled (meta-analytic) effect with heterogeneity, from one noisy release (NumPy)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.add.at
  - scipy.stats.norm.sf
  - numpy.random.Generator.normal
source_url: https://doi.org/10.1016/0197-2456(86)90046-2
source_license: DerSimonian & Laird 1986, Controlled Clinical Trials 7(3) (pooling and tau^2, textbook); Laplace mechanism from Dwork & Roth 2014
---

## Goal
Many disjoint sub-experiments (or units such as sites or firms), each with a treated and a control arm and a bounded outcome (a rating, or 0/1). Release the treated-minus-control difference per experiment with a noise-aware SE, a pooled effect, and a heterogeneity test that does not mistake DP noise for real variation. All sizes are private. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: experiment list, treated value, outcome bounds (L, U) from the scale or codebook. Clamp the outcome and rescale it to [0, 1].
2. One array of shape (experiments, 2 arms, 3): count, sum x, sum x^2 per cell.
3. One release: `noisy <- stats + rng.laplace(0, 6/EPS, shape)` with `rng <- numpy.random.default_rng()`.
4. Post-processing: `n <- max(noisy count, 2)`; `m <- clip(sum/n, 0, 1)`. One noisy sum of squares per cell is too noisy for a variance, so pool it per arm over all experiments (assumes a similar outcome variance across experiments): `sd2 <- clip(sum q / sum n - (sum s / sum n)^2)`.
5. Per cell: sampling variance `sd2/n`, noise variance `2*scale^2*(1 + m^2)/n^2` (delta method). Per experiment: `diff <- (m_t - m_c)*(U - L)`, `var <- (sum of both cells) * (U - L)^2`.
6. Pool with `w <- 1/var`: fixed effect `sum(w*diff)/sum(w)`, SE `1/sqrt(sum w)`; Cochran `Q <- sum(w*(diff - est)^2)`; DerSimonian-Laird `tau2 <- max(0, (Q - (E-1)) / (sum w - sum w^2 / sum w))`; random effects with `1/(var + tau2)`.
7. Q's p-value by parametric bootstrap, not the chi-square table: the Laplace noise is heavy-tailed, so Q has the right mean but a longer right tail and the chi-square reference over-rejects. Simulate B null data sets from the released values (released cell sizes shrunk toward their arm mean to remove the noise spread, factor `sqrt(max(0, 1 - 2*scale^2/var(sizes)))`; the pooled control mean for every experiment; treated = control + `est`, sampling noise `normal(0, sqrt(n*sd2))`, plus fresh `rng.laplace(0, scale)` on every count and sum), recompute Q each time with steps 4-6, and report `p <- (1 + #(Q_b >= Q)) / (B + 1)`.

## Privacy accounting
A row adds (1, x, x^2), at most 3 in L1, to one cell. Replace-one can move it to another experiment or arm, so the whole array has L1 sensitivity 6. One Laplace release: `epsilon_spent <- 6/scale`, assert <= EPS. Everything after is post-processing.

## Pitfalls
- Sampling-only SEs: the pooled test and especially the Q test reject far too often, and tau2 reports noise as heterogeneity.
- The chi-square table for Q even with noise-aware variances: about 7.7% rejections at nominal 5% with large n (Laplace tails); use the bootstrap of step 7.
- A DP mean tool per experiment and arm, each at EPS: under replace-one a row can move between cells, so this costs 2*EPS, and the tools treat each cell size as public.
- Exact cell sizes or bounds taken from the data.
- Experiments with tiny noisy counts: keep them with their large noise-aware variance (they get little influence) rather than dropping them by exact size.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
