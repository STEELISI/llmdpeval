---
id: diffprivlib-pooled-effect-many-experiments
title: Per-experiment mean differences and a pooled (meta-analytic) effect with heterogeneity, from one diffprivlib release
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
  - scipy.stats.norm.sf
  - scipy.stats.chi2.sf
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT; DerSimonian & Laird 1986, Controlled Clinical Trials 7(3) (pooling and tau^2, textbook)
---

## Goal
Many disjoint sub-experiments (or units such as sites or firms), each with treated and control rows and a bounded outcome. Release each experiment's treated-minus-control difference with a noise-aware SE, a pooled effect, and a heterogeneity test that does not count DP noise as real variation. All sizes are private. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: experiment list, treated value, outcome bounds (L, U). Clamp and rescale to [0, 1].
2. Array (experiments, 2 arms, 3): count, sum x, sum x^2.
3. `mech <- Laplace(epsilon=EPS, sensitivity=6)`; `noisy <- vectorize(mech.randomise)(array)`; one `acc.spend(EPS, 0)`; `scale <- sqrt(mech.variance(0)/2)`.
4. Post-processing: `n <- max(count, 2)`, `m <- clip(sum/n, 0, 1)`, per-arm variance `sd2` pooled over all experiments (one noisy sum of squares per cell is too noisy; assumes a similar outcome variance across experiments).
5. Cell variance `sd2/n + 2*scale^2*(1 + m^2)/n^2`; per experiment `diff`, `var` rescaled by `(U - L)`.
6. `w <- 1/var`: fixed effect and SE, Cochran Q vs chi-square(E-1), DerSimonian-Laird `tau2`, random-effects estimate.

## Privacy accounting
A row adds (1, x, x^2), at most 3 in L1, to one cell; replace-one can move it to another cell: L1 sensitivity 6. Laplace at that sensitivity on every entry is one EPS-DP release; report `acc.total()`, assert <= EPS.

## Pitfalls
- `tools.mean` per experiment and arm: each call treats its cell size as public and spends its own epsilon; with 2E cells and a row that can move between them, the cost is not EPS.
- Sampling-only SEs: pooled and Q tests reject far too often; tau2 absorbs the noise.
- `tools.var` per cell at a share of EPS: far too noisy; pool the variance over experiments instead.
- Bounds from the data.
