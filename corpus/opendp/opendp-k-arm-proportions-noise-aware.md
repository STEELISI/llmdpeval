---
id: opendp-k-arm-proportions-noise-aware
title: Outcome rates in k arms, each arm vs control, with a noise-aware z-test, confidence interval and Holm correction (OpenDP)
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
  - statistics.NormalDist
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT; Holm 1979, Scand. J. Stat. 6(2) (step-down correction, textbook)
---

## Goal
Compare the rate of a binary outcome across k arms (or groups): the rate per arm, every arm minus the control arm, a standard error and confidence interval that include the DP noise, and Holm-corrected p-values. Arm sizes are private. Replace-one neighbours (bounded DP, total n public).

## Steps
1. Public inputs: arm list with the control first (design, not data), outcome value, alpha.
2. `CELLS <- ["arm|1", "arm|0" for each arm]`; each row becomes one label `str(arm) + "|" + str(int(outcome == value))`.
3. `make(scale) <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=CELLS) >> then_laplace(scale=scale)`.
4. `scale <- binary_search_param(make, d_in=2, d_out=EPS)`; `counts <- meas(labels)[:-1]` (drop the null bin, which holds rows outside the list); reshape to k x 2; clamp at 0.
5. Post-processing with `v <- 2*scale^2`: `n_a <- max(yes_a + no_a, floor)`; `p_a <- yes_a/n_a`; `var_a <- p_a(1-p_a)/n_a + v*(yes_a^2 + no_a^2)/n_a^4`.
6. Per arm a: `diff <- p_a - p_0`, `se <- sqrt(var_a + var_0)`, `p <- 2*Phi(-|diff/se|)`, CI `diff +- z_(1-alpha/2)*se`.
7. Holm over the k-1 p-values (sort, multiply the i-th smallest by k-1-i, running max, cap at 1).

## Privacy accounting
Replace-one is symmetric distance 2: one changed row leaves one cell and enters another. One histogram over all 2k cells: `epsilon_spent <- meas.map(2)`, assert <= EPS. All rates and tests are post-processing.

## Pitfalls
- `d_in=1` / `map(1)` is add/remove-one; under replace-one it reports half the real epsilon.
- A test with sampling variance only: with small arms the noise dominates and false positives far exceed alpha.
- One `then_count` per arm at EPS each spends k*EPS; one histogram over all cells spends EPS.
- Dividing by exact arm sizes: they are private unless fixed by the design.
- The polars Context (`group_by(arm, outcome).agg(dp.len())` with public keys) is an equivalent alternative; read the noise scale from `query.summarize()`.
- Several outcomes on the same rows need separate tables: split EPS and correct over all tests.
