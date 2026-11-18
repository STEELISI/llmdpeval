---
id: opendp-two-by-two-table-odds-ratio
title: 2x2 contingency table counts and an odds ratio or relative risk from one OpenDP noisy histogram
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - math.erfc
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
Release the four cells of a 2x2 contingency table (exposure yes/no by outcome yes/no) with pure ε-DP, and statistics computed from them: odds ratio, relative risk, difference in proportions. Neighbouring relation: replace-one (bounded DP, n public) by default.

## Steps
1. Map each row to one cell label from its two binary values: `CELLS <- ["1|1", "1|0", "0|1", "0|0"]` (exposure|outcome), a public list.
2. `make(scale) <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=CELLS) >> then_laplace(scale=scale)`; drop the last (null) entry.
3. `scale <- binary_search_param(make, d_in=2, d_out=EPS)` (replace-one = symmetric distance 2; `d_in=1` only if the task says add/remove); `cells <- meas(CELL_LABELS)`.
4. Post-processing: clamp cells at 0; odds ratio `((c11+0.5)(c00+0.5)) / ((c10+0.5)(c01+0.5))`, relative risk `((c11+0.5)/(c11+c10+1)) / ((c01+0.5)/(c01+c00+1))`, both from the clamped noisy cells with the same correction.
5. Noise-aware significance of the odds ratio: with v the variance of the noise on each cell (Laplace: 2·scale²) and corrected cells c+0.5, `SE(log OR) <- sqrt(Σ 1/c + Σ v/c²)` (delta method); `p <- erfc(|log OR| / SE / √2)`. Apply any multiple-testing correction (e.g. Bonferroni) to these p-values.

## Privacy accounting
Replace-one (bounded DP, n public): one changed row leaves one cell and enters another, so the table has L1 sensitivity 2; the map reports it from `D_IN = 2`. One histogram: assert `epsilon_spent <- meas.map(2) <= EPS`. All ratios are post-processing of the four released cells.

## Pitfalls
- The classical Wald SE sqrt(Σ 1/c) alone: when cells are small the noise dominates and far too many odds ratios look significant.
- `d_in=1` / `map(1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Four separate count queries at ε each spend 4ε; one histogram over the four cells spends ε.
- Noisy cells can be 0 or negative: clamp at 0 and add a continuity correction (0.5) before forming ratios, and say so.
- Printing the exact table or exact odds ratio next to the private one.
- The two levels of each variable must be public (e.g. value / not value), not read from the data.
