---
id: opendp-group-rates-parity-difference
title: Per-group outcome rates and the statistical parity difference from two OpenDP noisy histograms
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
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
Release the rate of a binary outcome (for example a positive prediction or decision) in each group of a public group list, and a fairness gap built from those rates (statistical parity difference = rate of group A minus rate of group B; disparate-impact ratio = rate A / rate B), with pure ε-DP under replace-one (bounded DP, n public).

## Steps
1. `GROUPS`: the public list of every group value, declared before looking at the data (codebook, schema).
2. `space <- (vector_domain(atom_domain(T=str)), symmetric_distance())`; `make(scale) <- space >> then_count_by_categories(categories=GROUPS) >> then_laplace(scale=scale)`. The last output entry is the null bin (values outside GROUPS): drop it.
3. `scale <- binary_search_param(make, d_in=2, d_out=EPS/2)`: replace-one is symmetric distance 2; build two measurements `m_pos`, `m_all`.
4. `noisy_pos <- m_pos(group values of rows with the outcome)`; `noisy_all <- m_all(group values of all rows)`.
5. Post-processing on released counts only: `rate[g] <- clip(noisy_pos[g] / max(noisy_all[g], FLOOR), 0, 1)` with a public FLOOR; `spd <- rate[A] - rate[B]`; `ratio <- rate[A] / rate[B]`.

## Privacy accounting
Replace-one (bounded DP, n public): a changed row can leave one bin and enter another, so each histogram has L1 sensitivity 2, which the map reports from `D_IN = 2`. A row is in the "all" input and maybe in the "positive" input, so the two releases compose sequentially: `epsilon_spent <- m_pos.map(2) + m_all.map(2)`; assert it is <= EPS. Group sizes are not public: they come from the noisy `all` histogram. Rates, gaps and ratios computed from the released counts are free (post-processing).

## Pitfalls
- `d_in=1` / `map(1)` is add/remove-one; under replace-one it reports half the real epsilon.
- Dividing a noisy count by an exact group size, or printing exact rates next to the private ones.
- A group list from `unique()`: which groups occur is private.
- Adding fresh noise to the gap itself as well as to the rates spends budget twice for the same information.
- Small groups give mostly-noise rates; report the noisy size, never filter on exact sizes.
