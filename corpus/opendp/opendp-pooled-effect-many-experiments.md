---
id: opendp-pooled-effect-many-experiments
title: Per-experiment mean differences and a pooled (meta-analytic) effect with heterogeneity, from one OpenDP vector release
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.l1_distance
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - scipy.stats.norm.sf
  - scipy.stats.chi2.sf
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/additive-noise-mechanisms.rst
source_license: MIT; DerSimonian & Laird 1986, Controlled Clinical Trials 7(3) (pooling and tau^2, textbook)
---

## Goal
Many disjoint sub-experiments (or units such as sites or firms), each with treated and control rows and a bounded outcome. Release each experiment's treated-minus-control difference with a noise-aware SE, a pooled effect, and a heterogeneity test that does not count DP noise as real variation. All sizes are private. Replace-one neighbours (bounded DP, n public).

## Steps
1. Public inputs: experiment list, treated value, outcome bounds (L, U). Clamp and rescale the outcome to [0, 1].
2. Aggregate outside OpenDP: array (experiments, 2 arms, 3) of count, sum x, sum x^2.
3. `space <- (vector_domain(atom_domain(T=float, nan=False)), l1_distance(T=float))`; `scale <- binary_search_param(lambda s: space >> then_laplace(scale=s), d_in=6, d_out=EPS)`; `noisy <- meas(flattened array)`.
4. Post-processing: `n <- max(count, 2)`; `m <- clip(sum/n, 0, 1)`; pooled per-arm variance `sd2` from the sums over all experiments (one noisy sum of squares per cell is too noisy; assumes a similar outcome variance across experiments).
5. Variance per cell `sd2/n + 2*scale^2*(1 + m^2)/n^2`; per experiment `diff <- (m_t - m_c)(U - L)`, var = both cells times `(U - L)^2`.
6. Pool with `w <- 1/var`: fixed effect and SE; Cochran Q vs chi-square(E-1); DerSimonian-Laird `tau2`; random-effects estimate with `1/(var + tau2)`.

## Privacy accounting
A row adds (1, x, x^2), at most 3 in L1, to one cell; replace-one can move it to another cell, so `d_in = 6`. This bound is argued by hand because the aggregation runs outside OpenDP; OpenDP calibrates to it. `epsilon_spent <- meas.map(6)`, assert <= EPS.

## Pitfalls
- Sampling-only SEs: pooled and Q tests reject far too often and tau2 reports noise as heterogeneity.
- `then_laplace` on a symmetric-distance vector space fails; use the float L1 space above.
- `d_in = 3` (add/remove) under replace-one halves the real epsilon.
- The polars Context alternative (`group_by(cell).agg(dp.len(), x.dp.sum(...), (x*x).dp.sum(...))` with public keys, `unit_of(changes=1)`) also works; read scales from `query.summarize()`.
- Exact cell sizes, or bounds from the data.
