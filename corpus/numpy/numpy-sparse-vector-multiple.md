---
id: numpy-sparse-vector-multiple
title: Sparse vector technique with up to c positive answers (Lyu-Su-Li Alg. 1) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
source_url: https://arxiv.org/abs/1603.01699
source_license: original code (algorithm from cited paper)
---

## Goal
Scan a public stream of queries (sensitivity Δ each) and report which ones are above a public threshold T, stopping after c positive answers, under a total ε. Lyu, Su & Li, "Understanding the Sparse Vector Technique for Differential Privacy", VLDB 2017, Algorithm 1 (arXiv 1603.01699), which also shows why several earlier SVT variants are not private. Neighbouring relation: replace-one (n public).

## Steps
1. Public queries, public T, public cutoff c. Split ε = ε₁ + ε₂ (Lyu et al. suggest giving ε₂ the larger share when c > 1; ε/2 each is valid).
2. Threshold noise drawn ONCE: `rho <- rng.laplace(loc=0, scale=Δ/ε₁)`.
3. For each query: `nu <- rng.laplace(loc=0, scale=2cΔ/ε₂)` (use cΔ/ε₂ only if all queries are monotone, e.g. counts under add/remove). If q_i + nu ≥ T + rho, output ⊤ (index i) and increase the positive counter; else output ⊥. Stop after c positives.
4. Release only the ⊤/⊥ pattern (or the list of positive indices).

## Privacy accounting
Algorithm 1 is ε₁ + ε₂ = ε DP for any number of ⊥ answers and at most c ⊤ answers (Lyu et al., Thm. 2). Compute epsilon_spent = Δ/scale_ρ + 2cΔ/scale_ν in code and assert it.

## Pitfalls
- Query noise that does not grow with c: then it is only private for c = 1.
- Releasing noisy query values along with ⊤ (that is not SVT; answer them separately with fresh budget).
- Not stopping after c positives.
- Fresh threshold noise per query while keeping these scales: a different algorithm with a different analysis.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
