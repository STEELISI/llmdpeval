---
id: numpy-sparse-vector-above-threshold
title: Sparse vector technique - AboveThreshold (first query above a threshold) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Given a public, ordered stream of queries (each with sensitivity Δ) and a public threshold T, report the index of the first query whose answer is (approximately) above T, paying ε once no matter how many queries were below. AboveThreshold, Dwork & Roth 2014, Algorithm 1 and Theorem 3.23. Neighbouring relation: replace-one (n public).

## Steps
1. Public query list q_1, q_2, ... (fixed in advance or chosen adaptively from earlier outputs), each with sensitivity ≤ Δ; public threshold T.
2. Noisy threshold, drawn once: `T_hat <- T + rng.laplace(loc=0, scale=2Δ/ε)`.
3. For each query in order: `nu <- rng.laplace(loc=0, scale=4Δ/ε)`; if q_i(data) + nu ≥ T_hat, output i and STOP. Otherwise output "below" and continue.
4. If no query passes, report that none did.

## Privacy accounting
The whole run is ε-DP (Thm. 3.23): ε/2 for the threshold noise and ε/2 for the query noise, independent of how many "below" answers were given. It answers ONE "above" and then halts. Compute epsilon_spent from the two scales as Δ/scale_T + 2Δ/scale_q (ε/2 + ε/2) and assert it.

## Pitfalls
- Continuing after the first "above" with the same noisy threshold: that needs the multi-answer variant (`numpy-sparse-vector-multiple`) with its own scaling.
- Releasing the noisy answer q_i + nu: SVT releases only the index. Releasing the value needs a separate Laplace query with its own ε.
- Re-drawing the threshold noise for every query, or using the same noise scale for threshold and queries: several published SVT variants with such changes are not private (Lyu, Su & Li, VLDB 2017).
- Choosing T from the data.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
