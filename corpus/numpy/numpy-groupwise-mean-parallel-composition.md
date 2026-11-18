---
id: numpy-groupwise-mean-parallel-composition
title: Per-group bounded means with parallel composition over disjoint groups in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.maximum
source_url: https://doi.org/10.1145/1559845.1559850
source_license: original code (algorithm from cited paper)
---

## Goal
Release the mean of a bounded numeric column for every group in a public group list with ε-DP. Groups are disjoint, so each group's statistics use the full budget: parallel composition (McSherry, PINQ, SIGMOD 2009, Thm. 4). Per group: noisy clamped sum over noisy count. Default neighbours: replace-one (bounded DP, n public); group sizes stay private.

## Steps
1. Public group list and public bounds [L, U] for the value column; clamp values.
2. Per group: `s[g]` = sum of clamped values, `n[g]` = count. Zeros for empty groups.
3. Split ε = ε_sum + ε_cnt (sequential composition between the two statistics).
4. Sensitivity under add/remove (one record is in exactly one group, so only one entry of each vector moves):
   - sum vector: L1 sensitivity max(|L|, |U|);
   - count vector: 1.
   Under replace-one the record may move between groups: double both (2·max(|L|,|U|) and 2).
5. `noisy_s <- s + Lap(Δ_sum/ε_sum)` and `noisy_n <- n + Lap(Δ_cnt/ε_cnt)`, independent per group.
6. Post-process: `mean <- clip(noisy_s / maximum(noisy_n, m), L, U)` with a PUBLIC floor m ≥ 1.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ_sum = 2·max(|L|,|U|), Δ_cnt = 2, because a changed record can leave one group and join another. Parallel composition: queries on disjoint subsets cost the max of their epsilons, not the sum. Here that is exactly the statement that a histogram-shaped vector has L1 sensitivity "one record's worth". Total = ε_sum + ε_cnt. Compute it in code and assert.

## Pitfalls
- Charging ε per group (over-noising) or, worse, splitting nothing and running one query per group at ε each when groups are NOT disjoint (multi-label records).
- Group list from the data's distinct values.
- Replace-one with sensitivity "one group's worth": a replaced record can leave one group and join another.
- Using U − L as the sum sensitivity under add/remove when L < 0 < U: it is max(|L|, |U|).
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
