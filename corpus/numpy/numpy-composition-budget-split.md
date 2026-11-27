---
id: numpy-composition-budget-split
title: Splitting a total epsilon across several queries (sequential composition) in NumPy
framework: numpy
framework_version: "any"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.isclose
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Answer several DP queries on the same dataset while keeping the **total** privacy loss at a stated ε. Basic sequential composition: k mechanisms with ε_1..ε_k are together (Σ ε_i, 0)-DP (Dwork & Roth Cor. 3.15; Thm. 3.16 for (ε_i, δ_i)). Neighbouring relation: replace-one (n public).

## Steps
1. List every release the task asks for: each count, sum, mean, selection. A mean built from a noisy sum and a noisy count is two releases.
2. Choose a split EPS_LIST with Σ = ε before touching the data. Equal split is the default; give more to queries that need accuracy.
3. Check it: `assert numpy.isclose(sum(EPS_LIST), EPS)` (or ≤ with a small tolerance) and print the split.
4. Run query i with its own share: `rng.laplace(loc=0, scale=?)` with scale = Δ_i / ε_i, Δ_i its sensitivity under replace-one (a clamped sum: replace-one U − L; add/remove (n private) max(|L|, |U|)).
5. Optional: a small ledger (running total that raises when it would exceed ε), so later edits cannot quietly overspend.

## Privacy accounting
Replace-one by default. Total loss = Σ ε_i, whatever the queries are and whichever columns they touch. Post-processing of released values (dividing a noisy sum by a noisy count, sorting, rounding) is free (Prop. 2.1). Parallel composition (McSherry 2009, PINQ): queries on **disjoint** partitions defined by a public rule cost the maximum, not the sum — under add/remove. Under replace-one a changed record can move between two partitions, so charge 2ε or use sequential composition. Advanced composition (Dwork & Roth Thm. 3.20) needs δ > 0 and helps only for many queries; at δ = 0, basic composition is the tool.

## Pitfalls
- Giving every query the full ε when the task states a total.
- Hidden queries: a noisy count used as a denominator, a DP step that picks bounds or candidates.
- Re-running a mechanism until the output looks good: every run is a new release.
- Reusing one noise draw for two queries: q1 + n and q2 + n reveal q1 − q2 exactly. Each query needs a fresh draw.
- Splitting ε over the bins of one histogram: the bins form one vector query (see `numpy-laplace-histogram-public-categories`).
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
