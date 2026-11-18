---
id: numpy-privacy-accountant-ledger
title: A small privacy budget ledger that refuses to overspend (basic composition) in NumPy
framework: numpy
framework_version: "any"
mechanism: other
task_type: composition
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Keep a running total of (ε, δ) across every DP release in a script and fail before any release that would exceed the declared budget. Basic sequential composition for (ε_i, δ_i) mechanisms: the total is (Σε_i, Σδ_i) (Dwork & Roth 2014, Thm. 3.16). NumPy has no accountant, so the script must carry one. Neighbouring relation: replace-one (n public).

## Steps
1. Create the ledger with the total budget (ε_total, δ_total), declared before any data is touched.
2. Before each release, call `spend(ε_i, δ_i, label)`: it raises if the new totals would exceed the budget, otherwise records the entry. Only then draw the noise.
3. Derive each ε_i from the actual noise scale and the sensitivity for the stated relation (ε_i = Δ/scale for Laplace; a clamped sum: replace-one Δ = U − L, add/remove (n private) max(|L|, |U|); a count of rows meeting a condition: Δ = 1 under both), not from the intended value, so the ledger records what was really spent. The total row count is public under replace-one and costs nothing.
4. At the end, print the ledger (label, ε_i, δ_i) and the totals; return the total as epsilon_spent.

## Privacy accounting
Each entry must use the sensitivity for the stated relation (replace-one by default). This is the loosest valid composition rule but needs no assumptions about the mechanisms beyond their individual (ε_i, δ_i). Tighter options: advanced composition (`numpy-advanced-composition`) for many small queries, zCDP for Gaussian noise (`numpy-zcdp-gaussian-composition`). Statistics of disjoint groups can be one vector release, but under replace-one a row can move between two groups, so that vector's sensitivity doubles (see `numpy-groupwise-mean-parallel-composition`).

## Pitfalls
- Charging the ledger after drawing noise: a failure then leaves an unaccounted release in the output.
- Recording the intended ε instead of the one implied by the scale actually used.
- Forgetting releases made for "exploration" (printing a noisy count to pick a bin width): they count.
- Re-running a mechanism until the output looks good: every run is spent budget.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
