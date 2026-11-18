---
id: numpy-marginals-contingency-tables
title: Several two-way contingency tables (marginals) under one epsilon in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.zeros
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release m two-way contingency tables (cross-tabulations of pairs of categorical columns, each with public category lists) with ε-DP. Every record falls in one cell of EACH table, so the stacked vector of all cells has L1 sensitivity 2m under replace-one (bounded DP, n public; the default) and m under add/remove. Laplace mechanism on a vector query (Dwork & Roth 2014, Def. 3.3, Thm. 3.6).

## Steps
1. Public pairs of columns and public category lists for each column (zeros included; values outside the lists go to an explicit "other" category or are dropped).
2. For each pair, exact cell counts as a K_a × K_b array.
3. L1 sensitivity of all tables together: Δ₁ = m under add/remove (one cell per table moves by 1); 2m under replace-one.
4. Same scale for every cell of every table: scale = Δ₁/ε. Equivalent view: each table gets ε/m and has sensitivity 1 (or 2).
5. Post-process if wanted (clip at 0, round, make the shared one-way margins agree).

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₁ = 2m (m only under add/remove). One vector release, ε total; or m releases at ε/m each (basic composition): the same noise either way. Compute ε_spent = Δ₁/scale and assert it. Margins summed from a noisy table are free; releasing them separately from the data again is not.

## Pitfalls
- Charging ε once per table: a record appears in all m tables, so that spends m·ε.
- Charging per cell: cells inside one table are disjoint.
- Category lists from the observed values.
- Many pairs: noise per cell grows linearly with m; release only the tables you need.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
