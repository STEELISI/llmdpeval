---
id: numpy-marginals-consistent-nonnegative
title: Counts per category and two-way cross-tabulations released together under one epsilon, made consistent (non-negative, matching margins and totals) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.ravel_multi_index
  - numpy.bincount
  - numpy.maximum
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release counts per category of several categorical columns and two-way cross-tabulations of column pairs (one-way and two-way marginal tables, public category lists) together under one epsilon, for replace-one neighbours (bounded DP, n public), then post-process them so every cell is non-negative, every table sums to n, and tables that share a column agree on that column's margin. Laplace mechanism on a vector query (Dwork & Roth 2014, Thm. 3.6) plus post-processing (Prop. 2.1).

## Steps
1. Public inputs: marginals as column tuples `[(A,), (B,), (A, B), (B, C)]`, a category list per column, the public n.
2. Exact counts per table: column values -> list indices, `ravel_multi_index`, `bincount(minlength=K)`, reshape.
3. One scale for every cell of every table: `scale <- 2*M / EPS` (M tables; replace-one moves two cells per table).
4. `noisy[t] <- exact[t] + laplace(0, scale, shape)`.
5. From here on use only `noisy`. For each column C: collect its direct one-way table and the row or column sums of each two-way table with C. Average them with weights 1/K_other (a margin summed over K_other cells has K_other times the variance).
6. Clip the average at 0 and rescale it to sum to n: this is the target margin of C.
7. Each two-way table: clip at a small positive floor, then iterative proportional fitting: alternately scale rows to the target of the first column and columns to the target of the second, for a fixed number of rounds.
8. Report the targets as the one-way tables and the fitted tables as the two-way tables.

## Privacy accounting
Under replace-one a changed row moves two cells in each of the M tables, so the stacked L1 sensitivity is 2M (M under add/remove). `epsilon_spent <- 2*M / scale`; assert it is at most EPS. Steps 5-8 never read the data, so they cost nothing.

## Pitfalls
- Rescaling to the exact n when n is private: then n needs its own noisy count.
- Clipping at 0 alone biases totals upward; rescale after clipping.
- Rounding before fitting breaks consistency; round last, if at all.
- A plain average of margins ignores that summed margins are noisier.
- Least-squares consistency over many overlapping tables, MST/AIM and private-pgm graphical models need other libraries; this fitting only handles one-way plus two-way tables.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
