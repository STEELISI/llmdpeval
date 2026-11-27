---
id: diffprivlib-histogram-categorical
title: Counts per category via integer codes and tools.histogram
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release a noisy count for every category of a string column (a categorical histogram) with pure epsilon-DP, using one histogram query instead of many separate counts. Neighbours: replace-one (bounded DP, n public).

## Steps
1. `CATS <- [c_1, ..., c_K]`: the category list from public knowledge (codebook, task text), in a fixed order. Not from the column's unique values or value counts.
2. `codes <- position of each value in CATS` (0 .. K-1; values not in CATS get -1).
3. `EDGES <- [-0.5, 0.5, ..., K - 0.5]` so each code has its own bin.
4. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
5. `(dp_counts, _) <- tools.histogram(codes, epsilon=EPS/2, bins=EDGES, range=(-0.5, K - 0.5), accountant=acc)`
6. Pair each count with its name from CATS. Ranking or picking the largest is post-processing.

## Privacy accounting
The histogram is one query. diffprivlib calibrates it for add/remove (sensitivity 1 per bin). Under replace-one (bounded DP, n public) one changed record leaves one bin and enters another (L1 sensitivity 2), so pass EPS/2: the guarantee is `epsilon_spent <- 2 * acc.total()[0]` = EPS. Calling `count_nonzero` once per category instead spends epsilon per call, so K categories cost K times as much for the same noise. Confirm with `acc.total()`.

## Pitfalls
- A category list read from the data leaks which rare categories exist, and its order can leak through tie-breaking.
- Code -1 (unlisted values) falls outside the range and is dropped; that is intended.
- Passing the full EPS under replace-one: the real guarantee is then 2 x EPS. Pass the full EPS only if the task states add/remove-one (n private).
- A share per category (fraction with some property) needs two histograms, numerator and denominator, each with its own share of the budget (e.g. epsilon/2 each, total epsilon); divide afterwards.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
