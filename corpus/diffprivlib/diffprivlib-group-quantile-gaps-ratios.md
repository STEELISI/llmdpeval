---
id: diffprivlib-group-quantile-gaps-ratios
title: Several quantiles per group, quantile gaps between groups and percentile ratios (p90/p10) with diffprivlib
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.quantile
  - diffprivlib.tools.percentile
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/quantiles.py
source_license: MIT
---

## Goal
Release several quantiles (e.g. 10th, 50th, 90th percentiles) of a numeric column in each group of a public group list, then derive the gap between two groups at each quantile and each group's percentile ratio (p90/p10) or spread (p90 - p10). Neighbours: replace-one (bounded DP, n public).

## Steps
1. Public inputs: group list, `bounds=(L, U)` from domain knowledge (topcode, design limits), the list of quantiles `ALPHAS` in [0, 1].
2. For each group g: `acc_g <- BudgetAccountant()`; `x_g <- values of rows in g`;
   `q_g <- tools.quantile(x_g, ALPHAS, epsilon=EPS/2, bounds=(L, U), accountant=acc_g)`. The list is split evenly inside the call. `tools.percentile(x_g, [10, 50, 90], ...)` is the 0-100 form.
3. Post-processing: sort each `q_g` (independent draws can cross); `gap[A] = q_g1[A] - q_g2[A]`; `ratio[g] = q_g[hi] / q_g[lo]` if `q_g[lo] > 0`.

## Privacy accounting
Groups are disjoint, but under replace-one one changed row can leave group g1 and join g2, so two groups' releases change. Each group gets EPS/2; the worst case is the sum of the two largest per-group totals: `epsilon_spent <- sum of the two largest acc_g.total()[0]`, assert ≤ EPS. A single shared accountant would show K·EPS/2, which over-states the loss for K > 2 groups; under add/remove each group could take the full EPS.

## Pitfalls
- No `bounds`: PrivacyLeakWarning and bounds from the data.
- Passing 10/50/90 to `tools.quantile` raises; use fractions or `tools.percentile`.
- Calling `tools.quantile` once per quantile with the full ε each: spends k times the budget.
- A gap or ratio from exact quantiles of one group and DP quantiles of the other.
- Survey weights: `tools.quantile` has no weights; one heavy row moves a weighted rank by its weight. Drop the weights (unweighted estimand) or say it is not available.
- Group sizes are private; do not filter groups by their exact size. Small groups give near-random tail quantiles, and ratios inherit that noise.
- Wide bounds at small ε put probability on empty space; a p10 near L makes p90/p10 explode.
