---
id: diffprivlib-survival-curve-exit-histogram
title: Survival curves and life tables per group from one diffprivlib exit histogram
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release a survival curve (Kaplan-Meier style, one point per public time bin), at-risk counts, event counts and discrete-time hazards for each group of a public group list. Also covers a life table and year-by-year hazards. One noisy histogram over (group, exit bin, event flag) cells, then post-processing. Neighbours: replace-one (bounded DP, n public).

## Steps
1. Public inputs: time bin edges E from 0 to the public maximum follow-up (study design); the group list; event coding (1 = event, 0 = censored). Rows with time ≥ E_last are censored at the last bin.
2. Integer cell code per row: `cell <- (g*B + exit_bin)*2 + event`, with `exit_bin` from `searchsorted(E, t, "right") - 1` clipped to 0..B-1. Rows outside the group list are dropped.
3. `acc <- BudgetAccountant()`
4. `counts, _ <- tools.histogram(cell, epsilon=EPS/2, bins=K*B*2, range=(-0.5, K*B*2 - 0.5), accountant=acc)`. The range is public, so no edge is learned from the data. Output counts are non-negative integers.
5. Post-processing per group: `exits = events + censored`; `at_risk[0] = sum(exits)`; `at_risk[b] = at_risk[0] - cumsum(exits)[b-1]`; `hazard = clip(events / max(at_risk, FLOOR), 0, 1)`; `S = cumprod(1 - hazard)`.
6. Life table variant: denominator `at_risk - censored/2`. Hazard ratios / Cox model: not available in diffprivlib; a Poisson model fitted to the released table is post-processing.

## Privacy accounting
diffprivlib's histogram is calibrated for add/remove (one bin moves by 1). Under replace-one the changed person leaves one cell and enters another, so pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`; assert it is ≤ EPS. Everything after the histogram is free.

## Pitfalls
- Noising at-risk counts directly: a person is at risk in every bin up to exit, so that vector's sensitivity grows with the number of bins. Release exits and rebuild at-risk.
- One `histogram` call per bin or group with a share of ε: the cells are one release.
- `range=None`: bins from the data (PrivacyLeakWarning).
- Group sizes or the at-risk start from the exact data; use the noisy cell sums.
- Survey weights in the histogram (`weights=`): sensitivity becomes the largest weight, which diffprivlib does not account for. Drop or cap them and say so.
- Late bins with few at risk: report the noisy at-risk counts next to the curve.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
