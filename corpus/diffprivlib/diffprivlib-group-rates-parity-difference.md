---
id: diffprivlib-group-rates-parity-difference
title: Per-group outcome rates and the statistical parity difference from two diffprivlib histograms
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release the rate of a binary outcome (a positive prediction, an approval) in each group of a public group list, and a fairness gap from those rates (statistical parity difference = rate A minus rate B; disparate-impact ratio = rate A / rate B), with pure ε-DP under replace-one (bounded DP, n public).

## Steps
1. `GROUPS`: public list of every group value, declared before looking at the data; map each row's group to its index 0..K-1 (rows outside the list are dropped).
2. `acc <- BudgetAccountant()`; common histogram arguments: `epsilon=EPS/4, bins=K, range=(-0.5, K-0.5), accountant=acc`. The range is public, so no bin edge is learned from the data.
3. `noisy_pos, _ <- histogram(CODES_OF_ROWS_WITH_OUTCOME, ...)`; `noisy_all, _ <- histogram(CODES_OF_ALL_ROWS, ...)`.
4. Post-processing: `rate[g] <- clip(noisy_pos[g] / max(noisy_all[g], FLOOR), 0, 1)` with a public FLOOR; `spd <- rate[A] - rate[B]`.

## Privacy accounting
Replace-one (bounded DP, n public): a changed row can leave one bin and enter another, so each histogram has L1 sensitivity 2, while diffprivlib calibrates for 1. Pass EPS/4 to each histogram; each then guarantees EPS/2, and the two compose: `epsilon_spent <- 2 * acc.total()[0]`, assert <= EPS. The rates and the gap are post-processing.

## Pitfalls
- Passing EPS/2 per histogram under replace-one: the real total is 2 x EPS. EPS/2 each is right only for add/remove-one (n private).
- Omitting `range=`: the bins come from the data (PrivacyLeakWarning).
- Exact group sizes in the denominator, or exact rates printed next to private ones.
- Group list from the data (`unique()`).
- Calling `diffprivlib.tools.mean` on the outcome inside each group treats the group size as public; the group size is private.
- Small groups give mostly-noise rates; never drop groups using exact sizes.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
