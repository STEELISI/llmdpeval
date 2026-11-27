---
id: diffprivlib-means-by-group
title: Mean of a bounded column in each group of a public list from noisy sums and noisy counts in diffprivlib
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.sum
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the mean of a bounded numeric column for every group in a public group list with pure ε-DP under replace-one (bounded DP, n public), when group sizes are private.

## Steps
1. Public `GROUPS` (declared before seeing the data) and public bounds (L, U) for the value column.
2. Clip, then shift values to `v - L`, which lies in [0, U - L]. diffprivlib's `sum` charges sensitivity U - L; after the shift that equals the add/remove sensitivity of one row.
3. For each group g (one accountant per group), on FULL-LENGTH arrays of length n (public): `IN_G <- (group == g)`; `noisy_sum <- sum(where(IN_G, SHIFTED, 0), epsilon=EPS/4, bounds=(0, U-L), accountant=acc_g)`; `noisy_n <- count_nonzero(IN_G, epsilon=EPS/4, accountant=acc_g)`. Never pass the group's own rows.
4. Post-processing: `mean_g <- clip(noisy_sum / max(noisy_n, FLOOR) + L, L, U)` with a public FLOOR.

## Privacy accounting
Replace-one (bounded DP, n public): a changed row can leave one group and join another, so two groups change. Each group spends EPS/2 (EPS/4 + EPS/4, sequential); groups are disjoint, and one change touches at most two of them: `epsilon_spent <- 2 * max over groups of acc_g.total()[0]` = EPS. Add/remove alternative (n private): one row is in one group, so EPS/2 + EPS/2 per group and the max alone would do; but the array length n is then private and the tools' length-based truncation leaks it, so use one Laplace-vector release with `mechanisms.Laplace` (partition-ratio-of-sums) instead of the tools.

## Pitfalls
- Never pass a private subset (one group's rows) to `tools.sum`, `tools.count_nonzero` or `tools.mean`. Behaviour of diffprivlib 0.6.6: `sum` truncates its output to `[L*size, U*size]` of the array it receives and `count_nonzero` to `[0, size]`, so on a subset the noisy count never exceeds the true group size and an empty group always returns exactly 0: not DP at any epsilon. `tools.mean` also uses the subset size as if public. Use full-length masked arrays (step 3).
- One shared accountant across groups reports the sum over groups, which ignores parallel composition; that is conservative, not wrong.
- Without the shift, `sum`'s U - L sensitivity understates the add/remove sensitivity max(|L|, |U|) when both bounds have the same sign.
- Group list from `unique()`; bounds from the data.
