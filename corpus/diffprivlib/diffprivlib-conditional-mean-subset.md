---
id: diffprivlib-conditional-mean-subset
title: Mean of a numeric column within a subset of rows, as noisy sum over noisy count
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.sum
  - diffprivlib.tools.count_nonzero
  - diffprivlib.tools.mean
  - diffprivlib.accountant.BudgetAccountant
  - numpy.where
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release the mean of a numeric column among the rows that satisfy a condition (for example rows in one public category), with pure epsilon-DP, without treating the subset size as public. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Public bounds `(L, U)` for the value column with `L <= 0 <= U` (shift the column first if needed, and shift back at the end).
2. Row mask from the condition: `mask <- condition(df)`.
3. Masked values over ALL rows: `masked <- where(mask, clip(values, L, U), 0)`. The array length stays n (public), and rows outside the subset contribute 0.
4. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`; split `EPS_TOTAL` into `eps_sum + eps_count`.
5. `dp_sum <- tools.sum(masked, epsilon=eps_sum, bounds=(L, U), accountant=acc)`
6. `dp_count <- tools.count_nonzero(mask, epsilon=eps_count, accountant=acc)`
7. Post-processing: `dp_mean <- dp_sum / max(dp_count, 1)`, then clip to `(L, U)`.
8. Output true subset mean, DP mean, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): the masked array keeps length n, one replaced row moves the masked sum by at most U - L (0 lies in the bounds) and the count by at most 1. Add/remove alternative (n private): the tools truncate with the array length n, so use `mechanisms.Laplace` / `Geometric` on the sum and count instead. Two releases on the same rows: total = eps_sum + eps_count, recorded by the accountant (`acc.total()`). The ratio is post-processing.

## Pitfalls
- `tools.mean(values[mask], ...)` calibrates noise to the subset size, treating it as public. One record entering or leaving the subset is then not covered. Only use it if the subset size is public.
- Computing the subset mean exactly and adding noise to the mean alone has the same problem.
- Bounds must include 0 so that the masked-out rows are inside them; otherwise clipping moves the zeros.
- For a small subset the noisy count can be near 0; guard the division and report that the estimate is unreliable.
