---
id: diffprivlib-summary-statistics-split
title: A describe()-style summary (mean, std, median) under one total epsilon
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.mean
  - diffprivlib.tools.std
  - diffprivlib.tools.median
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/exploration.ipynb
source_license: MIT
---

## Goal
Release several summary statistics of one numeric column (mean, standard deviation, median) with a stated total epsilon, by splitting the budget across the calls (sequential composition). Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Public bounds `(L, U)` for the column, used by every call.
2. Decide the split before touching data, e.g. shares `(s_mean, s_std, s_median)` summing to 1. Statistics you care about more get larger shares.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. Calls, all with `bounds=(L, U), accountant=acc`:
   - `dp_mean <- tools.mean(values, epsilon=s_mean * EPS_TOTAL, ...)`
   - `dp_std <- tools.std(values, epsilon=s_std * EPS_TOTAL, ...)`
   - `dp_median <- tools.median(values, epsilon=s_median * EPS_TOTAL, ...)`
5. Output each true and DP statistic, and `acc.total()` and `acc.remaining()`.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Each call checks the remaining budget before running and spends its own epsilon. `acc.total()` is the sum (basic composition with `slack=0`). If the shares add to more than 1 the accountant raises `BudgetError` on the call that would overspend, so the script cannot silently exceed the total.

## Pitfalls
- Giving every call the full epsilon spends k times the budget; with a capped accountant this raises `BudgetError`, with no accountant it silently spends on the default accountant, whose budget is infinite.
- `tools.std` and `tools.var` use `ddof=0` like NumPy; the DP output can be 0 when noise pushes it to the lower clamp.
- The guarantee assumes n is public (bounded DP). If n must be private, add a DP count and give it a share.
- Bounds from `describe()` / min / max of the data are a leak.
