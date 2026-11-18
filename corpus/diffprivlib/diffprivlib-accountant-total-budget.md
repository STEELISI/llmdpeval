---
id: diffprivlib-accountant-total-budget
title: Enforcing a total epsilon across several queries with BudgetAccountant
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.utils.BudgetError
  - diffprivlib.tools.mean
  - diffprivlib.tools.count_nonzero
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/accountant.py
source_license: MIT
---

## Goal
Answer several DP queries on the same data under one fixed total budget, with the library refusing any query that would exceed it (sequential composition). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `BudgetAccountant(epsilon=inf, delta=1.0, slack=0.0, spent_budget=None)`

## Steps
1. Decide the split before touching the data: shares EPS_1 .. EPS_m that sum to EPS_TOTAL (equal or uneven, as the task says).
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. Every diffprivlib call gets its share and the accountant, e.g. `tools.mean(col, epsilon=EPS_1, bounds=(L, U), accountant=acc)`, `tools.count_nonzero(mask, epsilon=EPS_2, accountant=acc)`.
4. Mechanisms used directly: `acc.spend(EPS_i, 0)` after each `randomise`.
5. Before an optional extra query: `acc.remaining()` (budget left for one query) or `acc.remaining(k)` (per-query budget for k more).
6. Output at the end: `acc.total()`, `acc.spent_budget`, `len(acc)` (number of recorded spends).
7. The full public method list in 0.6.6: `spend(epsilon, delta)`, `check(epsilon, delta)`, `remaining(k=1)`, `total()`, `set_default()`, `load_default()`, `pop_default()`, plus the attributes `epsilon`, `delta`, `slack`, `spent_budget`. There is no `add` (`acc.add(...)` raises AttributeError); `check` needs both arguments (`check(0.1)` raises TypeError: missing `delta`).

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Each diffprivlib function calls `acc.check(epsilon, delta)` before computing and `acc.spend(epsilon, delta)` after. With `slack=0` the total is the plain sum of epsilons. A query that would push the total above the ceiling raises `diffprivlib.utils.BudgetError` and returns nothing, so the overspend never happens. Catch it if the script should continue.

## Pitfalls
- `BudgetAccountant()` with no arguments has an infinite budget: it tracks but never enforces.
- Forgetting `accountant=acc` on one call sends that spend to the global default accountant; `acc.total()` then under-reports.
- One accountant for the whole script. A new `BudgetAccountant()` per call, per group or per table tracks nothing about the total; a second `BudgetAccountant(epsilon=EPS_TOTAL)` for the next table spends the budget twice.
- Floating-point: equal shares can sum to a hair above the cap, e.g. 20 spends of 0.05 against 1.0, and the last call raises `BudgetError: Privacy spend of (0.05,0) not permissible`. `acc.remaining(k)` can hit the same error. Give each call `min(EPS_i, acc.remaining()[0])`, so the last one takes exactly what is left; never raise the cap to make it fit.
- The share passed is `SHARE * EPS_TOTAL`, not `1 - SHARE` or `EPS_TOTAL - SHARE`; and a loop over groups or tables charges once per iteration.
- Print `acc.total()[0]` as the epsilon spent. A hand-typed total, or a noise scale printed as epsilon, is not the accounting.
- Every tools function defaults to `epsilon=1.0`: five calls without `epsilon=` cost 5.0, not 1.0.
- Choose the split from the task, not from looking at early noisy results.
- Some calls split internally (quantile lists, `axis=`, multiclass models) but still record exactly the `epsilon` you passed.
