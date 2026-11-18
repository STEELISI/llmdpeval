---
id: diffprivlib-count-nonzero
title: Count rows matching a condition with tools.count_nonzero
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/utils.py
source_license: MIT
---

## Goal
Release how many rows satisfy a predicate (column equals a given value) with pure epsilon-DP. One record changes the count by at most 1, so this holds for add/remove-one and for replace-one neighbours.

Signature: `count_nonzero(array, epsilon=1.0, axis=None, keepdims=False, random_state=None, accountant=None)`

## Steps
1. `mask <- (column == VALUE)` as a boolean array. VALUE comes from the task, not from inspecting the data.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)` holding the task's total budget.
3. `dp_count <- tools.count_nonzero(mask, epsilon=?, accountant=acc)`
   - `array`: the boolean mask (nonzero = counted).
   - `epsilon`: this query's share of the budget.
   - Leave `random_state` out: a fixed value makes the DP output deterministic.
4. The result is an integer. Internally the library calls `tools.sum` with bounds (0, 1) and an integer dtype, so the noise is geometric (discrete Laplace) and the output is clamped to [0, n].
5. Output: true count, DP count, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes the count by at most 1. Add/remove alternative (n private): the sensitivity is also 1, but `count_nonzero` truncates to [0, n] with the array length n, so use `mechanisms.Geometric` instead. `count_nonzero` calls `accountant.check(epsilon, 0)` before running and `accountant.spend(epsilon, 0)` after. Read the spend with `acc.total()` (returns `(epsilon=..., delta=...)`) or `acc.spent_budget` (list of per-query tuples). If the spend would exceed the ceiling, `diffprivlib.utils.BudgetError` is raised and no answer is returned.

## Pitfalls
- There is no `diffprivlib.tools.count`; use `count_nonzero` on a mask.
- Passing `axis=` releases one count per slice and splits epsilon evenly across them.
- Do not add your own NumPy noise on top of, or instead of, the library call.
- Without `accountant=`, the spend goes to the global default accountant, which has an infinite budget, so nothing enforces the total.
- The exact row count is not private under add/remove DP; do not publish an exact denominator next to the DP count unless n is public in the task.
- `count_nonzero` returns one int, not a tuple (`a, b = count_nonzero(...)` raises "cannot unpack non-iterable int object"), and it takes no `bounds=` argument (TypeError).
