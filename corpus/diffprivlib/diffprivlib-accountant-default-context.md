---
id: diffprivlib-accountant-default-context
title: Default accountant, set_default and the with-block accountant
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.BudgetAccountant
  - diffprivlib.tools.mean
  - diffprivlib.models.GaussianNB
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/accountant.ipynb
source_license: MIT
---

## Goal
Track every spend in a script without passing `accountant=` to each call, while still knowing which accountant recorded what. Neighbouring relation: replace-one (bounded DP, n public); the accountant only adds up the epsilons it is given.

## Steps
1. How a diffprivlib function finds its accountant, in priority order:
   - an explicit `accountant=` argument;
   - an accountant made active by a `with` block or `set_default()`;
   - otherwise a global default accountant with an infinite budget.
2. Block form (scoped, recommended):
   - `with BudgetAccountant(epsilon=EPS_1 + EPS_2/2, delta=0) as acc:`
     - `tools.mean(col, epsilon=EPS_1, bounds=(L, U))`
     - `models.GaussianNB(epsilon=EPS_2/2, bounds=(0, U_vec - L_vec)).fit(X - L_vec, y)` (features shifted by their public lower bounds; one placeholder row per public class appended so every class is present)
   - after the block: report `acc.total()[0] + EPS_2/2`. The previous default is restored on exit.
3. Script-wide form: `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`, then `acc.set_default()`. Later calls without `accountant=` spend on `acc`. Undo with `BudgetAccountant.pop_default()`.
4. Current default: `BudgetAccountant.load_default(None)`.

## Privacy accounting
The active accountant checks and records each spend exactly as when passed explicitly; `acc.total()` and `acc.spent_budget` show the result, and `BudgetError` is raised on overspend. Replace-one (bounded DP, n public): `tools.mean` with public n is charged correctly as recorded. GaussianNB releases per-class statistics calibrated for add/remove; a replaced row can switch class, so pass EPS_2/2 and count it twice: report `acc.total()[0] + EPS_2/2` and assert it is <= EPS_TOTAL.

## Pitfalls
- Models look up the accountant when they are constructed, not when `fit` runs. A model built outside the `with` block spends on whatever was default at construction.
- The global default persists for the whole Python process; in notebooks, spends from earlier cells accumulate.
- An explicit `accountant=` always wins over the block or default.
- Mechanism objects (`diffprivlib.mechanisms.*`) ignore all accountants; spend for them manually.
- Behaviour of diffprivlib 0.6.6: GaussianNB noises only the classes present in y (an empty class keeps a count of exactly 0); use the public class list with one placeholder row per class, as in the gaussian-nb outline.
- Add/remove alternative (n private): the library's per-class statistics are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]` (cap the accountant at EPS_1 + EPS_2); under replace-one pass epsilon/2 and report 2x. Keep the shift to lower bound 0 and the placeholder rows either way.
