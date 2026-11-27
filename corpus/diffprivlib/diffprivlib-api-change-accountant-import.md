---
id: diffprivlib-api-change-accountant-import
title: "Where BudgetAccountant lives: diffprivlib.accountant (singular), not accountants / accounting / privacy / utils"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.BudgetAccountant
  - diffprivlib.utils.BudgetError
  - diffprivlib.utils.Budget
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/__init__.py
source_license: MIT
---

## Goal
Import the budget accountant and its error types from the module paths that exist in diffprivlib 0.6.6. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Real 0.6.6 names (checked against the generated API index):
   - `diffprivlib.accountant.BudgetAccountant`, also re-exported as `diffprivlib.BudgetAccountant`.
   - `diffprivlib.utils.BudgetError` (also importable from `diffprivlib.accountant`).
   - `diffprivlib.utils.Budget`, the `(epsilon, delta)` tuple returned by `acc.total()` and `acc.remaining()`; it prints as `(epsilon=..., delta=...)` but has no `.epsilon` attribute.
   - `diffprivlib.utils.PrivacyLeakWarning`, `diffprivlib.utils.DiffprivlibCompatibilityWarning`.
2. Names that do NOT exist (each crashes generated code):
   - `diffprivlib.accountants` (ModuleNotFoundError; a frequent diffprivlib crash in generated code),
   - `diffprivlib.accounting`, `diffprivlib.privacy` (ModuleNotFoundError),
   - `diffprivlib.utils.BudgetAccountant` (ImportError / AttributeError: utils holds `Budget` and `BudgetError`, not the accountant).
3. `PrivacyLeakWarning` is a warning class, not a module: `PrivacyLeakWarning.simplefilter(...)` raises AttributeError. Write `import warnings` and `warnings.simplefilter("error", PrivacyLeakWarning)`; without `from diffprivlib.utils import PrivacyLeakWarning` the name is undefined (NameError).
4. Pattern to use: import `BudgetAccountant` from `diffprivlib.accountant`, create `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`, pass `accountant=acc` to tools and models.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Unchanged by the import path: `acc.total()` returns a `Budget(epsilon, delta)`; use index `[0]` for the epsilon spent and `[1]` for delta (`acc.total().epsilon` raises AttributeError).

## Pitfalls
- The module name is singular: `accountant`. The class name is `BudgetAccountant`, not `Accountant` or `PrivacyAccountant`.
- `BudgetAccountant()` with no arguments has an infinite epsilon budget and refuses nothing; give it the task's total.
- There is no `diffprivlib.mechanism` (singular) module; mechanisms live in `diffprivlib.mechanisms` (plural). Singular `accountant`, plural `mechanisms`, `models`, `tools`.
