---
id: diffprivlib-accountant-resume-spent-budget
title: Carry a privacy ledger across sessions with spent_budget, and plan with total(spent_budget=...)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.utils.BudgetError
  - diffprivlib.tools.median
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/test_BudgetAccountant.py
source_license: MIT
---

## Goal
Keep one lifetime budget for a dataset when analyses happen in separate scripts or sessions: restore earlier spends into a new accountant, check that a planned set of queries still fits, then run them. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Earlier session: after releasing, save `acc.spent_budget` (a list of `(epsilon, delta)` tuples) to the project's ledger.
2. New session: `acc <- BudgetAccountant(epsilon=EPS_LIFETIME, delta=0, spent_budget=PREVIOUS_SPENDS)`. The constructor replays the old spends and raises `BudgetError` if they already exceed the ceiling.
3. Plan without spending: `planned <- acc.spent_budget + [(e1, 0), (e2, 0)]`; `acc.total(spent_budget=planned)` returns the epsilon the plan would reach. Compare it with `EPS_LIFETIME`, or call `acc.check(e1, 0)` for a single query.
4. If it fits, run the new releases with `accountant=acc` (e.g. `tools.median(values, epsilon=e1, bounds=(L, U), accountant=acc)`).
5. Output the releases, the new `acc.spent_budget` to store back, and `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. The lifetime total is `acc.total()` over old plus new spends. `total(spent_budget=...)` and `check` are pure calculations; they spend nothing.

## Pitfalls
- `spent_budget` must be a list (a tuple raises TypeError) of `(epsilon, delta)` pairs.
- Starting each script with a fresh accountant and the full budget resets the ledger: the real cost is the sum over all scripts.
- `acc.total()` uses the accountant's own `slack`; pass `slack=` to `total` to see a different composition bound.
- The default accountant is per Python process; it does not persist across sessions.
