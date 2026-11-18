---
id: diffprivlib-accountant-slack-advanced
title: Advanced composition through the accountant's slack parameter
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: approx
needs_bounds: false
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.tools.mean
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/accountant.py
source_license: MIT
---

## Goal
Run many small-epsilon queries and report a total epsilon below the plain sum, by accepting a small extra delta. The accountant implements the composition bounds of Kairouz, Oh and Viswanath (ICML 2015 / IEEE TIT 2017). Neighbouring relation: replace-one (bounded DP, n public); each query is calibrated for it and the accountant composes the per-query epsilons.

Signature: `BudgetAccountant(epsilon=inf, delta=1.0, slack=0.0, spent_budget=None)`

## Steps
1. Only use this if the task allows approximate DP (a nonzero delta).
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=DELTA_TOTAL, slack=SLACK)` with 0 <= SLACK <= DELTA_TOTAL.
3. Run the queries as usual, each with `accountant=acc`. Equal shares: `eps_each <- acc.remaining(k)[0]`, then give query i `min(eps_each, acc.remaining(k - i)[0])` so float rounding never pushes the last spend past the ceiling.
4. `acc.total()`: epsilon is the smallest of the plain sum and two advanced-composition bounds; delta includes the slack.
5. `acc.slack` can be changed later; the library refuses a value that would push the total past the ceiling (BudgetError).
6. Show the saving: `BudgetAccountant().total(spent_budget=acc.spent_budget, slack=0)` gives the plain-sum total for the same spends.

## Privacy accounting
Replace-one (bounded DP, n public): each query's epsilon is for one replaced row (for example `tools.mean` with public n); the accountant does not care which relation, it composes what it is given. All composition maths happens inside `BudgetAccountant.total()`; never sum epsilons by hand when slack > 0. The ceiling check in `acc.check` uses the same total, so the advanced bound is what gets enforced.

## Pitfalls
- With `slack=0` (default) the total is the plain sum: basic composition, which is optimal for pure DP.
- Slack only helps with many queries; for a handful the plain sum is usually smaller and is returned.
- A slack larger than the delta ceiling (e.g. `delta=0` with slack > 0) raises ValueError.
- Reporting an epsilon without the accompanying delta is wrong once slack is used.
- Floating-point: k shares of `acc.remaining(k)[0]` can add up to a hair above the ceiling (for example k = 6, 9 or 20), and the last spend raises `BudgetError`. Clamp each share to what is left (step 3); never raise the ceiling to make it fit.
