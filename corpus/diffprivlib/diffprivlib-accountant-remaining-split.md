---
id: diffprivlib-accountant-remaining-split
title: Split what is left of a budget evenly over k further queries with acc.remaining(k)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.tools.mean
  - diffprivlib.tools.var
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/accountant.py
source_license: MIT
---

## Goal
After some releases have already used part of a fixed total budget, give each of k remaining queries the largest equal epsilon that still fits, letting the accountant do the arithmetic (including slack, if one is set). Neighbouring relation: replace-one (bounded DP, n public).

Signature: `BudgetAccountant(epsilon=inf, delta=1.0, slack=0.0, spent_budget=None)`; method `remaining(k=1)`.

## Steps
1. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)` and run the first release(s) with `accountant=acc`.
2. `(eps_each, delta_each) <- acc.remaining(k)`: the per-query budget such that k more queries of that size exactly reach the ceiling.
   - With `slack=0` this is `(EPS_TOTAL - spent) / k`. With `slack > 0` the accountant solves for the per-query epsilon under its advanced composition bound, which is larger than the plain division.
3. Run each of the k queries with `epsilon=eps_each, accountant=acc` (e.g. `tools.mean` and `tools.var` on k columns with public bounds).
4. Output the releases, `acc.total()` (should equal the ceiling up to rounding) and `acc.remaining()` (about 0).

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. The accountant is the source of truth: it computed the split and it records every spend. `acc.total()` is the reported epsilon.

## Pitfalls
- `remaining(k)` returns a PER-QUERY budget for k queries, not the total left; `remaining()` with k=1 is the total left.
- Floating-point rounding can make the last query exceed the ceiling by a hair and raise `BudgetError` (with a total of 1.0 and 0.4 spent first, `remaining(20)` fails on the 20th query); the ceiling check has no tolerance, so never round up. Give every query `min(eps_each, acc.remaining()[0])`, as the skeleton does.
- Deciding k after looking at results is adaptive; that is still fine for basic composition as long as every query is charged.
- `k` must be an integer >= 1.
