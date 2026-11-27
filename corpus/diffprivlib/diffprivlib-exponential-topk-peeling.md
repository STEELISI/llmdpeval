---
id: diffprivlib-exponential-topk-peeling
title: Top-k candidates by row COUNT by repeated Exponential mechanism (peeling) with a split budget
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Exponential
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/exponential.py
source_license: MIT
---

## Goal
Release the k highest-scoring items from a public candidate list with pure epsilon-DP, by running the exponential mechanism k times and removing each winner ("peeling"). diffprivlib has no built-in top-k, so you split the budget and record it on an accountant. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. `CANDS`: every possible value, declared before looking at the data (codebook, schema), in a fixed (e.g. alphabetical) order. Never the distinct values found in the data.
2. `UTIL`: a count-based score per candidate from the private data; state its sensitivity SENS (1 for a count).
3. `eps_round <- EPS_SELECT / k`
4. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
5. Repeat k times:
   - `mech <- mechanisms.Exponential(epsilon=eps_round, sensitivity=SENS, utility=UTIL_left, candidates=CANDS_left)`
   - `winner <- mech.randomise()`; append to the result
   - `acc.spend(eps_round, 0)`
   - remove `winner` (and its score) from `CANDS_left` / `UTIL_left`
6. Output: the chosen k in order, the true top-k for comparison, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes each count score by at most 1, but two scores move in opposite directions; `Exponential` with `monotonic=False` keeps the factor 2 that covers this (also valid under add/remove). Each round is a separate release costing `eps_round`; sequential composition gives k times `eps_round`. The accountant records each spend; `acc.total()` must equal the selection budget, and `acc.spend` raises `BudgetError` on overspend.

## Pitfalls
- Floating-point: k equal shares of epsilon can add up to a hair above the accountant's cap (for example 9, 11, 18 or 20 shares of 1.0), and the last spend raises `BudgetError: Privacy spend of (...) not permissible`. Give each step `min(SHARE, acc.remaining()[0])`; never raise the cap to make it fit.
- Reusing the full epsilon in every round spends k times the budget.
- RNG in the loop should be one shared `RandomState` object (or None); the same int seed every round makes every round's randomness identical.
- If the task also asks for the scores of the selected items, release them with a separate DP query (e.g. a histogram) and budget it.
- A tie-break by data order leaks; the candidate order must be fixed in advance.
- Not for scores that are shares within a group (sensitivity 1 on a value in [0, 1], split over k picks, gives near-random picks): use diffprivlib-topk-rate-two-histograms.
