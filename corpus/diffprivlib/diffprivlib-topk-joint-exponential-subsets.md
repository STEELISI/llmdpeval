---
id: diffprivlib-topk-joint-exponential-subsets
title: Top-k set by COUNT with one exponential-mechanism draw over all k-subsets of a public list
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Exponential
  - diffprivlib.accountant.BudgetAccountant
  - itertools.combinations
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/exponential.py
source_license: MIT
---

## Goal
Release the set of k categories with the largest counts from a short public list, with pure epsilon-DP, in one selection instead of k peeling rounds. The exponential mechanism (McSherry & Talwar 2007) picks one candidate among all k-subsets. This is a simple joint selection in the spirit of Gillenwater et al. (ICML 2022), not their exact construction. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. `CATS`: the public category list (K entries) in a fixed public order.
2. `per_cat <- [count of rows in c for c in CATS]`, zeros included.
3. `SUBSETS <- combinations(range(K), k)`: every k-subset, built from the public list only. There are C(K, k) of them, so K must be small (K = 14, k = 3 gives 364).
4. `utility[s] <- sum(per_cat[i] for i in s)`.
5. Sensitivity of every utility is 1: one row changes one count by 1 (add/remove), or one count by -1 and another by +1 (replace-one), and a subset sum moves by at most 1. Use `monotonic=False` unless the relation is add/remove.
6. `mech <- mechanisms.Exponential(epsilon=EPS, sensitivity=1, utility=utility, monotonic=?, candidates=list(range(len(SUBSETS))))`
7. `pick <- SUBSETS[mech.randomise()]`; `acc.spend(EPS, 0)`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes each count score by at most 1, but two scores move in opposite directions; `Exponential` with `monotonic=False` keeps the factor 2 that covers this (also valid under add/remove). One exponential-mechanism draw costs EPS, whatever k is. Mechanisms do not charge an accountant, so spend EPS on a `BudgetAccountant` and report `acc.total()`.

## Pitfalls
- When each row adds to one candidate, Laplace noise on one histogram (`opendp-topk-count-from-histogram`) is at least as accurate as this joint draw and simpler (Gillenwater et al., ICML 2022, Conclusion). Both pay epsilon once, while peeling gives each round only EPS/k.
- The output is a SET. Ordering it by the true counts leaks; order it by the public list, or release counts separately with their own budget.
- Utility = sum of counts is valid for count scores only. Rates (proportions) have a different sensitivity; use a rate outline.
- C(K, k) grows fast: for long lists use a histogram or one-shot noisy top-k.
