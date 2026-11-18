---
id: numpy-topk-count-joint-exponential
title: Top-k set by COUNT in NumPy with one exponential-mechanism draw over all k-subsets
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.exp
  - itertools.combinations
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release the set of k categories with the largest counts from a short public list with pure epsilon-DP, in one exponential-mechanism draw (Dwork & Roth 2014, Def. 3.4, Thm. 3.10) over all k-subsets, so the budget is not split over k rounds as in peeling. Neighbouring relation: replace-one (n public).

## Steps
1. Public category list (K entries) in a fixed public order; exact counts for all of them, zeros included.
2. `SUBSETS <- all k-subsets of range(K)` (from the public list only). There are C(K, k); keep K small.
3. `u[s] <- sum of the counts in s`.
4. Utility sensitivity `Δu = 1`: one row changes one count by 1 (add/remove), or one by -1 and one by +1 (replace-one); a subset sum moves by at most 1.
5. Weights `w[s] <- exp(ε · u[s] / (2Δu))`. Subtract the largest exponent before `exp` to avoid overflow; this does not change the probabilities.
6. Draw one subset with probability `w / sum(w)` using a fresh, unseeded generator.
7. Release the subset in public-list order.

## Privacy accounting
Assumptions: utility sensitivity Δu = 1, scores not assumed monotone. Then sampling with `exp(ε·u/(2Δu))` is ε-DP, and the whole top-k set costs ε once. Compute epsilon_spent from the coefficient actually used (2·Δu·coefficient) and assert it equals the budget.

## Pitfalls
- Peeling at ε/k per round on a short list is noisier. When each row adds to one candidate, one noisy histogram is at least as accurate as this joint draw and simpler (Gillenwater et al., ICML 2022, Conclusion); both pay ε once.
- Ordering the released set by the true counts leaks the ranking.
- Using the factor ε·u/Δu (no 2) assumes monotone utilities, which holds only under add/remove.
- Rates (proportions) are not counts: their sensitivity is not 1.
- C(K, k) explodes for long lists; use a noisy histogram or Gumbel top-k then.
