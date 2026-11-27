---
id: numpy-randomized-response-kary
title: k-ary randomized response for a categorical attribute and frequency estimation (local DP) in NumPy
framework: numpy
framework_version: "any"
mechanism: other
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.random
  - numpy.random.Generator.integers
  - numpy.exp
  - numpy.log
source_url: https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao
source_license: original code (algorithm from cited paper)
---

## Goal
Protect each record's category (one of K public categories) with ε-local DP, then estimate the frequency of every category. Generalized (k-ary) randomized response, "direct encoding" in Wang, Blocki, Li & Jha, USENIX Security 2017.

## Steps
1. Public category list of size K; map each record to its index (records outside the list: map to an explicit "other" category included in K).
2. p = e^ε / (e^ε + K − 1) (report the truth); q = 1 / (e^ε + K − 1) (report one specific other category).
3. Per record: with probability p keep the true index; otherwise report one of the other K − 1 categories uniformly (draw an integer in [0, K − 2] and skip the true index).
4. Estimate each frequency: f̂_v = (n_v / N − q) / (p − q), where n_v is the reported count of v. Optionally clip at 0 and renormalise (post-processing, adds bias).

## Privacy accounting
For any two inputs and any report, the probability ratio is at most p/q = e^ε, so each report is ε-LDP. Compute ε_spent = ln(p/q) in code. The estimates are post-processing.

## Pitfalls
- Using binary randomized response's p = e^ε/(1+e^ε) with K > 2 categories: the guarantee is wrong.
- "Uniform over all K" in the else branch instead of over the other K − 1: then p and q change; recompute them.
- Large K: variance grows with K; Wang et al. recommend unary encoding or local hashing in that regime.
- Building the category list from the reported data.
