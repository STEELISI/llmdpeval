---
id: diffprivlib-fraction-of-rows-public-n
title: Fraction of rows matching a predicate (support, classifier accuracy) with count_nonzero over a public n in diffprivlib
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
Release the share of rows satisfying a row-level predicate (support of a rule, accuracy of a fixed or DP-trained classifier on private test rows) with pure ε-DP, with the row count N public (replace-one).

## Steps
1. Indicator per row from that row only, e.g. `pred == label`.
2. `acc <- BudgetAccountant(epsilon=EPS, delta=0)`; `noisy <- count_nonzero(INDICATORS, epsilon=EPS, accountant=acc)`. It is a bounded sum over (0, 1) with geometric noise, sensitivity 1.
3. `fraction <- clip(noisy / N, 0, 1)` with the public N.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes the count by at most 1. Add/remove alternative (n private): the sensitivity is also 1, but `count_nonzero` truncates to [0, n] with the array length n, so use `mechanisms.Geometric` instead. One release at EPS: `epsilon_spent <- acc.total()[0]`; assert <= EPS. If the same script also trains a DP model on the same rows, add the model's epsilon to the total (sequential composition); with disjoint train and test rows, say why they are disjoint.

## Pitfalls
- `model.score(X_test, y_test)` is NOT private: it is the exact accuracy.
- `diffprivlib.tools.mean(INDICATORS, bounds=(0, 1), ...)` is an equivalent release; do not do both.
- If N is not public, release a noisy row count too and split ε.
- Printing the exact fraction next to the private one.
