---
id: diffprivlib-logistic-regression-public-unit-norm
title: Logistic regression with public feature bounds, unit row norm, an explicit intercept column, and C chosen by a public rule
framework: diffprivlib
framework_version: "0.6.6"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.LogisticRegression
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.utils.PrivacyLeakWarning
  - numpy.expm1
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/logistic_regression.py
source_license: MIT
---

## Goal
Train a binary logistic regression with pure epsilon-DP by objective perturbation (Chaudhuri, Monteleoni & Sarwate, JMLR 2011, Algorithm 2), with every precondition of its privacy theorem met by construction and with the accuracy cost of DP reported separately from the cost of regularisation. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Public bounds `(L_j, U_j)` per feature, from documentation or a codebook. Declare them first; checking afterwards that the data fits inside them is fine, deriving them from the data is not.
2. Clamp each feature to its bounds (clamp, do not drop rows), min-max it to [0, 1], and multiply by `1/sqrt(2d)` for d features.
3. Append a constant intercept column `1/sqrt(2)`. Each row now has norm at most 1, the unit ball the paper assumes (Section 2).
4. `C <- min(C_WANTED, 4 * expm1(EPS / 8))`. With row norm 1 and logistic curvature bound c = 1/4, the library spends `2 * ln(1 + C/4)` of EPS on a slack term before adding noise; this public rule keeps that at most EPS/4.
5. `clf <- models.LogisticRegression(epsilon=EPS, data_norm=1.0, C=C, fit_intercept=False, max_iter=?, accountant=acc)`; `clf.fit(X, y)`.
6. Control: non-private sklearn `LogisticRegression(C=C, fit_intercept=False)` on the same matrix. Report three rows: default-C non-private, same-C control, DP.

## Privacy accounting
Replace-one (bounded DP, n public): objective perturbation with row norm <= 1 covers replacing one row (Chaudhuri, Monteleoni and Sarwate 2011), so no halving; the same epsilon holds under add/remove. `fit` spends EPS on the accountant (binary labels: no internal split). Read `acc.total()`. C, bounds and the scaling are public constants, so they cost nothing. Choosing C by looking at test accuracy is a selection on private data and is not covered.

## Pitfalls
- `fit_intercept=True`: the fitted intercept is not penalised, so the objective is not strongly convex in that direction (a precondition of the theorem), and the library raises the norm bound to sqrt(data_norm^2 + 1) > 1. Use an explicit column.
- `data_norm` above 1: 0.6.6 computes the slack as `2*ln(1 + c*q*C)` (mechanisms/vector.py) where the paper's proof needs `q^2`, so epsilon is under-charged. Unit norm makes the charge exact.
- `C = 1/n`: diffprivlib uses ridge strength `1/(C*n)`, so this is a unit ridge; even the non-private model can predict a single class.
- Small EPS with a large C: when the slack exceeds EPS the library silently switches to a fallback (EPS/2 plus extra ridge), and just above that point almost nothing is left for noise. Accuracy is not monotone in EPS there.
- Reporting DP accuracy against the default-C model calls the regularisation cost a privacy cost.
- `StandardScaler` or data min/max: data-dependent scaling leaks.
