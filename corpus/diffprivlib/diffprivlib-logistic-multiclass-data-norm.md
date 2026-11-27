---
id: diffprivlib-logistic-multiclass-data-norm
title: Multiclass (one-vs-rest) DP logistic regression with data_norm derived from public bounds
framework: diffprivlib
framework_version: "0.6.6"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.LogisticRegression
  - diffprivlib.validation.clip_to_norm
  - diffprivlib.validation.clip_to_bounds
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/logistic_regression.py
source_license: MIT
---

## Goal
Train a logistic regression classifier with three or more classes under pure epsilon-DP, choosing the required row-norm bound `data_norm` from public feature bounds instead of from the data. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Public per-feature bounds `(L_vec, U_vec)`.
2. Rescale every feature to [0, 1] with those bounds: `X01 <- (clip_to_bounds(X, (L_vec, U_vec)) - L_vec) / (U_vec - L_vec)`. Then every row has L2 norm at most `sqrt(d)`, a public number.
3. Divide by `sqrt(2d)` and append a constant intercept column `1/sqrt(2)`, so every row has norm at most 1 and `data_norm = 1`; `clip_to_norm(X, 1.0)` is a cheap safety net (diffprivlib also clips internally).
4. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
5. Public C rule: `C <- min(C, 4*expm1(EPS/K/8))` for K classes. `clf <- models.LogisticRegression(epsilon=EPS, data_norm=1.0, C=C, fit_intercept=False, accountant=acc)`; all arguments keyword-only.
6. `clf.fit(X01, y)`: with K > 2 classes diffprivlib trains K one-vs-rest models, each with `epsilon / K`. `coef_` has shape (K, d + 1) (last column = intercept).
7. Apply the SAME public rescaling to test rows, then `clf.predict`.
8. Output: accuracy, `acc.total()`.

## Privacy accounting
`fit` spends `epsilon` once on the accountant (split over the K binary models internally). Read `acc.total()`. Replace-one (bounded DP, n public): objective perturbation with row norm <= 1 covers replacing one row, so no halving; the same epsilon holds under add/remove.

## Pitfalls
- `data_norm=None` → `PrivacyLeakWarning`, max row norm measured on the data.
- `data_norm = np.linalg.norm(X, axis=1).max()` written by hand is the same leak with no warning.
- Only one-vs-rest exists. `multi_class='multinomial'`, `solver=` and `penalty=` are silently ignored apart from a `DiffprivlibCompatibilityWarning`.
- The class labels are read from `y` (no `classes=` argument); the label set is assumed public.
- With `fit_intercept=True` the library enlarges the norm bound to `sqrt(data_norm**2 + 1)` > 1. In 0.6.6 the slack term uses the norm q where the objective-perturbation proof needs q^2 (mechanisms/vector.py), so any norm above 1 under-charges epsilon. Keep the explicit column and `data_norm=1`.
- Each binary model gets `epsilon / K`; if its slack `2*ln(1 + C/4)` exceeds that, the library silently switches to its fallback (half the epsilon plus extra ridge). Lower C for many classes.
- Each extra class divides epsilon further; merge rare classes if accuracy collapses.
