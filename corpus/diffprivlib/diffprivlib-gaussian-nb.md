---
id: diffprivlib-gaussian-nb
title: Gaussian naive Bayes classifier with models.GaussianNB
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.GaussianNB
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/naive_bayes.py
source_license: MIT
---

## Goal
Train a Gaussian naive Bayes classifier with pure epsilon-DP under replace-one (bounded DP, n public). The library adds noise to the per-class counts, means and variances (Vaidya, Shafiq, Basu and Hong, WI-IAT 2013).

Signature: `GaussianNB(*, epsilon=1.0, bounds=None, priors=None, var_smoothing=1e-09, random_state=None, accountant=None)`

## Steps
1. Per-feature public bounds `(L_vec, U_vec)` and the public class list `CLASSES`. Clip, then shift: `X <- clip(X, L_vec, U_vec) - L_vec`, bounds `(0, U_vec - L_vec)`.
2. Keep every public class present: append one placeholder row per class in `CLASSES` (features at the public midpoint `(U_vec - L_vec)/2`); a label outside `CLASSES` is an input error. `acc <- BudgetAccountant(epsilon=EPS_TOTAL/2, delta=0)`
3. `clf <- models.GaussianNB(epsilon=EPS/2, bounds=(0, U_vec - L_vec), priors=?, accountant=acc)`
   - `priors`: pass public class priors only if the task gives them; otherwise leave it out.
4. `clf.fit(X_train, y_train)` with numeric features only; then `clf.theta_ += L_vec` so `predict` takes raw features.
5. `clf.predict(X_test)`, `clf.score(X_test, y_test)` are post-processing.
6. Compare with the non-private sklearn `GaussianNB` on the same split.

## Privacy accounting
`fit` checks then spends `epsilon` once on the accountant. Internally the budget is shared between the noisy class counts and the per-feature means and variances; the accountant records exactly `epsilon`. Replace-one (bounded DP, n public): the class counts, means and variances are released per class, calibrated for add/remove. A replaced row can switch class and change two classes' statistics, so the guarantee is 2 x the epsilon passed. The per-class sums use sensitivity U - L, which covers removing a value only when L <= 0 <= U: shift each feature by its public lower bound first (bounds (0, U - L)), then add L back to `theta_` after fitting (post-processing; predictions on raw features are unchanged). Pass EPS/2 and report `2 * acc.total()[0]`; assert it is <= EPS.

## Pitfalls
- Missing `bounds` → `PrivacyLeakWarning` and the bounds come from the training data: not DP.
- Bounds that do not contain 0 (e.g. (18, 90)) without the shift: removing a row changes a class sum by up to max(|L|, |U|), more than the U - L the library assumes.
- Each feature's mean and variance get part of the budget; many features at small epsilon give poor accuracy. Use only the features the task needs.
- Categorical features must be encoded as numbers first; one-hot columns then have bounds (0, 1).
- Behaviour of diffprivlib 0.6.6: `fit` takes the classes from `np.unique(y)` and adds noise only to classes present in y. A class with no rows keeps `class_count_` exactly 0, while a present class gets a noisy count of at least 1, so an empty class is visible in the output. Safe pattern: take the class list from the task and append one placeholder row per class (step 2), so every class is present in every neighbouring dataset; each class count then includes one extra row. Class sizes for reporting: release them with `tools.count_nonzero(y == c)` on the full-length label array (public n), never on a per-class subset.
- Each `partial_fit` call spends epsilon again.
- Add/remove alternative (n private): the library's per-class statistics are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x. Keep the shift to lower bound 0 and the placeholder rows either way.
