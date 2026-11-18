---
id: diffprivlib-gaussian-nb-partial-fit
title: Incremental DP naive Bayes over data batches with GaussianNB.partial_fit
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
Train a Gaussian naive Bayes classifier on data that arrives in batches (chunks, days, sites), updating the model per batch with pure epsilon-DP per batch, under replace-one (bounded DP, n public).

Signature: `GaussianNB(*, epsilon=1.0, bounds=None, priors=None, var_smoothing=1e-09, random_state=None, accountant=None)`

## Steps
1. Public per-feature bounds `bounds=(L_vec, U_vec)` and the public list of class labels.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL/2, delta=0)` (the accountant sees the halved shares)
3. Shift features by the public lower bounds (`X <- clip(X, L_vec, U_vec) - L_vec`); `clf <- models.GaussianNB(epsilon=EPS_BATCH/2, bounds=(0, U_vec - L_vec), accountant=acc)`
4. Keep every public class present in every batch: append one placeholder row per class (features at the public midpoint `(U_vec - L_vec)/2`) to each batch. Before each batch set `clf.epsilon = min(EPS_BATCH/2, acc.remaining()[0])`. First batch: `clf.partial_fit(X_1, y_1, classes=[c1, ..., cK])`; `classes` is required on the first call.
5. Later batches: `clf.partial_fit(X_b, y_b)`; each call adds noisy class counts, means and variances for that batch and merges them into the running estimates.
6. After the last batch: `clf.theta_ += L_vec` (post-processing); then `clf.predict(X_new)` on raw features.
7. Output: accuracy, `acc.total()`, and `acc.spent_budget` (one entry per batch).

## Privacy accounting
Every `partial_fit` call spends the epsilon passed (`EPS_BATCH/2`) on the accountant (one third for class counts, two thirds for per-feature means and variances, split inside the library). With B batches the accountant records B x EPS_BATCH/2. If the batches are disjoint sets of people and batch membership is fixed by position or arrival order (not by the record's own values), the true cost for any one person is EPS_BATCH (parallel composition); if batches are defined by a record value (site, event day), a replaced record can move between two batches, so two batches change (2 x EPS_BATCH). In every case the accountant reports the conservative sum; state which one you claim and why. Replace-one (bounded DP, n public): a replaced row can switch class and change two classes' statistics, and the per-class sums need a lower bound of 0, so shift features and pass EPS_BATCH/2; report `2 * acc.total()[0]`.

## Pitfalls
- Reporting `acc.total()` under replace-one without halving: half the real loss.
- Floating-point: k equal shares of epsilon can add up to a hair above the accountant's cap (for example 9, 11, 18 or 20 shares of 1.0), and the last spend raises `BudgetError: Privacy spend of (...) not permissible`. Before each `partial_fit` set `clf.epsilon = min(EPS_BATCH/2, acc.remaining()[0])`, i.e. give each step `min(SHARE, acc.remaining()[0])`; never raise the cap to make it fit.
- Behaviour of diffprivlib 0.6.6: each `partial_fit` call adds noise only to the classes present in that batch's y; a class with no rows in the batch gets no update and no noise, while a present class gets a noisy count of at least 1, so an empty class is visible. Small batches make this likely. Safe pattern: pass the public `classes=` on the first call and append one placeholder row per class to every batch (step 4); each class count then includes one extra row per batch.
- Calling `fit` resets the model; use `partial_fit` to accumulate.
- If the same person can appear in several batches, the sum is the right cost.
- Pick `EPS_BATCH` so that B x EPS_BATCH fits the accountant's cap, or the last batches raise `BudgetError`.
- Missing `bounds` → `PrivacyLeakWarning`.
- Very small batches give very noisy per-batch statistics; batch sizes are treated as public.
- Add/remove alternative (n private): the library's per-class statistics are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x. Keep the shift to lower bound 0 and the placeholder rows either way.
