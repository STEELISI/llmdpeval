---
id: diffprivlib-repeated-splits-disjoint-chunks
title: Repeated train/test evaluation of a DP classifier under one budget (what composes over repeats)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.GaussianNB
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
  - numpy.random.permutation
  - numpy.array_split
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/naive_bayes.py
source_license: MIT
---

## Goal
Reproduce an analysis that repeats a random train/test split R times (for example R = 50) and reports the mean and spread of test accuracy, with DP and one total epsilon. Replace-one (bounded DP, n public). Each repeat trains a DP model and releases a DP accuracy.

## Steps
1. Public: feature bounds, the class list, R, the test share, EPS.
2. Choose how the repeats compose (decide before running):
   - (a) Same rows in every repeat (resampled splits): every repeat reads every row, so the total is R × (model + evaluation epsilon). Give each repeat EPS/R.
   - (b) Disjoint chunks: `perm <- permutation(n)`; `chunks <- array_split(perm, R)`. The assignment uses row positions only, never values, so a replaced row stays in its chunk; each chunk can use the full EPS on n/R rows.
   - (c) One split only, reported once.
3. Per chunk: positions → train and test parts; `acc <- BudgetAccountant(epsilon=eps_model/2 + eps_eval, delta=0)`; shift features publicly so each lower bound is 0; append one placeholder row per public class (features at the public midpoint) to the training part; `GaussianNB(epsilon=eps_model/2, bounds=(0, U − L), accountant=acc).fit(...)`.
4. `correct <- predict(X_test) == y_test`; `dp_correct <- tools.count_nonzero(correct, epsilon=eps_eval, accountant=acc)`; accuracy = clip(dp_correct / n_test, 0, 1).
5. Report the R DP accuracies, their mean and standard deviation (post-processing), and the total.

## Privacy accounting
Replace-one (bounded DP, n public). (b): epsilon_spent = max over chunks of (2 × the GaussianNB accountant share + eps_eval) = EPS. Parallel composition holds here because chunks are fixed by position; it does not hold for partitions defined by a data value (group, class, cluster), where a replaced row can move between parts. GaussianNB, RandomForestClassifier and KMeans add noise per class, leaf or cluster with add/remove calibration, so under replace-one pass half the epsilon and count it twice. (a): sum the accountants.

## Pitfalls
- R repeats on the same rows, each at the full EPS: the real total is R × EPS.
- Averaging R noisy accuracies does not remove the extra cost of (a); it is the averaging attack.
- Exact test accuracy per repeat (`score`, `accuracy_score`) is not DP.
- Min-max scaling fitted on each split's own data: use public bounds.
- Chunks assigned by stratifying on the label or a group use private values to place rows; then (b) does not hold. Stratifying the train/test split inside a chunk is fine: both parts are charged to that chunk.
- Behaviour of diffprivlib 0.6.6: `GaussianNB` has no `classes=` argument in `fit`; it reads the labels from `np.unique(y)` and adds noise only to classes present, so a class missing from a chunk keeps a count of exactly 0 (visible). The label set must be public and present in every chunk: append one placeholder row per class (step 3).
- Fewer rows per chunk means noisier models; compare (a) and (b) at the same EPS.
- Add/remove alternative (n private): the library's per-class statistics are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x. Keep the shift to lower bound 0 and the placeholder rows either way.
