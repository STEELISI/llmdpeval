---
id: diffprivlib-dp-model-evaluation
title: Report a DP model's accuracy on private test rows with a DP count
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.GaussianNB
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/naive_bayes.ipynb
source_license: MIT
---

## Goal
Train a DP classifier and also release its test-set accuracy when the test rows are private too. The model's epsilon does not cover statistics computed on the test rows, so the accuracy needs its own DP release. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Public feature bounds, public class labels, a public test-set size `n_test` (or release a DP count of it separately).
2. Split EPS_TOTAL into `eps_model + eps_eval`; `acc <- BudgetAccountant(epsilon=eps_model/2 + eps_eval, delta=0)` (capped at what it records when the model is halved).
3. Train any diffprivlib model with `accountant=acc`, e.g. `models.GaussianNB(epsilon=eps_model/2, bounds=(0, U_vec - L_vec), accountant=acc).fit(X_train - L_vec, y_train)`. GaussianNB, RandomForestClassifier and KMeans release per class, leaf or cluster (add/remove calibration), so under replace-one pass half and count it twice; shift features so each lower bound is 0. Predict on `X_test - L_vec`. For GaussianNB, append one placeholder row per public class (features at the public midpoint) to the training rows so every class is present.
4. `correct <- (clf.predict(X_test - L_vec) == y_test)`: a boolean array over private test rows.
5. `dp_correct <- tools.count_nonzero(correct, epsilon=eps_eval, accountant=acc)`; one test row changes this count by at most 1.
6. Post-processing: `dp_accuracy <- clip(dp_correct / n_test, 0, 1)`.
7. Output: DP accuracy, `acc.total()`. If a non-private accuracy is also printed for debugging, label it as such.

## Privacy accounting
Train and test rows are disjoint people, so in principle each person is covered by one of the two releases (parallel composition). Replace-one (bounded DP, n public). Report the sum eps_model + eps_eval, which is always a valid (conservative) total: `acc.total()[0] + eps_model/2` for a halved GaussianNB. `count_nonzero` needs no halving: with n_test public, one replaced test row moves the count by at most 1.

## Pitfalls
- `accuracy_score(y_test, pred)` on private test rows is an exact statistic; publishing it is outside the DP guarantee.
- Several metrics (precision, recall, per-group rates): release one (group x true x predicted) table instead of one count per metric; see the group-confusion outline.
- Model selection on the test set (trying many models, keeping the best) costs more budget; see the model-selection outline.
- Behaviour of diffprivlib 0.6.6: GaussianNB takes the classes from `np.unique(y_train)` and adds noise only to classes present there; an empty class keeps a count of exactly 0, a present one gets at least 1, so an empty class is visible. Use the public class list and one placeholder row per class (step 3).
- For tiny test sets, the noise on the accuracy can exceed the differences you want to show.
- Add/remove alternative (n private): the library's per-class statistics are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x. Keep the shift to lower bound 0 and the placeholder rows either way. Under add/remove `n_test` is private too: release a DP count of the test rows and divide by it.
