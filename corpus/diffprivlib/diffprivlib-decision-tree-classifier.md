---
id: diffprivlib-decision-tree-classifier
title: Single DP decision tree with public feature bounds and public class labels
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.DecisionTreeClassifier
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/models/test_DecisionTreeClassifier.py
source_license: MIT
---

## Goal
Train one interpretable decision tree classifier with pure epsilon-DP. Splits are drawn at random inside the public feature bounds (no data used); only the leaf labels look at the data, through the permute-and-flip mechanism. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `DecisionTreeClassifier(max_depth=5, *, epsilon=1, bounds=None, classes=None, random_state=None, accountant=None, criterion=None, **unused_args)`

## Steps
1. Public feature bounds as two vectors `bounds=(L_vec, U_vec)` (one entry per feature), from the codebook or valid ranges.
2. Public list of class labels `classes=[c1, ..., cK]` (from the task or schema, never `unique(y)`).
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. `tree <- models.DecisionTreeClassifier(max_depth=D, epsilon=EPS/2, bounds=(L_vec, U_vec), classes=[...], accountant=acc)`; arguments after `max_depth` are keyword-only.
5. `tree.fit(X_train, y_train)`; features outside the bounds are clipped.
6. `tree.predict(X_test)`; evaluate accuracy on held-out rows (see the DP evaluation outline if the test rows are private too).
7. Output: accuracy, `acc.total()`.

## Privacy accounting
`fit` checks and spends `epsilon` once on the accountant. Replace-one (bounded DP, n public): a replaced row can move from one leaf to another, and each leaf label is a separate permute-and-flip release at the full epsilon (calibrated for add/remove), so the guarantee is 2 x the epsilon passed. Pass EPS/2 and report `2 * acc.total()[0]`; assert it is <= EPS.

## Pitfalls
- Reporting `acc.total()` alone under replace-one: half the real loss.
- Missing `bounds` or `classes` → `PrivacyLeakWarning` and values read from the training data.
- `criterion=` and sklearn-only options such as `min_samples_leaf` are accepted and ignored (`DiffprivlibCompatibilityWarning`); random splits do not use a criterion.
- Deep trees spread rows over 2^D leaves; with few rows per leaf the noisy labels are near random. Keep `max_depth` small, or use `RandomForestClassifier` for an ensemble.
- Unlike sklearn, the first positional argument is `max_depth`, not `criterion`.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
