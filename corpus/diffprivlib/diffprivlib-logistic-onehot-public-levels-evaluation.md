---
id: diffprivlib-logistic-onehot-public-levels-evaluation
title: DP logistic regression on numeric and one-hot categorical features with public levels, plus overall and per-group accuracy
framework: diffprivlib
framework_version: "0.6.6"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.LogisticRegression
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.utils.PrivacyLeakWarning
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/logistic_regression.py
source_license: MIT
---

## Goal
Train a binary logistic regression with pure ε-DP on a mix of numeric and categorical columns, and release its test accuracy overall and per group, where group sizes are private. Replace-one neighbours (bounded DP: train and test sizes public). Objective perturbation (Chaudhuri, Monteleoni & Sarwate 2011) is a replace-one guarantee.

## Steps
1. Public inputs: bounds `(L, U)` per numeric column and the full level list per categorical column (codebook), the group list, and the budget split (e.g. 0.8 model, 0.2 evaluation).
2. Design matrix with the SAME function for train and test: numeric clamped and min-maxed to [0, 1]; one 0/1 column per public level (unknown values give all zeros). Columns never depend on which values a split contains.
3. Norm: each block (numeric column or one-hot group) adds at most 1 to the squared row norm. Scale all by `1/sqrt(2d)` (d blocks) and add an intercept column `1/sqrt(2)`: every row has norm <= 1 by construction.
4. `C <- min(C_WANTED, 4·expm1(EPS_MODEL/8))` (public rule); `clf <- models.LogisticRegression(epsilon=EPS_MODEL, data_norm=1.0, C=C, fit_intercept=False, max_iter=?, accountant=acc)`; `clf.fit(X_train, y_train)`.
5. Evaluation: cell = 2·group_index + correct (one extra group for values outside the list). One `tools.histogram(cells, epsilon=EPS_EVAL/2, bins=k, range=(-0.5, k-0.5), accountant=acc)`.
6. Post-processing: clip cells at 0; accuracy = correct cells / public test size; group accuracy = correct / (correct + wrong) of that group, floor 1, clip to [0, 1].

## Privacy accounting
Model: EPS_MODEL, recorded by the accountant. Evaluation: diffprivlib's histogram charges one bin per row (add/remove); under replace-one a changed test row leaves one cell and enters another, so it gets EPS_EVAL/2 and its real cost is EPS_EVAL. `epsilon_spent <- EPS_MODEL + EPS_EVAL = acc.total()[0] + EPS_EVAL/2`, assert <= EPS.

## Pitfalls
- `pd.get_dummies` on train and test separately: different column counts and a shape error at predict time.
- Scaling with the data mean/std or min/max (private), or omitting `data_norm` (diffprivlib then reads it from the data).
- A non-private sklearn model feeding the accuracy: the model is a release too.
- Per-group accuracy divided by exact group sizes.
- Very large C: the slack term eats the budget; keep the public cap.
