---
id: diffprivlib-random-forest
title: Random forest classifier with public bounds and classes
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.RandomForestClassifier
  - diffprivlib.models.DecisionTreeClassifier
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/forest.py
source_license: MIT
---

## Goal
Train a tree ensemble classifier with pure epsilon-DP. Each training row goes to exactly one tree, tree splits are chosen at random inside public feature bounds (no data used), and each leaf's label is picked from the leaf's class counts with the `PermuteAndFlip` mechanism. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `RandomForestClassifier(n_estimators=10, *, epsilon=1.0, bounds=None, classes=None, n_jobs=1, verbose=0, accountant=None, random_state=None, max_depth=5, warm_start=False, shuffle=False, **unused_args)`

## Steps
1. Per-feature public bounds `bounds=(L_vec, U_vec)` (data is clipped to them) and the public class label list `classes=LABELS` from the task.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `rf <- models.RandomForestClassifier(n_estimators=?, epsilon=EPS/2, bounds=(L_vec, U_vec), classes=LABELS, max_depth=?, shuffle=?, accountant=acc)`
   - `n_estimators`: number of trees; rows are split disjointly among them, so more trees means fewer rows each.
   - `max_depth`: depth of each random tree; memory grows exponentially with it.
   - `shuffle=True` assigns rows to trees at random instead of in order.
4. `rf.fit(X_train, y_train)`; `rf.predict`, `rf.score` are post-processing.
5. `models.DecisionTreeClassifier` is the single-tree building block; its docstring says it is not meant to be used on its own. Prefer the forest (with one estimator if one tree is wanted).
6. Compare with the non-private sklearn classifier of the same name.

## Privacy accounting
`fit` checks and spends `epsilon` once on the accountant. Each row is used by exactly one tree, so every tree runs with the full epsilon (parallel composition over trees, by position). Replace-one (bounded DP, n public): rows are assigned to trees by position, so a replaced row stays in its tree, but it can move from one leaf to another. Each leaf's label is a separate permute-and-flip release at the full epsilon (calibrated for add/remove), so two leaves change: the guarantee is 2 x the epsilon passed. Pass EPS/2 and report `2 * acc.total()[0]`; assert it is <= EPS.

## Pitfalls
- No `bounds` or no `classes` → `PrivacyLeakWarning`; they are read from the training data.
- Features must be numeric; encode categoricals first, with bounds known in advance.
- Passing the full EPS and reporting `acc.total()` under replace-one: the real loss is 2 x EPS (per-leaf releases do not compose in parallel when a row can change leaf).
- Many trees with few rows each gives very noisy leaves.
- With `shuffle=False` and data sorted by label, trees see one class only; shuffle or pass rows in random order.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
