---
id: diffprivlib-pipeline-scaler
title: DP preprocessing and model in one sklearn Pipeline with a shared budget
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.StandardScaler
  - diffprivlib.models.PCA
  - diffprivlib.models.LogisticRegression
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/pipeline.ipynb
source_license: MIT
---

## Goal
Train a model whose preprocessing also learns from the private data (scaling, PCA) so that every data-dependent step is DP and the steps together stay within one total epsilon. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `StandardScaler(*, epsilon=1.0, bounds=None, copy=True, with_mean=True, with_std=True, random_state=None, accountant=None)`

## Steps
1. Replace each sklearn step that learns from data with its diffprivlib version: `models.StandardScaler`, `models.PCA`, then the model.
2. Fix the epsilon share of each step from the task: EPS_1 + EPS_2 + EPS_3 = EPS_TOTAL.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`, shared by every step:
   - `scaler <- models.StandardScaler(epsilon=EPS_1, bounds=(L_vec, U_vec), accountant=acc)`
   - `pca <- models.PCA(n_components=K, epsilon=EPS_2, data_norm=NORM_1, centered=True, accountant=acc)`
   - `unit <- FunctionTransformer(Z -> [Z / (sqrt(2) * NORM_2), 1/sqrt(2)])`: a fixed public map, so rows have norm <= 1 and carry an explicit intercept column
   - `clf <- models.LogisticRegression(epsilon=EPS_3, data_norm=1.0, C=min(C, 4*expm1(EPS_3/8)), fit_intercept=False, accountant=acc)` (public C cap: the slack must fit inside EPS_3)
   - `pipe <- sklearn Pipeline of [scaler, pca, unit, clf]`
4. `pipe.fit(X_train, y_train)`; `pipe.score(X_test, y_test)` is post-processing.
5. To sweep epsilon, change shares with `pipe.set_params(<step>__epsilon=?)`; every refit spends again.

## Privacy accounting
Each step's `fit` checks and spends its own epsilon on the shared accountant, so `acc.total()` after `pipe.fit` is the sum of the step budgets. Replace-one (bounded DP, n public): the scaler's means and variances use public n, PCA's covariance and the objective-perturbation fit cover one replaced row, so no step needs halving. If the sum would exceed EPS_TOTAL, the later step raises `BudgetError`.

## Pitfalls
- Bounds and norms must be public at every stage. The scaled values `(x - dp_mean) / dp_std` depend on the DP std, so the public bounds alone do not bound them. PCA clips rows to `data_norm`: choose NORM_1 publicly (a few units, since values are in std units) and accept clipping. NORM_2 follows from NORM_1, since the PCA projection does not increase row norms.
- `StandardScaler` with `with_std=True` splits its epsilon between mean and variance internally; the accountant still records the step epsilon.
- Cross-validation or grid search over the pipeline refits on private data many times; every fit spends budget.
- `centered=True` on PCA is valid only because the DP scaler already centred the data.
- `LogisticRegression` with `data_norm` above 1 or `fit_intercept=True` (norm becomes sqrt(q^2 + 1)) under-charges epsilon in 0.6.6: the slack term uses q where the objective-perturbation proof needs q^2.
- diffprivlib 0.6.6 needs scikit-learn < 1.8 and scipy < 1.18.
