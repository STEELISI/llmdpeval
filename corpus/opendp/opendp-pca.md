---
id: opendp-pca
title: DP PCA with dp.sklearn.decomposition.PCA
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: pca
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.extras.sklearn.decomposition.PCA
  - opendp.mod.binary_search_param
  - opendp.mod.enable_features
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/statistical-modeling/pca.rst
source_license: MIT
---

## Goal
Fit principal components of a numeric matrix with pure DP, using an sklearn-like wrapper (based on Amin et al. 2019, DP covariance estimation). Neighbouring relation: replace-one (n public): `n_samples` is part of the domain and the scale is searched at `d_in=2`.

## Steps
1. `enable_features("contrib", "idealized-numerics", "honest-but-curious")`, as in the docs. Needs scikit-learn and the numpy extra (randomgen).
2. Prepare X as a float numpy array of shape (N, D) with the columns to use. Rescale columns with public constants if their units differ a lot, because each row is clipped to a norm bound.
3. Public parameters: `N` (number of rows, treated as public), `D` (features), `ROW_NORM` (each row is clipped to this L2 norm; pick it from public scales of the columns), `K` (components).
4. `model <- PCA(epsilon=EPS, row_norm=ROW_NORM, n_samples=N, n_features=D, n_components=K)`
5. Check what the release costs: `meas <- model.measurement()`; read `meas.map(D_IN)`. If it is above the task's epsilon, search the constructor's `epsilon` argument with `binary_search_param(lambda e: PCA(epsilon=e, ...).measurement(), d_in=D_IN, d_out=EPS)`.
6. `model.fit(X)`; then use `model.components_`, `model.singular_values_`, `model.explained_variance_ratio_` like sklearn's PCA.

Signature for reference: `PCA(*, epsilon, row_norm, n_samples, n_features, n_components=None, n_changes=1, whiten=False)`.

## Privacy accounting
The budget is split internally between the mean and the eigen-decomposition. Do not assume the `epsilon` argument equals `meas.map(D_IN)` for your `D_IN`: the domain is sized, so pick `D_IN` for your neighbouring relation (2 for one replaced row), read the map, and report the map value.

## Pitfalls
- `n_samples` must equal the number of rows passed; it is part of the domain.
- Rows larger than `ROW_NORM` are clipped. A norm far above typical row norms wastes budget; far below distorts the components.
- Fitting twice on the same model raises "already been fitted"; create a new model per release.
- Without randomgen the fit fails at call time with an ImportError.
- `"idealized-numerics"` is required: without it the constructor raises and names the flag (checked on 0.16.0). It tells OpenDP to treat the wrapper's floating-point linear algebra as exact, so the guarantee ignores rounding error in the eigendecomposition. State this assumption next to the release.
- Needs `"honest-but-curious"` (the sklearn-style wrapper is built on user transformations). If the task forbids that flag, OpenDP has no PCA you may use; say so, or use the diffprivlib or NumPy PCA items if another framework is allowed.
