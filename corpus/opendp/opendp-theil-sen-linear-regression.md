---
id: opendp-theil-sen-linear-regression
title: Univariate linear regression with DP Theil-Sen (dp.sklearn.linear_model.LinearRegression)
framework: opendp
framework_version: "0.16.0"
mechanism: exponential
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.extras.sklearn.linear_model.LinearRegression
  - opendp.measures.max_divergence
  - opendp.mod.binary_search_param
  - opendp.mod.enable_features
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/src/opendp/extras/sklearn/linear_model/__init__.py
source_license: MIT
---

## Goal
Fit y ~ slope * x + intercept for one feature with pure DP, and predict at given x values. OpenDP's algorithm pairs points at random, predicts y at two fixed x positions from each pair's line, and releases the private median of each prediction column. It then solves for slope and intercept. Neighbouring relation: replace-one (n public), so the scale is searched and the map read at `d_in=2`.

## Steps
1. `enable_features("contrib", "honest-but-curious")`. The model is built from user transformations, so the second flag is required. Install scikit-learn.
2. Public bounds: `X_BOUNDS = [(LX, UX)]` (a list with exactly one tuple) and `Y_BOUNDS = (LY, UY)`. They set the x cut points and the candidate grid for y.
3. The constructor takes `scale`, not epsilon. Find the scale from the budget:
   `make <- lambda s: LinearRegression(max_divergence(), x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, scale=s).measurement`
   `S <- binary_search_param(make, d_in=2, d_out=EPS)` (one changed row = symmetric distance 2)
4. `model <- LinearRegression(max_divergence(), x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS, scale=S)`
5. Check `model.measurement.map(2) <= EPS`.
6. `fitted <- model.fit(X_2D, Y)` with X shaped (n, 1); it returns an sklearn `LinearRegression` with `coef_` and `intercept_`.
7. `fitted.predict(NEW_X_2D)` is post-processing.

Signature for reference: `LinearRegression(output_measure, x_bounds, y_bounds, scale, runs=1, candidates_count=100, fraction_bounds=(0.25, 0.75))`.

## Privacy accounting
Two private medians are composed; the map covers both. Report `model.measurement.map(2)` (replace-one). Alternative, add/remove-one (n private): search at `d_in=1` and report `map(1)`; at a fixed scale `map(2)` is twice `map(1)`. `runs` > 1 repeats the pairing and multiplies the stability, so it raises epsilon at a fixed scale. Search the scale with your real `runs`.

## Pitfalls
- Bounds from the data's min/max leak; use public ranges. Results outside `Y_BOUNDS` cannot be represented.
- `candidates_count` sets the resolution of the y grid; slope and intercept inherit that coarseness.
- Only one feature. For several features OpenDP has no DP linear regression; there is no DP logistic regression either.
- `fit` returns a plain sklearn model; do not refit it non-privately.
- Needs `"honest-but-curious"`. If the task forbids that flag, use the contrib-only noisy-moments regression (`opendp-ols-noisy-moments-se`): one Laplace release of the clamped sums, then solve.
