---
id: diffprivlib-linear-regression
title: Linear regression with models.LinearRegression (functional mechanism)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.models.LinearRegression
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/linear_regression.py
source_license: MIT
---

## Goal
Fit an ordinary least-squares model y ~ X with pure epsilon-DP on the training rows, then predict at new points. Neighbouring relation: replace-one (bounded DP, n public). The library perturbs the coefficients of the squared-error objective (functional mechanism, Zhang et al., VLDB 2012). For reported coefficients with standard errors, prefer diffprivlib-ols-noisy-moments-se.

Signature: `LinearRegression(*, epsilon=1.0, bounds_X=None, bounds_y=None, fit_intercept=True, copy_X=True, random_state=None, accountant=None, **unused_args)`

## Steps
1. Public bounds for every feature and for the target: `(L_vec, U_vec)` and `(Ly, Uy)`, from codebooks or physical limits.
2. Public shift so each feature lies in `[-(U - L), 0]`: `X_s <- clip(X, L_vec, U_vec) - U_vec`; append a constant intercept column of -1: `X_d <- [X_s, -1]`, bounds `bounds_X=([L_vec - U_vec, -1], [0, ..., 0])`.
3. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
4. `reg <- models.LinearRegression(epsilon=EPS, bounds_X=?, bounds_y=(Ly, Uy), fit_intercept=False, accountant=acc)`; `reg.fit(X_d, y)` (X 2-D even when d = 1).
5. Undo the shift (post-processing): `coef <- reg.coef_[:d]`, `intercept <- -reg.coef_[d] - coef . U_vec`; predict with `X_new @ coef + intercept`.
6. Fit the non-private sklearn `LinearRegression` on the same data for the true coefficients and predictions.

## Privacy accounting
Replace-one (bounded DP, n public). With `fit_intercept=False`, `fit` spends `epsilon` once and the whole epsilon goes to the noisy objective coefficients; each coefficient's sensitivity is the range of its monomial over the bounds, which covers one replaced row. Behaviour of diffprivlib 0.6.6 that the steps work around: (a) with `fit_intercept=True` it also spends `epsilon/(d+1)` on DP means of X and the same on y for centring, while the objective gets `epsilon*d/(d+1)` and the accountant records `epsilon`; the real total is `epsilon*(d+2)/(d+1)` (1.5x at d = 1). If you keep `fit_intercept=True`, pass `EPS*(d+1)/(d+2)` and report the real total. (b) the noise on each X'X diagonal entry uses `|lower|^2` of that feature's (centred) bounds, so when the upper side is wider it is under-noised, and with lower bound 0 it gets no noise. The shift in step 2 makes `|lower|` the largest magnitude. Add/remove alternative (n private): the same sensitivities hold, because every monomial range contains 0 after the shift, so the epsilon is the same.

## Pitfalls
- Missing `bounds_X` or `bounds_y` → `PrivacyLeakWarning`; the bounds are then read from the data: not DP.
- Bounds equal to the observed min and max of the data are also a leak, even when typed in by hand. Use documented ranges.
- Default `fit_intercept=True` with `acc.total()` reported as the cost: the accountant shows less than is spent (a).
- Features in `[0, U]` with `fit_intercept=False`: the X'X diagonal is released without noise (b).
- Each extra feature adds many noisy objective coefficients; with few rows and small epsilon the fit can be far off. Report it against the non-private fit.
- Predictions are only meaningful inside the bounds the model saw.
- diffprivlib 0.6.6 needs scikit-learn < 1.8 and scipy < 1.18; pin them.
