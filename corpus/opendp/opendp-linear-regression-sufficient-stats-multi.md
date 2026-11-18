---
id: opendp-linear-regression-sufficient-stats-multi
title: OLS linear regression with several covariates from noisy sufficient statistics (X^T X, X^T y) built from OpenDP bounded sums
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: regression
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-sum.ipynb
source_license: MIT
---

## Goal
Fit y ~ x1 + ... + xd + intercept with pure ε-DP. Neighbouring relation: replace-one (n public), so the domain is sized and every map is read at `d_in=2`. OpenDP 0.16 has no multivariate regression, so release every entry of XᵀX and Xᵀy (with a constant column for the intercept) as an OpenDP bounded sum, then solve the normal equations as post-processing (sufficient statistics perturbation; Wang 2018 analyses its utility).

## Steps
1. Public bounds for each covariate and for y (domain knowledge, never min/max of the data). Columns `Z <- [1, x1..xd, y]`, each clipped to its bounds.
2. Entries: every product `Z_i * Z_j` with i <= j, except y*y and the intercept entry 1*1 (= n, public under replace-one: set `A[0, 0] <- N` without noise). Each entry's public bounds are the min and max of the four corner products of the two intervals.
3. For each entry: `make(scale) <- (vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()) >> then_clamp(bounds=(LO, HI)) >> then_sum() >> then_laplace(scale=scale)`; `scale <- binary_search_param(make, d_in=2, d_out=EPS/M)` where M is the number of released entries. The sized sum is charged HI - LO.
4. Fill a symmetric matrix A (intercept + covariates) and a vector b from the noisy sums; `coef <- lstsq(A, b)`. Predict at new points with the same column order; do not clip the prediction inputs.

## Privacy accounting
One changed row changes every entry, so the M releases compose sequentially: `epsilon_spent <- sum(meas.map(2) for each entry)`; assert <= EPS. The solve is post-processing. Alternative, add/remove-one (n private): unsized domain, `d_in=1`, the 1*1 entry (n) must be released too, and each entry is charged max(|LO|, |HI|, HI - LO) by `meas.map(1)`.

## Pitfalls
- Bounds from the data, or clamping only some columns. Declared `atom_domain(bounds=...)` alone is not checked by OpenDP; clamp inside the pipeline.
- M grows quadratically with d: keep d small or the noise per entry swamps A.
- A noisy A can be ill-conditioned; a public ridge term added to A is fine (post-processing), a ridge tuned on the data is not.
- The y*y entry is not needed for the coefficients; releasing it wastes budget. Under replace-one the same holds for the 1*1 entry: it is the public n (noise on a constant).
- The unsized `d_in=1` route charges each entry max(|LO|, |HI|, HI - LO), never less than the sized HI - LO; with n public the sized route is the tighter one.
