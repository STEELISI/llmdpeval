---
id: diffprivlib-bingham-top-eigenvector
title: First principal direction of a covariance matrix with the Bingham mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: pca
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Bingham
  - diffprivlib.validation.clip_to_norm
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/models/utils.py
source_license: MIT
---

## Goal
Release only the leading eigenvector (first principal direction) of a data matrix's second-moment matrix with pure epsilon-DP, by sampling from a Bingham distribution (Kent, Ganeiber and Mardia). This is the building block `models.PCA` uses for each component. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `Bingham(*, epsilon, sensitivity=1.0, random_state=None)`

## Steps
1. Public row-norm bound `R` (from feature bounds: e.g. rescale features to [0, 1] and use `R = sqrt(d)`).
2. Centre with PUBLIC centres if needed (e.g. the midpoints of the public bounds), never the data mean.
3. `X <- clip_to_norm(X, R)` so every row has L2 norm at most R.
4. Normalised second-moment matrix: `A <- X.T @ X / R**2` (d x d). Under add/remove-one each row changes A by a rank-one matrix of norm at most 1, so `sensitivity=1` matches how diffprivlib calls it in `models.utils.covariance_eig`.
5. `mech <- mechanisms.Bingham(epsilon=?, sensitivity=1.0)`; `v <- mech.randomise(A)`, a unit vector of length d.
6. The sign of an eigenvector is arbitrary; flip it to a fixed convention (post-processing).
7. `acc.spend(epsilon, 0)`; output `v` and `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): for rows with norm <= 1, x x^T - y y^T (both PSD, spectral norm <= 1) has spectral norm <= 1, so `sensitivity=1` holds for a replaced row as well as for an added or removed one; the library samples with epsilon/(4*sensitivity), which is conservative. One Bingham sample costs `epsilon`. Mechanisms do not charge an accountant; record `acc.spend(epsilon, 0)`. For k components, `models.PCA` already splits epsilon across components and eigenvalues; use it instead of repeating this by hand.

## Pitfalls
- `randomise` needs a square 2-D NumPy array; lists or 1-D arrays raise errors.
- Skipping the norm clip or the division by R**2 breaks the sensitivity assumption.
- Centring with the private mean costs extra budget (PCA spends epsilon/2 on it when `centered=False`).
- The eigenvalue (variance explained) is NOT released by Bingham; it needs a separate DP release.
