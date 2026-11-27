---
id: diffprivlib-vector-objective-perturbation
title: Objective perturbation for your own convex loss with mechanisms.Vector
framework: diffprivlib
framework_version: "0.6.6"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Vector
  - diffprivlib.validation.clip_to_norm
  - diffprivlib.accountant.BudgetAccountant
  - scipy.optimize.minimize
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/vector.py
source_license: MIT
---

## Goal
Train a linear classifier with your own smooth convex loss and an L2 penalty, beyond what `models.LogisticRegression` offers, using objective perturbation (Chaudhuri, Monteleoni and Sarwate, JMLR 2011, Algorithm 2). Pure epsilon-DP when the slack term fits inside epsilon. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `Vector(*, epsilon, function_sensitivity, data_sensitivity=1.0, dimension, alpha=0.01, n=1, random_state=None)`

## Steps
1. Rescale rows with public constants so their norm is at most 1, then `X <- clip_to_norm(X, Q)` with `Q <= 1`; labels in {-1, +1}.
2. Loss requirements (from the paper): convex, twice differentiable in the margin, `|l'| <= 1`, `|l''| <= c`. Logistic loss has `c = 1/4`; smoothed hinge losses have other c.
3. Objective as a function of the coefficient vector w: `f(w) <- mean(l(y * X @ w)) + (lambda/2) * ||w||^2` (mean loss, ridge lambda). It may return `(value, gradient)`.
4. `mech <- mechanisms.Vector(epsilon=?, function_sensitivity=c, data_sensitivity=Q, dimension=d, alpha=n * lambda, n=n)`. `alpha` is the ridge on the SUM of losses, so it is n times the mean-objective lambda.
5. `noisy_f <- mech.randomise(f)`: a new function that adds `b . w / n` (random b) and, if needed, an extra ridge term.
6. `w_hat <- argmin noisy_f` with an unconstrained optimiser (scipy `minimize`, `jac=True` if f returns a gradient).
7. `acc.spend(epsilon, 0)`; output accuracy and `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): objective perturbation with row norm <= 1 covers replacing one row (Chaudhuri, Monteleoni and Sarwate 2011), so no halving; the same epsilon holds under add/remove. Vector computes `epsilon_p = epsilon - 2 log(1 + c * Q / alpha)` with `alpha = n * lambda` and uses it for the noise. If `epsilon_p <= 0` it silently switches to the paper's fallback (noise at `epsilon / 2` plus an extra ridge term added to the objective), which changes the model you get, so check `epsilon > 2 * log(1 + c * Q / (n * lambda))` before trusting the result, and keep `Q <= 1` (rescale the data). The paper's proof (Theorem 9) bounds this term with `Q^2`, not `Q`: for `Q <= 1` the library's charge is exact or conservative, for `Q > 1` it under-charges epsilon (the fallback ridge uses `Q` too). Mechanisms do not charge an accountant.

## Pitfalls
- `randomise` takes a callable, not a number; passing a value raises TypeError.
- Passing the mean-objective lambda as `alpha` (forgetting the factor n) makes the slack term far too large and silently triggers the fallback. `models.LogisticRegression` passes `alpha = 1 / C` with an l2 strength of `1 / (C * n)` on the mean loss.
- Hinge loss (not differentiable) and squared loss on unbounded labels violate the conditions.
- Data-dependent Q (the max row norm) is a leak.
