---
id: diffprivlib-validation-bounds-clip
title: Validate and apply public bounds with validation.check_bounds and clip_to_bounds
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.validation.check_bounds
  - diffprivlib.validation.clip_to_bounds
  - diffprivlib.mechanisms.LaplaceTruncated
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/test_check_bounds.py
source_license: MIT
---

## Goal
When writing a custom DP aggregate with a mechanism object, normalise and validate user-supplied public bounds exactly the way diffprivlib's own tools do, clip the data to them, and derive the sensitivity from the validated bounds. Example: a per-column vector of clipped totals. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. `lower_vec, upper_vec <- validation.check_bounds((L, U), shape=d)`
   - Accepts scalars or length-d vectors; returns two length-d float arrays (a scalar pair is broadcast).
   - Raises TypeError if bounds are not a tuple, ValueError if lower > upper or the length is wrong.
   - `shape=0` returns two plain floats instead.
2. `X_clipped <- validation.clip_to_bounds(X, (lower_vec, upper_vec))` (rows of shape (n, d)).
3. Per-column totals `T_j <- sum of column j`. Under replace-one, one row changes column j by at most `U_j - L_j`, so the L1 sensitivity of the whole vector is `sum_j (U_j - L_j)`.
4. Either split epsilon over the d columns (each column: sensitivity `U_j - L_j`, epsilon/d) or use one shared scale; the first is shown here.
5. Per column: `mechanisms.LaplaceTruncated(epsilon=eps/d, sensitivity=U_j - L_j, lower=n*L_j, upper=n*U_j).randomise(T_j)`.
6. `acc.spend(eps, 0)` once for the vector; output and `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. Mechanism objects do not charge an accountant; the d per-column releases compose to `eps`, recorded with one `acc.spend(eps, 0)` (or d spends of eps/d).

## Pitfalls
- `check_bounds([L, U])` with a list raises TypeError: bounds must be a tuple.
- A tuple of per-column pairs `((L1, U1), (L2, U2))` is read as lowers=(L1, U1), uppers=(L2, U2): wrong, and only caught if some lower exceeds its upper.
- `clip_to_bounds` does not validate; call `check_bounds` first.
- Bounds still must come from domain knowledge; validation does not make data-derived bounds private.
