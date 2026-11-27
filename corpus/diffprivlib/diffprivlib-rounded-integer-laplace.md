---
id: diffprivlib-rounded-integer-laplace
title: Integer output from a real-valued mechanism with transforms.RoundedInteger
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.transforms.RoundedInteger
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.LaplaceBoundedDomain
  - diffprivlib.mechanisms.Geometric
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/mechanisms/transforms/test_RoundedInteger.py
source_license: MIT
---

## Goal
Release an integer-valued statistic (a total of whole units, a bounded integer score) when the mechanism you need is real-valued, e.g. `LaplaceBoundedDomain` to keep the output inside [L, U], and the output must be an integer. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `RoundedInteger(parent)`; `parent` is a mechanism instance.

## Steps
1. Compute the exact statistic from data clipped to public bounds; state its sensitivity under the task's neighbouring relation.
2. Inner mechanism: `inner <- mechanisms.LaplaceBoundedDomain(epsilon=?, sensitivity=SENS, lower=L_OUT, upper=U_OUT)` (or plain `Laplace`).
3. `mech <- transforms.RoundedInteger(inner)`.
4. `dp_value <- mech.randomise(true_value)` returns a Python int (the inner output rounded).
5. `acc.spend(epsilon, 0)` on a `BudgetAccountant`, because mechanisms do not charge accountants.
6. Output the true value, DP integer, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. Rounding is post-processing, so the cost is the inner mechanism's epsilon. Record it manually with `acc.spend(epsilon, 0)` and report `acc.total()`.

## Pitfalls
- If no range constraint is needed, `mechanisms.Geometric` (two-sided geometric, sensitivity must be an integer) is the native integer mechanism and avoids floating-point sampling entirely.
- Rounding does not fix floating-point side channels of the inner Laplace sampler; it only changes the output type.
- The sensitivity is that of the exact statistic, not of the rounded one.
- Prefer `tools.sum(..., dtype=int)` for plain integer sums; it handles bounds, noise and accounting for you.
