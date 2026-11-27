---
id: diffprivlib-binary-randomized-response
title: Randomised response on a yes/no attribute with mechanisms.Binary, then debias the share
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Binary
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/binary.py
source_license: MIT
---

## Goal
Perturb each person's two-valued answer before it is collected (local DP, randomised response), then estimate the population share of one answer from the perturbed values. Each person gets pure epsilon-DP without trusting the collector. Local DP per record, which implies replace-one (bounded DP, n public) central DP at the same epsilon.

Signature: `Binary(*, epsilon, value0, value1, random_state=None)`

## Steps
1. Map the attribute to two STRING labels, e.g. `value0="no"`, `value1="yes"` (Binary works on strings; for 0/1 integers wrap it with `IntToString`).
2. `mech <- mechanisms.Binary(epsilon=?, value0="no", value1="yes")` (keyword-only).
3. For every record: `noisy_i <- mech.randomise(answer_i)`. The true label is kept with probability `p = e^eps / (1 + e^eps)` and flipped otherwise.
4. Observed share: `s_obs <- mean(noisy == "yes")`.
5. Unbiased estimate (post-processing): `s_hat <- (s_obs - (1 - p)) / (2p - 1)`, clipped to [0, 1].
6. Output the true share, the noisy share, the debiased estimate, and the epsilon per person.

## Privacy accounting
Replace-one (bounded DP, n public): each record is randomised on its own, so replacing one record changes one randomised value; the guarantee is epsilon under replace-one and under add/remove. Each person's value passes through the mechanism once, so each person has epsilon-DP for their own record; the collected column as a whole is also epsilon-DP (each record is touched once, parallel composition). Mechanisms do not charge an accountant: record it with `acc.spend(epsilon, 0)` once, not once per row.

## Pitfalls
- Randomising the same person twice with fresh noise and releasing both costs 2x epsilon (the person can be averaged out).
- Integer labels raise a TypeError; labels must be strings, or use `IntToString(Binary(...))`.
- `value0` and `value1` must differ.
- Local DP needs far more rows than central DP for the same accuracy; if a trusted curator exists, `tools.count_nonzero` on the true values is much more accurate.
- `p` must use the same epsilon that was passed to the mechanism.
