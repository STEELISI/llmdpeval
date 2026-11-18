---
id: diffprivlib-laplace-bounded-domain
title: Laplace noise restricted to a valid output range (LaplaceBoundedDomain, LaplaceTruncated)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.LaplaceBoundedDomain
  - diffprivlib.mechanisms.LaplaceTruncated
  - diffprivlib.mechanisms.LaplaceFolded
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Release a real-valued statistic whose true value is known to lie in a public interval [L, U] (a rate in [0, 1], a mean inside known bounds) so the DP output also stays in that interval. Pure epsilon-DP. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `LaplaceBoundedDomain(*, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None)`

## Steps
1. Fix `lower=L`, `upper=U` from public knowledge of the statistic's possible values.
2. Determine the statistic's global sensitivity SENS under the task's neighbouring relation (after clipping inputs to public bounds).
3. Choose a variant (same arguments for all three):
   - `mech <- mechanisms.LaplaceBoundedDomain(epsilon=?, sensitivity=SENS, lower=L, upper=U)`: draws directly from the interval by rejection sampling; the library searches for a larger noise scale so epsilon still holds (Holohan et al., "The Bounded Laplace Mechanism", 2020).
   - `mechanisms.LaplaceTruncated(...)`: adds Laplace noise and clamps to the interval.
   - `mechanisms.LaplaceFolded(...)`: reflects out-of-range values back in.
4. `dp_value <- mech.randomise(true_value)`; the true value must already be inside [L, U].
5. `acc.spend(?, 0)` with the same epsilon.
6. `mech.bias(true_value)` reports the bias these variants introduce near the edges; reporting or correcting it is post-processing.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row moves the mean by at most (U - L)/n, the sensitivity used. Add/remove alternative (n private): n is not public, so release a noisy sum and a noisy count instead. Mechanisms do not record spends. The cost is the `epsilon` passed, given a correct `sensitivity`. Use `acc.check(epsilon, 0)` / `acc.spend(epsilon, 0)` and report `acc.total()`.

## Pitfalls
- These variants are biased near the bounds (the output is pulled toward the middle); plain `Laplace` is unbiased but can leave the range.
- `lower`/`upper` must be public. Using the observed min/max of the data leaks them.
- `delta` must stay 0 for pure DP (it defaults to 0.0).
- `tools.mean` already uses `LaplaceTruncated` internally; do not add a second mechanism on top of a tools result.
