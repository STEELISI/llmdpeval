---
id: diffprivlib-geometric-integer-count
title: Integer-valued noise for counts with the Geometric mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Geometric
  - diffprivlib.mechanisms.GeometricTruncated
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/geometric.py
source_license: MIT
---

## Goal
Add discrete (two-sided geometric, "discrete Laplace") noise to an integer count you computed yourself, with pure epsilon-DP. Useful when the count is a custom query not covered by `tools.count_nonzero`. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `GeometricTruncated(*, epsilon, sensitivity=1, lower, upper, random_state=None)`

## Steps
1. `true_count <- exact count`, cast to a Python int.
2. State the sensitivity SENS: 1 for a plain row count. If one person can contribute several rows, cap contributions per person first; SENS is that cap.
3. `mech <- mechanisms.Geometric(epsilon=?, sensitivity=SENS)`; `sensitivity` must be an int.
4. `dp_count <- mech.randomise(true_count)`
5. To keep the output in a valid range: `mech <- mechanisms.GeometricTruncated(epsilon=?, sensitivity=SENS, lower=0, upper=N_MAX)`, with integer bounds. N_MAX must be public (a known population size), not the observed count.
6. `acc.spend(?, 0)` with the same epsilon.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes the count by at most 1 (c for a public per-person cap c); the same holds under add/remove. Mechanism objects do not spend budget. The cost is the `epsilon` passed, assuming the sensitivity is right. Check with `acc.check(epsilon, 0)`, spend with `acc.spend(epsilon, 0)`, and report `acc.total()`.

## Pitfalls
- `randomise` raises TypeError for floats (e.g. a NumPy float from a mean); pass an int.
- `delta` must be 0; Geometric does not accept a nonzero delta.
- `tools.count_nonzero` already applies this mechanism and spends on the accountant; prefer it for simple counts.
- Clamping a noisy count at 0 afterwards is allowed (post-processing); `GeometricTruncated` does this for you.
