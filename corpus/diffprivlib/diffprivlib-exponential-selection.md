---
id: diffprivlib-exponential-selection
title: Selecting the best candidate with the Exponential mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Exponential
  - diffprivlib.mechanisms.PermuteAndFlip
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/exponential.py
source_license: MIT
---

## Goal
Pick one item from a public list of candidates (the most common category, the best threshold) so that higher-scoring items are more likely, with pure epsilon-DP (McSherry and Talwar, FOCS 2007). Only the chosen label is released, not the scores. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `Exponential(*, epsilon, sensitivity, utility, monotonic=False, candidates=None, measure=None, random_state=None)`

## Steps
1. `CANDS <- [c_1, ..., c_K]`: from public knowledge, in a fixed order. Never from the private data's unique values.
2. `UTIL <- [u_1, ..., u_K]`: a score per candidate computed on the private data (e.g. its row count), as a plain Python list of floats.
3. State the utility's sensitivity SENS: how much one record can change any single score. A count has sensitivity 1. A fraction or rate has sensitivity that depends on the group size; use a count-based utility when you can.
4. `mech <- mechanisms.Exponential(epsilon=?, sensitivity=SENS, utility=UTIL, candidates=CANDS)`
5. `choice <- mech.randomise()` returns a label from `candidates` (or an index if `candidates` is omitted).
6. Alternative with the same arguments: `mechanisms.PermuteAndFlip(...)`, never worse in expected utility (McKenna and Sheldon, NeurIPS 2020).
7. `acc.spend(?, 0)` with the same epsilon.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced row changes each count score by at most 1, but two scores move in opposite directions; `Exponential` with `monotonic=False` keeps the factor 2 that covers this (also valid under add/remove). Mechanism objects do not touch the accountant. One `randomise()` call costs the `epsilon` passed, given the right `sensitivity`. Use `acc.check` / `acc.spend` on a `BudgetAccountant(epsilon=EPS_TOTAL, delta=0)` and report `acc.total()`. Calling `randomise()` again on the same object is a new release and costs epsilon again.

## Pitfalls
- `utility` must be a Python list; a NumPy array raises TypeError.
- `monotonic=True` lowers the noise, but only if adding a record can only increase every score (true for counts under add/remove, false under replace-one).
- Releasing the true scores alongside the choice defeats the mechanism.
- Small groups make rate-based utilities very sensitive; a candidate with one record can win by chance.
