---
id: diffprivlib-snapping-mechanism
title: Floating-point-safe Laplace noise with the Snapping mechanism
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.mechanisms.Snapping
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/snapping.py
source_license: MIT
---

## Goal
Release a real-valued statistic with pure epsilon-DP using Mironov's snapping mechanism (CCS 2012), which closes the floating-point side channel of textbook Laplace sampling by clamping and rounding the output to a coarse grid. Neighbouring relation: replace-one (bounded DP, n public); `sensitivity` must be the replace-one sensitivity of the statistic.

Signature: `Snapping(*, epsilon, sensitivity, lower, upper, random_state=None)`

## Steps
1. Compute the statistic from data clipped to public bounds; state its sensitivity.
2. Public output interval `[L, U]` that contains every plausible true value; the mechanism clamps both input and output to it.
3. `mech <- mechanisms.Snapping(epsilon=?, sensitivity=SENS, lower=L, upper=U)` (keyword-only).
4. `dp_value <- mech.randomise(true_value)`.
   - Outputs are multiples of a power of two at least as large as the noise scale, so they look coarse (e.g. only odd integers or multiples of 2). That is by design.
5. `acc.spend(epsilon, 0)`; mechanisms do not charge an accountant.
6. Output the true value, DP value, `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public): one replaced value moves the total by at most U - L, the sensitivity passed. Add/remove alternative (n private): use max(|L|, |U|) as the sensitivity. The guarantee is epsilon-DP for the epsilon passed: the library samples with a slightly smaller `mech.effective_epsilon()` (Mironov, section 5.2) to absorb the rounding error, which is why `epsilon` must exceed twice machine epsilon. Record `acc.spend(epsilon, 0)` (the epsilon passed, not the effective one) and report `acc.total()`.

## Pitfalls
- The docstring recommends installing the optional `crlibm` package for the most faithful implementation; without it a fallback log is used. Do not pip-install into a pinned environment without approval; just note it.
- Coarse outputs are expected; do not "fix" them by adding more noise or re-rounding.
- A too-narrow [L, U] clamps real values; a too-wide one increases the grid size. Choose it from domain knowledge.
- For integer counts, `Geometric` already avoids floating-point issues and is simpler.
