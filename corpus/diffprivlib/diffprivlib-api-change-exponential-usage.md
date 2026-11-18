---
id: diffprivlib-api-change-exponential-usage
title: "Exponential mechanism in 0.6.6: mechanisms.Exponential, utility as a list, randomise() with no value"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Exponential
  - diffprivlib.mechanisms.PermuteAndFlip
  - diffprivlib.mechanisms.ExponentialCategorical
source_url: https://github.com/IBM/differential-privacy-library/releases/tag/0.4.0
source_license: MIT
---

## Goal
Call the exponential mechanism with the 0.6.6 signature. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `Exponential(*, epsilon, sensitivity, utility, monotonic=False, candidates=None, measure=None, random_state=None)`

## Steps
1. The class is `diffprivlib.mechanisms.Exponential`. There is no `diffprivlib.tools.ExponentialMechanism` (AttributeError) and no `ExponentialMechanism` class anywhere.
2. `utility` must be a Python LIST of numbers, one per candidate. A NumPy array raises `TypeError: Utility must be a list, got a [...]`, and a lambda raises `TypeError: Utility must be a list, got a <function ...>`. Convert with `list(map(float, scores))`; utilities are computed by you before construction.
3. `candidates` is an optional list of labels of the same length; without it the mechanism returns an index.
4. `mech.randomise()` takes NO value (the scores are already inside the mechanism). Passing anything, e.g. a `random.Random()` object, raises `ValueError: Value to be randomised must be None`.
5. History: in 0.4.0 the old categorical exponential mechanism was renamed `ExponentialCategorical` (string domain, distance list `utility_list`) and the new `Exponential` / `PermuteAndFlip` for selection were added; 0.5.0 added `monotonic=`.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. `Exponential` and `PermuteAndFlip` do not touch any accountant; spend `epsilon` on a `BudgetAccountant` yourself (`acc.spend(epsilon, 0)`), once per selection.

## Pitfalls
- All constructor arguments are keyword-only.
- `sensitivity` is the sensitivity of the utility scores (1 for counts), not of the output.
- Higher utility means more likely in `Exponential`; in `ExponentialCategorical` larger values in `utility_list` mean LESS likely (they are distances).
