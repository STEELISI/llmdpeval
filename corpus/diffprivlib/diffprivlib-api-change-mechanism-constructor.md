---
id: diffprivlib-api-change-mechanism-constructor
title: "Mechanisms take keyword arguments at construction (no set_epsilon / set_sensitivity builders since 0.4)"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.mechanisms.Geometric
  - diffprivlib.mechanisms.Gaussian
  - diffprivlib.mechanisms.DPMechanism
source_url: https://github.com/IBM/differential-privacy-library/releases/tag/0.4.0
source_license: MIT
---

## Goal
Build mechanism objects with the post-0.4 API that 0.6.6 uses. Older tutorials and many model outputs use the removed builder style. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Removed in 0.4.0 (release notes): `Laplace().set_epsilon(1).set_sensitivity(1)` and similar `set_*` chains. In 0.6.6 `Laplace()` with no arguments raises `TypeError: missing 2 required keyword-only arguments`, and `set_epsilon` raises AttributeError.
2. 0.6.6 form, all keyword-only: `mech <- mechanisms.Laplace(epsilon=?, sensitivity=?)`, `mechanisms.LaplaceBoundedDomain(epsilon=?, sensitivity=?, lower=?, upper=?)`, `mechanisms.Gaussian(epsilon=?, delta=?, sensitivity=?)`, `mechanisms.Geometric(epsilon=?, sensitivity=?)` (integers). All of them subclass `mechanisms.DPMechanism`, whose constructor takes `epsilon` and `delta` as keywords.
3. Then `mech.randomise(value)`; one number in, one number out (an array raises `TypeError: Value to be randomised must be a number`: loop over the entries, or use the tools for vectors) (except `Exponential`/`PermuteAndFlip`, which take no value, and `Vector`/`Bingham`, which take a function / a matrix).
4. Since 0.5.0 models are keyword-only too: `models.LogisticRegression(epsilon=?, data_norm=1, fit_intercept=False)` (rows scaled to norm exactly 1 with an explicit intercept column: in 0.6.6 the epsilon accounting is only exact when the row norm bound is 1), not `LogisticRegression(1.0, 2.0)` (TypeError).
5. Since 0.6.0 every tool, model and mechanism accepts `random_state=`; do not set it in a real release.
6. Module name is `diffprivlib.mechanisms` (plural); `diffprivlib.mechanism` raises ModuleNotFoundError.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Mechanism objects never charge an accountant in any version: spend the epsilon (and delta) on a `BudgetAccountant` yourself.

## Pitfalls
- Mechanisms take no `accountant=` (`TypeError: Laplace.__init__() got an unexpected keyword argument 'accountant'`); only tools and models do. After `randomise`, call `acc.spend(epsilon, delta)`.
- Positional arguments to a mechanism, e.g. `Laplace(1.0, 1.0)`, raise TypeError.
- `Gaussian` requires `delta > 0` and raises for `epsilon > 1`; its calibration (Dwork and Roth, Theorem 3.22) is proved for epsilon in (0, 1), so at epsilon = 1 or above use `GaussianAnalytic`.
- `Wishart` was removed in 0.5.0 (it did not satisfy DP).
