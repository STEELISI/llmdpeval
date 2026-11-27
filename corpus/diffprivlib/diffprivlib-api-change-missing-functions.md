---
id: diffprivlib-api-change-missing-functions
title: "What diffprivlib 0.6.6 does NOT have: the complete tools and models lists"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.count_nonzero
  - diffprivlib.tools.histogram
  - diffprivlib.tools.quantile
  - diffprivlib.tools.nanmean
  - diffprivlib.models.RandomForestClassifier
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.validation.clip_to_bounds
  - diffprivlib.tools.sum
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/docs/modules/tools.rst
source_license: MIT
---

## Goal
Stop guessing function names. These are the complete public lists in 0.6.6 (from `inspect` on the installed package); anything else in these modules does not exist. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. `diffprivlib.tools` (15 functions): `count_nonzero`, `histogram`, `histogram2d`, `histogramdd`, `mean`, `nanmean`, `median`, `percentile`, `quantile`, `std`, `nanstd`, `sum`, `nansum`, `var`, `nanvar`.
   - Not present: `count`, `laplace` (`from diffprivlib.tools import laplace` raises ImportError), `noisy_count`, `nanmedian`, `nanquantile`, `nanpercentile`, `max`, `min`, `cov`, `corrcoef`, `top_k`, any `ExponentialMechanism`.
2. `diffprivlib.models` (8 classes): `LogisticRegression`, `LinearRegression`, `GaussianNB`, `KMeans`, `PCA`, `StandardScaler`, `RandomForestClassifier`, `DecisionTreeClassifier`.
   - Not present: `RandomForestRegressor`, `LinearSVC`, `Ridge`, `Lasso`, `MLPClassifier`, `DBSCAN`.
3. `diffprivlib.mechanisms`: `Binary`, `Bingham`, `Exponential`, `ExponentialCategorical`, `ExponentialHierarchical`, `Gaussian`, `GaussianAnalytic`, `GaussianDiscrete`, `Geometric`, `GeometricFolded`, `GeometricTruncated`, `Laplace`, `LaplaceBoundedDomain`, `LaplaceBoundedNoise`, `LaplaceFolded`, `LaplaceTruncated`, `PermuteAndFlip`, `Snapping`, `Staircase`, `Uniform`, `Vector` (plus base classes `DPMechanism`, `DPMachine`, `TruncationAndFoldingMixin`). Transforms are in `diffprivlib.mechanisms.transforms`.
   - Not present: `LaplaceBoundedSum`, `LaplaceMechanism`, any `*Sum` or `*Count` mechanism (a bounded sum is `tools.sum(..., bounds=(L, U))`).
   - `clip_to_bounds(array, bounds)` lives in `diffprivlib.validation`, not in `diffprivlib.mechanisms.transforms` (ImportError there).
4. `tools.count_nonzero(array, epsilon=1.0, axis=None, keepdims=False, random_state=None, accountant=None)` returns ONE number (an int), not a tuple: `a, b = count_nonzero(...)` raises `TypeError: cannot unpack non-iterable int object`. It takes no `bounds=` (TypeError: unexpected keyword argument); a count needs none.
5. Missing a statistic? Build it: clip to public bounds, compute the exact value, state its sensitivity, add noise with a mechanism, spend on an accountant. A min/max is better released as a high/low quantile.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. For built-from-mechanism releases, the epsilon is what you pass to the mechanism, valid only if your sensitivity is right; record it with `acc.spend`.

## Pitfalls
- `RandomForestClassifier(feature_domains=...)` is the pre-0.6.0 API; since 0.6.0 it takes `bounds=` (and `classes=`). In 0.6.6 `feature_domains` is silently ignored with a `DiffprivlibCompatibilityWarning`.
- There is no top-k helper; repeat `Exponential` with a split budget.
- `normed=` no longer works: in 0.6.6 it is silently ignored with a `DiffprivlibCompatibilityWarning` (no error, and the output is NOT normalised); use `density=`.
