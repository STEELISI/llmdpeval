---
id: diffprivlib-api-change-unused-args-silent
title: "Misspelled or sklearn-only keyword arguments are silently ignored (bounds_x, solver, criterion)"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.utils.DiffprivlibCompatibilityWarning
  - diffprivlib.utils.PrivacyLeakWarning
  - diffprivlib.utils.warn_unused_args
  - diffprivlib.models.LinearRegression
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/utils.py
source_license: MIT
---

## Goal
Catch argument-name mistakes that do not crash. Most diffprivlib tools and models end their signature with `**unused_args`, so an unknown keyword is accepted, a `DiffprivlibCompatibilityWarning` is printed, and the call goes on without it. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Observed in generated code: `LinearRegression(..., bounds_x=...)` (lower-case x). The model warned `Parameter 'bounds_x' is not functional in diffprivlib`, ignored it, then found `bounds_X=None` and fell back to bounds computed from the data with a `PrivacyLeakWarning`. The script ran and printed a result that was not DP.
2. Exact 0.6.6 spellings: `bounds_X` and `bounds_y` (LinearRegression), `data_norm` (LogisticRegression, PCA), `bounds` (tools, GaussianNB, KMeans, StandardScaler, PCA, forests), `range` (histograms), `classes` (forests).
3. Make both warnings fatal at the top of the script so such mistakes stop the run:
   - `warnings.simplefilter("error", PrivacyLeakWarning)`
   - `warnings.simplefilter("error", DiffprivlibCompatibilityWarning)`
4. The warning comes from `diffprivlib.utils.warn_unused_args`, which every model and tool calls on unknown arguments. Also ignored with this warning: sklearn options such as `solver`, `penalty`, `multi_class` (LogisticRegression), `criterion`, `min_samples_leaf` (trees), and the per-sample argument of `fit`.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Not affected directly, but an ignored bounds argument turns the release into a non-DP one while `acc.total()` still reports the epsilon as if it were DP. The accountant cannot detect this; only the warnings can.

## Pitfalls
- Promoting `DiffprivlibCompatibilityWarning` to an error also stops on harmless sklearn options; remove those options instead of suppressing the warning.
- Do not make ALL warnings errors (`warnings.simplefilter("error")` with no category): with SciPy 1.17, `models.LogisticRegression.fit` then stops on SciPy's `DeprecationWarning` about the L-BFGS-B `iprint` option. Name the two diffprivlib categories only.
- `warnings.filterwarnings("ignore")` hides exactly these signals; never use it in DP code.
- `bounds_X=(L, U)` must be a 2-tuple; a list raises `TypeError: Bounds must be specified as a tuple`, and a 1-tuple `(L,)` raises `ValueError: not enough values to unpack (expected 2, got 1)`.
