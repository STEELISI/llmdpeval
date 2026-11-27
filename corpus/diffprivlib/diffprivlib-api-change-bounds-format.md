---
id: diffprivlib-api-change-bounds-format
title: "The bounds argument format in 0.6.6: (lower, upper) as a tuple, vectors for several columns"
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.validation.check_bounds
  - diffprivlib.tools.mean
  - diffprivlib.models.LinearRegression
  - diffprivlib.models.GaussianNB
  - diffprivlib.tools.histogram2d
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/validation.py
source_license: MIT
---

## Goal
Pass bounds in the one shape every diffprivlib 0.6.6 tool and model accepts, so they are neither rejected nor silently misread. Derived from observed model errors. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. One column / scalar statistic: `bounds=(L, U)`, a TUPLE of two numbers (tools: `mean`, `sum`, `var`, `std`, `quantile`, `median`, ...).
2. Several columns: `bounds=(L_vec, U_vec)`, a tuple of two equal-length vectors, one entry per column (models: `GaussianNB`, `KMeans`, `StandardScaler`, `PCA`, forests; `LinearRegression(bounds_X=...)`; tools with `axis=0`). A scalar pair is broadcast to every column.
3. `LinearRegression` takes two arguments: `bounds_X=(L_vec, U_vec)` for the features and `bounds_y=(L, U)` for the target.
4. Histograms use `range=` (not `bounds=`): `range=(L, U)` for `histogram`, `range=((Lx, Ux), (Ly, Uy))` for `histogram2d`, a list of pairs for `histogramdd` (this is the NumPy convention: per-axis PAIRS here, unlike `bounds`).
5. To normalise and validate a user-supplied value the way the library does: `validation.check_bounds(bounds, shape=d)` returns two length-d arrays.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Bounds set the sensitivity inside the library. Wrong-shaped but accepted bounds give a release whose accountant entry looks right while the noise is calibrated to the wrong range.

## Pitfalls
- A list `[L, U]` raises `TypeError: Bounds must be specified as a tuple of (min, max)`.
- A 1-tuple `(L,)` raises `ValueError: not enough values to unpack (expected 2, got 1)`.
- Per-column pairs `((L1, U1), (L2, U2))` passed as `bounds` are read as lowers=(L1, U1), uppers=(L2, U2): accepted silently when each lower <= upper, and wrong.
- `ValueError: lower and upper bounds must have D element(s), got M.`: a model or `axis=0` tool got bound vectors whose length M differs from the number of columns D (one entry per column, intercept or one-hot columns included).
- `ValueError: lower and upper bounds must be scalar or a 1-dimensional array`: per-row or nested bounds; pass one vector of lowers and one of uppers.
- `bounds=None` (or a misspelled keyword, which is ignored) makes the library read min/max from the data with a `PrivacyLeakWarning`.
