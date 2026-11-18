---
id: diffprivlib-histogram2d-joint
title: Joint 2-D histogram of two numeric columns with tools.histogram2d
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.tools.histogram2d
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/notebooks/exploration.ipynb
source_license: MIT
---

## Goal
Release a noisy contingency table (joint bin counts) of two numeric columns, for example to show how one variable is distributed within ranges of another. Pure epsilon-DP for replace-one neighbours (bounded DP, n public).

Signature (abridged): `histogram2d(array_x, array_y, epsilon=1.0, bins=10, range=None, density=None, random_state=None, accountant=None, ...)`

## Steps
1. Fix bins per axis from public knowledge: `bins=(Bx, By)` integers plus `range=((Lx, Ux), (Ly, Uy))`, or explicit edge arrays `bins=(EDGES_X, EDGES_Y)`.
   - A small integer-coded column (codes 0..K-1) works as an axis with `bins=K` and range `(0, K)`.
2. `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`
3. `(dp_counts, edges_x, edges_y) <- tools.histogram2d(x, y, epsilon=EPS/2, bins=(Bx, By), range=((Lx, Ux), (Ly, Uy)), accountant=acc)`
   - Returns THREE values (counts of shape (Bx, By), x edges, y edges), unlike `tools.histogram` which returns two.
   - Every cell gets geometric noise truncated at 0, so counts are non-negative integers (as floats).
4. Optional post-processing: divide each row by its row total for conditional shares; any arithmetic on the noisy table is free.
5. Output: true table from `numpy.histogram2d` with the same bins and range, DP table, `acc.total()`.

## Privacy accounting
One call covers the whole table. diffprivlib calibrates it for add/remove (each record in one cell). Under replace-one (bounded DP, n public) a changed record leaves one cell and enters another, so pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`.

## Pitfalls
- Unpacking into two names (`h, e = histogram2d(...)`) raises "too many values to unpack".
- `range=None`, or a LIST range with a None axis, → `PrivacyLeakWarning`, range taken from the data. A TUPLE range with a None axis, e.g. `(None, (Ly, Uy))`, is NOT caught: that axis's range silently comes from the data.
- Passing the full EPS under replace-one gives a 2 x EPS guarantee. The full EPS is right only for add/remove-one (n private).
- The number of cells is Bx*By; with many cells and few rows most noisy cells are pure noise. Keep bins coarse.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
