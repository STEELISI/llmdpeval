---
id: numpy-errors-noise-arrays-pandas3
title: Common errors when adding noise to pandas/NumPy arrays (pandas 3, NumPy 2) and testing noisy counts (error message -> cause -> fix)
framework: numpy
framework_version: "any"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.laplace
  - numpy.nan_to_num
  - numpy.isnan
  - pandas.DataFrame.to_numpy
  - scipy.stats.fisher_exact
source_url: https://pandas.pydata.org/docs/user_guide/copy_on_write.html
source_license: BSD-3-Clause (pandas docs, paraphrased)
---

## Goal
Map errors seen when DP code adds noise to arrays taken from pandas DataFrames, or tests noisy counts, to their cause and fix. Reproduced with pandas 3.0, NumPy 2.4, SciPy 1.17.

## Steps
1. `ValueError: output array is read-only` from `DF.values += NOISE`, or `assignment destination is read-only` from `nan_to_num(DF.to_numpy(), copy=False)` -> pandas 3 (copy-on-write) can hand out read-only views. Fix: `ARR <- DF.to_numpy(dtype=float, copy=True)` and work on ARR, or build a new frame: `NOISY <- DF + laplace(0, SCALE, DF.shape)`.
2. `Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')` from `COUNTS += NOISE` -> in-place add on an integer array. Fix: `COUNTS <- COUNTS.astype(float) + NOISE`.
3. `ufunc 'isnan' not supported for the input types` -> an object-dtype array (mixed None/float, or built from a list of arrays). Fix: `ARR <- ARR.astype(float)` (None becomes NaN) before `isnan`.
4. `ValueError: All values in table must be nonnegative` from `fisher_exact` -> noisy cells can be negative. Clamping is not enough: an exact test on noisy counts ignores the noise and gives far too many significant results. Use a noise-aware test (sampling variance plus DP noise variance) on the clamped noisy cells. SciPy also accepts non-integer cells silently, which hides the problem.
5. `Length of values (A) does not match length of index (B)` -> two arrays on different item lists (the full public list vs a filtered one). Fix: reindex every count vector to ONE public item list (`Series.reindex(ITEMS, fill_value=0)`) before combining.
6. `KeyError` on a year or label used as a column, or `index must be a MultiIndex to unstack` -> group by the two keys together (`groupby([KEY1, KEY2]).size().unstack(fill_value=0)`), then `reindex` rows and columns to the public lists.

## Privacy accounting
None of these fixes change the privacy: the epsilon still follows from the stated sensitivity and the noise scale.

## Pitfalls
- Fixing a crash by dropping the noise, or by computing on exact values "just for the test".
- Reindexing to the values seen in the data instead of the public list: the set of values that occur can be private.
