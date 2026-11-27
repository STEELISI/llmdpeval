---
id: opendp-errors-polars-context
title: Common errors with OpenDP Context and polars, and their fixes (error message -> cause -> fix)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.extras.polars.Margin
  - opendp.extras.polars.LazyFrameQuery
  - opendp.context.Context.compositor
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.release
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Map the error messages that OpenDP 0.16.0 + polars 1.36 code most often raises to their cause and fix.

## Steps
1. `AttributeError: 'DataFrame' object has no attribute 'collect'` -> `.collect()` called on an eager polars DataFrame. Fix: call it only on a LazyFrame (`DF.lazy()` / `scan_csv`), or drop it; for a plain true answer use pandas `(DF[COL] == VALUE).sum()`.
2. `AttributeError: 'DataFrame' object has no attribute 'to_scalar'` -> not a polars method. Fix: `FRAME.select(pl.len()).item()` on a 1x1 result.
3. `ValueError: can only call .item() ... if the DataFrame has a single element` -> `.count()` on a frame returns one count PER COLUMN. Fix: `select(pl.len())` first, then `.item()`.
4. `TypeError: Context.__init__() got an unexpected keyword argument 'data'` -> constructor used directly. Fix: `Context.compositor(data=LF, privacy_unit=..., privacy_loss=..., split_evenly_over=K)`.
5. Printed output looks like `OnceFrame` / a query plan instead of a number -> the release was never collected. Fix: `query.release().collect().item()`.
6. `Predicate in binary search always raises an exception` -> building the chain fails for every scale. Fix: build the chain once with a fixed scale (e.g. 1.0) to see the real error; check `T` against the data, and use `dp.len(signed=True)` rather than `.count()` in Context queries.
7. `TypeError: inferred type is SymmetricDistance, expected i32` -> the metric was passed inside `vector_domain(...)`. Fix: `space = (vector_domain(atom_domain(T=str)), symmetric_distance())`.
8. `Unable to infer type T; pass the type T or bounds` -> OpenDP cannot guess the element type. Fix: pass `T=` explicitly or `bounds=(L, U)` of the right type.
9. `KeyError: 'Columns not found: ...'` -> a column name built from data values or booleans. Fix: select columns by their real names; turn a condition into a filter, not a column.
10. `'LazyFrameQuery' object has no attribute 'agg'` -> aggregate with `query.select(...)`, or `query.group_by(KEYS).agg(...)`.
11. `'OnceFrame' object has no attribute 'item'` -> `release()` returns a OnceFrame: call `.collect()` first, then `.item()`.
12. `Expr is not recognized at this time: col(X).sum()` (or `.strict_cast(Int64).sum()`, `.mean()`) -> a plain polars aggregate in a DP query. Use the DP forms: `pl.col(X).dp.sum((L, U))`, `pl.col(X).dp.mean((L, U))`, `dp.len(signed=True)`.
13. `Expr is not recognized at this time: col(X).pow(...)` -> the Context cannot square or transform with every expression. Add the derived column (e.g. X squared) to the LazyFrame BEFORE building the Context, with public bounds for it.
14. `The key-set of {col(G)} is private and cannot be released` -> a `group_by` whose keys are not public. Declare them: `margins=[dp.polars.Margin(by=[G], invariant="keys", max_length=N)]` with the public key list, or join against a public key frame.
15. `max_groups must be known when the metric is not sensitive to ordering` -> a group-by sum/mean needs the number of groups: `Margin(by=[G], invariant="keys", max_length=N, max_groups=K)` with K from the public key list.
16. `module 'opendp.extras.polars' has no attribute 'from_pandas'` / `No module named 'opendp.polars'` -> convert with polars itself, `LF <- pl.from_pandas(DF).lazy()`; Margin lives at `dp.polars.Margin`.

## Privacy accounting
None of these fixes change the privacy: after fixing, still read epsilon from `context.accountant.map(1)` or `meas.map(d_in)`.

## Pitfalls
- Do not catch these errors with try/except and print a message: the script then releases nothing.
- Do not swap in a synthetic table when the data file is missing; read the given file.
