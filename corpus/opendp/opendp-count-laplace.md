---
id: opendp-count-laplace
title: DP count of rows with the Laplace mechanism (Framework API, or the Context API)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.transformations.then_count
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.context.loss_of
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.LazyFrameQuery.filter
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.release
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/api-comparison/code/typical-workflow-framework.rst
source_license: MIT
---

## Goal
Release the number of records meeting a condition with epsilon-DP. Neighbouring relation: replace-one (n public), the default: one changed row moves the count by at most 1. Because n is public, a proportion or rate is this DP count divided by the public total n.

## Steps
Framework API (what the skeleton uses):
1. `enable_features("contrib")`.
2. Row-wise flags, no budget: `FLAGS <- (DF[COL] == VALUE).astype(int).tolist()`.
3. `space <- (vector_domain(atom_domain(T=int), size=N), symmetric_distance())`: a 2-tuple of (domain, metric), N = the public row count. The metric is NOT an argument of `vector_domain`.
4. `meas <- binary_search_chain(lambda s: space >> then_clamp((0, 1)) >> then_sum() >> then_laplace(s), d_in=2, d_out=EPS)`: replace-one = symmetric distance 2; OpenDP charges this sized 0/1 sum 1.
5. `dp_count <- meas(FLAGS)`: an integer (discrete Laplace noise).

Context API (alternative): `context <- Context.compositor(data=pl.from_pandas(DF).lazy(), privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=1)`, then `context.query().filter(COL == VALUE).select(dp.len(signed=True)).release().collect().item()`; epsilon = `context.accountant.map(context.d_in)`. Under `changes=1` OpenDP charges a filtered `dp.len` 2 (conservative: scale 2/EPS), so the Framework route above is twice as accurate.

The true count, for comparison only, is plain pandas: `TRUE <- int((DF[COL] == VALUE).sum())`.

## Privacy accounting
Replace-one: `binary_search_chain` picks the tightest scale with `meas.map(2) <= EPS`; assert it and report `meas.map(2)`. Do not derive the scale by hand.
Alternative, add/remove-one (n private): unsized domain `vector_domain(atom_domain(T=int))`, `d_in=1`; the count is also charged 1 (`map(1) == 1`), but n is then private, so do not divide by `len(DF)` as if it were public. In the Context this is `unit_of(contributions=1)` (filtered `dp.len` scale 1/EPS).

## Pitfalls
- With n public the plain row count is known; release counts of rows that meet a condition.
- `d_in=1` on symmetric distance is add/remove; using it with n public is exact for this single count but halves the epsilon of histograms and group-bys. Keep `d_in=2` for replace-one.
- Polars: a **DataFrame** has no `.collect()`; only a LazyFrame has. `.item()` works only on a 1x1 table. Count with `dp.len(signed=True)`, not `.count()` or `pl.len()`.
- "Predicate in binary search always raises an exception" means the chain itself fails: usually `T` does not match the data (list of str vs int), or a metric inside `vector_domain`.
- A sized domain rejects data whose length differs from N.
- A DP count can be negative or exceed n. Clip only after release (post-processing).
