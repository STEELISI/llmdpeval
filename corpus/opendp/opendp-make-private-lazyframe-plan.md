---
id: opendp-make-private-lazyframe-plan
title: Measurement from a polars LazyFrame plan without the Context (make_private_lazyframe)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: sum
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.measurements.make_private_lazyframe
  - opendp.domains.lazyframe_domain
  - opendp.domains.series_domain
  - opendp.domains.with_margin
  - opendp.extras.polars.Margin
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.sum
  - opendp.mod.binary_search_chain
  - opendp.metrics.change_one_distance
  - opendp.metrics.symmetric_distance
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/test_polars.py
source_license: MIT
---

## Goal
Build one OpenDP measurement directly from a polars query plan: describe the data with a LazyFrameDomain, write the query with `.dp` expressions, and search a single noise knob to hit epsilon. Useful when you want Framework-API control (composition, binary search) over polars queries. Neighbouring relation: replace-one (n public), the default: input metric `change_one_distance()` with `d_in=1` (one changed row).

## Steps
1. `enable_features("contrib")`.
2. Domain from public schema facts: `domain <- lazyframe_domain([series_domain(X, atom_domain(T=f64))])`, then `domain <- with_margin(domain, polars.Margin(by=[], max_length=N_MAX))`.
3. Plan on an EMPTY frame with the same schema: `plan <- LazyFrame(schema=LF.collect_schema()).select(dp.len(signed=True), col(X).dp.sum((L, U)))`.
4. `make <- lambda s: make_private_lazyframe(domain, change_one_distance(), max_divergence(), plan, global_scale=s)`
5. `meas <- binary_search_chain(make, d_in=1, d_out=EPS)` (d_in=1 = one changed row on `change_one_distance`).
6. `out <- meas(LF).collect()`; LF's columns and dtypes must match the domain (cast before).

Signature for reference: `make_private_lazyframe(input_domain, input_metric, output_measure, lazyframe, global_scale=None, threshold=None)`.

## Privacy accounting
`global_scale` multiplies the noise of every DP expression in the plan; `meas.map(1)` on `change_one_distance` is the total replace-one epsilon; `dp.len` costs nothing there, because n is public. Alternative, add/remove-one (n private): `symmetric_distance()` with `d_in=1`; `dp.len` then gets noise and a sum with bounds that straddle 0 is charged max(|L|, |U|) per row instead of U - L. Use `zero_concentrated_divergence()` for Gaussian noise (map returns rho). Group-by without public keys also needs `threshold`.

## Pitfalls
- Never build the plan on the private frame; only the schema is public.
- `symmetric_distance()` with `d_in=1` is add/remove; with n public it under-charges group-bys and sums whose bounds straddle 0 by 2x.
- Series dtypes must match exactly (f64 vs i64 fails at call time).
- Without a `max_length` margin some aggregates cannot bound their sensitivity.
- The Context API (`opendp-polars-context-basic`) does steps 2-5 for you.
