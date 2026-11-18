---
id: opendp-context-compositor-framework
title: dp.Context.compositor over a plain list (total epsilon split across queries)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: composition
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.Context.query
  - opendp.context.Context.remaining_privacy_loss
  - opendp.context.Query.release
  - opendp.context.Query.param
  - opendp.context.unit_of
  - opendp.context.loss_of
  - opendp.accuracy.discrete_laplacian_scale_to_accuracy
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/code/typical-workflow-context.rst
source_license: MIT
---

## Goal
Answer several queries on one column (a Python list) under a single total budget that OpenDP enforces. The Context derives each query's noise scale from its budget share, so no scale search is needed. Neighbouring relation: replace-one (n public), the default: privacy unit `unit_of(contributions=2)` (one changed row = symmetric distance 2) on a sized domain.

## Steps
1. `enable_features("contrib")`.
2. Privacy unit for a plain list (symmetric distance): replace-one is `unit_of(contributions=2)`; if a person can change R rows, `contributions=2*R`. (`unit_of(changes=1)` is the polars LazyFrame form; vectors do not accept it.) Alternative, add/remove-one (n private): `unit_of(contributions=1)`.
3. Total loss: `loss_of(epsilon=EPS)` (Laplace), `loss_of(rho=RHO)` (Gaussian) or `loss_of(epsilon=EPS, delta=DELTA)`.
4. `context <- Context.compositor(data=VALUES_AS_LIST, privacy_unit=UNIT, privacy_loss=LOSS, split_evenly_over=M, domain=vector_domain(atom_domain(T=float, nan=False), size=N))` (N = public row count), or `split_by_weights=[W_1, ..., W_M]` for an uneven split.
5. Each query chains transformation names without the `then_` prefix and ends in a noise step:
   - sum: `q <- context.query().clamp((L, U)).sum().laplace()` (charged U - L on the sized domain)
   - mean: `q <- context.query().clamp((L, U)).mean().laplace()` (charged (U - L) / N)
   - add/remove alternative (n private): `context.query().count().laplace()` first, then `.clamp((L, U)).resize(size=DP_N, constant=C).mean().laplace()` with the released DP count.
6. Before releasing: `q.param()` gives the scale the context chose; turn it into an error bound with `discrete_laplacian_scale_to_accuracy(scale=?, alpha=?)` (or the float version).
7. `value <- q.release()`. Each release uses one slot of the split.

Signature for reference: `Context.compositor(data, privacy_unit, privacy_loss, split_evenly_over=None, split_by_weights=None, domain=None, margins=None)`.

## Privacy accounting
The compositor holds the total; report `context.accountant.map(context.d_in)` (= the unit: 2 under replace-one, 1 under add/remove). `split_evenly_over=M` gives each query EPS / M; the context finds the scale so each query's map fits its slot. After M releases, a further `release()` raises "Privacy allowance has been exhausted". `context.remaining_privacy_loss()` (a method) shows what is left.

## Pitfalls
- M must equal the number of queries you will release. Too small: later queries fail. Too large: budget is wasted.
- `remaining_privacy_loss` and `current_privacy_loss` are methods; call them with `()`.
- Under replace-one n is public: declare `size=N` and skip the DP count. Under add/remove (n private) use the DP count for `resize(size=...)`, never `len(data)`.
- `unit_of(contributions=1)` with n public halves the epsilon for histograms and group-bys (on polars LazyFrames also for sums whose bounds straddle 0).
- Bounds and imputation constants must be public.
- For tables with several columns use a polars LazyFrame as `data` (see `opendp-polars-context-basic`).
