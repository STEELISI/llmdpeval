---
id: opendp-polars-zcdp-gaussian-context
title: Polars Context under zCDP - loss_of(rho) gives Gaussian noise; convert total to (epsilon, delta)
framework: opendp
framework_version: "0.16.0"
mechanism: gaussian
task_type: sum
dp_notion: zcdp
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.loss_of
  - opendp.context.unit_of
  - opendp.extras.polars.Margin
  - opendp.extras.polars.dp_len
  - opendp.extras.polars.DPExpr.sum
  - opendp.extras.polars.LazyFrameQuery.select
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.release
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/polars/gaussian-noise.rst
source_license: MIT
---

## Goal
Run polars Context queries with Gaussian instead of Laplace noise. OpenDP picks the noise from the privacy notion: `loss_of(epsilon=...)` gives Laplace, `loss_of(rho=...)` gives Gaussian. Neighbouring relation: replace-one (n public), the default: `unit_of(changes=1)`; alternative, add/remove-one (n private): `unit_of(contributions=1)`.

## Steps
1. `enable_features("contrib")`; data as a polars LazyFrame.
2. `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(rho=RHO), split_evenly_over=K, margins=[polars.Margin(max_length=N_MAX)])`
3. `q <- context.query().select(dp.len(signed=True), col(X).cast(float).dp.sum((L, U)))`
4. `q.summarize(alpha=0.05)` shows "Integer Gaussian" / "Float Gaussian" and an accuracy column.
5. `row <- q.release().collect()`.
6. If the task reports (epsilon, delta): `(eps, delta) <- make_fix_delta(make_zCDP_to_approxDP(context.accountant), delta=DELTA).map(context.d_in)`.

## Privacy accounting
`context.accountant.map(context.d_in)` is the total rho. Under `changes=1` the length is public (released without noise) and a sum is charged U - L; under `contributions=1` (add/remove) the length gets noise and a sum is charged max(|L|, |U|), which under-charges a replace-one task when L < 0 < U. Converting it once with the combinators gives the (epsilon, delta) guarantee of everything released, with no hand formula. Choose RHO so the converted epsilon meets the target, and check before releasing.

## Pitfalls
- Gaussian cannot give pure DP; if the task says epsilon-DP with delta = 0, use `loss_of(epsilon=...)`.
- Gaussian pays off most when one person touches many cells (L2 vs L1); see `opendp-polars-bound-per-group-unit`.
- `loss_of(rho=..., delta=...)` adds a delta for key release (group_by without public keys), not for the conversion.
- `dp.len()` without `signed=True` warns in 0.16 and clamps negatives to zero.
- Expressions that set the distribution explicitly (e.g. `.dp.laplace()`) do not switch.
