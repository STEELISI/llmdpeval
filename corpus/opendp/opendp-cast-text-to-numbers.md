---
id: opendp-cast-text-to-numbers
title: Casting text columns to numbers inside the pipeline (make_cast, cast_default, cast_inherent)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: mean
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.make_cast
  - opendp.transformations.then_cast
  - opendp.transformations.then_cast_default
  - opendp.transformations.then_cast_inherent
  - opendp.transformations.then_impute_constant
  - opendp.transformations.then_clamp
  - opendp.transformations.then_resize
  - opendp.transformations.then_mean
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/index.rst
source_license: MIT
---

## Goal
Release a DP statistic of numbers stored as text (raw file fields, survey codes). Parsing happens inside the OpenDP pipeline so a bad value cannot raise a data-dependent error. Neighbouring relation: replace-one (n public), so the domain is sized and the privacy unit is symmetric distance 2.

## Steps
1. `enable_features("contrib")`.
2. `space <- vector_domain(atom_domain(T=str), size=N), symmetric_distance()` (N = the public row count).
3. Pick a caster:
   - `then_cast(float)`: failed parse becomes `None` (output is an `OptionDomain`); must be followed by `then_impute_constant(C)` or `then_drop_null()`.
   - `then_cast_default(float)`: failed parse becomes the type's default (0.0).
   - `then_cast_inherent(float)`: failed parse becomes NaN; follow with an imputer.
4. `pre <- space >> then_cast(float) >> then_impute_constant(C) >> then_clamp((L, U)) >> then_mean()` (casting, imputing and clamping keep the size).
5. `meas <- binary_search_chain(lambda s: pre >> then_laplace(s), d_in=2, d_out=EPS)`
6. `dp_mean <- meas(LIST_OF_STRINGS)`.

Signature for reference: `make_cast(input_domain, input_metric, TOA)`.

## Privacy accounting
Replace-one (n public): casting and imputing are row-by-row and keep the distance; the sized, bounded mean sets the sensitivity ((U - L) / N for one changed row). Read epsilon from `meas.map(2)`.

## Pitfalls
- Casting in pandas first (`astype(float)`) raises on bad text, and that exception depends on private data. Cast in OpenDP.
- `then_cast_default` silently turns errors into 0.0; choose it only when 0 is a sensible value.
- C must lie inside (L, U); N is the public row count. Alternative, add/remove-one (n private): unsized domain, `then_resize(size=DP_COUNT, constant=C)` before `then_mean()`, `d_in=1`, mean charged (U - L) / DP_COUNT plus the count's own epsilon (see opendp-mean-unknown-size-resize).
- `then_drop_null()` after `then_cast` makes the length data-dependent, so the domain is no longer sized and the sum/mean is charged more; prefer an imputer.
- Dataframe helpers like `make_split_dataframe` / `make_select_column` are deprecated since 0.12 ("Use Polars instead").
