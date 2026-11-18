---
id: opendp-api-change-compositor-renames
title: OpenDP 0.14-0.16 renames - composition, noisy max, quantile and threshold constructors
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.combinators.make_composition
  - opendp.combinators.make_adaptive_composition
  - opendp.combinators.make_fully_adaptive_composition
  - opendp.measurements.make_noisy_max
  - opendp.measurements.make_noisy_top_k
  - opendp.measurements.make_private_quantile
  - opendp.measurements.make_laplace_threshold
  - opendp.measurements.make_noise
source_url: https://github.com/opendp/opendp/blob/v0.16.0/CHANGELOG.md
source_license: MIT
---

## Goal
Translate constructor names from OpenDP 0.10-0.13 code and tutorials to their 0.16.0 names (CHANGELOG "Migration" sections).

## Steps
1. Composition (renamed in 0.14):
   - `make_basic_composition([...])` -> `dp.c.make_composition([...])`.
   - `make_sequential_composition(...)` -> `dp.c.make_adaptive_composition(input_domain, input_metric, output_measure, d_in, d_mids)`.
   - New since then: `make_fully_adaptive_composition` (odometer) and `make_privacy_filter`.
2. Selection (0.14): `make_report_noisy_max_gumbel(...)` -> `dp.m.make_noisy_max(...)`; top-k is `dp.m.make_noisy_top_k(...)`.
3. `make_private_quantile` is parametrized with `output_measure` (0.14).
4. Thresholds (0.14): `make_laplace_threshold` and `make_gaussian_threshold` take `L0PInfDistance` input metrics (`dp.l01inf_distance(...)`, `dp.l02inf_distance(...)`).
5. 0.14 added `make_noise(input_domain, input_metric, output_measure, scale)`, which picks Laplace or Gaussian from the measure.
6. Older (0.8): `make_base_laplace` / `make_base_discrete_laplace` -> `make_laplace`; `make_base_gaussian` -> `make_gaussian`; `make_base_ptr` -> threshold mechanisms.

## Privacy accounting
Renames keep the same privacy maps; still re-derive scales with `binary_search_chain` and check `meas.map(d_in)`.

## Pitfalls
- `make_basic_composition`, `make_sequential_composition` and `make_report_noisy_max_gumbel` still import in 0.16.0 as deprecated aliases; prefer the new names so code survives the next release.
- 0.11 changed tuples to product ordering, which broke `binary_search_param` for `make_laplace_threshold`; search epsilon and delta parts separately.
- Code that calls `meas.check(d_in, d_out)` still works, but `map` gives the actual number to report.
