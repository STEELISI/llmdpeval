---
id: opendp-binary-search-scale
title: Getting the noise scale from epsilon with binary_search_param / binary_search_chain
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.mod.binary_search_param
  - opendp.mod.binary_search_chain
  - opendp.mod.binary_search
  - opendp.measurements.then_laplace
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/utilities/parameter-search.rst
source_license: MIT
---

## Goal
Find the free parameter of a pipeline (usually the noise scale) from a target budget, using the pipeline's own privacy map instead of a hand formula. Neighbouring relation: replace-one (n public), so `D_IN = 2`.

## Steps
1. Write the pipeline as a function of one number:
   `make <- lambda scale: PRE >> then_laplace(scale)` (PRE = the transformations, e.g. clamp and sum).
2. Decide `D_IN` from the neighbouring relation: replace-one (the default) is symmetric distance 2 on any domain; 2K if a person contributes up to K rows; alternative, add/remove-one (n private): unsized domain, `D_IN = 1`, a clamped sum charged max(|L|, |U|, U - L). With n public, use a sized domain `vector_domain(atom_domain(T=float, nan=False), size=N)` so a clamped sum is charged U - L and the target `D_OUT` (epsilon; rho for zCDP; an `(epsilon, delta)` tuple for approx-DP).
3. Get the measurement: `meas <- binary_search_chain(make, d_in=D_IN, d_out=D_OUT)`.
   Or just the number: `scale <- binary_search_param(make, d_in=D_IN, d_out=D_OUT)`.
4. Release with `meas(DATA)`.
5. Other uses from the docs: search the largest clipping bound for a fixed scale, the smallest n, or (with `binary_search(predicate)`) the largest `d_in` a fixed measurement tolerates.

Signature for reference: `binary_search_chain(make_chain, d_in, d_out, bounds=(None, None), T=None)`.

## Privacy accounting
The search relies on the map being monotone: it returns the smallest scale for which `make(scale).map(D_IN) <= D_OUT`. Under replace-one, assert `meas.map(2) <= D_OUT` and print the value, so the log shows what was actually spent.

## Pitfalls
- Searching with the wrong `D_IN` gives a valid-looking but wrong guarantee. `D_IN` is the privacy unit, not the sensitivity; OpenDP derives the sensitivity.
- Hand formulas like scale = sensitivity / epsilon skip the float-rounding terms, monotonicity factors, top-k's k factor and composition. Let the map decide.
- If the search raises, the budget may be too small for any scale, or `make` may fail at some values. Pass explicit `bounds=(LO, HI)` for the search range.
- For several releases, search each one against its share of the budget, or let a compositor handle the split.
