---
id: opendp-measure-casting
title: Composing pure-DP and zCDP/approx-DP queries by casting output measures
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: composition
dp_notion: approx
needs_bounds: false
api_calls:
  - opendp.combinators.make_pureDP_to_zCDP
  - opendp.combinators.make_approximate
  - opendp.combinators.make_zCDP_to_approxDP
  - opendp.combinators.make_fix_delta
  - opendp.combinators.make_composition
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/combinators/measure-casting.rst
source_license: MIT
---

## Goal
Combine releases that use different privacy definitions (e.g. a Laplace count and a Gaussian sum) into one composition with a single reported guarantee. Neighbouring relation: replace-one (n public): sized domain `vector_domain(atom_domain(T=float, nan=False), size=N)`, every search and map at `D_IN = 2`.

## Steps
1. Build each measurement on the same input domain and metric.
2. Pick one common measure and cast the others to it:
   - pure-DP into zCDP: `make_pureDP_to_zCDP(MEAS_LAPLACE)`; compose with Gaussian measurements; total is rho.
   - pure-DP into approx-DP: `make_approximate(MEAS_LAPLACE)` gives (epsilon, 0) pairs.
   - zCDP into approx-DP: `make_fix_delta(make_zCDP_to_approxDP(MEAS_GAUSS), delta=DELTA)` gives an (epsilon, delta) pair.
3. Compose the cast measurements: `comp <- make_composition([...])`.
4. Check `comp.map(D_IN)` against the task's total (rho, or an (epsilon, delta) tuple), then `answers <- comp(DATA)`.
5. Usually simpler: compose everything in zCDP first, then convert the whole composition once with `make_zCDP_to_approxDP` and `make_fix_delta`. Converting at the end is tighter than converting each piece.

Signature for reference: `make_fix_delta(measurement, delta)`.

## Privacy accounting
Each combinator carries its own privacy map, so `comp.map(2)` is the replace-one guarantee. Alternative, add/remove-one (n private): unsized domain and `d_in=1` throughout (`comp.map(1)`); a clamped sum is then charged max(|L|, |U|, U - L) instead of U - L. Use the scale search on each piece against its share (in the common measure), then check the composed map.

## Pitfalls
- `make_composition` of measurements with different output measures fails with "All output measures must be the same". It does not fall back silently.
- Converting each piece to (epsilon, delta) and then adding deltas and epsilons is valid but loose.
- Pure-DP into zCDP costs rho = epsilon^2 / 2 per piece, decided by the map, not by you.
- In the Context API, `loss_of(epsilon=?, delta=?)` or `loss_of(rho=?)` sets one measure for the whole context; mixing is not needed there.
