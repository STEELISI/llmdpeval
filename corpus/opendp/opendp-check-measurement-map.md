---
id: opendp-check-measurement-map
title: Checking privacy with measurement.map(d_in) and reporting accuracy
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.mod.Measurement.map
  - opendp.mod.Measurement.check
  - opendp.mod.Transformation.map
  - opendp.accuracy.laplacian_scale_to_accuracy
  - opendp.accuracy.discrete_laplacian_scale_to_accuracy
  - opendp.accuracy.accuracy_to_laplacian_scale
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/framework-api/core-structures.rst
source_license: MIT
---

## Goal
Verify that a measurement spends what you claim, and report how accurate the release is, using the framework's own maps rather than formulas. Neighbouring relation: replace-one (n public), so `D_IN = 2` (one changed row = symmetric distance 2).

## Steps
1. For every transformation, inspect the sensitivity it hands on: `trans.map(D_IN)`. This shows whether the domain (sized/unsized, bounds) is what you think. Example: a count of matching rows as a sized 0/1 sum, `vector_domain(atom_domain(T=int), size=N) >> then_clamp((0, 1)) >> then_sum()`, has `map(2) == 1`.
2. For every measurement, compute the privacy loss: `spent <- meas.map(D_IN)`; it returns epsilon (pure), rho (zCDP) or an (epsilon, delta) tuple (approx).
3. Gate the release: `assert spent <= EPS` (for tuples, compare both parts); print `spent`.
4. Or use the predicate form: `meas.check(D_IN, EPS)` returns True when the relation holds.
5. Accuracy for reporting: with the noise scale, `laplacian_scale_to_accuracy(scale=?, alpha=?)` (float noise) or `discrete_laplacian_scale_to_accuracy(scale=?, alpha=?)` (integer noise) gives a half-width that holds with probability 1 - alpha. `accuracy_to_laplacian_scale(accuracy=?, alpha=?)` goes the other way.
6. Polars Context users get the same numbers from `query.summarize(alpha=?)`.

## Privacy accounting
This is the accounting step. Under replace-one the value from `meas.map(2)` is the guarantee; a number typed into a print statement is not. For several releases, compose them (`make_composition`) and check the map of the composition, or use a Context that enforces the total.

## Pitfalls
- An assert that compares a value to itself, or checks a hand-computed epsilon, cannot fail and proves nothing. Check `meas.map(...)` of the measurement you actually invoke.
- Checking with a different `D_IN` than the privacy unit (e.g. 1 when replace-one needs 2): for histograms and group-bys (vector outputs) the d_in=1 value is half the replace-one epsilon.
- With n public the plain row count is known and needs no release; a DP count should count rows that match a condition.
- Alternative, add/remove-one (n private): unsized domain, `D_IN = 1`; a count (`then_count`) or 0/1 sum has `map(1) == 1`.
- Accuracy bounds cover noise only; clamping bias is extra.
- Using the scale from one query to report accuracy for another.
