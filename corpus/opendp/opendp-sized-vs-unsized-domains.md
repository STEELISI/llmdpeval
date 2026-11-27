---
id: opendp-sized-vs-unsized-domains
title: Sized vs unsized domains, neighbouring relations and why sensitivity changes
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-sum.ipynb
source_license: MIT
---

## Goal
Pick the domain and `d_in` that match your neighbouring relation, so OpenDP computes the right sensitivity. Two relations: unbounded DP (add or remove one record, n private) and bounded DP (replace one record, n public).

## Steps
1. Vector pipelines use `symmetric_distance()`, whose unit is one record added or removed.
   - Unbounded DP, one record per person: `D_IN = 1`.
   - Bounded DP: one replacement = one removal + one addition, so `D_IN = 2`.
2. Unsized domain (n private):
   `space <- vector_domain(atom_domain(T=?)), symmetric_distance()`
   `count <- space >> then_count()`; read `count.map(1)`.
3. Sized domain (n public): put the public size in the domain.
   `space <- vector_domain(atom_domain(bounds=(L, U)), size=N), symmetric_distance()`
   `total <- space >> then_sum()`; read `total.map(2)`.
   The sized bounded sum is charged by the width U - L per replacement, so a 0/1 indicator column has sensitivity 1 at `D_IN = 2`.
4. Counting rows that meet a condition under bounded DP: make a 0/1 column, use the sized domain with bounds (0, 1) and `then_sum`. `then_count` on a sized domain still maps `D_IN = 2` to 2, so it costs twice the noise.
5. Add noise with `then_laplace(scale=?)` and search the scale (see `opendp-binary-search-scale`).

## Privacy accounting
Never hard-code sensitivity. Read it with `trans.map(D_IN)`, then check the whole measurement: `meas.map(D_IN) <= EPS`, with the `D_IN` of your neighbouring relation.

## Pitfalls
- `D_IN = 1` with a sized domain: the sized sum maps an odd distance down (1 -> 0), so noise looks free and the privacy claim is wrong. Bounded DP means `D_IN = 2`.
- Unsized float sums are charged max(|L|, U, U - L), not U - L: bounds of mixed sign cost up to twice as much.
- N must be public knowledge, not `len(private_data)` used silently. If n is not public, release a DP count first (see `opendp-mean-unknown-size-resize`).
- `then_mean` divides by the domain's `size`, not the data length. Data of another length gives a wrong answer with no error.
