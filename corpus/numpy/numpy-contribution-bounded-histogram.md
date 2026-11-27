---
id: numpy-contribution-bounded-histogram
title: Histogram over a public item list when one unit lists many items (contribution bounding to K per unit)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.bincount
  - numpy.random.Generator.laplace
  - pandas.DataFrame.drop_duplicates
  - pandas.DataFrame.sample
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/identifier-truncation.rst
source_license: MIT
---

## Goal
Release, for every item in a public list, how many units (people, reports, households) list it, when one unit can list several items in long-format rows. The privacy unit is the whole unit, not one row. Replace-one unit (bounded DP: the number of units is public).

## Steps
1. `ITEMS`: the public item list (codebook), fixed before looking at the data. `K`: a public cap on items per unit, chosen by a rule fixed in advance (a design limit, or a high percentile from public prior data), never the maximum seen in the private data.
2. Per unit: drop repeated items, then keep at most K of its distinct items, chosen at random within the unit (or by a public priority order). The choice may look only at that unit's own rows.
3. `counts <- bincount(codes of kept items in ITEMS)`; items outside the list are dropped.
4. Sensitivity: one unit adds 1 to at most K bins. Replace-one removes one unit and adds another, so `SENS <- 2K` in L1 (K under add/remove-one).
5. `noisy <- counts + laplace(0, SENS/EPS)` per bin.
6. Post-processing: ratios, top-k, sorting on noisy counts only.

## Privacy accounting
One Laplace release of the count vector with L1 sensitivity 2K under replace-one at the unit level: `epsilon_spent <- SENS / scale`, assert <= EPS. A different budget split for two histograms (e.g. two time windows of disjoint units) follows the same rule per histogram; if the windows hold different units, parallel composition applies.

## Pitfalls
- Sensitivity 1 or 2 when a unit lists many items: the real epsilon is K times the printed one.
- K from the private data (its maximum or percentile): the cap itself leaks; use a public rule.
- Keeping "the K most common items overall" for each unit: that choice depends on other units' data.
- Counting rows instead of units: an item listed twice by one unit must count once.
- A large K keeps more data but adds noise in proportion; report how many items the cap dropped only as a noisy number.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
