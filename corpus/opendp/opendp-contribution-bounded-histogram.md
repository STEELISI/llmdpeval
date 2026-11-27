---
id: opendp-contribution-bounded-histogram
title: Histogram over a public item list when one unit lists many items (contribution bounding to K per unit)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
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
3. `hist <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=ITEMS, null_category=False)`; input = the kept items as one list of str.
4. Distance: after capping a unit is at most K rows. Replace-one removes one unit and adds another, so the input distance is `D_IN <- 2K` (K under add/remove-one). OpenDP turns it into the histogram's sensitivity.
5. `meas <- binary_search_chain(lambda s: hist >> then_laplace(s), d_in=D_IN, d_out=EPS)`; `noisy <- meas(KEPT_ITEMS)`. The polars Context alternative: `unit_of(changes=1, identifier=col(UNIT))` with `truncate_per_group(K)`.
6. Post-processing: ratios, top-k, sorting on noisy counts only.

## Privacy accounting
One Laplace release of the count vector, priced by OpenDP at input distance 2K (replace-one at the unit level): `epsilon_spent <- meas.map(2K)`, assert <= EPS. A different budget split for two histograms (e.g. two time windows of disjoint units) follows the same rule per histogram; if the windows hold different units, parallel composition applies.

## Pitfalls
- Sensitivity 1 or 2 when a unit lists many items: the real epsilon is K times the printed one.
- K from the private data (its maximum or percentile): the cap itself leaks; use a public rule.
- Keeping "the K most common items overall" for each unit: that choice depends on other units' data.
- Counting rows instead of units: an item listed twice by one unit must count once.
- A large K keeps more data but adds noise in proportion; report how many items the cap dropped only as a noisy number.
