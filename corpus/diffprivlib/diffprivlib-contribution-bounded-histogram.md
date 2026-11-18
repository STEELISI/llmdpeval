---
id: diffprivlib-contribution-bounded-histogram
title: Histogram over a public item list when one unit lists many items (contribution bounding to K per unit)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
  - pandas.DataFrame.drop_duplicates
  - pandas.DataFrame.sample
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release, for every item in a public list, how many units (people, reports, households) list it, when one unit can list several items in long-format rows. The privacy unit is the whole unit, not one row. Replace-one unit (bounded DP: the number of units is public).

## Steps
1. `ITEMS`: the public item list (codebook), fixed before looking at the data. `K`: a public cap on items per unit, chosen by a rule fixed in advance (a design limit, or a high percentile from public prior data), never the maximum seen in the private data.
2. Per unit: drop repeated items, then keep at most K of its distinct items, chosen at random within the unit (or by a public priority order). The choice may look only at that unit's own rows.
3. Map kept items to codes 0..k−1 of ITEMS; items outside the list are dropped.
4. Sensitivity: one unit adds 1 to at most K bins. Replace-one removes one unit and adds another, so `SENS <- 2K` in L1 (K under add/remove-one).
5. diffprivlib's `tools.histogram` assumes one row touches one bin (sensitivity 1). So `acc <- BudgetAccountant(epsilon=EPS/SENS, delta=0)` (capped at what it records); `noisy, _ <- tools.histogram(codes, epsilon=EPS/SENS, bins=k, range=(-0.5, k-0.5), accountant=acc)` and the real guarantee is SENS times what the accountant recorded.
6. Post-processing: ratios, top-k, sorting on noisy counts only.

## Privacy accounting
One geometric (discrete Laplace) release of the count vector (`tools.histogram`, truncated at 0) with L1 sensitivity 2K under replace-one at the unit level: `epsilon_spent <- SENS * acc.total()[0]`, assert <= EPS. A different budget split for two histograms (e.g. two time windows of disjoint units) follows the same rule per histogram; if the windows hold different units, parallel composition applies.

## Pitfalls
- Passing EPS straight to `tools.histogram` on capped long data: it charges one bin per unit.
- Sensitivity 1 or 2 when a unit lists many items: the real epsilon is K times the printed one.
- K from the private data (its maximum or percentile): the cap itself leaks; use a public rule.
- Keeping "the K most common items overall" for each unit: that choice depends on other units' data.
- Counting rows instead of units: an item listed twice by one unit must count once.
- A large K keeps more data but adds noise in proportion; report how many items the cap dropped only as a noisy number.
