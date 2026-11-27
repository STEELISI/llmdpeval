---
id: theory-contribution-bounding
title: Bound how much one person can contribute before releasing
topic: sensitivity
task_types: [histogram, count, sum]
sources:
  - "Dwork & Roth 2014, Definition 3.1 (global sensitivity) and Theorem 3.6 (Laplace mechanism)"
  - "OpenDP 0.16 documentation, identifier truncation (privacy unit with an identifier)"
  - "Wilson et al. 2020, PoPETs, DP SQL with bounded user contribution"
---
## Idea
The privacy unit is what DP protects: a person, a household, a report. When one unit can own many rows, for example one report listing several items, the sensitivity of a count per item is not 1. It is the number of rows a unit can touch. Without a cap that number is unbounded, so no finite noise gives a guarantee.

Contribution bounding fixes this. Choose a public cap K before looking at the data, from a design limit or from public prior data. Keep at most K rows per unit, choosing them by a rule that looks only at that unit's own rows, and drop repeats so one unit counts an item once. A histogram then changes by at most K under add or remove one unit, and by at most 2K under replace-one unit.

## Why it matters when writing DP code
- Group the rows by the unit identifier first, then cap, then count.
- Set the sensitivity from K and the neighbouring relation, never from the observed maximum.
- A larger K keeps more data but adds proportionally more noise.

## Common mistakes
- Sensitivity 1 for a histogram over long data where one unit has many rows.
- Taking K from the private data, such as its maximum or a percentile.
- Choosing which K rows to keep from data of other units, such as the most common items overall.
