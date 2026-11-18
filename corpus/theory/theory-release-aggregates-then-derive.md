---
id: theory-release-aggregates-then-derive
title: Release the underlying counts and sums once, then derive every reported number from them
topic: post-processing
task_types: [histogram, sum, mean, classification, inference, composition, other]
sources:
  - "Dwork & Roth 2014, Proposition 2.1 (post-processing) and Theorem 3.16 (sequential composition)"
  - "Dwork & Roth 2014, Example 3.2 (histogram queries over disjoint cells)"
  - "McSherry 2009, SIGMOD, Privacy Integrated Queries (parallel composition over disjoint subsets)"
  - "Hay, Rastogi, Miklau & Suciu 2010, PVLDB 3(1), consistency of noisy hierarchical counts"
---
## Idea
Many analyses print dozens or hundreds of numbers: rates per group, errors of one estimate against another, averages over areas, ranks. All of them are functions of a few tables of counts and sums over disjoint cells. Adding noise to each printed number needs a sensitivity for each, often guessed, and splits the budget into tiny slices. A per cell loop over thousands of cells also runs for a very long time.

The better plan: find the smallest set of tables the outputs are built from, release each table once as one vector (its sensitivity is set by how much one person can add across all its cells), and compute every printed number from the noisy tables. That computation is post-processing and costs nothing. Larger areas are sums of smaller ones; a regression on area-level shares is fitted on noisy shares.

## Why it matters when writing DP code
- Count the tables, not the printed cells, when splitting the budget.
- Keep the noise scale of each table to judge or test the derived numbers.
- Clip derived values to their valid range.

## Common mistakes
- Noise on each final rate, bias or error with a range bound and the budget divided by the number of cells.
- One query per group or per cell inside a loop.
- Exact group sizes used as denominators of derived rates.
- An exact group statistic copied onto every row of the group, then treated as a row value.
