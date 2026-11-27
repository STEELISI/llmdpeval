---
id: theory-histogram-public-categories
title: Histograms over a public category list, and the sort-order leak
topic: sensitivity
task_types: [histogram]
sources:
  - "Dwork & Roth 2014, Example 3.2 (histogram queries)"
  - "Dwork & Roth 2014, Section 3.4, remark after Definition 3.4 (the range must not depend on the data)"
---
## Idea
A histogram counts how many records fall in each of a set of disjoint cells. Each record lands in exactly one cell. Under add/remove neighbours one record changes one cell by 1, so the L1 sensitivity is 1 whatever the number of cells. Under replace-one it is 2, because one cell loses 1 and another gains 1. Laplace noise with scale Δ₁/ε on every cell gives an ε-DP histogram.

That argument needs the list of cells to be fixed before looking at the data. If the list is built from the values that appear, its contents reveal facts about individuals. A category held by one person shows up only when that person is present.

## Why it matters when writing DP code
- Take the category list from the task or the domain's documentation. Release a noisy count for every listed category, including those whose true count is zero.
- Map values not on the list to a public "other" cell, or drop them by a public rule.
- Order and tie-breaking must be public too. Sorting noisy counts is free post-processing. Breaking ties by the order categories first appear in the data, or by their true frequency, leaks. Break ties by a public order, such as alphabetical order of the public list.

## Common mistakes
- Using the distinct values found in the data as the category list.
- Dropping categories with zero true count.
- Listing categories in the data's frequency order or first-appearance order.
- Using sensitivity 1 under replace-one neighbours.
