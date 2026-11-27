---
id: theory-parallel-composition
title: Parallel composition on disjoint parts, and its catch under replace-one
topic: composition
task_types: [histogram, composition, classification]
sources:
  - "Dwork & Roth 2014, Example 3.2 (histogram queries over disjoint cells)"
  - "Dwork & Roth 2014, Theorem 2.2 (group privacy)"
---
## Idea
Suppose the records are split into disjoint parts by a rule fixed in advance from public information, and each part gets its own mechanism that reads only that part. Under add/remove neighbours, one record lives in exactly one part, so only one mechanism sees a change. If every part's mechanism is ε-DP, the whole release is ε-DP, not kε. The histogram is the standard case: disjoint cells give L1 sensitivity 1 (Example 3.2).

Under replace-one neighbours the picture changes. A replaced record can leave one part and join another, so two parts change. If each part's mechanism is ε-DP for add/remove, the release can cost up to 2ε. This matches the histogram's L1 sensitivity of 2 under replace-one.

## Why it matters when writing DP code
- Only claim the parallel saving when the parts really are disjoint and each record feeds exactly one output (forest trees on disjoint row subsets, one model per disjoint chunk).
- The splitting rule must be public: a fixed list of categories or fixed ranges, not groups found in the data.
- Under replace-one, budget for two parts changing. Separate per-part releases with equal shares ε₀ each then cost 2ε₀, the same as charging both shares; unequal shares cost twice the largest, more than their sum suggests.

## Common mistakes
- Claiming parallel composition for several statistics computed on the same records. That is sequential composition.
- Partitions that overlap, or that were chosen by looking at the data.
- Assuming the cost stays at ε under replace-one.
