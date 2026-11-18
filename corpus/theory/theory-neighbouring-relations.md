---
id: theory-neighbouring-relations
title: Neighbouring datasets, bounded versus unbounded, and public n
topic: neighbours
task_types: [count, sum, mean, variance, histogram]
sources:
  - "Dwork & Roth 2014, Definitions 2.3 and 2.4 (add/remove distance)"
  - "Dwork & Roth 2014, Theorem 2.2 (group privacy)"
  - "Chaudhuri, Monteleoni, Sarwate 2011, Section 2.3, Definition 3 (datasets differing in a single entry)"
---
## Idea
DP compares neighbouring datasets, and there are two common definitions.
- Unbounded (add/remove): one dataset has one extra record. The size n differs, so n itself is private.
- Bounded (replace-one): same size, one record changed. n is public.

A replacement is one removal plus one addition. So a mechanism that is ε-DP for add/remove is 2ε-DP for replace-one (group privacy with a group of 2). Some tools measure distance as the number of records added plus removed, and then count a replacement as distance 2.

Sensitivities under each relation (values clamped to public [L, U]):
- Count: 1 under both.
- Histogram over disjoint cells: L1 = 1 under add/remove, 2 under replace-one, because one cell loses 1 and another gains 1.
- Sum: max(|L|, |U|) under add/remove, U − L under replace-one.
- Mean with public n (replace-one only): (U − L)/n.

## Why it matters when writing DP code
Decide the relation first, from the task, and compute every sensitivity under it. State it in the output. Treat n as public only under replace-one.

## Common mistakes
- Using Δ = 1 for a histogram under replace-one.
- Dividing by the exact n under add/remove, which releases n without noise.
- Mixing relations, for example computing sensitivity under one and claiming the guarantee under the other.
- Doubling the count sensitivity when n is public and the relation is replace-one. That is safe but wastes accuracy.
