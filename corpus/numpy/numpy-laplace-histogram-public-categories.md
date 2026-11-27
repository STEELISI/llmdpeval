---
id: numpy-laplace-histogram-public-categories
title: Histogram over a public category list with the Laplace mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.sum
  - numpy.clip
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release a count for every category in a public list with epsilon-DP by adding independent Laplace noise to each bin. Dwork & Roth Example 3.2 (histogram queries, sensitivity 1 under their add/remove relation) with the Laplace mechanism (Def. 3.3, Thm. 3.6): each record falls in exactly one bin. Default neighbours: replace-one (bounded DP, n public), so Δ = 2.

## Steps
1. Fix the category list CATS from public knowledge (a codebook, a schema) in a fixed order, e.g. alphabetical. Never derive it from the private column (unique values, value counts): a category held by one person would reveal that person.
2. Count every listed category, zeros included: `counts[k] <- numpy.sum(x == CATS[k])` for each k. Values outside CATS go to an explicit "other" bin or are dropped (both per-record).
3. L1 sensitivity Δ of the whole count vector, by relation:
   - add/remove one record: one bin moves by 1, so Δ = 1;
   - replace one record: one bin −1 and another +1, so Δ = 2.
4. Same scale for every bin: `noisy <- counts + rng.laplace(loc=0, scale=?, size=K)` with scale = Δ / ε, K = number of categories.
5. Optional post-processing (free, Prop. 2.1): round, clip at 0, sort, arg-max.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ = 2, scale 2/ε; Δ = 1 only if the task states add/remove-one. The whole vector costs ε once, not ε per bin: Def. 3.3 adds i.i.d. Lap(Δ/ε) to each coordinate of a vector query with L1 sensitivity Δ. This holds because each record touches one bin. If a record can touch several bins (multi-label data), Δ is the maximum number of bins one record can change (times 2 under replace-one).

## Pitfalls
- Δ = 1 under replace-one: it must be 2.
- Charging ε per bin, or splitting ε across bins: both over-noise disjoint bins.
- Releasing only the non-empty categories: the key set itself leaks.
- Tie order taken from data order instead of a fixed public order.
- If only the top category is needed, report-noisy-max uses the same noise but releases less (see `numpy-report-noisy-max`).
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
