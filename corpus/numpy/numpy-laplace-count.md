---
id: numpy-laplace-count
title: Counting query with the Laplace mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.sum
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release the number of records that satisfy a predicate with epsilon-DP by adding Laplace noise to the exact count. Dwork & Roth Def. 3.3 (Laplace mechanism) and Thm. 3.6 (it is (epsilon, 0)-DP). Neighbouring relation: replace-one (n public).

## Steps
1. Exact count from a per-record boolean mask: `true_count <- numpy.sum(mask)`.
2. Fix the sensitivity Δ from the neighbouring relation, never from the data:
   - replace one record, N public (the default): a predicate count moves by at most 1 (one row can leave or join the predicate), so Δ = 1;
   - add/remove one record (alternative): also Δ = 1, so the calibration is the same.
   (Releasing the total row count N under replace-one is pointless: N is public there.)
3. One generator per run: `rng <- numpy.random.default_rng()` (no seed in a real release; see `numpy-rng-seeding`).
4. Noise: `dp_count <- true_count + rng.laplace(loc=0, scale=?)` with scale = Δ / ε. The scale is the Laplace parameter b, not the standard deviation (that is √2·b).
5. Optional post-processing (free, Prop. 2.1): round, clip at 0. It may use only the noisy value and public constants.

## Privacy accounting
This release costs exactly ε under the stated relation, provided Δ is a true worst-case bound. If the count is one of several queries on the same data, its ε adds to the others (sequential composition, Dwork & Roth Cor. 3.15; see `numpy-composition-budget-split`). Print ε, Δ and the scale next to the result so the accounting can be audited.

## Pitfalls
- scale = ε, or scale = 1 / (Δ·ε): the scale must grow as ε shrinks.
- Releasing the true count next to the DP count outside a clearly labelled test harness.
- Drawing twice and averaging: that spends 2ε.
- Legacy global `numpy.random.laplace` plus a hard-coded `numpy.random.seed`: the noise becomes predictable.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
