---
id: numpy-groupwise-rate-noisy-ratio
title: Per-group rates as noisy numerator over noisy denominator in NumPy (and their pitfalls)
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.maximum
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release, for every group in a public group list, the rate of records with some binary property (positives / group size) with ε-DP. Two Laplace histograms (Dwork & Roth 2014, Example 3.2, Thm. 3.6), sequential composition between them (Cor. 3.15) and post-processing for the division (Prop. 2.1). Default neighbours: replace-one (bounded DP, n public); group sizes stay private.

## Steps
1. Public, fixed group list. Per group: numerator `num[g]` = records in g with the property, denominator `den[g]` = records in g. Zeros included.
2. Sensitivity. Each vector is a histogram (one record sits in one group): L1 sensitivity 1 under add/remove, 2 under replace-one (the record may move to another group or flip its property).
3. Split ε = ε_num + ε_den. `noisy_num <- num + Lap(Δ₁/ε_num)` per group; `noisy_den <- den + Lap(Δ₁/ε_den)` per group.
4. Post-process: `rate <- clip(max(noisy_num, 0) / maximum(noisy_den, m), 0, 1)` with a PUBLIC floor m ≥ 1.
5. Optionally also report the noisy denominator so readers can see which rates are unreliable.
6. Fairness gaps from the released rates are post-processing: statistical parity difference `rate[A] - rate[B]`, disparate-impact ratio `rate[A] / rate[B]`. Do not noise the gap again.

## Privacy accounting
Replace-one (bounded DP, n public) by default: Δ₁ = 2 per histogram (1 only under add/remove). Two histogram releases: ε_num + ε_den = ε. Noising the rates directly would need per-group sensitivity 1 (a group may have one record) and scale 2/ε or more per rate, which drowns most rates; the ratio route is usually far more accurate for groups with many records. Compute epsilon_spent from both scales in code and assert it.

## Pitfalls
- Rate sensitivity 1/N or 1/n_g: n_g is private and can be 1.
- Dividing a noisy numerator by the exact group size: the size is released without noise.
- Small groups: the noisy ratio has large error and is biased; ranking groups by noisy ratios tends to favour small groups whose noise happened to be large. Use the noisy denominator as a reliability filter (with a public threshold) before comparing.
- A floor m taken from the data.
- Splitting ε per group: groups are disjoint, so one histogram costs ε once.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
