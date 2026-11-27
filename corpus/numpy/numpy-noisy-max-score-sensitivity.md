---
id: numpy-noisy-max-score-sensitivity
title: Score sensitivity for report-noisy-max - counts, sums, means and rates in NumPy
framework: numpy
framework_version: "any"
mechanism: report-noisy-max
task_type: selection
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.argmax
  - numpy.clip
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Pick one candidate from a public list by any per-candidate score with ε-DP (Laplace report-noisy-max, Dwork & Roth 2014, Section 3.3, Claim 3.9). The hard part is the score's sensitivity Δ: the largest change of ONE candidate's score when one record is added, removed or replaced. Neighbouring relation: replace-one (n public).

## Steps
1. Write the score as a function `score(df, candidate)` and derive Δ for it before touching the data:
   - count of records in the group: Δ = 1;
   - sum of a column clamped to public [L, U] inside the group: max(|L|, |U|) under add/remove; under replace-one max(U − L, |L|, |U|), because the record may change its value or leave/join the group;
   - mean or rate (fraction of the group with some property) inside a group: Δ = U − L for a mean and **Δ = 1 for a rate**, because a group can have a single member. It is NOT 1/N: N is the total, not the group size, and group sizes are private.
2. Monotone or not: counts under add/remove only go up when a record is added, so scale = Δ/ε. Sums of signed values, means, rates and anything under replace-one are not monotone: scale = 2·Δ/ε (Ding et al. 2019, arXiv 1904.12773, Theorem 2).
3. `noisy <- scores + rng.laplace(loc=0, scale=?, size=K)`; release only `candidates[numpy.argmax(noisy)]`.

## Privacy accounting
One selection costs ε for any score with per-candidate sensitivity Δ when the scale is 2Δ/ε; the factor 2 can be dropped only for monotone scores. epsilon_spent = factor·Δ / scale, computed in code and asserted ≤ ε.

## Pitfalls
- Rate sensitivity 1/N: wrong; a group of one record moves its rate from 0 to 1.
- An empty group scored as 0 when the score range is [L, U] with L > 0: use L, so the change stays within U − L.
- Because Δ = 1 for a rate is large compared with typical gaps between rates, selecting by rate is very noisy. Another route is to release noisy numerators and denominators and post-process (`numpy-groupwise-rate-noisy-ratio`), which has its own pitfalls.
- Monotone scale for a rate: adding a record can lower a rate.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
