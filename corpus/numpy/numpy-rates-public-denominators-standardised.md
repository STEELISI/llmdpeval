---
id: numpy-rates-public-denominators-standardised
title: Event rates against PUBLIC census population counts or person-time (stratum, crude, age-standardised, rate ratio) in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.bincount
  - numpy.sqrt
  - numpy.exp
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Only for denominators that are published (census population, person-time); if the group sizes come from the private data, use `numpy-groupwise-rate-noisy-ratio` instead. Release event rates (deaths, diagnoses, arrests per population or person-time) per group and stratum (e.g. age band), the crude rate and the directly standardised rate per group, and a standardised rate ratio with a confidence interval that counts the DP noise. The event records are private; the denominators are public. Default neighbours: replace-one on the event rows.

## Steps
1. Public inputs: group list, stratum list, denominator table `P[g][s]`, standard weights `w[s]` (a published standard population, summing to 1).
2. One cell per event row: `cell <- g*S + s`; `counts <- bincount(cell, minlength=G*S)`.
3. `d <- counts + rng.laplace(0, Δ₁/ε, size)` with Δ₁ = 2 (replace-one) or 1 (add/remove). Keep negative values for unbiased sums; clip only for display.
4. Post-processing: `rate[g,s] = d/P`; `crude[g] = sum_s d / sum_s P`; `asr[g] = sum_s w[s]*rate[g,s]`.
5. Noise-aware variance per cell: `var_d = max(d, 0)` (Poisson sampling) `+ 2*scale²` (Laplace). `var_asr[g] = sum_s w[s]² var_d / P²`.
6. Rate ratio `rr = asr[A]/asr[B]`; `se_log = sqrt(var_asr[A]/asr[A]² + var_asr[B]/asr[B]²)`; CI `rr*exp(±1.96*se_log)`. Undefined if either asr ≤ 0.
7. Private denominators (group sizes from the same data): use two noisy histograms and a noisy ratio instead (`numpy-groupwise-rate-noisy-ratio`).

## Privacy accounting
One event row sits in one cell; replacing it moves it to another cell, Δ₁ = 2 (1 under add/remove). One histogram costs ε; rates, standardisation, ratios and CIs use only released counts and public tables. Compute epsilon_spent = Δ₁/scale and assert it.

## Pitfalls
- A person with several events (repeat arrests, readmissions): cap events per person at a public K first and multiply Δ₁ by K, or count first events only.
- Noise on each rate with sensitivity 1/P per cell plus ε split across cells: the cells are one histogram.
- CI from Poisson variance only: too narrow when counts are small (see `theory-noise-aware-inference`).
- Denominators computed from the private file but treated as public.
- Standard weights chosen from the private data.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
