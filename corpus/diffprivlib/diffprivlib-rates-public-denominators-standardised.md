---
id: diffprivlib-rates-public-denominators-standardised
title: Incidence rates per population against PUBLIC census denominators (crude, age-standardised, rate ratio) in diffprivlib
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Only for published denominators; if the denominators are not published, use group-rates-parity-difference instead. Release event rates (deaths, diagnoses, arrests per population or person-time) per group and stratum (e.g. age band), the crude rate and the directly standardised rate per group, and a standardised rate ratio with a confidence interval that counts the DP noise. The event records are private; the denominators (census population, person-time) are public. Neighbouring relation: replace-one (bounded DP, n public) on the event rows.

## Steps
1. Public inputs: group list, stratum list, denominator table `P[g][s]`, standard weights `w[s]` (a published standard population, summing to 1).
2. Integer cell code per event row: `cell <- g*S + s`; rows outside the lists are dropped.
3. `acc <- BudgetAccountant()`; `counts, _ <- tools.histogram(cell, epsilon=EPS/2, bins=G*S, range=(-0.5, G*S - 0.5), accountant=acc)`.
4. Post-processing: `rate = counts/P`; `crude[g] = sum_s counts / sum_s P`; `asr[g] = sum_s w[s]*rate[g,s]`.
5. Noise-aware variance per cell: `counts` (Poisson sampling) `+ 2a/(1-a)²` with `a = exp(-EPS/2)` (two-sided geometric noise at the ε passed; the library truncates at 0, so this is slightly conservative near zero). `var_asr[g] = sum_s w[s]² var / P²`.
6. Rate ratio `rr = asr[A]/asr[B]`; `se_log = sqrt(var_asr[A]/asr[A]² + var_asr[B]/asr[B]²)`; CI `rr*exp(±1.96*se_log)`. Undefined if either asr is 0.
7. Private denominators: two histograms and a noisy ratio instead (`diffprivlib-group-rates-parity-difference`).

## Privacy accounting
diffprivlib calibrates the histogram for add/remove. Under replace-one an event row moves between two cells, so pass EPS/2 and report `epsilon_spent <- 2 * acc.total()[0]`; assert it is ≤ EPS. Rates, standardisation, ratios and CIs are post-processing.

## Pitfalls
- A person with several events: cap events per person at a public K and divide the ε passed by K, or count first events only.
- `tools.mean` or `tools.count_nonzero` per cell with a share of ε: the cells are one histogram.
- Truncation at 0 biases small cells upward; sums over many near-empty cells drift up.
- CI from Poisson variance only: too narrow when counts are small (see `theory-noise-aware-inference`).
- `range=None`, denominators or standard weights taken from the private data.
- Add/remove alternative (n private): the library's histograms and per-partition models are calibrated for add/remove, so under add/remove pass the full epsilon and report `acc.total()[0]`; under replace-one pass epsilon/2 and report 2x.
