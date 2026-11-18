---
id: opendp-rates-public-denominators-standardised
title: Event rates against PUBLIC population denominators (census counts, person-time) - stratum, crude and age-standardised rates and a noise-aware rate ratio in OpenDP
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
Use this only when the denominators are published (census population, person-time); if group sizes come from the private data, use `opendp-group-rates-parity-difference` instead. Release event rates (deaths, diagnoses, arrests per population or person-time) per group and stratum (e.g. age band), the crude rate and the directly standardised rate per group, and a standardised rate ratio with a confidence interval that counts the DP noise. The event records are private; the denominators are public. Neighbours: replace-one on the event rows, `d_in = 2`.

## Steps
1. `enable_features("contrib")`. Public inputs: group list, stratum list, denominator table `P[g][s]`, standard weights `w[s]` summing to 1.
2. Integer cell code per event row: `cell <- g*S + s`; `CATS <- list(range(G*S))`.
3. `space <- vector_domain(atom_domain(T=int)), symmetric_distance()`
   `make <- lambda s: space >> then_count_by_categories(categories=CATS, null_category=False) >> then_laplace(s)`
4. `scale <- binary_search_param(make, d_in=2, d_out=EPS)`; `meas <- make(scale)`; `d <- meas(cell_list)` reshaped to (G, S). Keep the scale: the CI needs it.
5. Post-processing: `rate = d/P`; `crude[g] = sum_s d / sum_s P`; `asr[g] = sum_s w[s]*rate[g,s]`.
6. Noise-aware variance per cell: `max(d, 0)` (Poisson sampling) `+ 2a/(1-a)²` with `a = exp(-1/scale)` (discrete Laplace). `var_asr[g] = sum_s w[s]² var / P²`.
7. Rate ratio `rr = asr[A]/asr[B]`; `se_log = sqrt(var_asr[A]/asr[A]² + var_asr[B]/asr[B]²)`; CI `rr*exp(±1.96*se_log)`. Undefined if either asr ≤ 0.
8. Private denominators: two histograms and a noisy ratio instead (`opendp-group-rates-parity-difference`).

## Privacy accounting
Replacing one event row moves it between two cells: the count vector moves by 2 in L1, and the map gives this from `d_in = 2`. Assert `epsilon_spent <- meas.map(2) <= EPS`. Rates, standardisation, ratios and CIs are free.

## Pitfalls
- A person with several events: cap events per person at a public K and evaluate the map at `d_in = 2K`, or count first events only.
- Noise on each finished rate, or one measurement per cell with ε split: the cells are one release.
- CI from Poisson variance only: too narrow when counts are small (see `theory-noise-aware-inference`).
- `map(1)` under replace-one reports half the real ε.
- Denominators or standard weights taken from the private data.
