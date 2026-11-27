---
id: numpy-survival-curve-exit-histogram
title: Survival curves and life tables per group from one noisy exit histogram in NumPy
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.searchsorted
  - numpy.bincount
  - numpy.cumsum
  - numpy.cumprod
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release a survival curve (Kaplan-Meier style, one point per public time bin), at-risk counts, event counts and discrete-time hazards for each group of a public group list. Also covers a life table and year-by-year hazards ("share with the event in year 1, 2, 3 among those still at risk"). One Laplace histogram (Dwork & Roth 2014, Thm. 3.6) plus post-processing (Prop. 2.1). Default neighbours: replace-one (bounded DP, n public).

## Steps
1. Public inputs: time bin edges E from 0 to the public maximum follow-up (study design, not the data); the group list; the event coding (1 = event, 0 = censored).
2. Administrative censoring: a row with time ≥ E_last is censored at the last bin.
3. Each person exits once. Cell = (group index, bin of exit time, event flag): `bin <- clip(searchsorted(E, t, "right") - 1, 0, B-1)`; `cell <- (g*B + bin)*2 + event`.
4. `counts <- bincount(cell, minlength=K*B*2)`; `noisy <- max(counts + rng.laplace(0, Δ₁/ε, size), 0)`.
5. Post-processing per group: `events[b]`, `exits[b] = events[b] + censored[b]`; `at_risk[0] = sum(exits)`, `at_risk[b] = at_risk[0] - cumsum(exits)[b-1]`; `hazard = clip(events / max(at_risk, FLOOR), 0, 1)`; `S = cumprod(1 - hazard)`.
6. Actuarial life table variant: use `at_risk - censored/2` as the denominator. Rate per person-time: release person-time per bin as a separate bounded sum (each person adds at most the bin width per bin).
7. Hazard ratios or a Cox model: no DP routine exists here. A piecewise-exponential (Poisson) model fitted to the released events and person-time tables is post-processing.

## Privacy accounting
Replace-one (bounded DP, n public): the changed person leaves one cell and enters another, Δ₁ = 2 (1 under add/remove). One release costs ε; at-risk counts, hazards, survival, medians of the curve and group differences are free. Compute epsilon_spent = Δ₁/scale and assert it.

## Pitfalls
- Noising the at-risk counts: a person is at risk in every bin up to exit, so that vector has L1 sensitivity up to B-1 under replace-one (B under add/remove), not 1. Release exits and rebuild at-risk by cumulative sums.
- One noisy query per bin or per group with ε split across them: the cells are one histogram.
- Bin edges or the maximum follow-up taken from the data.
- Group sizes or the at-risk start taken from the exact data; use the sum of the noisy cells.
- Survey or sampling weights: a weighted count has sensitivity equal to the largest weight. Drop the weights or cap them at a public value, and say the estimand changed.
- Late bins with few people at risk: the hazard is mostly noise. Report the noisy at-risk counts next to the curve.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
