---
id: numpy-group-quantile-gaps-ratios
title: Several quantiles per group, quantile gaps between groups and percentile ratios (p90/p10) with the exponential mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.searchsorted
  - numpy.sort
  - numpy.exp
source_url: https://doi.org/10.1109/FOCS.2007.66
source_license: original code (algorithm from cited paper)
---

## Goal
Release several quantiles (e.g. 10th, 50th, 90th percentiles) of a numeric column in each group of a public group list, then derive the gap between two groups at each quantile and each group's percentile ratio (p90/p10) or spread (p90 - p10). Exponential mechanism over a public grid (McSherry & Talwar 2007; Dwork & Roth 2014, Thm. 3.10). Default neighbours: replace-one (bounded DP, n public).

## Steps
1. Public sorted grid `G` between public bounds (topcode, design limits). For ratios, `geomspace(L, U, m)` gives equal relative precision.
2. Budget: `eps_q <- (EPS/2) / len(ALPHAS)` under replace-one (EPS / len(ALPHAS) under add/remove).
3. Per group g and alpha A: `x <- sort(values in g)`; `below <- searchsorted(x, G, "right")`; `u <- -abs(below - A*len(x))` (sensitivity 1); `logits <- eps_q*u/2`; subtract the max, `exp`, normalise; `q[g][A] <- G[rng.choice(m, p=probs)]`.
4. Post-processing: sort each group's quantiles; `gap[A] = q[g1][A] - q[g2][A]`; `ratio[g] = q[g][hi]/q[g][lo]` if `q[g][lo] > 0`.

## Privacy accounting
Within a group, the k quantiles compose: k·eps_q. Groups are disjoint, but under replace-one a changed row can leave one group and join another, so two groups change: total `2·k·eps_q = EPS` (sequential composition, Dwork & Roth Cor. 3.15). Compute epsilon_spent in code from eps_q and assert it.

## Pitfalls
- A grid from the observed values leaks them.
- Each group at the full EPS under replace-one: the real loss is 2·EPS.
- A gap or ratio from exact quantiles of one group and DP quantiles of the other.
- Survey weights: a weighted rank moves by the row's weight, so the utility sensitivity becomes the largest weight. Cap weights at a public W and use logits `eps_q*u/(2W)`, or drop weights and say the estimand changed.
- Small groups: near-random tail quantiles; ratios inherit the noise.
- Exponentiating without subtracting the max: overflow.
