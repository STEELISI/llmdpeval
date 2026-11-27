---
id: numpy-exponential-quantile
title: Quantile (median, quartiles) with the exponential mechanism over data intervals in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.random.Generator.uniform
  - numpy.clip
  - numpy.sort
  - numpy.concatenate
  - numpy.diff
  - numpy.arange
  - numpy.log
  - numpy.exp
  - numpy.errstate
source_url: https://arxiv.org/abs/2102.08244
source_license: "Gillenwater, Joseph & Kulesza 2021 (ICML) arXiv preprint; algorithm description paraphrased, no code copied"
---

## Goal
Release the q-th quantile (0 < q < 1) of a numeric column with epsilon-DP. Single-quantile exponential mechanism of Smith (STOC 2011) as described in Gillenwater et al. 2021, Section 5.1 ("IndExp"); privacy from the exponential mechanism (Dwork & Roth Thm. 3.10). Neighbouring relation: replace-one (n public).

## Steps
1. Public bounds [L, U]; clamp: `x_c <- numpy.clip(x, L, U)`.
2. End points: `z <- numpy.concatenate([[L], numpy.sort(x_c), [U]])`. This gives N + 1 intervals [z_j, z_{j+1}], j = 0..N; interval j has j sorted data points to its left. (Interval sampling is an exact sampler for the exponential mechanism over every real output in [L, U].)
3. Utility of interval j: u_j = −|j − q·N|, with `j <- numpy.arange(N + 1)`. Sensitivity Δu = 1 (Gillenwater et al. Lemma 7, replace-one; under add/remove it moves by at most max(q, 1 − q) ≤ 1).
4. Log-weights: log(interval length) + ε·u_j / (2·Δu). Lengths from `numpy.diff(z)`; take `numpy.log` inside `numpy.errstate(divide="ignore")` so zero-length intervals get probability 0. Subtract the max, `numpy.exp`, normalise to `p`.
5. `j* <- rng.choice(N + 1, p=p)`, then `dp_q <- rng.uniform(z[j*], z[j*+1])`.

## Privacy accounting
One quantile costs ε. Several quantiles (e.g. three quartiles) run the mechanism once each; with basic composition each gets ε/m (Dwork & Roth Cor. 3.15). Assert the shares sum to ≤ ε. Joint methods (Gillenwater et al.'s JointExp) do better but are more complex.

## Pitfalls
- Returning a data point instead of a uniform point inside a length-weighted interval.
- Bounds from the data's min/max, or forgetting to add L and U as end points.
- Dropping the interval-length factor: the output is then not the exponential mechanism over [L, U].
- Overflow from exponentiating without subtracting the max.
- Sorting several noisy quartiles afterwards is fine (post-processing); re-running until they look ordered is not.
