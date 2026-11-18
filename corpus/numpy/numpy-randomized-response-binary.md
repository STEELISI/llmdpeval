---
id: numpy-randomized-response-binary
title: Local DP survey answers by randomized response for a yes/no attribute and the unbiased share estimate in NumPy
framework: numpy
framework_version: "any"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.random
  - numpy.where
  - numpy.exp
  - numpy.log
source_url: https://doi.org/10.1080/01621459.1965.10480775
source_license: original code (algorithm from cited paper)
---

## Goal
Protect each record's yes/no value on its own (local DP) by randomized response: keep the true bit with probability p = e^ε/(1 + e^ε), flip it otherwise. Then estimate the population share of "yes" without bias. Warner, JASA 1965; ε-DP form in Dwork & Roth 2014, Section 3.2.

## Steps
1. Bit per record: b_i ∈ {0, 1} (from a public positive value).
2. p = e^ε / (1 + e^ε). For each record: `keep <- rng.random(N) < p`; `reported <- numpy.where(keep, b, 1 − b)`.
3. The reported column is the release (it may be shared row by row).
4. Unbiased estimate of the true share π: π̂ = (mean(reported) − (1 − p)) / (2p − 1). Clip to [0, 1] if you want a valid proportion (adds slight bias).
5. Standard error ≈ √(mean·(1 − mean)/N) / (2p − 1); report it.

## Privacy accounting
Each record's report satisfies ε-local DP: the likelihood ratio of any report is at most p/(1 − p) = e^ε. Compute ε_spent = ln(p/(1 − p)) in code. Each record is touched once, so the whole released column is ε-DP (parallel over records) and any estimate from it is post-processing.

## Pitfalls
- Using p = 0.75 "because Warner did" and claiming some other ε: ε = ln(p/(1−p)) = ln 3 ≈ 1.1 for p = 0.75.
- Reporting mean(reported) as the estimate: biased towards 0.5.
- Comparing its accuracy with central Laplace: local DP needs far more records for the same error (error ∝ 1/(ε√N) vs 1/(εN)).
- Applying it several times to the same record and averaging: each report costs ε.
