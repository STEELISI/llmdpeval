---
id: theory-laplace-mechanism
title: The Laplace mechanism for counting rows, sums and means, and its scale
topic: mechanism
task_types: [count, sum, mean, variance, histogram]
sources:
  - "Dwork & Roth 2014, Definition 3.2 (Laplace distribution, variance 2b²)"
  - "Dwork & Roth 2014, Definition 3.3 and Theorem 3.6 (Laplace mechanism is (ε, 0)-DP)"
  - "Dwork & Roth 2014, Example 3.1"
---
## Idea
The Laplace mechanism releases f(x) + Y, where each coordinate of Y is an independent draw from the Laplace distribution with scale b = Δ₁/ε. Here Δ₁ is the global L1 sensitivity of f under the chosen neighbour relation, and ε is the budget assigned to this release. The result is ε-DP with δ = 0.

The noise has mean 0 and variance 2b². The chance that one draw exceeds t in absolute value is e^(−t/b), so about 95% of draws fall within ±3b. For one count (Δ₁ = 1) at ε = 1, b = 1, so the error is usually a few units whatever the dataset size.

## Why it matters when writing DP code
- Compute b from Δ₁ and the ε share for this release, not from the total ε if the budget is split.
- For a vector output, use the vector's Δ₁ and add independent noise to every coordinate.
- Draw fresh noise for every release.
- Integer outputs are better served by discrete noise.
- Rounding or clipping the noisy result to a public range afterwards is free post-processing.

## Common mistakes
- Setting b = 1/ε when Δ₁ is not 1.
- Setting b = Δ₁ · ε. The scale divides by ε.
- Computing b from a sensitivity measured on the data.
- Using the scale for one query when the release is k queries, or the full ε when ε was split.
- Clipping the noisy result to the true minimum or maximum, which touches raw data.
