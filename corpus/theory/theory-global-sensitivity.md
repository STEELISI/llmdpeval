---
id: theory-global-sensitivity
title: Global L1 and L2 sensitivity
topic: sensitivity
task_types: [count, sum, mean, variance, histogram, pca]
sources:
  - "Dwork & Roth 2014, Definition 3.1 (L1 sensitivity)"
  - "Dwork & Roth 2014, Definition 3.8 (L2 sensitivity)"
  - "Dwork & Roth 2014, Example 3.1 (counting queries)"
---
## Idea
The global L1 sensitivity of a query f is Δ₁ = max ‖f(x) − f(y)‖₁, taken over every pair of neighbouring datasets x and y. L2 sensitivity Δ₂ uses the Euclidean norm. "Global" means the worst case over all possible datasets, not the change seen on the dataset you hold.

Examples, each with its assumption:
- One count: one record changes it by at most 1, so Δ₁ = 1.
- A sum of values clamped to public bounds [L, U]: Δ₁ = max(|L|, |U|) under add/remove neighbours, and U − L under replace-one neighbours.
- m counts that one person can all affect: Δ₁ can be as large as m (Example 3.1).

## Why it matters when writing DP code
- The sensitivity must come from public facts: the query, the neighbour relation and public bounds, never from the private data.
- Laplace noise is calibrated to Δ₁. Gaussian noise is calibrated to Δ₂.

## Common mistakes
- Using the largest value in the data, or the change seen when removing one actual row, as "the sensitivity". That is local sensitivity, and it leaks.
- Using Δ = 1 for a vector of several counts that one person can all change.
- Using L2 sensitivity with Laplace noise.
- 1/n for a rate over a private group size, or for a fitted estimate: release the counts and sums instead.
- A range bound on an unbounded quantity, such as a regression estimate, not clipped first.
- Ignoring the neighbour relation.
