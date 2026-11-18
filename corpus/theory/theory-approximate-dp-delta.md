---
id: theory-approximate-dp-delta
title: Delta and approximate differential privacy
topic: approx-dp
task_types: [mean, sum, composition]
sources:
  - "Dwork & Roth 2014, Definition 2.4 and the discussion after it (Section 2.3)"
  - "Dwork & Roth 2014, Theorem 3.16"
---
## Idea
Approximate DP adds a slack δ: a mechanism is (ε, δ)-DP if, for all neighbouring x and y and every output set S, Pr[M(x) in S] ≤ e^ε · Pr[M(y) in S] + δ. Pure DP is the case δ = 0. Informally, δ is a small probability mass where the ε bound is allowed to fail.

δ must be tiny compared with 1/n, where n is the number of records. Dwork and Roth warn that δ on the order of 1/n is "very dangerous": a mechanism that publishes the complete records of a few people can meet such a δ. They ask for δ smaller than the inverse of any polynomial in n.

## Why it matters when writing DP code
- Laplace and the exponential mechanism give pure DP (δ = 0). The Gaussian mechanism needs δ > 0. If the task gives only ε, or says δ = 0, a mechanism that needs δ > 0 does not meet the task.
- δ adds up under composition just as ε does: k releases with (ε_i, δ_i) give (Σε_i, Σδ_i).
- Report both parameters. Describing a result only by ε, when δ > 0 was used, overstates the guarantee.

## Common mistakes
- Using Gaussian noise and never stating δ, or setting δ = 0 for it.
- Choosing δ near 1/n or larger, for example 0.01 on a few thousand records.
- Forgetting to add up the δ values of several releases.
- Reading δ as "the chance someone is caught".
