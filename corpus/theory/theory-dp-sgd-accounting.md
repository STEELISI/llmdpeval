---
id: theory-dp-sgd-accounting
title: DP-SGD - per-example clipping, Gaussian noise and Renyi DP accounting
topic: approx-dp
task_types: [classification, regression, composition]
sources:
  - "Abadi et al. 2016, CCS, Algorithm 1 and Section 3.1 (moments accountant)"
  - "Mironov 2017, CSF, Propositions 1 and 3 (Renyi DP composition and conversion)"
  - "Mironov, Talwar and Zhang 2019, arXiv 1908.10530, Section 3 (sampled Gaussian mechanism)"
---
## Idea
DP-SGD trains a model by gradient steps on random batches. Each example's gradient is clipped to a fixed L2 norm C, the clipped gradients are summed, and Gaussian noise with standard deviation sigma times C is added. Clipping bounds the sensitivity of the sum; the noise makes each step a Gaussian mechanism; random batches amplify privacy, because a given row is used in a step only with probability q.

The steps compose. Renyi DP gives tight totals: the Renyi divergences of the steps add up, and the total converts to an (epsilon, delta) guarantee at the end. The Gaussian mechanism needs delta > 0, so DP-SGD never gives pure DP.

## Why it matters when writing DP code
- Fix C, sigma, batch rate, steps and delta before training, from public facts, and compute epsilon with an accountant.
- Standard accountants assume add/remove-one and Poisson batches. Under replace-one, use the two-step chain (remove, then add): epsilon doubles and delta becomes (1 + e^epsilon) times delta, with epsilon and delta the add/remove values.
- Every training run on the private data costs budget, including tuning runs.

## Common mistakes
- Clipping the batch gradient instead of each example's gradient.
- Adding noise without an accountant, or reporting epsilon for one step.
- Using delta = 0 or a delta near 1/n.
- Releasing exact accuracy on private test rows.
