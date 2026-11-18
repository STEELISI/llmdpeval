---
id: theory-report-noisy-max
title: Report noisy max releases only the winner
topic: selection
task_types: [selection]
sources:
  - "Dwork & Roth 2014, Claim 3.9 (Report Noisy Max)"
  - "Dwork & Roth 2014, Remark 3.1 (Gumbel noise matches the exponential mechanism)"
  - "Ding, Wang, Zhang, Kifer, Free Gap Information from the Differentially Private Sparse Vector and Noisy Max Mechanisms, PVLDB 13(3) 2019, arXiv 1904.12773: Section 5.1, Algorithm 1, Definition 7, Theorem 2 and proof"
---
## Idea
To find which of m counts is largest, add independent Laplace noise with scale 1/ε to every count and release only the index of the largest noisy count. Dwork and Roth show this is ε-DP (Claim 3.9). The cost does not grow with m.

That proof needs monotone scores: counts under add/remove, where adding one record raises some counts by at most 1 and lowers none. For scores with sensitivity Δ that are not monotone, such as proportions or per-candidate sums under replace-one, Laplace report noisy max needs scale 2Δ/ε for ε-DP; top-k needs 2kΔ/ε. Scale Δ/ε, or kΔ/ε for top-k, is enough only for monotone scores (Ding et al. 2019, Theorem 2 and its proof).

Adding Gumbel noise instead of Laplace and taking the maximum gives exactly the exponential mechanism's distribution (Remark 3.1).

## Why it matters when writing DP code
- The privacy saving comes from releasing only the index. The full vector of noisy counts has L1 sensitivity up to m when one person can affect every count, so releasing it needs much more noise.
- If the task also asks for the winning value, estimate it with a separate, budgeted noisy release.
- Selecting k items is k selections, not one (see the top-k card).

## Common mistakes
- Adding scale-1/ε noise to every count, then printing the whole sorted noisy list as if it cost ε.
- Using scale Δ/ε on scores that are not monotone, such as shares or ratios; they need 2Δ/ε.
- Reusing the same noise draws for a second selection.
