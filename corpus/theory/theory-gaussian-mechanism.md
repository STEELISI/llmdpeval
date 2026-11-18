---
id: theory-gaussian-mechanism
title: The Gaussian mechanism needs delta and L2 sensitivity
topic: approx-dp
task_types: [sum, mean, histogram, pca]
sources:
  - "Dwork & Roth 2014, Definition 3.8 and Theorem 3.22"
  - "Balle & Wang 2018, Theorem 1, Theorem 4 and Theorem 8 (analytic Gaussian mechanism)"
---
## Idea
The Gaussian mechanism releases f(x) plus independent normal noise with standard deviation σ on every coordinate. The classic calibration (Dwork & Roth, Theorem 3.22) is: for 0 < ε < 1 and 0 < δ < 1, the release is (ε, δ)-DP when σ ≥ c · Δ₂/ε with c² > 2 ln(1.25/δ). Δ₂ is the global L2 sensitivity under the chosen neighbour relation.

Three conditions sit inside that formula. It needs δ > 0, so it never gives pure DP. It uses L2 sensitivity, not L1. It is proved only for ε < 1; Balle and Wang show the same rate cannot be extended to larger ε (their Theorem 4). Their analytic calibration (Theorem 8) gives an exact condition for any ε > 0, solved numerically.

## Why it matters when writing DP code
- Use Gaussian noise only if the task allows δ > 0, and print δ with ε.
- Compute σ from Δ₂, ε and δ, all fixed from public facts before looking at the data.
- If ε ≥ 1, do not use the classic formula. Use a calibration proved for that range, or use Laplace.

## Common mistakes
- Adding Gaussian noise with no δ at all.
- Adding "a little" noise whose size is a fraction of the data's own spread. That scale depends on the data and gives no guarantee.
- Using Δ₁ in place of Δ₂, or the classic formula with ε ≥ 1.
- Forgetting that δ adds up across releases.
