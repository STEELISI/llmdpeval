---
id: numpy-gaussian-mechanism
title: Gaussian mechanism for (epsilon, delta)-DP sums of bounded columns (mean = noisy sum / public n) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: other
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.sqrt
  - numpy.log
  - numpy.linalg.norm
  - math.erfc
  - math.exp
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release the sums of one or more bounded numeric columns (or another vector statistic: counts, sufficient statistics) with (ε, δ)-DP by adding Gaussian noise calibrated to its **L2** sensitivity; a mean is the noisy sum divided by the public n. Neighbouring relation: replace-one (n public). Dwork & Roth Def. 3.8 (l2-sensitivity), Thm. 3.22 (proof in Appendix A, Thm. A.1). Use only when the task allows δ > 0.

## Steps
1. Bound the statistic (clamp inputs to public bounds) and derive its L2 sensitivity Δ2 = max over replace-one neighbours of ‖f(x) − f(y)‖₂. Examples under replace-one: the sum vector of clamped columns has Δ2 = √(Σ_j (U_j − L_j)²); a histogram has Δ2 = √2. (Add/remove alternative: √(Σ_j max(|L_j|, |U_j|)²) and 1.)
2. Choose δ much smaller than 1/N (Dwork & Roth warn that δ near 1/N permits publishing whole records).
3. Classic calibration (Thm. 3.22), valid only for **0 < ε < 1**: c² > 2·ln(1.25/δ) and σ ≥ c·Δ2/ε. The inequality on c is strict, so take c slightly above √(2·ln(1.25/δ)).
4. For ε ≥ 1, or a tighter σ, use the analytic Gaussian condition (Balle & Wang 2018, Thm. 8): N(0, σ²) noise is (ε, δ)-DP iff
   Φ(Δ2/(2σ) − ε·σ/Δ2) − e^ε · Φ(−Δ2/(2σ) − ε·σ/Δ2) ≤ δ,
   with Φ(t) = ½·erfc(−t/√2) (`math.erfc`). Find the smallest σ by bisection on this check.
5. `noisy <- f_x + rng.normal(loc=0, scale=?, size=D)` with scale = σ — the standard deviation, not the variance.

## Privacy accounting
Δ2 is the replace-one value. Put the condition you used (step 3 or 4) in the code as an assert, so a wrong σ fails loudly; for step 4, evaluate the left-hand side at the chosen σ and assert it is ≤ δ. Several Gaussian releases compose by adding epsilons and deltas (Dwork & Roth Thm. 3.16).

## Pitfalls
- Calibrating to the L1 sensitivity out of Laplace habit; Gaussian uses L2.
- Using the classic formula with ε ≥ 1: Thm. 3.22 does not cover it.
- Passing the variance as `scale`.
- Reporting the result as pure ε-DP: Gaussian noise never gives δ = 0.
- Writing Φ as ½·(1 + erf(·)): it rounds to 0 in the left tail, so the check can pass falsely as 0 − 0 ≤ δ. Use the erfc form.
