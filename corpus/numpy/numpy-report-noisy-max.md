---
id: numpy-report-noisy-max
title: Report noisy max (arg-max of Laplace-noised scores) in NumPy
framework: numpy
framework_version: "any"
mechanism: report-noisy-max
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.argmax
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release only the index of the (approximately) largest of K scores, e.g. the most common category in a public list, with epsilon-DP. Dwork & Roth Section 3.3, "Report Noisy Max", Claim 3.9. Default neighbours: replace-one (bounded DP, n public), so counts are not monotone and scale = 2Δ/ε.

## Steps
1. Fix the public, ordered candidate list (see `numpy-laplace-histogram-public-categories` for why it must not come from the data).
2. Exact scores `c[k]`, one per candidate, zeros included.
3. Noise scale from the per-score sensitivity Δ and monotonicity:
   - counts under add/remove (monotone: adding a record never lowers a count), Δ = 1: scale = 1/ε. Claim 3.9 proves exactly this case.
   - any other scores, including counts under replace-one (one count falls while another rises): scale = 2·Δ/ε. Ding, Wang, Zhang & Kifer (PVLDB 2019, arXiv 1904.12773, Theorem 2 and proof) show Laplace scale 2·Δ/ε is ε-DP for sensitivity-Δ scores in general, and Δ/ε suffices only for monotone scores.
4. `noisy <- c + rng.laplace(loc=0, scale=?, size=K)`.
5. Release only `idx <- numpy.argmax(noisy)` (and the candidate name).

## Privacy accounting
Replace-one (bounded DP, n public) by default: one count falls while another rises, so scale = 2Δ/ε (Δ/ε only for monotone counts under add/remove). Releasing only the arg-max costs ε even though every score was noised, because the noisy values stay secret. Publishing the noisy vector turns this into a full Laplace histogram release, accounted with the vector's L1 sensitivity. Releasing the winner's count too needs a fresh Laplace draw with its own ε (sequential composition, Cor. 3.15).

## Pitfalls
- Printing the noisy score vector "for debugging" in the released output.
- scale = Δ/ε for non-monotone scores (replace-one counts, differences, ratios).
- Ties have probability 0 under continuous noise; don't add a tie-break by data order.
- Repeating it k times for a top-k: each repetition costs its own ε; see `numpy-topk-selection`.
- Gumbel noise instead of Laplace gives the exponential mechanism exactly (Dwork & Roth Remark 3.1) with its own 2·Δ/ε rule; don't mix the calibrations.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
