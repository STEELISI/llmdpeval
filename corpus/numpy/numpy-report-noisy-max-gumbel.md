---
id: numpy-report-noisy-max-gumbel
title: Report noisy max with Gumbel noise (exact exponential mechanism) in NumPy
framework: numpy
framework_version: "any"
mechanism: report-noisy-max
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.gumbel
  - numpy.argmax
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Release the index of one high-scoring candidate from a public list with ε-DP by adding Gumbel noise to every score and taking the arg-max. This is an exact sampler for the exponential mechanism (McSherry & Talwar 2007; Dwork & Roth 2014, Def. 3.4, Thm. 3.10, and the remark on Gumbel noise in Section 3.4; Durfee & Rogers 2019, Lemma 4.1 for the monotone case). Neighbouring relation: replace-one (n public).

## Steps
1. Fix the candidate list from public knowledge, in a fixed order. Compute every score `u[r]`, zeros included.
2. Fix the score sensitivity Δu: the largest change of any ONE candidate's score when one record changes, under your stated relation. Counts: Δu = 1. Clamped per-candidate sums: max(|L|, |U|) (add/remove) or max(U − L, |L|, |U|) (replace-one: the record may also leave or join that candidate's group).
3. Noise scale:
   - general (non-monotone) scores: scale = 2·Δu / ε;
   - monotone scores (all scores move in the same direction when a record is added, e.g. counts under add/remove): scale = Δu / ε.
4. `winner <- numpy.argmax(u + rng.gumbel(loc=0, scale=?, size=K))`.
5. Release only the winner.

## Privacy accounting
argmax(u_r + Gumbel(β)) picks r with probability ∝ exp(u_r / β). With β = 2Δu/ε that is the exponential mechanism at ε (Thm. 3.10). NumPy has no accountant, so compute epsilon_spent = factor·Δu / scale in code and assert it is ≤ ε.

## Pitfalls
- Using the monotone scale Δu/ε for replace-one counts or rates: those are not monotone.
- Swapping Gumbel for Laplace with the same scale: Laplace noise gives report-noisy-max, a different mechanism with its own proof (`numpy-report-noisy-max`).
- Releasing the noisy scores: then the release is a noisy score vector, which needs vector sensitivity.
- Repeating it k times at full ε for a top-k list: each pick costs its ε (see `numpy-topk-peeling-laplace`).
