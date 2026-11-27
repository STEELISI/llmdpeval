---
id: numpy-topk-peeling-laplace
title: Top-k by peeling with Laplace report-noisy-max and a per-round epsilon split in NumPy
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
source_license: original code (algorithm from cited paper)
---

## Goal
Release an ordered list of k candidates with high scores from a public list under a TOTAL budget ε, by running Laplace report-noisy-max k times and removing each winner ("peeling"). Report-noisy-max: Dwork & Roth 2014, Claim 3.9; budget: sequential composition, Cor. 3.15 (adaptive choices allowed, Thm. 3.16). Default neighbours: replace-one (bounded DP, n public): counts are not monotone, so each round uses scale 2Δ/ε_round.

## Steps
1. Public, fixed candidate list; exact scores for all candidates, zeros included. Fix Δ (per-score sensitivity) and whether the scores are monotone (see `numpy-noisy-max-score-sensitivity`).
2. Per-round budget: ε_round = ε / k. Fix it before touching the data.
3. Per-round scale: scale = factor·Δ / ε_round with factor 1 for monotone scores, 2 otherwise. So scale grows linearly with k.
4. Loop k times on the remaining candidates:
   - `noisy <- scores[remaining] + rng.laplace(loc=0, scale=?, size=len(remaining))` (fresh noise every round);
   - move `remaining[numpy.argmax(noisy)]` to the output list.
5. Release only the ordered winners.

## Privacy accounting
Replace-one (bounded DP, n public) by default (monotone=False, scale 2Δ/ε_round; Δ/ε_round only for counts under add/remove). Each round is report-noisy-max at ε_round; the set of remaining candidates depends only on earlier outputs, so adaptive sequential composition gives k·ε_round = ε. Keep a running total in code and assert it.

## Pitfalls
- Full ε in every round: total k·ε.
- Reusing one noise draw for all rounds and sorting it: that is one-shot Laplace top-k, which this proof does not cover. Use one-shot Gumbel (`numpy-topk-selection`) or release a full noisy histogram and sort it (`numpy-topk-from-noisy-histogram`).
- Forgetting that scale grows with k: for small candidate lists, the noisy-histogram route has scale Δ₁/ε regardless of k and may be more accurate.
- Releasing the noisy scores of the winners.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
