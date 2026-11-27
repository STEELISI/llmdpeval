---
id: numpy-topk-selection
title: Top-k candidates by row COUNT (k > 1) by peeling or one-shot Gumbel noise in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.gumbel
  - numpy.argsort
source_url: https://arxiv.org/abs/1905.04273
source_license: "Durfee & Rogers 2019 (NeurIPS) arXiv preprint; results paraphrased, no code copied"
---

## Goal
Release the ordered list of the k best candidates (k > 1) from a public candidate list under a **total** budget ε. Two routes with the same privacy: peeling (k sequential selections) or one-shot Gumbel noise. Durfee & Rogers 2019, Lemma 4.2: one-shot Gumbel top-k has the same output distribution as k rounds of exponential-mechanism peeling. Neighbouring relation: replace-one (n public).

## Steps
1. Public, fixed candidate list (K entries) and exact scores `c` (counts, zeros included). Fix the per-score sensitivity Δ and the relation.
2. Split the budget: each of the k picks gets ε_pick = ε / k (basic sequential composition, Dwork & Roth Cor. 3.15).
3. Route A, one-shot Gumbel:
   - replace-one (default): counts are not monotone (one count goes down, another up), so scale = 2·Δ / ε_pick (= 2·k·Δ / ε);
   - only for monotone scores, e.g. counts under add/remove: scale = Δ / ε_pick (Durfee & Rogers Lemma 4.1).
   - `noisy <- c + rng.gumbel(loc=0, scale=?, size=K)`; `topk <- first k of numpy.argsort(-noisy)`.
4. Route B, peeling: repeat k times — one selection at ε_pick on the remaining candidates (`numpy-exponential-selection` or `numpy-report-noisy-max`), record the winner, remove it.
5. Release only the k indices, in order. Not the noisy or true scores.

## Privacy accounting
Replace-one, factor 2 in the scale. k picks at ε/k each total ε (pure DP, basic composition). Route A inherits this because it equals peeling in distribution. Assert k·ε_pick ≤ ε in code; print k, ε_pick, scale and the relation. Durfee & Rogers also give tighter (ε, δ) bounds via advanced composition; use them only if the task allows δ > 0.

## Pitfalls
- Spending the full ε on each pick (total k·ε).
- One-shot **Laplace** noise plus sorting is not covered by this argument; use Gumbel for one-shot, or peel.
- Monotone scale under replace-one: counts there are not monotone.
- Releasing the winners' noisy or true scores.
- Candidate list taken from categories observed in the data.
- Share-within-group scores (rates): sensitivity 1, not 1/N, and not monotone; at ε/k per pick the picks are near random. Use numpy-topk-rate-two-histograms.
