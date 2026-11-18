---
id: numpy-permute-and-flip
title: Permute-and-flip selection (a better exponential mechanism) in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.permutation
  - numpy.random.Generator.random
  - numpy.exp
source_url: https://arxiv.org/abs/2010.12603
source_license: original code (algorithm from cited paper)
---

## Goal
Select one candidate from a public list with ε-DP, never worse in expected utility than the exponential mechanism at the same ε. Permute-and-flip, McKenna & Sheldon, NeurIPS 2020 (Algorithm 1 and its privacy and utility theorems). Neighbouring relation: replace-one (n public).

## Steps
1. Public, fixed candidate list; exact utilities `q[r]` for all candidates; utility sensitivity Δ (largest change of one candidate's utility when one record changes).
2. `q_star <- max(q)` (used only inside the mechanism, never released).
3. Visit candidates in a fresh random order `rng.permutation(K)`. For each candidate r, flip a coin with heads probability `exp(ε·(q[r] − q_star) / (2Δ))`; return the first r that comes up heads.
4. The best candidate has heads probability 1, so the loop always stops.

## Privacy accounting
McKenna & Sheldon prove the output is ε-DP with the same 2Δ in the denominator as the exponential mechanism. epsilon_spent = 2Δ·(ε/(2Δ)) = ε; compute it from the constants in code rather than returning the input blindly.

## Pitfalls
- Dropping the factor 2 (the mechanism uses ε/(2Δ) for general utilities).
- A fixed visiting order (data order or alphabetical): the permutation must be random on every call.
- Releasing q_star or any utility.
- Using it k times for a top-k list at full ε each: split ε across rounds.
