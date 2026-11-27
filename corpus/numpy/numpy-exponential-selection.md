---
id: numpy-exponential-selection
title: Selecting one candidate with the exponential mechanism in NumPy
framework: numpy
framework_version: "any"
mechanism: exponential
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.choice
  - numpy.exp
  - numpy.max
  - numpy.sum
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: "Dwork & Roth 2014 textbook, author-hosted PDF; definitions paraphrased, no text copied"
---

## Goal
Release one element of a public candidate set R (a category, a threshold, a parameter value) that scores well on a utility u(x, r), with epsilon-DP. Exponential mechanism: McSherry & Talwar 2007; Dwork & Roth Def. 3.4 and Thm. 3.10. Neighbouring relation: replace-one (n public).

## Steps
1. Fix the candidate list R from public information before seeing the data. A candidate present for one dataset and absent for a neighbour breaks the guarantee (Dwork & Roth, remark after Def. 3.4).
2. Utilities: `u[r] <- utility(x, r)` for each r in R.
3. Fix the utility sensitivity Δu: the largest change of u(x, r), for any single r, when x is replaced by a neighbour (state add/remove or replace-one). For a count of records with property r, Δu = 1 under both relations.
4. Probabilities, in log space to avoid overflow:
   `logits <- ε·u / (2·Δu)`; `logits <- logits - numpy.max(logits)`; `p <- numpy.exp(logits)`; `p <- p / numpy.sum(p)`.
5. Sample: `idx <- rng.choice(len(R), p=p)`; release R[idx] only.

## Privacy accounting
Sampling with probability ∝ exp(ε·u / (2·Δu)) is (ε, 0)-DP (Thm. 3.10). The factor 2 may be dropped only for utilities monotone in the data (adding a record never lowers any score), such as counts under add/remove (Durfee & Rogers 2019, Lemma 4.1). Under replace-one, counts are not monotone; keep the 2. Releasing the winner's score as well needs a separate mechanism and more ε.

## Pitfalls
- Candidate set built from the private data, or ties broken by data order.
- Dropping the 2 for non-monotone utilities.
- Exponentiating ε·u directly: overflow gives inf/nan probabilities. Subtract the max first.
- argmax of u (no randomness) or argmax of p (the mode, not a sample): neither is private.
- Assuming Δu = 1/N for a ratio or share (count in group / group size): one record can move a small group's share across the whole [0, 1] range, so Δu = 1 in share units. Score with counts, use the rate with Δu = 1 (safe but noisier), or bound group sizes publicly and derive a smaller Δu.
- Several selections (top-k): each pick spends budget; see `numpy-topk-selection`.
