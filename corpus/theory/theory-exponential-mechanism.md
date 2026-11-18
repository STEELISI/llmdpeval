---
id: theory-exponential-mechanism
title: The exponential mechanism and utility sensitivity
topic: selection
task_types: [selection, quantile]
sources:
  - "McSherry & Talwar 2007 (introduces the exponential mechanism)"
  - "Dwork & Roth 2014, Definition 3.4, Theorem 3.10 and Corollary 3.12"
  - "Durfee & Rogers 2019, Lemma 4.1 (monotone utilities)"
---
## Idea
The exponential mechanism picks one item from a candidate set R. Each candidate r gets a utility score u(x, r) on dataset x. Its utility sensitivity Δu is the largest change in any single candidate's score between neighbouring datasets. The mechanism outputs r with probability proportional to exp(ε · u(x, r) / (2Δu)). This is ε-DP with δ = 0 (Theorem 3.10).

Accuracy: with probability at least 1 − e^(−t), the chosen item's utility is within (2Δu/ε)(ln|R| + t) of the best (Corollary 3.12). More candidates or a larger Δu means a worse choice.

The factor 2 in the exponent can be dropped only when utility is monotone: adding one record moves every score in the same direction, or leaves it unchanged. Durfee and Rogers prove this for add/remove neighbours. Under replace-one, scores can move in opposite directions, so keep the 2.

## Why it matters when writing DP code
- The candidate set must be fixed from public information. Dwork and Roth stress that the range must not depend on the data.
- Choose a utility whose Δu you can bound from public facts, and compute Δu under the task's neighbour relation.
- Release only the chosen item. Releasing the scores is a separate release with its own cost.

## Common mistakes
- Building the candidate list from values that appear in the data.
- Assuming Δu = 1 for a utility that is a share or a ratio.
- Dropping the factor 2 when utility is not monotone.
