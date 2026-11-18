---
id: theory-top-k-selection
title: Top-k selection by count or by proportion, how the budget splits over k, and when noisy histograms are better
topic: selection
task_types: [selection]
sources:
  - "Dwork & Roth 2014, Proposition 2.1 (post-processing), Corollary 3.15 and Theorem 3.16 (basic composition)"
  - "Durfee & Rogers 2019, NeurIPS, Lemma 4.2 and Corollary 4.1"
  - "Gillenwater, Joseph, Medina, Ribero 2022 (ICML), Section 2 setting, Lemma 3.1 and Conclusion"
  - "McKenna & Sheldon 2020, NeurIPS (permute-and-flip is never worse than the exponential mechanism)"
  - "Ding, Wang, Zhang & Kifer 2019, PVLDB 13(3) (noisy max and top-k can also release the gaps at no extra cost)"
---
## Idea
Peeling runs a one-item selection, removes the winner and repeats k times, so each round gets epsilon/k and its noise grows with k. One-shot Gumbel top-k has the same output distribution as peeling with the exponential mechanism, so it costs the same (Durfee and Rogers, Lemma 4.2).

One noisy histogram over all candidates, then a sort, costs epsilon once whatever k is; sorting is post-processing. A joint exponential mechanism samples a whole k-set in one draw. If each person adds to one candidate, Laplace noise on the counts beats joint and peeling (Gillenwater et al., Conclusion). If one person adds to many, histogram noise must cover all of them, so joint or peeling are better. If candidates come from the data, use a thresholded method with delta above zero.

## Why it matters when writing DP code
- For a single winner, permute-and-flip is never worse than the exponential mechanism (McKenna and Sheldon).
- Each peeling round has k times the noise of one histogram, so near-empty candidates win more often at small epsilon.
- Fix k, the candidates and their tie-break order from the task, not the data.

## Common mistakes
- Spending the full epsilon on each of the k rounds, or reusing a scale computed for k = 1.
- Peeling when each person adds to one candidate.
- Histogram noise sized for one candidate when one person adds to many.
- Rate scores with sensitivity 1: use two histograms instead.
