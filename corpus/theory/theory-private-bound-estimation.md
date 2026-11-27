---
id: theory-private-bound-estimation
title: Estimating bounds privately as a separate budgeted release
topic: bounds
task_types: [other]
sources:
  - "Dwork & Roth 2014, Definition 3.4 and Theorem 3.10 (exponential mechanism)"
  - "Dwork & Roth 2014, Theorem 3.16 (composition)"
---
## Idea
Sometimes no public bound is known. Then the bound itself must be found with a DP mechanism, and that search costs budget like any other release.

A common pattern: fix a public grid of candidate values, for example powers of two across a plausible range. Use a selection mechanism such as the exponential mechanism to pick a candidate near a high quantile of the data. Use a rank-based utility, such as how far the number of records below the candidate is from a target rank. Its sensitivity is small and can be bounded from public facts. Then clamp to the chosen bound and run the main statistic with the remaining budget. By sequential composition the total is ε_bound + ε_statistic.

A private quantile release is the same mechanism used as the final answer.

## Why it matters when writing DP code
- Split the budget before starting, for example a small share for the bound and the rest for the statistic. Both shares count toward the task's ε.
- The candidate grid must be public. Candidates taken from values in the data leak those values.
- Treat the chosen bound as a noisy output. Later steps may use it freely, since that is post-processing.

## Common mistakes
- Estimating bounds with ordinary percentiles and calling the result DP.
- Forgetting to charge the bound estimate to the budget.
- Building the candidate grid from the data's own values.
- Choosing a grid that cannot cover the true range, without noting the bias this causes.
