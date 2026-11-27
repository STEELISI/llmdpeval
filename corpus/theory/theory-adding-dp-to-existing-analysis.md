---
id: theory-adding-dp-to-existing-analysis
title: Adding DP to an existing analysis script, step by step
topic: composition
task_types: [composition, other]
sources:
  - "Dwork & Roth 2014, Theorem 3.16 (sequential composition) and Proposition 2.1 (post-processing)"
  - "Ponomareva et al. 2023, JAIR 77, How to DP-fy ML (models are releases)"
  - "Barrientos et al. 2019, JCGS 28(2), DP significance tests"
---
## Idea
Treat the script as a list of releases. Follow the private data to every value the script prints, returns, saves or plots. Each such value is a release: counts, means, tables, a trained model, its metrics, and test results.

Next sort every input into public or private. Public: the total row count when the task says so, codebook bounds, category lists, design constants. Private: everything computed from rows, including group sizes not published, data means, minimums and maximums, and the list of values that occur.

Keep preprocessing free of private statistics: clamp to public bounds, encode with public category lists, scale with bounds, never with the data mean or standard deviation. Then split one budget over all releases and give each a mechanism and a stated sensitivity under the task's neighbouring relation.

## Why it matters when writing DP code
- Print the original and the DP value of each release side by side, labelled, only when the task allows it.
- Where the analysis tests significance, use a noise-aware test.
- Clip post-processed values to their valid range.

## Common mistakes
- Adding noise to the metrics but releasing a non-private model.
- Forgetting a release, so the budget spent is larger than claimed.
- Feeding the mechanism a constant, a copied statistic or a random number instead of the private rows.
- ε printed as 0, noise drawn but never added, or a δ the task did not allow.
- Enabling a framework option the task rules out.
