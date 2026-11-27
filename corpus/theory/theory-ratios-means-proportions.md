---
id: theory-ratios-means-proportions
title: Means, shares and other ratios need care with the denominator
topic: sensitivity
task_types: [count, mean, classification]
sources:
  - "Dwork & Roth 2014, Definition 3.1 (global sensitivity)"
  - "Dwork & Roth 2014, Example 3.2 (histogram queries)"
  - "Dwork & Roth 2014, Proposition 2.1 and Theorem 3.16"
---
## Idea
A ratio's sensitivity depends on its denominator. With values clamped to public [L, U], replace-one neighbours and a public n, the mean has sensitivity (U − L)/n. Without a public n, or for a share inside a subgroup, the denominator is private and can be tiny. A share within a group of one person can jump from 0 to 1, so its global sensitivity can be as large as 1, not 1/n.

The safe pattern is to release the parts and divide afterwards. Release a noisy numerator and a noisy denominator, each with its share of the budget, then divide. Dividing noisy values is free post-processing. For shares across groups, count every (group, outcome) cell once as one histogram. Then derive group totals and shares by adding and dividing noisy cells. That histogram has L1 sensitivity 1 under add/remove and 2 under replace-one.

## Why it matters when writing DP code
- Never compute a sensitivity from the private group sizes.
- To rank groups by share, either rank shares computed from noisy cells, or use a selection mechanism with a utility whose sensitivity you have bounded.
- Small noisy denominators can give wild or negative shares. Clipping them to [0, 1] afterwards is allowed.

## Common mistakes
- Adding scale-1/ε noise to shares, as if each share were a count.
- Using the true subgroup size in the sensitivity or in the final division.
- Releasing numerator and denominator but charging the budget for only one.
- Releasing noisy shares for every group and charging only for one selection.
