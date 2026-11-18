---
id: theory-dp-regression
title: DP regression via sufficient statistics or perturbation
topic: mechanism
task_types: [regression]
sources:
  - "Chaudhuri, Monteleoni, Sarwate 2011, JMLR 12, Section 2.1, Algorithms 1 and 2, Theorems 6 and 9"
  - "Dwork & Roth 2014, Theorem 3.6 and Theorem 3.16"
  - "Dwork & Roth 2014, Proposition 2.1"
---
## Idea
Three common ways to fit a model privately:
- Sufficient statistics: least squares depends only on sums of feature products and of feature times target. Clamp to public bounds, add noise to the sums, and solve from them; solving is post-processing.
- Output perturbation (Algorithm 1): fit a regularised model, then add noise to the coefficients (Theorem 6).
- Objective perturbation (Algorithm 2): add a random linear term to the training objective, then minimise. Its guarantee (Theorem 9) needs a smooth convex loss with bounded derivatives and a strongly convex regulariser, and it spends part of ε on a slack term.

Both perturbation results assume each record's features have norm at most 1 (Section 2.1) and use replace-one neighbours.

## Why it matters when writing DP code
- Set the feature norm bound and value bounds from public facts, then scale and clamp. A constant column for the intercept counts toward the norm.
- Split ε across every noisy sum or step.
- An unstable solve from noisy sums may be fixed with a public adjustment.

## Common mistakes
- Scaling features by the data's largest row norm, or standardising with the data's mean and spread.
- Taking target bounds from the data's minimum and maximum.
- Reporting exact accuracy on the private training data as DP output.
- Labels in {-1, +1} with a gradient written for {0, 1}: the loss is then 2-Lipschitz and output-perturbation noise must double.
- A buggy hand-written optimiser (a flipped gradient sign) releases a vector the proof does not cover.
