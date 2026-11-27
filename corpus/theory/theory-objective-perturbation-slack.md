---
id: theory-objective-perturbation-slack
title: Objective perturbation, its slack term, and why regularisation is part of the privacy proof
topic: mechanism
task_types: [classification]
sources:
  - "Chaudhuri, Monteleoni, Sarwate 2011, JMLR 12, Section 2 (records in the unit ball), Algorithm 2 and Theorem 9"
  - "diffprivlib 0.6.6, mechanisms/vector.py, Vector.randomise (slack and fallback branch)"
  - "Ponomareva et al. 2023, JAIR 77 (regularisation and hyperparameter tuning are part of the DP training setup)"
---
## Idea
Objective perturbation adds a random linear term to the training loss and then solves the perturbed problem exactly. Its privacy proof needs three things. Every record lies in the unit ball. The loss has bounded first and second derivatives; logistic loss has curvature bound c = 1/4. And the L2 penalty makes the objective strongly convex in every coefficient.

Part of ε goes to a slack term before any noise. With row norm 1, regularisation parameter C and logistic loss, diffprivlib's slack is 2 ln(1 + C/4). The rest goes to the noise. If the slack is at least ε, the library silently switches to a fallback: half of ε plus an extra ridge term.

## Why it matters when writing DP code
- The regularisation strength belongs to the proof. Fix it by a public rule before training, for example keeping the slack below a quarter of ε.
- Just above the fallback point almost no budget is left for the noise, so the noise is huge. A smaller ε that triggers the fallback can then give a better model.
- To report the cost of privacy, compare against a non-private model at the same regularisation. Strong regularisation alone can cost more accuracy than the noise.

## Common mistakes
- Setting C to 1/n in diffprivlib, which is a unit ridge.
- The library's fitted intercept (unpenalised) or a norm above 1; version 0.6.6 then under-charges the slack.
- Tuning C on test accuracy.
- Comparing against a default-C baseline.
