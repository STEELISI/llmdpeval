---
id: theory-epsilon-meaning
title: What epsilon means
topic: epsilon
task_types: [other]
sources:
  - "Dwork & Roth 2014, Definition 2.4"
  - "Kairouz, Oh, Viswanath ICML 2015, Theorem 2.1"
---
## Idea
A mechanism M is ε-differentially private (pure DP, δ = 0) if, for every pair of neighbouring datasets x and y and every set of outputs S, Pr[M(x) in S] ≤ e^ε · Pr[M(y) in S]. Here ε ≥ 0 is the privacy-loss budget, and smaller means more private. The probability is over the mechanism's own random noise, not over the data.

One useful reading: an attacker who must guess which of two neighbouring datasets was used, starting from even odds, can be right with probability at most e^ε / (1 + e^ε). That is about 52.5% at ε = 0.1, 73.1% at ε = 1 and 95.3% at ε = 3. Put differently, seeing the output can multiply the attacker's prior odds by at most e^ε. This reading assumes δ = 0.

## Why it matters when writing DP code
- ε is a budget for the whole program's release, not a setting for each printed number. Every noisy value spends part of it.
- The guarantee comes only from the noise. A value computed from the private data and released without noise is not covered.
- Take ε from the task. Do not raise it, and do not split it so the parts add up to more than the task allows.

## Common mistakes
- Spending the full ε on each of several outputs instead of splitting it.
- Thinking a larger ε is more private. It is the opposite.
- Believing DP hides aggregate facts. It limits what one person's record can change, nothing more.
