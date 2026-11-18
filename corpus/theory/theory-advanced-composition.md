---
id: theory-advanced-composition
title: Advanced composition and when it does not help
topic: composition
task_types: [composition]
sources:
  - "Dwork & Roth 2014, Theorem 3.20 (Advanced Composition)"
  - "Kairouz, Oh, Viswanath ICML 2015, Theorem 3.3 (optimal composition)"
---
## Idea
Advanced composition trades a little δ for a smaller total ε when there are many releases. If each of k mechanisms is (ε, δ)-DP, then for any chosen δ' > 0 the k-fold combination is (ε', kδ + δ')-DP, with ε' = √(2k ln(1/δ')) · ε + k · ε · (e^ε − 1) (Theorem 3.20). The total grows roughly like √k instead of k, but only once k is large and ε is small.

For small k it is worse than plain addition. With 5 releases at ε = 0.2 each and δ' = 10⁻⁶, the formula gives about 2.57, while basic composition gives 1.0. Kairouz, Oh and Viswanath give the exact optimal bound (Theorem 3.3). With no δ slack at all, the plain sum kε is the best possible.

## Why it matters when writing DP code
- If the task requires δ = 0, use basic composition: the shares must sum to ε.
- Use an advanced bound only when the task allows δ > 0, and then report the δ it adds.
- For the handful of queries in a typical task, an even split under basic composition is usually best, and always correct.

## Common mistakes
- Applying advanced composition under a pure-DP (δ = 0) requirement.
- Using it for a few queries and assuming it must be tighter.
- Forgetting the kδ + δ' term in the reported δ.
- Treating the √k rate as if it held for any k and ε.
