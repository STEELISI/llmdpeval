---
id: theory-sequential-composition
title: Sequential composition and splitting a budget
topic: composition
task_types: [composition]
sources:
  - "Dwork & Roth 2014, Theorem 3.14, Corollary 3.15 and Theorem 3.16"
  - "Kairouz, Oh, Viswanath ICML 2015, Theorem 3.1 and the remark after it"
---
## Idea
Running k private mechanisms on the same data, where mechanism i is (ε_i, δ_i)-DP, gives a combined release that is (ε_1 + … + ε_k, δ_1 + … + δ_k)-DP. This still holds when each mechanism is chosen after seeing earlier outputs (Kairouz, Oh and Viswanath, Theorem 3.1). If no extra δ is allowed, the sum cannot be improved in general.

So a total budget ε must be split: choose ε_1, …, ε_k with ε_1 + … + ε_k ≤ ε, and calibrate each release to its own share.

## Why it matters when writing DP code
- Fix the split before touching the data, from public facts such as the number of queries.
- Each release's noise scale uses its own share. A count given ε/5 needs five times the noise.
- Count every noisy release that reads the data, including internal ones: a noisy denominator, a bound estimate, a selection step.
- Keep one running total for the whole script, charged once per release, loop iterations included. A fresh ledger per call tracks nothing.
- Print the ε the accounting actually holds, and make sure it equals the true loss of the noise you added.

## Common mistakes
- Giving every query the full ε.
- Counting only the printed numbers and forgetting internal releases.
- Computing a scale for one query and reusing it for all k.
- Passing 1 − share, or the noise scale, where the share of ε belongs; printing a scale as ε.
- Changing the split after seeing results.
