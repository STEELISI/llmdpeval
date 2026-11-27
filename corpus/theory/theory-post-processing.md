---
id: theory-post-processing
title: Post-processing is free only if it never touches raw data
topic: post-processing
task_types: [other]
sources:
  - "Dwork & Roth 2014, Proposition 2.1 (Post-Processing)"
---
## Idea
If M is (ε, δ)-DP and f is any function, possibly random, that does not use the private data, then f applied to M's output is still (ε, δ)-DP. Free post-processing includes rounding a noisy count, clipping it to a public range, sorting noisy values, adding noisy cells together, or dividing one noisy value by another.

## Why it matters when writing DP code
- Once the noisy outputs exist, compute everything else from them alone.
- Any step that reads the raw data again is a new release. It needs noise and its own share of the budget.
- An exact value computed from the data is a release with no privacy at all. If a task asks you to print the exact answer for checking, label it as non-private, keep it apart, and never feed it into the DP computation. The guarantee covers only the noisy outputs.

## Common mistakes
- Clipping a noisy result to the true minimum and maximum, or to the true total.
- Rescaling a noisy histogram so it sums to the exact n when n is private.
- Re-running the mechanism and keeping the output closest to the true value.
- Sorting by noisy values but breaking ties with true values.
- Reporting the exact value as part of the DP result.
- Adding the true value to a mechanism output that already contains it.
