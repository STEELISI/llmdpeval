---
id: theory-measuring-dp-utility
title: Measuring the accuracy of DP code honestly - repeated draws, theory checks and controls
topic: implementation
task_types: [other]
sources:
  - "Dwork & Roth 2014, Definition 3.2 and Example 3.1 (Laplace noise; expected error of Lap(1/ε) is 1/ε)"
  - "Dwork & Roth 2014, Theorem 3.16 (each call of a mechanism spends budget again)"
  - "Ding, Wang, Wang, Zhang & Kifer 2018, CCS (judging a mechanism from many repeated runs on fixed inputs)"
  - "Ponomareva et al. 2023, JAIR 77 (report non-private baselines and hyperparameter choices next to DP results)"
---
## Idea
A DP output is one draw from a distribution. A single error, good or bad, says little about the mechanism. To judge it, run it many times on the same data and look at the spread.

Theory gives a check that can fail. For Laplace noise with scale b, the mean absolute error is b. Over many runs the measured error should match the scale the framework chose. If a version with twice the sensitivity does not show twice the error, the calibration is wrong somewhere.

These repeated runs are simulation, not release. In a real deployment each private function runs once; every extra call spends the budget again.

## Why it matters when writing DP code
- State the expected error from the scale, then compare a run against it.
- Quote numbers from unseeded runs as ranges, or print them from the code.
- For models, report three rows: non-private at default settings, non-private at the same regularisation as the DP model, and the DP model. Show the majority-class rate too, because accuracy near it means nothing.
- A DP result that equals the exact result in every digit, where a group is small, means the noise is missing or far too small.

## Common mistakes
- Reporting one draw as the accuracy.
- Seeding a DP mechanism to get repeatable output.
- Calling the gap to a default-settings baseline the cost of privacy.
- Computing test metrics on private rows without noise.
