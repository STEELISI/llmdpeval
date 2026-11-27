---
id: theory-public-bounds-clamping
title: Clamping bounds and norm bounds must be public
topic: bounds
task_types: [sum, mean, variance, regression, classification, pca]
sources:
  - "Dwork & Roth 2014, Definition 3.1 (global sensitivity)"
  - "Dwork & Roth 2014, Section 3.4, remark after Definition 3.4 (the range must not depend on the data)"
  - "Chaudhuri, Monteleoni, Sarwate 2011, Section 2.1 (records assumed in the unit ball)"
---
## Idea
The sensitivity of a sum, mean, variance or model fit depends on how large one record's values can be. If values are unbounded, so is the sensitivity, and no finite noise works. The fix is to choose public bounds [L, U], clamp every value into them, and compute sensitivity from L and U. For vectors, choose a public norm bound C and scale each row down to norm at most C. Model-fitting results such as Chaudhuri, Monteleoni and Sarwate's assume every record has norm at most 1.

Public means chosen without looking at the private data: from the task text, domain knowledge, documentation, or a measurement scale.

## Why it matters when writing DP code
- Taking the bounds from the data's own minimum and maximum, or the norm bound from the largest row norm, breaks the guarantee. Those values are functions of single records. An outlier sets the bound, and the bound shapes the noise and the output range.
- Clamp, do not drop: clamping keeps n fixed, which replace-one analyses rely on.
- A constant column added to each row counts toward its norm.
- Wide bounds mean more noise. Narrow bounds mean bias from clamping. That is an accuracy trade-off, not a privacy issue.

## Common mistakes
- Bounds from the data's min and max, percentiles, or standard deviation.
- Norm bounds from the largest row norm in the data.
- Standardising features with the data's mean and spread before adding noise.
- Computing sensitivity without clamping first.
