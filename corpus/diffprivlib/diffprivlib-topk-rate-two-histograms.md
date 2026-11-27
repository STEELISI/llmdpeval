---
id: diffprivlib-topk-rate-two-histograms
title: Top-k groups by PROPORTION (rate, share of rows meeting a condition) from two diffprivlib histograms
framework: diffprivlib
framework_version: "0.6.6"
mechanism: geometric
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.accountant.BudgetAccountant
  - numpy.argsort
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/tools/histograms.py
source_license: MIT
---

## Goal
Release the k groups (from a public list) with the highest proportion (rate) of rows meeting a condition, ε-DP under replace-one (bounded DP, n public). Two diffprivlib histograms at ε/2 each (passed as ε/4), then rate = noisy numerator / noisy denominator as post-processing. Preferred route for rate scores.

## Steps
1. Public, fixed group list; map each row's group to its index 0..K-1 (rows outside the list dropped).
2. `acc <- BudgetAccountant()`; histogram arguments: `epsilon=ε/4, bins=K, range=(-0.5, K-0.5), accountant=acc`. The range is public, so no bins are learned from the data.
3. `noisy_num, _ <- histogram(CODES_OF_ROWS_MEETING_CONDITION, ...)`; `noisy_den, _ <- histogram(CODES_OF_ALL_ROWS, ...)`.
4. Post-processing: drop groups whose noisy size is below 10 noise scales (scale 1/(ε/4)), `rate <- noisy_num / noisy_den`, sort, keep k.

## Privacy accounting
Replace-one (bounded DP, n public): one changed row can lower one bin and raise another, L1 sensitivity 2 per histogram, while diffprivlib calibrates for 1. Pass ε/4 to each; each guarantees ε/2 and the two compose: `epsilon_spent <- 2 * acc.total()[0]`, assert <= ε. Under add/remove (n private) pass ε/2 each. The ratio, cut-off and sort are free.

## Pitfalls
- `diffprivlib.mechanisms.Exponential` peeling over rate scores with sensitivity 1 at ε/k per pick: valid but near-random picks when rates differ by a few percent.
- Omitting `range=`: diffprivlib then takes the range from the data (PrivacyLeakWarning).
- Ranking by the count of matching rows instead of the rate: a different query.
- The noisy-size cut-off must be a fixed rule, not tuned on the data.
