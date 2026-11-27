---
id: diffprivlib-group-confusion-fairness-metrics
title: Evaluating a trained classifier on test rows (no training) - accuracy, precision, recall and group fairness gaps from one DP (group x true x predicted) table
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.accountant.BudgetAccountant
  - numpy.add.at
  - numpy.vectorize
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Evaluate a binary classifier on private held-out rows and release overall accuracy, precision, recall and per-group fairness metrics (statistical parity difference, equal opportunity difference, average odds difference, disparate impact) for a public list of groups. All of them come from ONE noisy count table over (group, true label, predicted label). Replace-one (bounded DP, n public); group sizes are private.

## Steps
1. Train the model by any DP method; predict on the held-out rows (prediction itself is not a release). Public: the group list (declared before looking at the data), the positive label, the test-set size n, which group is unprivileged.
2. Per row (not a release): a token or index for (group, true label, predicted label). Rows whose group is outside the list are dropped.
3. Exact table with `add.at`; `mech <- Laplace(epsilon=EPS, sensitivity=2)`; `noisy <- vectorize(mech.randomise)(table)`; `acc.spend(EPS, 0)` once (mechanisms do not charge an accountant).
4. Post-processing from the one noisy table (clip negatives to 0; rates clipped to [0, 1]; denominators are noisy sums of cells, floored at 1):
   - accuracy = (TN + TP over all groups) / n; precision = TP / (TP + FP); recall = TP / (TP + FN);
   - per group: positive-prediction rate (TP + FP) / group size, true-positive rate TP / (TP + FN), false-positive rate FP / (FP + TN);
   - statistical parity difference = rate_unpriv − rate_priv; equal opportunity difference = TPR gap; average odds difference = mean of the FPR and TPR gaps; disparate impact = rate ratio.
5. Repeated train/test splits: see the repeated-splits outline; every repeat that reads private test rows is another release.

## Privacy accounting
Replace-one (bounded DP, n public): one changed row moves between two cells, L1 = 2 for the whole table, so Laplace with sensitivity 2 on every cell is one EPS-DP release. `tools.histogramdd` is calibrated for add/remove (sensitivity 1): passing EPS there guarantees only 2 x EPS under replace-one; use the mechanism as above, or pass EPS/2 and report twice the accountant. Report `acc.total()` and assert it is <= EPS. The model's own epsilon (training rows) and this table's epsilon (test rows) are separate releases; the total is their sum unless the train and test rows are different people and you claim parallel composition explicitly.

## Pitfalls
- `accuracy_score`, `precision_score`, `recall_score` or AIF360 metrics on private test rows: exact statistics, not DP.
- One count per metric numerator and denominator, each with a slice of epsilon: several times the error of one table.
- Exact group sizes as denominators; group sizes are private.
- Group list from the data (`unique()`); empty groups must still get a noisy cell.
- Very small groups give mostly-noise TPR and FPR; report the noisy denominators next to the gaps.
- Add/remove alternative (n private): a row is added to or removed from its cells instead of moving between them, so the L1 sensitivity is half the replace-one value used here; anything divided by n then needs a noisy n.
