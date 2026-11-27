---
id: numpy-group-confusion-fairness-metrics
title: Evaluating a trained classifier on test rows (no training) for accuracy, precision, recall and group fairness gaps from one DP (group x true x predicted) table
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.add.at
  - numpy.clip
source_url: https://doi.org/10.1561/0400000042
source_license: original code (Dwork & Roth 2014, Laplace mechanism)
---

## Goal
Evaluate a binary classifier on private held-out rows and release overall accuracy, precision, recall and per-group fairness metrics (statistical parity difference, equal opportunity difference, average odds difference, disparate impact) for a public list of groups. All of them come from ONE noisy count table over (group, true label, predicted label). Replace-one (bounded DP, n public); group sizes are private.

## Steps
1. Train the model by any DP method; predict on the held-out rows (prediction itself is not a release). Public: the group list (declared before looking at the data), the positive label, the test-set size n, which group is unprivileged.
2. Per row (not a release): a token or index for (group, true label, predicted label). Rows whose group is outside the list are dropped.
3. `table <- zeros(G, 2, 2)`; `add.at(table, (g, t, p), 1)`; `noisy <- table + rng.laplace(0, 2/EPS, table.shape)`.
4. Post-processing from the one noisy table (clip negatives to 0; rates clipped to [0, 1]; denominators are noisy sums of cells, floored at 1):
   - accuracy = (TN + TP over all groups) / n; precision = TP / (TP + FP); recall = TP / (TP + FN);
   - per group: positive-prediction rate (TP + FP) / group size, true-positive rate TP / (TP + FN), false-positive rate FP / (FP + TN);
   - statistical parity difference = rate_unpriv − rate_priv; equal opportunity difference = TPR gap; average odds difference = mean of the FPR and TPR gaps; disparate impact = rate ratio.
5. Repeated train/test splits: see the repeated-splits outline; every repeat that reads private test rows is another release.

## Privacy accounting
Replace-one (bounded DP, n public): a changed row leaves one cell and enters another, so the whole table has L1 sensitivity 2 and Laplace(2/EPS) on every cell is one EPS-DP release (Dwork & Roth 2014, Thm. 3.6). NumPy has no accountant: epsilon_spent = 2 / scale; state the relation next to it and assert it is <= EPS. All metrics are post-processing (Prop. 2.1). The model's own epsilon (training rows) and this table's epsilon (test rows) are separate releases; the total is their sum unless the train and test rows are different people and you claim parallel composition explicitly.

## Pitfalls
- `accuracy_score`, `precision_score`, `recall_score` or AIF360 metrics on private test rows: exact statistics, not DP.
- One count per metric numerator and denominator, each with a slice of epsilon: several times the error of one table.
- Exact group sizes as denominators; group sizes are private.
- Group list from the data (`unique()`); empty groups must still get a noisy cell.
- Very small groups give mostly-noise TPR and FPR; report the noisy denominators next to the gaps.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
