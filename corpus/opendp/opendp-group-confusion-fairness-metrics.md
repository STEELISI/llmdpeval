---
id: opendp-group-confusion-fairness-metrics
title: Evaluating a trained classifier on test rows (no training) - accuracy, precision, recall and group fairness gaps from one DP (group x true x predicted) table
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://docs.opendp.org/en/v0.16.0/api/python/opendp.transformations.html
source_license: MIT
---

## Goal
Evaluate a binary classifier on private held-out rows and release overall accuracy, precision, recall and per-group fairness metrics (statistical parity difference, equal opportunity difference, average odds difference, disparate impact) for a public list of groups. All of them come from ONE noisy count table over (group, true label, predicted label). Replace-one (bounded DP, n public); group sizes are private.

## Steps
1. Train the model by any DP method; predict on the held-out rows (prediction itself is not a release). Public: the group list (declared before looking at the data), the positive label, the test-set size n, which group is unprivileged.
2. Per row (not a release): a token or index for (group, true label, predicted label). Rows whose group is outside the list are dropped.
3. Public cell list `cells <- ["g|t|p" for every group, t in (0,1), p in (0,1)]`; `meas <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=cells, null_category=False) >> then_laplace(scale=?)`; scale from `binary_search_param(make, d_in=2, d_out=EPS)`; `noisy <- meas(tokens)` reshaped to (G, 2, 2).
4. Post-processing from the one noisy table (clip negatives to 0; rates clipped to [0, 1]; denominators are noisy sums of cells, floored at 1):
   - accuracy = (TN + TP over all groups) / n; precision = TP / (TP + FP); recall = TP / (TP + FN);
   - per group: positive-prediction rate (TP + FP) / group size, true-positive rate TP / (TP + FN), false-positive rate FP / (FP + TN);
   - statistical parity difference = rate_unpriv − rate_priv; equal opportunity difference = TPR gap; average odds difference = mean of the FPR and TPR gaps; disparate impact = rate ratio.
5. Repeated train/test splits: every repeat that reads private test rows is another release and composes sequentially; split the budget across repeats, or release from disjoint row chunks.

## Privacy accounting
Replace-one (bounded DP, n public) is symmetric distance 2 (one token removed, one added). The scale comes from `binary_search_param` at d_in=2 and epsilon_spent = `meas.map(2)`; assert it is <= EPS. Evaluating the map at d_in=1 reports half the real loss. OpenDP has no classifiers; train the model with another DP method (or treat the model as public) and use OpenDP only for the evaluation release. The model's own epsilon (training rows) and this table's epsilon (test rows) are separate releases; the total is their sum unless the train and test rows are different people and you claim parallel composition explicitly.

## Pitfalls
- `accuracy_score`, `precision_score`, `recall_score` or AIF360 metrics on private test rows: exact statistics, not DP.
- One count per metric numerator and denominator, each with a slice of epsilon: several times the error of one table.
- Exact group sizes as denominators; group sizes are private.
- Group list from the data (`unique()`); empty groups must still get a noisy cell.
- Very small groups give mostly-noise TPR and FPR; report the noisy denominators next to the gaps.
