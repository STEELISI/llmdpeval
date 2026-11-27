---
id: numpy-confusion-matrix-error-rates
title: Confusion matrix of one or more classifiers from one Laplace histogram each, with overall, false-negative and false-positive rates derived
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.add.at
  - numpy.random.Generator.laplace
  - numpy.clip
source_url: https://doi.org/10.1561/0400000042
source_license: Dwork & Roth 2014 (ideas paraphrased)
---

## Goal
Release how often each true label is predicted as each label, for one or several classifiers evaluated on the same private rows, and derive the error rates (overall, and per class false-negative and false-positive) from the noisy matrices. The labels may include "none" (no true label) and "no class" / "unclassified" predictions. Replace-one row (bounded DP, n public; class sizes private).

## Steps
1. Public: the true-label list and the predicted-label list (codebooks, including the "none" / "no class" entries), the classes to report, n.
2. Per row (not a release): the true label (e.g. from one-hot columns; all zeros -> "none") and each classifier's prediction (ties -> "no class", missing inputs -> "unclassified", as the original code does).
3. Per classifier: matrix of counts over (true, predicted) cells with `add.at`; `noisy <- matrix + laplace(0, 2/eps_each)` on every cell, `eps_each <- EPS / number_of_classifiers`.
4. Post-processing (free), clip each rate to [0, 1]:
   - overall error `<- 1 - (sum of noisy diagonal cells of the reported classes) / n`;
   - false-negative rate of class c `<- 1 - M[c, c] / (noisy row total of c)`;
   - false-positive rate of c `<- (noisy column total of c - M[c, c]) / (noisy total of the other rows)`.
   Denominators are noisy sums of cells, never exact class sizes.

## Privacy accounting
One row is in exactly one cell of each matrix; replace-one moves it to another cell: L1 sensitivity 2 per matrix. With m classifiers on the same rows, sequential composition: `eps_spent <- m * 2/scale`, assert <= EPS.

## Pitfalls
- A separate count for every numerator and denominator (overall, FN, FP per class), each with a slice of EPS: several times the error of one matrix per classifier, same privacy.
- Noise on each final rate with sensitivity 1 and EPS/rates: valid but useless for small classes.
- Exact class sizes as denominators: they are private.
- Dropping rows with no true label or no prediction from the matrix changes what the rates mean; keep them as their own row/column if the original code counts them.
- Sensitivity 1 per matrix: under replace-one it is 2.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
