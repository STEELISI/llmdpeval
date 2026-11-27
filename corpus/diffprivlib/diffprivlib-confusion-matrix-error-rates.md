---
id: diffprivlib-confusion-matrix-error-rates
title: Confusion matrix of one or more classifiers as one diffprivlib Laplace release each, with overall, false-negative and false-positive rates derived
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
Release how often each true label is predicted as each label, for one or several classifiers evaluated on the same private rows, and derive the error rates (overall, per class false-negative and false-positive) from the noisy matrices. Labels may include "none" (no true label) and "no class" / "unclassified" predictions. Replace-one row (bounded DP, n public; class sizes private).

## Steps
1. Public: true-label list and predicted-label list (with the "none" / "no class" entries), classes to report, n.
2. Per row (not a release): true label and each classifier's prediction, as the original code defines them.
3. `acc <- BudgetAccountant(epsilon=EPS, delta=0)`; `eps_each <- EPS / m` for m classifiers; per classifier `e <- min(eps_each, acc.remaining()[0])`, `acc.check(e, 0)`, `mech <- Laplace(epsilon=e, sensitivity=2)`, then `acc.spend(e, 0)` after the release.
4. Per classifier: exact matrix over (true, predicted) cells with `add.at`; `noisy <- vectorize(mech.randomise)(matrix)`; `acc.spend(eps_each, 0)` once per matrix.
5. Post-processing, clipped to [0, 1]: overall error `1 - (noisy diagonal of reported classes) / n`; FN of c `1 - M[c,c] / row_total(c)`; FP of c `(column_total(c) - M[c,c]) / (total of the other rows)`; totals are sums of noisy cells.

## Privacy accounting
A row is in one cell; replace-one moves it to another: L1 sensitivity 2 per matrix, so Laplace(2/eps_each) on every cell is one eps_each-DP release. Same rows in every matrix: sequential composition, `eps_spent <- acc.total()[0]`, assert <= EPS.

## Pitfalls
- `tools.histogram2d` / `tools.histogram` assume sensitivity 1 per row (add/remove); under replace-one pass half the epsilon and report twice the accountant, or use the mechanism as above.
- `count_nonzero` per numerator and denominator, each with a slice of EPS: several times the error of one matrix per classifier.
- `tools.mean` of an indicator inside a class: its denominator is the private class size.
- Floating-point: m equal shares of epsilon can add up to a hair above a capped accountant's limit and raise `BudgetError`. Keep the accountant capped at EPS and give each release `min(SHARE, acc.remaining()[0])`; assert the reported total <= EPS. An uncapped `BudgetAccountant()` tracks but never enforces; never raise the cap to make it fit.
- Noise on each final rate with a guessed sensitivity.
- Add/remove alternative (n private): a row is added to or removed from its cells instead of moving between them, so the L1 sensitivity is half the replace-one value used here; anything divided by n then needs a noisy n.
