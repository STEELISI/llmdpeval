---
id: diffprivlib-private-bounds-then-fit
title: Paying for bounds or data_norm with part of the budget, then clipping and fitting
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.tools.histogram
  - diffprivlib.models.GaussianNB
  - diffprivlib.models.LogisticRegression
  - diffprivlib.accountant.BudgetAccountant
  - numpy.clip
  - numpy.linalg.norm
source_url: https://github.com/DAI-Lab/Tiresias/blob/b1b8fae7416fb34e80e2d2e8a0c449a1e02a62d6/tiresias/core/classification.py
source_license: "none stated (DAI-Lab, MIT); steps paraphrased"
---

## Goal
Train a diffprivlib model when the task gives no feature bounds and no row-norm bound. Instead of letting the model read min/max or the max norm from the data (PrivacyLeakWarning) or computing them yourself (same leak, no warning), spend a stated part of epsilon to estimate them, clip, and fit with the rest. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Split `EPS_TOTAL = EPS_BOUNDS + EPS_MODEL` up front. Histograms and GaussianNB are add/remove-calibrated, so they get half their shares: `acc <- BudgetAccountant(epsilon=EPS_TOTAL/2, delta=0)` for the GaussianNB route.
2. Public, data-independent bin edges on a log scale, e.g. `EDGES = [-2^K, ..., -1, 0, 1, ..., 2^K]` covering any plausible magnitude.
3. For GaussianNB (per-feature `bounds`): for each of the D features,
   `counts, _ <- tools.histogram(x_j, epsilon=EPS_BOUNDS/D/2, bins=EDGES, range=(EDGES[0], EDGES[-1]), accountant=acc)`.
   Lower bound = left edge of the first bin whose noisy count exceeds a noise threshold; upper bound = right edge of the last such bin. Give each call `min(SHARE, acc.remaining()[0])`.
4. For LogisticRegression (`data_norm`): compute each row's L2 norm (`numpy.linalg.norm(X, axis=1)`), histogram the norms the same way with `EPS_BOUNDS/2`, and take the upper edge as `DATA_NORM`. Scale rows with norm above `DATA_NORM` down onto the ball.
5. Clip and shift: `X_c <- numpy.clip(X, LOWERS, UPPERS) - LOWERS`, bounds `(0, UPPERS - LOWERS)`. Append one placeholder row per public class (features at the midpoint) so every class is present.
6. Fit with the remainder on the same accountant:
   - `GaussianNB(epsilon=EPS_MODEL/2, bounds=(0, UPPERS - LOWERS), accountant=acc)`; afterwards `theta_ += LOWERS` (post-processing)
   - or divide rows by `sqrt(2) * DATA_NORM`, append an intercept column `1/sqrt(2)`, and fit `LogisticRegression(epsilon=EPS_MODEL, data_norm=1.0, fit_intercept=False, accountant=acc)` (objective perturbation with row norm <= 1 is charged correctly under replace-one, no halving): diffprivlib 0.6.6 under-charges epsilon when the row norm exceeds 1 or `fit_intercept=True`
7. Report the replace-one total and assert it is <= `EPS_TOTAL`.

## Privacy accounting
Replace-one (bounded DP, n public). A replaced row moves one count out of a bin and one into another, so each fixed-edge histogram costs 2 x the epsilon passed; GaussianNB releases per-class statistics and a replaced row can switch class, so it also costs 2 x. GaussianNB route: everything is halved, report `2 * acc.total()[0]`. LogisticRegression route: report `2 * (histogram share recorded) + EPS_MODEL`. Under sequential composition the total is `EPS_BOUNDS + EPS_MODEL`.

## Pitfalls
- Reporting `acc.total()` with full shares on the histograms or GaussianNB: half the real loss under replace-one.
- DP lower bounds are usually not 0. Without the shift, removing a row changes a class sum by up to max(|L|, |U|), more than the U - L GaussianNB assumes.
- Behaviour of diffprivlib 0.6.6: `GaussianNB` has no `classes=` argument in `fit`; labels come from `np.unique(y)` and only present classes get noise, so an empty class keeps a count of exactly 0. Use the public class list with one placeholder row per class (step 5).
- A fixed epsilon subtracted for `data_norm` (and an assert that epsilon exceeds it) makes small budgets impossible; split proportionally instead.
- `data_norm` set to the exact max row norm of the training data silences the warning but is still read from the private data.
- Edges derived from the data (quantiles, min/max) defeat the purpose.
- Coarse log bins give loose bounds; loose bounds mean more noise. Public domain ranges, when they exist, beat this.
- Add/remove alternative (n private): the library's histograms and GaussianNB are calibrated for add/remove, so under add/remove pass the full shares (`EPS_BOUNDS/D`, `EPS_MODEL`) and report `acc.total()[0]`; under replace-one pass half and report 2x. Keep the shift to lower bound 0 either way.
