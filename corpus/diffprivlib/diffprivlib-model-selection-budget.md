---
id: diffprivlib-model-selection-budget
title: Epsilon sweeps and hyperparameter search on private data - what each fit and the final pick cost
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - diffprivlib.accountant.BudgetAccountant
  - diffprivlib.models.LogisticRegression
  - diffprivlib.models.GaussianNB
  - diffprivlib.mechanisms.Exponential
source_url: https://github.com/DAI-Lab/Tiresias/blob/b1b8fae7416fb34e80e2d2e8a0c449a1e02a62d6/tiresias/core/classification.py
source_license: "none stated (DAI-Lab, MIT); steps paraphrased"
---

## Goal
Choose between several DP models (different C, different model types) trained on private data, and release one, with the total epsilon of the whole search stated. A common real-world pattern trains at many epsilons or many C values, keeps the best validation score, and reports only that model's epsilon. Neighbouring relation: replace-one (bounded DP, n public).

## Steps
1. Decide the candidate list in advance (K configurations) and the total budget. One shared `acc <- BudgetAccountant(epsilon=EPS_TOTAL, delta=0)`.
2. Split the rows once: a training part and a held-out validation part of M rows.
3. Train each candidate on the training part with `EPS_FIT` and `accountant=acc`, e.g. `LogisticRegression(epsilon=EPS_FIT, data_norm=1.0, C=C_k, fit_intercept=False, accountant=acc)` on rows publicly scaled to norm <= 1 with an explicit intercept column (0.6.6 under-charges epsilon for norms above 1) or `GaussianNB(epsilon=EPS_FIT/2, bounds=(0, U_vec - L_vec), accountant=acc)` on features shifted by their lower bounds (per-class releases: under replace-one pass half and count it twice; under add/remove, n private, pass the full EPS_FIT; append one placeholder row per public class, since diffprivlib 0.6.6 noises only classes present in y). For LogisticRegression, cap C by the public rule `C <- min(C, 4*expm1(EPS_FIT/8))` so the slack fits inside EPS_FIT. Each fit spends `EPS_FIT`; K fits on the same rows spend `K * EPS_FIT`.
4. Score each candidate on the validation part (accuracy in [0, 1]; changing one validation row moves each score by at most 1/M).
5. Pick privately: `mech <- Exponential(epsilon=EPS_SEL, sensitivity=1/M, utility=list(scores), monotonic=False)`; `k <- mech.randomise()`; then `acc.spend(EPS_SEL, 0)` because mechanisms used directly do not spend on the accountant.
6. Release only model k. If you refit model k on all rows, that is another `EPS_FIT` and must also be charged.
7. Report `acc.total()`.

## Privacy accounting
Replace-one (bounded DP, n public). Conservative total: `K * EPS_FIT + EPS_SEL (+ EPS_FIT for a final refit)`. The accountant records the fits automatically; add the selection by hand with `spend`. Reporting `EPS_FIT` alone for the chosen model under-states the cost by a factor of about K.

## Pitfalls
- Floating-point: k equal shares of epsilon can add up to a hair above the accountant's cap (for example 9, 11, 18 or 20 shares of 1.0), and the last spend raises `BudgetError: Privacy spend of (...) not permissible`. Give each step `min(SHARE, acc.remaining()[0])`; never raise the cap to make it fit.
- A common pattern: one model trained per epsilon in a list, the epsilon with the best validation score kept, then refit; the sweep, the non-private argmax and the refit are all unaccounted.
- Selection by an exponential mechanism is right, but if every candidate spends the full model share on the same rows and the winner is refit on all rows, the real total is several times the stated epsilon.
- Picking by `argmax` of validation scores is not private when the validation rows are private.
- Reusing one `random_state` across candidates makes their noise identical, not independent.
- C above the cap: the slack `2*ln(1 + C/4)` exceeds EPS_FIT and diffprivlib 0.6.6 silently switches to its fallback (half the epsilon plus extra ridge); privacy holds but candidates no longer differ only in C.
