---
id: numpy-logistic-onehot-public-levels-evaluation
title: DP logistic regression on numeric and one-hot categorical features with public levels, plus overall and per-group accuracy
framework: numpy
framework_version: "any"
mechanism: objective-perturbation
task_type: classification
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.gamma
  - numpy.random.Generator.laplace
  - numpy.bincount
  - numpy.linalg.solve
source_url: https://www.jmlr.org/papers/v12/chaudhuri11a.html
source_license: JMLR paper (algorithm paraphrased)
---

## Goal
Train a binary logistic regression with pure ε-DP on a mix of numeric and categorical columns, and release its test accuracy overall and per group, where group sizes are private. Replace-one neighbours (bounded DP: train and test sizes public). Objective perturbation (Chaudhuri, Monteleoni & Sarwate 2011) is a replace-one guarantee.

## Steps
1. Public inputs: bounds `(L, U)` per numeric column and the full level list per categorical column (codebook), the group list, and the budget split (e.g. 0.8 model, 0.2 evaluation).
2. Design matrix with the SAME function for train and test: numeric clamped and min-maxed to [0, 1]; one 0/1 column per public level (unknown values give all zeros). Columns never depend on which values a split contains.
3. Norm: each block (numeric column or one-hot group) adds at most 1 to the squared row norm. Scale all by `1/sqrt(2d)` (d blocks) and add an intercept column `1/sqrt(2)`: every row has norm <= 1 by construction.
4. Objective perturbation (CMS 2011, Alg. 2), labels in {−1, 1}, logistic curvature c = 1/4, public λ: `eps' <- EPS_MODEL − 2·ln(1 + c/(nλ))`; if eps' <= 0 add ridge `c/(n(e^(EPS_MODEL/4) − 1)) − λ` and set eps' = EPS_MODEL/2. Draw b with a uniform direction and `‖b‖ ~ Gamma(p, 2/eps')`. Minimise mean logistic loss + b·w/n + (λ + ridge)/2·‖w‖² (Newton steps).
5. Evaluation: cell = 2·group_index + correct (one extra group for values outside the list). `noisy <- bincount(cells) + laplace(0, 2/EPS_EVAL)` per cell (L1 sensitivity 2 under replace-one).
6. Post-processing: clip cells at 0; accuracy = correct cells / public test size; group accuracy = correct / (correct + wrong) of that group, floor 1, clip to [0, 1].

## Privacy accounting
Model: EPS_MODEL by the CMS theorem (row norm <= 1, c = 1/4, replace-one). Evaluation: one histogram with L1 sensitivity 2 under replace-one at scale 2/EPS_EVAL. `epsilon_spent <- EPS_MODEL + 2/scale`, assert <= EPS.

## Pitfalls
- `pd.get_dummies` on train and test separately: different column counts and a shape error at predict time.
- Scaling with the data mean/std or min/max (private): the row-norm bound then depends on private data.
- A non-private sklearn model feeding the accuracy: the model is a release too.
- Per-group accuracy divided by exact group sizes.
- Output perturbation or noisy weights with an invented sensitivity: use the theorem's conditions.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
