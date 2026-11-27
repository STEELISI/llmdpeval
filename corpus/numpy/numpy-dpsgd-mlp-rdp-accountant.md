---
id: numpy-dpsgd-mlp-rdp-accountant
title: DP-SGD for a small neural network in NumPy (per-example clipping, Poisson batches, RDP accountant)
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: classification
dp_notion: approx
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.random
  - numpy.random.Generator.normal
  - numpy.linalg.norm
  - numpy.logaddexp
  - scipy.special.gammaln
  - scipy.special.xlogy
  - scipy.special.xlog1py
source_url: https://arxiv.org/abs/1607.00133
source_license: original code (algorithm from cited papers)
---

## Goal
Train a small neural network classifier (an MLP) with (ε, δ)-DP by DP-SGD (Abadi et al., CCS 2016): Poisson-sampled mini-batches, per-example gradient clipping, Gaussian noise on the clipped sum, and a Rényi DP (RDP) accountant (Mironov, CSF 2017). Replace-one (bounded DP, N public). δ > 0 is required: the Gaussian mechanism has no pure-ε guarantee. OpenDP 0.16 and diffprivlib 0.6.6 have no DP-SGD or neural networks, so this is NumPy only (Opacus is a PyTorch library and not in the pinned environment).

## Steps
1. Public feature bounds; scale each feature to [−1, 1] and clamp. Label from a public positive value. N public.
2. Public hyperparameters, fixed before training: hidden size, epochs, expected batch size B, clip norm C, learning rate. Sampling rate q = B/N; steps T = epochs·N/B.
3. Calibrate σ before touching the data: bisection for the smallest σ whose accountant ε (below) is at most the target.
4. Each step: `mask <- rng.random(N) < q` (Poisson sampling: each row joins independently). Forward pass, then per-example gradients for every parameter (manual backprop, one row per example), flattened into a matrix G (batch × parameters).
5. Clip each row of G to L2 norm C: `G_i <- G_i · min(1, C/‖G_i‖)`.
6. `g <- (sum_i G_i + normal(0, σ·C, P)) / (q·N)`; the denominator is the public expected batch size, never the realised one.
7. Parameter update; repeat. Release only the final parameters.

## Privacy accounting
One step is the Poisson-subsampled Gaussian. For integer order α its RDP is log(Σ_k C(α,k)(1−q)^(α−k) q^k exp((k²−k)/(2σ²)))/(α−1) (Mironov, Talwar and Zhang 2019, integer-order formula). RDP adds over T steps; convert with ε = T·RDP(α) + log(1/δ)/(α−1), minimised over α (Mironov 2017, Prop. 3). This accountant is for add/remove-one. Under replace-one, go through the removal and the addition: (ε', δ') add/remove gives (2ε', δ'(1+e^ε')) replace-one, so target ε' = ε/2 and δ' = δ/(1+e^(ε/2)). Assert both totals in code.

## Pitfalls
- Clipping the summed or averaged gradient instead of each example's gradient: no sensitivity bound.
- Fixed shuffled batches with the subsampled accountant: the formula assumes Poisson sampling.
- Using an add/remove accountant number as a replace-one guarantee.
- δ = 0 or δ ≥ 1/N.
- Tuning epochs, C or the learning rate on the private data and reporting only the last run's ε.
- Feature scaling or one-hot levels fitted on the private data.
- Per-example gradients through batch normalisation mix rows; use no normalisation or a per-example one.
- Test accuracy on private rows is a further release.
