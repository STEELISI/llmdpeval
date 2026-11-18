---
id: numpy-per-record-noise-input-perturbation
title: Noise on every record (input perturbation) - local DP, its sensitivity, and why it is not a central release
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.clip
  - numpy.mean
source_url: https://github.com/Anishreddy-gith/ISRA-ITDS/blob/5c8dee205b63fc728fda4e40a058967fd1fcb93f/services/ml_engine/app/privacy/differential_privacy.py
source_license: MIT
---

## Goal
Release a perturbed copy of a table, one noisy value per record and column, so each record is protected on its own (local DP, Laplace input perturbation). Many projects call this "applying DP to the dataset"; it is valid only with the per-record sensitivity below, and it is much noisier than adding noise to an aggregate.

## Steps
1. Public bounds per column `(L_j, U_j)` from domain knowledge, never from the column's min/max.
2. Clip: `x_c <- numpy.clip(x_j, L_j, U_j)`.
3. Per-record sensitivity of one numeric value is the whole range: `Δ_j = U_j - L_j`. Changing a person changes that value anywhere in the range, so it is not 1 and not a fraction of the range.
4. Budget: each record releases D columns, so split `EPS` across columns, e.g. `EPS_j = EPS / D` (sequential composition within a record).
5. `rng <- numpy.random.default_rng()`; `noisy_j <- x_c + rng.laplace(0, Δ_j / EPS_j, size=n)`.
6. Optional post-processing: clip noisy values back to `(L_j, U_j)`; any statistic computed from the noisy table is free.
7. State in the output that this is local DP per record, with epsilon EPS per record.

## Privacy accounting
Formula (assumptions: one record per person, public bounds, Laplace with scale Δ_j/EPS_j per value): each record's released row is EPS-DP with respect to that record's true row. There is no further cost for aggregates of the noisy table. The error of a mean over n noisy values is about `sqrt(2) * Δ_j / EPS_j / sqrt(n)`, far larger than a central DP mean (`Δ_j / (n * EPS_j)`). If the task asks for an aggregate, prefer central DP.

## Pitfalls
- Seen in the wild (CSG-AISym4MED/PREDANO_Privacy_Enhancement_Data_Anonymization_Toolbox@d1913c6): per-record noise with sensitivity set to 1% of the column's observed range: no DP guarantee, and the range itself is read from the data.
- Seen in the wild (LaraujoM/ciber-app-poc@ef7bc8c): `LaplaceBoundedDomain(sensitivity=1, lower=min, upper=max)` applied to every value; sensitivity 1 is wrong for a numeric column and the bounds are the data's own min/max.
- Seen in the wild (MortenMunk/PE-capstone@b804283): options for Laplace, Gaussian or uniform noise all with scale `sensitivity/epsilon`; uniform jitter has no DP guarantee, and Gaussian needs delta and its own calibration.
- Seen in the wild (Anishreddy-gith/ISRA-ITDS@5c8dee2): per-feature epsilon = EPS / D with sensitivity = feature range, as above; but the features are standardised with a scaler fit on the private data first, so the bounds apply to data-dependent units, and the noise generator is seeded with a fixed `random_state=42`.
- Using the same noise draw for two columns, or a fixed seed: see `numpy-rng-seeding`.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
