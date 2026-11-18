---
id: opendp-private-bounds-log-histogram
title: Estimating clamp bounds privately with a log-band histogram (when no public bounds exist)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.mod.enable_features
  - opendp.mod.binary_search_param
  - opendp.accuracy.discrete_laplacian_scale_to_accuracy
source_url: https://github.com/Hands-on-Differential-Privacy-Book/hands-on-differential-privacy/blob/69c4042fd8843e2647a437f5aae00fe6330d2da6/scripts/ch06_private_bounds_via_histogram.py
source_license: "none stated (code companion to the O'Reilly book Hands-On Differential Privacy); steps paraphrased"
---

## Goal
Find usable lower and upper clamp bounds for a numeric column with a small, separately accounted slice of epsilon, so that the main query does not have to read `min()` / `max()` from the private data. Replace-one neighbours (bounded DP, n public). Use this only when the task gives no bounds and no public range exists; public bounds are always better.

## Steps
1. `enable_features("contrib")`. No other flag is needed: the binning is a public row-wise map applied before OpenDP.
2. Split the budget: `EPS_TOTAL = EPS_BOUNDS + EPS_QUERY`, stated in advance.
3. Binning in pandas/Python: map each value to a signed power-of-two band, e.g. band = sign(x) * (floor(log2|x|) + offset), with its own band for zero and a public rule for missing values. Each row maps to exactly one band, so the map does not change the dataset distance.
4. Count per band over the **full public list of bands** (fixed in advance, not observed bands): `space <- (vector_domain(atom_domain(T=int)), symmetric_distance())`, then `space >> then_count_by_categories(categories=ALL_BANDS, null_category=False) >> then_laplace(scale=?)`.
5. Get the scale for `EPS_BOUNDS` with `binary_search_param(..., d_in=2, d_out=EPS_BOUNDS)` and check `meas.map(2) <= EPS_BOUNDS` (replace-one is symmetric distance 2).
6. Post-process: pick a threshold that pure noise exceeds only with tiny probability (`discrete_laplacian_scale_to_accuracy(scale, alpha / number_of_bands)`). Lower bound = left edge of the first band whose noisy count passes it; upper bound = right edge of the last such band.
7. Use these bounds in the main pipeline (`then_clamp`), funded by `EPS_QUERY`. Compose both releases (`make_composition` or a Context).

## Privacy accounting
The bounds release is a histogram over a public category list; its epsilon comes from `meas.map(2)` under replace-one. The main query's epsilon is separate. The total is their sum under sequential composition; report both parts.

## Pitfalls
- Seen in the wild (Hands-on-Differential-Privacy-Book/hands-on-differential-privacy@69c4042): the script prints `meas.map(1)` for the bounds step (add/remove). Under replace-one evaluate at 2. The bounds are a release and cost budget.
- Seen in the wild (DAI-Lab/Tiresias@b1b8fae): the same log-histogram idea in NumPy, with the bounds epsilon subtracted from the model's epsilon before fitting.
- Wrapping the binning in `make_user_transformation` requires `"honest-but-curious"`, which many task specifications forbid; a public row-wise map before the chain needs no such flag.
- Bands are coarse (powers of two). The bounds can be up to 2x looser than the data range, which adds noise later.
- Small groups may have no band above the threshold; handle the empty case with a fallback public range.
- Do not take the band list from the data; that leaks which magnitudes occur.
