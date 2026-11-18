---
id: opendp-noisy-max-selection
title: Selecting the best candidate with report-noisy-max (exponential mechanism)
framework: opendp
framework_version: "0.16.0"
mechanism: report-noisy-max
task_type: selection
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.measurements.make_noisy_max
  - opendp.measurements.then_noisy_max
  - opendp.metrics.linf_distance
  - opendp.measures.max_divergence
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/noisy-max-mechanisms.rst
source_license: MIT
---

## Goal
Release the index of the highest-scoring candidate from a public candidate list, with pure DP under replace-one (bounded DP, n public). With `max_divergence()` OpenDP uses exponential noise (permute-and-flip), which is equivalent to or better than the classic exponential mechanism. Needs only the `"contrib"` feature.

## Steps
1. `enable_features("contrib")`. Do not add `"honest-but-curious"`: this pattern needs no user-defined transformation.
2. Fix `CANDS` from public knowledge, in a fixed order.
3. Build the score vector in pandas over the public list: `SCORES <- [count of rows equal to c, for c in CANDS]` (0 for a candidate that does not occur). Never take the list from the values seen in the data.
4. State the score sensitivity in L-infinity. Counts under replace-one: one row leaves one candidate and joins another, so each score moves by at most 1, in different directions. Use the non-monotonic metric and `d_in = 1`:
   `space <- (vector_domain(atom_domain(T=int)), linf_distance(T=int))`
5. `make <- lambda scale: space >> then_noisy_max(max_divergence(), scale)`
6. `meas <- binary_search_chain(make, d_in=1, d_out=EPS)`; `idx <- meas(SCORES)`; answer = `CANDS[idx]`.
7. To pick the lowest score, pass `negate=True`.

Signature for reference: `make_noisy_max(input_domain, input_metric, output_measure, scale, negate=False)`.

## Privacy accounting
OpenDP prices the release from your stated `d_in`: `meas.map(1)` is 2/scale for the non-monotonic metric (the monotonic metric halves it, but replace-one moves counts in both directions). Let the search pick the scale and assert `meas.map(1) <= EPS`. The `d_in` is your proof that one replaced row moves every score by at most 1; write it next to the code.

## Pitfalls
- Declaring `linf_distance(T=int, monotonic=True)` for replace-one counts, or for scores that can move in different directions (differences, ratios), understates epsilon.
- Rates or shares as scores: the only safe bound is the worst case, 1 per record (a group of one row). A 1/group-size bound read from the data is a leak. Sensitivity 1 swamps rate differences, so prefer two noisy histograms (`opendp-topk-rate-two-histograms`).
- The release is an index only. Releasing the winner's score needs a separate DP query and budget.
- `count_by_categories` outputs an L1 metric; it cannot be chained into noisy max (metric mismatch). Either build the scores as above, or release the noisy histogram and take its argmax (`opendp-topk-count-from-histogram`).
- A `make_user_transformation` score step also works but requires `"honest-but-curious"`; many task specifications forbid that flag.
