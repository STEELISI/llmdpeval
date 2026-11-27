---
id: opendp-select-private-candidate
title: Private selection from private candidates (make_select_private_candidate)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: selection
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.combinators.make_select_private_candidate
  - opendp.combinators.make_composition
  - opendp.transformations.then_count
  - opendp.transformations.then_clamp
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/test/extras/test_make_private_selection_numpy.py
source_license: MIT
---

## Goal
Repeatedly run a DP measurement that returns a `(score, candidate)` pair and release the first run whose noisy score passes a public threshold (Liu and Talwar, STOC 2019, Alg. 1). Typical uses: hyperparameter or model selection where each trial is itself DP. Neighbouring relation: replace-one (n public): sized domain, `d_in=2`.

## Steps
1. `enable_features("contrib")`; `space <- vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()` (N = public row count).
2. Build a scored measurement that releases `[score, candidate]` and satisfies pure DP with epsilon `E_INNER`:
   `m_score <- binary_search_chain(lambda s: space >> then_count() >> then_laplace(s), d_in=2, d_out=E_INNER/2)`
   `m_cand  <- binary_search_chain(lambda s: space >> then_clamp((L, U)) >> then_sum() >> then_laplace(s), d_in=2, d_out=E_INNER/2)`
   `m_scored <- make_composition([m_score, m_cand])`
   (A custom scorer via `make_user_measurement` must declare `TO=(float, "ExtrinsicObject")`.)
3. `m_sel <- make_select_private_candidate(m_scored, stop_probability=P, threshold=T)`
4. `released <- m_sel(DATA)`: a `(score, candidate)` pair, or `None` if it stopped early.

Signature for reference: `make_select_private_candidate(measurement, stop_probability, threshold)`.

## Privacy accounting
The wrapper costs twice the inner epsilon: `m_sel.map(2) == 2 * m_scored.map(2)`. So to spend EPS in total, give the scored measurement EPS/2. Read the total from `m_sel.map(2)`. Alternative, add/remove-one (n private): unsized domain and `d_in=1` throughout (`m_sel.map(1)`).

## Pitfalls
- The inner measurement must be pure DP (max divergence).
- `stop_probability=0` with a threshold that is never reached loops forever; use a small positive value.
- T must be public, not tuned on the data.
- The result can be `None`; handle it.
