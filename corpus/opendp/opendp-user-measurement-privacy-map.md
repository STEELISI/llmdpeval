---
id: opendp-user-measurement-privacy-map
title: Wrapping custom code with make_user_measurement without lying in the privacy map
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.measurements.make_user_measurement
  - opendp.transformations.make_user_transformation
  - opendp.domains.user_domain
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.metrics.l1_distance
  - opendp.measures.max_divergence
  - opendp.measurements.then_laplace
  - opendp.mod.enable_features
  - opendp.mod.binary_search_param
source_url: https://github.com/KnowledgeEdgeAI/PETs_for_Public_Health_Challenge/blob/9cef8e790d868d316af009935926873089a2d50a/src/DP_epidemiology/utilities.py
source_license: MIT
---

## Goal
Release a statistic that OpenDP has no built-in constructor for (a dataframe filter, a custom score) by wrapping your own Python function, while keeping the epsilon that `meas.map(d_in)` reports tied to real noise. Real projects use this pattern for pandas dataframes and grouped sums. Neighbouring relation: replace-one (n public), so maps are read at `d_in=2` on `symmetric_distance()`.

## Steps
1. `enable_features("contrib", "honest-but-curious")`. The second flag is required because OpenDP cannot check your function or your map.
2. Prefer a user **transformation** plus a built-in **measurement**:
   - `t <- make_user_transformation(input_domain=?, input_metric=symmetric_distance(), output_domain=vector_domain(atom_domain(T=?)), output_metric=l1_distance(T=?), function=f, stability_map=lambda d_in: d_in * C)`
     - `f` must clamp each record to public bounds `(L, U)` itself, so that each added or removed record moves the output by at most `C = max(|L|, |U|)` in L1. Under replace-one (`d_in=2`) a changed record can leave one group and join another, so the change is up to `2*C` = `stability_map(2)`.
   - `meas <- t >> then_laplace(scale=?)`; find the scale with `binary_search_param(..., d_in=2, ...)` and read epsilon with `meas.map(2)`.
3. If you must use `make_user_measurement` (e.g. the input is a dataframe in a `user_domain("...", member=?)`), build the noise from an OpenDP measurement and **derive the map from it**:
   - `inner <- space >> then_laplace(scale=?)` (space = the vector domain and l1 metric of the aggregate)
   - `make_user_measurement(input_domain=?, input_metric=symmetric_distance(), output_measure=max_divergence(), function=lambda df: inner(aggregate(df)), privacy_map=lambda d_in: inner.map(t_agg.map(d_in)))`
4. Print `meas.map(2)` next to the release. Alternative, add/remove-one (n private): search at `d_in=1` and report `meas.map(1)` (change up to C).

Signature: `make_user_measurement(input_domain, input_metric, output_measure, function, privacy_map, TO='ExtrinsicObject')`

## Privacy accounting
The privacy map of a user measurement is whatever the lambda returns. It is only correct if it is computed from the maps of the OpenDP pieces that actually add the noise (`inner.map(t.map(d_in))`). Any map written as a constant or as `d_in * EPS` is an assertion, not accounting.

## Pitfalls
- Seen in the wild (KnowledgeEdgeAI/PETs_for_Public_Health_Challenge@9cef8e7): `privacy_map=lambda d_in: d_in*epsilon` around an inner count whose Laplace scale is fixed at 1; the reported epsilon has no link to the noise. The same file's grouped sum does it right with `m_lap.map(t_sum.map(d_in))`.
- Seen in the wild (same repo): `numpy.random.laplace` called inside the user function. That bypasses OpenDP's samplers; use an OpenDP measurement for the noise.
- The stability map must cover the worst record. A filter or group-by that lets one person contribute many rows needs a larger `d_in`: set it to the maximum number of rows per person (times 2 under replace-one), and keep the stability map per unit of `d_in`.
- `honest-but-curious` is your promise that `function` and the map are correct. Do not enable it just to make an error go away.
- Many task specifications forbid `"honest-but-curious"`. Then do not wrap your function: compute row-wise maps or the aggregate in pandas, and feed the result to a built-in measurement over a vector space (e.g. `l1_distance(T=float)`) with `d_in` set to your stated sensitivity; that needs only `"contrib"`.
