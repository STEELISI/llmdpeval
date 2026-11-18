---
id: opendp-randomized-response
title: Local DP with randomized response (per-record release)
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: count
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.measurements.make_randomized_response_bool
  - opendp.measurements.make_randomized_response
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/randomized-response.rst
source_license: MIT
---

## Goal
Protect each individual's answer before it leaves them (local DP), then estimate a proportion from the noisy answers. Use this only when the task asks for local DP; central DP (count + Laplace) is far more accurate.

## Steps
1. `enable_features("contrib")`.
2. Boolean answer: `rr <- make_randomized_response_bool(prob=P)`; P in (0.5, 1) is the chance of reporting the truth.
3. Categorical answer: `rr <- make_randomized_response(categories=CATS, prob=P)` with a public category list.
4. Check the per-person cost: `rr.map(1)` returns epsilon for one person's answer.
5. Apply per record: `noisy <- [rr(v) for v in VALUES]`.
6. De-bias the aggregate as post-processing: for the boolean case, with observed share S of True, the estimated true share is (S - (1 - P)) / (2P - 1).

Signature for reference: `make_randomized_response_bool(prob, constant_time=False)`.

## Privacy accounting
The input metric is `discrete_distance` on one answer: the map gives the epsilon of one person's report. Each person is protected by that epsilon on their own, so there is no composition across people. Choose P by searching `binary_search_param(lambda p: make_randomized_response_bool(prob=p), d_in=1, d_out=EPS)` instead of solving a formula.

## Pitfalls
- Reporting the de-biased share without noting that its variance grows sharply as P nears 0.5.
- Local and central DP are different promises. Do not report a randomized-response epsilon as if it were a central count's epsilon, or the reverse.
- Category lists from the data's unique values leak rare values; use a public list.
- The de-biasing formula above holds only for the boolean mechanism.
