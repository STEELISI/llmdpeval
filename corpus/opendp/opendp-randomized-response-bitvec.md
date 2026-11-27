---
id: opendp-randomized-response-bitvec
title: Local DP for multi-select answers with bit-vector randomized response
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.measurements.make_randomized_response_bitvec
  - opendp.measurements.debias_randomized_response_bitvec
  - opendp.domains.bitvector_domain
  - opendp.metrics.discrete_distance
source_url: https://github.com/opendp/opendp/blob/v0.16.0/python/src/opendp/measurements.py
source_license: MIT
---

## Goal
Estimate how many records selected each item from a public list (multi-select survey, set membership) under local DP: each record's 0/1 vector is randomized before it leaves the user (RAPPOR-style per-bit flipping), and the analyst debiases the sum.

## Steps
1. `enable_features("contrib")`.
2. Encode each record as bits over the public item list; pack with numpy `packbits(BITS, bitorder="little")` (the debias step expects little-endian bits).
3. `m_rr <- make_randomized_response_bitvec(bitvector_domain(max_weight=M), discrete_distance(), f=F)`
   - `M`: public cap on the number of ones per record; `F` in (0, 1]: per-bit flip probability.
4. On each user's side: `report <- m_rr(PACKED_BYTES)`.
5. Analyst: `estimates <- debias_randomized_response_bitvec(LIST_OF_REPORTS, f=F)`; keep the first `len(items)` entries (padding bits come back too).

Signature for reference: `make_randomized_response_bitvec(input_domain, input_metric, f, constant_time=False)`.

## Privacy accounting
`m_rr.map(1)` is epsilon per record (local DP), computed by the library as `2 * M * ln((2 - F) / F)`. Larger F means more flipping and smaller epsilon. Each user is protected separately; there is no central budget to split.

## Pitfalls
- M must be a public bound on how many items one record can select; records with more ones fall outside the domain.
- Bit order: encode with `bitorder="little"` or the debiased counts land on the wrong items.
- Local DP needs far more records than central DP for the same accuracy.
- For a single yes/no or one-of-k answer, use `opendp-randomized-response`.
