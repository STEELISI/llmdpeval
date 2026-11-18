---
id: opendp-confusion-matrix-error-rates
title: Confusion matrix of one or more classifiers from one OpenDP count-by-categories each, with overall, false-negative and false-positive rates derived
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: classification
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation.rst
source_license: MIT
---

## Goal
Release how often each true label is predicted as each label, for one or several classifiers evaluated on the same private rows, and derive the error rates (overall, per class false-negative and false-positive) from the noisy matrices. Labels may include "none" (no true label) and "no class" / "unclassified" predictions. Replace-one row (bounded DP, n public; class sizes private).

## Steps
1. Public: true-label list and predicted-label list (with the "none" / "no class" entries), classes to report, n. `CELLS <- ["t|p" for t in TRUE for p in PRED]` (row-major).
2. Per row (not a release): true label and each classifier's prediction, as the original code defines them; one token `"t|p"` per row and classifier.
3. `make(s) <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=CELLS, null_category=False) >> then_laplace(s)`; `scale <- binary_search_param(make, d_in=2, d_out=EPS/m)` for m classifiers; `meas <- make(scale)`.
4. Per classifier: `noisy <- meas(TOKENS)` reshaped to TRUE x PRED. One measurement per classifier, not one per rate.
5. Post-processing, clipped to [0, 1]: overall error `1 - (noisy diagonal of reported classes) / n`; FN of c `1 - M[c,c] / row_total(c)`; FP of c `(column_total(c) - M[c,c]) / (total of the other rows)`; all totals are sums of noisy cells.

## Privacy accounting
Each row is one token; replace-one removes one token and adds one: symmetric distance 2. `eps_spent <- sum over classifiers of meas.map(2)`, assert <= EPS (sequential composition: the same rows feed every matrix).

## Pitfalls
- `d_in=1`: add/remove-one; under replace-one the matrix moves by 2.
- A polars Context query or a `then_sum` per numerator and denominator: many small budgets, much larger error, slow on millions of rows.
- polars Context on Int8/Int16 columns fails ("No match for concrete type i8"); for this pattern build the string tokens in pandas instead.
- `then_laplace` directly on precomputed counts in a `symmetric_distance` space fails; count with `then_count_by_categories`.
- Exact class sizes as denominators.
