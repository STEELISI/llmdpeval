---
id: opendp-quantiles-from-histogram
title: Many quantiles from a DP histogram or b-ary tree (CDF approach)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: quantile
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.then_drop_null
  - opendp.transformations.then_find_bin
  - opendp.transformations.then_index
  - opendp.transformations.then_count_by_categories
  - opendp.transformations.then_b_ary_tree
  - opendp.transformations.choose_branching_factor
  - opendp.transformations.make_consistent_b_ary_tree
  - opendp.transformations.make_cast_default
  - opendp.transformations.make_quantiles_from_counts
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-quantile.ipynb
source_license: MIT
---

## Goal
Release many quantiles at once by releasing a noisy histogram (or a noisy b-ary tree of counts) once, then post-processing it into quantiles. One epsilon covers all quantiles. Neighbours: replace-one (bounded DP, n public), evaluated at symmetric distance `d_in = 2`.

## Steps
1. `enable_features("contrib")`.
2. Public bin edges: `EDGES <- evenly spaced from L to U, NB+1 points`; `BIN_NAMES <- ["0", ..., str(NB-1)]`.
3. Histogram version:
   `make <- lambda scale: space >> then_drop_null() >> then_find_bin(edges=EDGES[1:-1]) >> then_index(BIN_NAMES, "0") >> then_count_by_categories(categories=BIN_NAMES, null_category=False) >> then_laplace(scale) >> make_cast_default(..., TOA=float) >> make_quantiles_from_counts(EDGES, alphas=ALPHAS)`
4. Tree version (usually more accurate): `B <- choose_branching_factor(size_guess=GUESS)` with a public guess; insert `then_b_ary_tree(leaf_count=NB, branching_factor=B)` before `then_laplace`, and replace the cast with `make_consistent_b_ary_tree(branching_factor=B)`.
5. `meas <- binary_search_chain(make, d_in=2, d_out=EPS)`; `dp_quantiles <- meas(VALUES_AS_LIST)`.

Signature for reference: `make_quantiles_from_counts(bin_edges, alphas, interpolation='linear', TA=None, F='float')`.

## Privacy accounting
Replace-one is one removal plus one addition (symmetric distance 2): the changed row leaves one bin and enters another, so the count vector moves by 2 in L1. Only the Laplace step spends budget; `make_consistent_b_ary_tree` and `make_quantiles_from_counts` are post-processing Functions. The b-ary tree transformation raises the sensitivity (each record touches one node per level) and the map accounts for it, so search the scale; never reuse the plain-histogram scale. Assert `epsilon_spent <- meas.map(2) <= EPS`.

## Pitfalls
- `d_in=1` / `map(1)` is add/remove-one (n private); under replace-one it reports half the real ε.
- `then_find_bin` returns 0 below its first edge and len(edges) at or above the last, so passing all NB+1 edges shifts every bin by one (the docs notebook does this). Pass only the interior cut points `EDGES[1:-1]` to get indices 0..NB-1; `make_quantiles_from_counts` still takes all of `EDGES`.
- Edges or the size guess from the private data leak; use public values.
- Too few bins: wide bins and interpolation error. Too many: noise piles up across scanned bins. The tree reduces this trade-off.
- For three or fewer quantiles the exponential-mechanism outline is usually better.
