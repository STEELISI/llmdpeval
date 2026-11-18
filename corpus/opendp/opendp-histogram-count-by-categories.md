---
id: opendp-histogram-count-by-categories
title: Histogram over a fixed public category list (count_by_categories + Laplace)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: histogram
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.make_count_by_categories
  - opendp.transformations.then_count_by_categories
  - opendp.transformations.then_find_bin
  - opendp.transformations.then_index
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_chain
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/transformations/aggregation-quantile.ipynb
source_license: MIT
---

## Goal
Release one noisy count per category of a categorical column, with the category list fixed in advance from public knowledge. Neighbouring relation: replace-one (bounded DP, n public), the usual setting when the row count is known.

## Steps
1. `enable_features("contrib")`.
2. `CATS`: the public list of categories, in a fixed order (e.g. sorted). Do not build it from the data's unique values.
3. `space <- vector_domain(atom_domain(T=str)), symmetric_distance()`
4. `make <- lambda scale: space >> then_count_by_categories(categories=CATS, null_category=?) >> then_laplace(scale)`
   - `null_category=True` appends one extra count for values not in CATS; with False they are dropped.
5. `meas <- binary_search_chain(make, d_in=2, d_out=EPS)`: under symmetric distance one changed row is one removal plus one addition, so replace-one is `D_IN = 2`.
6. `noisy_counts <- meas(VALUES_AS_LIST)`; position i belongs to CATS[i] (plus the trailing "other" count if enabled).

Signature for reference: `make_count_by_categories(input_domain, input_metric, categories, null_category=True, MO='L1Distance<int>', TOA='int')`.

## Privacy accounting
Replace-one (bounded DP, n public): one changed row leaves one cell and enters another, so the count vector moves by 2 in L1. The transformation's map gives that from `D_IN = 2`; the Laplace step uses L1 sensitivity, so the whole histogram costs one epsilon, not one per category. Assert `epsilon_spent <- meas.map(2) <= EPS`. Only if the task says add/remove-one (n private) use `D_IN = 1`.

## Pitfalls
- `d_in=1` / `map(1)` under replace-one: the printed epsilon is half the real one.
- Category list from `df[col].unique()` leaks which rare values exist; use a public list.
- Spending EPS / number_of_categories on each cell is wasteful: the histogram is one release.
- Sorting the noisy counts, taking ratios or top-k from them is post-processing and free, but only if it uses the noisy counts alone.
- For a histogram of a numeric column, bin first with `then_find_bin(edges=?)` and `then_index(...)`, using public edges.
