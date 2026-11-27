---
id: opendp-two-group-item-tables-noise-aware
title: Many 2x2 tables (group a vs b x item listed or not) from one contribution-bounded OpenDP histogram, with a noise-aware test
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.domains.vector_domain
  - opendp.domains.atom_domain
  - opendp.metrics.symmetric_distance
  - opendp.transformations.then_count_by_categories
  - opendp.transformations.then_sum
  - opendp.measurements.then_laplace
  - opendp.mod.binary_search_param
  - math.erfc
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/identifier-truncation.rst
source_license: MIT
---

## Goal
Compare, for every item in a public list (thousands possible), how often units of group a and of group b list it: one 2x2 table per item (A = units listing it, B = group size minus A), a ratio, and a significance verdict that counts the DP noise. One row = one unit; its items sit in one delimited column. Replace-one unit (bounded DP: n public; group sizes and all counts private).

## Steps
1. Public: item list, group labels, n, cap K from a rule fixed in advance, test level, "enough data to test" rule.
2. Per unit: split the column, keep distinct public items, at most K chosen at random within the unit. Emit one token `"g|item"` per kept item (g = 0 for a, 1 for b).
3. Group size: `make_g(s) <- (vector_domain(atom_domain(bounds=(0, 1)), size=N), symmetric_distance()) >> then_sum() >> then_laplace(s)` on the 0/1 "in group b" list; `s_g <- binary_search_param(make_g, d_in=2, d_out=eps_g)`; `n_a <- N - n_b`.
4. Histogram: `make_h(s) <- (vector_domain(atom_domain(T=str)), symmetric_distance()) >> then_count_by_categories(categories=CELLS, null_category=False) >> then_laplace(s)` with CELLS = every public (group, item) token; `s_h <- binary_search_param(make_h, d_in=2K, d_out=eps_h)`; input = the token list. ONE measurement for all items.
5. Keep `s_g`, `s_h`: the noise variance (about 2·s²) feeds the test.
6. Post-processing: clamp at 0; `T <- A_a + A_b`, `pi <- n_b/N`, `D <- A_b - pi*T`; variance = `T*pi*(1-pi)*(N-T)/(N-1)` + `((1-pi)^2 + pi^2)*v + (T/N)^2*v_n`; `p <- erfc(|D|/sqrt(var)/sqrt 2)`; test items whose noisy T passes the public rule; Bonferroni over them; ratio with 0.5 correction.

## Privacy accounting
`eps_spent <- make_g(s_g).map(2) + make_h(s_h).map(2K)`, assert <= EPS. After capping, replace-one removes up to K tokens and adds up to K: symmetric distance 2K; OpenDP turns it into the histogram sensitivity. Tests and ratios are post-processing.

## Pitfalls
- `d_in=1` or `d_in=K`: under replace-one the token list moves by 2K.
- A canonical-noise test, `then_noisy_max` or a polars query per item, each with a share of EPS: thousands of tiny budgets (and timeouts). Release once, test after.
- `then_laplace` straight on a vector of counts in a `symmetric_distance` space fails ("No match for concrete type SymmetricDistance"): count with `then_count_by_categories` first.
- Fisher exact on noisy counts: negative cells raise; naive tests give many false positives.
- `make_count_by_categories` takes `input_domain`, `input_metric`, `categories` (no `domain=`); metrics cannot be multiplied by a number.
- `then_count_by_categories` adds a null bin by default (one extra entry): pass `null_category=False` or drop the last entry.
- Two per-group histograms at `d_in=K` instead of one: valid only with EQUAL shares (cost = 2 x the larger share); one joint histogram at 2K avoids the trap.
- A polars Context on a list column (`List(String)`) fails; build the tokens in pandas.
- Noise on a constant (`meas(["1"])`) instead of the data is not DP of anything.
