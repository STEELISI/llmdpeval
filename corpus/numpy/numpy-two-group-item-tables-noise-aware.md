---
id: numpy-two-group-item-tables-noise-aware
title: Many 2x2 tables (group a vs b x item listed or not) from one contribution-bounded histogram, with a noise-aware test
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.random.Generator.permutation
  - scipy.stats.norm.sf
  - scipy.stats.norm.cdf
source_url: https://doi.org/10.1561/0400000042
source_license: Dwork & Roth 2014 (ideas paraphrased); score test = Pearson chi-square for a 2x2 table (textbook)
---

## Goal
Compare, for every item in a public list (thousands possible), how often units of group a and of group b list it: one 2x2 table per item (A = units listing it, B = group size minus A), a ratio per item, and a significance verdict that counts the DP noise. Each row is one unit whose items sit in one delimited column. Replace-one unit (bounded DP: total n public; group sizes and all counts private).

## Steps
1. Public inputs: item list (codebook), group labels (a, b), n, cap K from a rule fixed in advance (design limit or prior public data), test level, a public "enough data to test" rule.
2. Per unit: split the column, keep distinct public items, keep at most K chosen at random within that unit.
3. Keep only rows whose group is a or b (others are dropped, never counted as a). Release 1 (small share of EPS): both sizes as one vector, `(n_a, n_b) <- counts + rng.laplace(2/eps_g, 2)` (replace-one can move a unit from a to b: L1 2).
4. Release 2: one count matrix over all (group, item) cells, `noisy <- counts + laplace(2K/eps_h)` per cell. Every item is in this ONE vector; never loop over items with a share of EPS each.
5. Post-processing (free): clamp cells at 0; per item total `T <- A_a + A_b`, `n <- n_a + n_b` (noisy), `pi <- n_b/n`, `D <- A_b - pi*T`; normal part `sd^2 <- T*pi*(1-pi)*(n-T)/(n-1) + ((1-pi)^2 + pi^2)*(T/n)^2*v_n` (sampling plus the group-size noise, v_n = 2*scale_g^2); the cell noise of D is `e = (1-pi)*L_b - pi*L_a` with Laplace L at the cell scale. Laplace tails are heavier than normal, so a plain z-test over-rejects at small Bonferroni levels: draw e a few thousand times and take `p <- mean(norm.sf((|D| - e)/sd) + norm.cdf((-|D| - e)/sd))`.
6. Test only items whose NOISY total passes the public rule (e.g. >= 5 noise scales); Bonferroni over that number.
7. Ratio with a 0.5 correction on each cell. Free, no extra budget.

## Privacy accounting
Sequential composition of two Laplace releases: `eps_spent <- 2/scale_g + 2K/scale_h`, assert <= EPS. Replace-one unit: a capped unit leaves up to K cells and its replacement enters up to K cells (possibly in the other group), so L1 = 2K. Tests and ratios are post-processing of the two releases.

## Pitfalls
- Sensitivity 1 per cell ("each count moves by at most 1"): the vector moves by up to 2K in L1.
- Fisher exact or a textbook chi-square on noisy counts: far too many false positives when noise is large next to the counts; Fisher also fails on negative cells.
- Wald test on log ratio: zero cells break it; the score test above does not need the ratio.
- Item domain from the values seen in the data, or Bonferroni over items picked with exact counts.
- A separate exponential mechanism or noise on p-values/ratios with sensitivity 1: their sensitivity is not 1, and tests and ratios computed from the released histogram cost nothing more.
- Counts only for items seen in the data (`value_counts`, an inner join, NaN for absent items): reindex to the full public list with zeros, or the release shows which items occur.
- Two per-group histograms instead of one matrix: each at sensitivity K is valid only with EQUAL budget shares (a unit can sit in either group, so the cost is 2 x the larger share).
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
