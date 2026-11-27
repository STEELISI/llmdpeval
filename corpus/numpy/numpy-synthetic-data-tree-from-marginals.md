---
id: numpy-synthetic-data-tree-from-marginals
title: Synthetic rows from an independent or tree model fitted only to noisy marginals
framework: numpy
framework_version: "any"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.laplace
  - numpy.random.Generator.choice
  - numpy.ravel_multi_index
  - numpy.bincount
source_url: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
source_license: original code (algorithm from cited paper)
---

## Goal
Make a synthetic table of categorical columns: release noisy marginals once under epsilon (replace-one, bounded DP, n public), fit a simple model to the noisy tables only, and sample any number of rows. Every later analysis on the synthetic rows is post-processing. The Chow-Liu tree (maximum mutual-information spanning tree) keeps each chosen pair's dependence.

## Steps
1. Public inputs: columns, a category list per column, the model structure:
   - independent: no edges, release one one-way table per column;
   - public tree: `[(parent, child), ...]` from domain knowledge, release one table per edge plus one-way tables for roots;
   - learned tree: release ALL pairs plus the root, choose the tree afterwards (step 4).
2. `scale <- 2*T / EPS` for T tables; `noisy[t] <- exact[t] + laplace(0, scale, shape)`.
3. Post-processing: clip each table at 0, add a tiny floor, normalise to probabilities.
4. Learned tree: mutual information of each noisy pair table; Prim's maximum spanning tree from the root, keeping the order edges were added (parent first).
5. Sample roots from their noisy one-way tables (`rng.choice(K, size, p=...)`).
6. For each edge in order: conditional table `P(child | parent) <- joint / row sums`; draw the child by inverse CDF on the parent's sampled row.
7. Map indices back to the public category labels.
8. Utility (an evaluation step outside the DP release): compute every one-way and two-way marginal of the synthetic rows and of the real rows, report total variation distance per table (mean and worst), and compare against an independent model.

## Privacy accounting
Replace-one moves two cells in each of the T released tables: L1 sensitivity 2T, `epsilon_spent <- 2*T / scale`, assert it is at most EPS. Structure learning in step 4 reads only noisy tables, so it is free; the price is that all pairs share the budget. A public tree releases fewer tables, so each gets more budget.

## Pitfalls
- Choosing the tree from exact mutual information: that is an unpaid release.
- Sampling a child before its parent; give edges in sampling order.
- Pairs that are not tree edges are only kept through the chain; check their errors in step 8.
- The utility step uses the real data, so its numbers are not private; print them as a check, never as a release.
- MST, AIM, PrivBayes and private-pgm (graphical models over many marginals, adaptive selection) need other libraries; they are not in NumPy, OpenDP or diffprivlib.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
