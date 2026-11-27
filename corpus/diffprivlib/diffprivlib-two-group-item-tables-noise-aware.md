---
id: diffprivlib-two-group-item-tables-noise-aware
title: Many 2x2 tables (group a vs b x item listed or not) from one contribution-bounded diffprivlib release, with a noise-aware test
framework: diffprivlib
framework_version: "0.6.6"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.Laplace
  - diffprivlib.tools.count_nonzero
  - diffprivlib.accountant.BudgetAccountant
  - math.erfc
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/mechanisms/laplace.py
source_license: MIT
---

## Goal
Compare, for every item in a public list (thousands possible), how often units of group a and of group b list it: one 2x2 table per item (A = units listing it, B = group size minus A), a ratio, and a significance verdict that counts the DP noise. One row = one unit; its items sit in one delimited column. Replace-one unit (bounded DP: n public; group sizes and all counts private).

## Steps
1. Public: item list, group labels, n, cap K from a rule fixed in advance, test level, "enough data to test" rule.
2. Per unit: split the column, keep distinct public items, at most K chosen at random within the unit; fill a 2 x items count matrix.
3. `acc <- BudgetAccountant()`. Group size: `n_b <- count_nonzero(IN_GROUP_B, epsilon=eps_g, accountant=acc)`; with n public one replaced unit moves n_b by at most 1, so the tool's sensitivity 1 fits; `n_a <- N - n_b`.
4. Count matrix as ONE vector release: `mech <- Laplace(epsilon=eps_h, sensitivity=2K)`; `noisy <- mech.randomise(cell)` for every cell; then `acc.spend(eps_h, 0)` once. Noise variance `v <- mech.variance(0)`.
5. Post-processing: clamp at 0; `T <- A_a + A_b`, `pi <- n_b/N`, `D <- A_b - pi*T`; variance = `T*pi*(1-pi)*(N-T)/(N-1)` + `((1-pi)^2 + pi^2)*v + (T/N)^2*v_n`; `p <- erfc(|D|/sqrt(var)/sqrt 2)`; test items whose noisy T passes the public rule; Bonferroni over them; ratio with 0.5 correction.

## Privacy accounting
`eps_spent <- acc.total()[0]` = eps_g + eps_h, assert <= EPS. Laplace with sensitivity 2K on every cell of the matrix is one ε-DP release because a capped unit changes the matrix by at most 2K in L1 under replace-one (K cells out, up to K in, maybe in the other group).

## Pitfalls
- `tools.histogram` on the item codes assumes sensitivity 1 per element added or removed; with K items per unit it under-charges (K, and 2K under replace-one). Either divide its epsilon by 2K and report 2K times the accountant, or use the mechanism as above.
- `count_nonzero` returns one number, not a tuple.
- One `count_nonzero` per item with EPS/items each: tiny budgets, huge noise, and very slow.
- Fisher exact or a textbook test on noisy counts: false positives, and errors on negative cells.
- A separate accountant per query hides the total; use one.
- An exponential mechanism or noise with sensitivity 1 on p-values or ratios: wrong sensitivity, and not needed; tests and ratios from the noisy counts are post-processing.
- Passing EPS/K to a sensitivity-1 tool makes the accountant print EPS/K: print the true loss (K x, or 2K x under replace-one), not the accountant's number.
- Counts only for items seen in the data: reindex to the full public list with zeros.
- Add/remove alternative (n private): a row is added to or removed from its cells instead of moving between them, so the L1 sensitivity is half the replace-one value used here; anything divided by n then needs a noisy n.
