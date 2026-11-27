---
id: diffprivlib-exponential-categorical-local
title: Perturb each record's category with ExponentialCategorical and a distance list
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.ExponentialCategorical
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/mechanisms/test_ExponentialCategorical.py
source_license: MIT
---

## Goal
Replace each record's categorical value with a randomised one (local DP), where "nearby" categories are more likely substitutes than distant ones. Useful for releasing a perturbed microdata column. Local DP per record, which implies replace-one (bounded DP, n public) central DP at the same epsilon.

Signature: `ExponentialCategorical(*, epsilon, utility_list, random_state=None)`

## Steps
1. Write a public list of string categories and a public DISTANCE for every unordered pair:
   `utility_list <- [("a", "b", D_ab), ("a", "c", D_ac), ("b", "c", D_bc), ...]`, each distance > 0.
   - Despite the name, larger values make a substitution LESS likely: output y for input x has probability proportional to `exp(-eps * d(x, y) / (2 * max_d))` (the library normalises by the largest distance as sensitivity; the factor 2 is dropped when every category has the same normalising sum, e.g. equal distances), and y = x has distance 0.
2. `mech <- mechanisms.ExponentialCategorical(epsilon=?, utility_list=utility_list)` (keyword-only).
3. For each record: `noisy_i <- mech.randomise(value_i)`; values must be strings in the list.
4. Optional: `mech.randomise` many times is NOT done per person; one call per record.
5. Output the perturbed column, the epsilon per person, and a tally of true vs noisy categories for checking.

## Privacy accounting
Replace-one (bounded DP, n public): each record is randomised on its own, so replacing one record changes one randomised value; the guarantee is epsilon under replace-one and under add/remove. Each person's value is randomised once, giving epsilon-DP per person (local DP; parallel over records). Record it once with `acc.spend(epsilon, 0)`; the mechanism does not touch accountants.

## Pitfalls
- Every pair must be listed, or construction raises a ValueError about missing utilities.
- Integer category codes must be converted to strings (or wrapped with `IntToString`).
- This is NOT the mechanism for choosing the best item from scores; that is `mechanisms.Exponential(utility=[...], candidates=[...])`, where larger utility means MORE likely.
- Equal distances everywhere reduce to k-ary randomised response.
