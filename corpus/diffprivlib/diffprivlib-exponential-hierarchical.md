---
id: diffprivlib-exponential-hierarchical
title: Perturb a hierarchical category (region > subregion) with ExponentialHierarchical
framework: diffprivlib
framework_version: "0.6.6"
mechanism: exponential
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.ExponentialHierarchical
  - diffprivlib.mechanisms.ExponentialCategorical
  - diffprivlib.accountant.BudgetAccountant
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/mechanisms/test_ExponentialHierarchical.py
source_license: MIT
---

## Goal
Randomise each record's leaf category of a public hierarchy (for example area codes grouped into regions) so that substitutes in the same branch are more likely than substitutes in other branches. Local pure epsilon-DP per record. Local DP per record, which implies replace-one (bounded DP, n public) central DP at the same epsilon.

Signature: `ExponentialHierarchical(*, epsilon, hierarchy, random_state=None)`

## Steps
1. Write the public hierarchy as a nested list of strings, all leaves at the same depth: `hierarchy <- [["a1", "a2"], ["b1", "b2", "b3"]]`. A flat list `["a", "b", "c"]` is a one-level hierarchy (plain k-ary randomised response).
2. `mech <- mechanisms.ExponentialHierarchical(epsilon=?, hierarchy=hierarchy)` (keyword-only).
   - The library derives distances from the tree: leaves in the same branch have distance 1, leaves that split higher up have larger distances. These feed an `ExponentialCategorical` mechanism.
3. For each record: `noisy_i <- mech.randomise(leaf_i)` (strings only).
4. Coarse release is post-processing: map noisy leaves to their parent branch if only the branch level is needed.
5. Output the perturbed column and the epsilon per person.

## Privacy accounting
Replace-one (bounded DP, n public): each record is randomised on its own, so replacing one record changes one randomised value; the guarantee is epsilon under replace-one and under add/remove. One randomise call per person gives epsilon-DP for that person (local DP). Mechanisms do not touch accountants; record `acc.spend(epsilon, 0)` once.

## Pitfalls
- Leaves at different depths raise a ValueError; pad shallow branches so that all leaves sit at the same depth.
- Values not in the hierarchy raise a ValueError; map unknowns to a public "other" leaf first.
- Integer codes must be strings (or wrapped with `IntToString`).
- The hierarchy must be public (from a codebook), not built from the categories present in the data.
