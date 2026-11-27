---
id: opendp-clamp-public-bounds
title: Choosing bounds from public knowledge and clamping inside the pipeline
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.transformations.make_clamp
  - opendp.transformations.then_clamp
  - opendp.transformations.then_impute_constant
  - opendp.domains.atom_domain
  - opendp.mod.Transformation.map
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/tabular-data/essential-statistics.rst
source_license: MIT
---

## Goal
Limit how much one record can move a sum, mean or variance, using bounds that do not depend on the private data, and let OpenDP know the bounds so its maps are right. Neighbouring relation: replace-one (n public): use a sized domain and `D_IN = 2`.

## Steps
1. Get (L, U) from public sources, in this order of preference:
   - domain knowledge (physical or legal limits, survey codebooks, topcodes, units);
   - a comparable public dataset;
   - a separate DP release of coarse quantiles (spends budget; see the quantile outlines).
   The docs' rule: the only bad option is to look at the data.
2. Inside the pipeline, impute missing values first with a public constant C in [L, U], then clamp:
   `pre <- SPACE >> then_impute_constant(C) >> then_clamp((L, U))`
3. If the bounds are already guaranteed by the data's definition, you may declare them in the domain instead: `atom_domain(bounds=(L, U))`. Values outside them make the result undefined, so only do this when it is truly guaranteed.
4. Continue with `then_sum`, `then_mean` or `then_variance`, and check `pre_and_aggregate.map(2)` to see the sensitivity the bounds imply (U - L for a sum on a sized domain `vector_domain(atom_domain(T=float, nan=True), size=N)`).

Signature for reference: `make_clamp(input_domain, input_metric, bounds)`.

## Privacy accounting
Sensitivity comes from the bounds through the transformation's map. Narrower public bounds mean less noise. Under replace-one, verify with `.map(2)`; do not plug the width into a formula yourself.

## Pitfalls
- `min()`/`max()`/percentiles of the private column as bounds: a privacy violation even if the rest is correct. Bounds given in a task prompt that equal the exact data extremes have the same problem; say so if you notice it.
- Clamping outside OpenDP (numpy clip) does not inform the domain; the map then cannot see the bounds.
- On an unsized domain OpenDP charges a clamped sum max(|L|, |U|, U - L) per added or removed row, so `map(2)` doubles it; with n public declare the size and the replace-one charge is U - L.
- Alternative, add/remove-one (n private): unsized domain, `D_IN = 1`, the clamped sum is charged `map(1)` = max(|L|, |U|, U - L).
- Too narrow bounds bias the result; the accuracy estimate covers only noise.
- The imputed constant must be public and inside [L, U]; the mean of the private column is not public.
