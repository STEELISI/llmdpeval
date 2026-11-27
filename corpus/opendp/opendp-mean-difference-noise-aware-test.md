---
id: opendp-mean-difference-noise-aware-test
title: Difference of two group means with private group sizes and a noise-aware significance test (polars Context)
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: inference
dp_notion: pure
needs_bounds: true
api_calls:
  - opendp.context.Context.compositor
  - opendp.context.unit_of
  - opendp.context.loss_of
  - opendp.extras.polars.Margin
  - opendp.extras.polars.LazyFrameQuery.summarize
  - opendp.extras.polars.LazyFrameQuery.with_keys
  - math.erfc
source_url: https://arxiv.org/abs/1705.09561
source_license: arXiv preprint (Barrientos et al., JCGS 2019; ideas paraphrased)
---

## Goal
Release the mean of a bounded column in two groups, their difference, and a standard error and p-value for the difference that include the DP noise variance. Group sizes are private (only n is public). Replace-one neighbours (bounded DP, n public), set with `unit_of(changes=1)`.

## Steps
1. `enable_features("contrib")`; `context <- Context.compositor(data=LF, privacy_unit=unit_of(changes=1), privacy_loss=loss_of(epsilon=EPS), split_evenly_over=1, margins=[polars.Margin(max_length=N_MAX)])`.
2. `v <- (col(X).cast(float).clip(L, U) − L) / (U − L)` with public bounds, so v is in [0, 1].
3. `q <- context.query().group_by(G).agg(dp.len(signed=True).alias("n"), v.dp.sum((0, 1)).alias("s"), (v*v).dp.sum((0, 1)).alias("q")).with_keys(LazyFrame({G: [A, B]}))`.
4. `scales <- {column: scale}` from `q.summarize()`: OpenDP reports the Laplace scale of every noisy value. Then `rows <- q.release().collect()`.
5. Post-processing per group: floor n at a public minimum; `mean <- clip(s/n, 0, 1)`; sampling variance `(q/n − mean²)/n`; DP noise variance by the delta method `2·(scale_s² + mean²·scale_n²)/n²`.
6. `diff <- mean_A − mean_B`; `SE <- sqrt(sum of both groups' variances)`; `p <- erfc(|diff/SE|/√2)`. Scale means, diff and SE back by (U − L).

## Privacy accounting
Replace-one (bounded DP): a changed row can leave one group and join the other; `unit_of(changes=1)` makes OpenDP charge that. The grouped query is one release: `epsilon_spent <- context.accountant.map(context.d_in)`, assert <= EPS. The test uses only released values and the public scales, so it is free.

## Pitfalls
- `dp.mean` per group divides by a noisy length but hides the scales; use explicit sums and len so the noise variance is known.
- Sampling variance alone in the SE: small groups then give false significance.
- `unit_of(contributions=1)` under replace-one: the printed epsilon is half the real one.
- Group labels from the data instead of the two public labels.
