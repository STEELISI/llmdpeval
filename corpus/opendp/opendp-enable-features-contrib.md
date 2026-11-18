---
id: opendp-enable-features-contrib
title: Opting in to contrib (and other) features before building OpenDP pipelines
framework: opendp
framework_version: "0.16.0"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.mod.enable_features
  - opendp.mod.disable_features
  - opendp.mod.assert_features
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/getting-started/quickstart.rst
source_license: MIT
---

## Goal
Make OpenDP constructors usable. Most constructors in OpenDP 0.16.0 have not finished vetting and are marked "contrib"; they raise an exception until you opt in. The skeleton's example release uses replace-one (n public): a sized 0/1 sum read at `d_in=2`.

## Steps
1. Use the prelude namespace (conventionally `dp`), so constructors are reachable as `dp.t.*` (transformations), `dp.m.*` (measurements), `dp.c.*` (combinators) and `dp.Context`.
2. Before any constructor call: `enable_features("contrib")`.
3. Add a further flag only when a constructor's error message names it:
   - `"honest-but-curious"`: user-defined pieces (`make_user_transformation`, `make_user_measurement`, chaining a plain Python function after a measurement) and the sklearn-style `PCA` / `LinearRegression` wrappers.
   - `"idealized-numerics"`: the `PCA` wrapper and some numpy helpers; you accept that float arithmetic is treated as exact.
4. Flags are global for the process; `disable_features(...)` turns them off.

## Privacy accounting
Flags never change a privacy map. They only state what you trust. Epsilon still comes from `meas.map(D_IN)`: `map(2)` under replace-one (the default); alternative, add/remove-one (n private): unsized domain and `map(1)`.

## Pitfalls
- A missing flag raises an exception naming the flag; it does not silently give a wrong result. Add exactly the flag named.
- Do not enable `"honest-but-curious"` by reflex. It is your promise that your own Python functions and stability maps are correct; OpenDP cannot check them. Use it only where required and say why in a comment.
- Flag names are not validated: a misspelled or invented flag is silently accepted and enables nothing, so the later constructor still fails. The old name "floating-point" still works but is deprecated; use "idealized-numerics" (checked on 0.16.0).
- Optional packages fail at call time, not import time: polars (Context over LazyFrames), pyarrow (polars `summarize`), scikit-learn and randomgen (sklearn wrappers).
- If the task forbids `"honest-but-curious"`, every user-transformation, sklearn-wrapper (PCA, Theil-Sen) and user-measurement pattern is out. Contrib-only routes: public row-wise maps in pandas before the chain, built-in transformations, or a built-in measurement on a vector you aggregated yourself with a stated `d_in`.
