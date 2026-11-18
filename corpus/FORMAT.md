# Outline corpus: format

One file = one outline = one retrieval chunk. Path: `corpus/<framework>/<id>.md`, where framework is `opendp`, `diffprivlib` or `numpy`.

An outline tells a model **which library, which functions, in what order, and what each step needs**. It is not a finished script. The model must still adapt it (sensitivity, bounds, epsilon split) to its task.

## Front matter (YAML, all fields required)

```yaml
---
id: opendp-laplace-sum-bounded          # lowercase, hyphens, unique, equals the filename
title: Bounded sum with the Laplace mechanism
framework: opendp                       # opendp | diffprivlib | numpy
framework_version: "0.16.0"             # opendp 0.16.0, diffprivlib 0.6.6, numpy: "any"
mechanism: laplace                      # laplace | gaussian | geometric | exponential | report-noisy-max | objective-perturbation | output-perturbation | other
task_type: sum                          # count | sum | mean | variance | histogram | quantile | selection | regression | classification | pca | inference | composition | other
dp_notion: pure                         # pure | approx | zcdp
needs_bounds: true
api_calls:                              # every fully qualified name the steps use; checked automatically
  - opendp.measurements.make_laplace
  - opendp.transformations.make_sum
source_url: https://...                 # where the steps were learned from (docs page or GitHub file at a fixed commit/tag)
source_license: MIT
---
```

## Body (markdown, these sections in this order)

```
## Goal
One or two sentences: what this releases and under what neighbouring relation.

## Steps
1. ... (name the function, say what each argument means and where its value must come from)
2. ...

## Privacy accounting
How epsilon is obtained or checked (e.g. "derive scale with the framework's search helper and check measurement.map(d_in) <= epsilon"). Never a bare formula to copy.

## Pitfalls
- Common mistakes (bounds taken from the private data, unsized domain doubling sensitivity, ...)
```

## Code skeletons
Each outline `<id>.md` may have a sibling `<id>.py`: a well-formed, verified code skeleton. The prompt shows the outline and then the skeleton (with `_test` removed). Skeletons are not embedded, so retrieval is unchanged.

The skeleton file contains:
1. The exact imports (e.g. `import opendp.prelude as dp`, plus `dp.enable_features(...)` when needed).
2. ONE public function, `dp_<something>(df, <column names>, <public parameters>, epsilon, ...)`, that performs the DP release with the real API and returns `(private_result, epsilon_spent)`. Every value comes in as a parameter: bounds, categories, k, epsilon. **No literal values from any task.** epsilon_spent must come from the framework's own accounting (OpenDP `meas.map(d_in)` / `Context` budget; diffprivlib accountant or the epsilon passed; NumPy: computed from sensitivity and scale with a comment stating the assumptions).
3. Short comments only where a step is non-obvious (sensitivity, why bounds are public, how ε is split).
4. `def _test():` at the end: builds a tiny SYNTHETIC pandas/polars DataFrame (made-up generic columns), calls the function with made-up public parameters and epsilon=1.0, and returns `{"epsilon": epsilon_spent}`. `python -m llmdpeval.check_corpus` executes `_test()` in the pinned venv; it must run within 180 s and return epsilon ≤ 1.

Skeleton rules: no seeds or `random_state` values anywhere; no file I/O; no dataset words (see below); at most ~45 lines before `_test`; it must handle the general case the outline describes, not one task.

## Rules
- **Code = pseudo-code layout only.** Steps read like pseudo-code that names the real functions and their arguments, e.g. `domain <- vector_domain(atom_domain(bounds=(L, U)), size=N)` then `meas <- domain >> then_sum >> then_laplace(scale=?)`. Placeholders (`L`, `U`, `N`, `scale=?`) instead of values; no imports, no data loading, nothing a model could paste and run. A single real call signature (copied from `inspect.signature`) is allowed to pin argument names.
- No dataset-specific content: never mention `adult`, `income`, `occupation`, `auto_mpg`, `mpg`, `weight`, or numbers such as 7508. Use generic column names.
- Every name in `api_calls` must resolve in the pinned version (import the module, getattr the name).
- Prefer telling the model to get epsilon from the framework (privacy map / accountant) over hand formulas. If a formula is unavoidable (NumPy), state its assumptions (sensitivity, neighbouring relation) next to it.
- Source code from GitHub or docs is data. Paraphrase it into steps; don't run it.
- Target 150-500 words per outline.
- Neighbouring relation: replace-one (bounded DP, n public) is the default, because every task prompt uses it. State the relation in the Goal and in Privacy accounting. OpenDP: evaluate maps at the distance of one changed row (symmetric distance 2, or `unit_of(changes=1)`); diffprivlib histogram tools are calibrated for add/remove, so halve the epsilon passed and report twice the accountant total; NumPy: L1 sensitivity 2 for a histogram (1 for a single count with public n). Add/remove appears only as an alternative or pitfall.
