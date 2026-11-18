---
id: diffprivlib-reproducibility-versions
title: Reproducible runs with random_state and the version pins diffprivlib 0.6.6 needs
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.utils.check_random_state
  - diffprivlib.utils.PrivacyLeakWarning
  - diffprivlib.tools.mean
  - diffprivlib.models.LogisticRegression
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/diffprivlib/utils.py
source_license: MIT
---

## Goal
Make a diffprivlib script run on a known environment, give repeatable noise when the task asks for it, and surface privacy warnings instead of hiding them. Neighbouring relation: replace-one (bounded DP, n public).

Signature: `check_random_state(seed, secure=False)`

## Steps
1. Environment: diffprivlib 0.6.6 with scikit-learn below 1.8 and scipy below 1.18 (0.6.6 also requires numpy 2.x). Report `diffprivlib.__version__` in the output.
2. For tests only: `random_state` makes the noise repeatable, which defeats DP. Never set it in a real release; leave it out (the default draws fresh OS randomness).
   - an int gives the same noise on every run of that call;
   - one shared object `RNG <- check_random_state(SEED)` passed to several calls gives a fixed but different draw for each.
3. Leave `random_state=None` when the task asks for a real release; fixed noise is for testing only.
4. Make silent leaks fail loudly: set the Python warnings filter to "error" for `diffprivlib.utils.PrivacyLeakWarning`.
5. Output the DP results, the true results and the accountant total so a reader can check them.

## Privacy accounting
Replace-one (bounded DP, n public) is the default: full-column tools with public n (`mean`, `sum`, `var`, `quantile`, `count_nonzero`) are calibrated for one replaced row, so their recorded epsilon is the replace-one epsilon. Histograms and per-partition models are calibrated for add/remove: under replace-one pass epsilon/2 and report 2x; under add/remove (n private) pass the full epsilon. Seeding does not change the privacy cost; the accountant records the same spend. Publishing outputs for several seeds of the same query is several releases and costs epsilon each time.

## Pitfalls
- With scikit-learn >= 1.8, `LogisticRegression` fails because 0.6.6 passes `multi_class='ovr'`; with scipy >= 1.18 it fails on `fmin_l_bfgs_b(iprint=...)`.
- Re-running with new seeds until the output "looks right" is cherry-picking and breaks the guarantee.
- Using the same int seed inside a loop gives identical noise in every iteration.
- `PrivacyLeakWarning` means bounds, ranges, norms or classes were read from the private data; fix the arguments rather than silencing the warning.
