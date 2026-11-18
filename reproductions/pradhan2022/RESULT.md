# Gopher (Pradhan et al., SIGMOD 2022): reproduction result

Run 2026-11-07 on a SPHERE XDC (2 cores, 2 GB), authors' commit e6e7eb3, notebooks unchanged, Python 3.9.6 + their
requirements.txt pins (torch 1.9.0 CPU build). `python compare.py --rtol 1e-5` diffs every cell against the
outputs the authors saved in their notebooks (German credit data, as shipped).

**Verdict: final result MATCHES.**
- Classifier and fairness numbers (statistical parity, TPR parity, predictive parity, loss, accuracy), both notebooks:
  match to ~1e-7.
- Final top-5 removal explanations (removal_expl.ipynb, last table): same 5 predicate sets, same support, score,
  gt-score, influence and new accuracy (max relative difference 1.5e-6).
- Intermediate only: candidate counts differ by 0.2-0.7% (3-predicate 2422 vs 2418, 4-predicate 10854 vs 10779).
  These are patterns at the support/score cut-offs that flip on last-digit float differences (authors: macOS;
  here: Linux, CPU torch). They do not change the top-5.

Not run: the SQF dataset (the notebook ships with `dataset = 'german'`; SQF needs that one value changed,
a parameter change, not a code change). License: the repo has no license file, so
the code and data stay out of git (fetch.sh pins the commit).
