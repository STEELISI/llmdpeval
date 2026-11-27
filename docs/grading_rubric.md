# Grading rubric

Every runnable script is read by hand. Each grade gets a one-line reason in `hand_grades.csv` (columns: cell,
task, run, grade, reason). The tools in `llmdpeval.grade` and `llmdpeval.app_grade` only flag and suggest; they do
not assign grades.

## Marks
- **G**: runs, the true answer is correct, valid DP at a total ε ≤ 1, the requested framework is used, and the answer is useful (within the U thresholds).
- **P** (partial): valid DP and a correct, useful answer, but the ε the script *reports* is wrong or missing (e.g. it reads a fresh accountant and prints 0, or crashes while printing ε after the answer). Reported separately from G.
- **U**: like G, but too noisy to be useful. Thresholds:
  - Count: |dp − 7508| > 3.
  - Top-K: fewer than 3 of the true top 5.
  - LinReg: mean absolute error > 10% of the true predictions.
- **L**: runs and is correct, but the DP part does not use the requested framework (e.g. NumPy noise when OpenDP was asked).
- **W**: runs, but the answer or the DP is wrong. Examples:
  - wrong true answer;
  - ε spent > 1;
  - wrong sensitivity;
  - bounds taken from the data;
  - no private answer printed (e.g. a lazy frame object, or a crash caught inside try);
  - a fixed seed that makes the DP output deterministic;
  - "simulated" or mock DP;
  - extra exact values printed next to the private answer (e.g. the exact scores of the private picks). Printing the true answer is required by the prompt and is fine (rule 20).
- **E**: execution error. **N**: no code (timeout / no answer).

## ε is read from the framework, not from the comments
- **OpenDP:** use `measurement.map(d_in)` where the code builds a measurement. `dp.Context(privacy_loss=loss_of(epsilon=e), split_evenly_over=s)` enforces the total budget e; the privacy unit must match one person, so `contributions=1` (larger is valid, just noisier). `make_noisy_top_k(k, scale)`: the map grows linearly with k. With k = 5 and scale 2 (sensitivity 1, monotonic), the map gives ε = 5, so W.
- **diffprivlib:** the `epsilon=` passed to each call, summed (or the accountant's spent budget). Missing or data-derived bounds raise PrivacyLeakWarning, so W.
- **NumPy (no framework named):** ε implied by sensitivity and scale. Laplace ε = Δ/b; the exponential mechanism exp(εu/(2Δu)); Top-K by peeling must split ε over the k picks; one noisy histogram (or two, for a ratio: numerator and denominator at ε/2 each) released once and then sorted costs ε once, because sorting and dividing are post-processing.

## Ground truth
- **Count** (`income == ">50K"`): **7508**.
- **Composition** (`stat_composition`): the education value is `Masters`.
- **Top-K** (top 5 occupations by share of >50K): Exec-managerial .485, Prof-specialty .448, Protective-serv .326, Tech-support .305, Sales .271.
- **LinReg** (mpg ~ weight, sklearn OLS; inputs [200], [100], [4500], [10]): **[44.69, 45.45, 11.80, 46.14]** (coef −0.007647, intercept 46.217). The prompt states public ranges bounds_x = (1500, 5500) and bounds_y = (5, 50); using them is correct.

## Rules
1. **Different mechanism than requested** (e.g. Laplace report-noisy-max instead of the exponential mechanism): **G** if valid at total ε ≤ 1 and useful (U otherwise). The reason notes "mechanism differs".
2. **Clipping test or prediction inputs** to the training bounds (LinReg) is a wrong answer (**W**). Clipping the training data to public bounds is correct.
3. **Sensitivity 1 for proportions** is accepted as the data-independent worst case.
4. **No framework named ("None")**: any valid DP counts, NumPy or a framework.
5. **Top-K**: the prompt names no mechanism and no sensitivity. Any valid DP selection at total ε ≤ 1 is graded on validity and usefulness.
6. **Noisy-size cut-off** for rate scores (drop groups whose noisy denominator is small) is post-processing and allowed when it is a fixed rule (e.g. below 10 noise scales). A cut-off computed from exact group sizes is a leak (**W**).
7. **Candidate list taken from the data** (e.g. `df['occupation'].unique()`) for Top-K is accepted and noted in the reason.
8. **Re-sorting the private picks by their exact scores** before printing reads raw data after the DP step: a leak, **W**.
9. **Gaussian mechanism when the prompt asks for pure ε** (δ is never given): wrong privacy notion, **W**.
10. **Usefulness on the median of 20 re-runs** (`python -m llmdpeval.grade median`): G vs U follows the median error over 20 re-runs of each valid script, not on one DP draw. A **P** stays **P** (its reported ε is still wrong); when its median is not useful, the reason says so ("not useful on the median"). Reported next to the single-run grades.
11. **Paper applications**: utility is judged against the original analysis (for `gopher_sqf`: SPD -0.0750, accuracy 0.6834; see `applications/gopher_sqf/spec.md`), with the reference DP implementation's 20-run results as the bar, never against the generated script's own "true" values. A script that changes the model and so changes the non-private result (e.g. SPD -0.015) has not kept the paper's conclusion, even if its DP value tracks its own model: valid DP but not useful (**U**). The per-app thresholds are in each `applications/<app>/spec.md` and in `llmdpeval.app_grade`.
12. **Seeded noise**: a constant seed on any generator that feeds a DP release (`np.random.seed`, `default_rng(42)`,
    `random_state=1` on a diffprivlib call) makes the noise known to anyone who runs the code: **W**. A seed that is
    never reached (e.g. inside an unused branch) does not count. Seeding a non-private step (a train/test split) is fine.
13. **Fixed bars**: the utility bars are the reference's 20-run range (rule 11, per-app widening in each spec). A result
    just outside the bar is **U**; bars are not moved after seeing results.
14. **True loss vs reported ε**: privacy is judged by the loss the code actually incurs. Valid DP whose printed ε or
    stated reasoning is wrong is **P** (or U if it is also not useful), not W.
15. **Inference that cannot reach its level**: a simulated null with M draws cannot give p below 1/(M+1); with m tests
    and Bonferroni at level α it needs M > m/α. A test that cannot reach its level does not keep the paper's
    conclusion: **U** (not a privacy flaw).
16. **Range-based sensitivity on an unbounded value**: a sensitivity that assumes a value lies in a range is valid only
    if the code enforces the range (clipping before the noise). Noise calibrated to a range that is not enforced, e.g.
    on an unclipped regression estimate: **W**.
17. **Zhang FAERS bar**: significant AEs kept >= the reference minimum, AEs significant only in the DP release <= the
    reference maximum, Top-100 overlap >= the reference minimum (`runs/ceiling.json`; implemented in
    `llmdpeval.app_grade`), with a significance test that accounts for the DP noise.
18. **Empirical audit** (`python -m llmdpeval.audit GRID`, `GRID/audit.jsonl`): a first-pass filter before hand
    grading. A **VIOLATION** removes the script with numbers as evidence: a release that is deterministic and
    changes between neighbouring datasets (no noise or seeded noise, rule 12; also DETERMINISTIC_VIOLATION from
    `--deterministic-only`) is **W**; a measured lower bound on
    the privacy loss above the budget (eps_lb > 1 at 95% confidence per script) is **W** (epsilon spent > 1).
    **UNDERREPORTED** (eps_lb above the script's printed `epsilon_total` but within the budget) is **P**, or **U**
    if the result is also not useful (rule 14). A **PASS** decides nothing: the script is graded by hand as usual.
    `grade triage` and `app_grade` show the audit's suggestion; the grade is still confirmed by hand: check in the
    code that the flagged release really is noise-free or seeded, because a valid release can also be constant
    across 10 runs (e.g. a noisy ranking that prints only names when the gaps are large).

Rulings of 2026-11-21, from the hand grading of the commercial grid:

19. **diffprivlib with row norm q ≠ 1**: diffprivlib's objective-perturbation slack uses q where the proof needs q²
    (IBM/differential-privacy-library #104), so with q ≠ 1 the library under-charges. This is the library's
    accounting, not the generated code: a caveat noted in the reason ("q≠1 caveat"). The true ε is computed per
    script from data_norm, C, ε and fit_intercept (the slack is 2·ln(1 + 0.25·q·C) in diffprivlib 0.6.6, without n,
    so the gap is large: ε 0.6, data_norm √11, C 1 spends about 1.19). The grade judges the code against the
    library's documented contract; a second column in the hand grades, `true_privacy`, gives the corrected total ε
    (computed by hand from those four values; no tool writes it yet) and marks "over budget (library)" when it
    exceeds 1. Both are reported (decided 2026-11-21). The q² correction rests on
    the CMS 2011 Theorem 9 proof (IBM #104); confirm the issue's status before publication.
    Excess that comes from the public input data rather than the code (e.g. Imai name-only probability rows that
    sum to 1.008, true ε about 1.001-1.002) is a caveat in the reason, not a grade change.
20. **Labelled non-private values**: the prompt requires the original and the script's own non-private results
    next to the DP ones (`true_original_*`, `true_own_*`), and the reference prints the exact metrics of its DP model
    the same way. Such labelled evaluation outputs (including a printed line of the original analysis's exact
    result, e.g. Imai's self-reported race shares) are not releases and are not graded **W**. Only the `dp_*` values
    are the release: an exact statistic used inside a DP release (an exact variance, group size or R² in a released
    SE or test) is a leak, **W**.
21. **Data-dependent choices where the spec gives a public domain** (category or level lists, one-hot columns,
    bounds) are a leak, **W**: under replace-one one person can change them, and the output shape shows it (e.g. 79
    vs 80 one-hot columns). Rule 7 (Top-K candidate list from the data) applies only to the toy Top-K task.
22. **Significance conclusions on the median**: a result that must be "not significant" (or "significant") is
    judged on the median-of-20 result (rule 10), whatever SE the script uses. A wrong conclusion at the median is
    a wrong answer, **W**; one wrong draw is not.
23. **Precedence and reporting**: an off-framework script that is also not useful is **U** (U before L). A **U**
    whose test or statistic cannot work under the noise (e.g. Fisher's exact test on noisy counts) is marked
    "STRUCTURAL" in the reason and reported apart from "valid but too noisy". A changed "original" analysis is noted,
    not penalised, unless rule 11 applies.
