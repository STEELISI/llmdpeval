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
  - extra exact values printed next to the private answer (e.g. the exact scores of the private picks). Printing the true answer is required by the prompt and is fine.
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
