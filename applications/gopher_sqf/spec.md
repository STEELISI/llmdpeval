# gopher_sqf: DP spec (Level 1)

Source paper: Pradhan, Zhu, Glavic, Salimi, "Interpretable Data-Based Explanations for Fairness
Debugging" (Gopher), SIGMOD 2022, Table 3 (SQF, logistic regression, statistical parity).
Analysis code: `original.py` (a rewrite; the authors' code has no licence and is not in this repo).

## Data
- `sqf_train.csv` (48,605 rows) and `sqf_test.csv` (23,941 rows), from the authors' repo
  (`backup/Integration/`, commit e6e7eb3). Total 72,546 rows; the paper says 72,548 (the two header
  lines would explain the gap). Already integer-coded; no raw file or preprocessing script is in the repo.
- Label `frisked` (1 = frisked). Sensitive attribute `race` (0/1). 17 features.
- Public domain (codebook, not measured): `build`, `age`, `ht_feet`, `weight` take codes 0..2;
  every other column is 0/1. `age`: 0 = under 25, 1 = 25-45, 2 = over 45 (from Table 3).
  `inout_I`/`inout_O` and `sex_F`/`sex_M` are one-hot pairs.
- Race coding caveat: the authors' code treats `race == 0` as the protected group, and Table 3's
  patterns reproduce only when `race == 0` is read as "Black". But in the data the frisk rate is
  lower for `race == 0` (train 30.2%) than for `race == 1` (38.5%), and the model's SPD is negative,
  which does not fit the paper's text ("frisking blacks ... more often"). Unresolved. Results report
  the group codes ("protected group `race == 0`" / "privileged group `race == 1`") with this note, and
  make no claim about the direction of bias against a named race.

## Privacy setup
- Private data: all SQF rows (train and test).
- Unit of privacy: one row, treated as one individual. Caveat: a row is a stop, and one person can
  be stopped many times, so person-level protection is weaker than row-level epsilon suggests
  (group privacy: k stops cost k * epsilon). The data has no person ID, so row level is the finest unit available.
- Neighbouring relation: replace-one (bounded DP). Row counts of both files are public.
- Budget: epsilon = 1 total, delta = 0, basic sequential composition over train and test together
  (conservative: parallel composition over the fixed train/test split would also be valid).
- Public inputs: the codebook domains above, the label/feature names, the row counts.
  Nothing is measured from the private data to set bounds, scaling or norms.

## Expected releases (answer key for task identification)

1. Query type: Logistic regression (model training)
   Query definition: fit a logistic regression that predicts `frisked` from the other 17 columns
   Target columns: [`frisked`, `race`, `build`, `age`, `ht_feet`, `weight`, `perobs`, `cs_objcs`,
   `cs_descr`, `cs_casng`, `cs_lkout`, `cs_drgtr`, `cs_vcrim`, `ac_proxm`, `inout_I`, `inout_O`, `sex_F`, `sex_M`]
   Conditions: [rows of `sqf_train.csv`]
2. Query type: Accuracy (count of correct predictions divided by the public row count)
   Query definition: fraction of test rows where (predicted frisk probability > 0.5) equals `frisked`
   Target columns: [`frisked`, model prediction]
   Conditions: [rows of `sqf_test.csv`]
3. Query type: Statistical parity difference (difference of two group means)
   Query definition: mean predicted frisk probability over `race == 0` minus the same over `race == 1`
   Target columns: [model predicted probability, `race`]
   Conditions: [rows of `sqf_test.csv`; groups `race == 0` and `race == 1`]
   Needs: sum of predicted probability per group and the group size (only n total is public).

Out of scope for Level 1: pattern supports and bias reductions (Level 2).

## Utility target
"The model is biased by SPD of about -0.075 (`race == 0` gets a mean predicted frisk probability
about 7.5 points lower than `race == 1`) at test accuracy about 0.683." Numbers from
`runs/original_*_env.json`. A DP release keeps the conclusion when the SPD sign is the same and
its size is close; see `runs/ceiling.json` for what the reference achieves.
Grading rule: utility is judged against the ORIGINAL analysis above (SPD -0.0750,
accuracy 0.6834) with the reference's 20-run results as the bar, never against the generated script's own
"true" value. A script that swaps the model and changes the non-private SPD (e.g. to -0.015) has not kept
the paper's conclusion, even if its DP value tracks its own model: valid DP but not useful (U).

## Reference DP implementation (`reference_dp.py`)
- Model: diffprivlib 0.6.6 `LogisticRegression` (objective perturbation), epsilon 0.6.
  Features divided by their public max code, intercept column added, every row divided by sqrt(18),
  so the max row norm bound is q = 1 and diffprivlib's q-vs-q^2 slack term is exact
  (`data_norm=1.0`, `fit_intercept=False`). C = 0.1, picked from the slack formula 2 ln(1 + 0.25 C)
  = 0.049 (under 10% of the model budget). Caveat: a sweep over C in {0.001, 0.01, 0.1, 1, 10} was
  also run on this data while designing the reference, so the choice is not fully data-blind.
- Accuracy: OpenDP 0.16.0 sized bounded sum of 0/1 correctness + Laplace, epsilon 0.1.
- SPD: three OpenDP sized bounded sums + Laplace at epsilon 0.1 each: sum of p over `race == 0`,
  sum of p over `race == 1`, count of `race == 0` (count of `race == 1` = n minus it).
- Every epsilon is read back from the framework (diffprivlib accountant, OpenDP `map(2)`), and the
  script asserts each share and the total.
- Budget split: model 0.6, accuracy 0.1, SPD 0.3 (0.1 + 0.1 + 0.1). Total 1.0.

Framework rule: under the OpenDP condition the model may use diffprivlib or NumPy objective perturbation (OpenDP
has no logistic regression); the other releases are graded as OpenDP. The prompt says so after the framework line.

## Facts stated in the prompt
These facts remove ambiguities without stating a DP design. The JSON-line contract (requirement 6) is listed in
`prompt.txt`.
- The 17 feature columns are listed (all columns but `frisked`, `race` included) and X has shape (rows, 17)
  before any intercept.
- The size of every subgroup (for example the number of rows with `race` = 0) is private.
- Preprocessing must not use statistics computed from the data (mean, std, min, max); the listed codes are public.
- The trained model is a release (its weights are released).
- Accuracy and SPD are defined in one line each, on `sqf_test.csv`, for the released model; SPD = mean predicted
  probability of `race` = 0 minus `race` = 1.
- Requirement 5 asks for three labelled values per output: the original analysis's result, the script's own
  analysis without noise, and the DP value. Requirement 6 names the keys `true_original_*`, `true_own_*`, `dp_*`
  and `epsilon_total`. Grading uses the original; `true_own_*` shows whether a utility loss comes from DP or from
  a change of model. `app_grade` reads a plain `true_*` key as the original.

## Grading thresholds (`llmdpeval.app_grade`)
Utility keeps the conclusion when DP SPD < 0 and lies in the reference's 20-run range widened by 0.01
([-0.104, -0.059]), and DP accuracy is within 0.01 of the original 0.6834. Otherwise U (if the privacy is valid).
