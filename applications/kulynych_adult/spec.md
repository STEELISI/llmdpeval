# kulynych_adult: DP spec (Level 1)

Source paper: Kulynych, Yaghini, Cherubin, Veale, Troncoso, "Disparate Vulnerability to Membership Inference
Attacks", PoPETs 2022(1):460-480, Table 2 (adult half), DOI 10.2478/popets-2022-0023.
Authors' code: github.com/spring-epfl/disparate-vulnerability, commit 69dea17, MIT licence (not copied here).
Analysis code: `original.py` (a rewrite of their non-private logistic regression setup).
Framing: LLM vs expert DP. Three-way comparison: non-private LR vs the authors' own DP LR vs LLM-added DP, with
the reference DP implementation as the bar. Scope: logistic regression only (no neural nets, no DP-SGD).

## Data
- `train.csv` (32,561 rows) and `test.csv` (16,281 rows) from the authors' repo (`data/adult/`), pooled:
  48,842 rows. Missing values are empty cells in `work-class` (2,799), `occupation` (2,809) and
  `native-country` (857). After dropping them: 45,222 complete rows, the count UCI's `adult.names` states.
- Label `income` (`>50K` = 1). Sensitive attribute `race` (5 groups).
- Features (their `loaders/adult.py`): 7 categorical columns (`work-class`, `marital-status`, `occupation`,
  `relationship`, `race`, `sex`, `native-country`) and 4 numeric columns (`hours-per-week`, `capital-loss`,
  `capital-gain`, `education-num`). `age`, `fnlwgt` and `education` are not used.
- Their preprocessing, kept in `original.py`: StandardScaler on the numeric columns fitted on all 48,842 rows
  (before dropping missing rows), `get_dummies(drop_first=True)`, then `.astype(int)` on the whole matrix. That
  cast truncates the standardised numeric values to integers (e.g. education-num becomes -3..2). It looks
  unintended, but it is what produced their numbers, so it is kept. 79 features.
- Public domains (from `adult.names`, not measured): the categorical levels listed in `prompt.txt` and
  `reference_dp.py` (`DOMAINS`). `adult.names` gives no numeric ranges. Adult was extracted from the 1994
  Census/CPS database, so the money ranges come from the Current Population Survey March/April 1994 codebook
  (IPUMS CPS, https://cps.ipums.org/cps/resources/codebooks/cpsmarapr94.pdf), data dictionary lines
  "D CAP-GAIN 5 689 (0:99999)" and "D CAP-LOSS 4 694 (0:9999)": `capital-gain` 0..99999, `capital-loss` 0..9999.
  `hours-per-week` 1..99 and `education-num` 1..16 (the 16 education levels listed in `adult.names`).

## How the authors form their numbers
- Each of 200 reps is a fresh random 50/50 train/test split of the 45,222 rows
  (`train_test_split(test_size=0.5)`, seed `np.random.RandomState(i)` with `i` from `enumerate(range(200))`,
  i.e. seed tuple (k, k)). The model is fit on the train half; accuracy, generalisation gap and the
  membership-inference vulnerability are measured on both halves. Table 2 reports means over the 200 reps.
- `original.py` runs one split (`random_state=0`). `check_original.py` re-runs all 200 of their seeds.

## Check against the authors' results (non-private LR)
`check_original.py`, 200 reps, sklearn 1.7.2 / pandas 3.0.6 (theirs: 0.24.2 / 1.2.5), `runs/check_original.json`.
Mean test accuracy, this rewrite vs their `results/adult_paper_snapshot.csv` (model `lr`):

| Group | rewrite | theirs | paired diff, max abs | identical reps |
|---|---|---|---|---|
| All | 0.84043 | 0.84039 | 0.00097 | 19/200 |
| White | 0.83251 | 0.83249 | 0.00098 | 21/200 |
| Black | 0.90633 | 0.90611 | 0.00330 | 80/200 |
| Asian-Pac-Islander | 0.82690 | 0.82671 | 0.00767 | 53/200 |
| Amer-Indian-Eskimo | 0.90633 | 0.90669 | 0.00971 | 161/200 |
| Other | 0.89168 | 0.89146 | 0.01538 | 138/200 |

The splits are the same (several reps identical); the small per-rep differences are consistent with solver
changes between sklearn versions (their versions were not installed for this check). A full reproduction of their
pipeline in their pinned environment is in `reproductions/kulynych2022/RESULT.md`.

## Privacy setup
- Private data: the 45,222 complete Adult rows (train and test halves).
- Unit of privacy: one row = one person (Adult has one row per person).
- Neighbouring relation: replace-one (bounded DP). n = 45,222 is public (stated in `adult.names`); the split sizes
  (22,611 / 22,611) follow from n. Caveat: the complete-case filter is treated as public, because the public data
  dictionary states the count.
- Budget: epsilon = 1 total, delta = 0, basic sequential composition over both halves.
- Public inputs: the domains and ranges above, column names, n. Nothing is measured from the private data to
  set bounds, scaling or norms.

## Expected releases (answer key for task identification)

1. Query type: Logistic regression (model training)
   Query definition: fit a logistic regression that predicts `income` from the 7 categorical and 4 numeric columns
   Target columns: [`income`, `work-class`, `marital-status`, `occupation`, `relationship`, `race`, `sex`,
   `native-country`, `hours-per-week`, `capital-loss`, `capital-gain`, `education-num`]
   Conditions: [complete rows; train half of a 50/50 split]
2. Query type: Accuracy (count of correct predictions divided by the public test-half size)
   Query definition: fraction of test rows where the predicted label equals `income`
   Target columns: [`income`, model prediction]
   Conditions: [test half]
3. Query type: Accuracy per group (count of correct predictions per group divided by the group size)
   Query definition: accuracy over the test rows of each race group
   Target columns: [`income`, model prediction, `race`]
   Conditions: [test half; groups White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other]
   Needs: correct count per group and the group size (group sizes are not public).
4. Query type: Count per group (histogram)
   Query definition: number of rows per race group
   Target columns: [`race`]
   Conditions: [all complete rows; 5 race groups from the public domain]

## Utility target
"The non-private LR has test accuracy about 0.84 overall, with per-race accuracy from about 0.82
(Asian-Pac-Islander) to about 0.91 (Black, Amer-Indian-Eskimo); race counts White 38,903, Black 4,228,
Asian-Pac-Islander 1,303, Amer-Indian-Eskimo 435, Other 353." Numbers from `original.py` on its split
(`runs/original_run.txt`): All 0.8399, White 0.8320, Black 0.9085, Asian-Pac-Islander 0.8240,
Amer-Indian-Eskimo 0.9039, Other 0.8951.
Grading rule (as for gopher_sqf): utility is judged against the ORIGINAL analysis above with the reference's
20-run results (`runs/ceiling.json`) as the bar, never against the generated script's own "true"
value. A script that changes the model and so changes the non-private accuracies has not kept the paper's
analysis, even if its DP values track its own model: valid DP but not useful (U).

## Reference DP implementation (`reference_dp.py`)
- Model: diffprivlib 0.6.6 `LogisticRegression` (objective perturbation), epsilon 0.6. Numeric columns mapped to
  [0, 1] by the public ranges (clamped), full one-hot over the public domains (exactly one 1 per categorical
  column), an intercept column, then every row divided by sqrt(12) (4 + 7 + 1 entries of size <= 1). So the max
  row norm is q = 1 and diffprivlib's q-vs-q^2 slack term is exact (`data_norm=1.0`, `fit_intercept=False`).
  C = 0.1 from the slack rule used in gopher_sqf: 2 ln(1 + 0.25 C) = 0.049, under 10% of the model budget.
  No sweep over C on this data.
- Accuracy overall and per race: one OpenDP 0.16.0 sized count-by-categories over the 10 cells race x correct on
  the test half, discrete Laplace, epsilon 0.2. Per-race accuracy = noisy correct / (noisy correct + noisy wrong),
  overall = sum of noisy correct / public test size (post-processing).
- Race counts: OpenDP sized count-by-categories over the 5 races on all rows, discrete Laplace, epsilon 0.2.
- Every epsilon is read back from the framework (diffprivlib accountant, OpenDP `map(2)`); the script asserts each
  share and the total. Budget split: model 0.6, test histogram 0.2, race counts 0.2. Total 1.0.
- Noise check: the race-count noise standard deviation over 300 draws was 12.8 to 14.3 per cell, against 14.1
  for Laplace scale 10 (= 2 / 0.2).

### Ceiling (`ceiling.py`, 20 unseeded runs, `runs/ceiling.json`, capital-loss bound 0..9999)
| Group | original | no-noise control (public features, C=0.1) | DP release, median [IQR] | DP model, exact eval, median |
|---|---|---|---|---|
| All | 0.8399 | 0.8227 | 0.8128 [0.8079, 0.8160] | 0.8127 |
| White | 0.8320 | 0.8150 | 0.8067 [0.8031, 0.8092] | 0.8066 |
| Black | 0.9085 | 0.8935 | 0.8847 [0.8751, 0.8901] | 0.8848 |
| Asian-Pac-Islander | 0.8240 | 0.7988 | 0.7741 [0.7541, 0.7916] | 0.7685 |
| Amer-Indian-Eskimo | 0.9039 | 0.8777 | 0.8208 [0.6904, 0.8685] (range 0.60-1.00) | 0.8188 |
| Other | 0.8951 | 0.8704 | 0.8605 [0.8252, 0.9230] (range 0.51-1.00) | 0.8580 |

Race counts, DP median [IQR]: White 38,906 [38,899, 38,915], Black 4,232 [4,221, 4,233], Asian-Pac-Islander
1,301 [1,291, 1,307], Amer-Indian-Eskimo 437 [426, 446], Other 355 [350, 362]. Median absolute error 5-11 per group.
With a capital-loss bound of 0..99999 instead (`runs/ceiling_capLoss99999.json`) the overall median is also 0.8128.
Reading: about 0.017 of the overall accuracy drop is the change of features and C (control), and about 0.010 more
is DP noise. The two smallest groups (229 and 162 test rows on this split) are where DP hurts: their accuracy swings by
more than 0.1 between runs, from both model noise and count noise.

## Expert DP (authors')
The authors' DP model (`model_zoo.DpClassifierFactory`, used in `adult.py` for eps 1, 2, 10) is
`Pipeline(MinMaxScaler(), diffprivlib.LogisticRegression(max_iter=500, epsilon=eps, data_norm=sqrt(11)))` on the
same int-cast design matrix as the non-private LR (default C = 1, `fit_intercept=True`). Only the model is DP; their
accuracies and race counts are exact.
Framework rule: under the OpenDP condition the model may use diffprivlib or NumPy objective perturbation (OpenDP
has no logistic regression); the other releases are graded as OpenDP. The prompt says so after the framework line.

## Facts stated in the prompt
These facts remove ambiguities without stating a DP design. The JSON-line contract (requirement 6) is listed in
`prompt.txt`.
- The 4 numeric and 7 categorical input columns are listed, one-hot encoded over the public domains, so train,
  test and every subset have the same columns.
- The size of every subgroup (a race group, or a race group of the test half) is private.
- Preprocessing must not use statistics computed from the data; use the public domains.
- The trained model is a release (its weights are released).
- The model is trained on a random half and the accuracies are measured on the other half, as the share of correct
  predictions (overall and among the test rows of each race) for the released model; race counts are over all
  45,222 rows, not the test half.
- Requirement 5 asks for three labelled values per output: the original analysis's result, the script's own
  analysis without noise, and the DP value. Requirement 6 names the keys `true_original_*`, `true_own_*`, `dp_*`
  and `epsilon_total`. Grading uses the original; `true_own_*` shows whether a utility loss comes from DP or from
  a change of model. `app_grade` reads a plain `true_*` key as the original.

## Grading thresholds (`llmdpeval.app_grade`)
Utility keeps the analysis when: overall DP accuracy >= the reference's 20-run minimum - 0.01; each per-race DP
accuracy >= the reference minimum for that race - 0.02 (the small groups swing by > 0.1 at epsilon 1); each DP race
count lies in the reference's 20-run range widened on each side by the range's own width.
