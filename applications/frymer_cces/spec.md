# frymer_cces: DP spec (Level 1)

Source paper: Frymer and Grumbach, "Labor Unions and White Racial Politics", American Journal of Political
Science 65(1), 2021, Table 1 (CCES 2010 + 2012 cross-section, white respondents, OLS).
Replication package: Harvard Dataverse doi:10.7910/DVN/VJUOOV (v2). It states no licence, so neither the data nor
the authors' code is in this repo. Reproduction of their R code: `reproductions/frymer2021/RESULT.md`.
Analysis code: `original.py` (a Python port of `Labor_White_Racial_Politics_Code.R` lines 50-65, the three
`lm` models on `data[data$race==1,]`, plus the union / non-union means reported in the paper).

## Data
- `build_data.py` reads `cces_crosssectional_replication.RDS` (checked against the sha256 in
  `reproductions/frymer2021/checksums.sha256`), keeps `race == 1` (white) and the 9 columns Table 1 uses, and
  writes `data/cces_white.csv`: 170,299 rows. Rows with missing values stay; `original.py` drops them per model,
  like R's `lm`. `data/` is gitignored.
- Environment: `build_data.py` needs `pyreadr` (0.5.9 was used, with pandas 3.0.6), which is not in
  `requirements-lock.txt`. Everything else (`original.py`, `check_original.py`, `reference_dp.py`, `ceiling.py`)
  runs in the locked environment (OpenDP 0.16.0, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1). No statsmodels: the OLS is plain numpy.
- Complete rows: 71,431 for model 1 and 61,463 for models 2 and 3. Both counts are printed in Table 1.

## Check against the paper (`check_original.py`, `runs/check_original.txt`)
38 of the 40 printed Table 1 values match after rounding to 3 decimals: every coefficient, SE, N and R2, except
the `age` coefficient in models 2 and 3. The port gives 0.00046 and 0.00048; the paper prints 0.001. The age SEs
(0.000) and all other values match, so the models are the same. The authors' own stargazer output from the
reproduction (`reproductions/frymer2021/outputs/union_white_racialresentment_cces_new.html`) prints age 0.0005,
which the port's values round to; the paper's 0.001 is a printing round-up. Note: `reproductions/frymer2021/compare.py` uses a tolerance of 0.0011, so
its "every coefficient equals the printed value" also covers this case.
Factor coding follows R: the lowest code is the reference level (gender 1, state FIPS 1, year 2010).
`educ` is stored as an object column of integer codes in the RDS; the paper's single "education" coefficient
shows R treated it as numeric, and so does the port.

## Public domains (from the authors' `Codebook.pdf`, Dataverse file 3826163, not measured)
- `racial_resentment`: "Racial resentment index [0,1]", built from two Likert items (`racial_resent_special_favors`,
  `racial_resent_slavery`). Read as the mean of two 5-point items, so 9 values 0, 1/8, ..., 1 (the values in
  the file agree).
- `union_member`, `past_union_member`: 0/1 ("Respondent is a union member" / "was union member in the past").
  The reference maps each row to one status (current, else past, else never), so the two can never both be 1.
- `gender`: the codebook says "1 = Female, 0 = Male", but the file has the codes 1 and 2. The paper labels the
  `factor(gender)2` coefficient "female", so 2 is read as female (only the label depends on this).
- `famincome`: codebook codes 1 to 18 (income brackets; 17 and 18 are open top brackets).
- `age`: "Respondent age in years", no range in the codebook; public bounds 18..100, clamped.
- `educ`: codes 1 to 6 (No HS ... Post-grad).
- `state_pre`: state FIPS code; the 51 codes of the 50 states and DC (standard FIPS list).
- `year`: the paper's Table 1 uses the 2010 and 2012 surveys.

## Privacy setup
- Private data: the white CCES respondents in the complete-case rows of each model.
- Unit of privacy: one respondent (one row).
- Neighbouring relation: replace-one (bounded DP). Public: the complete-case counts printed in the paper
  (71,431 and 61,463). Not public: any other count, such as the number of union members, or of respondents per
  state or year (per-group sizes not printed in the paper are private). Caveat, as in kulynych_adult: the
  complete-case filter is treated as public because the paper prints the counts.
- Budget: epsilon = 1 total, delta = 0, basic sequential composition.
- Public inputs: the domains above, column names, the two counts. Nothing is measured from the private data to set
  bounds or scaling.

## Inference requirement
The paper's conclusions rest on significance. A DP release must come with a noise-aware standard error / test:
SE^2 = sampling variance + DP noise variance. A naive test that uses only the sampling variance is miscalibrated
(see the null calibration under Ceiling below).
The prompt states this requirement (requirement 2).

## Expected releases (answer key for task identification)

1. Query type: Linear regression (OLS), model 1
   Query definition: regress `racial_resentment` on `union_member` and `past_union_member`; coefficients, SEs,
   p-values, R2
   Target columns: [`racial_resentment`, `union_member`, `past_union_member`]
   Conditions: [white respondents with no missing value in these columns (n = 71,431)]
2. Query type: Linear regression (OLS), model 2
   Query definition: model 1 plus `female` (gender == 2), `famincome`, `age`, `educ` and state fixed effects
   (50 dummies)
   Target columns: [`racial_resentment`, `union_member`, `past_union_member`, `gender`, `famincome`, `age`, `educ`,
   `state_pre`]
   Conditions: [complete rows (n = 61,463)]
3. Query type: Linear regression (OLS), model 3
   Query definition: model 2 plus year fixed effects (one dummy, 2012)
   Target columns: [as model 2, plus `year`]
   Conditions: [complete rows (n = 61,463)]
4. Query type: Mean per group (difference of two group means)
   Query definition: mean `racial_resentment` of union members (`union_member == 1`) and of non-members, and the
   difference, with SE and p-value
   Target columns: [`racial_resentment`, `union_member`]
   Conditions: [rows of model 1; group sizes are private]
5. Query type: Hypothesis test (noise-aware)
   Query definition: SE and p-value of the union coefficient in each model and of the mean difference, with the DP
   noise variance added to the sampling variance
   Target columns: [the outputs of 1-4]

## Utility target
"White union members score lower on racial resentment than non-members: the `union_member` coefficient is
-0.063 (bivariate), -0.047 (controls + state FE) and -0.048 (+ year FE), all significant (SE 0.005); past members
-0.020 to -0.025. Mean resentment 0.626 (union) vs 0.683 (non-union), difference -0.057 (SE 0.005)."
Numbers from `runs/original_run.txt`. A DP release keeps the conclusion when the union coefficient (or the mean
difference) keeps its negative sign and stays significant at 0.05 under a noise-aware test (model 1, means). For
models 2 and 3 the bar is an honest noise-aware "not significant" (rule 1 below).
Grading rule (as for the other apps): judged against the ORIGINAL analysis above, with the reference's
20-run results (`runs/ceiling.json`) as the bar, never against the generated script's own "true"
value. A significant DP result from a naive test (SE without the DP noise) does not count as keeping the conclusion.

## Reference DP implementation (`reference_dp.py`)
Two releases, OpenDP 0.16.0, each epsilon read back from the framework's privacy map and asserted:
1. Model 1 and the group means, epsilon 0.2. One OpenDP sized count-by-categories over 27 cells
   (status current / past / never x the 9 resentment values) + discrete Laplace, d_in = 2 (replace-one).
   Scale 10. Model 1 is saturated in the 3 groups, so its coefficients are differences of group means; the
   coefficients, SEs, R2 and the union / non-union means are post-processing of the 27 noisy counts. The
   never-member count is the public n minus the two noisy group counts.
2. Models 2 and 3, epsilon 0.8. Noisy sufficient statistics (as studied by Sheffet 2019 and Alabi and Vadhan 2022
   for DP OLS): every regressor scaled to [0, 1] by its public bounds (clamped), then the column sums, the
   cross-products of different columns, and the squares of the non-binary columns, for the 58 columns
   (union, past, female, income, age, educ, year 2012, 50 state dummies, y): 489 numbers. Products that are 0 or
   duplicates for every possible row are left out (two different states, current x past, squares of 0/1 columns).
   A row has at most 8 non-zero entries, so its contribution has L1 norm at most 8 + 28 + 4 = 40, and replace-one
   moves the vector by at most 80. OpenDP vector Laplace (`make_laplace` on `l1_distance`), `map(80) <= 0.8`,
   scale 100. Model 2 is model 3 without the year entries (post-processing). Coefficients of the scaled income,
   age and educ are mapped back to the original units.
   Note: the row -> moment map is plain Python, not an OpenDP transformation; OpenDP accounts only for the
   Laplace step at the derived sensitivity 80. The script asserts the per-row bound on every row. An OpenDP
   user transformation would put the map inside the framework but needs the "honest-but-curious" feature, which
   the reference does not enable (rule 4).
- Noise-aware SE (a simplification, not the exact procedure of the papers above): plug-in sampling
  variance from the noisy statistics (sigma^2 (X'X)^-1, or the group variances), plus the DP-noise variance
  estimated by Monte Carlo: 500 times, add fresh noise with the released (public) scale to the noisy statistics
  and refit; the variance of the refits is the noise term. p-values use a normal reference. Known gap: the normal approximation is poor when DP noise dominates.
- Budget split 0.2 / 0.8, chosen from the sensitivities (2 vs 80) before any DP run. Total 1.0, asserted.

### Ceiling (`ceiling.py`, 20 unseeded runs, `runs/ceiling.json`)
| Release | original | DP median [IQR] | noise-aware SE, median | significant (noise-aware) | sign kept |
|---|---|---|---|---|---|
| model 1 `union_member` | -0.0627 | -0.0618 [-0.0641, -0.0604] | 0.0055 | 20/20 | 20/20 |
| model 1 `past_union_member` | -0.0203 | -0.0203 [-0.0207, -0.0199] | 0.0028 | 20/20 | 20/20 |
| model 2 `union_member` | -0.0475 | 0.0233 [-0.3527, 0.6855] | 18.18 | 0/20 | 10/20 |
| model 3 `union_member` | -0.0481 | 0.0193 [-0.3418, 0.4834] | 26.11 | 0/20 | 10/20 |
| mean difference (union - non-union) | -0.0573 | -0.0567 [-0.0589, -0.0553] | 0.0058 | 20/20 | 20/20 |

Group means, DP median: union 0.6263 (original 0.6256), non-union 0.6828 (original 0.6828).
For models 2 and 3 a naive test (sampling SE only) called the union coefficient significant in 4/20 runs, with
either sign; the noise-aware SE correctly says the DP release cannot answer the question.

Null calibration (no union effect; `null_calibration` in `runs/ceiling.json`), false positives at 0.05:
| Test | naive | noise-aware |
|---|---|---|
| model 1 union coefficient (2,000 synthetic histograms, real group sizes, y from the pooled distribution) | 11.25% | 5.20% |
| mean difference (same simulations) | 12.0% | 5.15% |
| model 2 union coefficient (300 permutations of y over the real regressors) | 98.0% (254 runs; 46 had no valid naive SE) | 0.67% |

Why models 2 and 3 fail (`model2_noise_diagnostic`, exact statistics + Laplace at scale 100 on a subset of
entries, 20 draws): SD of the union coefficient 1.58 with noise only on the 450 state-related entries, 0.063 with
noise only on the 39 other entries (the true coefficient is -0.047, SE 0.005). The state entries hurt most: the
smallest states have about a hundred complete rows, against noise of SD 141 per entry, so the noisy X'X is badly
conditioned. Even without the state entries, noise of this size swamps the union coefficient. A positive-definite
projection of the noisy X'X with an eigenvalue floor reduced the spread but biased the coefficient (it changed
sign at a large floor) and the floor would have to be tuned on the private data, so the reference does not use it.

## Rules
1. The task asks for all three models. At eps = 1 models 2 and 3 lose the union effect (0/20 in the ceiling), so for
   them the correct DP answer is a noise-aware SE that reports NO significance; claiming a significant effect for
   model 2/3 (e.g. from a naive SE) is graded W. Model 1 and the group means must keep the conclusion (20/20 in the
   ceiling).
2. `age` upper bound 100 (public assumption, clamp above), lower bound 18 (CCES adult sample).
3. `famincome` domain 1 to 18 (codebook); not narrowed from the data (rule: bounds from the codebook).
4. The row -> moment map stays plain Python with the bound asserted per row; no OpenDP user transformation,
   because that needs the "honest-but-curious" feature.

## Facts stated in the prompt
These facts remove ambiguities without stating a DP design. The JSON-line contract (requirement 6) is listed in
`prompt.txt`.
- The design matrix of each model is stated: model 1 has 3 columns, model 2 has 57 (constant, union, past union,
  female, famincome, age, educ, 50 state dummies with state 1 as reference; checked on the data: 51 states in the
  61,463 rows), model 3 has 58 (plus a 2012 dummy). The group means use the rows of model 1.
- No count other than the two complete-case counts is public (requirement 1).
- Preprocessing must not use statistics computed from the data; use the public domains.
- Requirement 5 asks for three labelled values per output: the original analysis's result, the script's own
  analysis without noise, and the DP value. Requirement 6 names the keys `true_original_*`, `true_own_*`, `dp_*`
  and `epsilon_total`. Grading uses the original; `true_own_*` shows whether a utility loss comes from DP or from
  a change of model. `app_grade` reads a plain `true_*` key as the original.

## Grading thresholds (`llmdpeval.app_grade`)
Utility keeps the conclusion when the model-1 `union_member` coefficient and the union - non-union mean difference
are both negative with p < 0.05. A p < 0.05 for model 2 or 3 (state fixed effects) is graded W (rule 1).
A model-1 SE below 0.9 x the reference's smallest noise-aware SE is reported as "SE may leave out the DP noise"
for the hand check.
