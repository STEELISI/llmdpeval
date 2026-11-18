# imai_fl: DP spec (Level 1)

Source paper: Imai & Khanna, "Improving Ecological Inference by Predicting Individual Ethnicity from Voter
Registration Records", Political Analysis 24(2) 2016, Tables 1 and 2. Replication package: Harvard Dataverse
doi:10.7910/DVN/SVY5VF v1.0 (R code + `FL.Anon.RData`). Terms of access: "Only academic use permitted", so the
data never goes in git, a public demo or any public place. R reproduction: `reproductions/imai2016/RESULT.md`.
Analysis code: `original.py` (a Python port; the authors' code has no licence and is not copied here).

Framing: the case where DP is believed to ruin a Census-based analysis
(Kenny et al., Science Advances 2021, doi 10.1126/sciadv.abk3283, on the 2020 Census Disclosure Avoidance
System). Two DP placements: the Census side (the public block tables the method uses) and the analyst side
(aggregates of the private voter file). The LLM task (`prompt.txt`) is the analyst side.

## What Kenny et al. tested (arXiv 2105.14197)
- They did test this method: BISG with the R package wru, block-level Census priors, on North Carolina's
  February 2021 voter file (about 5.8 million registrants with self-reported race). Florida's voter file was only
  one of six states' files used to build name-race probabilities, not for the evaluation.
- Census inputs: 2010 Census vs the Demonstration Data at epsilon 4.5 and 12.2 (2021-04-28) and 19.61
  (2021-08-12). Misclassification with last name only: 15.1% (2010), 15.6% (DAS-4.5), 15.1% (DAS-12.2),
  15.5% (DAS-19.61). Their abstract: the DAS "does not degrade the overall predictive accuracy of an
  individual's race", but individual-level predictions "differ significantly". Their concern was redistricting
  and majority-minority districts (East Ramapo case), not turnout by race.
- So for BISG accuracy, Kenny et al. themselves found little loss; the Census-side result below agrees.
  They did not estimate turnout by race (Table 2 here) or use Florida for evaluation. That part is new.

## Data
- `data/voters.parquet`, built by `build_data.py` from `FL.Anon.RData` (get it with
  `reproductions/imai2016/fetch.sh`; md5 + sha256 checked). Needs pyreadr, which is not in the repo lock:
  run it in a separate env, e.g. `micromamba create -p /data/envs/imai_tools -c conda-forge python=3.11
  pyreadr pandas=3.0 numpy pyarrow matplotlib`.
- Rows: the 9,247,810 voters with self-reported race (the authors' `Predict.R` filter; the count in the paper's
  Table 1 notes). 8,828 precincts, 25 districts, 290,917 blocks; 1,017 precincts lie in more than one district
  (9,931 district-precinct parts). Self-reported shares: White 0.681, Black 0.130, Latino 0.133, Asian 0.016,
  Other 0.039; 7,612 voters have 0 in all five `SR.*` columns.
- 30 columns: `District`, `Precinct`, `Block` (de-identified ids), `PID`, `vote08`, `SR.*` (5), `p_*`
  (Pr(race | surname)), `r_prc_*`, `r_blk_*` (Pr(geo | race)), `r_pid_*` (Pr(party | race)). Dropped: voter id,
  county, tract, surname code, age, gender, `SR.NAT` and the probability columns Tables 1-2 do not use.
  `r_*` are missing for 870 voters (one block / precinct with no Census match); their geo predictions are NaN
  and they drop out exactly as in R.
- Size: parquet 108 MB. Full-data cost on the SPHERE eval node (`/usr/bin/time`): `original.py` 14 s, peak 7.4 GB;
  `reference_dp.py` 27-29 s, peak 13.6 GB. No subset needed. Memory limit for generated scripts: 32 GB
  (`LLMDPEVAL_SANDBOX_MEMORY_GB=32`, rule 5 below).
- Public inputs: `p_*` (2000 Census surname list), `r_prc_*` / `r_blk_*` (2010 Census counts; see Census side),
  `r_pid_*` (wru's bundled party table; equal to wru 0.0-1's `pid` table, see `reproductions/imai2016` note A5),
  the precinct / district / block ids, the race labels, and n = 9,247,810.
- Caveat on `r_pid_*`: it is public because it ships in the wru package, but it was probably estimated from
  voter-file data (possibly including Florida's). It is treated as public like the Census tables (rule 4).

## Check against the paper and the R run (`check_original.py`, `runs/check_original.txt`)
- Table 1: 55 / 55 cells equal the paper at 3 dp (and the R run).
- Table 2, Goodman / name-only / Bayesian: 60 / 60 cells equal the paper at 3 dp; against the R run at full
  precision the max abs difference is 2.7e-13.
- Port notes: wru 0.0-1 `race.pred` with `census = ""` (source read from the CRAN archive tarball): surname-only
  returns `p_*` unnormalised (row sums up to 1.008); otherwise p * r (* r_pid) over its row sum (0/0 -> NaN).
  `classify.R`: strict maximum, ties -> no class. `error.R`: the false-negative / false-positive denominators
  include unclassified voters. Turnout.R district loops use `District` (R adapter A4 in `reproductions/imai2016`). Goodman: one simple OLS
  per race per district over district-precinct parts; split precincts get the n_k-weighted average.

## Privacy setup (analyst side)
- Private data: the voter file rows (self-reported race, turnout, party, location, surname).
- Unit of privacy: one voter (one row). Replace-one (bounded DP). Only n = 9,247,810 is public; every group size
  (race, precinct, district, district-precinct part) is private.
- Budget: epsilon = 1 total, delta = 0, basic sequential composition.
- Predicting one voter's race from public Census / surname / party tables plus that voter's own record is a
  per-record computation, not a release. Only aggregates over voters are releases.
- No hypothesis tests: Tables 1 and 2 report rates, bias and RMSE without significance. The paper's claim is a
  comparison (the Bayesian prediction has smaller bias / RMSE than Goodman's regression and King's EI, and than
  the name-only prediction; King's EI is not part of this app, rule 1). The ceiling below checks whether DP keeps the direction of each
  comparison, so the noise-aware test rule has no test to apply here.

## Expected releases (answer key for task identification)

1. Query type: Count per group (confusion counts), 5 times
   Query definition: for each prediction (name; name + precinct; name + block; name + precinct + party;
   name + block + party), the number of voters per (self-reported race, predicted class) cell, including
   unclassified voters; Table 1 rates are ratios of these counts.
   Target columns: [`SR.*`, `p_*`, `r_prc_*`, `r_blk_*`, `r_pid_*`, `PID`]
   Conditions: [all rows]
   Needs: per-race positives and negatives (race group sizes are private; n is public).
2. Query type: Count and sum per group (observed turnout by race)
   Query definition: voters and voters who voted, per self-reported race, per precinct and per district
   Target columns: [`SR.*`, `vote08`, `Precinct`, `District`]
   Conditions: [all rows; 8,828 precincts, 25 districts]
3. Query type: Sum per group (predicted turnout by race)
   Query definition: per precinct and district, sum of Pr(race) and of Pr(race) * `vote08`, for the name-only
   and the Bayesian (name + precinct + party) predictions; estimate = ratio
   Target columns: [`p_*`, `r_prc_*`, `r_pid_*`, `vote08`, `Precinct`, `District`]
   Conditions: [all rows]
4. Query type: Linear regression per group (Goodman's regression)
   Query definition: per district, regress district-precinct turnout on its share of each race; estimate =
   intercept + slope; needs per-part voter counts, voters who voted and race counts
   Target columns: [`vote08`, `SR.*`, `Precinct`, `District`]
   Conditions: [district-precinct parts within each district]
5. Query type: Weighted mean of errors (bias, RMSE)
   Query definition: precinct level: mean of (estimate - observed) and its square, weighted by voters of that
   race; district level: unweighted mean over 25 districts; for Goodman, name-only and Bayesian
   Target columns: [outputs of 2-4]
   Conditions: [per race]
Out of scope for Level 1: King's EI column (dropped, rule 1), Figure 1 AUCs, appendix tables,
predictions 6-9.

## Utility target
"Predictions with name + geography (+ party) misclassify 14.5-15.8% of voters vs 21.5% with names only
(Table 1). For turnout by race, the Bayesian prediction has a much smaller RMSE than Goodman's regression
(precinct: White 0.012 vs 0.069, Black 0.039 vs 0.162, Latino 0.035 vs 0.251, Asian 0.111 vs 0.713,
Other 0.094 vs 0.499; district: 0.004 / 0.010 / 0.021 / 0.046 / 0.053 vs 0.037 / 0.197 / 0.463 / 0.808 /
0.527) and a smaller one than the name-only prediction." Numbers from `original.py` (`runs/check_original.txt`,
equal to the paper). Grading rule (as gopher_sqf): utility is judged against the ORIGINAL analysis, with the
reference's 20-run results (`runs/ceiling.json`) as the bar, never against the script's own "true" values.
Required for "useful" (rule 2): Table 1 and the district rows of Table 2. Precinct rows of Table 2 are
reported against the ceiling but need not keep the paper's ranking. The bar is the UNCORRECTED reference
(rule 3); the SIMEX numbers are an expert ceiling only.

## Reference DP implementation (`reference_dp.py`, OpenDP 0.16.0)
- Table 1: one discrete Laplace histogram per prediction over 6 x 7 cells (self-reported race or none x
  predicted race / no class / unclassified), d_in = 2, epsilon 0.02 each (0.1 total). Rates are ratios.
- Table 2: one Laplace release of a 9,931 x 32 vector of sums per district-precinct part: voters and voters who
  voted by self-reported race (6 groups incl. none, one-hot), and sums of Pr(race) and Pr(race) * voted for the
  name-only and Bayesian predictions. One voter adds at most 6.02 to the vector's L1 norm (name-only rows clipped
  to L1 <= 1.01, a no-op on this data), so d_in = 12.04 under replace-one; parts are disjoint, so one release
  covers all parts (parallel composition). epsilon 0.9. Counts clipped at 0, turnout ratios clipped to [0, 1].
- Everything in Table 2 is post-processing of that vector. Output `table2` = this uncorrected result: the
  grading bar. Output `table2_simex` = expert ceiling only: SIMEX (Cook & Stefanski, JASA 1994) adds extra
  noise at 1.5-3x the release's variance, refits Table 2 and extrapolates a quadratic to zero noise. It uses only
  the DP output and the public scale, so it costs no privacy.
- Every epsilon is read from `measurement.map(d_in)` and asserted; total 1.0.
- Check: with no noise the post-processing gives back `original.py` (max diff 2.7e-13, `ceiling.py` asserts it).

### Ceiling (`ceiling.py`, 20 unseeded runs, `runs/ceiling.json` / `.txt`, plot `runs/ceiling.png`)
The BAR is the uncorrected reference (rule 3). Cells: original | DP median [IQR].
- Table 1 (required): every cell's DP median is within 0.001 of the original, e.g. overall error
  name + block + party 0.145 | 0.145 [0.145, 0.145], name only 0.215 | 0.215; Asian FN 0.532 | 0.532;
  Other FN 0.968 | 0.969.
- Table 2, district rows (required), Bayesian RMSE: White 0.004 | 0.006 [0.005, 0.008], Black 0.010 | 0.021
  [0.019, 0.022], Latino 0.021 | 0.027 [0.026, 0.031], Asian 0.046 | 0.068 [0.060, 0.074], Other 0.053 | 0.114
  [0.109, 0.124]. Goodman RMSE: 0.037 | 0.031, 0.197 | 0.129, 0.463 | 0.168, 0.808 | 0.219, 0.527 | 0.212.
  "Bayesian RMSE < Goodman RMSE" is kept in 20/20 runs for all five races. "Bayesian < name-only" is kept in
  20/20 (White), 11/20 (Black), 19/20 (Latino), 11/20 (Asian), 8/20 (Other).
- Table 2, precinct rows (reported, ranking not required), Bayesian RMSE: White 0.012 | 0.070, Black 0.039 |
  0.197, Latino 0.035 | 0.201, Asian 0.111 | 0.472, Other 0.094 | 0.454. DP noise in the observed precinct
  turnout dominates every RMSE (median Asian voters per precinct is 7, noise scale 13.4 per sum), and the
  ranking is lost (e.g. Bayesian < Goodman in 0/20 runs for Black and Other, 2/20 for Asian).
- Goodman under DP: noisy race shares bend its slopes, so its DP RMSE is often lower than the original's
  (district Asian 0.808 | 0.219). That is distortion, not improvement.
- Small groups suffer most: at district level the Other (0.053 -> 0.114) and Asian (0.046 -> 0.068) RMSEs
  grow most; at precinct level all groups are noise-dominated.
- Expert ceiling (SIMEX, not the bar), Bayesian RMSE: district 0.004 / 0.010 / 0.018 / 0.041 / 0.084;
  precinct 0.045 / 0.142 / 0.144 / 0.403 / 0.386 (White / Black / Latino / Asian / Other).

## Census side (`census_fetch.sh`, `census_dp.py`, `runs/census_dp.json`, plot `runs/census_dp.png`)
- Sources (public, no account, no key): 2010 PL 94-171 for Florida (table P2, same counts as SF1 P5), and the
  Census Bureau's 2010 Demonstration Data PPMF at production settings (2021-06-08, the data behind Kenny et
  al.'s DAS-19.61), streamed (13.4 GB, 169 s) and cut to Florida block x race counts.
- The method's Census input is only Pr(block | race) and Pr(precinct | race). Florida race totals rebuilt as
  1 / (smallest positive r_blk) equal PL P2 exactly (White 10,884,722, Black 2,851,100, Hispanic 4,223,806,
  Asian + NHPI 454,941, AIAN + other 95,727; two or more races left out), and r * total gives whole numbers, so
  the counts behind the authors' columns are rebuilt exactly. The authors' blocks are de-identified and are not
  linked to real blocks.
- DAS noise transplant: for each cell (block or precinct x race) with true count c, one error drawn from real
  Florida cells of the same race and the same true count in (Demonstration Data - PL). Real Florida errors: mean
  absolute error per block cell 0.26 (Other) to 1.91 (White); per VTD cell 2.7 to 11.8 (484,481 blocks, 9,435 VTDs).
- Results with the DAS noise (3 draws; original -> DAS): Table 1 overall error name + block + party
  0.145 -> 0.156, name + block 0.152 -> 0.162, precinct predictions unchanged at 3 dp (0.158 -> 0.159,
  0.151 -> 0.151). Asian false negatives rise most with block geography (0.532 -> 0.594). 4.4% of voters change
  predicted race with block geography, 0.3% with precinct geography. Table 2 (Bayesian uses precinct geography):
  every RMSE and bias moves by at most 0.0015.
- Discrete Laplace sweep on the block x race counts (OpenDP, d_in = 2, epsilon read from map(2)): overall error
  name + block + party 0.218-0.219 at epsilon 0.1, 0.157 at 1, 0.146 at 10; Table 2 Bayesian RMSE moves by
  0.0044 at most (Other, epsilon 0.1) over the whole grid.
- Reading: as in Kenny et al. for North Carolina, block-level DAS noise costs about 1 point of overall accuracy
  and flips a few percent of individual predictions; precinct-level inputs and the turnout-by-race estimates are
  essentially unaffected. Caveats: transplanted errors are independent per cell (the real DAS errors are
  correlated within a block and across levels); blocks with no voters are not in the authors' data, so their
  noise (which flows into precinct totals in the real DAS) is only covered through the VTD-level transplant.

## Rules
1. King's EI is dropped: it is R-only, slow (31 min in the R run), and the R run reproduced only 7 of its 20
   Table 2 cells. `original.py`, the prompt and the grading leave the EI column out; the paper's claim that the
   Bayesian prediction beats EI is not checked here.
2. Utility = Table 1 + the district rows of Table 2. Precinct rows are reported against the ceiling, but a
   script does not have to keep the paper's precinct-level ranking (the reference does not either).
3. Bar = the UNCORRECTED DP reference (`table2` in `reference_dp.py`, "BAR" in `runs/ceiling.txt`). SIMEX is
   reported only as an "expert ceiling" (`table2_simex`, `table2_expert_simex` in `runs/ceiling.json`).
4. `r_pid_*` is public (it ships with the wru package), with the caveat under Data.
5. Memory limit 32 GB per generated script (`LLMDPEVAL_SANDBOX_MEMORY_GB=32`); full data, no subset. Parquet
   data is fine. The Census side is a separate experiment, not an LLM task: the prompt treats the Census
   tables (`r_*`) as public inputs.

## Grading thresholds (`llmdpeval.app_grade`)
Utility keeps the conclusion when each Table 1 overall error is within 0.01 of the original and the name-only
prediction stays the worst, and at district level the Bayesian RMSE is below Goodman's for all five races. Precinct
cells inside the reference's 20-run range are counted and reported only (rule 2).

## Prompt
`prompt.txt` asks for three labelled values per output (original analysis, own analysis without noise, DP) and the `true_original_*` / `true_own_*` keys. Evaluation: 600 s and 32 GB per script (`eval.json`).
