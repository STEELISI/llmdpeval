# Imai & Khanna (Political Analysis 24(2) 2016), "Improving Ecological Inference by Predicting Individual Ethnicity from Voter Registration Records": reproduction result

Run 2026-11-11 on a SPHERE experiment node (16 cores / 62 GB), R 3.5.1 (conda-forge env + CRAN Archive
tarballs pinned to 2016-01-20, see env/), the authors' scripts from Dataverse doi:10.7910/DVN/SVY5VF v1.0 in the
Replication Guide's order (`run.sh`). Inputs are pinned by `checksums.md5` (Dataverse's own md5s) and
`checksums.sha256`. `compare.py OUTPUT_DIR` compares the run outputs with the paper's printed values, typed into
`expected/`. Data terms: "Only academic use permitted", so `original/`, `work/` and `outputs/` stay out of git.

**Verdict: Table 1 and Figure 1 MATCH exactly. Table 2 matches in 67 of 80 cells; all 13 differences are King's
EI cells, and the three columns that do not use ei (Goodman, name-only, Bayesian) match 60 of 60.**

| Target | Cells matching the paper | Max abs difference |
|---|---|---|
| Table 1 (classification error rates, 5 predictions x 11 rows, 3 dp) | 55 / 55 | 0.0000 |
| Figure 1 AUCs (3 predictions x 3 races, 2 dp) | 9 / 9 | 0.0049 (rounding) |
| Table 2, Goodman's regression (bias, RMSE; precinct + district) | 20 / 20 | < 0.0005 |
| Table 2, name-only prediction | 20 / 20 | < 0.0005 |
| Table 2, Bayesian prediction (name, precinct, party) | 20 / 20 | < 0.0005 |
| Table 2, King's EI | 7 / 20 | 0.0725 |
| Sample counts: 9,247,810 voters with self-reported race; 8,828 precincts; 25 districts; race shares 68/13/13/2/4% | all | 0 |

King's EI differences (paper -> this run): White and Black and Latino cells are off by at most 0.002 (precinct White
bias 0.041 -> 0.0418, RMSE 0.062 -> 0.0625; precinct Black bias -0.133 -> -0.1340; district Black -0.215 -> -0.2160,
0.267 -> 0.2676). Asian and Other are off by 0.01 to 0.07: precinct Asian bias -0.470 -> -0.415, RMSE
0.550 -> 0.523; precinct Other -0.338 -> -0.328, 0.450 -> 0.441; district Asian -0.459 -> -0.386, 0.530 -> 0.499;
district Other -0.342 -> -0.326, 0.448 -> 0.432. ei() fits by optimization then importance sampling, seeded once
(`set.seed(3)`) before 125 fits, so any difference in ei's version or in the numeric dependencies changes every
later draw. Likely cause: this run uses ei 1.3 (first CRAN release, 2016-01-20) and ucminf 1.1-4, while the authors ran a
pre-CRAN ei (KingEI.R is dated 2016-01-08) on R 3.2. The small groups (Asian 2%, Other 4%) have the least data per
precinct, so their EI estimates move most. Not tried: ei 1.3-2 (2016-03-25).

Runtimes and peak memory on the node (one R process per script; logs in `logs/`):

| Script | Wall time | Peak RSS | Status |
|---|---|---|---|
| Predict.R | 22 min | 29.0 GB | ok (with A5) |
| Table1.R / Figure1.R / Table7.R | 6 / 10 / 1 min | 11.4 / 9.6 / 4.8 GB | ok |
| Precincts.R / Goodman.R | 5 min / 9 s | 7.3 / 0.1 GB | ok |
| KingEI.R | 31 min | 0.3 GB | ok (with A2 and ucminf 1.1-4) |
| Turnout.R | 2 h 38 min | 10.7 GB | ok (with A1, A4) |
| Table2.R / Table5.R / Table6.R | seconds | 0.1 GB | ok |
| Residuals.R | 3.5 min | 18.1 GB | ok |
| Figure2.R | | | see "Appendix" |
| Figure3.R | 21 s | | FAILS: author bug (below) |
| L2.Demo.R -> Predict.L2.R -> Table3.R | ~5 h (estimated) | | see "Appendix" |

Tables 5, 6 and 7 (appendix) ran, but the Supplementary Appendix is not in the paper PDF used here, so they are not compared.

## Environment
- R 3.5.1 from conda-forge (oldest full conda-forge R; the code dates from Dec 2015 - Jan 2016, R 3.2.x). R < 3.6
  keeps the old `sample()` RNG, which matters only for KingEI.R (`set.seed(3)`).
- From the CRAN Archive at the version current on 2016-01-20: wru 0.0-1, ei 1.3, eiPack 0.1-7, xtable 1.8-0,
  ROCR 1.0-7, mvtnorm 1.0-4, msm 1.6, mnormt 1.5-3, cubature 1.1-2, tmvtnorm 1.4-10, gmm 1.5-2,
  sandwich 2.3-4. Posit's dated CRAN snapshots start 2017-10-10, so exact tarballs are used instead.
- Everything else (plotting/iteration imports: rgl, sp, foreach, plotrix, ellipse, gplots, survival, expm, coda,
  MASS, Matrix) is conda-forge's R 3.5.1 build; exact builds in `env/explicit-lock.txt`, R library versions in
  `env/r-packages.csv`.
- ei 1.3 (2016-01-20) is the first ei release on CRAN; KingEI.R is dated 2016-01-08, so the authors ran an earlier,
  pre-CRAN ei. This is the closest available version.
- One version is later than Jan 2016: ucminf 1.1-4 (2016-08-18) instead of 1.1-3 (2012). 1.1-3's C interface lets
  Fortran overwrite its input vectors in place, including `.stepmax`, which aliases the literal `1` in `ucminf()`'s
  own code. On R 3.5 this breaks ei(): with byte-compiled 1.1-3 the first KingEI.R fit (district 1, White) stops
  with "lower>upper integration limits"; with 1.1-3 installed `--no-byte-compile` the 5th fit stops with "sigma
  must be a symmetric matrix" (both seen in this run). 1.1-4 only adds
  `duplicate()` of those in/out arguments in `interface.c` (and namespaced numDeriv calls on a path ei does not
  use); the optimizer is unchanged. With it all 125 ei fits ran.

## wru: no network, no Census key
Predict.R calls `race.pred(voters, name.clean = FALSE, ...)` with the default `census = ""`. In wru 0.0-1 that
path only multiplies columns already in FL.Anon (`p_*` = Pr(race | surname), `r_*` = Pr(geo | race), and for
`party = "PID"` the bundled `pid` table merged into the data). The Census API code (`census.helper.api`) runs only
when `census = "tract"/"block"`, which the authors never pass. So the 2016 API runs offline, and no Census data or
key is needed. (Current wru renamed the function `predict_race` and needs a key; it is not used here.)

## Adapters (outside the authors' files; `original/` is never edited)
- A1 `Turnout.R` uses `SR`, which only `Predict.R` defines (the authors ran everything in one R session). run.sh
  defines the same `eth` and `SR` vectors before sourcing it.
- A2 `KingEI.R` line 59 selects column `"Distrct"` and stops with "undefined columns selected" after saving
  `ei.cd.RData` but before `ei.prc.RData` (needed by Table2.R). run.sh fixes the word in `work/KingEI.R`.
- A4 `Turnout.R` selects districts with `df$CD == d`, but FL.Anon has no `CD` column (it is `District`). Unmodified,
  every district-level observed and predicted turnout is NaN, so Table 2's district rows cannot be computed.
  run.sh replaces `$CD == d` by `$District == d` in `work/Turnout.R` (5 lines).
- A5 `Predict.R` predictions 4-5 call `race.pred(..., party = "PID")`. wru (0.0-1 and every later 0.x) then
  merges its bundled `pid` table (columns `r_pid_*`) into the data, but FL.Anon already has `r_pid_*`, so `merge()`
  makes `r_pid_*.x`/`.y` and race.pred stops ("undefined columns selected"). FL.Anon's `r_pid_*` equal wru's `pid`
  table exactly (3 PID values, same 5 numbers each, checked on the data). run.sh defines a global `race.pred`
  wrapper that drops the data's `r_pid_*` for `party=` calls only and then calls `wru::race.pred` unchanged; the
  result is what the merge computes. (The authors presumably added `r_pid_*` to FL.Anon after running Predict.R,
  since predictions 8-9 need them.)
- A3 `Figure1.R` computes the AUCs only for the PDF legend; run.sh writes its `AUC` data.frame to
  `figure1_auc.csv` after the script ends, for compare.py.

A1, A2, A4 and A5 are bugs in the shipped package (each stops the script or yields NaN in any R version); all four
are worked around outside `original/`. Table 2's district rows match the paper only with A4, which confirms that
`District` is the column the authors meant.

## Appendix
- Table 4: not run. `Predict.EM*.R` use `foreach` + `doMPI` on a cluster, and the guide warns they take a long time.
- Table 3: `./run.sh appendix` runs L2.Demo.R -> Predict.L2.R -> Table3.R. L2.Demo.R loops over 8,828 precincts
  at about 31 per minute (measured), so it needs about 4.8 hours; the other two take minutes. No comparison target
  (appendix); not compared.
- Figure 2: Residuals.R ran (3.5 min, 18.1 GB); Figure2.R draws qqplots of tens of millions of residuals. Not
  compared.
- Figure 3: Figure3.R line 13 is `census.prc[order(census.prc$Precinct)]` (missing comma, so it selects 8,828
  columns) and stops with "undefined columns selected". Author bug; not worked around, no comparison target.

## Reproduce
```
./fetch.sh                      # 32 files from Dataverse, md5 + sha256 checked
MAMBA_ROOT_PREFIX=/data/mamba ./env/setup.sh
./run.sh all                    # Predict, then the Table 1/Figure 1 and Table 2 chains in parallel (3 R processes)
python3 compare.py outputs
./run.sh appendix               # optional, about 5 h
```
Peak memory is Predict.R (29 GB); Turnout.R is the long pole (2.6 h, single-threaded precinct loops).
