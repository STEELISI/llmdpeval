# Frymer & Grumbach (AJPS 65(1) 2021), "Labor Unions and White Racial Politics": reproduction result

Run 2026-11-07 to 2026-11-11 on a SPHERE XDC (2 cores, 2 GB), R 3.5.1 (conda-forge env + 2019-04-01 CRAN snapshot,
see env/), the authors' `Labor_White_Racial_Politics_Code.R` unchanged, run in 11 sections with workspace checkpoints
(`run_sections.sh`). Sections 4-5 (the two `mediate(sims=1000)` calls, peak ~9.3 GB) do not fit in 2 GB; they ran
from `checkpoint_3.RData` in native Windows R 3.5.1 (`env/install_win_r351.R`). Section 8 (genetic matching, `GenMatch`) took about 3.8 days.
Input files are pinned by `checksums.sha256`. `compare.py OUTPUT_DIR` compares the run outputs with the paper's printed
tables, typed into `expected/main_tables.csv`. The run outputs are not kept in the repo.

**Verdict: Tables 1-4 MATCH (60 of 61 printed values; the one difference is a printing slip in the paper).**
- Tables 1, 3, 4 (OLS, deterministic): every coefficient, N and R-squared equals the printed 3-decimal value.
- Table 2 (genetic matching, unseeded bootstrap in the original): estimates -0.041 (current members) and -0.020
  (past members) on N = 61,463, the paper's values.
- The difference: Table 3 model 5, `lost_union`, the paper prints 0.003; the authors' own stargazer output from this
  run prints 0.0003 (SE 0.015, not significant either way). Read as a dropped zero in the paper, not a reproduction
  failure.

Environment adjustments (the authors' code is not changed):
- `ltm` (loaded at line 549) and five other packages have no R 3.5 conda-forge build; `env/cran_packages.R`
  installs them from the CRAN snapshot.
- `ltm`'s dependency `msm` (2019) fails to link with GCC >= 10 (`multiple definition of expm`); `env/Makevars` adds
  `-fcommon`.

Not checked: the figures (PDF) and the appendix tables. License: see the authors' Dataverse terms.

## Reproduce
```
./env/setup.sh                  # pinned R 3.5.1 env in env/r351
./run_sections.sh               # fetches and checks the inputs; or ./run.sh in one R process (needs ~10 GB)
python3 compare.py outputs
```
