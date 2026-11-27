# Zhang, Sumathipala, Zitnik (Nat. Comput. Sci. 2021), "Population-scale identification of differential adverse events before and during a pandemic": reproduction status

**Verdict (run 2026-11-12): the main results REPRODUCE.** Notebooks 3-7 ran unchanged on a SPHERE experiment node
(16 cores / 64 GB) in 3 h 04 min (notebook 3: 2 h 19 min, notebook 6: 44 min, the rest about 1 min;
`outputs/timing.log`). A 2 GB SPHERE XDC cannot run them: the memory limit restarts the container when notebook 3
loads `patient_safety.pk`. Full output of `compare.py`: `outputs/compare.txt`.

| Paper result (Dataverse CSV) | Match |
|---|---|
| 1. AE vs pandemic, all patients (7,761 AEs) | 2x2 counts 7761/7761 exact; beta within CSV rounding (max 0.005); significance verdict 7761/7761 |
| 2. Top100 AEs | 97/100 significant (same as the authors' own pickle); counts exact; rank rho 0.9999 |
| 3. Identified AEs per population | male, female, young, adult: identical sets. all patients 65 vs 64 and elderly 20 vs 19 (one extra AE each: MedDRA 10043089, 10022004); E/P labels, beta, PAEAI all agree on shared AEs |
| 4. Drug-AE pairs vs pandemic | 363 matched pairs, counts exact, delta within rounding. The run's pickle has 699 pairs, the CSV 41,860; 336 of the run's pairs are not in the CSV (not explained) |
| 5. AE-drug associations (faiss-matched controls) | A, B exact on 277/277 matched pairs; control counts C, D differ slightly on 266/277 (e.g. C 100 vs 107); gamma median gap 0.18; significance verdict 276/277 |

Reading: everything that is plain counting matches exactly. The drift is only in step 5's nearest-neighbour control
group, where ties between identical normalised feature vectors are broken by faiss/BLAS internals (see
"Sources of drift"). The OpenBLAS used here (conda-forge 0.3.34) is not the authors'. The one changed conclusion
(1 of 277 verdicts) and the two extra identified AEs come from that drift or from p-values at the threshold; not
traced further.

## Setup
- `fetch.sh`: authors' code at commit d3a62d0 (2021-10-11, MIT) into `original/`. 16 Dataverse files
  (doi:10.7910/DVN/G9SHDA v1.0, CC0) into `data/`, each checked against Dataverse's MD5, with sha256 in
  `checksums.sha256`. Downloaded: the 4 inputs the notebooks read (patient_safety.pk, AE_dic.pk, AE_mapping.pk,
  drug_mapping.pk) and the authors' 12 result CSVs. AE_mapping carries MedDRA terms (non-commercial licence), so
  `data/` is gitignored.
- `env/setup.sh`: Python 3.7.12 (README: "tested ... with Python 3.7"). The notebooks' metadata says 3.8.2, but
  numpy 1.16.5 and faiss 1.5.3 from `requirments.txt` have no 3.8 wheels, so 3.7 is the version consistent with
  the pins. Only the packages the notebooks import, plus nbconvert/ipykernel, are installed, with the authors' file
  as a pip constraints file. All 55 installed packages carry exactly the authors' versions (`env/pip-freeze.txt`):
  numpy 1.16.5, pandas 1.0.3, scipy 1.4.1, statsmodels 0.12.0, scikit-learn 0.23.0, faiss 1.5.3, tqdm 4.46.0,
  matplotlib 3.2.1, seaborn 0.10.1, plotly 4.11.0, nbconvert 5.6.1, ipykernel 5.3.4. Conda lock: `env/explicit-lock.txt`.
  - Adapter: the faiss 1.5.3 wheel needs the OS libraries `libopenblas.so.0` and `libomp.so`. The env installs
    conda-forge libopenblas 0.3.34 and llvm-openmp, and `env/activate-faiss.sh` puts `env/py37/lib` on
    LD_LIBRARY_PATH. faiss imports and searches correctly after that.
- `run.sh`: builds `work/Code` (copies of notebooks 3-7) next to `work/Data/{curated,pandemic/results,parsed}`
  (the layout in the README), symlinks the 4 inputs into `Data/curated`, and runs each notebook with
  `jupyter nbconvert --execute` (no timeout). Executed copies go to `outputs/`, timings to `outputs/timing.log`.
  The authors' notebooks are not edited. `NOTEBOOKS="5 6 7" ./run.sh` resumes on an existing `work/`.
- `compare.py`: compares the run with the Dataverse CSVs (sections below).

## Files the notebooks name but Dataverse lacks
- `../Data/curated/drug_dic.pk` (notebook 4, cell 2) is on a commented line. Notebook 4 never uses drug_dic.
  Notebooks 3, 5 and 6 load `drug_mapping.pk`, which is on Dataverse.
- `results/SE_<pop>_historical.pk` (notebook 5, cell 7) is also commented out. The live line loads
  `../Data/pandemic/SE_<pop>.pk`, which notebook 3 writes (cells 21/38/43/48).
- `../Data/parsed/drug_code_dic.pk` is written by notebook 3 cell 8. Notebook 6 only mentions it in a comment.

## Step 6 is required, not optional
Notebook 7 filters each population's AEs with `sig_drugSE_<pop>.pk` and builds the drug-AE pairs from
`top_SE_drug_*.pk` and `sig_SE_drug_pair*2020.pk`. Notebook 6 writes all three. Without step 6 there are no
"identified AE" lists, so `run.sh` runs 3-7 in order.

## What compare.py checks
1. AE vs pandemic, all patients (notebook 4 `SE_uncondition_2019_volcano.pk` vs
   `Associations_between_adverse_events_and_the_pandemic.csv`): 2x2 counts must be exact, beta (= ROR) within 0.005
   (the CSV rounds to 2 decimals), Bonferroni p within rounding, plus agreement on the p<0.05 verdict.
2. Top100 table: membership in the run's significant set, rank agreement (Spearman), exact counts.
3. Identified AEs per population (final AE set = SEs in notebook 7's `<pop>_2019_sig_{over,under}_drug_pair.pk`,
   with beta/p/PAEAI from notebook 5's `_step2.pk`): set overlap/Jaccard, E/P agreement, max |beta| and |PAEAI|
   differences, p-value log10 gap, PAEAI Spearman.
4. Drug-AE pairs vs pandemic (`top_SE_drug_uncondition_*`): exact counts and delta.
5. AE-drug associations with faiss-matched controls (A, B, C, D, gamma).

Check of compare.py itself: sections 1-2 on the authors' own notebook-4 output (shipped in their repo as
`figure/figuredata/SE_uncondition_2019_volcano.pk`) match their CSVs: all 7,761 rows with identical 2x2 counts, max
|beta difference| 0.0050 over the 6,207 finite betas (the other 1,554 are printed as "- -"), max |p difference|
0.0083 (rounding), Top100 97/100 significant with rho 0.9999. So the CSVs come from the same pipeline as the pickle,
and compare.py parses them correctly.

## Sources of drift
Notebook 6 picks 10 nearest controls per report with faiss `IndexFlatIP` on normalised report features. Ties
between identical feature vectors are common, and which neighbour wins depends on faiss and BLAS internals. faiss
is pinned to the authors' 1.5.3; the BLAS is not theirs. Gamma (section 5) is the number most likely to drift.

## How to run (a 64 GB node was used; peak memory not measured)
```bash
cd reproductions/zhang2021
./env/setup.sh            # also runs fetch.sh (about 2.4 GB download, checksummed)
./run.sh                  # notebooks 3-7; or NOTEBOOKS="6 7" ./run.sh to resume
~/bin/micromamba run -p env/py37 python compare.py
```
