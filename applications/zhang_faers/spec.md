# zhang_faers: DP spec (Level 1)

Source paper: Zhang, Sumathipala, Zitnik, "Population-scale identification of differential adverse events before
and during a pandemic", Nature Computational Science 1, 666-677 (2021). Main table: "Associations between adverse
events and the pandemic" (all patients, 7,761 AEs) and its Top100 version (Dataverse doi:10.7910/DVN/G9SHDA v1.0).
Authors' code: github.com/mims-harvard/patient-safety, commit d3a62d0, MIT (not copied here; notebooks 3 and 4 are
the analysis). Analysis code: `original.py` (a rewrite of notebook 4 for all patients).
Scope: all-patients main table (2x2 counts per AE, ROR, significance) + Top-100. Populations
(sex, age), drug-AE pairs and the faiss-matched controls are out of scope.
Full reproduction of the authors' pipeline: `reproductions/zhang2021/RESULT.md`.

## Data
- Built by `build_data.py` from the Dataverse inputs (`patient_safety.pk` 9,325,731 FAERS reports, `AE_mapping.pk`,
  both sha256-checked) and the drop list in the authors' notebook 4. Output in `data/` (gitignored: MedDRA terms are
  under a non-commercial licence). Log: `runs/build_data.log` (4 min 17 s, 14.3 GB peak RAM on the SPHERE eval node).
- `reports.csv`: 432,072 rows, 12.2 MB. One row per US report filed by a health professional (qualify 1-3),
  received 2019-03-11..2019-09-30 (220,920 reports) or 2020-03-11..2020-09-30 (211,152). Columns `window`
  (2019/2020) and `aes` (MedDRA preferred-term IDs, `;`-joined, deduplicated). Same counts as notebook 3 cell 17.
- `ae_terms.csv`: the 24,313 MedDRA preferred terms of `AE_mapping.pk` with `name`, `soc` and two public flags:
  - `drop_before_test`: the authors' manual drop list (979 IDs, notebook 4 cell 7; COVID terms, device and product
    issues, and others). These AEs are never tested.
  - `drop_from_table` (8,681 terms): tested and counted in the Bonferroni correction, but not shown in the published
    table. Rule found by comparing notebook 4's output (9,943 tested AEs) with the CSV (7,761): every AE of the
    system organ classes Investigations, Product issues, Social circumstances, Surgical and medical procedures,
    plus 8 more terms (listed in `build_data.py`). The paper's caption only says "manually exclusion based on
    expertise". No other rule explains these 8.
- What the private data looks like (for the reader, not used to set anything): mean 2.58 AEs per report, median 2,
  max 155; 1,999 reports list no AE.

## How the paper forms its numbers (notebooks 3-4)
- Per AE: A19 / A20 = reports in the 2019 / 2020 window that list it; B = reports in the window minus A.
- ROR = (A20 x B19) / (B20 x A19) (the paper calls it beta); 95% CI exp(log ROR +- 1.96 sqrt(1/A19+1/B19+1/A20+1/B20)).
- Two-sided Fisher exact test on [[A19, B19], [A20, B20]]; Bonferroni over all 9,943 tested AEs; significant if
  p <= 0.05 / 9,943. Over/under-reported is split by the sign of A20 - A19 (not by ROR).
- Top-100 (inferred; the CSV does not state it): among main-table AEs with max(A19, A20) >= 100, the 100 with the
  smallest p, shown in descending ROR. This rule reproduces all 100 rows (thresholds 100 to 102 give the same set).
  3 of the 100 are not significant (Bonferroni p 0.052-0.060).

## Check against the authors' results (`check_original.py`, `runs/check_original.txt`)
| Target | Result |
|---|---|
| notebook 4 output (reproduction run, `SE_uncondition_2019_volcano.pk`) | same 9,943 AEs; 2x2 counts 9,943/9,943 exact; ROR identical; Fisher p max log10 gap 7.5e-10 (scipy 1.17 vs 1.4); verdict 9,943/9,943 (196 significant) |
| published main table (CSV) | same 7,761 AEs; counts 7,761/7,761 exact; ROR max diff 0.0050 (CSV has 2 decimals); 1,554 "- -" rows are ROR = inf here (A19 = 0); verdict 7,761/7,761 (159 significant) |
| published Top100 (CSV) | same 100 AEs; counts exact; 78/100 positions identical, the other 22 are neighbours with equal ROR at 2 decimals |
`original.py` runs in 6.3 s (650 MB) on the SPHERE eval node (`runs/original_time.txt`).

## Privacy setup
- Private data: the 432,072 reports in `reports.csv`.
- Unit of privacy: one report, with at most K of its AEs kept. FAERS has no patient ID, so a
  patient with several reports is protected only at the group level (several reports cost several epsilons).
  Sensitivity: replacing one report changes at most 2K (window, AE) counts by 1 each (L1 2K); the count of reports
  per window changes by 1.
- **K = 6**. Public rule: the 95th percentile of AEs per report (after the authors' drop list)
  in FAERS reports outside the analysed windows, with the same filters (US, health professional, received
  March 11 - September 30 of 2013-2018; 990,495 reports). Those years count as public prior data (FDA
  publishes FAERS quarterly). Source: `public_k.py`, `runs/public_k.txt`: median 1, 75th pct 2, 90th pct 4, 95th
  pct 6, 99th pct 13. The prompt states K = 6 as a public parameter, like epsilon. The K sweep (below) ran on the
  private reports and was not used to choose K.
- Neighbouring relation: replace-one (bounded DP). n = 432,072 (both windows) is public.
- Budget: epsilon = 1 total, delta = 0, basic sequential composition.
- Public inputs: n, the MedDRA term list with the two flags (`ae_terms.csv`), K. The number of reports in each
  window and every AE count are private (not published in the paper as such; counts not
  published are private). Which AEs occur at all is private too: the DP domain is the public term list.

## Expected releases (answer key for task identification)

1. Query type: Count per category (histogram), two of them
   Query definition: for every AE in the public term list (not `drop_before_test`), the number of reports in the
   2019 window that list it, and the same for the 2020 window
   Target columns: [`aes`, `window`]
   Conditions: [all reports; AEs of `ae_terms.csv` with `drop_before_test` = 0]
   Needs: contribution bounding (at most K AEs per report), sensitivity 2K under replace-one.
2. Query type: Count (per group)
   Query definition: number of reports in each window (n19; n20 = n - n19), needed for B = n_window - A
   Target columns: [`window`]
   Conditions: [all reports]
3. Query type: Ratio / hypothesis test (post-processing of 1-2)
   Query definition: ROR per AE from the noisy 2x2 counts, and a significance verdict with Bonferroni correction
   whose test includes the DP noise variance (noise-aware)
   Target columns: [noisy A19, A20, n19, n20]
4. Query type: Top-k selection
   Query definition: the 100 main-table AEs (`drop_from_table` = 0) with >= 100 reports in one window and the
   smallest p, sorted by ROR. Accepted forms: noisy histogram + sort (post-processing of 1-3, no extra budget), or
   report-noisy-max / exponential mechanism with its own budget share.

## Utility target
"Of 7,761 AEs in the main table, 159 change significantly between the 2019 and 2020 windows (128 more reported in
2020, 31 less), and the Top-100 lists the strongest changes among common AEs (>= 100 reports in a window)."
Numbers from `runs/original_run.txt`. A DP release keeps the conclusion to the extent it keeps significant AEs
without adding false ones, and recovers the Top-100. The bar is the reference's 20-run result (`runs/ceiling.txt`),
judged against the ORIGINAL analysis, never against the generated script's own "true" values.

### Ceiling (`ceiling.py`, 20 unseeded runs, K = 6, `runs/ceiling.json`, `runs/ceiling.txt`, `runs/ceiling.png`)
| Metric (vs original) | DP release, median [IQR] (range) | no-noise control (cap K = 6, same test) |
|---|---|---|
| significant AEs kept (of 159) | 35.5 [34, 37] (31-42) = 22% | 112 = 70% |
| significant in DP only (not in original) | 8.5 [7.75, 10] (5-12) | 10 |
| verdict agreement over 7,761 AEs | 0.983 [0.983, 0.983] (0.982-0.984) | 0.993 |
| Top-100 overlap | 64.5 [62.75, 66] (60-69) | 73 |

|log DP ROR - log ROR| by the AE's original 2019 + 2020 count (median over AEs, then over runs):
rare (< 100, 6,927 AEs) 0.81, but only 28% of them get a finite DP ROR; mid (100-999, 705 AEs) 0.155;
common (>= 1,000, 129 AEs) 0.072.

K sweep, diagnostic only (run on the private reports, not used to choose K; 5 DP runs each; `runs/ceiling.txt`,
top panels of `runs/ceiling.png`): significant AEs kept, median 44 / 46 / 48 / 38 / 33 for K = 1 / 2 / 3 / 5 / 10
(control 56 / 78 / 94 / 106 / 130); Top-100 overlap 45 / 52 / 58 / 64 / 55 (control 43 / 52 / 58 / 72 / 85). Small K
loses AEs through truncation; large K loses them to noise. AEs significant only in the DP release: 18 / 14 / 12 /
11 / 4; the no-noise control has about as many (16 / 14 / 15 / 12 / 3), so they come from truncation (it changes
the 2x2 tables), not from noise.

Null check (bottom right of `runs/ceiling.png`; AE totals kept, 2020 counts redrawn with no real change, the
reference's noise added): share of p < 0.05 is 2.9% without DP, 51.3% for a naive test on the DP counts, 2.1% for
the noise-aware test. So the noise-aware test is conservative here, and the naive test is badly wrong.

Reading: the cost of DP here is mostly on rare AEs. Common AEs keep their ROR within about 7-17%, and about two
thirds of the Top-100 survive. Most of the 159 significant AEs are rare (many have a zero cell), and noise of scale
12.6 per count hides them.

## Reference DP implementation (`reference_dp.py`)
- Contribution bounding: each report keeps its AEs that are in the public domain (term list minus
  `drop_before_test`); if more than K = 6, a uniform random subset of K.
- Reports per window: OpenDP 0.16.0 sized count-by-categories over {2019, 2020} + discrete Laplace, epsilon 0.05
  (scale 40). n20 = n - n19.
- AE x window histogram: tokens "window|AE" (at most K per report) into an OpenDP count-by-categories over all
  2 x 23,334 public cells + discrete Laplace, d_in = 2K (symmetric distance on the token multiset), epsilon 0.95
  (scale 12.63 at K = 6).
- Every epsilon is read back with `map(d_in)`; the script asserts each share and the total (1.0000).
- Test (post-processing): score test of the 2x2 table. Given the AE's total T = A19 + A20, under "no change" A20
  has mean pi*T (pi = n20 / n) and hypergeometric variance; z^2 is the Pearson chi-square. The noise-aware version
  adds the DP variance of the numerator (discrete Laplace variance of both counts, and of n19) to the denominator.
  Normal approximation (known gap: it tilts when noise dominates; here it errs conservative).
  Vu & Slavkovic (2009) and Gaboardi et al. (2016) show that tests on noisy tables need such corrections; the
  reference implements only this normal-approximation version, not their methods.
  The score test is used, not a Wald test on log ROR, because the Wald test with a 0.5 correction lost 46 of the 159
  significant AEs even without noise or cap (zero cells); the score test without noise or cap keeps 154/159 with no
  extra ones and the Top-100 100/100 (Wald: Top-100 99/100). Source: `test_choice.py`, `runs/test_choice.txt`.
- AEs tested: noisy A19 + A20 >= 5 noise scales (a public rule; m = number tested, for Bonferroni).
- Top-100: the paper's rule applied to the noisy counts and the noise-aware p (noisy histogram + sort).
- Runtime on the SPHERE eval node: 12.2 s, 704 MB (`runs/reference_time.txt`). Ceiling: 252 s for 20 runs + sweep.

## Grading notes
- Valid DP needs: a contribution bound per report (the full AE list of one report can be 155 AEs; using
  sensitivity 1 or 2 per AE count without bounding is NOT DP); sensitivity matching the bound (2K for the
  histogram under replace-one); window sizes treated as private; the AE domain from the public term list (keeping
  only AEs observed in the data leaks which AEs occur); epsilon read from the framework and summing to <= 1.
- Fisher exact or a naive chi-square on noisy counts: the release is DP, but the significance verdicts are not
  valid (null check: 51% false positives). Grade the verdicts as U (not useful) if no noise-aware test is used.
- Printing true counts next to DP counts is expected (requirement 5); it is not part of the DP release.
- Top-100 by noisy histogram + sort is post-processing; a separate exponential mechanism must have its own budget.

## Grading thresholds (`llmdpeval.app_grade`)
Utility keeps the conclusion when the Top-100 overlap with the original is >= the reference's 20-run minimum (60),
and, when the original's 159 significant AEs can be checked, significant AEs kept >= the reference minimum (31) and
AEs significant in the DP release only <= the reference maximum (12).

## Prompt
`prompt.txt` asks for three labelled values per output (original analysis, own analysis without noise, DP) and the `true_original_*` / `true_own_*` keys. Evaluation timeout 600 s per script (`eval.json`).
