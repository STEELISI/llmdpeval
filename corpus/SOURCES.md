# Corpus sources and licences

Every outline names its source in its front matter (`source_url`, `source_license`). Outlines are paraphrased
pseudo-code (FORMAT.md rule), and the `.py` skeletons are newly written against the pinned APIs; no third-party
code is copied in. The main sources:

- OpenDP 0.16.0 documentation and code (MIT licence).
- diffprivlib 0.6.6 documentation and code (MIT licence).
- NumPy documentation (BSD-3-Clause licence).
- pandas documentation, copy-on-write guide (BSD-3-Clause licence): `numpy-errors-noise-arrays-pandas3`.
- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014), and DP papers on arXiv: paraphrased
  for the theory cards and some outlines.
- Published papers and textbook methods, listed below.
- Public GitHub repositories, listed below.

## Papers and textbook methods

Cited in item front matter (`source_url`, `source_license`, theory-card `sources`) or in the item text. Only the
method is paraphrased; no code or data from these papers is used.

| Reference | Used for |
|---|---|
| McSherry & Talwar, FOCS 2007 (exponential mechanism) | quantile and selection items, `*-group-quantile-gaps-ratios` |
| Holm, Scand. J. Statistics 6(2), 1979 (step-down multiple-comparison correction) | `*-k-arm-proportions-noise-aware` |
| DerSimonian & Laird, Controlled Clinical Trials 7(3), 1986 (random-effects pooling, tau^2) | `*-pooled-effect-many-experiments` |
| Gaboardi, Lim, Rogers & Vadhan, ICML 2016 (DP goodness-of-fit and independence tests; Monte Carlo null) | `*-rxc-independence-test-noise-aware` |
| Pearson chi-square / score test for 2x2 and r x c tables (textbook) | `*-two-group-item-tables-noise-aware`, `*-rxc-independence-test-noise-aware` |
| Kaplan-Meier estimator and life tables (textbook) | `*-survival-curve-exit-histogram` |
| Direct age standardisation (textbook epidemiology) | `*-rates-public-denominators-standardised` |
| Iterative proportional fitting; Chow-Liu trees (textbook) | `numpy-marginals-consistent-nonnegative`, `numpy-synthetic-data-tree-from-marginals` |
| Hay, Rastogi, Miklau & Suciu, PVLDB 3(1), 2010 (consistent noisy counts) | `theory-release-aggregates-then-derive` |
| Abadi et al., CCS 2016 (DP-SGD) | `numpy-dpsgd-mlp-rdp-accountant`, `theory-dp-sgd-accounting` |
| Mironov, CSF 2017 (Renyi DP) | `numpy-dpsgd-mlp-rdp-accountant`, `theory-dp-sgd-accounting` |
| Mironov, Talwar & Zhang, arXiv 1908.10530, 2019 (sampled Gaussian mechanism) | `numpy-dpsgd-mlp-rdp-accountant`, `theory-dp-sgd-accounting` |
| Chaudhuri, Monteleoni & Sarwate, JMLR 12, 2011 (objective perturbation) | `theory-objective-perturbation-slack` |
| Ding, Wang, Wang, Zhang & Kifer, CCS 2018 (detecting DP violations from repeated runs) | `theory-measuring-dp-utility` |
| Ponomareva et al., JAIR 77, 2023 (How to DP-fy ML) | `theory-measuring-dp-utility`, `theory-objective-perturbation-slack` |
| Haeberlen, Pierce & Narayan, USENIX Security 2011 (side channels around DP queries) | `theory-side-channels` |
| McKenna & Sheldon, NeurIPS 2020 (permute-and-flip) | `theory-top-k-selection` |
| Ding, Wang, Zhang & Kifer, PVLDB 13(3), 2019 (free gap information) | `theory-top-k-selection` |
| Gillenwater, Joseph, Medina & Ribero, ICML 2022 (joint top-k selection) | `theory-top-k-selection`, `diffprivlib-topk-joint-exponential-subsets`, `numpy-topk-count-joint-exponential` |
| Durfee & Rogers, NeurIPS 2019 (one-shot top-k) | `theory-top-k-selection` |

Three outlines paraphrase steps from code with no stated licence (two from DAI-Lab/Tiresias, one from the
*Hands-On Differential Privacy* code companion).

The `api_index/` files list the public API of the pinned framework versions. No code reads them; they were used
to check API names while writing outlines.

## GitHub sources

Read via the GitHub API at the commit shown (HEAD on 2026-11-02); no code was run. "Used for" names the outline
the repository is the source of, or whose pitfall it shows.

| Repo | Commit | Used for | Licence |
|---|---|---|---|
| advaithsrao/Fraud-Detector | 61d9cec | diffprivlib-model-selection-budget (pitfall) | Apache-2.0 |
| Anishreddy-gith/ISRA-ITDS | 5c8dee2 | numpy-per-record-noise-input-perturbation (source) | MIT |
| camelop/dp_lab | dec9b95 | opendp-legacy-api-names (source) | Apache-2.0 |
| capialbino/trustworthy-risk-classification | 763e851 | diffprivlib-private-bounds-then-fit (pitfall: data_norm = max row norm) | none stated |
| CSG-AISym4MED/PREDANO_Privacy_Enhancement_Data_Anonymization_Toolbox | d1913c6 | numpy-per-record-noise-input-perturbation (pitfall) | none stated |
| DAI-Lab/Tiresias | b1b8fae | diffprivlib-private-bounds-then-fit, diffprivlib-model-selection-budget (source) | none stated |
| different7/DSL-rewriting-to-inject-DP-mechanisms | a6a6845 | opendp-legacy-api-names (pitfall) | Apache-2.0 |
| eth-sri/Delta-Siege | 358377f | opendp-legacy-api-names (pitfall) | MIT |
| Hands-on-Differential-Privacy-Book/hands-on-differential-privacy | 69c4042 | opendp-private-bounds-log-histogram (source) | none stated |
| InseeFrLab/diff-privacy-OpenDP | 992407b | opendp-legacy-api-names (pitfall) | none stated |
| KnowledgeEdgeAI/PETs_for_Public_Health_Challenge | 9cef8e7 | opendp-user-measurement-privacy-map (source) | MIT |
| LaraujoM/ciber-app-poc | ef7bc8c | numpy-per-record-noise-input-perturbation (pitfall) | none stated |
| MacroAcon/Tavren | e32e4c5 | opendp-legacy-api-names (pitfall) | NOASSERTION |
| MinhNguyenOH/GNBvsLG | b550b30 | diffprivlib-private-bounds-then-fit (pitfall: data_norm = max row norm) | none stated |
| MortenMunk/PE-capstone | b804283 | numpy-per-record-noise-input-perturbation (pitfall) | none stated |
