# Kulynych et al. (PoPETs 2022), "Disparate Vulnerability to Membership Inference Attacks": reproduction result

Run 2026-11-07 20:23-21:19Z on a SPHERE XDC (2 cores, 2 GB), authors' commit 69dea17, `adult.py` unchanged,
Python 3.8.20 with their poetry.lock pins (installed by pip `--no-deps`, see env/). Thread pools capped to 2.
`python compare.py outputs/adult_repro.csv <their results/adult_paper_snapshot.csv>` compares this run with the
results file the authors committed (9 models x 6 subgroups x 200 reps, same shape as this run). The run outputs are not kept in the repo.

**Verdict: adult (Table 2, adult half) MATCHES within noise.**
- Per (model, subgroup) means over 200 reps of vulnerability, test accuracy and generalisation gap: all 54 groups
  agree within 1.6 standard errors of the difference (largest absolute gap 0.032, test accuracy of DP LR eps=1 on
  Amer-Indian-Eskimo, a small subgroup where DP noise is large).
- Paper's significance verdict (bold in Table 2, disparity F-test p < 0.01): the same two models, the 8- and 32-neuron
  networks, in both runs. The p-values themselves move with the unseeded attack noise (e.g. DP LR eps=10:
  0.17 here, 0.042 in their snapshot; both not bold).
- Note: the current code labels models with display names ("8-Neuron NN"); the snapshot uses short ids ("nn_8").
  compare.py maps them.

Not run: texas-50K and the synthetic experiments (the other snapshot files). License: MIT.
