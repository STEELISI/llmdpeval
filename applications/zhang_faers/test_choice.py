"""Why the reference uses a score test: compare tests on the exact counts (no DP noise, no cap on AEs per report).

For the 9,943 tested AEs, Bonferroni at 0.05 over 9,943 tests, compared with the original Fisher exact verdicts on
the 7,761 main-table AEs (159 significant) and the original Top-100:
  - Wald test on log ROR with a 0.5 correction on every cell;
  - score test (reference_dp.noise_aware_test with zero noise variance; z^2 = Pearson chi-square).
Usage: python test_choice.py <data_dir>   > runs/test_choice.txt
"""
import sys

import numpy as np
from scipy.stats import norm

import original
from reference_dp import noise_aware_test


def wald_p(A19, A20, n19, n20):
    """Two-sided Wald p-value of log ROR with 0.5 added to every cell."""
    a19, a20, b19, b20 = A19 + 0.5, A20 + 0.5, n19 - A19 + 0.5, n20 - A20 + 0.5
    z = np.log(a20 * b19 / (b20 * a19)) / np.sqrt(1 / a19 + 1 / b19 + 1 / a20 + 1 / b20)
    return 2 * norm.sf(np.abs(z))


def main(data_dir):
    reports, terms = original.load(data_dir)
    n, A = original.counts(reports)
    t = original.two_by_two(n, A, terms)
    m = len(t)
    A19, A20 = t.A19.to_numpy(float), t.A20.to_numpy(float)
    tab_mask = (t.drop_from_table == 0).to_numpy()
    true_sig = set(t.index[tab_mask & t.significant.to_numpy()])
    true_top = set(original.top100(original.published(t)).index)
    print(f"tested AEs {m}; main table {tab_mask.sum()}, significant with Fisher (original) {len(true_sig)}")
    for name, p in [("Fisher exact (original)", t.p.to_numpy()), ("Wald log ROR, +0.5", wald_p(A19, A20, n[2019], n[2020])),
                    ("score test", noise_aware_test(A19, A20, n[2019], n[2020], 0.0, 0.0)[1])]:
        tt = t.assign(p=p, significant=p <= original.ALPHA / m)
        sig = set(tt.index[tab_mask & tt.significant.to_numpy()])
        top = set(original.top100(original.published(tt)).index)
        print(f"{name:24s} significant {len(sig):3d}, kept of {len(true_sig)}: {len(sig & true_sig):3d}, "
              f"not in original: {len(sig - true_sig):2d}, Top-100 overlap {len(top & true_top)}/100")


if __name__ == "__main__":
    main(sys.argv[1])
