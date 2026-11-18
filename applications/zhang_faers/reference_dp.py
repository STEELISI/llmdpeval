"""Reference DP version of original.py: per-AE 2x2 counts, ROR, significance, Top-100.

Privacy unit: one report, keeping at most K of its adverse events (AEs); replace-one (bounded) DP. Public: the total
number of reports in the two windows (n), the MedDRA term list with the authors' drop flags (ae_terms.csv), K.
Everything else (reports per window, every AE count) is private.
Budget: epsilon = 1, delta = 0, basic sequential composition over two OpenDP 0.16.0 measurements:
  - reports per window: sized count-by-categories over the window labels, discrete Laplace (epsilon 0.05);
  - AE x window histogram: each report contributes at most K (window, AE) tokens, so replacing one report changes
    the token multiset by at most 2K (symmetric distance d_in = 2K); count-by-categories over all
    2 x (public AE terms) cells, discrete Laplace (epsilon 0.95).
Everything after that is post-processing of the noisy counts: ROR, a noise-aware score test (sampling variance
plus the DP noise variance; see noise_aware_test), Bonferroni over the AEs tested, and the Top-100 (noisy
histogram + sort, with the paper's selection rule applied to the noisy counts).

Usage: python reference_dp.py <data_dir> [K]
"""
import json
import sys

import numpy as np
import opendp.prelude as dp
from scipy.stats import norm

import original

dp.enable_features("contrib")
EPS_TOTAL = 1.0
BUDGET = {"window_counts": 0.05, "ae_histogram": 0.95}
assert abs(sum(BUDGET.values()) - EPS_TOTAL) < 1e-12
K = 6  # AEs kept per report: public rule in spec.md (95th pct of AEs per report, FAERS 2013-2018; runs/public_k.txt)
TEST_MIN = 5  # an AE is tested when its noisy 2019 + 2020 count is at least TEST_MIN noise scales (public rule)


def bounded_tokens(reports, domain, k, rng):
    """'window|AE' tokens, at most k per report (a uniform random subset when a report lists more)."""
    tokens = []
    for w, aes in zip(reports.window, reports.aes):
        kept = [a for a in aes.split(";") if a in domain]
        if len(kept) > k:
            kept = rng.choice(kept, k, replace=False)
        tokens += [f"{w}|{a}" for a in kept]
    return tokens


def laplace_after(trans, d_in, eps):
    """trans >> discrete Laplace, with the scale searched by OpenDP so that map(d_in) <= eps; returns (meas, scale)."""
    make = lambda s: trans >> dp.m.then_laplace(s)
    scale = dp.binary_search_param(make, d_in=d_in, d_out=eps)
    return make(scale), scale


def dlap_var(scale):
    """Variance of OpenDP's discrete Laplace with this scale (pmf proportional to exp(-|x| / scale))."""
    if scale == 0:
        return 0.0
    q = np.exp(-1 / scale)
    return 2 * q / (1 - q) ** 2


def noise_aware_test(A19, A20, n19, n20, var_a, var_n):
    """ROR and a two-sided p from a noise-aware score test.

    Under "no change", given the AE's total T = A19 + A20, A20 is hypergeometric with mean pi*T, pi = n20 / n.
    z = (A20 - pi*T) / sd, where sd^2 = sampling variance T*pi*(1-pi)*(n-T)/(n-1) (z^2 is the Pearson chi-square
    of the 2x2 table) + DP noise variance of the numerator: (1-pi)*e20 - pi*e19 from the AE counts and T*d(pi)
    from the window count."""
    n = n19 + n20
    pi, T = n20 / n, A19 + A20
    with np.errstate(divide="ignore", invalid="ignore"):
        ror = (A20 * (n19 - A19)) / ((n20 - A20) * A19)
    var = T * pi * (1 - pi) * (n - T) / (n - 1) + var_a * ((1 - pi) ** 2 + pi ** 2) + var_n * (T / n) ** 2
    return ror, 2 * norm.sf(np.abs(A20 - pi * T) / np.sqrt(np.maximum(var, 1e-12)))


def release(data_dir, k=K, noise=True, rng=None):
    """DP release (noise=False: same pipeline without noise = control for the cost of capping and the test)."""
    rng = rng or np.random.default_rng()
    reports, terms = original.load(data_dir)
    test_terms = terms[terms.drop_before_test == 0]
    ids = list(test_terms.index)
    n = len(reports)  # public
    windows = reports.window.astype(str).tolist()
    tokens = bounded_tokens(reports, set(ids), k, rng)
    cells = [f"{w}|{a}" for w in (2019, 2020) for a in ids]

    spent = {}
    if noise:
        win_space = dp.vector_domain(dp.atom_domain(T=str), size=n), dp.symmetric_distance()
        m_win, scale_win = laplace_after(win_space >> dp.t.then_count_by_categories(categories=["2019", "2020"],
                                                                         null_category=False), 2, BUDGET["window_counts"])
        tok_space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
        m_ae, scale_ae = laplace_after(tok_space >> dp.t.then_count_by_categories(categories=cells, null_category=False),
                             2 * k, BUDGET["ae_histogram"])
        spent = {"window_counts": m_win.map(2), "ae_histogram": m_ae.map(2 * k)}
        for key, e in spent.items():
            assert e <= BUDGET[key] + 1e-12, (key, e)
        assert sum(spent.values()) <= EPS_TOTAL + 1e-9, spent
        n19 = float(np.clip(m_win(windows)[0], 0, n))
        h = np.asarray(m_ae(tokens), dtype=float)
    else:
        scale_win = scale_ae = 0.0
        n19 = float(windows.count("2019"))
        pos = {c: i for i, c in enumerate(cells)}
        h = np.bincount([pos[t] for t in tokens], minlength=len(cells)).astype(float)
    n20 = n - n19
    D = len(ids)
    A19, A20 = np.clip(h[:D], 0, n19), np.clip(h[D:], 0, n20)
    ror, p = noise_aware_test(A19, A20, n19, n20, dlap_var(scale_ae), dlap_var(scale_win))
    tested = (A19 + A20) >= max(TEST_MIN * scale_ae, 1)
    m = int(tested.sum())
    sig = tested & (p <= original.ALPHA / m)

    t = test_terms[["name", "drop_from_table"]].assign(A19=A19, A20=A20, ror=ror, p=p, tested=tested, significant=sig)
    tab = t[tested & (t.drop_from_table == 0)]
    cand = tab[tab[["A19", "A20"]].max(axis=1) >= original.MIN_REPORTS]
    top = cand.sort_values(["p", "ror"], ascending=[True, False]).head(original.TOP_K).sort_values("ror", ascending=False)
    return {"k": k, "n19": n19, "n_tested": m, "table": t, "dp_significant": list(tab.index[tab.significant]),
            "dp_top100": list(top.index), "scale": {"window_counts": scale_win, "ae_histogram": scale_ae},
            "epsilon_spent": spent, "epsilon_total": float(sum(spent.values()))}


def true_values(data_dir):
    """Non-private counts, table, significant AEs and Top-100, for comparison."""
    reports, terms = original.load(data_dir)
    n, A = original.counts(reports)
    t = original.two_by_two(n, A, terms)
    tab = original.published(t)
    return {"n": n, "table": t, "true_significant": list(tab.index[tab.significant]),
            "true_top100": list(original.top100(tab).index)}


if __name__ == "__main__":
    k = int(sys.argv[2]) if len(sys.argv) > 2 else K
    true, out = true_values(sys.argv[1]), release(sys.argv[1], k)
    tt, dt = true["table"], out["table"]
    print(f"K = {k}; reports in 2019 window: true {true['n'][2019]}  DP {out['n19']:.0f}")
    print(f"main table significant AEs: true {len(true['true_significant'])}  DP {len(out['dp_significant'])}  "
          f"both {len(set(true['true_significant']) & set(out['dp_significant']))}")
    print(f"Top-100 overlap {len(set(true['true_top100']) & set(out['dp_top100']))}/100")
    print("Top-100 of the original, true vs DP (2019 count, 2020 count, ROR, significant):")
    for ae in true["true_top100"]:
        a, b = tt.loc[ae], dt.loc[ae]
        print(f"{ae} {a['name'][:36]:36s} true {a.A19:5d} {a.A20:5d} {a.ror:6.2f} {a.significant!s:5s} | "
              f"DP {b.A19:7.0f} {b.A20:7.0f} {b.ror:6.2f} {b.significant!s:5s}")
    print(f"epsilon spent {out['epsilon_spent']}  total {out['epsilon_total']:.4f} (budget {EPS_TOTAL}, delta 0)")
    print(json.dumps({"true_significant": true["true_significant"], "dp_significant": out["dp_significant"],
                      "true_top100": true["true_top100"], "dp_top100": out["dp_top100"],
                      "epsilon_total": out["epsilon_total"]}))
