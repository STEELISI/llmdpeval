import math

import opendp.prelude as dp

dp.enable_features("contrib")   # the binning is a public row-wise map done before OpenDP: no user transformation


def dp_estimate_bounds(df, column, max_exp, epsilon, alpha=1e-3):
    """Coarse DP clamp bounds from a noisy histogram of signed power-of-two bands, replace-one (n public).

    Use only when no public bounds exist; fund the main query from a SEPARATE epsilon and
    compose both. Magnitudes are capped to 2**-max_exp .. 2**max_exp (max_exp is public).
    Returns ((lower, upper), epsilon_spent).
    """
    def band(x):          # 0 for zero; +-(e + max_exp + 1) for |x| in [2**e, 2**(e+1))
        if x == 0:
            return 0
        e = max(-max_exp, min(max_exp - 1, math.floor(math.log2(abs(x)))))
        return int(math.copysign(e + max_exp + 1, x))

    all_bands = list(range(-2 * max_exp, 2 * max_exp + 1))   # full public band list, not observed
    # Each row maps to exactly one band by a public rule, so this map does not change the distance.
    bands = [band(x) for x in df[column].astype(float).fillna(0.0)]
    space = (dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance())
    hist = space >> dp.t.then_count_by_categories(categories=all_bands, null_category=False)
    # Replace-one = symmetric distance 2 (one band loses a row, another gains one).
    scale = dp.binary_search_param(lambda s: hist >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    meas = hist >> dp.m.then_laplace(scale)
    epsilon_spent = meas.map(2)
    counts = meas(bands)
    # Post-processing: keep bands that pure noise passes only with probability ~alpha overall.
    cut = dp.discrete_laplacian_scale_to_accuracy(scale=scale, alpha=alpha / len(all_bands))
    hit = [b for b, c in zip(all_bands, counts) if c > cut] or [0]   # empty: caller falls back

    def edges(b):         # (left, right) edge of band b
        if b == 0:
            return 0.0, 0.0
        inner, outer = 2.0 ** (abs(b) - max_exp - 1), 2.0 ** (abs(b) - max_exp)
        return (inner, outer) if b > 0 else (-outer, -inner)

    return (edges(min(hit))[0], edges(max(hit))[1]), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 100) for i in range(1000)]})
    (lo, hi), epsilon_spent = dp_estimate_bounds(df, "value", max_exp=20, epsilon=1.0)
    assert lo <= hi
    return {"epsilon": float(epsilon_spent)}
