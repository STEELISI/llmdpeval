import opendp.prelude as dp

dp.enable_features("contrib")


def dp_bounded_sum(df, column, bounds, epsilon, n_is_public=True):
    """DP sum of a float column clamped to public bounds.

    Default, n_is_public=True: replace-one neighbours (bounded DP), sized domain, d_in = 2,
        charged U - L.
    n_is_public=False: add/remove-one neighbours (n private), unsized domain, d_in = 1,
        charged max(|L|, |U|, U - L).
    Returns (private_sum, epsilon_spent).
    """
    values = df[column].astype(float).tolist()          # must contain no NaN (nan=False domain)
    atom = dp.atom_domain(T=float, nan=False)
    domain = dp.vector_domain(atom, size=len(values)) if n_is_public else dp.vector_domain(atom)
    d_in = 2 if n_is_public else 1
    pre = (domain, dp.symmetric_distance()) >> dp.t.then_clamp(bounds) >> dp.t.then_sum()
    print("sum sensitivity OpenDP charges:", pre.map(d_in))   # includes float rounding terms

    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon)
    epsilon_spent = meas.map(d_in)
    assert epsilon_spent <= epsilon
    private_sum = meas(values)
    return private_sum, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, eps_sized = dp_bounded_sum(df, "value", (-5.0, 5.0), epsilon=1.0)
    _, eps_unsized = dp_bounded_sum(df, "value", (-5.0, 5.0), epsilon=1.0, n_is_public=False)
    return {"epsilon": float(max(eps_unsized, eps_sized))}
