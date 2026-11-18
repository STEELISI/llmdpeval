import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_where(df, column, value, epsilon, n_is_public=True):
    """DP count of rows where df[column] == value, as a sum of 0/1 flags.

    Default: replace-one neighbours (n public): sized domain, d_in = 2, sensitivity 1.
    n_is_public=False: add/remove-one neighbours (n private): unsized domain, d_in = 1, sensitivity 1.
    Returns (private_count, epsilon_spent).
    """
    flags = (df[column].astype(str) == str(value)).astype(int).tolist()   # row-wise, no budget
    atom = dp.atom_domain(T=int)
    domain = dp.vector_domain(atom, size=len(flags)) if n_is_public else dp.vector_domain(atom)
    d_in = 2 if n_is_public else 1
    pre = (domain, dp.symmetric_distance()) >> dp.t.then_clamp((0, 1)) >> dp.t.then_sum()
    meas = dp.binary_search_chain(lambda s: pre >> dp.m.then_laplace(s), d_in=d_in, d_out=epsilon)
    # Read epsilon back from OpenDP's own accounting, not from hand arithmetic.
    epsilon_spent = meas.map(d_in)
    assert epsilon_spent <= epsilon
    private_count = meas(flags)                          # integer: discrete Laplace noise
    return private_count, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    c1, eps_r1 = dp_count_where(df, "group", "a", epsilon=1.0)
    c2, eps_ar = dp_count_where(df, "group", "a", epsilon=1.0, n_is_public=False)
    assert isinstance(c1, int) and isinstance(c2, int)
    return {"epsilon": float(max(eps_r1, eps_ar))}
