import opendp.prelude as dp

dp.enable_features("contrib")


def dp_sparse_key_counts(df, column, keys_to_ask, total_limit, value_limit, epsilon):
    """Approximate Laplace Projection: release a DP hashed sketch of counts over an unknown,
    possibly huge key space, then ask it about any keys. Pure epsilon-DP; no key list and no delta.
    Replace-one neighbours (bounded DP, n public): d_in = 2.

    total_limit: public upper bound on the number of rows; value_limit: public cap on any one count.
    Returns ({key: noisy count}, epsilon_spent).
    """
    space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
    counts = space >> dp.t.then_count_by()          # {key: count}, metric L01Inf
    meas = dp.binary_search_chain(
        lambda s: counts >> dp.m.then_alp_queryable(scale=s, total_limit=total_limit, value_limit=value_limit),
        d_in=2, d_out=epsilon)
    sketch = meas(df[column].astype(str).tolist())  # a queryable; asking it costs nothing more
    answers = {k: sketch(k) for k in keys_to_ask}   # unseen keys usually come back near 0
    return answers, meas.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"name": ["p"] * 60 + ["q"] * 30 + ["r"] * 3})
    _, epsilon_spent = dp_sparse_key_counts(df, "name", ["p", "q", "zzz"], total_limit=1000,
                                            value_limit=200, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
