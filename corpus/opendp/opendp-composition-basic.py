import opendp.prelude as dp

dp.enable_features("contrib")


def dp_count_and_sum(df, column, bounds, epsilon, weights=(0.5, 0.5)):
    """Two non-adaptive queries (count, clamped sum) under one total epsilon.

    Replace-one neighbours (n public): sized domain, d_in = 2 (add/remove, n private: unsized, d_in = 1).
    weights is a public split of the budget. Returns ([count, sum], epsilon_spent).
    """
    values = df[column].astype(float).tolist()
    d_in = 2                                           # one changed row = symmetric distance 2
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=len(values)), dp.symmetric_distance()
    shares = [epsilon * w / sum(weights) for w in weights]
    pres = [
        space >> dp.t.then_count(),                    # charged 2 at d_in=2; filtered count: 0/1 sum (1)
        space >> dp.t.then_clamp(bounds) >> dp.t.then_sum(),
    ]
    # Each query searches its scale against its own share; all start from the same space.
    parts = [dp.binary_search_chain(lambda s, p=p: p >> dp.m.then_laplace(s), d_in=d_in, d_out=e)
             for p, e in zip(pres, shares)]
    comp = dp.c.make_composition(parts)
    epsilon_spent = comp.map(d_in)                     # basic composition: the sum of the parts
    assert epsilon_spent <= epsilon
    answers = comp(values)
    return answers, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(100)]})
    _, epsilon_spent = dp_count_and_sum(df, "value", (-5.0, 5.0), epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
