import opendp.prelude as dp

dp.enable_features("contrib")   # no "honest-but-curious": no user-defined transformation is needed


def dp_most_frequent(df, column, candidates, epsilon):
    """Pick the public candidate that occurs most often in a column, with pure DP, replace-one
    neighbours (bounded DP, n public).

    candidates is a public list in a fixed order. Returns (winning_candidate, epsilon_spent).
    """
    # Score vector built outside OpenDP over the PUBLIC list (absent candidates score 0).
    counts = df[column].astype(str).value_counts()
    scores = [int(counts.get(str(c), 0)) for c in candidates]   # str(c): match the str-cast column
    # Replace-one: one count goes down by 1 and another goes up by 1, so the scores move by at most 1
    # in L-infinity, in different directions (not monotonic). d_in = 1 is this stated bound.
    space = (dp.vector_domain(dp.atom_domain(T=int)), dp.linf_distance(T=int))
    meas = dp.binary_search_chain(
        lambda s: space >> dp.m.then_noisy_max(dp.max_divergence(), scale=s), d_in=1, d_out=epsilon)
    epsilon_spent = meas.map(1)
    index = meas(scores)
    return candidates[index], epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a"] * 50 + ["b"] * 20 + ["c"] * 5})
    winner, epsilon_spent = dp_most_frequent(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert winner in ["a", "b", "c", "d"]
    return {"epsilon": float(epsilon_spent)}
