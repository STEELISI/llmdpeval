from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_top_k(df, column, candidates, k, epsilon):
    """Top-k public candidates by row count, by peeling: k exponential-mechanism rounds of epsilon / k.

    candidates: the PUBLIC list of every possible value, declared before looking at the data
    (codebook, schema), in a fixed order. Never df[column].unique(): which values occur is private.
    Returns (list of k picks, epsilon_spent).
    """
    sensitivity = 1.0   # one row changes one count by 1
    counts = df[column].value_counts()
    utility = [float(counts.get(c, 0)) for c in candidates]   # absent candidates score 0
    left = list(candidates)
    eps_round = epsilon / k  # basic sequential composition over the k rounds
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    chosen = []
    for _ in range(k):
        e = min(eps_round, acc.remaining()[0])  # k * (epsilon / k) can round above the cap: take what is left
        mech = mechanisms.Exponential(epsilon=e, sensitivity=sensitivity, utility=utility, candidates=left)
        winner = mech.randomise()
        acc.spend(e, 0)  # mechanisms do not charge an accountant themselves
        chosen.append(winner)
        i = left.index(winner)
        del left[i], utility[i]
    return chosen, acc.total()[0]


def _test():
    import pandas as pd
    CANDIDATES = ["a", "b", "c", "d", "e"]   # public list, declared up front
    df = pd.DataFrame({"group": list("aaaabbbccd") * 20})
    top, eps = dp_top_k(df, "group", CANDIDATES, k=3, epsilon=1.0)
    assert len(set(top)) == 3
    return {"epsilon": eps}
