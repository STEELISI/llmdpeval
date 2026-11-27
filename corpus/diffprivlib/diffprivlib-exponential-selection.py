from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_select_most_common(df, column, candidates, epsilon, sensitivity=1):
    """Pick the candidate value of df[column] with the highest row count, via the exponential mechanism.

    candidates: public list in a fixed order, never the column's unique values. The utility is a count,
    so one person changes any score by at most sensitivity (1). Returns (chosen_candidate, epsilon_spent).
    """
    counts = df[column].value_counts()
    utility = [float(counts.get(c, 0)) for c in candidates]  # plain list, same order as candidates
    # mechanisms.PermuteAndFlip takes the same arguments and is never worse in expected utility.
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.Exponential(epsilon=epsilon, sensitivity=sensitivity, utility=utility,
                                  candidates=list(candidates))
    choice = mech.randomise()
    acc.spend(epsilon, 0)  # mechanisms do not charge an accountant themselves
    return choice, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 20})
    choice, epsilon_spent = dp_select_most_common(df, "group", ["a", "b", "c", "d"], epsilon=1.0)
    assert choice in ["a", "b", "c", "d"]
    return {"epsilon": epsilon_spent}
