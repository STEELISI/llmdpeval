import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms.transforms import IntToString


def dp_flip_binary_codes(df, column, epsilon):
    """Local-DP randomised response on a 0/1 integer column, via Binary wrapped in IntToString.

    Binary only accepts string labels; IntToString converts int -> str -> int around it.
    Returns (noisy integer codes, epsilon_spent). Each row is randomised once.
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = IntToString(mechanisms.Binary(epsilon=epsilon, value0="0", value1="1"))
    noisy = np.array([mech.randomise(int(v)) for v in df[column]])
    acc.spend(epsilon, 0)  # per person: one randomise call each
    return noisy, acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"flag": np.arange(300) % 2})
    noisy, epsilon_spent = dp_flip_binary_codes(df, "flag", epsilon=1.0)
    assert set(np.unique(noisy)) <= {0, 1}
    return {"epsilon": epsilon_spent}
