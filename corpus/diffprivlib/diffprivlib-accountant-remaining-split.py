import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_first_then_rest(df, first_column, other_columns, bounds_by_column, epsilon, first_share=0.4):
    """One DP mean on first_column, then the remaining budget split evenly (acc.remaining(k)) over DP means
    of other_columns. bounds_by_column: {column: (lower, upper)}, all public.

    Returns (dict column -> private mean, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = lambda c: df[c].to_numpy(dtype=float)
    out = {first_column: float(tools.mean(values(first_column), epsilon=first_share * epsilon,
                                          bounds=bounds_by_column[first_column], accountant=acc))}
    eps_each, _ = acc.remaining(len(other_columns))  # per-query budget for k more queries
    for c in other_columns:
        e = min(eps_each, acc.remaining()[0])  # float rounding: the last query takes exactly what is left
        out[c] = float(tools.mean(values(c), epsilon=e, bounds=bounds_by_column[c], accountant=acc))
    return out, acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    a = np.arange(300) % 30
    df = pd.DataFrame({"a": a, "b": a + 5.0, "c": 30 - a})
    bounds = {"a": (0, 30), "b": (0, 40), "c": (0, 30)}
    out, epsilon_spent = dp_first_then_rest(df, "a", ["b", "c"], bounds, epsilon=1.0)
    assert len(out) == 3
    return {"epsilon": epsilon_spent}
