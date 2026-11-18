import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning
from diffprivlib.validation import check_epsilon_delta

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_planned_mean_var(df, column, bounds, epsilon, shares):
    """DP mean and variance of df[column] under a plan shares={"mean": s1, "var": s2}, validated first.

    bounds=(lower, upper) are public. Returns (dict of statistics, epsilon_spent).
    """
    plan = {q: (s * epsilon, 0.0) for q, s in shares.items()}
    for e, d in plan.values():
        check_epsilon_delta(e, d)  # one pair at a time: type and range only
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    planned_total, _ = acc.total(spent_budget=list(plan.values()))  # composition check, spends nothing
    assert planned_total <= epsilon * (1 + 1e-9), f"plan spends {planned_total} > {epsilon}"
    values = df[column].to_numpy(dtype=float)
    stats = {"mean": float(tools.mean(values, epsilon=plan["mean"][0], bounds=bounds, accountant=acc)),
             "var": float(tools.var(values, epsilon=plan["var"][0], bounds=bounds, accountant=acc))}
    return stats, acc.total()[0]


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(300) % 25})
    stats, epsilon_spent = dp_planned_mean_var(df, "value", (0, 30), 1.0, {"mean": 0.5, "var": 0.5})
    assert stats["var"] >= 0
    return {"epsilon": epsilon_spent}
