import warnings

from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_median_with_ledger(df, column, bounds, epsilon, lifetime_epsilon, previous_spends):
    """DP median charged to a lifetime ledger restored from previous_spends (list of (eps, delta)).

    bounds=(lower, upper) are public. Raises BudgetError if the plan does not fit.
    Returns ((private_median, updated_ledger), epsilon_spent_by_this_call).
    """
    acc = BudgetAccountant(epsilon=lifetime_epsilon, delta=0, spent_budget=list(previous_spends))
    planned_total, _ = acc.total(spent_budget=acc.spent_budget + [(epsilon, 0)])  # planning only
    assert planned_total <= lifetime_epsilon, f"plan would reach {planned_total} > {lifetime_epsilon}"
    before = acc.total()[0]
    median = tools.median(df[column].to_numpy(dtype=float), epsilon=epsilon, bounds=bounds, accountant=acc)
    return (float(median), acc.spent_budget), acc.total()[0] - before


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(400) % 50})
    (m, ledger), epsilon_spent = dp_median_with_ledger(df, "value", (0, 60), 0.5, 2.0, [(0.5, 0), (0.25, 0)])
    assert len(ledger) == 3 and 0 <= m <= 60
    return {"epsilon": epsilon_spent}
