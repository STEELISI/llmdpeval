import warnings

import numpy as np
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning

warnings.simplefilter("error", PrivacyLeakWarning)  # a missing bound must fail, never silently read the data


def dp_quantiles(df, column, quantiles, bounds, epsilon):
    """DP quantiles of df[column]; quantiles is a list of values in [0, 1] (use tools.median for 0.5 alone).

    bounds=(lower, upper) are public; every output lies inside them. Replace-one (n public): the rank utility
    has sensitivity 1 under both relations, so no halving. Equal shares, one call per quantile, each clamped to
    what is left (k shares of epsilon/k can round above the cap). Returns (array of private quantiles, epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    values = df[column].to_numpy(dtype=float)
    share = epsilon / len(quantiles)
    private_q = [tools.quantile(values, quant=q, epsilon=min(share, acc.remaining()[0]), bounds=bounds,
                                accountant=acc) for q in quantiles]
    epsilon_spent = acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return np.asarray(private_q), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(300) % 80 - 40})
    for k in (3, 9, 11, 18, 20, 40):   # these k used to round above a capped accountant
        private_q, epsilon_spent = dp_quantiles(df, "value", list(np.linspace(0.02, 0.98, k)), (-50, 50), 1.0)
        assert private_q.shape == (k,) and ((-50 <= private_q) & (private_q <= 50)).all()
    return {"epsilon": epsilon_spent}
