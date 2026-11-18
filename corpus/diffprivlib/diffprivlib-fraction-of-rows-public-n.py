from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import count_nonzero


def dp_fraction_matching(indicators, n_public, epsilon):
    """DP share of rows with indicator 1 (support of a predicate, accuracy as pred == label).

    Replace-one neighbours with a PUBLIC n; count_nonzero is a bounded (0, 1) sum, sensitivity 1.
    Returns (private_fraction, epsilon_spent).
    """
    assert len(indicators) == n_public, "n must be public and match the data"
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    noisy = count_nonzero(indicators, epsilon=epsilon, accountant=acc)
    epsilon_spent = acc.total()[0]
    assert epsilon_spent <= epsilon + 1e-9
    return min(max(noisy / n_public, 0.0), 1.0), epsilon_spent


def _test():
    import numpy as np
    pred, label = np.array([1, 0, 1, 1] * 50), np.array([1, 0, 0, 1] * 50)
    acc, eps = dp_fraction_matching((pred == label).astype(int), len(pred), epsilon=1.0)
    assert 0.0 <= acc <= 1.0
    return {"epsilon": float(eps)}
