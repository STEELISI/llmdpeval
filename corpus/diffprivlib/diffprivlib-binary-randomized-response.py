import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_randomized_response_share(df, column, positive_label, negative_label, epsilon):
    """Local-DP randomised response on a two-valued string column, then a debiased share of `positive_label`.

    Each row is randomised once, so every person has epsilon-DP. Returns ((noisy_column, share_estimate), epsilon_spent).
    """
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    mech = mechanisms.Binary(epsilon=epsilon, value0=negative_label, value1=positive_label)
    noisy = np.array([mech.randomise(str(v)) for v in df[column]])
    p_keep = np.exp(epsilon) / (1 + np.exp(epsilon))  # probability the mechanism keeps the true label
    observed = np.mean(noisy == positive_label)
    estimate = float(np.clip((observed - (1 - p_keep)) / (2 * p_keep - 1), 0, 1))  # post-processing
    # Mechanisms do not charge an accountant; each row is touched once, so charge epsilon once.
    acc.spend(epsilon, 0)
    return (noisy, estimate), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"answer": np.where(np.arange(1000) % 4 == 0, "yes", "no")})
    (noisy, share), epsilon_spent = dp_randomized_response_share(df, "answer", "yes", "no", epsilon=1.0)
    assert len(noisy) == 1000 and 0 <= share <= 1
    return {"epsilon": epsilon_spent}
