import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant


def dp_sum_with_planned_epsilon(df, column, bounds, max_noise_std, epsilon_budget, grid=20):
    """Pick the smallest epsilon <= epsilon_budget whose Laplace noise std meets max_noise_std (no data used),
    then release a clipped sum once. bounds=(lower, upper) are public.

    Returns ((private_sum, chosen_epsilon, predicted_std), epsilon_spent).
    """
    lower, upper = bounds
    sensitivity = upper - lower  # replace-one: one clipped value moves the sum by at most upper - lower
    chosen = epsilon_budget
    for e in np.linspace(epsilon_budget / grid, epsilon_budget, grid):  # planning only: no data touched
        if np.sqrt(mechanisms.Laplace(epsilon=e, sensitivity=sensitivity).variance(0.0)) <= max_noise_std:
            chosen = float(e)
            break
    mech = mechanisms.Laplace(epsilon=chosen, sensitivity=sensitivity)
    true_sum = float(np.clip(df[column].to_numpy(dtype=float), lower, upper).sum())
    private_sum = mech.randomise(true_sum)  # the only data access
    acc = BudgetAccountant(epsilon=epsilon_budget, delta=0)
    acc.spend(chosen, 0)
    return (private_sum, chosen, float(np.sqrt(mech.variance(0.0)))), acc.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": np.arange(200) % 10})
    (s, e, std), epsilon_spent = dp_sum_with_planned_epsilon(df, "value", (0, 10), 20.0, epsilon_budget=1.0)
    assert 0 < e <= 1 and std <= 20.0
    return {"epsilon": epsilon_spent}
