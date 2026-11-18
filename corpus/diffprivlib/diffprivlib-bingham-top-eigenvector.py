import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.validation import clip_to_norm


def dp_top_direction(df, columns, lower_bounds, upper_bounds, epsilon):
    """DP first principal direction via the Bingham mechanism. Bounds (one per feature) are public.

    Features are rescaled to [-1, 1] around the public midpoints, so rows have L2 norm <= sqrt(d) = R.
    Returns (unit vector, epsilon_spent).
    """
    lower, upper = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    X = df[list(columns)].to_numpy(dtype=float)
    X = (np.clip(X, lower, upper) - (lower + upper) / 2) / ((upper - lower) / 2)  # public centring and scaling
    norm_bound = np.sqrt(X.shape[1])
    X = clip_to_norm(X, norm_bound)
    second_moment = X.T @ X / norm_bound ** 2  # one row changes this by norm <= 1: sensitivity 1, as in diffprivlib
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.check(epsilon, 0)  # before the release: refuse an overspend
    v = mechanisms.Bingham(epsilon=epsilon, sensitivity=1.0).randomise(second_moment)
    v = v if v[np.argmax(np.abs(v))] >= 0 else -v  # fix the arbitrary sign (post-processing)
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return v, acc.total()[0]


def _test():
    import pandas as pd
    a = np.linspace(-5, 5, 500)
    df = pd.DataFrame({"x": a, "y": a * 0.9 + np.sin(a), "z": np.cos(a)})
    v, epsilon_spent = dp_top_direction(df, ["x", "y", "z"], [-5, -6, -1], [5, 6, 1], epsilon=1.0)
    assert v.shape == (3,) and np.isclose(np.linalg.norm(v), 1)
    return {"epsilon": epsilon_spent}
