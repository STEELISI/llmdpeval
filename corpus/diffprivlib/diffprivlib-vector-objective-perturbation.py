import numpy as np
from diffprivlib import mechanisms
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.validation import clip_to_norm
from scipy.optimize import minimize


def dp_custom_logistic(df, feature_columns, label_column, norm_bound, epsilon, ridge=0.01):
    """Objective perturbation with mechanisms.Vector on an L2-regularised logistic loss (|l''| <= 1/4).

    norm_bound is a public bound on each row's L2 norm, at most 1: Vector's slack uses the norm q where the
    proof (CMS 2011, Thm 9) needs q^2, so q > 1 under-charges epsilon. Rescale rows publicly first.
    Labels must be 0/1. Returns (coefficients, epsilon_spent).
    """
    assert 0 < norm_bound <= 1, "rescale rows to norm <= 1: Vector under-charges epsilon for norms above 1"
    X = clip_to_norm(df[list(feature_columns)].to_numpy(dtype=float), norm_bound)
    y = np.where(df[label_column].to_numpy() > 0, 1.0, -1.0)
    n, d = X.shape
    alpha = n * ridge  # Vector's alpha is the ridge on the SUM of losses (LogisticRegression: 1/C)
    slack = 2 * np.log(1 + 0.25 * norm_bound / alpha)
    assert epsilon > slack, "epsilon too small for this ridge/norm: Vector would silently change the guarantee"

    def objective(w):
        m = y * (X @ w)
        value = np.mean(np.logaddexp(0, -m)) + ridge / 2 * w @ w
        grad = -(X * (y / (1 + np.exp(m)))[:, None]).mean(axis=0) + ridge * w
        return value, grad

    mech = mechanisms.Vector(epsilon=epsilon, function_sensitivity=0.25, data_sensitivity=norm_bound,
                             dimension=d, alpha=alpha, n=n)
    w = minimize(mech.randomise(objective), np.zeros(d), jac=True).x
    acc = BudgetAccountant(epsilon=epsilon, delta=0)
    acc.spend(epsilon, 0)  # mechanisms never charge an accountant themselves
    return w, acc.total()[0]


def _test():
    import pandas as pd
    a = np.linspace(-1, 1, 400)
    df = pd.DataFrame({"f1": a * 0.6, "f2": -a * 0.6, "label": (a > 0).astype(int)})
    w, epsilon_spent = dp_custom_logistic(df, ["f1", "f2"], "label", norm_bound=1.0, epsilon=1.0)
    assert w.shape == (2,) and np.all(np.isfinite(w))
    return {"epsilon": epsilon_spent}
