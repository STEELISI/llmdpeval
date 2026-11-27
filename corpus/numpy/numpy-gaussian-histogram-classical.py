import numpy as np


def dp_gaussian_histogram(df, column, categories, epsilon, delta, relation="replace_one"):
    """(eps, delta)-DP counts over a PUBLIC category list, classical Gaussian calibration.

    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Valid only for 0 < epsilon < 1 (Dwork & Roth Thm. 3.22). Returns ({category: count}, epsilon_spent).
    """
    if not 0 < epsilon < 1:
        raise ValueError("classical Gaussian calibration needs epsilon < 1; use the analytic one")
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    counts = np.array([np.sum(x == c) for c in categories], dtype=float)   # zeros included
    # L2 sensitivity: 1 (add/remove), sqrt(2) (replace-one: one bin -1, another +1)
    sens2 = 1.0 if relation == "add_remove" else np.sqrt(2.0)
    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = sens2 * c / epsilon                      # standard deviation
    noisy = np.clip(np.rint(counts + rng.normal(0.0, sigma, size=len(categories))), 0, None)
    epsilon_spent = sens2 * c / sigma
    assert epsilon_spent <= epsilon + 1e-9
    return dict(zip(categories, noisy.tolist())), float(epsilon_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 40})
    res, eps = dp_gaussian_histogram(df, "group", ["a", "b", "c", "d"], 0.9, 1e-6, "replace_one")
    assert set(res) == {"a", "b", "c", "d"}
    return {"epsilon": eps}
