import numpy as np


def _discrete_gaussian(rng, sigma):
    """One draw from the discrete Gaussian N_Z(0, sigma^2), Canonne-Kamath-Steinke Alg. 3 (float coins)."""
    t = np.floor(sigma) + 1
    p = 1 - np.exp(-1 / t)
    while True:
        y = int(rng.geometric(p)) - int(rng.geometric(p))      # discrete Laplace(t)
        if rng.random() < np.exp(-((abs(y) - sigma ** 2 / t) ** 2) / (2 * sigma ** 2)):
            return y


def dp_discrete_gaussian_histogram(df, column, categories, epsilon, delta, relation="replace_one"):
    """Integer counts over a PUBLIC category list + discrete Gaussian noise; rho-zCDP -> (eps, delta).

    relation: "replace_one" (bounded DP, n public; default) or "add_remove".
    Returns ({category: noisy_int_count}, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = df[column].to_numpy()
    counts = [int(np.sum(x == c)) for c in categories]          # zeros included
    sens2 = 1.0 if relation == "add_remove" else np.sqrt(2.0)   # L2 sensitivity of the count vector
    log_d = np.log(1 / delta)
    rho = (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2
    sigma = sens2 / np.sqrt(2 * rho)
    noisy = [c + _discrete_gaussian(rng, sigma) for c in counts]
    rho_spent = sens2 ** 2 / (2 * sigma ** 2)
    epsilon_spent = rho_spent + 2 * np.sqrt(rho_spent * log_d)
    assert epsilon_spent <= epsilon + 1e-9
    return dict(zip(categories, noisy)), float(epsilon_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"group": ["a", "b", "a", "c", "a"] * 40})
    res, eps = dp_discrete_gaussian_histogram(df, "group", ["a", "b", "c", "d"], 1.0, 1e-6)
    assert all(isinstance(v, int) for v in res.values())
    return {"epsilon": eps}
