import numpy as np


def rho_for(epsilon, delta):
    """Largest rho whose zCDP -> (eps, delta) conversion (Bun & Steinke Prop. 1.3) stays within epsilon."""
    log_d = np.log(1 / delta)
    return (np.sqrt(log_d + epsilon) - np.sqrt(log_d)) ** 2


def eps_from_rho(rho, delta):
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def dp_gaussian_sums_zcdp(df, columns, bounds, epsilon, delta, relation="replace_one", shares=None):
    """Several clamped sums with Gaussian noise under ONE (eps, delta), accounted in zCDP.

    bounds: {column: (lo, hi)} PUBLIC. shares: fractions of rho per column (default equal).
    Returns ({column: noisy_sum}, epsilon_spent).
    """
    rng = np.random.default_rng()
    rho = rho_for(epsilon, delta)
    shares = shares or [1.0 / len(columns)] * len(columns)
    out, rho_spent = {}, 0.0
    for col, share in zip(columns, shares):
        lo, hi = bounds[col]
        x = np.clip(df[col].to_numpy(dtype=float), lo, hi)
        sens = hi - lo if relation == "replace_one" else max(abs(lo), abs(hi))   # L2 = L1 for a scalar
        sigma = sens / np.sqrt(2 * share * rho)
        out[col] = float(x.sum() + rng.normal(0.0, sigma))
        rho_spent += sens ** 2 / (2 * sigma ** 2)                    # zCDP adds up
    epsilon_spent = eps_from_rho(rho_spent, delta)
    assert epsilon_spent <= epsilon + 1e-9
    return out, float(epsilon_spent)


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 4.0, 2.5, 9.0] * 25, "b": [-3.0, 0.5, 2.0, 1.0] * 25})
    res, eps = dp_gaussian_sums_zcdp(df, ["a", "b"], {"a": (0, 10), "b": (-5, 5)}, 1.0, 1e-6)
    assert set(res) == {"a", "b"}
    return {"epsilon": eps}
