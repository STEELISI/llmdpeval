import numpy as np


def dp_above_threshold(df, queries, threshold, epsilon, sensitivity=1.0):
    """AboveThreshold (Dwork & Roth Alg. 1): index of the first query whose noisy answer passes
    a noisy PUBLIC threshold, or None. queries: list of functions df -> number, each with
    sensitivity <= `sensitivity`. Only the index is released. Returns (index or None, epsilon_spent).
    """
    rng = np.random.default_rng()
    scale_t = 2 * sensitivity / epsilon              # threshold noise: eps / 2
    scale_q = 4 * sensitivity / epsilon              # query noise: eps / 2, any number of "below"s
    t_hat = threshold + rng.laplace(0.0, scale_t)    # drawn ONCE
    epsilon_spent = sensitivity / scale_t + 2 * sensitivity / scale_q
    assert epsilon_spent <= epsilon + 1e-9
    for i, q in enumerate(queries):
        if q(df) + rng.laplace(0.0, scale_q) >= t_hat:
            return i, epsilon_spent                  # halt after the first "above"
    return None, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(300) % 100).astype(float)})
    cutoffs = [90, 70, 50, 30, 10]
    queries = [lambda d, c=c: float(np.sum(d["value"] > c)) for c in cutoffs]   # counts, sensitivity 1
    idx, eps = dp_above_threshold(df, queries, threshold=150, epsilon=1.0)
    assert idx is None or 0 <= idx < len(cutoffs)
    return {"epsilon": eps}
