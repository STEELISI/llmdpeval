import numpy as np


def advanced_epsilon(eps0, k, delta):
    """Dwork-Rothblum-Vadhan bound (Dwork & Roth Thm. 3.20) for k eps0-DP mechanisms at delta."""
    return np.sqrt(2 * k * np.log(1 / delta)) * eps0 + k * eps0 * (np.exp(eps0) - 1)


def dp_many_counts(df, predicates, epsilon, delta):
    """k Laplace counts (sensitivity 1 each) under a total (epsilon, delta).

    predicates: list of functions df -> boolean array. Per-query eps0 = max(eps / k, advanced eps0).
    Returns ([noisy counts], epsilon_spent).
    """
    rng = np.random.default_rng()
    k = len(predicates)
    lo, hi = 0.0, epsilon
    for _ in range(100):                             # largest eps0 with advanced bound <= epsilon
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if advanced_epsilon(mid, k, delta) <= epsilon else (lo, mid)
    eps0 = max(epsilon / k, lo)
    scale = 1.0 / eps0                               # count sensitivity 1
    answers = [float(np.sum(p(df)) + rng.laplace(0.0, scale)) for p in predicates]
    eps_each = 1.0 / scale
    epsilon_spent = float(min(k * eps_each, advanced_epsilon(eps_each, k, delta)))
    assert epsilon_spent <= epsilon + 1e-9
    return answers, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(500) % 97).astype(float)})
    preds = [lambda d, c=c: d["value"].to_numpy() > c for c in range(0, 97, 2)]   # 49 queries
    answers, eps = dp_many_counts(df, preds, epsilon=1.0, delta=1e-6)
    assert len(answers) == len(preds)
    return {"epsilon": eps}
