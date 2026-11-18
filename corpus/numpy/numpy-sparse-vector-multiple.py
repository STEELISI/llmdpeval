import numpy as np


def dp_sparse_vector(df, queries, threshold, c, epsilon, sensitivity=1.0, threshold_share=0.5,
                     monotone=False):
    """SVT with at most c positives (Lyu, Su & Li 2017, Alg. 1). queries: functions df -> number,
    each with sensitivity <= `sensitivity`; PUBLIC threshold. Returns ([indices above], epsilon_spent).
    """
    rng = np.random.default_rng()
    eps1, eps2 = threshold_share * epsilon, (1 - threshold_share) * epsilon
    scale_rho = sensitivity / eps1
    factor = 1.0 if monotone else 2.0                # monotone queries: c*Δ/eps2; general: 2c*Δ/eps2
    scale_nu = factor * c * sensitivity / eps2
    rho = rng.laplace(0.0, scale_rho)                # threshold noise, drawn ONCE
    above = []
    for i, q in enumerate(queries):
        if q(df) + rng.laplace(0.0, scale_nu) >= threshold + rho:
            above.append(i)
            if len(above) == c:
                break                                # stop after c positives
    epsilon_spent = sensitivity / scale_rho + factor * c * sensitivity / scale_nu
    assert epsilon_spent <= epsilon + 1e-9
    return above, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": (np.arange(300) % 100).astype(float)})
    queries = [lambda d, k=k: float(np.sum(d["value"] > k)) for k in range(95, 0, -10)]
    above, eps = dp_sparse_vector(df, queries, threshold=120, c=3, epsilon=1.0)
    assert len(above) <= 3
    return {"epsilon": eps}
