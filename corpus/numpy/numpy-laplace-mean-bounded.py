import numpy as np


def dp_mean(df, column, lower, upper, epsilon, relation="replace_one", public_n=None, sum_share=0.5):
    """DP mean of a column clamped to PUBLIC bounds [lower, upper].

    replace_one (default; N public by design, passed as public_n): noise the mean directly.
    add_remove (alternative, N private): noisy sum / noisy count, epsilon split sum_share : 1 - sum_share.
    Returns (private_mean, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = np.clip(df[column].to_numpy(dtype=float), lower, upper)
    if relation == "add_remove":
        sens_sum = max(abs(lower), abs(upper))              # sum sensitivity under add/remove
        scale_sum = sens_sum / (sum_share * epsilon)
        scale_cnt = 1.0 / ((1 - sum_share) * epsilon)       # count sensitivity 1
        noisy_sum = np.sum(x) + rng.laplace(0.0, scale_sum)
        noisy_cnt = len(x) + rng.laplace(0.0, scale_cnt)
        private_mean = noisy_sum / max(noisy_cnt, 1.0)      # post-processing
        epsilon_spent = sens_sum / scale_sum + 1.0 / scale_cnt   # sequential composition
    else:
        assert public_n is not None and public_n == len(x), "replace-one needs a public N"
        sensitivity = (upper - lower) / public_n            # one value moves from L to U
        scale = sensitivity / epsilon
        private_mean = np.sum(x) / public_n + rng.laplace(0.0, scale)
        epsilon_spent = sensitivity / scale
    assert epsilon_spent <= epsilon + 1e-9
    return float(np.clip(private_mean, lower, upper)), epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [3.0, 7.5, -2.0, 12.0, 5.0] * 20})
    m1, e1 = dp_mean(df, "value", -5.0, 5.0, epsilon=1.0, public_n=len(df))   # bounds straddle 0
    m2, e2 = dp_mean(df, "value", -5.0, 5.0, epsilon=1.0, relation="add_remove")
    assert -5.0 <= m1 <= 5.0 and -5.0 <= m2 <= 5.0
    return {"epsilon": float(max(e1, e2))}
