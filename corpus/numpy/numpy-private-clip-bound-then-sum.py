import numpy as np


def dp_sum_private_bound(df, column, bound_grid, epsilon, q=0.95, bound_share=0.2):
    """DP sum of a NON-NEGATIVE column whose clipping bound is chosen privately.

    Step 1 (bound_share * eps): exponential mechanism picks C from a PUBLIC grid near the q-quantile.
    Step 2 (rest): clamp to [0, C] and add Laplace noise with scale C / eps_sum.
    Returns ({"sum", "clip_bound"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    x = np.clip(df[column].to_numpy(dtype=float), 0.0, None)
    grid = np.asarray(bound_grid, dtype=float)
    eps_b, eps_s = bound_share * epsilon, (1 - bound_share) * epsilon
    below = np.searchsorted(np.sort(x), grid, side="right")
    logits = eps_b * -np.abs(below - q * len(x)) / 2.0     # utility sensitivity 1
    p = np.exp(logits - logits.max())
    clip_c = float(grid[rng.choice(len(grid), p=p / p.sum())])
    sens = clip_c                                           # values in [0, C]: add/remove and replace-one
    scale = sens / eps_s
    noisy_sum = float(np.clip(x, 0.0, clip_c).sum() + rng.laplace(0.0, scale))
    epsilon_spent = eps_b + sens / scale                    # sequential composition
    assert epsilon_spent <= epsilon + 1e-9
    return {"sum": noisy_sum, "clip_bound": clip_c}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"amount": np.abs(np.arange(300) % 50 - 7.0) * 3})
    res, eps = dp_sum_private_bound(df, "amount", 2.0 ** np.arange(0, 12), epsilon=1.0)
    assert res["clip_bound"] > 0
    return {"epsilon": eps}
