import numpy as np


def dp_randomized_response(df, column, positive_value, epsilon):
    """Local eps-DP randomized response on a yes/no attribute, plus an unbiased share estimate.

    Returns ({"reported": 0/1 array, "share", "std_error"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    b = (df[column].to_numpy() == positive_value).astype(int)
    p = np.exp(epsilon) / (1 + np.exp(epsilon))      # probability of reporting the true bit
    keep = rng.random(len(b)) < p
    reported = np.where(keep, b, 1 - b)              # this column is the release
    m = reported.mean()
    share = float(np.clip((m - (1 - p)) / (2 * p - 1), 0.0, 1.0))   # post-processing
    std_error = float(np.sqrt(m * (1 - m) / len(b)) / (2 * p - 1))
    epsilon_spent = float(np.log(p / (1 - p)))       # worst-case likelihood ratio per record
    assert epsilon_spent <= epsilon + 1e-9
    return {"reported": reported, "share": share, "std_error": std_error}, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"answer": ["yes", "no", "no", "yes", "no"] * 100})
    res, eps = dp_randomized_response(df, "answer", "yes", epsilon=1.0)
    assert 0.0 <= res["share"] <= 1.0
    return {"epsilon": eps}
