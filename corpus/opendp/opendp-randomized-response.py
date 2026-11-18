import opendp.prelude as dp

dp.enable_features("contrib")


def dp_share_randomized_response(df, column, epsilon):
    """Local DP: each boolean answer is randomized on its own, then the share of True is de-biased.

    Each person is protected by epsilon for their own report (no composition across people).
    Returns (estimated_true_share, epsilon_spent).
    """
    # Search the truth probability p from epsilon instead of solving a formula by hand.
    prob = dp.binary_search_param(
        lambda p: dp.m.make_randomized_response_bool(prob=p), d_in=1, d_out=epsilon,
        bounds=(0.5 + 1e-9, 1.0 - 1e-9))
    rr = dp.m.make_randomized_response_bool(prob=prob)
    epsilon_spent = rr.map(1)                         # epsilon of one person's report
    noisy = [rr(bool(v)) for v in df[column]]
    observed = sum(noisy) / len(noisy)
    # De-biasing is post-processing; its variance blows up as prob nears 0.5.
    estimate = (observed - (1 - prob)) / (2 * prob - 1)
    return estimate, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"flag": [True, False, False, True, True] * 20})
    _, epsilon_spent = dp_share_randomized_response(df, "flag", epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
