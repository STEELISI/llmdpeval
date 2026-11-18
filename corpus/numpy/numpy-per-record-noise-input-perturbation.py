import numpy as np


def dp_perturb_columns(df, columns, bounds, epsilon):
    """Local DP: a noisy copy of numeric columns, one Laplace draw per record and column.

    bounds: {column: (lower, upper)}, public. Each record's released row is epsilon-DP.
    Returns (noisy DataFrame, epsilon_spent per record).
    """
    rng = np.random.default_rng()
    out = df[columns].astype(float).copy()
    eps_col = epsilon / len(columns)                     # a record releases D values: split epsilon
    epsilon_spent = 0.0
    for col in columns:
        lower, upper = bounds[col]
        x = np.clip(out[col].to_numpy(), lower, upper)
        sens = upper - lower                             # per-record sensitivity: the whole range
        scale = sens / eps_col
        noisy = x + rng.laplace(0.0, scale, size=len(x))
        out[col] = np.clip(noisy, lower, upper)          # post-processing
        epsilon_spent += sens / scale                    # sequential composition within a record
    assert epsilon_spent <= epsilon + 1e-9
    return out, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, 4.0, 2.5, 9.0] * 25, "b": [-3.0, 0.5, 2.0, 1.0] * 25})
    noisy, epsilon_spent = dp_perturb_columns(df, ["a", "b"], {"a": (0.0, 10.0), "b": (-5.0, 5.0)},
                                              epsilon=1.0)
    assert noisy.shape == (100, 2)
    return {"epsilon": epsilon_spent}
