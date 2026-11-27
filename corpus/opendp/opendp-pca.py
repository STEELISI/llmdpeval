import opendp.prelude as dp
from opendp.extras.sklearn.decomposition import PCA

# As in the OpenDP docs for the PCA wrapper. Needs scikit-learn and randomgen installed.
# Both extra flags are required by the wrapper (it raises without them): user transformations need
# "honest-but-curious"; "idealized-numerics" treats its float linear algebra as exact.
dp.enable_features("contrib", "idealized-numerics", "honest-but-curious")


def dp_pca(df, columns, row_norm, n_samples, n_components, epsilon):
    """DP PCA of numeric columns. n_samples is public; each row is clipped to L2 norm row_norm
    (choose it from public scales). Replace-one neighbours on the sized domain: d_in = 2.

    Returns ({"components", "explained_variance_ratio"}, epsilon_spent).
    """
    n_features = len(columns)

    def make(e):
        return PCA(epsilon=e, row_norm=row_norm, n_samples=n_samples, n_features=n_features,
                   n_components=n_components).measurement()

    # The constructor's epsilon is not the map at the task's d_in; search it against the real d_in.
    # (In 0.16.0 the map at d_in=2 is about 2x the argument.) Bounds only narrow the search.
    eps_arg = dp.binary_search_param(make, d_in=2, d_out=epsilon, bounds=(1e-6, epsilon))
    model = PCA(epsilon=eps_arg, row_norm=row_norm, n_samples=n_samples, n_features=n_features,
                n_components=n_components)
    epsilon_spent = model.measurement().map(2)
    X = df[columns].to_numpy(dtype=float)
    assert X.shape[0] == n_samples, "row count must equal the public n_samples"
    model.fit(X)                                       # one fit per model
    result = {"components": model.components_, "explained_variance_ratio": model.explained_variance_ratio_}
    return result, epsilon_spent


def _test():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(200, 3)), columns=["x1", "x2", "x3"])
    _, epsilon_spent = dp_pca(df, ["x1", "x2", "x3"], row_norm=2.0, n_samples=200,
                              n_components=2, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
