import opendp.prelude as dp
import pandas as pd

# A user-defined Python function is wrapped, so "honest-but-curious" is required: OpenDP cannot check
# the function or its stability map; the author vouches for both. Many task specs forbid this flag.
dp.enable_features("contrib", "honest-but-curious")


def dp_grouped_sum(df, group_column, value_column, groups, bounds, epsilon):
    """DP sum of value_column per public group, on a pandas DataFrame. Replace-one (n public): d_in = 2;
    a changed row can leave one group and join another, so the L1 change is up to 2*c = stability(2).
    (Add/remove-one, n private: d_in = 1, change up to c.)

    Returns ({group: sum}, epsilon_spent). epsilon comes from OpenDP's Laplace map applied to
    the stated stability map, not from a constant.
    """
    lower, upper = bounds
    c = max(abs(lower), abs(upper))   # one row moves one group's clamped sum by at most c (L1)

    def grouped(frame):
        sums = frame[value_column].clip(lower, upper).groupby(frame[group_column]).sum()
        return [float(sums.get(g, 0.0)) for g in groups]   # fixed public order, missing -> 0

    t = dp.t.make_user_transformation(
        input_domain=dp.user_domain("PandasDataFrame", member=lambda x: isinstance(x, pd.DataFrame)),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
        output_metric=dp.l1_distance(T=float),
        function=grouped,
        stability_map=lambda d_in: float(d_in) * c,
    )
    # Noise from an OpenDP measurement, never numpy.random inside the function.
    meas = dp.binary_search_chain(lambda s: t >> dp.m.then_laplace(s), d_in=2, d_out=epsilon)
    epsilon_spent = meas.map(2)                       # = laplace.map(t.map(2))
    return dict(zip(groups, meas(df))), epsilon_spent


def _test():
    df = pd.DataFrame({"group": ["a", "b", "a", "c"] * 25, "value": [1.0, -4.0, 7.0, 20.0] * 25})
    _, epsilon_spent = dp_grouped_sum(df, "group", "value", ["a", "b", "c", "d"], (-5.0, 10.0), 1.0)
    return {"epsilon": float(epsilon_spent)}
