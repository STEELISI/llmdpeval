import opendp.prelude as dp
from opendp.extras.numpy.canonical import BinomialCND

dp.enable_features("contrib")


def dp_proportion_test(df, indicator_column, size, theta0, alpha, epsilon, delta):
    """Private hypothesis test and confidence interval for a proportion (Awan and Vadhan 2023).
    The count of 1s in a 0/1 column gets canonical (Tulap) noise; the release then supports exact
    p-values. Replace-one neighbours (n public, d_in = 2); size must equal the number of rows.

    theta0: proportion under the null; H0: p <= theta0 (right tail). Returns
    ({"estimate", "p_value", "ci"}, (epsilon_spent, delta_spent)).
    """
    values = df[indicator_column].astype(float).tolist()
    assert len(values) == size, "n is public under replace-one and must match the data"
    space = dp.vector_domain(dp.atom_domain(T=float, nan=False), size=size), dp.symmetric_distance()
    count = space >> dp.t.then_clamp((0.0, 1.0)) >> dp.t.then_sum()   # number of 1s
    d_mid = count.map(2)               # sensitivity under replace-one (sized domain): 1
    noise = dp.m.make_canonical_noise(count.output_domain, count.output_metric,
                                      d_in=d_mid, d_out=(epsilon, delta))
    meas = count >> noise
    z = meas(values)
    # Post-processing: inference uses only the release and public (d_in, d_out, size).
    cnd = BinomialCND(estimate=z, d_in=d_mid, d_out=(epsilon, delta), size=size)
    result = {"estimate": z / size, "p_value": cnd.p_value(theta0, tail="right"),
              "ci": cnd.confidence_interval(alpha)}
    return result, meas.map(2)


def _test():
    import pandas as pd
    df = pd.DataFrame({"flag": [1, 0, 0, 1, 0] * 40})
    _, (eps, _) = dp_proportion_test(df, "flag", size=200, theta0=0.3, alpha=0.05, epsilon=1.0, delta=1e-6)
    return {"epsilon": float(eps)}
