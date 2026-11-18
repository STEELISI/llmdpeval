import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_lazyframe_plan_release(df, column, bounds, max_length, epsilon, n_is_public=True):
    """Framework-level polars release without the Context: describe the data with a LazyFrameDomain,
    write the query as a polars plan with .dp expressions, and let make_private_lazyframe turn it
    into one measurement. One noise knob (global_scale) is searched to hit epsilon.
    Default: replace-one (n public): change_one_distance, d_in = 1 (one changed row); the length is
    then public and released without noise. n_is_public=False: add/remove-one (n private):
    symmetric_distance, d_in = 1.

    Returns (one-row result dict, epsilon_spent).
    """
    lf = pl.from_pandas(df[[column]]).lazy().with_columns(pl.col(column).cast(pl.Float64))
    # Public schema facts: column type, plus a public upper bound on the number of rows.
    domain = dp.lazyframe_domain([dp.series_domain(column, dp.atom_domain(T=dp.f64))])
    domain = dp.with_margin(domain, dp.polars.Margin(by=[], max_length=max_length))
    # The plan is built on an EMPTY frame with the same schema, never on the private data.
    plan = pl.LazyFrame(schema=lf.collect_schema()).select(
        dp.len(signed=True), pl.col(column).dp.sum(bounds))

    metric = dp.change_one_distance() if n_is_public else dp.symmetric_distance()

    def make(scale):
        return dp.m.make_private_lazyframe(domain, metric, dp.max_divergence(), plan, global_scale=scale)

    meas = dp.binary_search_chain(make, d_in=1, d_out=epsilon)
    row = meas(lf).collect().to_dicts()[0]
    return row, meas.map(1)


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 11) - 5.0 for i in range(200)]})
    _, eps_r1 = dp_lazyframe_plan_release(df, "value", (-5.0, 5.0), 1000, 1.0)
    _, eps_ar = dp_lazyframe_plan_release(df, "value", (-5.0, 5.0), 1000, 1.0, n_is_public=False)
    return {"epsilon": float(max(eps_r1, eps_ar))}
