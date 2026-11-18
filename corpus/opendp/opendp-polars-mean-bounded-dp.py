import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_mean_public_length(df, column, bounds, max_length, epsilon):
    """DP mean when the row count is public (bounded DP, replace-one neighbours).

    Do any row filtering on df BEFORE calling this; a filter inside the query makes the
    length private again. Returns (private_mean, epsilon_spent).
    """
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        # Replace-one. With invariant="lengths", contributions=1 would be charged zero noise.
        privacy_unit=dp.unit_of(changes=1),
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length, invariant="lengths")],
    )
    query = context.query().select(pl.col(column).cast(float).dp.mean(bounds=bounds))
    summary = query.summarize()
    # Sum row must have a nonzero scale; Length row has scale 0 because the length is public.
    # The trap shows a tiny float (about 4e-9), not exactly 0, so "> 0" would never fail.
    assert summary["scale"][0] > 1e-6, "zero-noise trap: check privacy_unit vs invariant"
    private_mean = query.release().collect().item()
    epsilon_spent = context.accountant.map(context.d_in)   # d_in of the privacy unit (changes=1 -> 1)
    return private_mean, epsilon_spent


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [float(i % 10) for i in range(100)]})
    _, epsilon_spent = dp_mean_public_length(df, "value", (0.0, 10.0), max_length=1000, epsilon=1.0)
    return {"epsilon": float(epsilon_spent)}
