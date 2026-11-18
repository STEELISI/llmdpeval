import opendp.prelude as dp
import polars as pl

dp.enable_features("contrib")


def dp_counts_by_recoded_label(df, code_column, code_to_label, default_label, max_length, epsilon):
    """Counts per label after recoding raw codes with a public codebook (replace_strict). Codes not in
    the codebook map to a public default label, so the output key set is fully known in advance and
    passed as keys (pure DP, no delta). One person = one row (replace-one: bounded DP, n public).

    code_to_label: public dict {code: label}. Returns ({label: count}, epsilon_spent).
    """
    labels = sorted(set(code_to_label.values()) | {default_label})
    context = dp.Context.compositor(
        data=pl.from_pandas(df).lazy(),
        privacy_unit=dp.unit_of(changes=1),   # replace-one: OpenDP doubles the add/remove noise
        privacy_loss=dp.loss_of(epsilon=epsilon),
        split_evenly_over=1,
        margins=[dp.polars.Margin(max_length=max_length)],
    )
    # Without default=, OpenDP turns unknown codes into null (polars would raise a data-dependent error).
    label = pl.col(code_column).replace_strict(code_to_label, default=default_label)
    query = (context.query()
             .group_by(label)
             .agg(dp.len(signed=True))
             .with_keys(pl.LazyFrame({code_column: labels})))
    out = query.release().collect()
    return dict(zip(out[code_column].to_list(), out["len"].to_list())), context.accountant.map(context.d_in)


def _test():
    import pandas as pd
    df = pd.DataFrame({"code": [1, 2, 9, 4] * 50})
    _, epsilon_spent = dp_counts_by_recoded_label(df, "code", {1: "low", 2: "mid", 9: "missing"},
                                                  "other", 1000, 1.0)
    return {"epsilon": float(epsilon_spent)}
