import numpy as np
import opendp.prelude as dp

dp.enable_features("contrib")


def dp_multiselect_frequencies(df, item_columns, f):
    """Local DP for multi-select answers: each record is a 0/1 vector over public items; every bit
    is flipped (RAPPOR-style) before it leaves the user. The analyst debiases the sum.
    Privacy is per record: neighbouring inputs are any two bit vectors with at most max_weight ones.

    item_columns: 0/1 columns, one per public item. f: per-bit flip probability in (0, 1].
    Returns (estimated count per item, epsilon_spent).
    """
    max_weight = len(item_columns)          # public cap on how many items one record may select
    m_rr = dp.m.make_randomized_response_bitvec(
        dp.bitvector_domain(max_weight=max_weight), dp.discrete_distance(), f=f)
    reports = []
    for row in df[item_columns].astype(int).values:
        packed = np.packbits(row, bitorder="little")    # the debias step expects little-endian bits
        reports.append(m_rr(packed.tobytes()))         # this call runs on the user's device
    estimates = dp.m.debias_randomized_response_bitvec(reports, f=f)[: len(item_columns)]
    return dict(zip(item_columns, estimates)), m_rr.map(1)   # epsilon per record (local DP)


def _test():
    import pandas as pd
    df = pd.DataFrame({"x1": [1, 0, 1, 1] * 25, "x2": [0, 0, 1, 0] * 25, "x3": [1, 1, 1, 0] * 25})
    _, epsilon_spent = dp_multiselect_frequencies(df, ["x1", "x2", "x3"], f=0.95)
    return {"epsilon": float(epsilon_spent)}
