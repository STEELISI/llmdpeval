import numpy as np


class BudgetLedger:
    """Basic sequential composition ledger: totals are sums of (eps_i, delta_i)."""

    def __init__(self, epsilon, delta=0.0):
        self.epsilon, self.delta, self.entries = epsilon, delta, []

    def spend(self, eps_i, delta_i=0.0, label=""):
        eps_used = sum(e for _, e, _ in self.entries) + eps_i
        delta_used = sum(d for _, _, d in self.entries) + delta_i
        if eps_used > self.epsilon + 1e-9 or delta_used > self.delta + 1e-15:
            raise RuntimeError(f"budget exceeded by {label!r}: eps {eps_used:.4f} > {self.epsilon}")
        self.entries.append((label, eps_i, delta_i))     # charge BEFORE drawing noise

    def total(self):
        return sum(e for _, e, _ in self.entries), sum(d for _, _, d in self.entries)


def dp_summary(df, column, lower, upper, threshold, epsilon, shares=(0.3, 0.7), relation="replace_one"):
    """Noisy count of rows with value >= threshold and noisy clamped sum, under one ledger.

    relation: "replace_one" (default, n public) or "add_remove" (n private). PUBLIC bounds [lower, upper]
    and threshold. Returns ({"count", "sum", "ledger"}, epsilon_spent).
    """
    rng = np.random.default_rng()
    ledger = BudgetLedger(epsilon)
    raw = df[column].to_numpy(dtype=float)
    x = np.clip(raw, lower, upper)
    # count of rows meeting a condition: 1 under both; sum: replace-one U - L, add/remove max(|L|, |U|)
    sens = {"count": 1.0, "sum": upper - lower if relation == "replace_one" else max(abs(lower), abs(upper))}
    exact = {"count": float(np.sum(raw >= threshold)), "sum": float(x.sum())}
    out = {}
    for name, share in zip(["count", "sum"], shares):
        scale = sens[name] / (share * epsilon)
        ledger.spend(sens[name] / scale, label=name)            # epsilon implied by the real scale
        out[name] = exact[name] + rng.laplace(0.0, scale)
    out["ledger"] = ledger.entries
    return out, ledger.total()[0]


def _test():
    import pandas as pd
    df = pd.DataFrame({"value": [3.0, 7.5, -2.0, 12.0, 5.0] * 20})
    res, eps = dp_summary(df, "value", -5.0, 5.0, 4.0, epsilon=1.0)   # bounds straddle 0
    try:
        BudgetLedger(1.0).spend(1.5, label="too much")
        raise AssertionError("ledger should refuse")
    except RuntimeError:
        pass
    return {"epsilon": eps}
