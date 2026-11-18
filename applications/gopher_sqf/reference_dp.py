"""Reference DP version of original.py (Level 1: model, accuracy, SPD).

Privacy unit: one row of sqf_train.csv / sqf_test.csv, replace-one (bounded) DP,
row counts public. Total budget epsilon = 1, delta = 0, split by basic sequential
composition (see BUDGET). Frameworks: diffprivlib 0.6.6 (model), OpenDP 0.16.0 (counts).

Usage: python reference_dp.py <data_dir>
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import opendp.prelude as dp
from diffprivlib import BudgetAccountant
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.linear_model import LogisticRegression

import original

dp.enable_features("contrib")
EPS_TOTAL = 1.0
BUDGET = {"model": 0.6, "accuracy": 0.1, "spd_sum_protected": 0.1,
          "spd_sum_privileged": 0.1, "spd_count_protected": 0.1}
assert abs(sum(BUDGET.values()) - EPS_TOTAL) < 1e-12
C = 0.1  # diffprivlib slack 2*ln(1 + 0.25*C) = 0.049 at q = 1 (spec.md: how C was chosen)
D_IN = 2  # replace-one on a sized dataset = symmetric distance 2 in OpenDP
MAX_ITER = 500  # diffprivlib's default 100 can stop L-BFGS early; same cap as kulynych_adult

# Public domain from the SQF coding (not measured): 3-level codes 0..2, all others 0/1.
THREE_LEVEL = {"build", "age", "ht_feet", "weight"}


def public_scale(df, features):
    """Map every feature to [0, 1] by its public max code, add an intercept column,
    divide by sqrt(d) so every row has L2 norm <= 1 (q = 1, exact diffprivlib accounting)."""
    X = np.column_stack([df[c].to_numpy(float) / (2 if c in THREE_LEVEL else 1) for c in features])
    X = np.clip(X, 0.0, 1.0)
    X = np.hstack([X, np.ones((len(X), 1))])
    return X / np.sqrt(X.shape[1])


def sum_meas(n, bounds, eps):
    """OpenDP sized bounded sum + Laplace, scale searched so map(D_IN) <= eps."""
    space = dp.vector_domain(dp.atom_domain(bounds=bounds), size=n), dp.symmetric_distance()
    m = dp.binary_search_chain(lambda s: space >> dp.t.then_sum() >> dp.m.then_laplace(s),
                               d_in=D_IN, d_out=eps)
    return m


def release(data_dir):
    """DP model, accuracy and SPD on the data in data_dir, with the epsilon spent per release."""
    train_df = pd.read_csv(f"{data_dir}/sqf_train.csv")
    test_df = pd.read_csv(f"{data_dir}/sqf_test.csv")
    features = [c for c in train_df.columns if c != original.LABEL]
    X_train, X_test = public_scale(train_df, features), public_scale(test_df, features)
    assert np.linalg.norm(X_train, axis=1).max() <= 1.0 + 1e-12
    y_train, y_test = train_df[original.LABEL].to_numpy(), test_df[original.LABEL].to_numpy()
    race = test_df[original.SENSITIVE].to_numpy()
    n = len(test_df)

    # 1. Model (diffprivlib objective perturbation), budget read back from its accountant.
    acc = BudgetAccountant()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf = DPLogisticRegression(epsilon=BUDGET["model"], data_norm=1.0, C=C, fit_intercept=False,
                                   max_iter=MAX_ITER, accountant=acc).fit(X_train, y_train)
    assert clf.n_iter_.max() < MAX_ITER, f"DP model did not converge in {MAX_ITER} iterations"
    spent = {"model": acc.total()[0]}
    p = clf.predict_proba(X_test)[:, 1]  # post-processing of the DP model on test rows

    # 2. Accuracy and SPD inputs (OpenDP), each epsilon read from measurement.map(D_IN).
    meas = {
        "accuracy": (sum_meas(n, (0, 1), BUDGET["accuracy"]), ((p > 0.5) == y_test).astype(int).tolist()),
        "spd_sum_protected": (sum_meas(n, (0.0, 1.0), BUDGET["spd_sum_protected"]), (p * (race == 0)).tolist()),
        "spd_sum_privileged": (sum_meas(n, (0.0, 1.0), BUDGET["spd_sum_privileged"]), (p * (race == 1)).tolist()),
        "spd_count_protected": (sum_meas(n, (0, 1), BUDGET["spd_count_protected"]), (race == 0).astype(int).tolist()),
    }
    noisy = {}
    for k, (m, x) in meas.items():
        spent[k] = m.map(D_IN)
        assert spent[k] <= BUDGET[k] + 1e-12, (k, spent[k])
        noisy[k] = m(x)
    assert sum(spent.values()) <= EPS_TOTAL + 1e-9, spent

    n0 = min(max(noisy["spd_count_protected"], 1), n - 1)
    dp_acc = noisy["accuracy"] / n
    dp_spd = noisy["spd_sum_protected"] / n0 - noisy["spd_sum_privileged"] / (n - n0)
    exact = {"accuracy": float(((p > 0.5) == y_test).mean()),
             "spd": float(p[race == 0].mean() - p[race == 1].mean())}
    return {"dp_accuracy": float(dp_acc), "dp_spd": float(dp_spd),
            "dp_model_exact_accuracy": exact["accuracy"], "dp_model_exact_spd": exact["spd"],
            "dp_model_coef": clf.coef_.ravel().round(4).tolist(),
            "epsilon_spent": {k: float(v) for k, v in spent.items()},
            "epsilon_total": float(sum(spent.values()))}


def true_values(data_dir):
    """Non-private answers of original.py (no patterns)."""
    train_df = pd.read_csv(f"{data_dir}/sqf_train.csv")
    test_df = pd.read_csv(f"{data_dir}/sqf_test.csv")
    features = [c for c in train_df.columns if c != original.LABEL]
    mu, sd = train_df[features].mean(), train_df[features].std(ddof=0)
    model = original.train(((train_df[features] - mu) / sd).to_numpy(), train_df[original.LABEL].to_numpy())
    p = original.predict_proba(model, ((test_df[features] - mu) / sd).to_numpy())
    y_test, race = test_df[original.LABEL].to_numpy(), test_df[original.SENSITIVE].to_numpy()
    # Control: same public scaling and C as the DP model, no noise (separates DP cost from model change).
    ctl = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(
        public_scale(train_df, features), train_df[original.LABEL].to_numpy())
    pc = ctl.predict_proba(public_scale(test_df, features))[:, 1]
    return {"accuracy": float(((p > 0.5) == y_test).mean()), "spd": original.spd(p, race),
            "control_accuracy": float(((pc > 0.5) == y_test).mean()), "control_spd": original.spd(pc, race)}


if __name__ == "__main__":
    true = true_values(sys.argv[1])
    out = release(sys.argv[1])
    print(f"true accuracy {true['accuracy']:.4f}  DP accuracy {out['dp_accuracy']:.4f}")
    print(f"true SPD {true['spd']:.4f}  DP SPD {out['dp_spd']:.4f}")
    print(f"epsilon spent {out['epsilon_spent']}  total {out['epsilon_total']:.4f} (budget {EPS_TOTAL}, delta 0)")
    print(json.dumps({"true": true, "dp": out}))
