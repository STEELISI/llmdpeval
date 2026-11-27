"""Fairness debugging on NYC Stop-Question-Frisk (SQF).

Train a logistic regression that predicts whether a stopped person is frisked,
measure its statistical parity difference (SPD) across race, then check how much
of that bias comes from three subsets of the training data: remove each subset,
retrain, and measure the drop in SPD.

Usage: python original.py <data_dir>   (data_dir holds sqf_train.csv and sqf_test.csv)
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

LABEL = "frisked"
SENSITIVE = "race"          # 0 = protected group, 1 = privileged group
LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, SEED = 0.05, 0.9, 0.03, 100, 1

# Subsets to test (all columns are already integer-coded in the csv files).
PATTERNS = {
    "black_nodescr_outside_age<25":   {"race": 0, "cs_descr": 0, "inout_O": 1, "age": 0},
    "black_nodescr_outside_age25-45": {"race": 0, "cs_descr": 0, "inout_O": 1, "age": 1},
    "white_noviolent_casing_noprox":  {"race": 1, "cs_vcrim": 0, "cs_casng": 1, "ac_proxm": 0},
}


def train(X, y, init=None):
    """Full-batch gradient descent on the log loss with L2 weight decay."""
    torch.manual_seed(SEED)
    model = torch.nn.Linear(X.shape[1], 1)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(Xt)).squeeze(1), yt)
        loss.backward()
        opt.step()
    return model


def predict_proba(model, X):
    """Predicted frisk probability of each row of X."""
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze(1).numpy()


def spd(p, race):
    """Mean predicted frisk probability, protected minus privileged."""
    return float(p[race == 0].mean() - p[race == 1].mean())


def main(data_dir):
    train_df = pd.read_csv(f"{data_dir}/sqf_train.csv")
    test_df = pd.read_csv(f"{data_dir}/sqf_test.csv")
    features = [c for c in train_df.columns if c != LABEL]

    # Standardise with training-set mean/std.
    mu = train_df[features].mean()
    sd = train_df[features].std(ddof=0)
    X_train = ((train_df[features] - mu) / sd).to_numpy()
    X_test = ((test_df[features] - mu) / sd).to_numpy()
    y_train = train_df[LABEL].to_numpy()
    y_test = test_df[LABEL].to_numpy()
    race_test = test_df[SENSITIVE].to_numpy()

    model = train(X_train, y_train)
    p = predict_proba(model, X_test)
    accuracy = float(((p > 0.5) == y_test).mean())
    spd_before = spd(p, race_test)
    print(f"train rows {len(train_df)}, test rows {len(test_df)}")
    print(f"accuracy {accuracy:.4f}  SPD {spd_before:.4f}")

    results = {}
    for name, cond in PATTERNS.items():
        mask = np.ones(len(train_df), dtype=bool)
        for col, val in cond.items():
            mask &= train_df[col].to_numpy() == val
        support = 100 * mask.mean()
        # Retrain without the subset, starting from the original model's weights.
        m2 = train(X_train[~mask], y_train[~mask], init=model.state_dict())
        p2 = predict_proba(m2, X_test)
        spd_after = spd(p2, race_test)
        reduction = 100 * (spd_before - spd_after) / spd_before
        results[name] = {"support_pct": round(support, 2), "spd_after": round(spd_after, 5),
                         "bias_reduction_pct": round(reduction, 2),
                         "accuracy_after": round(float(((p2 > 0.5) == y_test).mean()), 4)}
        print(f"{name}: support {support:.2f}%  SPD after {spd_after:.4f}  bias reduction {reduction:.1f}%")

    print(json.dumps({"accuracy": round(accuracy, 4), "spd": round(spd_before, 5), "patterns": results}))


if __name__ == "__main__":
    main(sys.argv[1])
