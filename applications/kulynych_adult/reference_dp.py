"""Reference DP version of original.py: model, test accuracy overall and per race, counts by race.

Privacy unit: one row of the complete-case Adult data (one person), replace-one (bounded) DP,
n = 45,222 public (stated in the UCI adult.names file), split sizes public (fixed 50/50 split of n).
Total budget epsilon = 1, delta = 0, split by basic sequential composition (see BUDGET).
Frameworks: diffprivlib 0.6.6 (model), OpenDP 0.16.0 (counts).

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
from sklearn.model_selection import train_test_split

import original

dp.enable_features("contrib")
EPS_TOTAL = 1.0
BUDGET = {"model": 0.6, "test_race_x_correct": 0.2, "race_counts": 0.2}
assert abs(sum(BUDGET.values()) - EPS_TOTAL) < 1e-12
C = 0.1  # slack 2*ln(1 + 0.25*C) = 0.049 at q = 1 (under 10% of the model budget); not tuned on the data
D_IN = 2  # replace-one on a sized dataset = symmetric distance 2 in OpenDP

# Public domains from the UCI adult.names data dictionary (not measured from the data).
DOMAINS = {
    "work-class": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                   "Without-pay", "Never-worked"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                       "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                       "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
}
# Public ranges (sources in spec.md; capital-gain/capital-loss from the IPUMS CPS March/April 1994 codebook,
# "D CAP-GAIN 5 689 (0:99999)" and "D CAP-LOSS 4 694 (0:9999)"; values outside are clamped).
BOUNDS = {"hours-per-week": (1, 99), "capital-loss": (0, 9999), "capital-gain": (0, 99999),
          "education-num": (1, 16)}


def load_raw(data_dir):
    """Same rows as original.load (complete cases), numeric columns left unscaled."""
    df = pd.concat([pd.read_csv(f"{data_dir}/train.csv"), pd.read_csv(f"{data_dir}/test.csv")],
                   ignore_index=True)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    return df[original.CATEGORICAL + original.NUMERIC + ["income"]].dropna()


def public_features(df):
    """Numeric columns mapped to [0, 1] by public bounds, full one-hot over public domains (exactly one 1
    per categorical column; out-of-domain values give all zeros), intercept column, then every row divided
    by sqrt(12): at most 4 + 7 + 1 = 12 entries of size <= 1, so the max row norm is q = 1 exactly."""
    parts = [np.clip((df[c].to_numpy(float) - lo) / (hi - lo), 0.0, 1.0)[:, None] for c, (lo, hi) in BOUNDS.items()]
    parts += [(df[c].to_numpy()[:, None] == np.array(v)[None, :]).astype(float) for c, v in DOMAINS.items()]
    parts.append(np.ones((len(df), 1)))
    return np.hstack(parts) / np.sqrt(len(BOUNDS) + len(DOMAINS) + 1)


def split(df):
    """The 50/50 train/test split of original.py (same seed)."""
    return train_test_split(df.index, test_size=0.5, random_state=original.SEED)


def histogram_meas(n, categories, eps):
    """OpenDP sized count-by-categories + (discrete) Laplace, scale searched so map(D_IN) <= eps."""
    space = dp.vector_domain(dp.atom_domain(T=str), size=n), dp.symmetric_distance()
    return dp.binary_search_chain(
        lambda s: space >> dp.t.then_count_by_categories(categories=categories, null_category=False)
        >> dp.m.then_laplace(s), d_in=D_IN, d_out=eps)


def release(data_dir):
    """DP model, accuracies and race counts on the data in data_dir, with the epsilon spent per release."""
    df = load_raw(data_dir)
    tr, te = split(df)
    X_tr, X_te = public_features(df.loc[tr]), public_features(df.loc[te])
    assert np.linalg.norm(X_tr, axis=1).max() <= 1.0 + 1e-12
    y_tr, y_te = df.loc[tr, "income"].to_numpy(), df.loc[te, "income"].to_numpy()
    race_te = df.loc[te, "race"].to_numpy()
    races = original.RACES

    # 1. Model (diffprivlib objective perturbation), epsilon read back from its accountant.
    acc = BudgetAccountant()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf = DPLogisticRegression(epsilon=BUDGET["model"], data_norm=1.0, C=C, fit_intercept=False,
                                   max_iter=500, accountant=acc).fit(X_tr, y_tr)
    assert clf.n_iter_.max() < 500, "DP model did not converge in 500 iterations"
    spent = {"model": acc.total()[0]}
    correct = clf.predict(X_te) == y_te  # post-processing of the DP model on test rows

    # 2. Test half: histogram over race x correct (10 cells) -> per-race correct counts and group sizes.
    cells = [f"{r}|{k}" for r in races for k in (1, 0)]
    m_test = histogram_meas(len(te), cells, BUDGET["test_race_x_correct"])
    # 3. Whole data: histogram over race (5 cells).
    m_race = histogram_meas(len(df), races, BUDGET["race_counts"])
    for k, m in [("test_race_x_correct", m_test), ("race_counts", m_race)]:
        spent[k] = m.map(D_IN)
        assert spent[k] <= BUDGET[k] + 1e-12, (k, spent[k])
    assert sum(spent.values()) <= EPS_TOTAL + 1e-9, spent

    h = dict(zip(cells, m_test([f"{r}|{int(c)}" for r, c in zip(race_te, correct)])))
    counts = dict(zip(races, m_race(df["race"].tolist())))

    dp_acc = {"All": sum(max(h[f"{r}|1"], 0) for r in races) / len(te)}
    for r in races:
        c1, c0 = max(h[f"{r}|1"], 0), max(h[f"{r}|0"], 0)
        dp_acc[r] = c1 / max(c1 + c0, 1)
    exact = {"All": float(correct.mean()), **{r: float(correct[race_te == r].mean()) for r in races}}
    return {"dp_accuracy": {k: float(v) for k, v in dp_acc.items()},
            "dp_race_counts": {k: int(v) for k, v in counts.items()},
            "dp_model_exact_accuracy": exact,
            "epsilon_spent": {k: float(v) for k, v in spent.items()},
            "epsilon_total": float(sum(spent.values()))}


def true_values(data_dir):
    """Non-private answers of original.py, plus a no-noise control with the reference's public features and C
    (separates the cost of DP noise from the cost of changing the features and regularisation)."""
    df = original.load(data_dir)
    X, y = original.design_matrix(df)
    acc = original.evaluate(df, X, y, original.SEED)
    counts = df.groupby("race").size().reindex(original.RACES).astype(int).to_dict()
    raw = load_raw(data_dir)
    tr, te = split(raw)
    ctl = LogisticRegression(C=C, fit_intercept=False, max_iter=1000).fit(
        public_features(raw.loc[tr]), raw.loc[tr, "income"].to_numpy())
    correct = ctl.predict(public_features(raw.loc[te])) == raw.loc[te, "income"].to_numpy()
    race_te = raw.loc[te, "race"].to_numpy()
    control = {"All": float(correct.mean()), **{r: float(correct[race_te == r].mean()) for r in original.RACES}}
    return {"accuracy": acc, "race_counts": counts, "control_accuracy": control}


if __name__ == "__main__":
    true = true_values(sys.argv[1])
    out = release(sys.argv[1])
    for g in ["All"] + original.RACES:
        print(f"accuracy {g}: true {true['accuracy'][g]:.4f}  DP {out['dp_accuracy'][g]:.4f}")
    print("race counts: true", true["race_counts"], " DP", out["dp_race_counts"])
    print(f"epsilon spent {out['epsilon_spent']}  total {out['epsilon_total']:.4f} (budget {EPS_TOTAL}, delta 0)")
    print(json.dumps({"true": true, "dp": out}))
