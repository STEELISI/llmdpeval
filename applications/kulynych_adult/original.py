"""Income prediction on UCI Adult, with accuracy broken down by race.

Train a logistic regression that predicts whether a person earns more than $50K
a year, on a random half of the complete-case Adult data, then report its test
accuracy overall and for each race group, and the number of people in each race group.

Usage: python original.py <data_dir>   (data_dir holds train.csv and test.csv)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL = ["work-class", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
NUMERIC = ["hours-per-week", "capital-loss", "capital-gain", "education-num"]
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEED = 0


def load(data_dir):
    """Pool the two files, standardise the numeric columns, drop rows with missing values."""
    df = pd.concat([pd.read_csv(f"{data_dir}/train.csv"), pd.read_csv(f"{data_dir}/test.csv")],
                   ignore_index=True)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df[NUMERIC] = StandardScaler().fit_transform(df[NUMERIC])
    return df[CATEGORICAL + NUMERIC + ["income"]].dropna()


def design_matrix(df):
    """One-hot encode the categorical columns (first level dropped); cast everything to int."""
    X = pd.get_dummies(df[CATEGORICAL + NUMERIC], drop_first=True).astype(int)
    return X, df["income"]


def evaluate(df, X, y, random_state):
    """Fit on a random 50% of rows, return test accuracy overall and per race."""
    train_idx, test_idx = train_test_split(df.index, test_size=0.5, random_state=random_state)
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=500)
    clf.fit(X.loc[train_idx].values, y.loc[train_idx].values)
    correct = clf.predict(X.loc[test_idx].values) == y.loc[test_idx].values
    race = df.loc[test_idx, "race"].values
    acc = {"All": float(correct.mean())}
    for r in RACES:
        acc[r] = float(correct[race == r].mean())
    return acc


def main(data_dir):
    df = load(data_dir)
    X, y = design_matrix(df)
    counts = df.groupby("race").size().reindex(RACES).astype(int).to_dict()
    acc = evaluate(df, X, y, SEED)
    print(f"rows {len(df)}, features {X.shape[1]}")
    for g, a in acc.items():
        print(f"test accuracy {g}: {a:.4f}")
    print("people per race:", counts)
    print(json.dumps({"accuracy": {g: round(a, 4) for g, a in acc.items()}, "race_counts": counts}))


if __name__ == "__main__":
    main(sys.argv[1])
