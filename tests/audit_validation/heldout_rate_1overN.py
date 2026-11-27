"""Held-out known-bad (Kulynych data): per-race rate of a fixed public rule (predict >50K when education-num >= 13)
with Laplace noise calibrated to 1 / n (the total row count) instead of the group size, which is private. The true
sensitivity of a group rate is about 1 / (group size), so the small groups are under-charged by about 100x."""
import json
import sys

import numpy as np
import pandas as pd

EPS_EACH = 0.2
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
df = pd.concat([pd.read_csv(f"{sys.argv[1]}/train.csv"), pd.read_csv(f"{sys.argv[1]}/test.csv")]).dropna()
correct = ((df["education-num"] >= 13) == df["income"].astype(str).str.startswith(">50K"))
rng = np.random.default_rng()
dp = {r: float(correct[df["race"] == r].mean() + rng.laplace(0, (1 / len(df)) / EPS_EACH)) for r in RACES}
print(json.dumps({"dp_accuracy": dp, "epsilon_total": EPS_EACH * len(RACES)}))
