"""Known-bad (toy, add/remove): count of income ">50K" with Laplace noise of scale 0.5 while reporting
epsilon 1: true epsilon 2."""
import json

import numpy as np
import pandas as pd

SCALE = 0.5  # half the scale epsilon 1 needs: true epsilon 2
df = pd.read_csv("adult_clean.csv")
print(json.dumps({"dp_count": float((df["income"] == ">50K").sum() + np.random.default_rng().laplace(0, SCALE)),
                  "epsilon_total": 1.0}))
