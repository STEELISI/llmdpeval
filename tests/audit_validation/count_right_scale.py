"""Control (toy, add/remove): count of income ">50K" with Laplace noise of scale 1 (sensitivity 1, epsilon 1). Valid."""
import json

import numpy as np
import pandas as pd

SCALE = 1.0
df = pd.read_csv("adult_clean.csv")
print(json.dumps({"dp_count": float((df["income"] == ">50K").sum() + np.random.default_rng().laplace(0, SCALE)),
                  "epsilon_total": 1.0}))
