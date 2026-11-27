"""Known-bad: race counts of the Kulynych Adult data, OpenDP count-by-categories + discrete Laplace calibrated for
add/remove (d_in = 1) although the relation is replace-one: true epsilon 2."""
import json
import sys

import opendp.prelude as dp
import pandas as pd

D_IN = 1  # add/remove calibration under replace-one: true epsilon 2
dp.enable_features("contrib")
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
df = pd.concat([pd.read_csv(f"{sys.argv[1]}/train.csv"), pd.read_csv(f"{sys.argv[1]}/test.csv")])
space = dp.vector_domain(dp.atom_domain(T=str)), dp.symmetric_distance()
make = lambda s: space >> dp.t.then_count_by_categories(categories=RACES, null_category=False) >> dp.m.then_laplace(s)
m = make(dp.binary_search_param(make, d_in=D_IN, d_out=1.0))
counts = m(df["race"].astype(str).tolist())
print(json.dumps({"dp_race_counts": dict(zip(RACES, counts)), "epsilon_total": m.map(D_IN)}))
