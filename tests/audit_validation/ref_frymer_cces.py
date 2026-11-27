"""Audit wrapper: the reference DP release of frymer_cces as the prompt's dp_* JSON line (union coefficient, its
noise-aware SE and p for models 1-3, and the group means)."""
import json
import sys

import reference_dp as ref

out = ref.release(sys.argv[1])
models = {"model1": out["model1"], "model2": out["model2"], "model3": out["model3"]}
u = {m: v["union_member"] for m, v in models.items()}
g = out["group_means"]
print(json.dumps({"dp_union_coef": {m: x["value"] for m, x in u.items()},
                  "dp_union_se": {m: x["se_noise_aware"] for m, x in u.items()},
                  "dp_union_p": {m: x["p_noise_aware"] for m, x in u.items()},
                  "dp_group_means": {"union": g["mean_union"]["value"], "nonunion": g["mean_nonunion"]["value"],
                                     "diff": g["diff"]["value"]},
                  "dp_diff_se": g["diff"]["se_noise_aware"], "dp_diff_p": g["diff"]["p_noise_aware"],
                  "epsilon_total": out["epsilon_total"]}))
