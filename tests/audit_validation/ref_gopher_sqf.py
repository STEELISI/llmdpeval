"""Audit wrapper: the reference DP release of gopher_sqf, printed as the prompt's dp_* JSON line, plus the DP model's
coefficients (the model is a release too; captured unrounded from the fitted estimator).
Usage: python ref_gopher_sqf.py <data_dir>"""
import json
import sys

import reference_dp as ref

fitted = []
_fit = ref.DPLogisticRegression.fit
ref.DPLogisticRegression.fit = lambda self, *a, **k: fitted.append(_fit(self, *a, **k)) or fitted[-1]
out = ref.release(sys.argv[1])
print(json.dumps({"dp_accuracy": out["dp_accuracy"], "dp_spd": out["dp_spd"],
                  "dp_model_coef": fitted[0].coef_.ravel().tolist(), "epsilon_total": out["epsilon_total"]}))
