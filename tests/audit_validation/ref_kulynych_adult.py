"""Audit wrapper: the reference DP release of kulynych_adult as the prompt's dp_* JSON line, plus the DP model's
coefficients (captured unrounded from the fitted estimator)."""
import json
import sys

import reference_dp as ref

fitted = []
_fit = ref.DPLogisticRegression.fit
ref.DPLogisticRegression.fit = lambda self, *a, **k: fitted.append(_fit(self, *a, **k)) or fitted[-1]
out = ref.release(sys.argv[1])
print(json.dumps({"dp_accuracy": out["dp_accuracy"], "dp_race_counts": out["dp_race_counts"],
                  "dp_model_coef": fitted[0].coef_.ravel().tolist(), "epsilon_total": out["epsilon_total"]}))
