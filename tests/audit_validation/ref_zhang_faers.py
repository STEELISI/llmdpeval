"""Audit wrapper: the reference DP release of zhang_faers as the prompt's dp_* JSON line."""
import json
import sys

import reference_dp as ref

NOISE = True
out = ref.release(sys.argv[1], noise=NOISE)
print(json.dumps({"dp_significant": out["dp_significant"], "dp_top100": out["dp_top100"],
                  "epsilon_total": out["epsilon_total"]}))
