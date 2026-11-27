"""Known-bad control: the zhang_faers reference pipeline without noise (reference_dp.release(noise=False))."""
import json
import sys

import reference_dp as ref

out = ref.release(sys.argv[1], noise=False)
print(json.dumps({"dp_significant": out["dp_significant"], "dp_top100": out["dp_top100"],
                  "epsilon_total": 1.0}))
