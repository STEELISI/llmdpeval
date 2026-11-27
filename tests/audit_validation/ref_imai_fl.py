"""Audit wrapper: the reference DP release of imai_fl (Table 1 and Table 2) as the prompt's dp_* JSON line."""
import json
import sys

import original as og
import reference_dp as ref

df = og.load(sys.argv[1])
out = ref.release(ref.exact_stats(df), len(df))
print(json.dumps({"dp_table1": out["table1"], "dp_table2": out["table2"], "epsilon_total": out["epsilon_total"]},
                 default=float))
