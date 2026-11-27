"""Compare our adult.py run with the authors' adult_paper_snapshot.csv.
Usage: python compare.py REPRO_CSV PAPER_CSV
Per (model, subgroup): mean vuln / test acc / overfitting gap over reps, plus the per-model F-test p-value.
The README says sklearn is not deterministic, so we use tolerances, not exact equality."""
import sys
import pandas as pd

TOL = {"vuln": 0.01, "target_test_acc": 0.01, "overfitting_gap": 0.01}  # floor = TOL/10 for zero-variance groups
r, p = (pd.read_csv(f, index_col=0) for f in sys.argv[1:3])
# The snapshot uses short ids; the current code prints display names (same 9 models, same order in adult.py).
NAMES = {"Control": "control", r"$\varepsilon=1$": "eps1", r"$\varepsilon=2$": "eps2", r"$\varepsilon=10$": "eps10",
         "Logistic Regression (LR)": "lr", "Fair LR (Dem. Parity)": "lr_dm_threshold",
         "Fair LR (Equalized Odds)": "lr_eo_threshold", "8-Neuron NN": "nn_8", "32-Neuron NN": "nn_32"}
r["model"] = r["model"].map(NAMES)
assert r["model"].notna().all(), "unmapped model name"
key = ["model", "subgroup"]
g = r.groupby(key)[list(TOL)].mean().join(p.groupby(key)[list(TOL)].mean(), lsuffix="_ours", rsuffix="_paper")
ok = True
# Tolerance per group: 4 standard errors of the difference of two 200-rep means (DP training and the attack are
# unseeded, so small subgroups such as Amer-Indian-Eskimo are much noisier than "All").
sd = r.groupby(key)[list(TOL)].std().join(p.groupby(key)[list(TOL)].std(), lsuffix="_ours", rsuffix="_paper")
n = r.groupby(key).size()
for c, t in TOL.items():
    diff = (g[c + "_ours"] - g[c + "_paper"]).abs()
    se = ((sd[c + "_ours"] ** 2 + sd[c + "_paper"] ** 2) / n) ** 0.5
    bad = diff > (4 * se).clip(lower=t / 10)
    ok &= not bad.any()
    z = (diff / se).max()
    print(f"{c:16s} max |ours-paper| = {diff.max():.4f}, max z = {z:.2f} over {len(diff)} groups"
          + (f"  FAIL: {list(diff[bad].index)}" if bad.any() else ""))
fp = r.groupby("model")["p"].first().to_frame("p_ours").join(p.groupby("model")["p"].first().rename("p_paper"))
# Paper Table 2: bold = disparity F-test p < 0.01. The p-value itself moves with the unseeded attack noise,
# so we compare the verdict (which models are bold), not the p-value.
fp["significant_ours"] = fp.p_ours < 0.01
fp["significant_paper"] = fp.p_paper < 0.01
print(fp.to_string())
ok &= (fp.significant_ours == fp.significant_paper).all()
print("MATCH (within tolerance)" if ok else "MISMATCH")
assert set(r.groupby(key).groups) == set(p.groupby(key).groups), "different model/subgroup sets"
