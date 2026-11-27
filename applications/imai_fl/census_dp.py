"""Census side: does DP noise in the 2010 Census block tables change Table 1 and Table 2?

The method only uses the Census through r_blk_* = Pr(block | race) and r_prc_* = Pr(precinct | race),
i.e. block / precinct counts by race divided by Florida's total for that race. The authors' blocks and
precincts are de-identified, so we cannot (and do not try to) link them to real Census block IDs. Instead:

1. Rebuild the counts behind r_*: Florida race totals N_k = 1 / (smallest positive r_blk_k); they must equal
   the public 2010 PL 94-171 totals (table P2); then count = r * N_k must be a whole number.
2. Real DAS noise ("ddp"): for every real Florida block (and voting district, VTD) compare the original 2010
   counts with the Census Bureau's 2010 Demonstration Data (2021-06-08 production settings). For each cell of
   the authors' data with true count c, draw one error from real cells of the same race with true count c
   (or the nearest counts, window w), so the noise has the size the 2020 system gives a cell of that size.
   No block is matched to a real block.
3. Documented mechanism ("laplace"): one OpenDP discrete Laplace release of the block x race count vector.
   Replace-one: one person changes at most 2 cells (L1 = 2), and blocks are disjoint, so one vector release
   covers all blocks (parallel composition). epsilon is read from measurement.map(2). Precinct counts are sums
   of the noisy block counts (post-processing). Negative counts are set to 0 (post-processing).
Then Pr(geo | race) is recomputed from the noisy counts and the predictions, Table 1 and Table 2 are re-run.

Usage: python census_dp.py <data_dir> <census_dir> [reps=3] [out=runs/census_dp.json]
"""
import json
import sys

import numpy as np
import pandas as pd
import opendp.prelude as dp

import original as og

dp.enable_features("contrib")
ETH = og.ETH
EPS_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
GEO_PREDS = ["name_precinct", "name_block", "name_precinct_party", "name_block_party"]


def pl_counts(census_dir):
    """Original 2010 PL 94-171 P2 counts for Florida: state totals, blocks (with VTD), in our 5 groups."""
    geo = {}
    with open(f"{census_dir}/flgeo2010.pl", encoding="latin-1") as f:
        for line in f:
            if line[8:11] in ("040", "750"):
                geo[line[18:25]] = (line[8:11], line[29:32], line[54:60], line[61:65], line[161:167].strip())
    rows = []
    with open(f"{census_dir}/fl000012010.pl", encoding="latin-1") as f:
        for line in f:
            v = line.rstrip("\n").split(",")
            g = geo.get(v[4])
            if g:
                p2 = [int(x) for x in v[76:86]]  # P0020001..P0020010
                rows.append(g + (p2[4], p2[5], p2[1], p2[7] + p2[8], p2[6] + p2[9]))
    pl = pd.DataFrame(rows, columns=["sumlev", "county", "tract", "block", "vtd"] + ETH)
    state = pl[pl.sumlev == "040"][ETH].iloc[0].to_numpy()
    return state, pl[pl.sumlev == "750"].drop(columns="sumlev").reset_index(drop=True)


def ddp_counts(census_dir, pl_blocks):
    """Demonstration Data block x race counts for Florida, aligned to the PL 94-171 block order."""
    d = pd.read_csv(f"{census_dir}/ddp_fl_block_long.csv", header=None, dtype=str,
                    names=["county", "tract", "block", "race", "n"])
    d["n"] = d["n"].astype(int)
    d = d[d.race != "two"].pivot_table(index=["county", "tract", "block"], columns="race", values="n",
                                       aggfunc="sum", fill_value=0)
    key = pd.MultiIndex.from_frame(pl_blocks[["county", "tract", "block"]])
    missing = (~d.index.isin(key)).sum()
    assert missing == 0, f"{missing} DDP blocks not in the PL file"
    return d.reindex(key, fill_value=0)[ETH].to_numpy()


def draw_errors(c, ref_true, ref_err, rng, w):
    """For each true count c, one error drawn from reference cells with the same true count,
    or from the w reference cells nearest in true count when fewer than w have exactly c."""
    o = np.argsort(ref_true, kind="stable")
    t, e = ref_true[o], ref_err[o]
    lo, hi = np.searchsorted(t, c, "left"), np.searchsorted(t, c, "right")
    small = hi - lo < w
    lo = np.where(small, np.clip((lo + hi) // 2 - w // 2, 0, len(t) - w), lo)
    hi = np.where(small, lo + w, hi)
    return e[lo + (rng.random(len(c)) * (hi - lo)).astype(int)]


def rebuild(df):
    """True counts behind r_blk / r_prc, one row per block / precinct of the authors' data."""
    b = df.drop_duplicates("Block").set_index("Block").sort_index()
    p = df.drop_duplicates("Precinct").set_index("Precinct").sort_index()
    rb, rp = og.cols(b, "r_blk"), og.cols(p, "r_prc")
    N = np.round(1 / np.nanmin(np.where(rb > 0, rb, np.nan), axis=0))
    cb, cp = rb * N, rp * N
    ok = ~np.isnan(cb).any(axis=1)
    assert np.abs(cb[ok] - np.round(cb[ok])).max() < 1e-4 and np.abs(np.nan_to_num(cp - np.round(cp))).max() < 1e-3
    blk_prc = b["Precinct"].to_numpy()
    return N, b.index.to_numpy(), np.round(cb), p.index.to_numpy(), np.round(cp), blk_prc


def run_tables(df, r_blk_by_block, r_prc_by_prc, blk_ids, prc_ids, sr):
    """Map the (noisy) Pr(geo | race) back to voters, re-run predictions, Table 1 and Table 2 (Bayesian)."""
    rb = r_blk_by_block[np.searchsorted(blk_ids, df["Block"].to_numpy())]
    rp = r_prc_by_prc[np.searchsorted(prc_ids, df["Precinct"].to_numpy())]
    preds = {"name_precinct": og.predict(df, "prc", False, rp), "name_block": og.predict(df, "blk", False, rb),
             "name_precinct_party": og.predict(df, "prc", True, rp), "name_block_party": og.predict(df, "blk", True, rb)}
    cls = {k: og.classify(v) for k, v in preds.items()}
    t1 = {k: og.table1_column(c, sr) for k, c in cls.items()}
    t2 = og.table2(df, {"bayes": preds["name_precinct_party"]})
    return t1, t2, cls


def main(data_dir, census_dir, reps=3, out="runs/census_dp.json"):
    reps = int(reps)
    df = og.load(data_dir)
    sr = df[og.SR].to_numpy(float)
    N, blk_ids, cb, prc_ids, cp, blk_prc = rebuild(df)
    state, pl = pl_counts(census_dir)
    print("Florida totals rebuilt from r_blk:", dict(zip(ETH, N.astype(int))), "PL 94-171 P2:", dict(zip(ETH, state)))
    assert (N == state).all()
    ddp = ddp_counts(census_dir, pl)
    plc = pl[ETH].to_numpy()
    err_blk = ddp - plc
    vtd = pl.groupby(["county", "vtd"]).ngroup().to_numpy()
    nv = vtd.max() + 1
    plv = og.group_sum(vtd, nv, plc)
    err_vtd = og.group_sum(vtd, nv, ddp) - plv
    ddp_state = ddp.sum(axis=0)
    summary = {"florida_totals_pl": dict(zip(ETH, state.tolist())), "florida_totals_ddp": dict(zip(ETH, ddp_state.tolist())),
               "real_blocks": len(pl), "real_vtds": int(nv), "anon_blocks": len(blk_ids), "anon_precincts": len(prc_ids),
               "ddp_block_mae": dict(zip(ETH, np.abs(err_blk).mean(axis=0).round(3).tolist())),
               "ddp_vtd_mae": dict(zip(ETH, np.abs(err_vtd).mean(axis=0).round(2).tolist())),
               "ddp_block_err_sd_by_true_count": {
                   k: {str(c): float(err_blk[plc[:, i] == c, i].std()) for c in [0, 1, 5, 20, 100] if (plc[:, i] == c).any()}
                   for i, k in enumerate(ETH)}}
    print(json.dumps(summary))

    ok_b = ~np.isnan(cb).any(axis=1)
    cb0 = np.nan_to_num(cb).astype(np.int64)
    hidden = cp - og.group_sum(np.searchsorted(prc_ids, blk_prc), len(prc_ids), cb0)  # people in blocks with no voters
    base_t1, base_t2, base_cls = run_tables(df, cb / N, cp / N, blk_ids, prc_ids, sr)
    check = og.table1_column(og.classify(og.predict(df, "blk", True)), sr)["overall_error"]
    assert abs(base_t1["name_block_party"]["overall_error"] - check) < 1e-9, "rebuilt counts do not reproduce r_blk"
    results = {"summary": summary, "original": {"table1": base_t1, "table2": base_t2}, "ddp": [], "laplace": []}
    rng = np.random.default_rng()

    def record(kind, r_b, r_p, extra):
        t1, t2, cls = run_tables(df, r_b, r_p, blk_ids, prc_ids, sr)
        changed = {k: float(np.nanmean((cls[k] != base_cls[k]).any(axis=1))) for k in cls}
        results[kind].append({**extra, "table1": t1, "table2": t2, "share_voters_class_changed": changed})
        print(kind, extra, {k: round(v["overall_error"], 4) for k, v in t1.items()},
              "bayes prc rmse", [round(t2[f"precinct_{r}"]["bayes_rmse"], 3) for r in og.RACE])

    for rep in range(reps):
        nb = np.stack([np.maximum(cb0[:, k] + draw_errors(cb0[:, k], plc[:, k], err_blk[:, k], rng, 100), 0)
                       for k in range(5)], axis=1).astype(float)
        nb[~ok_b] = np.nan
        npc = np.stack([np.maximum(cp[:, k] + draw_errors(cp[:, k], plv[:, k], err_vtd[:, k], rng, 25), 0)
                        for k in range(5)], axis=1)
        record("ddp", nb / ddp_state, npc / ddp_state, {"rep": rep})

    space = dp.vector_domain(dp.atom_domain(T=int)), dp.l1_distance(T=int)
    for eps in EPS_GRID:
        m = dp.binary_search_chain(lambda s: dp.m.make_laplace(*space, scale=s), d_in=2, d_out=eps)
        spent = m.map(2)
        assert spent <= eps + 1e-9
        for rep in range(reps):
            noisy = np.array(m(cb0[ok_b].ravel().tolist())).reshape(-1, 5)
            nb = np.full(cb.shape, np.nan)
            nb[ok_b] = np.maximum(noisy, 0)
            npc = og.group_sum(np.searchsorted(prc_ids, blk_prc), len(prc_ids), nb) + hidden
            tot = N + np.nansum(nb - cb, axis=0)  # simplification: noise of blocks without voters is not simulated
            record("laplace", nb / tot, npc / tot, {"epsilon": eps, "epsilon_spent": spent, "rep": rep})

    with open(out, "w") as f:
        json.dump(json.loads(json.dumps(results, default=float)), f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
