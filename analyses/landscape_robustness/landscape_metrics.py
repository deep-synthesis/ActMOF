"""
landscape_metrics.py -- compare every variant map with the reference V0.
Writes landscape_metrics.csv and temporal_holdout.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from landscape_lib import (EMU_MASK, N_GRID, SHAPE, basins, calc_q, flat_index, grid_points,
                           seeds_recording_order, tier_thresholds)

HERE = Path(__file__).resolve().parent
MAP_DIR = HERE / "maps"
RNG = np.random.default_rng(20260930)
EMU_IDX = np.where(EMU_MASK)[0]
SAMPLE = np.sort(RNG.choice(EMU_IDX, 200_000, replace=False))


def load(vid):
    return np.load(MAP_DIR / f"{vid}.npy").astype(np.float64)


def top_set(q, thr):
    return (q >= thr) & EMU_MASK


def jacc(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def cheb(i, j):
    G = grid_points()
    return int(np.abs(G[i].astype(int) - G[j].astype(int)).max())


def main():
    var = pd.read_csv(HERE / "variants.csv")
    q0 = load("V0"); th0 = tier_thresholds(q0)
    t01_0, t1_0 = top_set(q0, th0["top01"]), top_set(q0, th0["top1"])
    am0 = int(np.argmax(np.where(EMU_MASK, q0, -1)))
    nz0 = q0 > 0
    rows = []
    for _, v in var.iterrows():
        q = load(v.vid); th = tier_thresholds(q)
        t01, t1 = top_set(q, th["top01"]), top_set(q, th["top1"])
        union_nz = np.where(EMU_MASK & (nz0 | (q > 0)))[0]
        sub = RNG.choice(union_nz, min(200_000, len(union_nz)), replace=False)
        nb, big = basins(t01)
        am = int(np.argmax(np.where(EMU_MASK, q, -1)))
        rows.append(dict(
            vid=v.vid, group=v.group, label=v.label,
            spearman_sample=spearmanr(q0[SAMPLE], q[SAMPLE]).statistic,
            spearman_nonzero=spearmanr(q0[sub], q[sub]).statistic,
            jaccard_top01=jacc(t01, t01_0), jaccard_top1=jacc(t1, t1_0),
            top01_ref_in_var_top1=(t01_0 & t1).sum() / t01_0.sum(),
            zero_frac=1 - np.mean(q[EMU_MASK] > 0),
            n_top01=int(t01.sum()), n_basins=nb, largest_basin=big,
            thr_top1=th["top1"], thr_top01=th["top01"],
            max_emulated_q=float(q[am]), argmax_dist_to_ref=cheb(am, am0),
        ))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "landscape_metrics.csv", index=False)

    # temporal hold-out: the 35 later seeds as predicted by the map built from the first 60
    P, I, F, entry = seeds_recording_order()
    qt = load("TEMP60")
    hold = np.arange(60, 95)
    qtrue = calc_q(I[hold], F[hold]); qpred = qt[flat_index(P[hold])]
    df = pd.DataFrame(dict(entry=entry[hold], q_true=qtrue, q_pred=qpred,
                           e_log=np.abs(np.log1p(qpred) - np.log1p(qtrue))))
    df.to_csv(HERE / "temporal_holdout.csv", index=False)
    cr = df[df.q_true > 0]
    print("temporal hold-out: n", len(df), "crystalline", len(cr),
          "zeros true/pred-zero", int((df.q_true == 0).sum()), int(((df.q_true == 0) & (df.q_pred == 0)).sum()),
          "crystalline predicted 0", int((cr.q_pred == 0).sum()),
          "median e_log (cryst)", cr.e_log.median(), "within x2", np.mean(cr.e_log <= np.log(2)),
          "spearman (cryst)", spearmanr(cr.q_true, cr.q_pred).statistic,
          "spearman (all)", spearmanr(df.q_true, df.q_pred).statistic)


if __name__ == "__main__":
    main()
