"""
temporal_analysis.py -- prospective (time-ordered) tests of the landscape construction.

(1) First 60 seeds -> later 35: emulator vs naive baselines.
(2) Rolling split: first N = 40, 50, 60, 70, 80 seeds -> the next batch (N+1 ... N+10;
    the last cut predicts the remaining 15), plus all later seeds.

"Seeds" are the 95 unique conditions in the recording order of Table S3 (entry 31, a duplicate
of entry 26, dropped). The emulator is the unchanged construction (RF + rules + thresholds)
trained on the first N seeds only; predictions are read from the full rebuilt grid.

Naive baselines (use only the first N seeds):
  nearest seed  -- q of the nearest earlier seed in Chebyshev distance (grid steps); ties: mean q
  always product / constant -- for the product call: predict q > 0 everywhere
Outputs: temporal_first60_compare.csv, temporal_rolling.csv, temporal_rolling_points.csv,
         FigS_temporal_rolling.png/pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from landscape_lib import Variant, build_landscape, calc_q, flat_index, seeds_recording_order

HERE = Path(__file__).resolve().parent
P, I, F, ENTRY = seeds_recording_order()
Q = calc_q(I, F)
CUTS = [40, 50, 60, 70, 80]


def nearest_seed(train, test):
    out = []
    for t in test:
        d = np.abs(P[train] - P[t]).max(axis=1)
        out.append(Q[train][d == d.min()].mean())
    return np.array(out)


def metrics(qt, qp):
    el = np.abs(np.log1p(qp) - np.log1p(qt)); cr = qt > 0
    return dict(n=len(qt), n_cryst=int(cr.sum()), n_zero=int((~cr).sum()),
                spearman=spearmanr(qt, qp).statistic,
                spearman_cryst=spearmanr(qt[cr], qp[cr]).statistic if cr.sum() > 2 else np.nan,
                median_elog_cryst=float(np.median(el[cr])), mean_elog=float(el.mean()),
                within2_cryst=float(np.mean(el[cr] <= np.log(2))),
                product_call_acc=float(np.mean((qp > 0) == cr)),
                cryst_called_zero=int(np.sum(cr & (qp == 0))), zero_called_product=int(np.sum(~cr & (qp > 0))))


def main():
    rows, pts = [], []
    maps = {}
    for N in CUTS:
        maps[N] = build_landscape(Variant(f"T{N}", "temporal", f"first {N}", train_idx=np.arange(N)))["q"]
        print("built", N, flush=True)
    for N in CUTS:
        q = maps[N]
        for target, idx in [("next", np.arange(N, min(N + 10, 95)) if N < 80 else np.arange(80, 95)),
                            ("all_later", np.arange(N, 95))]:
            qt = Q[idx]; preds = {"emulator": q[flat_index(P[idx])], "nearest_seed": nearest_seed(np.arange(N), idx),
                                  "always_product": np.full(len(idx), np.median(Q[:N][Q[:N] > 0]))}
            for name, qp in preds.items():
                rows.append(dict(N=N, target=target, predictor=name, **metrics(qt, qp)))
            el_e = np.abs(np.log1p(preds["emulator"]) - np.log1p(qt)); el_n = np.abs(np.log1p(preds["nearest_seed"]) - np.log1p(qt))
            rows[-3]["wilcoxon_p_vs_nearest"] = wilcoxon(el_e, el_n).pvalue if np.any(el_e != el_n) else 1.0
            if target == "next":
                for k, i in enumerate(idx):
                    pts.append(dict(N=N, entry=ENTRY[i], q_true=qt[k], q_emulator=preds["emulator"][k],
                                    q_nearest=preds["nearest_seed"][k]))
    df = pd.DataFrame(rows); df.to_csv(HERE / "temporal_rolling.csv", index=False)
    pd.DataFrame(pts).to_csv(HERE / "temporal_rolling_points.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())

    # first 60 -> later 35, per point, with 95% bootstrap CI of the Spearman difference
    idx = np.arange(60, 95); qt = Q[idx]; qe = maps[60][flat_index(P[idx])]; qn = nearest_seed(np.arange(60), idx)
    rng = np.random.default_rng(0); diff = []
    for _ in range(5000):
        b = rng.integers(0, len(idx), len(idx))
        diff.append(spearmanr(qt[b], qe[b]).statistic - spearmanr(qt[b], qn[b]).statistic)
    print("first60: spearman emulator - nearest, 95% bootstrap CI", np.nanpercentile(diff, [2.5, 50, 97.5]))
    pd.DataFrame(dict(entry=ENTRY[idx], q_true=qt, q_emulator=qe, q_nearest=qn)).to_csv(HERE / "temporal_first60_compare.csv", index=False)


if __name__ == "__main__":
    main()
