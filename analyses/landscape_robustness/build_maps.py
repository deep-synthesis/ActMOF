"""
build_maps.py -- build every robustness variant of the full 6,101,172-point
landscape and save q as maps/<vid>.npy (float32, C order of the grid).
Also writes variants.csv (definitions) and emulator_cv.csv (5-fold CV of the
alternative emulator families, as Table S5 for the RF).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

from landscape_lib import Variant, build_landscape, seeds_recording_order, make_models

HERE = Path(__file__).resolve().parent
MAP_DIR = HERE / "maps"; MAP_DIR.mkdir(exist_ok=True)


def variant_list():
    V = [Variant("V0", "Reference", "Reference (RF, 80:20, 50:50, I 3000, FWHM 3)")]
    for s in range(10):
        V.append(Variant(f"RF{s}", "RF random state", f"random_state = {s}", rf_state=s))
    n_seeds = 95
    for b in range(20):
        rng = np.random.default_rng(1000 + b)
        V.append(Variant(f"BS{b:02d}", "Seed bootstrap", f"bootstrap {b}",
                         train_idx=np.sort(rng.integers(0, n_seeds, n_seeds))))
    V += [
        Variant("R1_60", "Rule parameters", "Rule 1 60:40", w1=0.6),
        Variant("R1_100", "Rule parameters", "Rule 1 100:0", w1=1.0),
        Variant("R2_30", "Rule parameters", "Rule 2 30:70", w2=0.3),
        Variant("R2_70", "Rule parameters", "Rule 2 70:30", w2=0.7),
        Variant("I2000", "Rule parameters", "I threshold 2000", I_thr=2000.0),
        Variant("I5000", "Rule parameters", "I threshold 5000", I_thr=5000.0),
        Variant("F2", "Rule parameters", "FWHM threshold 2", F_thr=2.0),
        Variant("F5", "Rule parameters", "FWHM threshold 5", F_thr=5.0),
        Variant("GB", "Emulator family", "Gradient boosting", model="gb"),
        Variant("KNN", "Emulator family", "kNN (k = 5, distance-weighted)", model="knn"),
        Variant("TEMP60", "Temporal hold-out", "First 60 seeds (recording order)", train_idx=np.arange(60)),
    ]
    return V


def emulator_cv():
    P, I, F, _ = seeds_recording_order()
    order = np.lexsort(P.T[::-1]); P, I, F = P[order], I[order], F[order]
    rows = []
    for kind in ["rf", "gb", "knn"]:
        for tname, y in [("I", I), ("FWHM", F)]:
            r2s, rm = [], []
            for tr, te in KFold(5, shuffle=True, random_state=42).split(P):
                m = make_models(kind, 42)[0].fit(P[tr], y[tr]); p = m.predict(P[te])
                r2s.append(r2_score(y[te], p)); rm.append(np.sqrt(mean_squared_error(y[te], p)))
            rows.append(dict(model=kind, target=tname, r2_mean=np.mean(r2s), r2_sd=np.std(r2s),
                             rmse_mean=np.mean(rm), rmse_sd=np.std(rm)))
    return pd.DataFrame(rows)


def main():
    cv = emulator_cv(); cv.to_csv(HERE / "emulator_cv.csv", index=False); print(cv)
    rows = []
    for v in variant_list():
        rows.append(dict(vid=v.vid, group=v.group, label=v.label, model=v.model, rf_state=v.rf_state,
                         w1=v.w1, w2=v.w2, I_thr=v.I_thr, F_thr=v.F_thr,
                         n_train=95 if v.train_idx is None else len(v.train_idx),
                         n_unique_seeds=95 if v.train_idx is None else len(np.unique(v.train_idx)),
                         train_idx="" if v.train_idx is None else " ".join(map(str, v.train_idx))))
        fp = MAP_DIR / f"{v.vid}.npy"
        if fp.exists() and v.vid != "V0":
            continue
        t = time.time()
        out = build_landscape(v)
        np.save(fp, out["q"].astype(np.float32))
        print(f"{v.vid:8s} built in {time.time() - t:.0f}s  nonzero={np.mean(out['q'] > 0):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "variants.csv", index=False)


if __name__ == "__main__":
    main()
