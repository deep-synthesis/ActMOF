"""prepare_percentile_band.py -- conditions of the landscape between its top-1% and top-0.1% thresholds.

Writes percentile_band_coords.npy (the conditions with q between the 99.0th and 99.9th percentile of the landscape)
and percentile_band_summary.txt. run_warmstart.py draws the starting batches of the `same_quantile` start from it.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, POOL_DIR, POOL_GLOB

HERE = Path(__file__).resolve().parent
OUT_COORDS = HERE / "percentile_band_coords.npy"
OUT_SUMMARY = HERE / "percentile_band_summary.txt"

LOW_PERCENTILE = 99.0
HIGH_PERCENTILE = 99.9


def main() -> None:
    files = sorted(glob.glob(str(POOL_DIR / POOL_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {POOL_DIR / POOL_GLOB}")

    print(f"[pass 1/2] computing percentile thresholds over {len(files)} file(s)...")
    q_chunks = []
    for fp in files:
        for chunk in pd.read_csv(fp, usecols=["q_final"], chunksize=500_000):
            q_chunks.append(chunk["q_final"].to_numpy())
    q_all = np.concatenate(q_chunks)
    n_total = len(q_all)
    p_low = np.percentile(q_all, LOW_PERCENTILE)
    p_high = np.percentile(q_all, HIGH_PERCENTILE)
    print(f"  n_total={n_total:,}  P{LOW_PERCENTILE}={p_low:.1f}  P{HIGH_PERCENTILE}={p_high:.1f}")
    del q_all, q_chunks

    print("[pass 2/2] collecting band members (q_final in [P99.0, P99.9))...")
    coord_chunks = []
    for fp in files:
        for chunk in pd.read_csv(fp, usecols=FEATURES + ["q_final"], chunksize=500_000):
            mask = (chunk["q_final"] >= p_low) & (chunk["q_final"] < p_high)
            if mask.any():
                coord_chunks.append(chunk.loc[mask, FEATURES].to_numpy(dtype=np.int16))
    coords = np.concatenate(coord_chunks, axis=0)
    print(f"  band size: {len(coords):,}")

    np.save(OUT_COORDS, coords)
    summary = (
        f"n_total={n_total}\nP{LOW_PERCENTILE}={p_low}\nP{HIGH_PERCENTILE}={p_high}\n"
        f"band_size={len(coords)}\n"
    )
    OUT_SUMMARY.write_text(summary)
    print(f"Wrote {OUT_COORDS} and {OUT_SUMMARY}")
    print(summary)


if __name__ == "__main__":
    main()
