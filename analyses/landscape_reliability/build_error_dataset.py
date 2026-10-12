#!/usr/bin/env python3
"""build_error_dataset.py -- leave-one-out test of the landscape (SI Section S2.5).

Each of the 95 measured conditions is removed in turn; the two random forests are refitted on the other 94 with five
random states, the neighbor rules and thresholds of the landscape are applied, and the removed condition is predicted.
Writes leave_one_out_predictions.csv with, for every condition: measured values, mean prediction, distance to the
nearest other measured condition, share of the five forests beyond each threshold, and the errors.

    python build_error_dataset.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
SEEDS = [0, 1, 2, 3, 4]
N_TREES = 400

# The 96 rows embedded in the source-of-truth notebook.  Keeping them here makes
# this script independent of the notebook file and, importantly, of the pseudo grid.
EXPERIMENTS_CSV = """35,5,22,9,14,0,30
68,7,15,12,24,0,30
12,9,1,7,29,785,2.2
36,10,24,3,18,0,30
55,11,8,2,15,0,30
69,6,23,1,20,605,30
14,12,2,11,15,0,30
42,7,27,6,17,0,30
9,14,7,4,11,182,30
45,6,11,8,28,0,30
27,12,1,2,25,880,3.9
41,14,5,3,16,0,30
5,5,0,1,30,1457,0.32
5,15,0,1,30,0,30
75,15,0,1,30,0,30
16,7,22,3,20,2345,0.37
16,9,23,4,21,960,1.1
15,7,12,11,19,16751,0.22
6,5,29,11,27,7575,0.18
62,7,26,11,26,0,30
5,15,2,11,16,0,30
33,10,7,11,18,8060,0.25
9,6,1,12,19,1318,0.4
48,14,3,12,23,920,4.6
19,8,12,11,16,0,30
15,7,14,11,24,23969,0.21
12,8,10,11,18,7305,0.15
15,7,13,11,18,7703,0.27
14,7,11,11,20,12434,0.24
7,6,20,11,15,0,30
15,7,14,11,24,23969,0.21
15,6,10,11,14,0,30
15,7,10,11,12,0,30
15,8,12,11,19,1103,0.8
9,8,13,7,20,0,30
21,6,13,12,19,17269,0.21
6,6,12,11,20,1077,0.48
15,8,9,12,19,903,2
75,12,6,11,18,0,30
25,13,14,7,24,0,30
17,13,16,12,24,0,30
38,13,16,8,24,0,30
65,5,0,12,24,0,30
74,7,23,1,16,0,30
5,7,21,12,20,0,30
15,7,16,12,29,31082,0.2
16,7,15,11,18,12791,0.3
62,10,12,10,17,0,30
11,7,12,11,24,28179,0.26
17,7,16,11,27,29258,0.29
13,7,10,12,25,24240,0.24
13,6,13,11,24,37307,0.18
15,5,0,12,24,15489,0.28
21,5,26,11,23,9906,0.26
16,7,8,11,24,21878,0.21
28,7,0,12,30,11411,0.22
10,7,0,12,24,688,0.6
23,7,30,11,24,18169,0.2
20,7,30,11,29,10048,0.22
23,7,14,11,25,12005,0.24
16,7,18,12,25,35900,0.22
17,7,18,12,24,35967,0.21
24,7,18,12,24,13410,0.22
16,7,14,11,30,2331,0.3
15,7,13,9,23,23301,0.29
15,7,7,7,26,11663,0.23
13,6,25,12,24,23658,0.28
13,7,11,11,23,17558,0.26
14,7,7,11,26,7418,0.29
15,6,23,12,24,32110,0.3
15,5,8,12,23,20121,0.21
15,6,14,12,21,19118,0.25
13,5,10,11,24,34595,0.14
17,6,11,11,22,17880,0.22
19,5,2,12,24,6978,0.16
6,6,19,12,30,0,30
25,7,18,11,24,8241,0.26
27,7,27,12,26,7910,0.23
15,5,8,11,23,26277,0.22
15,5,11,10,21,18112,0.29
24,5,30,12,24,4394,0.31
13,7,15,12,25,27452,0.24
16,7,11,12,24,16348,0.28
14,6,15,12,26,30033,0.23
14,7,15,11,26,28764,0.29
18,7,10,12,23,28299,0.22
10,7,13,12,29,1082,0.41
23,7,2,12,19,20334,0.27
18,7,1,4,19,19300,0.32
19,6,4,12,25,12874,0.21
17,7,10,12,25,28836,0.26
16,6,6,5,23,16918,0.31
20,6,0,12,23,11668,0.3
17,7,20,11,21,26346,0.28
24,7,22,8,21,17895,0.31
25,7,19,10,27,11657,0.34"""


def calculate_q(intensity: float, fwhm: float) -> int:
    if fwhm == 30 or fwhm <= 0 or intensity <= 0:
        return 0
    return int(round(float(intensity) / float(fwhm)))


def load_and_aggregate_experiments() -> pd.DataFrame:
    columns = FEATURES + ["Intensity_exp", "FWHM_exp"]
    raw = pd.read_csv(StringIO(EXPERIMENTS_CSV), header=None, names=columns)
    data = raw.groupby(FEATURES, as_index=False).agg(
        Intensity_exp=("Intensity_exp", "mean"), FWHM_exp=("FWHM_exp", "mean")
    )
    data["q_true"] = [calculate_q(i, f) for i, f in zip(data.Intensity_exp, data.FWHM_exp)]
    data = data.sort_values(FEATURES).reset_index(drop=True)
    data.insert(0, "experiment_id", [f"Q{i:03d}" for i in range(1, len(data) + 1)])
    return data


def train_rf_pair(x: np.ndarray, intensity: np.ndarray, fwhm: np.ndarray, seed: int):
    kwargs = dict(n_estimators=N_TREES, random_state=seed, n_jobs=1)
    return (RandomForestRegressor(**kwargs).fit(x, intensity),
            RandomForestRegressor(**kwargs).fit(x, fwhm))


def tree_prediction(model: RandomForestRegressor, point: np.ndarray) -> tuple[float, float]:
    values = np.fromiter((tree.predict(point)[0] for tree in model.estimators_), dtype=float)
    return float(values.mean()), float(values.var())


def compute_neighbor_features(point: np.ndarray, remaining: np.ndarray):
    diffs = np.abs(remaining - point[0])
    rule1 = np.all(diffs <= 1, axis=1)
    at_two = diffs == 2
    rule2 = (np.all(diffs <= 2, axis=1) & np.isin(at_two.sum(axis=1), (1, 2))
             & np.all((diffs <= 1) | at_two, axis=1))
    return diffs, rule1, rule2


def apply_actmof_rules(rf_i: float, rf_f: float, rule1: np.ndarray, rule2: np.ndarray,
                       intensity: np.ndarray, fwhm: np.ndarray) -> tuple[float, float, int]:
    if rule1.any():
        return 0.8 * intensity[rule1].mean() + 0.2 * rf_i, 0.8 * fwhm[rule1].mean() + 0.2 * rf_f, 1
    if rule2.any():
        return 0.5 * intensity[rule2].mean() + 0.5 * rf_i, 0.5 * fwhm[rule2].mean() + 0.5 * rf_f, 2
    return rf_i, rf_f, 0


def postprocess_prediction(intensity_pre: float, fwhm_pre: float) -> tuple[int, float, int]:
    intensity, fwhm = float(intensity_pre), float(fwhm_pre)
    if intensity < 3000:
        intensity, fwhm = 0.0, 30.0
    elif fwhm > 3:
        fwhm = 30.0
    intensity_final = int(np.rint(intensity))
    if fwhm == 30:
        fwhm_final = 30.0
    elif fwhm < 1:
        fwhm_final = float(np.round(fwhm, 2))
    else:
        fwhm_final = float(np.round(fwhm, 1))
    q = 0 if fwhm_final == 30 or intensity_final <= 0 else int(np.rint(intensity_final / fwhm_final))
    return intensity_final, fwhm_final, q


def run_single_loso_seed(job: tuple[int, int, pd.DataFrame]) -> dict:
    held_out, seed, data = job
    train = data.drop(index=held_out)
    point = data.loc[held_out, FEATURES].to_numpy(dtype=float).reshape(1, -1)
    x = train[FEATURES].to_numpy(dtype=float)
    y_i = train.Intensity_exp.to_numpy(dtype=float)
    y_f = train.FWHM_exp.to_numpy(dtype=float)
    rf_i_model, rf_f_model = train_rf_pair(x, y_i, y_f, seed)
    rf_i, var_i = tree_prediction(rf_i_model, point)
    rf_f, var_f = tree_prediction(rf_f_model, point)
    diffs, rule1, rule2 = compute_neighbor_features(point, x)
    i_pre, f_pre, rule = apply_actmof_rules(rf_i, rf_f, rule1, rule2, y_i, y_f)
    i_final, f_final, q_pred = postprocess_prediction(i_pre, f_pre)
    q_true = int(data.loc[held_out, "q_true"])
    return {
        "held_out": held_out, "seed": seed, "q_pred": q_pred,
        "Intensity_pred": i_final, "FWHM_pred": f_final,
        "rf_intensity_variance": var_i, "rf_fwhm_variance": var_f,
        "nearest_chebyshev_distance": float(np.max(diffs, axis=1).min()),
        "rule1_neighbor_count": int(rule1.sum()), "rule2_neighbor_count": int(rule2.sum()),
        "rule_applied": rule, "cross_i": int(i_pre < 3000),
        "cross_f": int(i_pre >= 3000 and f_pre > 3),
        "abs_error": abs(q_pred - q_true),
        "log_abs_error": abs(np.log1p(q_pred) - np.log1p(q_true)),
    }


def aggregate_results(data: pd.DataFrame, results: Iterable[dict]) -> pd.DataFrame:
    runs = pd.DataFrame(results)
    mean_cols = ["q_pred", "Intensity_pred", "FWHM_pred", "rf_intensity_variance",
                 "rf_fwhm_variance", "cross_i", "cross_f", "abs_error", "log_abs_error"]
    means = runs.groupby("held_out", sort=True)[mean_cols].mean()
    deterministic = runs.groupby("held_out", sort=True)[
        ["nearest_chebyshev_distance", "rule1_neighbor_count", "rule2_neighbor_count", "rule_applied"]
    ].first()
    out = data.join(means).join(deterministic)
    out = out.rename(columns={"q_pred": "q_pred_mean", "Intensity_pred": "intensity_pred_mean",
                              "FWHM_pred": "fwhm_pred_mean", "cross_i": "intensity_3000_cross_rate",
                              "cross_f": "fwhm_3_cross_rate", "abs_error": "y_abs_error",
                              "log_abs_error": "y_log_abs_error"})
    ordered = (["experiment_id"] + FEATURES + ["Intensity_exp", "FWHM_exp", "q_true", "q_pred_mean",
        "intensity_pred_mean", "fwhm_pred_mean", "rf_intensity_variance", "rf_fwhm_variance",
        "nearest_chebyshev_distance", "rule1_neighbor_count", "rule2_neighbor_count", "rule_applied",
        "intensity_3000_cross_rate", "fwhm_3_cross_rate", "y_abs_error", "y_log_abs_error"])
    return out[ordered].sort_values("experiment_id").reset_index(drop=True)


def detect_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in {"", "-1"}:
        return [x.strip() for x in visible.split(",") if x.strip()]
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        return [x.strip() for x in result.stdout.splitlines() if x.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def initialize_worker(device_queue) -> None:
    """Give each process exactly one of the requested CUDA devices."""
    if device_queue is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_queue.get())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("leave_one_out_predictions.csv"))
    parser.add_argument("--gpus", help="Comma-separated CUDA device IDs (default: all visible GPUs)")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Workers used when no GPU is available (default: min(4, CPU count))")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers_per_gpu < 1 or args.cpu_workers < 1:
        raise SystemExit("Worker counts must be positive")
    gpus = ([x.strip() for x in args.gpus.split(",") if x.strip()]
            if args.gpus is not None else detect_gpus())
    # sklearn's RF is the notebook-faithful backend. GPU IDs determine safe worker
    # concurrency, but are not claimed as RF acceleration.
    workers = len(gpus) * args.workers_per_gpu if gpus else args.cpu_workers
    data = load_and_aggregate_experiments()
    if len(data) != 95:
        raise RuntimeError(f"Expected 95 unique experiments, found {len(data)}")
    jobs = [(held_out, seed, data) for held_out in data.index for seed in SEEDS]
    started = time.perf_counter()
    manager = mp.Manager() if gpus else None
    device_queue = manager.Queue() if manager else None
    if device_queue is not None:
        for _ in range(args.workers_per_gpu):
            for gpu in gpus:
                device_queue.put(gpu)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=(device_queue,)) as pool:
        results = list(tqdm(pool.map(run_single_loso_seed, jobs, chunksize=1), total=len(jobs),
                            desc="ActMOF LOSO", unit="job"))
    if manager is not None:
        manager.shutdown()
    output = aggregate_results(data, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - started
    print(f"Completed {len(jobs)} LOSO reconstructions.")
    print(f"Unique held-out experiments: {len(output)}")
    print(f"Backend: scikit-learn RandomForestRegressor (CPU); workers: {workers}; visible GPUs: {gpus or 'none'}")
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
