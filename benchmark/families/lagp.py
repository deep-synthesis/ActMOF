"""lagp.py -- the two local approximate Gaussian-process strategies (LAGP_ALC_EI, LAGP_NN_EI).

The surrogate is laGP::aGP (R, called through rpy2): for every candidate a small GP is fitted to the 10-60 nearest
tested conditions, chosen by the ALC criterion or as nearest neighbours; EI is computed from its prediction.
Because one local model is fitted per candidate, 4,000 candidates are scored per iteration instead of 15,000.
Protocol as in core29_gp_ml_tree_rl_mof.py: 5 runs, 3 starting experiments + 39 iterations of 3 experiments.

Requires R with the laGP package and rpy2. The R start-up below is written for Windows; set ACTMOF_R_HOME to the
folder of the R installation. ACTMOF_POOL_DIR and ACTMOF_OUTPUT_DIR as in the other scripts.
"""

from __future__ import annotations

import ctypes
import json
import os
import glob
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# ==============================================================================
# 0. R / rpy2 bootstrap
# ==============================================================================

R_HOME = Path(os.environ.get("ACTMOF_R_HOME", r"C:\Program Files\R\R-4.6.1"))
R_LIBS_USER = Path.home() / "R" / "win-library" / "4.6"
os.environ["R_HOME"] = str(R_HOME)
os.environ["R_LIBS_USER"] = str(R_LIBS_USER)
os.environ["R_LIBS"] = str(R_LIBS_USER)

_bin_x64 = R_HOME / "bin" / "x64"
os.add_dll_directory(str(_bin_x64))
os.environ["PATH"] = str(_bin_x64) + os.pathsep + os.environ.get("PATH", "")
for _dll in ["R.dll", "Rblas.dll", "Rlapack.dll", "Riconv.dll"]:
    ctypes.CDLL(str(_bin_x64 / _dll))

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri

R_CONVERTER = default_converter + numpy2ri.converter
with R_CONVERTER.context():
    ro.r('.libPaths(c("{}", .libPaths()))'.format(str(R_LIBS_USER).replace("\\", "/")))
    ro.r("library(laGP)")

import os as _os
LAGP_OMP_THREADS = max(1, (_os.cpu_count() or 4) - 2)  # leave 2 cores free; pure
                                                          # performance knob, does
                                                          # not change results for
                                                          # a fixed seed (measured
                                                          # ~4.5x speedup on 12 cores)

_R_LAGP_PREDICT = ro.r("""
function(Xnew, Xref, Zref, start, end, method, ompt) {
    p <- laGP::aGP(
        X = Xref,
        Z = Zref,
        XX = Xnew,
        start = start,
        end = end,
        method = method,
        omp.threads = ompt,
        verb = 0
    )
    mu <- as.numeric(p[["mean"]])
    if ("var" %in% names(p)) {
        sd <- sqrt(pmax(as.numeric(p[["var"]]), 0))
    } else if ("s2" %in% names(p)) {
        sd <- sqrt(pmax(as.numeric(p[["s2"]]), 0))
    } else if ("sd" %in% names(p)) {
        sd <- as.numeric(p[["sd"]])
    } else {
        fallback <- stats::sd(mu)
        if (!is.finite(fallback) || fallback <= 0) fallback <- 1
        sd <- rep(fallback, length(mu))
    }
    sd[!is.finite(sd)] <- 1
    sd[sd <= 0] <- 1e-9
    cbind(mu, sd)
}
""")


def lagp_predict(X_new: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                  start: int, end: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
    with R_CONVERTER.context():
        result = _R_LAGP_PREDICT(X_new, X_train, y_train, int(start), int(end), method, LAGP_OMP_THREADS)
    arr = np.asarray(result)
    return arr[:, 0], arr[:, 1]


# ==============================================================================
# ==============================================================================

RANDOM_STATE = 42
BUDGET_TOTAL_EXPERIMENTS = 120
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 39 iterations: 120 experiments = 3 starting experiments + 39 x 3
N_RUNS = 5
ACQ_CANDIDATES_PER_ITER_SLOW = 4_000   # laGP is flagged "slow" in the registry
DIVERSITY_LAMBDA = 0.03
MIN_STD = 1e-9
USE_LOG1P_TARGET = True

METHODS = [
    {"name": "LAGP_ALC_EI", "lagp_method": "alc", "start": 10, "end": 60},
    {"name": "LAGP_NN_EI", "lagp_method": "nn", "start": 10, "end": 60},
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75),
    "modulator": (5, 15),
    "add_solvent": (0, 30),
    "reaction_time": (1, 12),
    "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "lagp")))


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


# ==============================================================================
# 2. Candidate pool oracle (same O(1) lookup trick as the window ablations)
# ==============================================================================

def build_oracle_lookup() -> np.ndarray:
    shape = tuple(hi - lo + 1 for lo, hi in FEATURE_RANGES.values())
    oracle = np.full(shape, np.nan, dtype=np.float32)
    files = sorted(glob.glob(str(POOL_DIR / POOL_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {POOL_DIR / POOL_GLOB}")
    usecols = FEATURES + [TARGET]
    mins = [lo for lo, _ in FEATURE_RANGES.values()]
    print(f"[pool] loading {len(files)} file(s)")
    for fp in files:
        df = pd.read_csv(fp, usecols=usecols)
        idx = tuple((df[c].to_numpy(dtype=np.int64) - m) for c, m in zip(FEATURES, mins))
        oracle[idx] = df[TARGET].to_numpy(dtype=np.float32)
        print(f"[pool]   {os.path.basename(fp)}: {len(df):,} rows")
    print(f"[pool] filled: {oracle.size - int(np.isnan(oracle).sum()):,}/{oracle.size:,}")
    return oracle


def q_of(oracle: np.ndarray, cond: Dict[str, int]) -> float:
    idx = tuple(cond[c] - FEATURE_RANGES[c][0] for c in FEATURES)
    val = oracle[idx]
    return 0.0 if np.isnan(val) else float(val)


def scale_row(cond: Dict[str, int]) -> np.ndarray:
    """MinMax-scale to [0,1] using the known full-grid bounds - equivalent to
    fitting sklearn's MinMaxScaler on the whole pool, since the pool spans
    exactly these bounds in every dimension (full factorial design)."""
    return np.array([(cond[c] - lo) / (hi - lo) for c, (lo, hi) in FEATURE_RANGES.items()], dtype=float)


def y_to_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.maximum(y, 0.0)) if USE_LOG1P_TARGET else y.copy()


def gaussian_ei(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    improvement = np.asarray(mu, dtype=float) - best - xi
    z = improvement / sd
    return improvement * norm.cdf(z) + sd * norm.pdf(z)


def greedy_diverse_batch(cand_local_idx: np.ndarray, acq: np.ndarray, X_scaled_cand: np.ndarray,
                          k: int, rng: np.random.Generator, diversity_lambda: float) -> np.ndarray:
    if len(cand_local_idx) <= k:
        return cand_local_idx
    acq = np.where(np.isfinite(acq), acq, -np.inf)
    acq = acq + rng.normal(0.0, 1e-12, size=len(acq))
    acq_std = np.nanstd(acq)
    acq_z = (acq - np.nanmean(acq)) / (acq_std + 1e-12)

    chosen_local: List[int] = []
    remaining = np.arange(len(cand_local_idx))
    for _ in range(k):
        if not chosen_local or diversity_lambda <= 0:
            best_local = remaining[np.argmax(acq_z[remaining])]
        else:
            X_chosen = X_scaled_cand[np.asarray(chosen_local)]
            X_rem = X_scaled_cand[remaining]
            dist = np.sqrt(((X_rem[:, None, :] - X_chosen[None, :, :]) ** 2).sum(axis=2))
            min_dist = dist.min(axis=1)
            score = acq_z[remaining] + diversity_lambda * min_dist
            best_local = remaining[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]
        if len(remaining) == 0:
            break
    return cand_local_idx[np.asarray(chosen_local, dtype=int)]


def sample_untried_candidates(rng: np.random.Generator, tried: set, n: int) -> List[Dict[str, int]]:
    """Rejection-sample n distinct, untried, in-range condition dicts.
    Pool has 6.1M combinations and at most 120 are ever tried, so collisions
    are rare and this converges essentially immediately."""
    out = []
    seen_local = set()
    while len(out) < n:
        cond = {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}
        t = tuple(cond[c] for c in FEATURES)
        if t in tried or t in seen_local:
            continue
        seen_local.add(t)
        out.append(cond)
    return out


# ==============================================================================
# 3. One active-learning run for one laGP method
# ==============================================================================

def run_single_arm(oracle: np.ndarray, method_spec: Dict[str, Any], run_idx: int, log_f) -> List[Dict[str, Any]]:
    method = method_spec["name"]
    seed = (RANDOM_STATE + 10_000 * run_idx + stable_int(method)) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    tried: set = set()
    history_conditions: List[Dict[str, int]] = []
    history_y_raw: List[float] = []
    best_q = 0.0
    rows = []

    def add_experiment(cond: Dict[str, int]) -> float:
        nonlocal best_q
        t = tuple(cond[c] for c in FEATURES)
        tried.add(t)
        q = q_of(oracle, cond)
        history_conditions.append(cond)
        history_y_raw.append(q)
        best_q = max(best_q, q)
        return q

    for c in sample_untried_candidates(rng, tried, INITIAL_EXPERIMENTS):
        add_experiment(c)

    rows.append({"method": method, "run": run_idx, "iteration": 0,
                 "n_experiments": len(history_conditions), "best_q": best_q, "status": "initial_random"})

    for iteration in range(1, N_ITERATIONS + 1):
        X_train = np.array([scale_row(c) for c in history_conditions])
        y_train_model = y_to_model_scale(np.array(history_y_raw))

        candidates = sample_untried_candidates(rng, tried, ACQ_CANDIDATES_PER_ITER_SLOW)
        X_cand = np.array([scale_row(c) for c in candidates])

        # Same adaptive start/end clamping as the original
        # RLaGPSurrogate.predict() - aGP() requires nrow(X) > end, which is
        # false in early iterations when the training set is still small.
        n_train = len(X_train)
        eff_end = max(2, min(method_spec["end"], n_train - 1))
        eff_start = max(2, min(method_spec["start"], eff_end - 1))

        status = "lagp_ok"
        try:
            mu, sd = lagp_predict(X_cand, X_train, y_train_model,
                                   eff_start, eff_end, method_spec["lagp_method"])
        except Exception as e:
            status = f"lagp_error: {str(e)[:150]}"
            mu = np.zeros(len(candidates))
            sd = np.ones(len(candidates))

        y_best_model = float(np.max(y_train_model))
        acq = gaussian_ei(mu, sd, y_best_model)

        chosen_local = greedy_diverse_batch(np.arange(len(candidates)), acq, X_cand, BATCH_SIZE, rng, DIVERSITY_LAMBDA)
        batch = [candidates[i] for i in chosen_local]
        for c in batch:
            add_experiment(c)

        rows.append({"method": method, "run": run_idx, "iteration": iteration,
                     "n_experiments": len(history_conditions), "best_q": best_q, "status": status})

        log_f.write(json.dumps({
            "method": method, "run": run_idx, "iteration": iteration,
            "batch": batch, "best_q_after": best_q, "status": status,
        }, default=float) + "\n")
        log_f.flush()

        print(f"[{method} run={run_idx}] iter {iteration:>2}/{N_ITERATIONS} "
              f"n={len(history_conditions):>3} best_q={best_q:>10,.0f} status={status}")

    return rows


# ==============================================================================
# 4. Main - resumable
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "lagp_iteration_metrics.csv"
    log_path = OUTPUT_DIR / "lagp_history.jsonl"

    done_pairs = set()
    prev = pd.DataFrame()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        for (m, r), g in prev.groupby(["method", "run"]):
            if g["iteration"].max() >= N_ITERATIONS:
                done_pairs.add((m, r))
        print(f"[resume] {len(done_pairs)} (method, run) pair(s) already complete")

    oracle = build_oracle_lookup()
    all_rows: List[Dict[str, Any]] = prev.to_dict("records") if not prev.empty else []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for run_idx in range(N_RUNS):
            for spec in METHODS:
                if (spec["name"], run_idx) in done_pairs:
                    print(f"[skip] {spec['name']} run={run_idx} already done")
                    continue
                rows = run_single_arm(oracle, spec, run_idx, log_f)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)
    print("\n=== DONE ===")
    final = df[df["iteration"] == N_ITERATIONS]
    print(final.groupby("method")["best_q"].agg(["mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()
