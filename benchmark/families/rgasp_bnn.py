"""rgasp_bnn.py -- two strategies: RGASP_Matern52_EI_nugget and BNN_like_MLPensemble_EI.

RGASP_Matern52_EI_nugget uses the RobustGaSP package of R (Matern-5/2 kernel, estimated nugget) through rpy2;
BNN_like_MLPensemble_EI uses an ensemble of eight small MLPs fitted to bootstrap samples, whose spread serves as
uncertainty. Both use EI on 15,000 candidates per iteration.
Protocol as in core29_gp_ml_tree_rl_mof.py: 5 runs, 3 starting experiments + 39 iterations of 3 experiments.

Requires R with the RobustGaSP package and rpy2. The R start-up below is written for Windows; set ACTMOF_R_HOME to the
folder of the R installation. ACTMOF_POOL_DIR and ACTMOF_OUTPUT_DIR as in the other scripts.
Set SMOKE_TEST = True for a short trial run.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor

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
from rpy2.robjects.packages import importr

R_CONVERTER = default_converter + numpy2ri.converter
with R_CONVERTER.context():
    ro.r('.libPaths(c("{}", .libPaths()))'.format(str(R_LIBS_USER).replace("\\", "/")))
    ro.r("library(RobustGaSP)")
RobustGaSP = importr("RobustGaSP")

_R_RGASP_PREDICT = ro.r("""
function(model, Xnew) {
    p <- predict(model, Xnew)
    mu <- as.numeric(p[["mean"]])
    if ("sd" %in% names(p)) {
        sd <- as.numeric(p[["sd"]])
    } else if ("upper95" %in% names(p) && "lower95" %in% names(p)) {
        sd <- as.numeric(p[["upper95"]] - p[["lower95"]]) / (2 * 1.96)
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


class RGASPSurrogate:
    def __init__(self, max_eval: int = 60):
        self.max_eval = max_eval

    def fit(self, X: np.ndarray, y: np.ndarray):
        with R_CONVERTER.context():
            self.model = RobustGaSP.rgasp(
                design=X, response=y,
                **{"nugget.est": True, "kernel_type": "matern_5_2", "max_eval": int(self.max_eval)},
            )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with R_CONVERTER.context():
            result = _R_RGASP_PREDICT(self.model, X)
        arr = np.asarray(result)
        return arr[:, 0], arr[:, 1]


# ==============================================================================
# 1. BNN-like surrogate
# ==============================================================================

MIN_STD = 1e-9


class MLPSurrogateBNNLike:
    def __init__(self, random_state: int = 0, n_models: int = 8):
        self.random_state = random_state
        self.n_models = n_models

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models = []
        for j in range(self.n_models):
            m = MLPRegressor(
                hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
                learning_rate_init=0.01, max_iter=500, random_state=self.random_state + j,
            )
            boot = np.random.default_rng(self.random_state + j).choice(len(X), size=len(X), replace=True)
            m.fit(X[boot], y[boot])
            self.models.append(m)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.asarray([m.predict(X) for m in self.models], dtype=float)
        mu = preds.mean(axis=0)
        sd = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.full(X.shape[0], 1.0)
        return mu, np.maximum(sd, MIN_STD)


# ==============================================================================
# 2. Config
# ==============================================================================

RANDOM_STATE = 42
BUDGET_TOTAL_EXPERIMENTS = 120
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 39 iterations: 120 experiments = 3 starting experiments + 39 x 3
N_RUNS = 5
ACQ_CANDIDATES_PER_ITER = 15_000   # equalized - was 4_000 for these two methods
DIVERSITY_LAMBDA = 0.03
USE_LOG1P_TARGET = True

METHODS = [
    {"name": "RGASP_Matern52_EI_nugget", "surrogate": "rgasp", "max_eval": 60},
    {"name": "BNN_like_MLPensemble_EI", "surrogate": "bnn_like", "n_models": 8},
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "rgasp_bnn")))

BASELINE = {
    "RGASP_Matern52_EI_nugget": (155042.0, 51289.4),
    "BNN_like_MLPensemble_EI": (140257.8, 18871.7),
}


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def make_surrogate(spec: Dict[str, Any], seed: int):
    if spec["surrogate"] == "rgasp":
        return RGASPSurrogate(max_eval=spec.get("max_eval", 60))
    return MLPSurrogateBNNLike(random_state=seed, n_models=spec.get("n_models", 8))


class ConstantSurrogate:
    """when the training targets have ~zero variance (very common at early iterations, since ~94% of the pool has q_final=0), skip real model fitting - RobustGaSP's marginal- likelihood optimizer errors out on a degenerate (zero-variance) response (confirmed by direct repro: "Expecting a single value: [extent=0]"), and a constant model is the only sensible fit anyway when every observed point has the same value."""
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        return np.full(n, self.value), np.full(n, MIN_STD)


# ==============================================================================
# 3. Candidate pool oracle
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
    print(f"[pool] filled: {oracle.size - int(np.isnan(oracle).sum()):,}/{oracle.size:,}")
    return oracle


def q_of(oracle: np.ndarray, cond: Dict[str, int]) -> float:
    idx = tuple(cond[c] - FEATURE_RANGES[c][0] for c in FEATURES)
    val = oracle[idx]
    return 0.0 if np.isnan(val) else float(val)


def scale_row(cond: Dict[str, int]) -> np.ndarray:
    return np.array([(cond[c] - lo) / (hi - lo) for c, (lo, hi) in FEATURE_RANGES.items()], dtype=float)


def y_to_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.maximum(y, 0.0)) if USE_LOG1P_TARGET else y.copy()


def gaussian_ei(mu, sd, best, xi=0.01):
    sd = np.maximum(sd, MIN_STD)
    imp = mu - best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def greedy_diverse_batch(cand_local_idx, acq, X_scaled_cand, k, rng, diversity_lambda):
    if len(cand_local_idx) <= k:
        return cand_local_idx
    acq = np.where(np.isfinite(acq), acq, -np.inf) + rng.normal(0.0, 1e-12, size=len(acq))
    acq_z = (acq - np.nanmean(acq)) / (np.nanstd(acq) + 1e-12)
    chosen_local: List[int] = []
    remaining = np.arange(len(cand_local_idx))
    for _ in range(k):
        if not chosen_local or diversity_lambda <= 0:
            best_local = remaining[np.argmax(acq_z[remaining])]
        else:
            X_chosen = X_scaled_cand[np.asarray(chosen_local)]
            X_rem = X_scaled_cand[remaining]
            dist = np.sqrt(((X_rem[:, None, :] - X_chosen[None, :, :]) ** 2).sum(axis=2))
            score = acq_z[remaining] + diversity_lambda * dist.min(axis=1)
            best_local = remaining[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]
        if len(remaining) == 0:
            break
    return cand_local_idx[np.asarray(chosen_local, dtype=int)]


def sample_untried_candidates(rng: np.random.Generator, tried: set, n: int) -> List[Dict[str, int]]:
    out, seen_local = [], set()
    while len(out) < n:
        cond = {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}
        t = tuple(cond[c] for c in FEATURES)
        if t in tried or t in seen_local:
            continue
        seen_local.add(t)
        out.append(cond)
    return out


# ==============================================================================
# 4. One active-learning run for one method
# ==============================================================================

def run_single_arm(oracle: np.ndarray, method_spec: Dict[str, Any], run_idx: int, log_f,
                    max_iterations: int) -> List[Dict[str, Any]]:
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

    for iteration in range(1, max_iterations + 1):
        X_train = np.array([scale_row(c) for c in history_conditions])
        y_train_model = y_to_model_scale(np.array(history_y_raw))

        t0 = time.perf_counter()
        if np.nanstd(y_train_model) < 1e-12:
            model = ConstantSurrogate().fit(X_train, y_train_model)
            surrogate_status = "constant_target_fallback"
        else:
            model = make_surrogate(method_spec, seed + iteration).fit(X_train, y_train_model)
            surrogate_status = "ok"
        fit_s = time.perf_counter() - t0

        candidates = sample_untried_candidates(rng, tried, ACQ_CANDIDATES_PER_ITER)
        X_cand = np.array([scale_row(c) for c in candidates])

        t0 = time.perf_counter()
        status = surrogate_status
        try:
            mu, sd = model.predict(X_cand)
        except Exception as e:
            status = f"error: {str(e)[:150]}"
            mu = np.zeros(len(candidates))
            sd = np.ones(len(candidates))
        acq_s = time.perf_counter() - t0

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
            "fit_seconds": round(fit_s, 2), "predict_seconds": round(acq_s, 2),
        }, default=float) + "\n")
        log_f.flush()

        print(f"[{method} run={run_idx}] iter {iteration:>2}/{max_iterations} "
              f"n={len(history_conditions):>3} best_q={best_q:>10,.0f} status={status} "
              f"fit={fit_s:.2f}s predict={acq_s:.2f}s")

    return rows


# ==============================================================================
# 5. Timing canary - run BEFORE committing to the full loop
# ==============================================================================

def run_timing_canary(oracle: np.ndarray) -> None:
    """One realistic-scale fit+predict per method (n_train~30, 15000
    candidates) with a hard wall-clock budget, so a laGP-style cubic-scaling
    surprise gets caught here instead of after an hour of real running."""
    print("[canary] timing check: 1 fit + 1 predict (15,000 candidates) per method, n_train=30 ...")
    rng = np.random.default_rng(0)
    dummy_conditions = [
        {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()} for _ in range(30)
    ]
    X_train = np.array([scale_row(c) for c in dummy_conditions])
    y_train = y_to_model_scale(np.array([q_of(oracle, c) for c in dummy_conditions]))
    candidates = [
        {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}
        for _ in range(ACQ_CANDIDATES_PER_ITER)
    ]
    X_cand = np.array([scale_row(c) for c in candidates])

    for spec in METHODS:
        t0 = time.perf_counter()
        model = make_surrogate(spec, 0).fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.predict(X_cand)
        pred_s = time.perf_counter() - t0
        total = fit_s + pred_s
        print(f"[canary] {spec['name']:<28} fit={fit_s:.2f}s predict={pred_s:.2f}s total={total:.2f}s/call")
        est_full = total * N_RUNS * N_ITERATIONS
        print(f"[canary]   -> extrapolated full run (5 runs x {N_ITERATIONS} iters): ~{est_full/60:.1f} min "
              f"(rough - n_train grows over the run, real time will be somewhat higher)")
        if total > 15:
            raise RuntimeError(f"[CANARY FAILED] {spec['name']}: {total:.1f}s per call is too slow "
                                f"(laGP-style scaling risk) - investigate before running the full loop.")
    print("[canary] OK - both methods within acceptable per-call time budget.\n")


# ==============================================================================
# 6. Main - resumable
# ==============================================================================

SMOKE_TEST = False


def main():
    oracle = build_oracle_lookup()
    run_timing_canary(oracle)

    n_runs = N_RUNS
    n_iterations = N_ITERATIONS
    if SMOKE_TEST:
        print("=== SMOKE_TEST True: 1 run x 3 iterations per method ===")
        n_runs = 1
        n_iterations = 3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "rgasp_bnn_iteration_metrics.csv"
    log_path = OUTPUT_DIR / "rgasp_bnn_history.jsonl"

    done_pairs = set()
    prev = pd.DataFrame()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        for (m, r), g in prev.groupby(["method", "run"]):
            if g["iteration"].max() >= n_iterations:
                done_pairs.add((m, r))
        print(f"[resume] {len(done_pairs)} (method, run) pair(s) already complete")

    all_rows: List[Dict[str, Any]] = prev.to_dict("records") if not prev.empty else []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for run_idx in range(n_runs):
            for spec in METHODS:
                if (spec["name"], run_idx) in done_pairs:
                    print(f"[skip] {spec['name']} run={run_idx} already done")
                    continue
                rows = run_single_arm(oracle, spec, run_idx, log_f, n_iterations)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)
    print("\n=== DONE ===")
    if not SMOKE_TEST:
        final = df[df["iteration"] == N_ITERATIONS]
        for method, (bmean, bstd) in BASELINE.items():
            sub = final[final["method"] == method]
            if not sub.empty:
                print(f"{method}: new mean={sub['best_q'].mean():,.0f} std={sub['best_q'].std():,.0f}  "
                      f"(baseline mean={bmean:,.0f} std={bstd:,.0f})")


if __name__ == "__main__":
    main()
