"""
================================================================================
Full 5-run rerun of RVM_like_ARD_RFF_EI

Root cause of the rerun: in the real executed benchmark, run=3 of this method
hit a PermissionError while write_checkpoint() tried to write
'bo_benchmark_outputs\\bo_benchmark_errors.csv' (most likely a transient
OneDrive file lock on the synced output directory - not a code or algorithm
bug). The outer per-method-run try/except caught it, logged
"[SKIP] Method-run failed and was skipped: RVM_like_ARD_RFF_EI run=3", and
moved on - but that run had already reached iteration 39/49 (120/150
experiments) at that point, and its partial best_q (85,081) was folded into
final_summary.csv's 5-run mean with no disclosure that it never finished.

Original 5 final values: [34396, 39102, 123796, 85081@iter39(truncated), 88560]
  -> reported mean 74,187 (includes the truncated run)
  -> mean excluding the truncated run: 71,464 (~3.9% difference - real but
     not conclusion-changing; this method ranks near the bottom of all 39)

Per the user's decision: rerun all 5 arms from scratch (not just run=3) for
internal consistency (same code path exercised uniformly), following the
established smoke-test -> canary-test -> full-run discipline.

Surrogate (RVMLikeSurrogate) reused verbatim from Benchmark v3.ipynb:
  RBF random features (RBFSampler, gamma via median-heuristic
  estimate_rbf_gamma) + ARDRegression (sparse Bayesian learning shrinks
  irrelevant basis functions). n_components=256, acq="ei". Not flagged
  "slow" in the original registry -> uses the standard 15,000-candidate
  pool (same as ~33 other non-"slow" methods, no pool-size confound here).

Pure Python/sklearn, no R, no API calls - cheapest of all the reruns this
session, should complete in minutes.

Protocol: full budget (150 experiments, 49 iterations after the initial 3),
5 runs, same seed formula as Benchmark v3.ipynb:
  seed = (RANDOM_STATE + 10_000*run + stable_int(method_name)) % (2**31-1)
================================================================================
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ARDRegression

# ==============================================================================
# 1. RVM-like surrogate (pure sklearn, identical to Benchmark v3.ipynb)
# ==============================================================================

MIN_STD = 1e-9


def estimate_rbf_gamma(X: np.ndarray) -> float:
    """Verbatim from Benchmark v3.ipynb - median-heuristic RBF bandwidth."""
    if X.shape[0] < 2:
        return 1.0
    sample = X
    if len(sample) > 500:
        rng = np.random.default_rng(123)
        sample = sample[rng.choice(len(sample), size=500, replace=False)]
    d = pdist(sample, metric="euclidean")
    d = d[d > 0]
    if len(d) == 0:
        return 1.0
    med = float(np.median(d))
    return float(1.0 / max(2.0 * med**2, 1e-6))


class RVMLikeSurrogate:
    """RVM-inspired model: RBF random features + ARDRegression.
    ARD shrinks irrelevant basis functions, similar in spirit to sparse
    Bayesian learning. Verbatim from Benchmark v3.ipynb."""

    def __init__(self, random_state: int = 0, n_components: int = 256):
        self.random_state = random_state
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        gamma = estimate_rbf_gamma(X)
        self.featurizer = RBFSampler(gamma=gamma, n_components=self.n_components, random_state=self.random_state)
        Z = self.featurizer.fit_transform(X)
        self.model = ARDRegression()
        self.model.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = self.featurizer.transform(X)
        try:
            mu, sd = self.model.predict(Z, return_std=True)
        except Exception:
            mu = self.model.predict(Z)
            sd = np.full(len(mu), np.std(mu) if len(mu) > 1 else 1.0)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


class ConstantSurrogate:
    """Matches Benchmark v3.ipynb's fallback: when training targets have
    ~zero variance (common at early iterations, ~94% of the pool has
    q_final=0), skip real model fitting and predict the constant mean."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        return np.full(n, self.value), np.full(n, MIN_STD)


# ==============================================================================
# 2. Config
# ==============================================================================

RANDOM_STATE = 42
BUDGET_TOTAL_EXPERIMENTS = 150
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 49
N_RUNS = 5
ACQ_CANDIDATES_PER_ITER = 15_000  # not flagged "slow" in the original registry
DIVERSITY_LAMBDA = 0.03
USE_LOG1P_TARGET = True

METHOD_SPEC = {"name": "RVM_like_ARD_RFF_EI", "surrogate": "rvm_like", "n_components": 256}

# Portability note (only change vs original rvm_like_rerun.py): POOL_DIR/
# OUTPUT_DIR now default to ../../data and ../../outputs inside this repo,
# overridable via ACTMOF_POOL_DIR / ACTMOF_OUTPUT_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "rvm_like")))

BASELINE_FINAL_VALUES = [34396.0, 39102.0, 123796.0, 85081.0, 88560.0]  # run3 was truncated @iter39
BASELINE_MEAN_WITH_TRUNCATION = 74187.0
BASELINE_MEAN_EXCLUDING_TRUNCATED = 71464.0


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


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
# 4. One active-learning run
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
            model = RVMLikeSurrogate(random_state=seed + iteration,
                                      n_components=method_spec["n_components"]).fit(X_train, y_train_model)
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
# 5. Timing canary
# ==============================================================================

def run_timing_canary(oracle: np.ndarray) -> None:
    print("[canary] timing check: 1 fit + 1 predict (15,000 candidates), n_train=30 ...")
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

    t0 = time.perf_counter()
    model = RVMLikeSurrogate(random_state=0, n_components=METHOD_SPEC["n_components"]).fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.predict(X_cand)
    pred_s = time.perf_counter() - t0
    total = fit_s + pred_s
    print(f"[canary] {METHOD_SPEC['name']:<28} fit={fit_s:.2f}s predict={pred_s:.2f}s total={total:.2f}s/call")
    est_full = total * N_RUNS * N_ITERATIONS
    print(f"[canary]   -> extrapolated full run (5 runs x {N_ITERATIONS} iters): ~{est_full/60:.1f} min "
          f"(rough - n_train grows over the run, real time will be somewhat higher)")
    if total > 15:
        raise RuntimeError(f"[CANARY FAILED] {METHOD_SPEC['name']}: {total:.1f}s per call is too slow - "
                            f"investigate before running the full loop.")
    print("[canary] OK.\n")


# ==============================================================================
# 6. Smoke test: 1 run x 3 iterations, verify no errors/crashes before full run
# ==============================================================================

def run_smoke_test(oracle: np.ndarray) -> None:
    print("=== SMOKE TEST: 1 run x 3 iterations ===")
    log_path = OUTPUT_DIR / "smoke_test_history.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        rows = run_single_arm(oracle, METHOD_SPEC, run_idx=999, log_f=log_f, max_iterations=3)
    statuses = {r["status"] for r in rows}
    bad = {s for s in statuses if s.startswith("error")}
    if bad:
        raise RuntimeError(f"[SMOKE TEST FAILED] statuses seen: {statuses}")
    print(f"[smoke] statuses seen: {statuses}")
    print("[smoke] OK - no errors.\n")


# ==============================================================================
# 7. Main - resumable
# ==============================================================================

def main():
    oracle = build_oracle_lookup()
    run_timing_canary(oracle)
    run_smoke_test(oracle)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "rvm_like_iteration_metrics.csv"
    log_path = OUTPUT_DIR / "rvm_like_history.jsonl"

    done_runs = set()
    prev = pd.DataFrame()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        for r, g in prev.groupby("run"):
            if g["iteration"].max() >= N_ITERATIONS:
                done_runs.add(r)
        print(f"[resume] {len(done_runs)} run(s) already complete: {sorted(done_runs)}")

    all_rows: List[Dict[str, Any]] = prev.to_dict("records") if not prev.empty else []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for run_idx in range(N_RUNS):
            if run_idx in done_runs:
                print(f"[skip] run={run_idx} already done")
                continue
            rows = run_single_arm(oracle, METHOD_SPEC, run_idx, log_f, N_ITERATIONS)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)
    print("\n=== DONE ===")
    final = df[df["iteration"] == N_ITERATIONS]
    print(f"New 5-run final best_q: {sorted(final['best_q'].tolist())}")
    print(f"New mean={final['best_q'].mean():,.0f} std={final['best_q'].std():,.0f}")
    print(f"(baseline WITH truncated run3: mean={BASELINE_MEAN_WITH_TRUNCATION:,.0f}; "
          f"baseline EXCLUDING truncated run3: mean={BASELINE_MEAN_EXCLUDING_TRUNCATED:,.0f})")


if __name__ == "__main__":
    main()
