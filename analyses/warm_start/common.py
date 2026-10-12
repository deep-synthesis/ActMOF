"""common.py -- shared code of the warm-start runs (SI Section S4.6).

Six strategies (Matern-5/2 and Matern-3/2 GPs with EI and PI, conformal random forest, ExtraTrees) and random search
are run from a given starting batch instead of three random conditions. The surrogates, acquisition functions and
batch selection are those of ../../benchmark/families/core29_gp_ml_tree_rl_mof.py; they are repeated here so that the
starting batch and the seed can be set from outside: within a run, all strategies share the starting batch and the
seed. Budget: 3 starting experiments + 39 iterations of 3 experiments.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, RBF, WhiteKernel

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]          # root of this repository

RANDOM_STATE = 42
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
N_ITERATIONS = 39  # 120 experiments = 3 starting experiments + 39 iterations x 3
ACQ_CANDIDATES_PER_ITER = 15_000
DIVERSITY_LAMBDA = 0.03
MIN_STD = 1e-9
USE_LOG1P_TARGET = True

FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"

N_REPLICATES = 20

# =============================================================================
# The 7 strategies: the n=20 statistically-indistinguishable leading group
# =============================================================================
METHOD_SPECS: List[Dict[str, Any]] = [
    {"name": "GP_Matern52_PI", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern52", "acq": "pi", "n_restarts": 1},
    {"name": "GP_Matern52_EI_BASELINE", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "matern52", "acq": "ei", "n_restarts": 1},
    {"name": "GP_Matern32_PI", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern32", "acq": "pi", "n_restarts": 1},
    {"name": "GP_Matern32_EI", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "matern32", "acq": "ei", "n_restarts": 1},
    {"name": "CONFRF_EI", "family": "Machine Learning", "surrogate": "confrf", "acq": "ei", "n_estimators": 300},
    {"name": "BART_like_ExtraTrees_EI", "family": "Basis Regression", "surrogate": "bart_like", "acq": "ei", "n_estimators": 300},
    {"name": "RANDOM_SEARCH", "family": "Baseline", "surrogate": "random", "acq": "random"},
]
METHOD_NAMES = [m["name"] for m in METHOD_SPECS]


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


def y_to_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.maximum(y, 0.0)) if USE_LOG1P_TARGET else y.copy()


def gaussian_ei(mu, sd, best, xi=0.01):
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    improvement = np.asarray(mu, dtype=float) - best - xi
    z = improvement / sd
    return improvement * norm.cdf(z) + sd * norm.pdf(z)


def gaussian_pi(mu, sd, best, xi=0.01):
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    z = (np.asarray(mu, dtype=float) - best - xi) / sd
    return norm.cdf(z)


def acquisition_scores(kind: str, mu, sd, y_best: float, rng: np.random.Generator, spec: Dict[str, Any]) -> np.ndarray:
    if kind == "ei":
        return gaussian_ei(mu, sd, y_best, xi=spec.get("xi", 0.01))
    if kind == "pi":
        return gaussian_pi(mu, sd, y_best, xi=spec.get("xi", 0.01))
    if kind == "random":
        return rng.random(len(mu))
    raise ValueError(f"Unknown acquisition {kind} (only ei, pi and random are used here)")


def greedy_diverse_batch(candidate_idx: np.ndarray, acq: np.ndarray, X_scaled: np.ndarray, k: int,
                          rng: np.random.Generator, diversity_lambda: float) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    acq = np.asarray(acq, dtype=float)
    if len(candidate_idx) <= k:
        return candidate_idx
    if not np.all(np.isfinite(acq)):
        acq = np.where(np.isfinite(acq), acq, -np.inf)
    acq = acq + rng.normal(0.0, 1e-12, size=len(acq))
    chosen_local: List[int] = []
    remaining = np.arange(len(candidate_idx))
    acq_std = np.nanstd(acq)
    acq_z = (acq - np.nanmean(acq)) / (acq_std + 1e-12)
    for _ in range(k):
        if len(chosen_local) == 0 or diversity_lambda <= 0:
            best_local = remaining[np.argmax(acq_z[remaining])]
        else:
            chosen_global_idx = candidate_idx[np.asarray(chosen_local)]
            X_chosen = X_scaled[chosen_global_idx]
            X_rem = X_scaled[candidate_idx[remaining]]
            dist_to_chosen = np.sqrt(((X_rem[:, None, :] - X_chosen[None, :, :]) ** 2).sum(axis=2))
            min_dist = dist_to_chosen.min(axis=1)
            score = acq_z[remaining] + diversity_lambda * min_dist
            best_local = remaining[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]
        if len(remaining) == 0:
            break
    return candidate_idx[np.asarray(chosen_local, dtype=int)]


class ConstantSurrogate:
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
        s = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
        self.sd_ = max(s, 1.0)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_), np.full(X.shape[0], self.sd_)


class SklearnGPSurrogate:
    def __init__(self, kernel_name="matern52", random_state=0, n_restarts=1):
        self.kernel_name = kernel_name
        self.random_state = random_state
        self.n_restarts = n_restarts

    def _kernel(self, p):
        if self.kernel_name == "matern52":
            base = Matern(length_scale=np.ones(p) * 0.5, length_scale_bounds=(1e-2, 10.0), nu=2.5)
        elif self.kernel_name == "matern32":
            base = Matern(length_scale=np.ones(p) * 0.5, length_scale_bounds=(1e-2, 10.0), nu=1.5)
        else:
            raise ValueError(f"Unknown GP kernel {self.kernel_name}")
        return C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))

    def fit(self, X, y):
        self.model = GaussianProcessRegressor(
            kernel=self._kernel(X.shape[1]), alpha=1e-8, normalize_y=True,
            n_restarts_optimizer=self.n_restarts, random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        mu, sd = self.model.predict(X, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


class ConformalRFSurrogate:
    def __init__(self, random_state=0, n_estimators=300):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, bootstrap=True, oob_score=True,
                                            max_features="sqrt", random_state=self.random_state, n_jobs=-1)
        self.model.fit(X, y)
        if hasattr(self.model, "oob_prediction_"):
            resid = np.abs(y - self.model.oob_prediction_)
            resid = resid[np.isfinite(resid)]
            self.conformal_q_ = float(np.quantile(resid, 0.90)) if len(resid) else float(np.std(y))
        else:
            self.conformal_q_ = float(np.std(y))
        self.conformal_q_ = max(self.conformal_q_, MIN_STD)
        return self

    def predict(self, X):
        preds = np.asarray([tree.predict(X) for tree in self.model.estimators_], dtype=float)
        mu = preds.mean(axis=0)
        tree_sd = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.full(X.shape[0], 0.0)
        sd = np.sqrt(tree_sd**2 + self.conformal_q_**2)
        return mu, np.maximum(sd, MIN_STD)


class ExtraTreesBARTLikeSurrogate:
    def __init__(self, random_state=0, n_estimators=300):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators, bootstrap=True, max_features="sqrt",
                                          random_state=self.random_state, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.asarray([tree.predict(X) for tree in self.model.estimators_], dtype=float)
        mu = preds.mean(axis=0)
        sd = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.full(X.shape[0], 1.0)
        return mu, np.maximum(sd, MIN_STD)


def make_surrogate(spec: Dict[str, Any], seed: int):
    kind = spec["surrogate"]
    if kind == "gp":
        return SklearnGPSurrogate(kernel_name=spec.get("kernel", "matern52"), random_state=seed, n_restarts=spec.get("n_restarts", 1))
    if kind == "confrf":
        return ConformalRFSurrogate(random_state=seed, n_estimators=spec.get("n_estimators", 300))
    if kind == "bart_like":
        return ExtraTreesBARTLikeSurrogate(random_state=seed, n_estimators=spec.get("n_estimators", 300))
    raise ValueError(f"Unsupported surrogate {kind!r} for the 7 strategies used here")


GATE_METADATA_PATH = Path(os.environ.get(
    "ACTMOF_GATE_METADATA",
    str(HERE.parent / "landscape_reliability" / "gate_metadata.npz")
))

_GATE_RANGES = [FEATURE_RANGES[c] for c in FEATURES]
_GATE_SIZES = [hi - lo + 1 for lo, hi in _GATE_RANGES]
_GATE_N_TOTAL = int(np.prod(_GATE_SIZES))


def _natural_index(coords: np.ndarray) -> np.ndarray:
    """Flat index into the Cartesian product in FEATURES order -- matches
    build_design_grid.py's grid_key_iter() nesting exactly."""
    coords = np.asarray(coords)
    idx = np.zeros(coords.shape[0], dtype=np.int64)
    for i, (lo, _hi) in enumerate(_GATE_RANGES):
        idx = idx * _GATE_SIZES[i] + (coords[:, i] - lo)
    return idx


class GateLookup:
    def __init__(self, npz_path: Path = GATE_METADATA_PATH):
        data = np.load(npz_path)
        self.d_min = data["d_min"]
        self.boundary_risk = data["boundary_risk"]
        self.suspicious = data["suspicious"]
        n_total = len(self.d_min)
        if n_total != _GATE_N_TOTAL:
            raise RuntimeError(f"gate_metadata.npz has {n_total:,} rows, expected {_GATE_N_TOTAL:,}")

        exp = pd.read_csv(REPO_ROOT / "data" / "anchors_mof321.csv").drop_duplicates(subset=FEATURES)
        exp_coords = exp[FEATURES].to_numpy(dtype=np.int64)
        self._exp_natural_sorted = np.sort(_natural_index(exp_coords))
        if len(self._exp_natural_sorted) != 95:
            raise RuntimeError(f"expected 95 unique experiments, got {len(self._exp_natural_sorted)}")
        # sanity: the last 95 entries of gate_metadata must be the real points (d_min==0)
        if not np.all(self.d_min[-95:] == 0.0):
            raise RuntimeError("gate_metadata.npz's last 95 rows are not all d_min==0 -- pool_idx convention mismatch")

    def pool_idx_of(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords)
        nat = _natural_index(coords)
        pos_in_exp = np.searchsorted(self._exp_natural_sorted, nat)
        pos_clipped = np.clip(pos_in_exp, 0, len(self._exp_natural_sorted) - 1)
        is_exp = (pos_in_exp < len(self._exp_natural_sorted)) & (self._exp_natural_sorted[pos_clipped] == nat)
        n_before = np.searchsorted(self._exp_natural_sorted, nat, side="right")
        return np.where(is_exp, (_GATE_N_TOTAL - 95) + pos_in_exp, nat - n_before)

    def lookup(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.pool_idx_of(coords)
        return self.d_min[p], self.boundary_risk[p], self.suspicious[p]


def chebyshev_dmin_to_set(point: np.ndarray, reference_set: np.ndarray) -> float:
    """with a static landscape, a newly-acquired point is an emulator output, not new experimental support, so it must not shrink the distance used to decide Suspicious."""
    if len(reference_set) == 0:
        return float("inf")
    return float(np.max(np.abs(reference_set.astype(np.int64) - point.astype(np.int64)), axis=1).min())
