"""core29_gp_ml_tree_rl_mof.py -- 29 of the 39 search strategies of the ActMOF benchmark.

Gaussian processes (Matern-5/2, Matern-3/2, RBF and rational-quadratic kernels; EI, PI, UCB, Thompson sampling,
quantile, CVaR and greedy acquisition), tree and other machine-learning surrogates (bootstrap and conformal random
forests, ExtraTrees, quantile gradient boosting), approximate and linear models (random Fourier features, Nystroem,
Bayesian linear model), bandits (LinUCB and chemistry-feature bandits) and random search. The other 10 strategies
(laGP, RobustGaSP, MLP ensemble, ARD regression on random features, LLMs and GP-LLM hybrids) have their own scripts in
this folder; they are listed in METHOD_SPECS only so that all 39 names appear in one place, and are skipped here.

Protocol (the same in every script of this folder): 3 random starting experiments, then 39 iterations of 3 experiments
(120 experiments per run); in every iteration 15,000 untried conditions are drawn at random, scored by the acquisition
function, and a batch of three is chosen by score plus a small distance bonus. The seed of run r of method m is
42 + 10,000 r + stable_int(m).

Environment variables:
    ACTMOF_POOL_DIR        folder with design_grid_part*.csv (default: ../../data)
    ACTMOF_OUTPUT_DIR      output folder (default: ../../outputs/core29_gp_ml_tree_rl_mof)
    ACTMOF_N_RUNS          number of runs per method (default 5; 20 for the n = 20 runs of the paper)
    ACTMOF_RUN_IDX_START   first run index (default 0; to split the runs over several workers)

Set SMOKE_TEST = True for a short trial run.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import pdist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C,
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import RBFSampler, Nystroem

# ==============================================================================
# 0. CONFIG
# ==============================================================================

RANDOM_STATE = 42
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
BUDGET_TOTAL_EXPERIMENTS = 120
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 39 iterations: 120 experiments = 3 starting experiments + 39 x 3
N_RUNS = int(os.environ.get("ACTMOF_N_RUNS", "5"))        # 5 runs per method; 20 for the n = 20 runs
RUN_IDX_START = int(os.environ.get("ACTMOF_RUN_IDX_START", "0"))  # first run index (to split runs over workers)

ACQ_CANDIDATES_PER_ITER = 15_000
DIVERSITY_LAMBDA = 0.03
MIN_STD = 1e-9
USE_LOG1P_TARGET = True

_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "core29_gp_ml_tree_rl_mof")))

SMOKE_TEST = False

METHODS_OF_OTHER_SCRIPTS = {
    "RGASP_Matern52_EI_nugget",
    "LAGP_ALC_EI", "LAGP_NN_EI",
    "RVM_like_ARD_RFF_EI",
    "BNN_like_MLPensemble_EI",
    "LLM_PURE_gpt-4o", "LLM_PURE_gpt-5.5", "LLM_PURE_gpt-5-mini",
    "HYBRID_GP_Matern52_EI_ALT_gpt-5.5", "HYBRID_GP_Matern52_PI_ALT_gpt-5.5",
}


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


# ==============================================================================
# 1. Oracle candidate-pool lookup
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


# ==============================================================================
# 2. Acquisition math
# ==============================================================================

def gaussian_ei(mu, sd, best, xi=0.01):
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    improvement = np.asarray(mu, dtype=float) - best - xi
    z = improvement / sd
    return improvement * norm.cdf(z) + sd * norm.pdf(z)


def gaussian_pi(mu, sd, best, xi=0.01):
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    z = (np.asarray(mu, dtype=float) - best - xi) / sd
    return norm.cdf(z)


def gaussian_ucb(mu, sd, beta=2.0):
    return np.asarray(mu, dtype=float) + beta * np.maximum(np.asarray(sd, dtype=float), MIN_STD)


def gaussian_quantile(mu, sd, tau=0.90):
    return np.asarray(mu, dtype=float) + norm.ppf(tau) * np.maximum(np.asarray(sd, dtype=float), MIN_STD)


def gaussian_cvar_upper(mu, sd, tau=0.90):
    z = norm.ppf(tau)
    return np.asarray(mu, dtype=float) + np.maximum(np.asarray(sd, dtype=float), MIN_STD) * norm.pdf(z) / max(1.0 - tau, 1e-6)


def acquisition_scores(kind: str, mu, sd, y_best: float, rng: np.random.Generator, spec: Dict[str, Any]) -> np.ndarray:
    if kind == "ei":
        return gaussian_ei(mu, sd, y_best, xi=spec.get("xi", 0.01))
    if kind == "pi":
        return gaussian_pi(mu, sd, y_best, xi=spec.get("xi", 0.01))
    if kind == "ucb":
        return gaussian_ucb(mu, sd, beta=spec.get("beta", 2.0))
    if kind == "ts":
        return rng.normal(mu, np.maximum(sd, MIN_STD))
    if kind == "greedy_mu":
        return np.asarray(mu, dtype=float)
    if kind == "quantile":
        return gaussian_quantile(mu, sd, tau=spec.get("tau", 0.90))
    if kind == "cvar":
        return gaussian_cvar_upper(mu, sd, tau=spec.get("tau", 0.90))
    if kind == "random":
        return rng.random(len(mu))
    raise ValueError(f"Unknown acquisition {kind}")


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


# ==============================================================================
# 3. Surrogate model classes
# ==============================================================================

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
        elif self.kernel_name == "rbf":
            base = RBF(length_scale=np.ones(p) * 0.5, length_scale_bounds=(1e-2, 10.0))
        elif self.kernel_name == "rq":
            base = RationalQuadratic(length_scale=0.5, alpha=1.0)
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


class BootRFSurrogate:
    def __init__(self, random_state=0, n_estimators=300):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, bootstrap=True, max_features="sqrt",
                                            min_samples_leaf=1, random_state=self.random_state, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.asarray([tree.predict(X) for tree in self.model.estimators_], dtype=float)
        mu = preds.mean(axis=0)
        sd = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.full(X.shape[0], 1.0)
        return mu, np.maximum(sd, MIN_STD)


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


def estimate_rbf_gamma(X: np.ndarray) -> float:
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


class RFFGPSurrogate:
    def __init__(self, random_state=0, n_components=512):
        self.random_state = random_state
        self.n_components = n_components

    def fit(self, X, y):
        gamma = estimate_rbf_gamma(X)
        self.featurizer = RBFSampler(gamma=gamma, n_components=self.n_components, random_state=self.random_state)
        Z = self.featurizer.fit_transform(X)
        self.model = BayesianRidge()
        self.model.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.featurizer.transform(X)
        mu, sd = self.model.predict(Z, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


class FITCGPLikeSurrogate:
    def __init__(self, random_state=0, n_components=120):
        self.random_state = random_state
        self.n_components = n_components

    def fit(self, X, y):
        gamma = estimate_rbf_gamma(X)
        ncomp = max(2, min(self.n_components, X.shape[0]))
        self.featurizer = Nystroem(kernel="rbf", gamma=gamma, n_components=ncomp, random_state=self.random_state)
        Z = self.featurizer.fit_transform(X)
        self.model = BayesianRidge()
        self.model.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.featurizer.transform(X)
        mu, sd = self.model.predict(Z, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


class BLMSurrogate:
    def fit(self, X, y):
        self.model = BayesianRidge()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        mu, sd = self.model.predict(X, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


class QuantileBoostingSurrogate:
    def __init__(self, random_state=0, alpha_low=0.10, alpha_high=0.90):
        self.random_state = random_state
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

    def fit(self, X, y):
        base = dict(n_estimators=250, max_depth=3, learning_rate=0.03, random_state=self.random_state)
        self.med = GradientBoostingRegressor(loss="squared_error", **base)
        self.low = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_low, **base)
        self.high = GradientBoostingRegressor(loss="quantile", alpha=self.alpha_high, **base)
        self.med.fit(X, y); self.low.fit(X, y); self.high.fit(X, y)
        return self

    def predict(self, X):
        mu = self.med.predict(X)
        lo = self.low.predict(X); hi = self.high.predict(X)
        sd = np.maximum((hi - lo) / (norm.ppf(self.alpha_high) - norm.ppf(self.alpha_low)), MIN_STD)
        return np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)


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


class ContextualLinUCBSurrogate:
    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def fit(self, X, y):
        Phi = np.c_[np.ones(len(X)), X]
        lam = 1.0
        self.A = lam * np.eye(Phi.shape[1]) + Phi.T @ Phi
        self.b = Phi.T @ y
        self.A_inv = np.linalg.pinv(self.A)
        self.beta = self.A_inv @ self.b
        resid = y - Phi @ self.beta
        self.noise_sd = float(np.std(resid, ddof=1)) if len(y) > Phi.shape[1] else float(np.std(y) + 1e-6)
        self.noise_sd = max(self.noise_sd, MIN_STD)
        return self

    def predict(self, X):
        Phi = np.c_[np.ones(len(X)), X]
        mu = Phi @ self.beta
        var = np.einsum("ij,jk,ik->i", Phi, self.A_inv, Phi)
        sd = self.noise_sd * np.sqrt(np.maximum(var, MIN_STD))
        return np.asarray(mu, dtype=float), np.maximum(sd, MIN_STD)


class MOFChemContextualBanditSurrogate:
    """`scaler=None` here (this script has no precomputed pool_df to fit a MinMaxScaler on) -- _safe_raw_features()'s fallback branch below reconstructs raw values from FEATURE_RANGES directly, which is exactly equivalent to a MinMaxScaler fit on those same bounds."""

    def __init__(self, scaler=None, ridge=1.0, success_weight=2.0, top_weight=4.0,
                 top_quantile=0.80, include_pairwise=True, include_distance_features=True):
        self.scaler = scaler
        self.ridge = float(ridge)
        self.success_weight = float(success_weight)
        self.top_weight = float(top_weight)
        self.top_quantile = float(top_quantile)
        self.include_pairwise = bool(include_pairwise)
        self.include_distance_features = bool(include_distance_features)

    def _safe_raw_features(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            try:
                return self.scaler.inverse_transform(X)
            except Exception:
                pass
        raw = np.zeros_like(X, dtype=float)
        raw[:, 0] = 5 + X[:, 0] * (75 - 5)
        raw[:, 1] = 5 + X[:, 1] * (15 - 5)
        raw[:, 2] = 0 + X[:, 2] * (30 - 0)
        raw[:, 3] = 1 + X[:, 3] * (12 - 1)
        raw[:, 4] = 10 + X[:, 4] * (30 - 10)
        return raw

    def _feature_map(self, X):
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        raw = self._safe_raw_features(X)
        metal, modulator, solvent, rtime, temp = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
        feats = [np.ones(n), X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4],
                 X[:, 0]**2, X[:, 1]**2, X[:, 2]**2, X[:, 3]**2, X[:, 4]**2]
        if self.include_pairwise:
            for i in range(p):
                for j in range(i + 1, p):
                    feats.append(X[:, i] * X[:, j])
        metal_mod_ratio = np.clip((metal / np.maximum(modulator, 1.0)) / 15.0, 0.0, 1.0)
        solvent_metal_ratio = np.clip((solvent / np.maximum(metal, 1.0)) / 6.0, 0.0, 1.0)
        mod_solvent_ratio = np.clip((modulator / np.maximum(solvent + 1.0, 1.0)) / 15.0, 0.0, 1.0)
        thermal_dose = np.clip((temp * rtime) / (30.0 * 12.0), 0.0, 1.0)
        time_temp_interaction = np.clip(((rtime - 1.0) / 11.0) * ((temp - 10.0) / 20.0), 0.0, 1.0)
        metal_solvent_balance = np.clip((metal / 75.0) * (solvent / 30.0), 0.0, 1.0)
        mod_solvent_balance = np.clip(((modulator - 5.0) / 10.0) * (solvent / 30.0), 0.0, 1.0)
        crystal_growth_proxy = np.clip(((modulator - 5.0) / 10.0) * ((rtime - 1.0) / 11.0) * ((temp - 10.0) / 20.0), 0.0, 1.0)
        feats.extend([metal_mod_ratio, solvent_metal_ratio, mod_solvent_ratio, thermal_dose,
                      time_temp_interaction, metal_solvent_balance, mod_solvent_balance, crystal_growth_proxy])
        if self.include_distance_features and hasattr(self, "best_x_scaled_"):
            dist_to_best = np.sqrt(((X - self.best_x_scaled_[None, :]) ** 2).sum(axis=1)) / np.sqrt(p)
            dist_to_success_centroid = np.sqrt(((X - self.success_centroid_[None, :]) ** 2).sum(axis=1)) / np.sqrt(p)
            feats.extend([dist_to_best, dist_to_success_centroid,
                          1.0 / (1.0 + 5.0 * dist_to_best), 1.0 / (1.0 + 5.0 * dist_to_success_centroid)])
        Phi = np.vstack(feats).T
        Phi = np.asarray(Phi, dtype=float)
        Phi[~np.isfinite(Phi)] = 0.0
        return Phi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) == 0:
            raise ValueError("MOFChemContextualBanditSurrogate received no training data.")
        self.best_x_scaled_ = X[int(np.argmax(y))].copy()
        success_mask = y > 0
        self.success_centroid_ = X[success_mask].mean(axis=0) if np.any(success_mask) else X.mean(axis=0)
        Phi = self._feature_map(X)
        d = Phi.shape[1]
        weights = np.ones(len(y), dtype=float)
        weights += self.success_weight * success_mask.astype(float)
        if len(y) >= 3:
            tail_cut = float(np.quantile(y, self.top_quantile))
            weights += self.top_weight * (y >= tail_cut).astype(float)
        weights = np.maximum(weights, 1e-6)
        sqrt_w = np.sqrt(weights)
        Phi_w = Phi * sqrt_w[:, None]
        y_w = y * sqrt_w
        A = self.ridge * np.eye(d) + Phi_w.T @ Phi_w
        b = Phi_w.T @ y_w
        self.A_inv_ = np.linalg.pinv(A)
        self.beta_ = self.A_inv_ @ b
        pred = Phi @ self.beta_
        resid = y - pred
        denom = max(float(np.sum(weights) - d), 1.0)
        self.noise_sd_ = float(np.sqrt(np.sum(weights * resid ** 2) / denom))
        self.noise_sd_ = max(self.noise_sd_, 1e-6)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Phi = self._feature_map(X)
        mu = Phi @ self.beta_
        epistemic_var = np.maximum(np.einsum("ij,jk,ik->i", Phi, self.A_inv_, Phi), 1e-12)
        sd = self.noise_sd_ * np.sqrt(1.0 + epistemic_var)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


def make_surrogate(spec: Dict[str, Any], seed: int):
    kind = spec["surrogate"]
    if kind == "gp":
        return SklearnGPSurrogate(kernel_name=spec.get("kernel", "matern52"), random_state=seed, n_restarts=spec.get("n_restarts", 1))
    if kind == "bootrf":
        return BootRFSurrogate(random_state=seed, n_estimators=spec.get("n_estimators", 300))
    if kind == "confrf":
        return ConformalRFSurrogate(random_state=seed, n_estimators=spec.get("n_estimators", 300))
    if kind == "rffgp":
        return RFFGPSurrogate(random_state=seed, n_components=spec.get("n_components", 512))
    if kind == "fitcgp_like":
        return FITCGPLikeSurrogate(random_state=seed, n_components=spec.get("n_components", 120))
    if kind == "blm":
        return BLMSurrogate()
    if kind == "qboost":
        return QuantileBoostingSurrogate(random_state=seed)
    if kind == "bart_like":
        return ExtraTreesBARTLikeSurrogate(random_state=seed, n_estimators=spec.get("n_estimators", 300))
    if kind == "linucb":
        return ContextualLinUCBSurrogate(alpha=spec.get("alpha", 1.5))
    if kind == "mof_chem_bandit":
        return MOFChemContextualBanditSurrogate(
            scaler=None, ridge=spec.get("ridge", 1.0), success_weight=spec.get("success_weight", 2.0),
            top_weight=spec.get("top_weight", 4.0), top_quantile=spec.get("top_quantile", 0.80),
        )
    if kind == "constant":
        return ConstantSurrogate()
    raise ValueError(
        f"Unknown surrogate type: {kind!r} "
        f"(rgasp/lagp/hetgp/gpytorch_exact/svigp/ngboost/llm/hybrid_gp_llm are "
        f"implemented in the other scripts of this folder)"
    )


# ==============================================================================
# 4. Method registry
# ==============================================================================

METHOD_SPECS: List[Dict[str, Any]] = [
    {"name": "GP_Matern32_EI", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "matern32", "acq": "ei", "n_restarts": 1},
    {"name": "GP_Matern52_EI_BASELINE", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "matern52", "acq": "ei", "n_restarts": 1},
    {"name": "BOOTRF_EI_BASELINE", "family": "Machine Learning", "surrogate": "bootrf", "acq": "ei", "n_estimators": 300},
    {"name": "GP_RBF_EI", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "rbf", "acq": "ei", "n_restarts": 1},
    {"name": "GP_RationalQuadratic_EI", "family": "Gaussian Processes", "surrogate": "gp", "kernel": "rq", "acq": "ei", "n_restarts": 1},
    {"name": "GP_Matern52_PI", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern52", "acq": "pi", "n_restarts": 1},
    {"name": "GP_Matern52_UCB", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern52", "acq": "ucb", "beta": 2.0, "n_restarts": 1},
    {"name": "GP_Matern32_PI", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern32", "acq": "pi", "n_restarts": 1},
    {"name": "GP_Matern32_UCB", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern32", "acq": "ucb", "beta": 2.0, "n_restarts": 1},
    {"name": "GP_Matern52_TS", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern52", "acq": "ts", "n_restarts": 1},
    {"name": "GP_Matern32_TS", "family": "Acquisition Ablation", "surrogate": "gp", "kernel": "matern32", "acq": "ts", "n_restarts": 1},
    {"name": "MOF_Greedy_GP_Matern52", "family": "MOF-specific", "surrogate": "gp", "kernel": "matern52", "acq": "greedy_mu", "n_restarts": 1},
    {"name": "MOF_Quantile90_GP_Matern52", "family": "MOF-specific", "surrogate": "gp", "kernel": "matern52", "acq": "quantile", "tau": 0.90, "n_restarts": 1},
    {"name": "MOF_CVaR90_GP_Matern52", "family": "MOF-specific", "surrogate": "gp", "kernel": "matern52", "acq": "cvar", "tau": 0.90, "n_restarts": 1},
    {"name": "MOF_Diversity_EI_GP_Matern52", "family": "MOF-specific", "surrogate": "gp", "kernel": "matern52", "acq": "ei", "diversity_lambda": 0.10, "n_restarts": 1},
    {"name": "RGASP_Matern52_EI_nugget", "family": "Gaussian Processes", "surrogate": "rgasp", "in_other_script": True},
    {"name": "LAGP_ALC_EI", "family": "Approximate GPs", "surrogate": "lagp", "in_other_script": True},
    {"name": "LAGP_NN_EI", "family": "Approximate GPs", "surrogate": "lagp", "in_other_script": True},
    {"name": "RFFGP_EI", "family": "Approximate GPs", "surrogate": "rffgp", "acq": "ei", "n_components": 512},
    {"name": "FITCGP_Nystroem_EI", "family": "Approximate GPs", "surrogate": "fitcgp_like", "acq": "ei", "n_components": 120},
    {"name": "BLM_EI", "family": "Basis Regression", "surrogate": "blm", "acq": "ei"},
    {"name": "RVM_like_ARD_RFF_EI", "family": "Basis Regression", "surrogate": "rvm_like", "in_other_script": True},
    {"name": "BART_like_ExtraTrees_EI", "family": "Basis Regression", "surrogate": "bart_like", "acq": "ei", "n_estimators": 300},
    {"name": "CONFRF_EI", "family": "Machine Learning", "surrogate": "confrf", "acq": "ei", "n_estimators": 300},
    {"name": "BNN_like_MLPensemble_EI", "family": "Machine Learning", "surrogate": "bnn_like", "in_other_script": True},
    {"name": "GBC_like_QuantileBoosting_EI", "family": "Machine Learning", "surrogate": "qboost", "acq": "ei"},
    {"name": "GBC_like_QuantileBoosting_Q90", "family": "Machine Learning", "surrogate": "qboost", "acq": "quantile", "tau": 0.90},
    {"name": "RL_Contextual_LinUCB", "family": "Reinforcement Learning", "surrogate": "linucb", "acq": "ucb", "alpha": 1.75},
    {"name": "RL_Contextual_LinUCB_Quantile90", "family": "Reinforcement Learning", "surrogate": "linucb", "acq": "quantile", "tau": 0.90, "alpha": 1.75},
    {"name": "MOF_ChemBandit_UCB_TailWeighted", "family": "Reinforcement Learning", "surrogate": "mof_chem_bandit", "acq": "ucb", "beta": 2.75, "ridge": 1.0, "success_weight": 2.0, "top_weight": 4.0, "top_quantile": 0.80},
    {"name": "MOF_ChemBandit_TS_TailWeighted", "family": "Reinforcement Learning", "surrogate": "mof_chem_bandit", "acq": "ts", "ridge": 1.0, "success_weight": 2.0, "top_weight": 4.0, "top_quantile": 0.80},
    {"name": "MOF_ChemBandit_Quantile90_TailWeighted", "family": "Reinforcement Learning", "surrogate": "mof_chem_bandit", "acq": "quantile", "tau": 0.90, "ridge": 1.0, "success_weight": 2.0, "top_weight": 4.0, "top_quantile": 0.80},
    {"name": "MOF_ChemBandit_CVaR90_TailWeighted", "family": "Reinforcement Learning", "surrogate": "mof_chem_bandit", "acq": "cvar", "tau": 0.90, "ridge": 1.0, "success_weight": 2.0, "top_weight": 4.0, "top_quantile": 0.80},
    {"name": "RANDOM_SEARCH", "family": "Baseline", "surrogate": "random", "acq": "random"},
    {"name": "LLM_PURE_gpt-4o", "family": "LLM", "surrogate": "llm", "in_other_script": True},
    {"name": "LLM_PURE_gpt-5.5", "family": "LLM", "surrogate": "llm", "in_other_script": True},
    {"name": "LLM_PURE_gpt-5-mini", "family": "LLM", "surrogate": "llm", "in_other_script": True},
    {"name": "HYBRID_GP_Matern52_EI_ALT_gpt-5.5", "family": "LLM-GP Hybrid", "surrogate": "hybrid_gp_llm", "in_other_script": True},
    {"name": "HYBRID_GP_Matern52_PI_ALT_gpt-5.5", "family": "LLM-GP Hybrid", "surrogate": "hybrid_gp_llm", "in_other_script": True},
]
assert len(METHOD_SPECS) == 39, f"expected 39 methods, got {len(METHOD_SPECS)}"


# ==============================================================================
# 5. One method x one run
# ==============================================================================

def run_single_method(oracle: np.ndarray, spec: Dict[str, Any], run_idx: int,
                       metrics_writer_rows: List[Dict[str, Any]], detail_writer) -> None:
    method = spec["name"]
    family = spec.get("family", "")
    seed_base = (RANDOM_STATE + 10_000 * run_idx + stable_int(method)) % (2**31 - 1)
    rng = np.random.default_rng(seed_base)

    tried: set = set()
    history_conditions: List[Dict[str, int]] = []
    history_y_raw: List[float] = []
    best_q = 0.0

    def add_experiment(cond: Dict[str, int]) -> float:
        nonlocal best_q
        t = tuple(cond[c] for c in FEATURES)
        tried.add(t)
        q = q_of(oracle, cond)
        history_conditions.append(cond)
        history_y_raw.append(q)
        best_q = max(best_q, q)
        return q

    for _ in range(INITIAL_EXPERIMENTS):
        c = {f: int(rng.integers(lo, hi + 1)) for f, (lo, hi) in FEATURE_RANGES.items()}
        q = add_experiment(c)
        detail_writer.write(json.dumps({
            "method": method, "family": family, "run": run_idx, "iteration": 0,
            "selection_stage": "initial_random",
            **{f: c[f] for f in FEATURES}, "q_final": q,
        }, default=float) + "\n")
    detail_writer.flush()
    metrics_writer_rows.append({"method": method, "family": family, "run": run_idx, "iteration": 0,
                                 "n_experiments": len(history_conditions), "best_q": best_q, "status": "initial_random"})

    t_start = time.perf_counter()
    for iteration in range(1, N_ITERATIONS + 1):
        try:
            if spec["surrogate"] == "random":
                batch = sample_untried_candidates(rng, tried, BATCH_SIZE)
                status = "random"
            else:
                X_train = np.array([scale_row(c) for c in history_conditions])
                y_train_model = y_to_model_scale(np.array(history_y_raw))

                if np.nanstd(y_train_model) < 1e-12:
                    model = ConstantSurrogate().fit(X_train, y_train_model)
                    status = "constant_target_fallback"
                else:
                    model = make_surrogate(spec, seed_base + iteration).fit(X_train, y_train_model)
                    status = "ok"

                candidates = sample_untried_candidates(rng, tried, ACQ_CANDIDATES_PER_ITER)
                X_cand = np.array([scale_row(c) for c in candidates])
                mu, sd = model.predict(X_cand)
                y_best_model = float(np.max(y_train_model))
                acq = acquisition_scores(spec["acq"], mu, sd, y_best_model, rng, spec)

                diversity_lambda = spec.get("diversity_lambda", DIVERSITY_LAMBDA)
                chosen_local = greedy_diverse_batch(np.arange(len(candidates)), acq, X_cand, BATCH_SIZE, rng, diversity_lambda)
                batch = [candidates[i] for i in chosen_local]

        except Exception as e:
            batch = sample_untried_candidates(rng, tried, BATCH_SIZE)
            status = f"method_error_random_fallback: {str(e)[:200]}"
            print(f"[ERROR] {method} run={run_idx} iter={iteration}: {traceback.format_exc(limit=2)}")

        for bp, c in enumerate(batch):
            q = add_experiment(c)
            detail_writer.write(json.dumps({
                "method": method, "family": family, "run": run_idx, "iteration": iteration,
                "batch_position": bp, "selection_stage": "model" if spec["surrogate"] != "random" else "random",
                **{f: c[f] for f in FEATURES}, "q_final": q,
            }, default=float) + "\n")
        detail_writer.flush()

        metrics_writer_rows.append({"method": method, "family": family, "run": run_idx, "iteration": iteration,
                                     "n_experiments": len(history_conditions), "best_q": best_q, "status": status})

        if iteration % 10 == 0 or iteration == N_ITERATIONS:
            elapsed = time.perf_counter() - t_start
            print(f"[{method} run={run_idx}] iter {iteration:>2}/{N_ITERATIONS} n={len(history_conditions):>3} "
                  f"best_q={best_q:>10,.0f} status={status:<28} elapsed={elapsed:.1f}s")


# ==============================================================================
# 6. Main
# ==============================================================================

def main() -> None:
    global N_RUNS, N_ITERATIONS

    if SMOKE_TEST:
        print("=== SMOKE_TEST True: 1 run x 3 iterations, first 3 non-skipped methods only ===")
        N_RUNS_LOCAL = 1
        N_ITERATIONS = 3
    else:
        N_RUNS_LOCAL = N_RUNS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / ("smoke_metrics.csv" if SMOKE_TEST else "unified_bo_iteration_metrics.csv")
    detail_path = OUTPUT_DIR / ("smoke_detail.jsonl" if SMOKE_TEST else "unified_bo_selected_conditions.jsonl")

    oracle = build_oracle_lookup()

    done_pairs = set()
    prev_rows: List[Dict[str, Any]] = []
    if metrics_path.exists() and not SMOKE_TEST:
        prev = pd.read_csv(metrics_path)
        for (m, r), g in prev.groupby(["method", "run"]):
            if g["iteration"].max() >= N_ITERATIONS:
                done_pairs.add((m, r))
        prev_rows = prev.to_dict("records")
        print(f"[resume] {len(done_pairs)} (method, run) pair(s) already complete")

    all_rows: List[Dict[str, Any]] = list(prev_rows)
    smoke_method_budget = 3

    with open(detail_path, "a", encoding="utf-8") as detail_writer:
        for run_idx in range(RUN_IDX_START, RUN_IDX_START + N_RUNS_LOCAL):
            n_smoke_done = 0
            for spec in METHOD_SPECS:
                method = spec["name"]

                if spec.get("in_other_script", False):
                    continue

                if method in METHODS_OF_OTHER_SCRIPTS:
                    continue

                if SMOKE_TEST:
                    if n_smoke_done >= smoke_method_budget:
                        continue
                    n_smoke_done += 1

                if (method, run_idx) in done_pairs:
                    print(f"[skip] {method} run={run_idx} already done")
                    continue

                run_single_method(oracle, spec, run_idx, all_rows, detail_writer)
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)
    print("\n=== DONE ===")
    print(f"Methods x runs completed: {len(df.groupby(['method','run']))}")


if __name__ == "__main__":
    main()
