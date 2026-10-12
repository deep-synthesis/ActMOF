"""landscape_lib.py -- rebuild the full 6,101,172-condition landscape under other construction choices (SI Section S4.4).

The reference construction (variant "V0") is that of ../../dataset_construction/build_design_grid.py:
  (i)   two RandomForestRegressor(n_estimators=400, random_state=42) on the integer codes, one for the intensity I,
        one for the FWHM;
  (ii)  Rule 1: anchors at Chebyshev distance <= 1 -> w1 * mean + (1 - w1) * model, w1 = 0.8; otherwise Rule 2:
        anchors with all |differences| <= 2 and one or two variables at exactly 2 -> w2 * mean + (1 - w2) * model, w2 = 0.5;
  (iii) I < 3000 -> (0, 30); FWHM > 3 -> FWHM 30; rounding; q = rint(I / FWHM); anchors keep their measured values.
A variant changes one of these choices. The neighbor search is done with fixed offset stencils, so that one full
grid takes about a minute.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseudolab_common import X as X_RAW, max_intensity as I_RAW, FWHM as F_RAW  # noqa: E402

COLS = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
LO = np.array([5, 5, 0, 1, 10], dtype=np.int64)
HI = np.array([75, 15, 30, 12, 30], dtype=np.int64)
SHAPE = tuple(int(h - l + 1) for l, h in zip(LO, HI))          # (71, 11, 31, 12, 21)
N_GRID = int(np.prod(SHAPE))                                   # 6,101,172


# ---------------------------------------------------------------- seeds
def seeds_recording_order():
    """95 unique seeds in the recording order of Table S3 (first occurrence
    kept; entries 26 and 31 are identical, so entry 31 is dropped).
    Returns (params E x 5 int, I E, F E, entry_no E)."""
    seen, keep = {}, []
    for i, row in enumerate(X_RAW):
        t = tuple(int(v) for v in row)
        if t in seen:
            seen[t].append(i)
            continue
        seen[t] = [i]
        keep.append(i)
    P = X_RAW[keep].astype(np.int64)
    # duplicates are averaged exactly as build_design_grid.py's groupby-mean
    I = np.array([I_RAW[seen[tuple(int(v) for v in X_RAW[i])]].astype(float).mean() for i in keep])
    F = np.array([F_RAW[seen[tuple(int(v) for v in X_RAW[i])]].astype(float).mean() for i in keep])
    return P, I, F, np.array(keep) + 1


def calc_q(I, F):
    I = np.asarray(I, float); F = np.asarray(F, float)
    q = np.zeros_like(I)
    ok = (F != 30) & (I > 0) & (F > 0)
    q[ok] = np.rint(I[ok] / F[ok])
    return q


# ---------------------------------------------------------------- grid
def flat_index(P: np.ndarray) -> np.ndarray:
    return np.ravel_multi_index(tuple((P - LO).T), SHAPE)


_GRID = None
def grid_points() -> np.ndarray:
    """All grid points as int16 (N, 5), C order == build_design_grid.py's nested loops."""
    global _GRID
    if _GRID is None:
        axes = [np.arange(l, h + 1, dtype=np.int16) for l, h in zip(LO, HI)]
        mesh = np.meshgrid(*axes, indexing="ij")
        _GRID = np.stack([m.ravel() for m in mesh], axis=1)
    return _GRID


def _stencils():
    offs = np.array(list(itertools.product(range(-2, 3), repeat=5)), dtype=np.int64)
    a = np.abs(offs)
    r1 = offs[np.all(a <= 1, axis=1)]
    n2 = (a == 2).sum(axis=1)
    r2 = offs[(n2 >= 1) & (n2 <= 2)]          # others automatically <= 1
    return r1, r2
R1_OFFS, R2_OFFS = _stencils()


def neighbour_sums(P: np.ndarray, I: np.ndarray, F: np.ndarray, offs: np.ndarray):
    """For every grid point: count, sum(I), sum(F) over seeds whose offset to it is in `offs`."""
    cnt = np.zeros(N_GRID, np.int64); sI = np.zeros(N_GRID, np.float32); sF = np.zeros(N_GRID, np.float32)
    for p, i, f in zip(P, I, F):
        pts = p[None, :] + offs
        ok = np.all((pts >= LO) & (pts <= HI), axis=1)
        idx = flat_index(pts[ok])
        cnt[idx] += 1; sI[idx] += np.float32(i); sF[idx] += np.float32(f)  # float32 as in build_design_grid.py (exp_I/exp_F are float32)
    return cnt, sI, sF


# ---------------------------------------------------------------- emulators
def make_models(kind: str, random_state: int = 42):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    if kind == "rf":
        mk = lambda: RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)
    elif kind == "gb":
        mk = lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                               subsample=1.0, random_state=random_state)
    elif kind == "knn":
        scale = FunctionTransformer(lambda Z: (np.asarray(Z, float) - LO) / (HI - LO))
        mk = lambda: make_pipeline(scale, KNeighborsRegressor(n_neighbors=5, weights="distance"))
    else:
        raise ValueError(kind)
    return mk(), mk()


@dataclass
class Variant:
    vid: str
    group: str
    label: str
    model: str = "rf"
    rf_state: int = 42
    w1: float = 0.8
    w2: float = 0.5
    I_thr: float = 3000.0
    F_thr: float = 3.0
    train_idx: Optional[np.ndarray] = None   # indices into the 95 seeds (with repeats for bootstrap)
    note: str = ""


def build_landscape(v: Variant, predict_batch: int = 1_000_000):
    """Returns dict(q=float64 N, I=..., F=..., measured_idx=flat idx of seeds kept as measured,
    model fit info)."""
    P, I, F, _ = seeds_recording_order()
    tr = np.arange(len(P)) if v.train_idx is None else np.asarray(v.train_idx)
    uniq = np.unique(tr)
    # model training set: build_design_grid.py trains on the 95 unique (sorted) rows;
    # row order does not change a RF with fixed random_state? It does (bootstrap draws by
    # position), so keep the same sorted order as agg_exp for V0 reproducibility.
    Ptr, Itr, Ftr = P[tr], I[tr], F[tr]
    order = np.lexsort(Ptr.T[::-1])
    Ptr, Itr, Ftr = Ptr[order], Itr[order], Ftr[order]
    mI, mF = make_models(v.model, v.rf_state)
    mI.fit(Ptr, Itr); mF.fit(Ptr, Ftr)

    G = grid_points()
    rfI = np.empty(N_GRID); rfF = np.empty(N_GRID)
    for s in range(0, N_GRID, predict_batch):
        rfI[s:s + predict_batch] = mI.predict(G[s:s + predict_batch])
        rfF[s:s + predict_batch] = mF.predict(G[s:s + predict_batch])
    # Clip to the range of the training targets. A forest or kNN average can never leave
    # this range (no-op for V0); gradient boosting can, and an extrapolated FWHM of ~0.01
    # deg would give unphysical q of several million.
    np.clip(rfI, Itr.min(), Itr.max(), out=rfI)
    np.clip(rfF, Ftr.min(), Ftr.max(), out=rfF)

    Pu, Iu, Fu = P[uniq], I[uniq], F[uniq]
    c1, sI1, sF1 = neighbour_sums(Pu, Iu, Fu, R1_OFFS)
    c2, sI2, sF2 = neighbour_sums(Pu, Iu, Fu, R2_OFFS)
    Ib, Fb = rfI.copy(), rfF.copy()
    m1 = c1 > 0
    Ib[m1] = v.w1 * (sI1[m1] / c1[m1]) + (1 - v.w1) * rfI[m1]
    Fb[m1] = v.w1 * (sF1[m1] / c1[m1]) + (1 - v.w1) * rfF[m1]
    m2 = (~m1) & (c2 > 0)
    Ib[m2] = v.w2 * (sI2[m2] / c2[m2]) + (1 - v.w2) * rfI[m2]
    Fb[m2] = v.w2 * (sF2[m2] / c2[m2]) + (1 - v.w2) * rfF[m2]

    # hard boundaries + rounding (postprocess_and_round_vec)
    low = Ib < v.I_thr
    Ib[low] = 0.0; Fb[low] = 30.0
    Fb[(Fb > v.F_thr) & ~low] = 30.0
    If = np.rint(Ib)
    Ff = np.where(Fb == 30.0, 30.0, np.where(Fb < 1.0, np.round(Fb, 2), np.round(Fb, 1)))
    q = np.zeros(N_GRID)
    ok = (Ff != 30) & (If > 0) & (Ff > 0)
    q[ok] = np.rint(If[ok] / Ff[ok])

    # measured seed positions keep measured values
    midx = flat_index(Pu)
    q[midx] = calc_q(Iu, Fu)
    rule = np.zeros(N_GRID, np.int8); rule[m1] = 1; rule[m2] = 2
    return {"q": q, "rule": rule, "measured_idx": midx, "models": (mI, mF)}


# ---------------------------------------------------------------- landscape metrics
_ALL_SEED_IDX = flat_index(seeds_recording_order()[0])
EMU_MASK = np.ones(N_GRID, bool); EMU_MASK[_ALL_SEED_IDX] = False      # 6,101,077 emulated points


def tier_thresholds(q):
    e = q[EMU_MASK]
    return {k: float(np.quantile(e, p)) for k, p in
            [("top5", 0.95), ("top1", 0.99), ("top01", 0.999), ("top001", 0.9999)]}


def basins(top_mask_flat):
    lab, n = ndimage.label(top_mask_flat.reshape(SHAPE), structure=np.ones((3,) * 5, bool))
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int(sizes.max())
