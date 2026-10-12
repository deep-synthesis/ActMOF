"""pseudolab_common.py -- the 96 recorded MOF-321 syntheses (recording order) and the blending rules of the landscape.

X, max_intensity and FWHM are the measured data, as in ../../dataset_construction/build_design_grid.py; the functions
apply the neighbor rules and the thresholds of the landscape to a single condition. Used by landscape_lib.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORDERED_COLS = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}

X = np.array([
    [35,5,22,9,14],[68,7,15,12,24],[12,9,1,7,29],[36,10,24,3,18],[55,11,8,2,15],
    [69,6,23,1,20],[14,12,2,11,15],[42,7,27,6,17],[9,14,7,4,11],[45,6,11,8,28],
    [27,12,1,2,25],[41,14,5,3,16],
    [5,5,0,1,30],[5,15,0,1,30],[75,15,0,1,30],
    [16,7,22,3,20],[16,9,23,4,21],[15,7,12,11,19],
    [6,5,29,11,27],[62,7,26,11,26],[5,15,2,11,16],
    [33,10,7,11,18],[9,6,1,12,19],[48,14,3,12,23],
    [19,8,12,11,16],[15,7,14,11,24],[12,8,10,11,18],
    [15,7,13,11,18],[14,7,11,11,20],[7,6,20,11,15],
    [15,7,14,11,24],[15,6,10,11,14],[15,7,10,11,12],
    [15,8,12,11,19],[9,8,13,7,20],[21,6,13,12,19],
    [6,6,12,11,20],[15,8,9,12,19],[75,12,6,11,18],
    [25,13,14,7,24],[17,13,16,12,24],[38,13,16,8,24],
    [65,5,0,12,24],[74,7,23,1,16],[5,7,21,12,20],
    [15,7,16,12,29],[16,7,15,11,18],[62,10,12,10,17],
    [11,7,12,11,24],[17,7,16,11,27],[13,7,10,12,25],
    [13,6,13,11,24],[15,5,0,12,24],[21,5,26,11,23],
    [16,7,8,11,24],[28,7,0,12,30],[10,7,0,12,24],
    [23,7,30,11,24],[20,7,30,11,29],[23,7,14,11,25],
    [16,7,18,12,25],[17,7,18,12,24],[24,7,18,12,24],
    [16,7,14,11,30],[15,7,13,9,23],[15,7,7,7,26],
    [13,6,25,12,24],[13,7,11,11,23],[14,7,7,11,26],
    [15,6,23,12,24],[15,5,8,12,23],[15,6,14,12,21],
    [13,5,10,11,24],[17,6,11,11,22],[19,5,2,12,24],
    [6,6,19,12,30],[25,7,18,11,24],[27,7,27,12,26],
    [15,5,8,11,23],[15,5,11,10,21],[24,5,30,12,24],
    [13,7,15,12,25],[16,7,11,12,24],[14,6,15,12,26],
    [14,7,15,11,26],[18,7,10,12,23],[10,7,13,12,29],
    [23,7,2,12,19],[18,7,1,4,19],[19,6,4,12,25],
    [17,7,10,12,25],[16,6,6,5,23],[20,6,0,12,23],
    [17,7,20,11,21],[24,7,22,8,21],[25,7,19,10,27]
], dtype=np.int16)
max_intensity = np.array([
    0,0,785,0,0,605,0,0,182,0,880,0,
    1457,0,0,
    2345,960,16751,
    7575,0,0,
    8060,1318,920,
    0,23969,7305,
    7703,12434,0,
    23969,0,0,
    1103,0,17269,
    1077,903,0,
    0,0,0,
    0,0,0,
    31082,12791,0,
    28179,29258,24240,
    37307,15489,9906,
    21878,11411,688,
    18169,10048,12005,
    35900,35967,13410,
    2331,23301,11663,
    23658,17558,7418,
    32110,20121,19118,
    34595,17880,6978,
    0,8241,7910,
    26277,18112,4394,
    27452,16348,30033,
    28764,28299,1082,
    20334,19300,12874,
    28836,16918,11668,
    26346,17895,11657
], dtype=np.float32)
FWHM = np.array([
    30,30,2.2,30,30,30,30,30,30,30,3.9,30,
    0.32,30,30,
    0.37,1.1,0.22,
    0.18,30,30,
    0.25,0.40,4.6,
    30,0.21,0.15,
    0.27,0.24,30,
    0.21,30,30,
    0.8,30,0.21,
    0.48,2,30,
    30,30,30,
    30,30,30,
    0.2,0.3,30,
    0.26,0.29,0.24,
    0.18,0.28,0.26,
    0.21,0.22,0.6,
    0.2,0.22,0.24,
    0.22,0.21,0.22,
    0.3,0.29,0.23,
    0.28,0.26,0.29,
    0.3,0.21,0.25,
    0.14,0.22,0.16,
    30,0.26,0.23,
    0.22,0.29,0.31,
    0.24,0.28,0.23,
    0.29,0.22,0.41,
    0.27,0.32,0.21,
    0.26,0.31,0.30,
    0.28,0.31,0.34
], dtype=np.float32)


def load_unique_experiments() -> pd.DataFrame:
    """95 unique conditions after grouping the 96 raw measurements -- identical
    aggregation to build_design_grid.py's agg_exp."""
    exp_df = pd.DataFrame(X, columns=ORDERED_COLS)
    exp_df["Intensity_exp"] = max_intensity.astype(float)
    exp_df["FWHM_exp"] = FWHM.astype(float)

    def _calc_q(intensity, fwhm):
        if fwhm == 30 or intensity <= 0 or fwhm <= 0:
            return 0
        return int(round(float(intensity) / float(fwhm)))

    agg = exp_df.groupby(ORDERED_COLS, as_index=False).agg(
        Intensity_exp=("Intensity_exp", "mean"), FWHM_exp=("FWHM_exp", "mean"))
    agg["q_exp"] = [_calc_q(i, f) for i, f in zip(agg["Intensity_exp"], agg["FWHM_exp"])]
    return agg.sort_values(ORDERED_COLS).reset_index(drop=True)


def scale_features(arr: np.ndarray) -> np.ndarray:
    lo = np.array([FEATURE_RANGES[c][0] for c in ORDERED_COLS], dtype=float)
    hi = np.array([FEATURE_RANGES[c][1] for c in ORDERED_COLS], dtype=float)
    return (arr.astype(float) - lo) / (hi - lo)


def postprocess_and_round(I: float, F: float) -> tuple[float, float, float]:
    """Single-point version of build_design_grid.py's postprocess_and_round_vec."""
    if I < 3000.0:
        I, F = 0.0, 30.0
    elif F > 3.0:
        F = 30.0
    I_final = float(np.rint(I))
    if F == 30.0:
        F_final = 30.0
    elif F < 1.0:
        F_final = float(np.round(F, 2))
    else:
        F_final = float(np.round(F, 1))
    q = 0.0
    if F_final != 30.0 and F_final > 0 and I_final > 0:
        q = float(np.rint(I_final / F_final))
    return I_final, F_final, q


def apply_neighbor_rules(point: np.ndarray, raw_I: float, raw_F: float,
                          neighbor_params: np.ndarray, neighbor_I: np.ndarray, neighbor_F: np.ndarray
                          ) -> tuple[float, float, int]:
    """Single-point version of build_design_grid.py's apply_rules_batch.
    neighbor_* are the (94, ...) remaining real experiments used as the
    Rule-1/Rule-2 lookup for this held-out point. Returns (I_blend, F_blend, rule_flag)."""
    diffs = np.abs(point.astype(np.float32)[None, :] - neighbor_params.astype(np.float32))  # (94,5)

    r1_mask = np.all(diffs <= 1.0 + 1e-9, axis=1)
    if r1_mask.any():
        I_blend = 0.8 * neighbor_I[r1_mask].mean() + 0.2 * raw_I
        F_blend = 0.8 * neighbor_F[r1_mask].mean() + 0.2 * raw_F
        return I_blend, F_blend, 1

    le2 = np.all(diffs <= 2.0 + 1e-9, axis=1)
    eq2 = np.isclose(diffs, 2.0)
    count2 = eq2.sum(axis=1)
    not2 = ~eq2
    non2_max = np.max(np.where(not2, diffs, -np.inf), axis=1)
    others_le1 = non2_max <= 1.0 + 1e-9
    r2_mask = le2 & others_le1 & ((count2 == 1) | (count2 == 2))
    if r2_mask.any():
        I_blend = 0.5 * neighbor_I[r2_mask].mean() + 0.5 * raw_I
        F_blend = 0.5 * neighbor_F[r2_mask].mean() + 0.5 * raw_F
        return I_blend, F_blend, 2

    return raw_I, raw_F, 0


def reconstruct_q(point: np.ndarray, raw_I: float, raw_F: float,
                   neighbor_params: np.ndarray, neighbor_I: np.ndarray, neighbor_F: np.ndarray) -> float:
    """Full pipeline for ONE held-out point: neighbor-rule blend -> post-process -> q."""
    I_blend, F_blend, _ = apply_neighbor_rules(point, raw_I, raw_F, neighbor_params, neighbor_I, neighbor_F)
    _, _, q = postprocess_and_round(I_blend, F_blend)
    return q
