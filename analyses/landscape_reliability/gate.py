"""Fixed ActMOF reliability features and Experiment-A gate.

Definitions follow the original error-dataset construction workflow:
raw-coordinate Chebyshev d_min and five-RF-seed pre-threshold crossing rates.
"""

from __future__ import annotations

import numpy as np


DMIN_THRESHOLD = 4.0
BOUNDARY_THRESHOLD = 0.0


def compute_dmin(
    x: np.ndarray,
    experimental_x: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Nearest-experiment L-infinity distance in raw synthesis coordinates."""
    x = np.asarray(x)
    experimental_x = np.asarray(experimental_x)
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        diffs = np.abs(x[start:stop, None, :] - experimental_x[None, :, :])
        out[start:stop] = diffs.max(axis=2).min(axis=1)
    return out


def compute_boundary_features(
    x: np.ndarray,
    experimental_x: np.ndarray,
    experimental_intensity: np.ndarray,
    experimental_fwhm: np.ndarray,
    intensity_models: list,
    fwhm_models: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute established B_I/B_F crossing rates for one candidate chunk.

    Each of the five RF seeds is blended through the original local-neighbor
    rules. B_I is the fraction with pre-threshold intensity below 3000; B_F is
    the fraction with intensity at least 3000 and pre-threshold FWHM above 3.
    """
    x = np.asarray(x)
    experimental_x = np.asarray(experimental_x)
    diffs = np.abs(x[:, None, :] - experimental_x[None, :, :])

    rule1 = np.all(diffs <= 1, axis=2)
    r1_has = rule1.any(axis=1)
    r1_count = np.maximum(rule1.sum(axis=1), 1)
    r1_i = (rule1 @ experimental_intensity) / r1_count
    r1_f = (rule1 @ experimental_fwhm) / r1_count

    at_two = diffs == 2
    rule2 = (
        np.all(diffs <= 2, axis=2)
        & np.isin(at_two.sum(axis=2), (1, 2))
        & np.all((diffs <= 1) | at_two, axis=2)
    )
    r2_has = rule2.any(axis=1)
    r2_count = np.maximum(rule2.sum(axis=1), 1)
    r2_i = (rule2 @ experimental_intensity) / r2_count
    r2_f = (rule2 @ experimental_fwhm) / r2_count

    cross_i = np.zeros(len(x), dtype=np.uint8)
    cross_f = np.zeros(len(x), dtype=np.uint8)
    for model_i, model_f in zip(intensity_models, fwhm_models):
        i_pre = model_i.predict(x)
        f_pre = model_f.predict(x)
        i_pre[r1_has] = 0.8 * r1_i[r1_has] + 0.2 * i_pre[r1_has]
        f_pre[r1_has] = 0.8 * r1_f[r1_has] + 0.2 * f_pre[r1_has]
        use_r2 = ~r1_has & r2_has
        i_pre[use_r2] = 0.5 * r2_i[use_r2] + 0.5 * i_pre[use_r2]
        f_pre[use_r2] = 0.5 * r2_f[use_r2] + 0.5 * f_pre[use_r2]
        cross_i += i_pre < 3000.0
        cross_f += (i_pre >= 3000.0) & (f_pre > 3.0)

    n_models = len(intensity_models)
    return cross_i.astype(np.float32) / n_models, cross_f.astype(np.float32) / n_models


def classify_gate(
    d_min: np.ndarray,
    b_i: np.ndarray,
    b_f: np.ndarray,
    dmin_threshold: float = DMIN_THRESHOLD,
    boundary_threshold: float = BOUNDARY_THRESHOLD,
) -> np.ndarray:
    """Return True only for sparse AND boundary-unstable candidates."""
    boundary_risk = np.maximum(b_i, b_f)
    return (np.asarray(d_min) > dmin_threshold) & (boundary_risk > boundary_threshold)
