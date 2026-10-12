"""Paths of this repository and the benchmark code, shared by the scripts of this folder."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ACTMOF_POOL_DIR", str(REPO / "data"))
sys.path.insert(0, str(REPO / "benchmark" / "families"))
import core29_gp_ml_tree_rl_mof as core   # noqa: E402  (the benchmark loop, surrogates and acquisition functions)

import numpy as np   # noqa: E402

FEATURES = core.FEATURES
HERE = Path(__file__).resolve().parent


def seed_base(method, run):
    """seed of run `run` of `method` in the benchmark (independent starts)"""
    return (core.RANDOM_STATE + 10_000 * run + core.stable_int(method)) % (2**31 - 1)


def oracle():
    """the landscape as a 71 x 11 x 31 x 12 x 21 array of q (cached in this folder after the first call)"""
    f = HERE / "_oracle.npy"
    if not f.exists():
        np.save(f, core.build_oracle_lookup())
    return np.load(f)
