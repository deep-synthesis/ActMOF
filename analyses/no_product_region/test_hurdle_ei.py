"""test_hurdle_ei.py -- unit tests of feasibility.py: the identity of hurdle_ei.md (Monte-Carlo check) and edge cases
of the policies M1 and M3.    python -m pytest test_hurdle_ei.py -q
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark" / "families"))
import feasibility as F   # noqa: E402

warnings.filterwarnings("ignore")


def test_hurdle_ei_monte_carlo():
    rng = np.random.default_rng(0)
    n, S, ok = 1000, 200_000, 0
    for _ in range(n):
        p = rng.uniform(0, 1)
        mu = rng.uniform(0, 8)
        sd = rng.uniform(0.05, 3)
        yb = rng.uniform(0, 10)
        zero = rng.random(S) >= p
        y = np.where(zero, 0.0, rng.normal(mu, sd, S))
        imp = np.maximum(y - yb - F.XI, 0.0)
        mc, se = imp.mean(), imp.std(ddof=1) / np.sqrt(S)
        val = float(F.hurdle_ei(p, mu, sd, yb))
        assert np.isfinite(val)
        # se == 0 when no sample improves (y+ far above mu): then the true EI is < 1e-6, below MC resolution;
        # an absolute tolerance of 1e-6 covers that case only (the 3-se criterion is unchanged otherwise)
        ok += abs(val - mc) <= 3 * se + 1e-6
    assert ok / n >= 0.99, ok / n


# ---------------------------------------------------------------- helpers
def _data(q_list, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((len(q_list), 5))
    Xc = rng.random((200, 5))
    return X, np.array(q_list, dtype=float), Xc


def test_m1_no_product():
    X, q, Xc = _data([0, 0, 0])
    acq, info = F.m1_scores(X, q, Xc, 0)
    assert acq is None and info["n_pos"] == 0 and not info["gpc_fitted"]


def test_m1_one_product():
    X, q, Xc = _data([0, 0, 500.0, 0])
    acq, info = F.m1_scores(X, q, Xc, 0)
    assert info["n_pos"] == 1 and info["reg"] == "const" and np.all(np.isfinite(acq))
    assert np.allclose(info["c"]["ei"], info["c"]["ei"][0])          # constant EI_+, p drives the ranking


def test_m1_all_crystalline():
    X, q, Xc = _data([10.0, 200.0, 3000.0])
    acq, info = F.m1_scores(X, q, Xc, 0)
    assert not info["gpc_fitted"] and np.allclose(info["c"]["p"], 1.0) and info["reg"] == "gpr"
    assert np.allclose(acq, info["c"]["ei"])


def test_m1_all_equal_y():
    X, q, Xc = _data([0, 50.0, 50.0, 50.0, 0])
    acq, info = F.m1_scores(X, q, Xc, 0)
    assert info["reg"] == "const" and info["gpc_fitted"] and np.all(np.isfinite(acq))


def test_m1_mixed_normal_case():
    X, q, Xc = _data([0, 0, 10.0, 400.0, 0, 3000.0])
    acq, info = F.m1_scores(X, q, Xc, 0)
    assert info["gpc_fitted"] and info["reg"] == "gpr"
    assert np.all((info["c"]["p"] >= 0) & (info["c"]["p"] <= 1)) and np.allclose(acq, info["c"]["p"] * info["c"]["ei"])


def test_m3_no_product_is_constant():
    X, q, Xc = _data([0, 0, 0])
    ei, info = F.m3_scores(X, q, Xc, 0)
    assert info["n_pos"] == 0 and np.allclose(ei, ei[0])


def test_m3_one_product_constant_surrogate():
    # one crystalline y, failures padded to the same value -> all y equal -> constant surrogate
    X, q, Xc = _data([0, 0, 500.0])
    ei, info = F.m3_scores(X, q, Xc, 0)
    assert info["n_pos"] == 1 and np.all(np.isfinite(ei)) and np.allclose(ei, ei[0])


def test_m3_floor_padding():
    X, q, Xc = _data([0, 20.0, 400.0, 0])
    ei, info = F.m3_scores(X, q, Xc, 0)
    assert info["n_pos"] == 2 and np.all(np.isfinite(ei))


def test_m1_diagnostic_neighbourhood_does_not_change_selection_scores():
    X, q, Xc = _data([0, 0, 10.0, 400.0, 0, 3000.0])
    a1, _ = F.m1_scores(X, q, Xc, 0)
    a2, i2 = F.m1_scores(X, q, Xc, 0, Xn=np.random.default_rng(3).random((50, 5)))
    assert np.allclose(a1, a2) and len(i2["nb"]["acq"]) == 50


def test_hurdle_ei_near_mu_monte_carlo():
    """y+ drawn within mu +/- 2 sigma (non-degenerate MC): original 3-SE criterion, no absolute tolerance, >= 99 %"""
    rng = np.random.default_rng(1)
    n, S, ok = 500, 200_000, 0
    for _ in range(n):
        p = rng.uniform(0.05, 1)
        mu = rng.uniform(0.5, 8)
        sd = rng.uniform(0.05, 3)
        yb = max(0.0, mu + rng.uniform(-2, 2) * sd)
        zero = rng.random(S) >= p
        y = np.where(zero, 0.0, rng.normal(mu, sd, S))
        imp = np.maximum(y - yb - F.XI, 0.0)
        mc, se = imp.mean(), imp.std(ddof=1) / np.sqrt(S)
        val = float(F.hurdle_ei(p, mu, sd, yb))
        assert np.isfinite(val) and se > 0
        ok += abs(val - mc) <= 3 * se
    assert ok / n >= 0.99, ok / n
