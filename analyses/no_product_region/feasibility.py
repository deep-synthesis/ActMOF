"""feasibility.py -- policies that model "product / no product" separately (SI Section S4.7).

    B0   Matern-5/2 GP with EI on y = log(1 + q); an experiment without product enters as y = 0 (the baseline)
    M1   score = p(x) * EI+(x): a GP classifier gives the probability p of a product, a GP regression fitted to the
         crystalline experiments only gives EI+ (see hurdle_ei.md)
    M3   an experiment without product is entered with the lowest y of the products found so far; then GP with EI
    M1_avoid, M1_gamma0.5, M1_gamma2   variants of M1, not reported in the paper

All policies use the loop, surrogates and batch selection of ../../benchmark/families/core29_gp_ml_tree_rl_mof.py and
shared starts: the seed depends on the run index only, so that in every run all policies begin with the same three
experiments. Before the first product, all policies follow the same rule. Budget: 3 + 39 x 3 experiments.
"""
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # root of this repository
FAM = ROOT / "benchmark" / "families"
os.environ.setdefault("ACTMOF_POOL_DIR", str(ROOT / "data"))
sys.path.insert(0, str(FAM))
import core29_gp_ml_tree_rl_mof as core     # noqa: E402

from sklearn.gaussian_process import GaussianProcessClassifier   # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern   # noqa: E402

FEATURES = core.FEATURES
XI = 0.01
SPEC_B0 = next(s for s in core.METHOD_SPECS if s["name"] == "GP_Matern52_EI_BASELINE")

METHODS = {                       # name -> (policy, gamma)
    "B0": ("b0", 1.0),
    "M1": ("m1", 1.0),
    "M3": ("m3", 1.0),
    "M1_avoid": ("m1_avoid", 1.0),        # exploratory
    "M1_gamma0.5": ("m1", 0.5),           # exploratory
    "M1_gamma2": ("m1", 2.0),             # exploratory
}


# ------------------------------------------------------------------ oracle
def load_oracle(cache=None):
    cache = Path(cache) if cache else HERE / "_oracle.npy"
    if not cache.exists():
        np.save(cache, core.build_oracle_lookup())
    return np.load(cache)


def oracle_sha256():
    """SHA-256 of each design_grid_part*.csv (the landscape has no single file)"""
    out = {}
    for f in sorted(Path(os.environ["ACTMOF_POOL_DIR"]).glob("design_grid_part*.csv")):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[f.name] = h.hexdigest()
    return out


# ------------------------------------------------------------------ building blocks
def ei_pos(mu, sd, y_best, xi=XI):
    """EI_+ (Eq. S-EI) -- identical to core.gaussian_ei"""
    return core.gaussian_ei(mu, sd, y_best, xi=xi)


def hurdle_ei(p, mu, sd, y_best, xi=XI):
    """Acquisition of M1: p * EI_+  (exact EI of the hurdle model, docs/hurdle_ei.md)"""
    return np.asarray(p, dtype=float) * ei_pos(mu, sd, y_best, xi)


def make_gpc(seed):
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.5] * 5, length_scale_bounds=(0.01, 10), nu=2.5)
    return GaussianProcessClassifier(kernel=k, n_restarts_optimizer=1, random_state=seed)


def fit_regressor(X, y, seed):
    if len(y) >= 2 and np.nanstd(y) >= 1e-12:
        return core.make_surrogate(SPEC_B0, seed).fit(X, y), "gpr"
    return core.ConstantSurrogate().fit(X, y), "const"


BOUNDS = dict(amp=(1e-3, 1e3), ls=(0.01, 10.0))


def _bound_flags(amp, ls):
    """names of fitted GPC kernel parameters within 1 % (relative) of a bound"""
    fl = []
    for name, v, (lo, hi) in [("amp", amp, BOUNDS["amp"])] + [(f"ls{i+1}", l, BOUNDS["ls"]) for i, l in enumerate(ls)]:
        if np.isclose(v, lo, rtol=1e-2) or np.isclose(v, hi, rtol=1e-2):
            fl.append(name)
    return ";".join(fl)


def m1_scores(X, q_obs, Xc, seed, gamma=1.0, Xn=None):
    """M1 acquisition on candidates Xc (used for selection). Xn: optional DIAGNOSTIC points (never used for
    selection) scored with the same fitted models. Returns (acq, info); acq is None if n_pos == 0 (cold start)."""
    z = (q_obs > 0).astype(int)
    n_pos, n_neg = int(z.sum()), int((1 - z).sum())
    info = dict(n_pos=n_pos, gpc_fitted=False, gpc_failed=False, gpc_warn=0, n_train=np.nan)
    if n_pos == 0:
        return None, info
    nC = len(Xc)
    Xall = Xc if Xn is None or len(Xn) == 0 else np.vstack([Xc, Xn])
    if n_neg == 0:
        p = np.ones(len(Xall))
    else:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gpc = make_gpc(seed).fit(X, z)
                info["n_train"] = int(gpc.base_estimator_.X_train_.shape[0])
                assert info["n_train"] == len(q_obs) == len(X), "GPC training-set size != number of queries"
                p = gpc.predict_proba(Xall)[:, 1]
            info["gpc_warn"] = len(w)
            info["gpc_fitted"] = True
            k = gpc.kernel_
            info["gpc_amp"] = float(k.k1.constant_value)
            ls = np.atleast_1d(k.k2.length_scale)
            for i, l in enumerate(ls):
                info[f"gpc_ls{i+1}"] = float(l)
            info["gpc_bounds"] = _bound_flags(info["gpc_amp"], ls)
            if not np.all(np.isfinite(p)):
                raise FloatingPointError("non-finite p")
        except AssertionError:
            raise                                          # data-flow violations are never swallowed
        except Exception as e:                         # logged, fallback p = 1 (= plain EI_+)
            info["gpc_failed"] = True
            info["gpc_err"] = repr(e)[:120]
            p = np.ones(len(Xall))
    pos = z == 1
    y_pos = core.y_to_model_scale(q_obs[pos])
    reg, kind = fit_regressor(X[pos], y_pos, seed)
    mu, sd = reg.predict(Xall)
    y_best = float(np.max(y_pos))
    ei = ei_pos(mu, sd, y_best)
    a_all = (p ** gamma) * ei
    info.update(reg=kind, max_p=float(np.max(p[:nC])), max_ei_pos=float(np.max(ei[:nC])))
    info["c"] = dict(p=p[:nC], mu=mu[:nC], sd=sd[:nC], ei=ei[:nC])
    if len(Xall) > nC:
        info["nb"] = dict(acq=a_all[nC:], p=p[nC:], mu=mu[nC:], sd=sd[nC:], ei=ei[nC:])
    return a_all[:nC], info


def m3_scores(X, q_obs, Xc, seed):
    """M3: failures get y = min crystalline y observed so far; standard GP-EI with y+ = max y"""
    n_pos = int((q_obs > 0).sum())
    y = core.y_to_model_scale(q_obs)
    if n_pos == 0:
        model = core.ConstantSurrogate().fit(X, y)
    else:
        y_floor = float(np.min(y[q_obs > 0]))
        y = np.where(q_obs > 0, y, y_floor)
        model, _ = fit_regressor(X, y, seed)
    mu, sd = model.predict(Xc)
    ei = ei_pos(mu, sd, float(np.max(y)))
    return ei, dict(n_pos=n_pos, max_ei_pos=float(np.max(ei)))


def avoid_scores(X, Xc):
    """M1-avoid (exploratory, n_pos == 0): maximin scaled distance to all observed failures"""
    d = np.sqrt(((Xc[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)


# ------------------------------------------------------------------ diagnostics helpers
def cheb_to_set(pts, ref):
    """Chebyshev distance (integer grid steps) from each row of pts to the nearest row of ref; NaN if ref empty"""
    if len(ref) == 0:
        return np.full(len(pts), np.nan)
    return np.abs(pts[:, None, :] - ref[None, :, :]).max(-1).min(1)


_OFFS = np.array(np.meshgrid(*[np.arange(-2, 3)] * 5, indexing="ij")).reshape(5, -1).T     # 3125 x 5
_LO = np.array([core.FEATURE_RANGES[f][0] for f in FEATURES])
_HI = np.array([core.FEATURE_RANGES[f][1] for f in FEATURES])
_SC = (_HI - _LO).astype(float)


def neighbourhood(cryst_int, tried):
    """untried grid points within Chebyshev distance <= 2 of any observed crystalline point (diagnostic only)"""
    if len(cryst_int) == 0:
        return np.zeros((0, 5), dtype=int)
    pts = (cryst_int[:, None, :] + _OFFS[None]).reshape(-1, 5)
    pts = pts[((pts >= _LO) & (pts <= _HI)).all(1)]
    pts = np.unique(pts, axis=0)
    keep = np.array([tuple(r) not in tried for r in pts])
    return pts[keep]


# ------------------------------------------------------------------ one run
def run_one(method, run, oracle, n_iter=39, verbose=False):
    """Returns (query_rows, iter_rows). Iteration 0 = the 3 starting experiments.
    Data-flow assertions (every iteration): number of queries = 3 + 3(it-1); n_neg = previous n_neg + failures of the
    previous batch; GPC training-set size = number of queries (inside m1_scores)."""
    policy, gamma = METHODS[method]
    sb = (core.RANDOM_STATE + 10_000 * run) % (2 ** 31 - 1)     # SI S4.4 paired seed: depends on the run only
    rng = np.random.default_rng(sb)
    tried, conds, q_obs = set(), [], []
    rows, irows = [], []
    best = 0.0

    def add(c, it, meta):
        nonlocal best
        tried.add(tuple(c[f] for f in FEATURES))
        q = core.q_of(oracle, c)
        conds.append(c)
        q_obs.append(q)
        best = max(best, q)
        rows.append(dict(method=method, run=run, iteration=it, **{f"x{i+1}": c[f] for i, f in enumerate(FEATURES)},
                         q=q, y=float(np.log1p(q)), best_so_far_q=best, n_pos=0, **meta))

    nan = dict(p_chosen=np.nan, ei_pos_chosen=np.nan, acquisition_chosen=np.nan, gpc_fitted=False, wall_time_s=np.nan,
               mu_pos_chosen=np.nan, sd_pos_chosen=np.nan, cheb_to_cryst=np.nan)
    for _ in range(core.INITIAL_EXPERIMENTS):
        add({f: int(rng.integers(lo, hi + 1)) for f, (lo, hi) in core.FEATURE_RANGES.items()}, 0, dict(nan))
    n0 = int(sum(v > 0 for v in q_obs))
    for r in rows:
        r["n_pos"] = n0
    prev_neg_expected = int(sum(v <= 0 for v in q_obs))   # n_neg after the starting batch

    for it in range(1, n_iter + 1):
        t0 = time.time()
        X = np.array([core.scale_row(c) for c in conds])
        qo = np.array(q_obs)
        n_neg = int((qo <= 0).sum())
        # ---- data-flow checks
        assert len(conds) == len(q_obs) == 3 + 3 * (it - 1), (method, run, it, len(conds))
        assert n_neg == prev_neg_expected, (method, run, it, n_neg, prev_neg_expected)
        cand = core.sample_untried_candidates(rng, tried, core.ACQ_CANDIDATES_PER_ITER)
        Xc = np.array([core.scale_row(c) for c in cand])
        cand_int = np.array([[c[f] for f in FEATURES] for c in cand])
        cryst_int = np.array([[c[f] for f in FEATURES] for c, v in zip(conds, q_obs) if v > 0], dtype=int).reshape(-1, 5)
        info = dict(n_pos=int((qo > 0).sum()), gpc_fitted=False, gpc_failed=False, gpc_warn=0)
        p = ei = mu_c = sd_c = None
        nb_rec = {}
        seed = sb + it

        if policy == "m3":
            acq, i3 = m3_scores(X, qo, Xc, seed)
            info.update(i3); ei = acq
        elif policy in ("m1", "m1_avoid"):
            Xn_int = neighbourhood(cryst_int, tried)
            Xn = (Xn_int - _LO) / _SC if len(Xn_int) else None
            acq, i1 = m1_scores(X, qo, Xc, seed, gamma, Xn)
            info.update({k: v for k, v in i1.items() if k not in ("c", "nb")})
            if acq is None:                                         # n_pos == 0: cold start
                if policy == "m1_avoid":
                    acq = avoid_scores(X, Xc)
                else:                                               # same as B0: constant surrogate
                    y = core.y_to_model_scale(qo)
                    mu, sd = core.ConstantSurrogate().fit(X, y).predict(Xc)
                    acq = ei_pos(mu, sd, float(np.max(y)))
            else:
                c = i1["c"]
                p, ei, mu_c, sd_c = c["p"], c["ei"], c["mu"], c["sd"]
                if "nb" in i1:
                    nb = i1["nb"]
                    j = int(np.argmax(nb["acq"]))
                    nb_rec = dict(nb_n=len(nb["acq"]), nb_acq=float(nb["acq"][j]), nb_p=float(nb["p"][j]),
                                  nb_mu=float(nb["mu"][j]), nb_sd=float(nb["sd"][j]), nb_ei=float(nb["ei"][j]),
                                  nb_cheb=float(cheb_to_set(Xn_int[j:j + 1], cryst_int)[0]))
        else:                                                        # B0 -- identical to replay.py
            y = core.y_to_model_scale(qo)
            if np.nanstd(y) < 1e-12:
                model = core.ConstantSurrogate().fit(X, y)
            else:
                model = core.make_surrogate(SPEC_B0, seed).fit(X, y)
            mu, sd = model.predict(Xc)
            acq = ei_pos(mu, sd, float(np.max(y)))
            ei = acq

        ch = core.greedy_diverse_batch(np.arange(len(cand)), acq, Xc, core.BATCH_SIZE, rng, core.DIVERSITY_LAMBDA)
        dt = time.time() - t0
        cheb = cheb_to_set(cand_int[ch], cryst_int)
        n_new_neg = 0
        for k, i in enumerate(ch):
            add(cand[i], it, dict(p_chosen=float(p[i]) if p is not None else np.nan,
                                  ei_pos_chosen=float(ei[i]) if ei is not None else np.nan,
                                  acquisition_chosen=float(acq[i]), gpc_fitted=info["gpc_fitted"], wall_time_s=dt,
                                  mu_pos_chosen=float(mu_c[i]) if mu_c is not None else np.nan,
                                  sd_pos_chosen=float(sd_c[i]) if sd_c is not None else np.nan,
                                  cheb_to_cryst=float(cheb[k])))
            rows[-1]["n_pos"] = info["n_pos"]          # n_pos before the batch (what the policy saw)
            n_new_neg += int(q_obs[-1] <= 0)
        prev_neg_expected = n_neg + n_new_neg
        irows.append(dict(method=method, run=run, iteration=it, n_pos=info["n_pos"], n_neg=n_neg,
                          n_queries=len(qo), n_train=info.get("n_train", np.nan), dataflow_ok=True,
                          gpc_fitted=info["gpc_fitted"], gpc_failed=info["gpc_failed"], gpc_warn=info["gpc_warn"],
                          gpc_amp=info.get("gpc_amp", np.nan),
                          **{f"gpc_ls{i+1}": info.get(f"gpc_ls{i+1}", np.nan) for i in range(5)},
                          gpc_bounds=info.get("gpc_bounds", ""),
                          max_p=float(np.max(p)) if p is not None else np.nan,
                          max_ei_pos=float(np.max(ei)) if ei is not None else np.nan,
                          wall_time_s=dt,
                          pool_first5=";".join("-".join(str(c[f]) for f in FEATURES) for c in cand[:5]),
                          chosen_idx=",".join(map(str, ch)), **nb_rec))
        if verbose:
            print(method, run, it, info["n_pos"], f"{dt:.1f}s", f"best={best:.0f}", flush=True)
    return rows, irows
