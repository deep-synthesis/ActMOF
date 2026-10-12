"""
replay.py -- re-execute one (method, run) of the benchmark loop with the original functions of
core29_gp_ml_tree_rl_mof.py, recording what the production run did not log: the posterior mean mu and
standard deviation sd (log1p scale) of every chosen point, the incumbent y_best, and the acquisition values.

Optional multiplicative observation noise (log-normal, mean 1, coefficient of variation `noise_cv`) turns the
free-query landscape into a noisy "wet-lab" oracle; the surrogate then learns from noisy q, while the best-so-far
is always scored on the true landscape q. The noise stream has its own generator, so with noise_cv = 0 the replay
is bit-identical to the production loop (checked against the logged selections).
"""
import numpy as np
import common as s

core = s.core
SPEC = {sp["name"]: sp for sp in core.METHOD_SPECS}


def replay(method, run, n_iter, oracle, noise_cv=0.0, keep_models_at=(), paired=False):
    """paired=True: seed from the run index only, so every method starts from the same batch and candidate stream"""
    spec = SPEC[method]
    sb = (core.RANDOM_STATE + 10_000 * run) % (2**31 - 1) if paired else s.seed_base(method, run)
    rng = np.random.default_rng(sb)
    rng_noise = np.random.default_rng(sb + 7_777_777)
    sig = np.sqrt(np.log1p(noise_cv ** 2)) if noise_cv > 0 else 0.0

    tried, conds, y_obs, y_true = set(), [], [], []

    def add(c):
        t = tuple(c[f] for f in core.FEATURES)
        tried.add(t)
        qt = core.q_of(oracle, c)
        qo = qt * np.exp(rng_noise.normal(-sig ** 2 / 2, sig)) if (sig > 0 and qt > 0) else qt
        conds.append(c); y_true.append(qt); y_obs.append(qo)

    for _ in range(core.INITIAL_EXPERIMENTS):
        add({f: int(rng.integers(lo, hi + 1)) for f, (lo, hi) in core.FEATURE_RANGES.items()})

    rec, kept = [], {}
    best_true = [max(y_true)]
    for it in range(1, n_iter + 1):
        X = np.array([core.scale_row(c) for c in conds])
        y = core.y_to_model_scale(np.array(y_obs))
        if np.nanstd(y) < 1e-12:
            model = core.ConstantSurrogate().fit(X, y); status = "constant_target_fallback"
        else:
            model = core.make_surrogate(spec, sb + it).fit(X, y); status = "ok"
        cand = core.sample_untried_candidates(rng, tried, core.ACQ_CANDIDATES_PER_ITER)
        Xc = np.array([core.scale_row(c) for c in cand])
        mu, sd = model.predict(Xc)
        yb = float(np.max(y))
        acq = core.acquisition_scores(spec["acq"], mu, sd, yb, rng, spec)
        lam = spec.get("diversity_lambda", core.DIVERSITY_LAMBDA)
        ch = core.greedy_diverse_batch(np.arange(len(cand)), acq, Xc, core.BATCH_SIZE, rng, lam)
        if it in keep_models_at:
            kept[it] = dict(model=model, X=X.copy(), y=y.copy(), y_best=yb, conds=list(conds),
                            cand=Xc, mu=mu, sd=sd, acq=acq, chosen=ch)
        for bp, i in enumerate(ch):
            rec.append(dict(method=method, run=run, iteration=it, bp=bp, status=status, y_best=yb,
                            mu=float(mu[i]), sd=float(sd[i]), acq=float(acq[i]),
                            **{f: cand[i][f] for f in core.FEATURES}))
        for i in ch:
            add(cand[i])
        best_true.append(max(y_true))
    return rec, best_true, kept
