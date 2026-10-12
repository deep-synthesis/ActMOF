"""run_warmstart.py -- warm-start runs of SI Section S4.6.

Two starts, 20 runs each, for six strategies and random search:
    faithful        the three conditions proposed by the MOF-321 prior, (18,7,17,10,21), (23,6,15,10,20),
                    (20,7,19,12,29), identical in all runs
    same_quantile   three conditions drawn from the band between the top-1% and top-0.1% thresholds of the landscape,
                    one draw per run (the band is built by prepare_percentile_band.py)

    python prepare_percentile_band.py
    python run_warmstart.py --arm faithful --replicate-start 0 --replicate-count 20
    python run_warmstart.py --arm same_quantile --replicate-start 0 --replicate-count 20

Outputs: metrics_<arm>.csv (best-so-far of every run and iteration), detail_<arm>.jsonl (every selected condition)
and start_points_<arm>.csv.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from common import (ACQ_CANDIDATES_PER_ITER, BATCH_SIZE, DIVERSITY_LAMBDA, FEATURE_RANGES, FEATURES,
                     INITIAL_EXPERIMENTS, METHOD_SPECS, N_ITERATIONS, RANDOM_STATE,
                     ConstantSurrogate, GateLookup, acquisition_scores, build_oracle_lookup,
                     chebyshev_dmin_to_set, greedy_diverse_batch, make_surrogate,
                     sample_untried_candidates, scale_row, y_to_model_scale)

HERE = Path(__file__).resolve().parent
PERCENTILE_BAND_PATH = HERE / "percentile_band_coords.npy"

FAITHFUL_START = [
    {"metal_amount": 18, "modulator": 7, "add_solvent": 17, "reaction_time": 10, "reaction_temperature": 21},
    {"metal_amount": 23, "modulator": 6, "add_solvent": 15, "reaction_time": 10, "reaction_temperature": 20},
    {"metal_amount": 20, "modulator": 7, "add_solvent": 19, "reaction_time": 12, "reaction_temperature": 29},
]

STARTPOINT_SEED_BASE = 900_000_000  # deliberately far from RANDOM_STATE+10_000*replicate_idx's range


def get_arm2_start(replicate_idx: int, band_coords: np.ndarray) -> List[Dict[str, int]]:
    """Independent RNG, seeded ONLY by replicate_idx, drawn to completion
    before any strategy runs. Never touches the per-strategy algorithmic rng."""
    rng = np.random.default_rng(STARTPOINT_SEED_BASE + replicate_idx)
    idx = rng.choice(len(band_coords), size=3, replace=False)
    rows = band_coords[idx]
    return [dict(zip(FEATURES, (int(v) for v in row))) for row in rows]


def pairwise_chebyshev(points: List[Dict[str, int]]) -> Dict[str, float]:
    arr = np.array([[p[f] for f in FEATURES] for p in points], dtype=np.int64)
    out = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            out[f"d({i},{j})"] = float(np.max(np.abs(arr[i] - arr[j])))
    return out


def run_one_strategy(arm: str, spec: Dict[str, Any], replicate_idx: int, initial_points: List[Dict[str, int]],
                      oracle: np.ndarray, gate: GateLookup,
                      metrics_rows: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]]) -> None:
    from common import q_of  # local import, avoids re-importing at module top for clarity of what run_one_strategy needs

    method = spec["name"]
    seed_base = (RANDOM_STATE + 10_000 * replicate_idx) % (2**31 - 1)  # NO stable_int(method) -- see module docstring
    rng = np.random.default_rng(seed_base)

    tried: set = set()
    history_conditions: List[Dict[str, int]] = []
    history_y_raw: List[float] = []
    best_q = 0.0

    def log_point(iteration: int, cond: Dict[str, int], q: float, batch_position: int | None,
                  selection_stage: str, status: str) -> None:
        coord_arr = np.array([[cond[f] for f in FEATURES]])
        d_min_fixed95, boundary_risk, suspicious = gate.lookup(coord_arr)
        # d_min_Dt: distance to the measured set BEFORE this point is added (history_conditions
        # at call time already excludes cond, since we log before add_experiment below)
        prior_set = np.array([[c[f] for f in FEATURES] for c in history_conditions], dtype=np.int64)
        d_min_Dt = chebyshev_dmin_to_set(coord_arr[0], prior_set)
        detail_rows.append({
            "arm": arm, "method": method, "replicate": replicate_idx, "iteration": iteration,
            "batch_position": batch_position, "selection_stage": selection_stage,
            **{f: cond[f] for f in FEATURES}, "q_final": q,
            "d_min_fixed95": float(d_min_fixed95[0]), "boundary_risk": float(boundary_risk[0]),
            "suspicious": bool(suspicious[0]), "d_min_Dt": d_min_Dt, "status": status,
        })

    def add_experiment(cond: Dict[str, int]) -> float:
        nonlocal best_q
        t = tuple(cond[c] for c in FEATURES)
        tried.add(t)
        q = q_of(oracle, cond)
        history_conditions.append(cond)
        history_y_raw.append(q)
        best_q = max(best_q, q)
        return q

    for cond in initial_points:
        q_placeholder = q_of(oracle, cond)
        log_point(0, cond, q_placeholder, None, "initial_fixed", "initial_fixed")
        add_experiment(cond)
    metrics_rows.append({"arm": arm, "method": method, "replicate": replicate_idx, "iteration": 0,
                          "n_experiments": len(history_conditions), "best_q": best_q, "status": "initial_fixed"})

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
            print(f"[ERROR] {arm}/{method} replicate={replicate_idx} iter={iteration}: {traceback.format_exc(limit=2)}")

        for bp, c in enumerate(batch):
            q = q_of(oracle, c)
            log_point(iteration, c, q, bp, "model" if spec["surrogate"] != "random" else "random", status)
            add_experiment(c)

        metrics_rows.append({"arm": arm, "method": method, "replicate": replicate_idx, "iteration": iteration,
                              "n_experiments": len(history_conditions), "best_q": best_q, "status": status})

        if iteration % 10 == 0 or iteration == N_ITERATIONS:
            elapsed = time.perf_counter() - t_start
            print(f"[{arm}/{method} rep={replicate_idx}] iter {iteration:>2}/{N_ITERATIONS} "
                  f"n={len(history_conditions):>3} best_q={best_q:>10,.0f} status={status:<28} elapsed={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["faithful", "same_quantile"], required=True)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-count", type=int, default=20)
    parser.add_argument("--methods", default=None, help="comma-separated subset of method names; default all 7")
    parser.add_argument("--n-iterations", type=int, default=None, help="override N_ITERATIONS (for smoke tests)")
    parser.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = parser.parse_args()

    if args.n_iterations is not None:
        import common
        common.N_ITERATIONS = args.n_iterations
        global N_ITERATIONS
        N_ITERATIONS = args.n_iterations

    specs = METHOD_SPECS
    if args.methods:
        wanted = set(args.methods.split(","))
        specs = [s for s in METHOD_SPECS if s["name"] in wanted]
        missing = wanted - {s["name"] for s in specs}
        if missing:
            raise SystemExit(f"Unknown method(s): {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building oracle lookup...")
    oracle = build_oracle_lookup()
    print("Loading frozen gate metadata...")
    gate = GateLookup()

    band_coords = None
    if args.arm == "same_quantile":
        band_coords = np.load(PERCENTILE_BAND_PATH)

    metrics_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    start_point_rows: List[Dict[str, Any]] = []

    replicates = range(args.replicate_start, args.replicate_start + args.replicate_count)
    for replicate_idx in replicates:
        if args.arm == "faithful":
            initial_points = FAITHFUL_START
        else:
            initial_points = get_arm2_start(replicate_idx, band_coords)

        start_row = {"arm": args.arm, "replicate": replicate_idx}
        for i, p in enumerate(initial_points):
            for f in FEATURES:
                start_row[f"p{i}_{f}"] = p[f]
        if args.arm == "same_quantile":
            start_row.update(pairwise_chebyshev(initial_points))
        start_point_rows.append(start_row)

        per_strategy_initial_sets = []
        for spec in specs:
            # Pass a copy so no strategy's in-place list mutations (history_conditions
            # starts as a fresh list per strategy anyway) can affect another's view.
            run_one_strategy(args.arm, spec, replicate_idx, [dict(p) for p in initial_points],
                              oracle, gate, metrics_rows, detail_rows)
            first3 = [r for r in detail_rows if r["arm"] == args.arm and r["method"] == spec["name"]
                      and r["replicate"] == replicate_idx and r["iteration"] == 0]
            per_strategy_initial_sets.append(tuple(sorted(tuple(r[f] for f in FEATURES) for r in first3)))

        if len(set(per_strategy_initial_sets)) != 1:
            raise RuntimeError(
                f"[INVALID REPLICATE] arm={args.arm} replicate={replicate_idx}: strategies do not share an "
                f"identical initial 3-point set -- {dict(zip([s['name'] for s in specs], per_strategy_initial_sets))}"
            )
        print(f"[OK] arm={args.arm} replicate={replicate_idx}: all {len(specs)} strategies share one initial set.")

    metrics_path = out_dir / f"metrics_{args.arm}_rep{args.replicate_start}-{args.replicate_start+args.replicate_count-1}.csv"
    detail_path = out_dir / f"detail_{args.arm}_rep{args.replicate_start}-{args.replicate_start+args.replicate_count-1}.jsonl"
    starts_path = out_dir / f"start_points_{args.arm}_rep{args.replicate_start}-{args.replicate_start+args.replicate_count-1}.csv"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    with open(detail_path, "w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, default=float) + "\n")
    pd.DataFrame(start_point_rows).to_csv(starts_path, index=False)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {starts_path}")


if __name__ == "__main__":
    main()
