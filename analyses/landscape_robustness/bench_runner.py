"""
bench_runner.py -- run the leading group + random search on a rebuilt landscape.

Uses run_single_method() from ../family_scripts/core29_gp_ml_tree_rl_mof.py
unchanged (same surrogates, acquisition, candidate sampling, seed formula
s = 42 + 10000 r + h(m)); only the oracle passed to it is swapped, and the
loop stops at iteration 39 (120 experiments), the horizon reported in the SI.
Because the seed formula is unchanged, run r of method m starts from the same
three conditions on every landscape (paired design across variants).

Worker usage:  python bench_runner.py <jobs.txt> <worker_id> <n_workers>
each line of jobs.txt: variant_id method run_idx
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[k] = "1"

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("core29", HERE.parents[1] / "benchmark" / "families" / "core29_gp_ml_tree_rl_mof.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
core.N_ITERATIONS = 39

METHODS = ["GP_Matern52_PI", "GP_Matern52_EI_BASELINE", "GP_Matern32_PI", "GP_Matern32_EI",
           "CONFRF_EI", "BART_like_ExtraTrees_EI", "RANDOM_SEARCH"]
SPEC = {s["name"]: s for s in core.METHOD_SPECS}
MAP_DIR = HERE / "maps"
RES_DIR = HERE / "bench_results"


class _NullWriter:
    def write(self, s): pass
    def flush(self): pass


def load_oracle(vid: str) -> np.ndarray:
    # variants.csv is written only after build_maps.py has saved every map
    while vid != "V0" and not (HERE / "variants.csv").exists():
        time.sleep(30)
    q = np.load(MAP_DIR / f"{vid}.npy").astype(np.float32)
    return q.reshape(71, 11, 31, 12, 21)


def run_job(oracle, method, run_idx):
    rows = []
    t = time.perf_counter()
    core.run_single_method(oracle, SPEC[method], run_idx, rows, _NullWriter())
    best = [float(r["best_q"]) for r in sorted(rows, key=lambda r: r["iteration"])]
    status = [r["status"] for r in rows]
    n_err = sum(1 for s in status if str(s).startswith("method_error"))
    return {"best_q": best, "n_err": n_err, "sec": time.perf_counter() - t}


def main():
    jobs_file, wid, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jobs = [l.split() for l in Path(jobs_file).read_text().splitlines() if l.strip()]
    jobs = jobs[wid::nw]
    RES_DIR.mkdir(exist_ok=True)
    out_fp = RES_DIR / f"worker_{wid:02d}.jsonl"
    done = set()
    for fp in RES_DIR.glob("worker_*.jsonl"):
        for l in fp.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add((d["variant"], d["method"], d["run"]))
    cache_vid, oracle = None, None
    with open(out_fp, "a") as fh:
        for vid, method, run in jobs:
            run = int(run)
            if (vid, method, run) in done:
                continue
            if vid != cache_vid:
                oracle, cache_vid = load_oracle(vid), vid
            r = run_job(oracle, method, run)
            fh.write(json.dumps({"variant": vid, "method": method, "run": run, **r}) + "\n"); fh.flush()
            print(f"[w{wid}] {vid} {method} run={run} best={r['best_q'][-1]:.0f} {r['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
