"""
run_feasibility_paired.py -- run policies of benchmark/families/feasibility.py under the SI S4.4 paired seeding.

    python scripts/run_feasibility_paired.py --methods B0 --runs 0-19 --tag cp2 --workers 6
    python scripts/run_feasibility_paired.py --methods M1 M3 --runs 0-2 --tag cp5_pilot

Output (never touches existing result folders): results/feasibility_paired/<tag>/runs.csv, iterations.csv
Runs already present in the output are skipped (resume).
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import feasibility as F    # noqa: E402

OUT = ROOT / "results"


def job(a):
    m, r = a
    t = time.time()
    rows, irows = F.run_one(m, r, F.load_oracle())
    return m, r, rows, irows, time.time() - t


def parse_runs(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, choices=list(F.METHODS))
    ap.add_argument("--runs", default="0-19")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    d = OUT / a.tag
    d.mkdir(parents=True, exist_ok=True)
    fr, fi = d / "runs.csv", d / "iterations.csv"
    done = set()
    if fr.exists():
        old = pd.read_csv(fr, usecols=["method", "run"]).drop_duplicates()
        done = set(zip(old.method, old.run))
    todo = [(m, r) for r in parse_runs(a.runs) for m in a.methods if (m, r) not in done]
    print(f"{len(todo)} runs to do -> {d}", flush=True)
    F.load_oracle()                                   # build cache once before forking
    with Pool(a.workers) as pool:
        for m, r, rows, irows, dt in pool.imap_unordered(job, todo):
            pd.DataFrame(rows).to_csv(fr, mode="a", header=not fr.exists(), index=False)
            pd.DataFrame(irows).to_csv(fi, mode="a", header=not fi.exists(), index=False)
            print(f"{m} run {r}: {dt:.0f}s, final best q = {rows[-1]['best_so_far_q']:.0f}", flush=True)
