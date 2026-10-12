"""
Noisy-oracle benchmark for Section S4.5 (previewed in S3.3).

GP Matern-5/2 with PI or EI, n = 20 paired runs (both acquisitions start from the same batch and candidate
stream), 39 iterations (3 + 39 x 3 = 120 experiments). Each query returns either the landscape q (noise-free) or
q x log-normal noise with CV = 30 % (one-loading scatter measured in Section S2.1.2). The best-so-far is always
scored on the true landscape q. Results are appended run by run to _cache/noise_benchmark.csv.
"""
import os, sys, time
from multiprocessing import Pool
import pandas as pd
import common as s
from replay import replay

N_IT, N_RUNS, CV = 39, 20, 0.30
OUTF = os.path.join(str(s.HERE), "noise_benchmark.csv")
ARMS = [(m, cv) for m in ["GP_Matern52_PI", "GP_Matern52_EI_BASELINE"] for cv in (0.0, CV)]


def job(args):
    m, cv, r = args
    o = s.oracle()
    t = time.time()
    rec, best, _ = replay(m, r, N_IT, o, noise_cv=cv, paired=True)
    return [dict(method=m, noise_cv=cv, run=r, iteration=i, n_experiments=3 + 3 * i, best_true_q=b)
            for i, b in enumerate(best)], rec, time.time() - t


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUTF):
        old = pd.read_csv(OUTF)
        done = set(zip(old.method, old.noise_cv, old.run))
    todo = [(m, cv, r) for r in range(N_RUNS) for m, cv in ARMS if (m, cv, r) not in done]
    print(f"{len(todo)} runs to do", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for rows, rec, dt in pool.imap_unordered(job, todo):
            pd.DataFrame(rows).to_csv(OUTF, mode="a", header=not os.path.exists(OUTF), index=False)
            rf = os.path.join(str(s.HERE), "noise_benchmark_selections.csv")
            pd.DataFrame(rec).assign(noise_cv=rows[0]["noise_cv"]).to_csv(rf, mode="a", header=not os.path.exists(rf), index=False)
            print(rows[0]["method"], rows[0]["noise_cv"], rows[0]["run"], f"{dt:.0f}s best={rows[-1]['best_true_q']:.0f}", flush=True)
