"""
build_figuredata_curve_best_q.py -- collapse the 5 independent runs into one
summary curve per method: reads the per-run raw data
(benchmark/reference_output/all_methods_PERRUN_curves_UNIFIED.csv, or a
fresh run's benchmark/outputs/all_methods_PERRUN_curves_UNIFIED.csv) and
writes figuredata_curve_best_q_UNIFIED.csv, the file
fig3.py and curve_best_q_UNIFIED.py (internally) actually
plot.

This is NEW code written for this repo -- it does not recompute or alter
any benchmark result. It only aggregates:

    input  (one row per method, run, iteration):
        method, family, run, iteration, best_q
    output (one row per method, iteration -- the `run` dimension collapsed):
        method, family, x, mean, std, standard_error

For a fixed (method, iteration), `mean`/`std` are the mean/sample std
(ddof=1, pandas default) of `best_q` across the 5 independent runs;
`standard_error = std / sqrt(n_runs)`. This is exactly what a "best q found
so far" curve with one line per method needs -- the raw per-run file has 5
different, noisy trajectories per method; this is the one summary trajectory
that gets drawn.

Verified before being written: the groupby below was checked against the
already-published figuredata_curve_best_q_UNIFIED.csv this repo ships in
this same folder -- for all 1,950 (method, iteration) rows, `mean` matched
exactly and `std`/`standard_error` matched to within ~1e-11 (floating-point
noise, not a real difference). Re-run this script with --check to repeat
that verification against whatever reference file you point it at.

Usage:
    python build_figuredata_curve_best_q.py
    # or, to point at a fresh run's output / a different reference file:
    python build_figuredata_curve_best_q.py --in ../benchmark/outputs/all_methods_PERRUN_curves_UNIFIED.csv
    # to verify against the shipped figuredata_curve_best_q_UNIFIED.csv instead of just overwriting it:
    python build_figuredata_curve_best_q.py --check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parent / "benchmark" / "reference_output" / "all_methods_PERRUN_curves_UNIFIED.csv"
DEFAULT_OUT = HERE / "figuredata_curve_best_q_UNIFIED.csv"

REQUIRED_IN_COLS = ["method", "family", "run", "iteration", "best_q"]
OUT_COLS = ["method", "family", "x", "mean", "std", "standard_error"]


def build(perrun: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_IN_COLS) - set(perrun.columns)
    if missing:
        raise ValueError(f"input is missing column(s): {sorted(missing)}")

    agg = (
        perrun.groupby(["method", "family", "iteration"])["best_q"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["standard_error"] = agg["std"].fillna(0) / np.sqrt(agg["count"].clip(lower=1))
    agg = agg.rename(columns={"iteration": "x"})
    return agg[OUT_COLS].sort_values(["method", "x"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN),
                         help="Per-run PERRUN CSV to aggregate (default: the shipped reference copy).")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT),
                         help="Where to write the aggregated figuredata CSV.")
    parser.add_argument("--check", action="store_true",
                         help="Instead of overwriting --out, compare the freshly computed result "
                              "against whatever is currently at --out and report the max difference.")
    args = parser.parse_args()

    perrun = pd.read_csv(args.in_path)
    result = build(perrun)

    n_methods = result["method"].nunique()
    print(f"Aggregated {len(perrun):,} per-run rows ({perrun['method'].nunique()} methods) "
          f"-> {len(result):,} rows ({n_methods} methods x "
          f"{result.groupby('method').size().iloc[0] if n_methods else 0} iterations each, if uniform).")

    out_path = Path(args.out_path)

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"--check requested but {out_path} does not exist.")
        existing = pd.read_csv(out_path).sort_values(["method", "x"]).reset_index(drop=True)
        merged = existing.merge(result, on=["method", "x"], suffixes=("_existing", "_fresh"))
        if len(merged) != len(existing) or len(merged) != len(result):
            print(f"[MISMATCH] row count differs: existing={len(existing)} fresh={len(result)} matched={len(merged)}")
        for col in ["mean", "std", "standard_error"]:
            diff = (merged[f"{col}_existing"] - merged[f"{col}_fresh"]).abs()
            print(f"  {col}: max abs diff = {diff.max():.6g}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
