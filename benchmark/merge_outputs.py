"""
merge_outputs.py -- stitch all 7 family scripts' per-run outputs (1 covering
29 methods, the other 6 covering 10 methods between them) into the one CSV
the plotting code and paper figures read: all_methods_PERRUN_curves_UNIFIED.csv
(columns: method, family, run, iteration, best_q).

This is NEW glue code written for this repo -- it does not reimplement any
method. It only concatenates each family script's already-final
*_iteration_metrics.csv (each already contains the correct cumulative best_q
per run/iteration -- no metric is recomputed here) and attaches a `family`
label. The concatenation logic and the method-family map below were verified,
row for row, against the original all_methods_PERRUN_curves_UNIFIED.csv this
project already published (see docs/ for how that dataset was produced) --
every one of the 10 non-core methods' rows matched exactly (max abs diff in
best_q: 0.0) before this script was written.

Usage:
    python merge_outputs.py
    # or, to read/write somewhere other than the repo-relative defaults:
    python merge_outputs.py --outputs-dir /path/to/outputs --out /path/to/all_methods_PERRUN_curves_UNIFIED.csv

Run this after some or all of the 7 scripts in families/ have produced their
output (via run_all_methods.py or manually). Any family whose output file is
missing is skipped with a warning, not a hard failure, so you can merge a
partial run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Method -> family, for the 10 methods whose own output files don't carry a
# `family` column (unlike core29_gp_ml_tree_rl_mof's output, which already
# does). Matches this project's published family taxonomy exactly.
FAMILY_OVERRIDES = {
    "HYBRID_GP_Matern52_EI_ALT_gpt-5.5": "LLM-GP Hybrid",
    "HYBRID_GP_Matern52_PI_ALT_gpt-5.5": "LLM-GP Hybrid",
    "LLM_PURE_gpt-4o": "LLM",
    "LLM_PURE_gpt-5.5": "LLM",
    "LLM_PURE_gpt-5-mini": "LLM",
    "LAGP_ALC_EI": "Approximate GPs",
    "LAGP_NN_EI": "Approximate GPs",
    "RGASP_Matern52_EI_nugget": "Gaussian Processes",
    "BNN_like_MLPensemble_EI": "Machine Learning",
    "RVM_like_ARD_RFF_EI": "Basis Regression",
}

# (subfolder under outputs/, filename, has_family_column, fixed_method_name)
# fixed_method_name is set only for llm_pure_gpt5mini's output file, which
# (unlike every other family script) never writes a `method` column at all
# because it only ever runs one method.
SOURCES = [
    ("core29_gp_ml_tree_rl_mof", "unified_bo_iteration_metrics.csv", True, None),
    ("llm_pure_gpt4o_gpt55", "llm_pure_iteration_metrics.csv", False, None),
    ("llm_pure_gpt5mini", "iteration_metrics_gpt5mini.csv", False, "LLM_PURE_gpt-5-mini"),
    ("hybrid_gp_llm", "hybrid_iteration_metrics.csv", False, None),
    ("lagp", "lagp_iteration_metrics.csv", False, None),
    ("rgasp_bnn", "rgasp_bnn_iteration_metrics.csv", False, None),
    ("rvm_like", "rvm_like_iteration_metrics.csv", False, None),
]

REQUIRED_COLS = ["method", "family", "run", "iteration", "best_q"]


def load_one(outputs_dir: Path, subfolder: str, filename: str, has_family: bool,
             fixed_method_name: str | None) -> pd.DataFrame | None:
    path = outputs_dir / subfolder / filename
    if not path.exists():
        print(f"[skip] {path} not found yet -- run this family's script first.")
        return None

    df = pd.read_csv(path)

    if fixed_method_name is not None:
        df["method"] = fixed_method_name

    if not has_family:
        unknown = sorted(set(df["method"].unique()) - set(FAMILY_OVERRIDES))
        if unknown:
            raise ValueError(
                f"{path}: no family mapping for method(s) {unknown}. "
                "Add them to FAMILY_OVERRIDES in this script."
            )
        df["family"] = df["method"].map(FAMILY_OVERRIDES)

    return df[REQUIRED_COLS].copy()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=str(repo_root / "outputs"),
                         help="Parent directory holding each family's own output subfolder.")
    parser.add_argument("--out", default=str(repo_root / "outputs" / "all_methods_PERRUN_curves_UNIFIED.csv"),
                         help="Where to write the merged CSV.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    frames = []
    for subfolder, filename, has_family, fixed_method_name in SOURCES:
        df = load_one(outputs_dir, subfolder, filename, has_family, fixed_method_name)
        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit(
            "No family output files found under "
            f"{outputs_dir}. Run the scripts in families/ (directly, or via "
            "run_all_methods.py) before merging."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["method", "run", "iteration"]).reset_index(drop=True)

    n_methods = merged["method"].nunique()
    print(f"Merged {len(frames)}/{len(SOURCES)} family output file(s), "
          f"{n_methods}/39 method(s), {len(merged):,} rows.")
    if n_methods < 39:
        print("[note] fewer than 39 methods present -- this is a partial merge, "
              "not an error; re-run after the remaining family script(s) finish.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
