"""
run_all_methods.py -- single entry point for all 39 ActMOF benchmark methods.

This is NEW orchestration code written for this repo. It does not contain or
reimplement any method logic itself -- it only runs each of the 7 family
scripts under families/ (1 covering 29 methods, the other 6 covering 10
methods between them) as a separate process (exactly as if you had typed
`python families/<script>.py` yourself) and then calls merge_outputs.py.
Each family script is a verbatim (config-paths-only-changed) copy of one of
this project's own validated, seed-controlled rerun scripts -- see docs/ for
what "validated" means here and where each script's data has been checked
against the previously-published results.

Why 6 separate scripts instead of one shared codebase: each family script was
independently validated (including, for the 5 LLM-touching methods, a live
"canary" API call that hard-fails the run if the server silently drops the
seed) against real re-runs. Merging their internals into shared helper
functions was judged higher-risk than keeping them separate, since even
small refactors could silently change the RNG consumption pattern (see
core29_gp_ml_tree_rl_mof.py's own docstring for a worked example of exactly
that happening between two earlier drafts of this codebase). This script
only adds an orchestration layer on top; nothing below it changes any
method's math.

Cost / requirements per family (see benchmark/README.md for detail):
    core29_gp_ml_tree_rl_mof   29 methods, pure Python, no R, $0
    lagp                        2 methods, needs R + laGP,  $0
    rgasp_bnn                   2 methods, needs R + RobustGaSP, $0
    rvm_like                    1 method,  pure Python, $0
    llm_pure_gpt4o_gpt55        2 methods, needs OPENAI_API_KEY, real $ cost
    llm_pure_gpt5mini           1 method,  needs OPENAI_API_KEY, real $ cost (cheap)
    hybrid_gp_llm               2 methods, needs OPENAI_API_KEY, real $ cost

Usage:
    python run_all_methods.py                       # run everything, then merge
    python run_all_methods.py --only core29_gp_ml_tree_rl_mof,rvm_like
    python run_all_methods.py --skip lagp,rgasp_bnn  # e.g. no R available
    python run_all_methods.py --list                # show families and exit
    python run_all_methods.py --merge-only           # just (re)build the merged CSV
    python run_all_methods.py --dry-run              # print commands, run nothing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (key, script path relative to this file, human note)
FAMILIES = [
    ("core29_gp_ml_tree_rl_mof", "families/core29_gp_ml_tree_rl_mof.py",
     "29 methods (GP/ML/tree/RL/MOF-specific) -- pure Python, no R, $0"),
    ("lagp", "families/lagp.py",
     "2 methods (laGP ALC/NN) -- needs R 4.6 + laGP installed, $0"),
    ("rgasp_bnn", "families/rgasp_bnn.py",
     "2 methods (RobustGaSP, BNN-like) -- needs R 4.6 + RobustGaSP installed, $0"),
    ("rvm_like", "families/rvm_like.py",
     "1 method (RVM-like ARD+RFF) -- pure Python, $0"),
    ("llm_pure_gpt4o_gpt55", "families/llm_pure_gpt4o_gpt55.py",
     "2 methods (LLM-only gpt-4o, gpt-5.5) -- needs OPENAI_API_KEY, real $ cost"),
    ("llm_pure_gpt5mini", "families/llm_pure_gpt5mini.py",
     "1 method (LLM-only gpt-5-mini) -- needs OPENAI_API_KEY, real $ cost (cheap)"),
    ("hybrid_gp_llm", "families/hybrid_gp_llm.py",
     "2 methods (GP-LLM hybrid, EI/PI x gpt-5.5) -- needs OPENAI_API_KEY, real $ cost"),
]
FAMILY_KEYS = [key for key, _, _ in FAMILIES]


def parse_keys(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    unknown = keys - set(FAMILY_KEYS)
    if unknown:
        raise SystemExit(f"Unknown family key(s): {sorted(unknown)}. Valid keys: {FAMILY_KEYS}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="Comma-separated family keys to run (default: all).")
    parser.add_argument("--skip", default=None, help="Comma-separated family keys to skip.")
    parser.add_argument("--list", action="store_true", help="Print family keys/notes and exit.")
    parser.add_argument("--merge-only", action="store_true", help="Skip running families; just merge existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run, without running them.")
    args = parser.parse_args()

    if args.list:
        for key, script, note in FAMILIES:
            print(f"{key:28s} {script:40s} {note}")
        return

    only = parse_keys(args.only)
    skip = parse_keys(args.skip) or set()

    to_run = [
        (key, script) for key, script, _ in FAMILIES
        if (only is None or key in only) and key not in skip
    ]

    if not args.merge_only:
        for key, script in to_run:
            cmd = [sys.executable, str(HERE / script)]
            print(f"\n=== [{key}] {' '.join(cmd)} ===")
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True, cwd=str(HERE))

    print("\n=== merge_outputs.py ===")
    merge_cmd = [sys.executable, str(HERE / "merge_outputs.py")]
    if args.dry_run:
        print(" ".join(merge_cmd))
        return
    subprocess.run(merge_cmd, check=True, cwd=str(HERE))


if __name__ == "__main__":
    main()
