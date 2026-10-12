"""run_all_methods.py -- single entry point for the 39 strategies of the ActMOF benchmark.

Runs each of the 7 scripts in families/ as a separate process, then merge_outputs.py. It contains no method logic.
The strategies are kept in separate scripts because they need different software (R, OpenAI API) and consume random
numbers in their own way; sharing code between them could change their results.

    python run_all_methods.py                          # everything
    python run_all_methods.py --only core29_gp_ml_tree_rl_mof,rvm_like
    python run_all_methods.py --skip lagp,rgasp_bnn
    python run_all_methods.py --list | --dry-run

Requirements per family (details in README.md):
    core29_gp_ml_tree_rl_mof   29 strategies, Python only
    rvm_like                    1 strategy,   Python only
    lagp                        2 strategies, R + laGP
    rgasp_bnn                   2 strategies, R + RobustGaSP
    llm_pure_gpt4o_gpt55        2 strategies, OPENAI_API_KEY (billed)
    llm_pure_gpt5mini           1 strategy,   OPENAI_API_KEY (billed)
    hybrid_gp_llm               2 strategies, OPENAI_API_KEY (billed)
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
