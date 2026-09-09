# ActMOF

ActMOF is a crystallinity-guided active-learning benchmark for metal-organic-framework (MOF) synthesis. 95 real MOF-321 microwave-synthesis experiments were used to construct a virtual sandbox of roughly 6.1 million conditions via two Random Forest models. **Upon this sandbox, 39 active-learning and Bayesian-optimization strategies, spanning Gaussian processes, tree ensembles, approximate GPs, reinforcement learning, LLM-only methods, and GP-LLM hybrid methods, are compared for their capacity to reach high-crystallinity synthesis conditions.** The objective, `q_final` (`Intensity_final / FWHM_final`), is such that higher values are preferable.

## Pipeline

```
data/                          the ~6.1M-row virtual sandbox, shipped in this repo
    design_grid_part*.csv          (~233MB total; see "Where the data comes from" below)
        |
benchmark/                     (1) run all 39 methods, 5 repeats each
    run_all_methods.py             (single entry point)
    families/*.py                   7 scripts: 1 covers 29 methods, the other 6 cover
                                     10 methods between them (1-3 methods each) -- see
                                     benchmark/README.md's table; 7 scripts, 39 methods
        |
        v  all_methods_PERRUN_curves_UNIFIED.csv   9,750 rows = 39 methods x 5 runs x 50 iterations
           (written to outputs/, via merge_outputs.py; a verified copy ships in
           benchmark/reference_output/ so you don't have to run this stage first)
        |
plotting/                      (2) turn the merged results into a figure
    build_figuredata_curve_best_q.py   collapses the 5 runs -> 1,950 rows (39 x 50), one
                                        mean/std/SE curve per method (a verified copy of
                                        its output ships here too)
    curve_best_q_UNIFIED.py            all 39 methods
    fig3.py             the published Figure 3 (16 of the 39 methods)
```

Each stage's own README provides further detail, namely [`dataset_construction/README.md`](dataset_construction/README.md), [`benchmark/README.md`](benchmark/README.md), and [`plotting/README.md`](plotting/README.md).

## Quickstart

```bash
pip install -r requirements.txt

# data/design_grid_part*.csv is already in this repo -- no build step needed.

# 1. run the benchmark (needs R for 4/39 methods, an OpenAI key for 5/39 -- see benchmark/README.md)
#    to start cheap: the 30 methods needing neither R nor an API key
python benchmark/run_all_methods.py --only core29_gp_ml_tree_rl_mof,rvm_like

# 2. plot -- curve_best_q_UNIFIED.py is self-sufficient (all 39 methods);
#    fig3.py (the published Figure 3, 16/39 methods) needs
#    build_figuredata_curve_best_q.py's output first -- both already ship a
#    verified copy of that intermediate file, so this step is optional:
python plotting/curve_best_q_UNIFIED.py
python plotting/build_figuredata_curve_best_q.py   # only if you reran the benchmark yourself
python plotting/fig3.py
```

## Where the data comes from

`data/design_grid_part*.csv` is committed directly in this repository, comprising roughly 233MB across seven files, each comfortably under GitHub's 50MB and 100MB size thresholds, obviating the need for Git LFS and allowing the repository to be cloned and used without regeneration. Should independent verification be preferred over trusting the shipped copy, `dataset_construction/build_design_grid.py` rebuilds the sandbox from scratch, carrying the ninety-five real experiments through two RandomForestRegressor models to the full grid. The process is fully deterministic, with `random_state=42` fixed throughout, such that a rebuild is expected to reproduce these files exactly, albeit subject to floating-point or version drift across substantially different scikit-learn releases.

`benchmark/README.md` details how the remaining nine methods, four R-backed and five LLM-backed, are run, along with their associated cost and the manner in which reproducibility, with respect to seed, temperature, and reasoning_effort, is handled for the five LLM-touching methods.

## What's in `docs/`

The `docs/` directory contains (i) [`data_dictionary.md`](docs/data_dictionary.md), giving the column schema for the virtual sandbox and the benchmark output, along with feature ranges and the `q_final` definition; and (ii) [`method_registry.csv`](docs/method_registry.csv), listing all thirty-nine methods, their family, and the script under `benchmark/families/` by which each is produced.

## Provenance and what "unchanged implementation" means here

**Every method's implementation, comprising surrogate model code, acquisition functions, LLM prompts and parsing, and the RF-based dataset construction, was copied verbatim from this project's own previously validated scripts, with no method gaining, losing, or having a parameter altered in the process of assembling this repository.** The only edits made anywhere concerned where files are read from and written to, specifically hardcoded absolute Windows paths converted to paths relative to this repository and overridable via environment variables, so that the code runs for anyone cloning this folder rather than solely on the original machine. Each such edit was verified line by line against the original scripts, with diffs confined exactly to the intended path and import lines, and the full merge pipeline was rerun and checked to reproduce this project's previously published results exactly, the maximum difference being zero, prior to this README being written. The one piece of genuinely new code, `plotting/build_figuredata_curve_best_q.py`, for which no saved script existed for this aggregation step before this repository was created, was checked two independent ways against the already published `figuredata_curve_best_q_UNIFIED.csv` prior to being shipped; see `plotting/README.md`.

## Requirements

See `requirements.txt` for the full list of core Python dependencies. Beyond these, R 4.6.x together with the `laGP` and `RobustGaSP` CRAN packages is required for four of the thirty-nine methods, and an OpenAI API key is required for five of the thirty-nine, both being optional, given that thirty of the thirty-nine methods run with `pip install -r requirements.txt` alone. The key should be set via the `OPENAI_API_KEY` environment variable; no key is stored anywhere in this repository.

## License

This repository is released under the MIT license; see [`LICENSE`](LICENSE).
