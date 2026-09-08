# benchmark

This directory runs all thirty-nine active-learning and Bayesian-optimization methods compared in the ActMOF benchmark, producing the single CSV from which every downstream plot and table is built. See `../docs/data_dictionary.md` for the protocol, comprising a 150-experiment budget, batch size three, and five repeats, together with the column schema.

## Layout

```
benchmark/
├── run_all_methods.py     <- single entry point (runs the 7 scripts below, then merges)
├── merge_outputs.py       <- stitches the 7 scripts' outputs into one CSV
├── check_llm_model_availability.py  <- optional pre-flight sanity check for the OpenAI models
├── reference_output/      <- the already-published result of the above, shipped for reference
│   └── all_methods_PERRUN_curves_UNIFIED.csv   (9,750 rows -- see its own README)
└── families/
    ├── core29_gp_ml_tree_rl_mof.py   29 methods: GP / ML / tree ensembles / RL / MOF-specific
    ├── lagp.py                        2 methods: laGP (ALC, NN)              -- needs R
    ├── rgasp_bnn.py                   2 methods: RobustGaSP, BNN-like MLP    -- needs R
    ├── rvm_like.py                    1 method:  RVM-like (ARD + RFF)
    ├── llm_pure_gpt4o_gpt55.py        2 methods: LLM-only, gpt-4o / gpt-5.5  -- needs OpenAI key, costs $
    ├── llm_pure_gpt5mini.py           1 method:  LLM-only, gpt-5-mini        -- needs OpenAI key, costs $ (cheap)
    └── hybrid_gp_llm.py               2 methods: GP<->LLM hybrid, EI/PI x gpt-5.5 -- needs OpenAI key, costs $
```

Twenty-nine methods from `core29`, together with two, two, one, two, one, and two from the remaining scripts, sum to **thirty-nine methods**. `../docs/method_registry.csv` lists every method by name, family, and the script by which it is produced.

## Why 7 separate scripts instead of one

Each of the six non-`core29` scripts is a **verbatim copy**, with only `POOL_DIR`, `OUTPUT_DIR`, and `R_HOME` repointed at this repository's layout, diffed against the originals line by line with zero other changes, of a script individually validated in this project against a real rerun. For the five LLM-touching methods, this validation included a live canary API call that hard-fails the run should the server silently drop the `seed` or `reasoning_effort` parameters. Nothing about any method's mathematics, acquisition function, prompt, or model parameter was altered in packaging it here. See `../docs/` for the history of what was found and fixed, comprising an LLM seed and temperature bug and a laGP argument-name bug, both genuine, both remedied, and both re-validated with real reruns before this repository existed.

## How seed, temperature, and reasoning_effort are actually handled

**Reproducibility rests on several concrete mechanisms.** The seed is sent on every call, computed deterministically as `RANDOM_STATE + 10_000*run_idx + stable_int(method_name) [+ iteration]`, the same formula recurring in every family script. Temperature is sent only where the API accepts it, namely `0.2` for `gpt-4o`, and is withheld entirely for `gpt-5.5` and `gpt-5-mini`, confirmed via real API probing to be rejected with a 400 error for these reasoning-effort models, such that they fall back to the model's own default, which is the intended behavior rather than an oversight. The `reasoning_effort` parameter is set explicitly to each model's own minimum value, `"none"` for gpt-5.5 and `"minimal"` for gpt-5-mini, so the model cannot silently spend its entire output budget on hidden reasoning tokens. Every LLM-touching script runs a canary check, a single real API call, before the main loop, raising immediately should the server silently drop `seed` or `reasoning_effort`, such that a broken reproducibility guarantee fails loudly rather than silently. The non-LLM half of every method, comprising GP fits, RF fits, and the like, uses `random_state=seed` or `np.random.default_rng(seed)` throughout, an instance of ordinary scikit-learn and numpy determinism.

### Resolved model snapshots (observed 2026-09-08)

The three model names used in code, `gpt-4o`, `gpt-5.5`, and `gpt-5-mini`, are **generic aliases rather than pinned dated snapshots**, and OpenAI can silently move an alias to a newer underlying model at any time, constituting a genuine reproducibility gap (see `../docs/`). As of the date above, they resolved as follows.

| Alias in code | Resolved snapshot | How confirmed |
|---|---|---|
| `gpt-4o` | `gpt-4o-2024-08-06` | **Confirmed**: real `chat.completions.create` call, `response.model` read back directly. |
| `gpt-5.5` | `gpt-5.5-2026-04-23` | **Confirmed**: real `chat.completions.create` call (`reasoning_effort="none"`, matching `hybrid_gp_llm.py`'s actual call shape), `response.model` read back directly. |
| `gpt-5-mini` | `gpt-5-mini-2025-08-07` | **Confirmed**: real `chat.completions.create` call (`reasoning_effort="minimal"`, matching `llm_pure_gpt5mini.py`'s actual call shape), `response.model` read back directly. |

The scripts continue to call the bare alias, in `llm_pure_gpt4o_gpt55.py`, `llm_pure_gpt5mini.py`, and `hybrid_gp_llm.py`, rather than these dated snapshots; the table above records what was observed, not a pin. Should this benchmark be rerun well after the date above, it is advisable to recheck with `client.models.list()`, which is free, or a real probe call costing a few cents, such as `check_llm_model_availability.py`, before assuming the same snapshot remains in use.

## Setup

```bash
pip install -r ../requirements.txt
```

The R-backed methods, `lagp.py` and `rgasp_bnn.py`, additionally require (i) R, this codebase having been validated against R 4.6.1, with `ACTMOF_R_HOME` to be set if it is installed elsewhere than `C:\Program Files\R\R-4.6.1`; (ii) the CRAN packages `laGP`, for `lagp.py`, and `RobustGaSP`, for `rgasp_bnn.py`; and (iii) `pip install rpy2`.

Should R, laGP, or RobustGaSP be unavailable, these two scripts may be skipped via `--skip lagp,rgasp_bnn`, yielding thirty-five of the thirty-nine methods and omitting `LAGP_ALC_EI`, `LAGP_NN_EI`, `RGASP_Matern52_EI_nugget`, and `BNN_like_MLPensemble_EI`. The last of these belongs to the "Machine Learning" family rather than Approximate GPs or RGASP, yet its data likewise derives from `rgasp_bnn.py` (see `../docs/method_registry.csv`); the resulting subset suffices to reproduce every figure not specifically concerned with these four methods.

The LLM-touching methods, in `llm_pure_gpt4o_gpt55.py`, `llm_pure_gpt5mini.py`, and `hybrid_gp_llm.py`, require an OpenAI API key with access to `gpt-4o`, `gpt-5.5`, and `gpt-5-mini`.

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

These five methods incur real, billed API calls, amounting to roughly one to ten dollars in total for a full five-run reproduction depending on the models involved, with cost estimates given in each script's own header comment. Before committing to a full run, it is advisable to confirm that the models remain callable and accept `seed` and `temperature` in the manner this code expects.

```bash
python check_llm_model_availability.py
```

The `design_grid_part*.csv` pool is already committed under `../data/`, so nothing need be built beforehand. Should independent verification be preferred over trusting the shipped copy, see `../dataset_construction/`.

**Before a full run, billed for the LLM families,** most family scripts carry a `SMOKE_TEST = False` flag near the top of their CONFIG section, present in `core29_gp_ml_tree_rl_mof.py`, `llm_pure_gpt4o_gpt55.py`, `llm_pure_gpt5mini.py`, `hybrid_gp_llm.py`, and `rgasp_bnn.py`. Setting it to `True` runs a small fraction of the full protocol, a handful of iterations, runs, or methods, allowing the environment, API key, and R setup to be confirmed before the full time or expense is committed.

## Running

```bash
# everything (needs R + an OpenAI key set up as above)
python run_all_methods.py

# just the free, pure-Python 30 methods (skip R and LLM/cost families)
python run_all_methods.py --only core29_gp_ml_tree_rl_mof,rvm_like

# skip the R-backed families only
python run_all_methods.py --skip lagp,rgasp_bnn

# see what would run without running it
python run_all_methods.py --list
python run_all_methods.py --dry-run
```

Each family script writes its own output under `../outputs/<family>/`, comprising an iteration-metrics CSV, a per-call history JSONL, and a run log, and each is safe to rerun independently. `run_all_methods.py` runs the selected families in sequence and then calls `merge_outputs.py`, which is likewise runnable directly at any point, for instance after a partial run or after rerunning a single family, to build or rebuild `../outputs/all_methods_PERRUN_curves_UNIFIED.csv`.

This merge logic was verified, prior to being written, against the dataset this project had already published, with every one of the ten non-`core29` methods' per-run, per-iteration `best_q` values matching exactly, the maximum absolute difference being zero, by simple concatenation; no metric is recomputed by `merge_outputs.py`, which only relabels each row with a `family` column and stacks the result. That published dataset is shipped in `reference_output/`, against which a fresh run's `../outputs/all_methods_PERRUN_curves_UNIFIED.csv` may be diffed at any time to check one's own run against this one.
