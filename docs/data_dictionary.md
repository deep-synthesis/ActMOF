# Data dictionary

## The five synthesis variables (feature space)

| Column | Meaning | Range | Unit |
|---|---|---|---|
| `metal_amount` | metal precursor amount | 5-75 | eq |
| `modulator` | modulator amount | 5-15 | eq |
| `add_solvent` | extra solvent added | 0-30 | mL |
| `reaction_time` | microwave heating time | 1-12 | h |
| `reaction_temperature` | reaction temperature | 10-30 | degC |

These are exactly the `FEATURES` / `FEATURE_RANGES` used throughout every
script in `dataset_construction/` and `benchmark/`.

## design_grid_part*.csv (the virtual sandbox)

Produced by `dataset_construction/build_design_grid.py`. Columns:

```
metal_amount, modulator, add_solvent, reaction_time, reaction_temperature,
label_id, Intensity_exp, FWHM_exp, q_exp,
Intensity_final, FWHM_final, q_final, rule_applied
```

- Full grid size: 6,101,172 rows (every integer combination of the five
  variables within their ranges above).
- `label_id`: rows starting with `Q`/`q` are the 95 real experimental
  conditions (ground truth, confirmed by direct count across the shipped
  files); everything else (`Y0000001`, `Y0000002`, ...) is RF-predicted or
  rule-blended.
- To keep only real experimental rows: `df['label_id'].astype(str).str.startswith(('Q','q'))`.
  To exclude them (the usual "simulation pool" view): negate that mask.
- `q_final = round(Intensity_final / FWHM_final)`, with the fallback rule
  `Intensity_final < 3000 -> q_final = 0` and `FWHM_final = 30` (the sentinel
  "failed synthesis" value whenever `FWHM_final > 3` and the low-intensity
  rule didn't already fire). See `build_design_grid.py`'s
  `postprocess_and_round_vec()` for the exact vectorized logic.
- `rule_applied`: 0 = experimental row or plain RF prediction with no nearby
  real neighbor; 1 = Rule 1 (Chebyshev-distance-1 neighbor of a real
  experiment -> 80% neighbor average + 20% RF); 2 = Rule 2 (distance-2
  neighbor -> 50/50 blend).

## Active-learning benchmark protocol (all `benchmark/families/*.py`)

- Batch size: 3 proposed experiments per iteration.
- Total budget: 150 experiments (3 initial + 49 active-learning iterations
  x 3 per iteration).
- Objective: maximize `q_final` (the crystallinity index -- higher is better).
- 5 independent repeats (`N_RUNS = 5`) per method, same seed formula
  (`RANDOM_STATE + 10_000*run_idx + stable_int(method_name)`, then
  `+ iteration` per call) across every family script.
- `all_methods_PERRUN_curves_UNIFIED.csv` (built by
  `benchmark/merge_outputs.py`) columns: `method, family, run, iteration, best_q`
  -- `best_q` is the best `q_final` found so far, cumulative up to that
  iteration, for that run.
