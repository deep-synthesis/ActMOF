# Data dictionary

## The five synthesis variables

Every condition is a vector of five integer codes. One step of a code is one increment of the laboratory quantity.

| Column | Laboratory quantity | Code | One step | Physical range |
|---|---|---|---|---|
| `metal_amount` | AlCl3.6H2O stock solution, 100 mg/mL in DI water | 5-75 | 0.1 mL | 0.5-7.5 mL |
| `modulator` | NaOH stock solution, 100 mg/mL in DI water | 5-15 | 0.1 mL | 0.5-1.5 mL |
| `add_solvent` | additional DI water | 0-30 | 0.1 mL | 0-3.0 mL |
| `reaction_time` | microwave hold time | 1-12 | 5 min | 5-60 min |
| `reaction_temperature` | microwave temperature | 10-30 | 5 degC | 50-150 degC |

The MOF-321 syntheses used 1 mmol of linker. The six validation campaigns used the same codes at one tenth of the
scale: 0.1 mmol of linker and 10 uL per step of the first three codes.

## `data/anchors_mof321.csv`

The 96 recorded MOF-321 syntheses (the "anchor experiments" of the paper), in recording order.

| Column | Meaning |
|---|---|
| `entry` | number of the entry (1-96) |
| `metal_amount` ... `reaction_temperature` | the five codes |
| `AlCl3_6H2O_stock_mL`, `NaOH_stock_mL`, `added_water_mL`, `hold_time_min`, `temperature_C` | the same condition in laboratory units |
| `intensity` | intensity at the maximum of the diagnostic PXRD reflection (0 if no reflection) |
| `FWHM_deg` | full width at half maximum of that reflection in degrees 2-theta (30 is a placeholder for "no reflection") |
| `q` | crystallinity index, intensity / FWHM (0 if no product) |
| `repeat_of_entry` | entry 31 repeats entry 26; the landscape uses the 95 unique conditions |

## `data/design_grid_part*.csv` (the landscape)

One row for each of the 71 x 11 x 31 x 12 x 21 = 6,101,172 conditions; written by
`dataset_construction/build_design_grid.py`.

| Column | Meaning |
|---|---|
| `metal_amount` ... `reaction_temperature` | the five codes |
| `label_id` | `Q...` for the 95 measured conditions, `Y...` for all others |
| `Intensity_exp`, `FWHM_exp`, `q_exp` | measured values (measured conditions only) |
| `Intensity_final`, `FWHM_final` | values of the landscape: measured for the 95 anchors, emulated for the others |
| `q_final` | crystallinity index of the landscape, `round(Intensity_final / FWHM_final)`; 0 means no product |
| `rule_applied` | 0, measured condition or forest prediction alone; 1, Rule 1 (an anchor within one step in every variable: 80% anchor mean + 20% forest); 2, Rule 2 (an anchor two steps away in one or two variables: 50/50) |

A condition is set to no product (`q_final = 0`) when `Intensity_final < 3000` or `FWHM_final > 3`; `FWHM_final = 30`
then marks "no reflection". To select the measured conditions: `df["label_id"].astype(str).str.startswith("Q")`.

Thresholds of the quality tiers (quantiles of `q_final` over the 6,101,077 emulated conditions): top 5%, 4,167;
top 1%, 42,463; top 0.1%, 100,288; top 0.01%, 144,556. 6.3% of the conditions have `q_final > 0`.

## Benchmark protocol and outputs

- 3 starting experiments drawn at random, then 39 iterations of 3 experiments: 120 experiments per run. Iteration `t`
  ends after `3 + 3t` experiments (`t = 3`: 12 experiments, the budget of a wet-lab campaign).
- The models are fitted to `log(1 + q)`; the objective is the highest `q_final` found.
- 5 runs per strategy (20 for six leading strategies and random search). The seed of run `r` of method `m` is
  `42 + 10000 r + stable_int(m)`, so different methods start from different conditions.
- `all_methods_PERRUN_curves_UNIFIED.csv`: columns `method`, `family`, `run`, `iteration`, `best_q`, where `best_q`
  is the highest `q_final` among all experiments of the run up to that iteration.

## `docs/method_registry.csv`

| Column | Meaning |
|---|---|
| `method` | name in the code and in the result files |
| `name_in_paper`, `class_in_paper` | name and class of the strategy in the paper and its Supporting Information |
| `family` | grouping used in the result files |
| `produced_by_script` | script in `benchmark/families/` that runs the strategy |
