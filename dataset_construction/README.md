# dataset_construction

**Nothing here need be run to use this repository.** The virtual sandbox this script produces is already committed at `../data/design_grid_part*.csv` (~233MB, see the top-level README's "Where the data comes from" section). This folder exists so the sandbox remains independently reproducible and auditable, not because it constitutes a required setup step.

One script, `build_design_grid.py`, turns the ninety-five real MOF-321 microwave synthesis experiments, hardcoded near the top of the file as `X`, `max_intensity`, and `FWHM`, into the full ActMOF virtual sandbox, `design_grid_part001.csv` through `design_grid_part007.csv`, roughly 6.1 million rows, one row per integer combination of the five synthesis variables.

## How it works

The script proceeds through four stages. It fits (i) **two independent `RandomForestRegressor` models** on the real experiments, `rf_int` predicting `Intensity_exp` and `rf_fwhm` predicting `FWHM_exp`, with five-fold cross-validation run first and its R2 and RMSE printed purely as a diagnostic before the final models are refit on all real data; (ii) it then walks every point in the full five-variable grid, reusing the real measured values verbatim for genuine experimental conditions and, for every other point, obtaining an RF prediction and blending it with nearby real experiments, whereby a Chebyshev distance no greater than one from a real point yields an eighty-percent real-neighbor average combined with twenty-percent RF, and a distance no greater than two yields an even blend, the exact rule being given in the module docstring and in `apply_rules_batch()`; (iii) it applies the failure-value post-processing in `postprocess_and_round_vec()` and computes `q_final` as `Intensity_final` divided by `FWHM_final` for every row; and (iv) it streams the result to `design_grid_part*.csv` in roughly one-million-row chunks.

See `../docs/data_dictionary.md` for the exact column schema and rules.

## Provenance

Copied verbatim from the first code cell of this project's original `construct dataset with time tracker.ipynb`. The only change made here concerns where the output CSVs are written (`OUTPUT_DIR`, described below); no RF hyperparameter, blending rule, or post-processing rule was touched.

## Running it (optional, to rebuild or verify instead of trusting the shipped copy)

```bash
pip install -r ../requirements.txt
python build_design_grid.py
```

Output goes to `../data/` by default, so the benchmark scripts in `../benchmark/families/` find it with no extra configuration. Override with the `ACTMOF_DATA_DIR` environment variable should a different location be preferred.

**Expect this to take a while.** The script predicts and rule-blends over roughly 6.1 million rows in batches of `BATCH_ROWS`, default 50,000, requiring anywhere from tens of minutes to a few hours depending on the CPU involved. There is no meaningful way to shorten this without changing `FULL_GRID` or `BOUNDS`, which would alter the dataset itself rather than merely the time taken to build it.

Only `numpy`, `pandas`, and `scikit-learn` are required; neither R nor an API key is needed.
