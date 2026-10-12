# dataset_construction

`build_design_grid.py` builds the ActMOF landscape, `design_grid_part001.csv` to `design_grid_part007.csv`, from the
95 measured MOF-321 syntheses. The landscape is already shipped in `../data/`, so this script only needs to be run to
rebuild or check it.

## What the script does

1. Two random forests (400 trees, `random_state = 42`) are fitted to the 95 measured conditions, one for the intensity
   and one for the FWHM of the diagnostic PXRD reflection. Five-fold cross-validation is printed first.
2. For each of the 6,101,172 conditions of the grid: a measured condition keeps its measured values; any other
   condition gets the forest prediction, blended with nearby measured conditions (Rule 1: a measured condition within
   one step in every variable -> 80% mean of these + 20% forest; Rule 2: one two steps away in one or two variables
   -> 50/50).
3. Conditions with intensity below 3000 or FWHM above 3 are set to no product (`q_final = 0`); otherwise
   `q_final = round(intensity / FWHM)`.

The columns are described in `../docs/data_dictionary.md`.

## Running

```bash
pip install -r ../requirements.txt
python build_design_grid.py
```

The files are written to `../data/` (or to the folder given in the environment variable `ACTMOF_DATA_DIR`).
The run takes from tens of minutes to a few hours, depending on the CPU. All random states are fixed, so a rebuild
reproduces the shipped files, apart from small floating-point differences between versions of scikit-learn.
