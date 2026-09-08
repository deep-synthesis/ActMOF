# ============================================
# build_design_grid.py -- ActMOF virtual sandbox construction
# ============================================
# Provenance: copied verbatim (algorithm untouched) from the first code cell
# of `construct dataset with time tracker.ipynb` (the project's original
# dataset-construction notebook). Only the output-path handling near the
# bottom (OUTPUT_DIR / write_chunk) was made configurable for portability;
# no numerical logic, model, or rule was changed.
#
# What this does, in one paragraph: fits two RandomForestRegressor models
# (rf_int -> Intensity_exp, rf_fwhm -> FWHM_exp) on the ~96 unique real
# microwave-synthesis experiments hardcoded below (X / max_intensity / FWHM
# arrays), then walks the full 5-variable grid implied by BOUNDS
# (6,101,172 conditions) and, for every non-experimental point, predicts
# Intensity/FWHM with those two RFs and blends the prediction with nearby
# real experiments (Rule 1: neighbors within Chebyshev distance 1 -> 80%
# experimental average + 20% RF; Rule 2: distance-2 neighbors -> 50/50),
# then applies the failure-value rules (Intensity<3000 -> q=0, FWHM>3 -> 30)
# to get q_final = Intensity_final / FWHM_final for every row. Output is the
# design_grid_part*.csv series consumed by every script under ../benchmark/.
#
# Runtime: this streams and predicts over ~6.1M rows -- expect it to take
# from tens of minutes to a few hours depending on BATCH_ROWS/CPU. There is
# no way to shrink this without changing FULL_GRID/BOUNDS, which would
# change the resulting dataset, not just how fast you get it.
# ============================================
# Fast full-grid generator with vectorized rules, batching, progress + ETA
# - Rule 1: all five params within 1 step -> 80% EXP + 20% RF
# - Rule 2: all diffs ≤ 2 and exactly 1 or 2 dims at distance 2, others ≤ 1 -> 50% EXP + 50% RF
# - Overwrites and rounding preserved
# - Streams to CSV or XLSX in parts
# ============================================

import os
import time
import sys
import math
from pathlib import Path

# 0) Setup
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "scikit-learn", "-q"])
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error

# 1) Config
BOUNDS = {
    "metal_amount":        (5, 75, 1),
    "modulator":           (5, 15, 1),
    "add_solvent":         (0, 30, 1),
    "reaction_time":       (1, 12, 1),
    "reaction_temperature":(10, 30, 1)
}
ORDERED_COLS = ["metal_amount","modulator","add_solvent","reaction_time","reaction_temperature"]

FULL_GRID = True             # True -> 6,101,172 rows. False -> only experiments + Chebyshev radius 2 shell
OUTPUT_FORMAT = "csv"        # "csv" is much faster. Use "xlsx" only if required.
CSV_ROWS_PER_FILE = 1_000_000
XLSX_ROWS_PER_FILE = 1_000_000  # Excel max 1,048,576
OUTPUT_BASENAME = "design_grid_part"

# Where the design_grid_part*.csv files are written. Defaults to the `data/`
# folder that ships alongside this repo (ActMOF_Code/data/), so the benchmark
# scripts in ../benchmark/families/ find the output with zero configuration.
# Override with the ACTMOF_DATA_DIR environment variable if you want it
# elsewhere.
OUTPUT_DIR = Path(os.environ.get("ACTMOF_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_ROWS = 50_000          # prediction batch size. Tune for memory and speed
PROGRESS_EVERY_ROWS = 100_000
N_TREES = 400

# 2) Experimental dataset (96 unique runs)
X = np.array([
    [35,5,22,9,14],[68,7,15,12,24],[12,9,1,7,29],[36,10,24,3,18],[55,11,8,2,15],
    [69,6,23,1,20],[14,12,2,11,15],[42,7,27,6,17],[9,14,7,4,11],[45,6,11,8,28],
    [27,12,1,2,25],[41,14,5,3,16],
    [5,5,0,1,30],[5,15,0,1,30],[75,15,0,1,30],
    [16,7,22,3,20],[16,9,23,4,21],[15,7,12,11,19],
    [6,5,29,11,27],[62,7,26,11,26],[5,15,2,11,16],
    [33,10,7,11,18],[9,6,1,12,19],[48,14,3,12,23],
    [19,8,12,11,16],[15,7,14,11,24],[12,8,10,11,18],
    [15,7,13,11,18],[14,7,11,11,20],[7,6,20,11,15],
    [15,7,14,11,24],[15,6,10,11,14],[15,7,10,11,12],
    [15,8,12,11,19],[9,8,13,7,20],[21,6,13,12,19],
    [6,6,12,11,20],[15,8,9,12,19],[75,12,6,11,18],
    [25,13,14,7,24],[17,13,16,12,24],[38,13,16,8,24],
    [65,5,0,12,24],[74,7,23,1,16],[5,7,21,12,20],
    [15,7,16,12,29],[16,7,15,11,18],[62,10,12,10,17],
    [11,7,12,11,24],[17,7,16,11,27],[13,7,10,12,25],
    [13,6,13,11,24],[15,5,0,12,24],[21,5,26,11,23],
    [16,7,8,11,24],[28,7,0,12,30],[10,7,0,12,24],
    [23,7,30,11,24],[20,7,30,11,29],[23,7,14,11,25],
    [16,7,18,12,25],[17,7,18,12,24],[24,7,18,12,24],
    [16,7,14,11,30],[15,7,13,9,23],[15,7,7,7,26],
    [13,6,25,12,24],[13,7,11,11,23],[14,7,7,11,26],
    [15,6,23,12,24],[15,5,8,12,23],[15,6,14,12,21],
    [13,5,10,11,24],[17,6,11,11,22],[19,5,2,12,24],
    [6,6,19,12,30],[25,7,18,11,24],[27,7,27,12,26],
    [15,5,8,11,23],[15,5,11,10,21],[24,5,30,12,24],
    [13,7,15,12,25],[16,7,11,12,24],[14,6,15,12,26],
    [14,7,15,11,26],[18,7,10,12,23],[10,7,13,12,29],
    [23,7,2,12,19],[18,7,1,4,19],[19,6,4,12,25],
    [17,7,10,12,25],[16,6,6,5,23],[20,6,0,12,23],
    [17,7,20,11,21],[24,7,22,8,21],[25,7,19,10,27]
], dtype=np.int16)
max_intensity = np.array([
    0,0,785,0,0,605,0,0,182,0,880,0,
    1457,0,0,
    2345,960,16751,
    7575,0,0,
    8060,1318,920,
    0,23969,7305,
    7703,12434,0,
    23969,0,0,
    1103,0,17269,
    1077,903,0,
    0,0,0,
    0,0,0,
    31082,12791,0,
    28179,29258,24240,
    37307,15489,9906,
    21878,11411,688,
    18169,10048,12005,
    35900,35967,13410,
    2331,23301,11663,
    23658,17558,7418,
    32110,20121,19118,
    34595,17880,6978,
    0,8241,7910,
    26277,18112,4394,
    27452,16348,30033,
    28764,28299,1082,
    20334,19300,12874,
    28836,16918,11668,
    26346,17895,11657
], dtype=np.float32)
FWHM = np.array([
    30,30,2.2,30,30,30,30,30,30,30,3.9,30,
    0.32,30,30,
    0.37,1.1,0.22,
    0.18,30,30,
    0.25,0.40,4.6,
    30,0.21,0.15,
    0.27,0.24,30,
    0.21,30,30,
    0.8,30,0.21,
    0.48,2,30,
    30,30,30,
    30,30,30,
    0.2,0.3,30,
    0.26,0.29,0.24,
    0.18,0.28,0.26,
    0.21,0.22,0.6,
    0.2,0.22,0.24,
    0.22,0.21,0.22,
    0.3,0.29,0.23,
    0.28,0.26,0.29,
    0.3,0.21,0.25,
    0.14,0.22,0.16,
    30,0.26,0.23,
    0.22,0.29,0.31,
    0.24,0.28,0.23,
    0.29,0.22,0.41,
    0.27,0.32,0.21,
    0.26,0.31,0.30,
    0.28,0.31,0.34
], dtype=np.float32)

# Build experimental DataFrame and unique means
exp_df = pd.DataFrame(X, columns=ORDERED_COLS)
exp_df["Intensity_exp"] = max_intensity.astype(float)
exp_df["FWHM_exp"] = FWHM.astype(float)

def _calc_q(intensity, fwhm):
    if fwhm == 30 or intensity <= 0:
        return 0
    if fwhm <= 0:
        return 0
    return int(round(float(intensity) / float(fwhm)))

group_cols = ORDERED_COLS
agg_exp = exp_df.groupby(group_cols, as_index=False).agg(
    Intensity_exp=("Intensity_exp", "mean"),
    FWHM_exp=("FWHM_exp", "mean")
)
agg_exp["q_exp"] = [_calc_q(i, f) for i, f in zip(agg_exp["Intensity_exp"], agg_exp["FWHM_exp"])]
agg_exp = agg_exp.sort_values(group_cols).reset_index(drop=True)
agg_exp["label_id"] = ["Q" + str(i + 1).zfill(3) for i in range(len(agg_exp))]

# Fast lookups
exp_params = agg_exp[ORDERED_COLS].to_numpy(dtype=np.int16)       # E x 5
exp_I = agg_exp["Intensity_exp"].to_numpy(dtype=np.float32)       # E
exp_F = agg_exp["FWHM_exp"].to_numpy(dtype=np.float32)            # E

# Map from tuple -> (label, I, F, q)
exp_lookup = {
    tuple(int(v) for v in row[:5]): (lab, float(i), float(f), int(q))
    for row, lab, i, f, q in zip(agg_exp[ORDERED_COLS].to_numpy(), agg_exp["label_id"], agg_exp["Intensity_exp"], agg_exp["FWHM_exp"], agg_exp["q_exp"])
}

# 3) Train RFs with progress
X_train = agg_exp[ORDERED_COLS].to_numpy()
y_int = agg_exp["Intensity_exp"].to_numpy()
y_fwhm = agg_exp["FWHM_exp"].to_numpy()

def cv_with_progress(X, y, n_estimators, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s, rmses = [], []
    t0 = time.time()
    print(f"[TRAIN] {label}: 5-fold CV with {n_estimators} trees")
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        m = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2 = r2_score(y[te], p)
        rmse = mean_squared_error(y[te], p, squared=False)
        r2s.append(r2); rmses.append(rmse)
        elapsed = time.time() - t0
        rate = fold / max(elapsed, 1e-6)
        eta = (5 - fold) / max(rate, 1e-6)
        print(f"[TRAIN] {label}: fold {fold}/5 | elapsed {elapsed:.1f}s | ETA {eta:.1f}s | R2 {np.mean(r2s):.3f} RMSE {np.mean(rmses):.2f}")
    return np.mean(r2s), np.std(r2s), np.mean(rmses), np.std(rmses)

r2_int_m, r2_int_s, rmse_int_m, rmse_int_s = cv_with_progress(X_train, y_int, N_TREES, "Intensity")
r2_fwhm_m, r2_fwhm_s, rmse_fwhm_m, rmse_fwhm_s = cv_with_progress(X_train, y_fwhm, N_TREES, "FWHM")

print("[TRAIN] Fitting final RF models...")
t_fit = time.time()
rf_int = RandomForestRegressor(n_estimators=N_TREES, random_state=42, n_jobs=-1).fit(X_train, y_int)
rf_fwhm = RandomForestRegressor(n_estimators=N_TREES, random_state=42, n_jobs=-1).fit(X_train, y_fwhm)
print(f"[TRAIN] Final fit done in {time.time() - t_fit:.1f}s")

# 4) Utilities
bounds_ranges = [range(BOUNDS[c][0], BOUNDS[c][1] + 1, BOUNDS[c][2]) for c in ORDERED_COLS]
n_total_grid = np.prod([len(r) for r in bounds_ranges])

def in_bounds_block(keys):
    # keys: B x 5
    return (
        (keys[:,0] >= BOUNDS["metal_amount"][0]) & (keys[:,0] <= BOUNDS["metal_amount"][1]) &
        (keys[:,1] >= BOUNDS["modulator"][0]) & (keys[:,1] <= BOUNDS["modulator"][1]) &
        (keys[:,2] >= BOUNDS["add_solvent"][0]) & (keys[:,2] <= BOUNDS["add_solvent"][1]) &
        (keys[:,3] >= BOUNDS["reaction_time"][0]) & (keys[:,3] <= BOUNDS["reaction_time"][1]) &
        (keys[:,4] >= BOUNDS["reaction_temperature"][0]) & (keys[:,4] <= BOUNDS["reaction_temperature"][1])
    )

def postprocess_and_round_vec(Ip, Fp):
    # Overwrites and rounding vectorized
    Ip = Ip.astype(np.float64)
    Fp = Fp.astype(np.float64)
    # Intensity < 3000 -> 0 and FWHM 30
    mask_lowI = Ip < 3000.0
    Ip[mask_lowI] = 0.0
    Fp[mask_lowI] = 30.0
    # FWHM > 3 -> 30 (only where not already set by lowI)
    mask_hiF = (Fp > 3.0) & (~mask_lowI)
    Fp[mask_hiF] = 30.0
    # Round intensity to int
    I_final = np.rint(Ip).astype(np.int64)
    # Round FWHM per rule
    F_final = np.empty_like(Fp)
    m30 = Fp == 30.0
    mlt1 = (Fp < 1.0) & (~m30)
    mother = (~mlt1) & (~m30)
    F_final[m30] = 30
    F_final[mlt1] = np.round(Fp[mlt1], 2)
    F_final[mother] = np.round(Fp[mother], 1)
    # q index
    q = np.zeros_like(I_final)
    valid = (F_final != 30) & (I_final > 0)
    q[valid] = np.rint(I_final[valid] / F_final[valid]).astype(np.int64)
    return I_final, F_final, q

# 5) Neighbor rules, vectorized over a batch
def apply_rules_batch(keys_batch, rfI_pred, rfF_pred):
    """
    keys_batch: B x 5 int16
    rfI_pred, rfF_pred: B arrays
    returns I_final, F_final, q_final, rule_flags
    rule_flags: 0 none, 1 rule1, 2 rule2
    """
    B = keys_batch.shape[0]
    E = exp_params.shape[0]
    kf = keys_batch.astype(np.int16)

    # Diffs: B x E x 5
    # Use float32 to save RAM
    diffs = np.abs(kf[:, None, :].astype(np.float32) - exp_params[None, :, :].astype(np.float32))  # B,E,5

    # Rule 1: all five ≤ 1
    r1_mask_BE = np.all(diffs <= 1.0 + 1e-9, axis=2)                  # B,E
    r1_has = np.any(r1_mask_BE, axis=1)                               # B

    # Means over experimental neighbors for rule 1
    r1_counts = np.maximum(r1_mask_BE.sum(axis=1), 1)                 # avoid div0
    r1_I_mean = (r1_mask_BE @ exp_I) / r1_counts
    r1_F_mean = (r1_mask_BE @ exp_F) / r1_counts

    # Rule 2: all ≤ 2 and exactly 1 or 2 dims at 2, others ≤ 1
    le2 = np.all(diffs <= 2.0 + 1e-9, axis=2)                         # B,E
    eq2 = np.isclose(diffs, 2.0)                                      # B,E,5
    count2 = np.sum(eq2, axis=2)                                      # B,E
    # others ≤ 1 means for positions not equal to 2, the diff ≤ 1
    not2 = ~eq2
    # For each row, ensure max diff on non-2 positions ≤ 1
    # Use big negative for masked positions so max ignores them correctly
    non2_max = np.max(np.where(not2, diffs, -np.inf), axis=2)         # B,E
    others_le1 = (non2_max <= 1.0 + 1e-9)
    r2_mask_BE = le2 & others_le1 & ((count2 == 1) | (count2 == 2))   # B,E
    r2_has = np.any(r2_mask_BE, axis=1)

    r2_counts = np.maximum(r2_mask_BE.sum(axis=1), 1)
    r2_I_mean = (r2_mask_BE @ exp_I) / r2_counts
    r2_F_mean = (r2_mask_BE @ exp_F) / r2_counts

    # Blend
    I_blend = rfI_pred.copy()
    F_blend = rfF_pred.copy()
    rule_flags = np.zeros(B, dtype=np.int8)

    # Apply Rule 1 where true
    idx1 = np.where(r1_has)[0]
    I_blend[idx1] = 0.8 * r1_I_mean[idx1] + 0.2 * rfI_pred[idx1]
    F_blend[idx1] = 0.8 * r1_F_mean[idx1] + 0.2 * rfF_pred[idx1]
    rule_flags[idx1] = 1

    # For the rest, apply Rule 2 where true and Rule 1 not already applied
    idx2 = np.where(~r1_has & r2_has)[0]
    I_blend[idx2] = 0.5 * r2_I_mean[idx2] + 0.5 * rfI_pred[idx2]
    F_blend[idx2] = 0.5 * r2_F_mean[idx2] + 0.5 * rfF_pred[idx2]
    rule_flags[idx2] = 2

    return I_blend, F_blend, rule_flags

# 6) Generators for keys
def grid_key_iter():
    for a in bounds_ranges[0]:
        for b in bounds_ranges[1]:
            for c in bounds_ranges[2]:
                for d in bounds_ranges[3]:
                    for e in bounds_ranges[4]:
                        yield (a,b,c,d,e)

def neighborhood_keys():
    # Experiments plus Chebyshev radius 2 shell around them
    seen = set(tuple(int(v) for v in row) for row in exp_params.tolist())
    offs = [-2,-1,0,1,2]
    from itertools import product
    for base in exp_params:
        ba = base.astype(int)
        for o in product(offs, offs, offs, offs, offs):
            nb = (ba[0]+o[0], ba[1]+o[1], ba[2]+o[2], ba[3]+o[3], ba[4]+o[4])
            # bounds check
            if (BOUNDS["metal_amount"][0] <= nb[0] <= BOUNDS["metal_amount"][1] and
                BOUNDS["modulator"][0] <= nb[1] <= BOUNDS["modulator"][1] and
                BOUNDS["add_solvent"][0] <= nb[2] <= BOUNDS["add_solvent"][1] and
                BOUNDS["reaction_time"][0] <= nb[3] <= BOUNDS["reaction_time"][1] and
                BOUNDS["reaction_temperature"][0] <= nb[4] <= BOUNDS["reaction_temperature"][1]):
                seen.add(nb)
    for k in sorted(seen):
        yield k

# 7) Writers
def write_chunk(df, file_idx, header_written_map):
    if OUTPUT_FORMAT == "csv":
        rows_per_file = CSV_ROWS_PER_FILE
        out = str(OUTPUT_DIR / f"{OUTPUT_BASENAME}{str(file_idx).zfill(3)}.csv")
        mode = "a" if header_written_map.get(out, False) else "w"
        df.to_csv(out, index=False, mode=mode, header=(mode=="w"))
        header_written_map[out] = True
    elif OUTPUT_FORMAT == "xlsx":
        try:
            import xlsxwriter  # noqa
        except Exception:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter", "-q"])
            import xlsxwriter  # noqa
        rows_per_file = XLSX_ROWS_PER_FILE
        # open, write, close per chunk
        out = str(OUTPUT_DIR / f"{OUTPUT_BASENAME}{str(file_idx).zfill(3)}.xlsx")
        # if file exists and header written, append by reading and rewriting is too slow
        # instead, we create new file per chunk index boundary
        # so we rely on caller to roll file_idx when chunk row count would exceed limit
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
    else:
        raise ValueError("OUTPUT_FORMAT must be 'csv' or 'xlsx'")

# 8) Main streaming loop
def run_stream():
    global_count = 0
    file_idx = 1
    rows_in_current_file = 0
    header_written_map = {}
    Y_counter = 1
    RULE1_COUNT = 0
    RULE2_COUNT = 0

    total_target = n_total_grid if FULL_GRID else None
    t0 = time.time()

    # choose key source
    key_source = grid_key_iter() if FULL_GRID else neighborhood_keys()

    # batch buffers
    batch_keys = []

    print("[WRITE] Streaming output...")
    print("[PREDICT] Starting batched prediction...")

    for key in key_source:
        batch_keys.append(key)
        if len(batch_keys) >= BATCH_ROWS:
            # process batch
            keys_arr = np.array(batch_keys, dtype=np.int16)
            B = keys_arr.shape[0]

            # Split mask: experimental vs synthetic
            is_exp = np.array([tuple(int(x) for x in k) in exp_lookup for k in batch_keys], dtype=bool)

            # Prepare outputs
            I_final = np.zeros(B, dtype=np.int64)
            F_final = np.zeros(B, dtype=np.float64)
            q_final = np.zeros(B, dtype=np.int64)
            rule_applied = np.zeros(B, dtype=np.int8)
            label = np.empty(B, dtype=object)

            # Experimental rows
            if np.any(is_exp):
                idx = np.where(is_exp)[0]
                exp_vals_list = [exp_lookup[tuple(int(x) for x in k)] for k in keys_arr[idx]]
                lab_arr = [t[0] for t in exp_vals_list]
                I_e = np.array([t[1] for t in exp_vals_list])
                F_e = np.array([t[2] for t in exp_vals_list])
                q_e = np.array([t[3] for t in exp_vals_list], dtype=np.int64)

                label[idx] = lab_arr
                I_final[idx] = np.rint(I_e).astype(np.int64)
                # keep FWHM formatting: if 30, int
                Ftmp = F_e.astype(np.float64)
                Ftmp = np.where(Ftmp==30.0, 30.0, Ftmp)
                F_final[idx] = Ftmp
                q_final[idx] = q_e
                rule_applied[idx] = 0

            # Synthetic rows: RF + neighbor rules
            if np.any(~is_exp):
                idx = np.where(~is_exp)[0]
                K_syn = keys_arr[idx].astype(np.int16)
                # RF predictions in a vectorized call
                rfI = rf_int.predict(K_syn)
                rfF = rf_fwhm.predict(K_syn)

                # Apply rules vectorized
                I_blend, F_blend, rflags = apply_rules_batch(K_syn, rfI, rfF)
                # Overwrites + rounding
                I_pp, F_pp, q_pp = postprocess_and_round_vec(I_blend, F_blend)

                I_final[idx] = I_pp
                F_final[idx] = F_pp
                q_final[idx] = q_pp
                rule_applied[idx] = rflags

                # labels Y...
                labY = np.array([f"Y{str(i).zfill(7)}" for i in range(Y_counter, Y_counter + len(idx))], dtype=object)
                label[idx] = labY
                Y_counter += len(idx)

                # update counters
                RULE1_COUNT += int(np.sum(rflags == 1))
                RULE2_COUNT += int(np.sum(rflags == 2))

            # Assemble DataFrame
            df = pd.DataFrame({
                "metal_amount": keys_arr[:,0].astype(int),
                "modulator": keys_arr[:,1].astype(int),
                "add_solvent": keys_arr[:,2].astype(int),
                "reaction_time": keys_arr[:,3].astype(int),
                "reaction_temperature": keys_arr[:,4].astype(int),
                "label_id": label,
            })

            # Fill experimental columns
            I_exp_col = np.where(is_exp, I_final, np.nan)
            F_exp_col = np.where(is_exp, F_final, np.nan)
            q_exp_col = np.where(is_exp, q_final, np.nan)

            # Final columns
            df["Intensity_exp"] = I_exp_col
            # For formatting consistency, if F_exp is 30.0 display as 30
            F_exp_fmt = np.where((~np.isnan(F_exp_col)) & (np.isclose(F_exp_col, 30.0)), 30, F_exp_col)
            df["FWHM_exp"] = F_exp_fmt
            df["q_exp"] = q_exp_col

            df["Intensity_final"] = I_final
            df["FWHM_final"] = np.where(np.isclose(F_final, 30.0), 30, F_final)
            df["q_final"] = q_final
            df["rule_applied"] = rule_applied

            # File rolling
            rows_per_file = CSV_ROWS_PER_FILE if OUTPUT_FORMAT == "csv" else XLSX_ROWS_PER_FILE
            if rows_in_current_file + B > rows_per_file:
                # start a new file
                file_idx += 1
                rows_in_current_file = 0

            write_chunk(df, file_idx, header_written_map)
            rows_in_current_file += B

            # Progress
            global_count += B
            elapsed = time.time() - t0
            rate = global_count / max(elapsed, 1e-6)
            if total_target:
                rem = max(total_target - global_count, 0)
                eta = rem / max(rate, 1e-6)
                print(f"[WRITE] rows {global_count:,}/{total_target:,} | files {file_idx} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s | rate {rate:,.0f} rows/s | R1 {RULE1_COUNT:,} R2 {RULE2_COUNT:,}")
            else:
                print(f"[WRITE] rows {global_count:,} | files {file_idx} | elapsed {elapsed:.1f}s | rate {rate:,.0f} rows/s | R1 {RULE1_COUNT:,} R2 {RULE2_COUNT:,}")

            # reset batch
            batch_keys = []

    # Flush tail
    if batch_keys:
        keys_arr = np.array(batch_keys, dtype=np.int16)
        B = keys_arr.shape[0]
        is_exp = np.array([tuple(int(x) for x in k) in exp_lookup for k in batch_keys], dtype=bool)

        I_final = np.zeros(B, dtype=np.int64)
        F_final = np.zeros(B, dtype=np.float64)
        q_final = np.zeros(B, dtype=np.int64)
        rule_applied = np.zeros(B, dtype=np.int8)
        label = np.empty(B, dtype=object)

        if np.any(is_exp):
            idx = np.where(is_exp)[0]
            exp_vals_list = [exp_lookup[tuple(int(x) for x in k)] for k in keys_arr[idx]]
            lab_arr = [t[0] for t in exp_vals_list]
            I_e = np.array([t[1] for t in exp_vals_list])
            F_e = np.array([t[2] for t in exp_vals_list])
            q_e = np.array([t[3] for t in exp_vals_list], dtype=np.int64)
            label[idx] = lab_arr
            I_final[idx] = np.rint(I_e).astype(np.int64)
            F_final[idx] = np.where(np.isclose(F_e, 30.0), 30.0, F_e)
            q_final[idx] = q_e
            rule_applied[idx] = 0

        if np.any(~is_exp):
            idx = np.where(~is_exp)[0]
            K_syn = keys_arr[idx].astype(np.int16)
            rfI = rf_int.predict(K_syn)
            rfF = rf_fwhm.predict(K_syn)
            I_blend, F_blend, rflags = apply_rules_batch(K_syn, rfI, rfF)
            I_pp, F_pp, q_pp = postprocess_and_round_vec(I_blend, F_blend)
            I_final[idx] = I_pp
            F_final[idx] = F_pp
            q_final[idx] = q_pp
            rule_applied[idx] = rflags
            labY = np.array([f"Y{str(i).zfill(7)}" for i in range(Y_counter, Y_counter + len(idx))], dtype=object)
            label[idx] = labY

        df = pd.DataFrame({
            "metal_amount": keys_arr[:,0].astype(int),
            "modulator": keys_arr[:,1].astype(int),
            "add_solvent": keys_arr[:,2].astype(int),
            "reaction_time": keys_arr[:,3].astype(int),
            "reaction_temperature": keys_arr[:,4].astype(int),
            "label_id": label,
        })
        I_exp_col = np.where(is_exp, I_final, np.nan)
        F_exp_col = np.where(is_exp, F_final, np.nan)
        q_exp_col = np.where(is_exp, q_final, np.nan)
        df["Intensity_exp"] = I_exp_col
        df["FWHM_exp"] = np.where((~np.isnan(F_exp_col)) & (np.isclose(F_exp_col, 30.0)), 30, F_exp_col)
        df["q_exp"] = q_exp_col
        df["Intensity_final"] = I_final
        df["FWHM_final"] = np.where(np.isclose(F_final, 30.0), 30, F_final)
        df["q_final"] = q_final
        df["rule_applied"] = rule_applied

        rows_per_file = CSV_ROWS_PER_FILE if OUTPUT_FORMAT == "csv" else XLSX_ROWS_PER_FILE
        if rows_in_current_file + B > rows_per_file:
            file_idx += 1
            rows_in_current_file = 0
        write_chunk(df, file_idx, header_written_map)
        global_count += B

    elapsed_total = time.time() - t0
    print("====================================")
    print("Streaming complete")
    print("Rows written:", f"{global_count:,}")
    print("Files created:", file_idx)
    print("Elapsed:", f"{elapsed_total:.1f}s")
    return file_idx, global_count

# 9) Run
if FULL_GRID:
    print(f"[INFO] Full-grid total conditions: {n_total_grid:,}")
else:
    print("[INFO] Neighborhood mode (experiments + radius 2)")

files_created, rows_written = run_stream()

# 10) Summary metrics
print("====================================")
print("Unique experimental conditions:", len(agg_exp))
print("Random Forest accuracy (5-fold CV):")
print(f"  Intensity -> R2 {r2_int_m:.3f} ± {r2_int_s:.3f}, RMSE {rmse_int_m:.1f} ± {rmse_int_s:.1f}")
print(f"  FWHM     -> R2 {r2_fwhm_m:.3f} ± {r2_fwhm_s:.3f}, RMSE {rmse_fwhm_m:.3f} ± {rmse_fwhm_s:.3f}")
print(f"Output format: {OUTPUT_FORMAT}")
print(f"Output pattern: {OUTPUT_BASENAME}NNN.{ 'csv' if OUTPUT_FORMAT=='csv' else 'xlsx'}")
print("Columns: ['metal_amount','modulator','add_solvent','reaction_time','reaction_temperature',"
      "'label_id','Intensity_exp','FWHM_exp','q_exp','Intensity_final','FWHM_final','q_final','rule_applied']")
print("====================================")
