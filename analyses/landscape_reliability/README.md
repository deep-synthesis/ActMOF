# landscape_reliability (SI Section S2.5)

| File | Content |
|---|---|
| `leave_one_out_predictions.csv` | one row per measured condition: measured values, leave-one-out prediction (mean of five forests), distance to the nearest other measured condition, share of the forests beyond each threshold, errors |
| `build_error_dataset.py` | produces `leave_one_out_predictions.csv` (`python build_error_dataset.py`) |
| `gate.py` | distance to the nearest measured condition and vote of the forests for arbitrary conditions |
| `gate_metadata.npz` | `d_min`, `B_I`, `B_F` for all 6,101,172 conditions of the landscape |
| `support_by_region.csv` | SI Table S8: support of the landscape by the measured conditions, by region |

`leave_one_out_predictions.csv` is the file behind the SI. A rerun of `build_error_dataset.py` (about 2 min on 4 CPU cores)
gives the same intensity predictions and classifications; the FWHM predictions can differ in the second decimal place
with the scikit-learn version, which shifts individual predicted `q` slightly.
