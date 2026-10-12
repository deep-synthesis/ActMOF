# wetlab

Data of the six validation campaigns (CAU-10, CAU-23, MIL-53-muc, MIL-160, MTV-MIL-160, MTV-MOF-303).

| Folder | Content |
|---|---|
| `campaigns/` | one CSV per MOF with the 12 experiments of the campaign (R00001-R00012) as recorded by the program: round, the five codes, measured intensity and FWHM, `q`, the prediction of the model for the proposed condition, and in `notes` the calibration used for the proposal |
| `pxrd/patterns/` | the PXRD pattern of every experiment (`<MOF>_R<n>.xy`: 2-theta, intensity); `pxrd/source_mapping.csv` gives the original file of each pattern |
| `pxrd/rasx_process.py` | reads a Rigaku `.rasx` file and returns the intensity and FWHM of the diagnostic reflection |
| `pxrd/reflux/` | patterns of the reflux reference products (repeated loadings) |
| `pxrd/cifs/` | structures used for the simulated patterns |
| `sorption/` | N2 isotherms (instrument exports) of the four samples of SI Section S8 and their BET and Langmuir areas |

Rounds: round 0 is proposed from the MOF-321 data alone; rounds 1-3 from the calibrated prior with the results of the
new MOF. The program itself (web and desktop versions) is at https://github.com/Mr-Thiol/ActMOF_Website.
