# plotting

`curve_best_q_UNIFIED.py` produces the all-thirty-nine-methods "best q found so far" curve. It reads `../outputs/all_methods_PERRUN_curves_UNIFIED.csv`, produced by `../benchmark/merge_outputs.py`, together with `method_style_map.csv`, the per-method color, marker, and linestyle assignments this project has used consistently across every version of this figure, and writes `curve_best_q_UNIFIED.png` alongside itself.

```bash
python curve_best_q_UNIFIED.py
```

No API calls and no R are involved; this is pure pandas and matplotlib, running in seconds. **The script is self-sufficient, requiring no intermediate file beyond the raw benchmark output**, and was used to validate the whole pipeline in this repository end to end, with `merge_outputs.py`'s output fed directly into it and compared against the previously published figure before this repository was finalized.

`build_figuredata_curve_best_q.py` builds `figuredata_curve_best_q_UNIFIED.csv` from `../benchmark/reference_output/all_methods_PERRUN_curves_UNIFIED.csv`, or from a fresh run's own PERRUN csv, collapsing the five independent runs per method into one summary curve via `groupby(method, family, iteration)['best_q'].agg(mean, std, count)` together with `standard_error = std/sqrt(count)`. The five raw per-run trajectories are noisy and are not themselves plotted; rather, this mean, one number per method per iteration, is the line that `fig3.py` and `curve_best_q_UNIFIED.py` actually draw.

```bash
python build_figuredata_curve_best_q.py            # (re)writes figuredata_curve_best_q_UNIFIED.csv
python build_figuredata_curve_best_q.py --check    # compare against the shipped file instead of overwriting it
```

**Verified two independent ways before being shipped.** First, `--check` against the already published `figuredata_curve_best_q_UNIFIED.csv` shows `mean` matching exactly, with a maximum difference of zero, while `std` and `standard_error` match to roughly 1e-11, attributable to floating-point noise. Second, eight randomly sampled method-iteration pairs, recomputed by hand with plain Python `statistics.stdev` rather than pandas `.agg()`, so as to avoid trusting the same library code path twice, all matched to within 1e-6. The currently shipped `figuredata_curve_best_q_UNIFIED.csv` was deliberately left untouched rather than overwritten with this script's output, given that it is already pixel-verified against the published Figure 3 below and the two differ only by float noise.

`fig3.py` produces the actual published Figure 3, covering sixteen of the thirty-nine methods (`SELECTED_METHODS` in the script) and iterations 0 through 41 only, the established display convention for this figure, albeit the underlying data extends to iteration 49. It reads `figuredata_curve_best_q_UNIFIED.csv`, built above. All color, linestyle, label, and legend-grouping choices are inline in the script by design, with no external style file, unlike `curve_best_q_UNIFIED.py`.

```bash
python fig3.py
```

Verified pixel for pixel against the figure this project had already published, the `PIL.ImageChops.difference` bounding box returning `None`, before this copy was finalized.
