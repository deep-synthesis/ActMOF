"""
curve_best_q_UNIFIED.py -- the all-39-methods "cool" overview plot, rebuilt on
the UNIFIED (single-candidate-sampling-mechanism) dataset.

Same script as curve_best_q_POSTFIX.py (which itself replicated the ORIGINAL
Benchmark v3.ipynb plot_metric_curve() function's look: original saved
per-method color/marker/linestyle from figuredata_method_style_map.csv, SE
shading at alpha=0.08, full iteration range 0-49) -- only the data source
changed, from all_methods_PERRUN_curves_POSTFIX.csv to
all_methods_PERRUN_curves_UNIFIED.csv (all 39 methods now share one
candidate-sampling mechanism; see fig3.py docstring for the
full explanation). Visual identity (colors/markers/linestyles/shading) is
unchanged from curve_best_q_POSTFIX.py -- only the 29 previously-untouched
methods' curve values differ.

Cost: $0 -- pure data-wrangling + matplotlib, no API calls.

Portability note: this is otherwise byte-identical to the original
curve_best_q_UNIFIED.py -- only ROOT/STYLE_PATH/DATA_PATH/OUT_PATH below were
repointed at this repo's own relative layout (method_style_map.csv shipped
alongside this script; the PERRUN csv now read from ../outputs/, where
../benchmark/merge_outputs.py writes it) instead of the original hardcoded
absolute path. No plotting/aggregation logic changed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
STYLE_PATH = os.path.join(_HERE, "method_style_map.csv")
DATA_PATH = os.path.join(_REPO_ROOT, "outputs", "all_methods_PERRUN_curves_UNIFIED.csv")
OUT_PATH = os.path.join(_HERE, "curve_best_q_UNIFIED.png")

style = pd.read_csv(STYLE_PATH).sort_values("method_order")
METHOD_ORDER = style["method"].tolist()
METHOD_COLORS = dict(zip(style["method"], style["saved_color"]))
METHOD_MARKERS = dict(zip(style["method"], style["saved_marker"]))
METHOD_LINESTYLES = dict(zip(style["method"], style["saved_linestyle"]))

perrun = pd.read_csv(DATA_PATH)
agg = perrun.groupby(["method", "iteration"])["best_q"].agg(["mean", "std", "count"]).reset_index()

plt.figure(figsize=(13, 7))

for method in METHOD_ORDER:
    s = agg[agg["method"] == method].sort_values("iteration")
    if len(s) == 0:
        continue
    color = METHOD_COLORS.get(method)
    marker = METHOD_MARKERS.get(method, "o")
    linestyle = METHOD_LINESTYLES.get(method, "-")

    plt.plot(
        s["iteration"], s["mean"],
        marker=marker, linestyle=linestyle, linewidth=1.35, markersize=3.5,
        color=color, label=method,
    )
    if s["count"].max() > 1:
        se = s["std"].fillna(0) / np.sqrt(s["count"].clip(lower=1))
        plt.fill_between(s["iteration"], s["mean"] - se, s["mean"] + se, color=color, alpha=0.08)

plt.xlabel("Active-learning iteration, batch size = 3")
plt.ylabel("Best q found so far")
plt.title("Best q found so far  (unified candidate-sampling dataset, all 39 methods)")
plt.grid(True, alpha=0.25)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True, ncol=1)
plt.figtext(0.005, 0.005, 'curve_best_q_UNIFIED  |  unified candidate-sampling dataset',
            fontsize=7, color='#999999', ha='left', va='bottom')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
print(f"saved {OUT_PATH}")
