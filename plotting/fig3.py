"""
Figure 3 main -- Best q Curve, v7
====================================================
DATA SOURCE CHANGE vs v5/v6 (this is the important change in this version):
  v5 and earlier used figuredata_curve_best_q_POSTFIX.csv, built from a mix of
  (a) the original Benchmark v3.ipynb's own pool_df/rng.choice candidate
      sampling for ~29 never-bugged methods, and (b) 10 separately-patched
      methods (HYBRID x2, LLM_PURE x3, laGP x2, RGASP, BNN, RVM_like) each
      rerun via its own one-off script using a DIFFERENT candidate-sampling
      mechanism (continuous rejection sampling, no pool_df).
  v7 uses figuredata_curve_best_q_UNIFIED.csv instead: the same 10 methods
  (unchanged, still from their own validated rerun scripts) PLUS the other 29
  methods rerun under unified_bo_rerun.py -- ALL 39 methods now share the
  identical candidate-sampling mechanism (rejection sampling over
  FEATURE_RANGES + greedy_diverse_batch), so there is one consistent upstream
  RNG/candidate-drawing scheme across the entire corrected dataset, not two.
  Consequence: the 29 previously-untouched methods' exact curve VALUES change
  from v5/v6 (new valid random draws under the new mechanism, not a
  correction of an error -- see chat history for the seed/RNG-consumption
  analysis). Overall method rankings are qualitatively similar, but exact
  numbers for those 29 methods are not the same as v5/v6.
  Full protocol iterations (0-49) are now available in the underlying data
  (unlike some earlier partial reruns); this figure still displays x<=41 to
  match the established Fig3 display convention -- unchanged from v5.

Changes vs v4 (fig3_bestq_shade_v4_rl_POSTFIX_withLAGP.py):
  - tick marks added on both axes (seaborn 'white' style hides them by default)
  - right-side twin y-axis, normalized to the max value ever plotted (%) --
    lets the reader see at a glance how close each curve is to the best
    charted point without reading two separate scales
  - GP Matern family: 4-shade blue ramp (unchanged conceptually, cleaned up)
  - GP-LLM Hybrid: BOTH EI and PI variants now shown, gold retired -> two
    teal shades (dark=PI, light=EI)
  - line style now follows one consistent rule (was a mix before):
      solid    = GP-only + GP-LLM Hybrid (statistical-exploitation family)
      dash-dot = ML / Tree ensembles
      dashed   = LLM-only
      dotted   = everything else (Approximate GPs / RL / Random baseline)
  - parenthetical annotations removed from legend labels (seed-fixed, bug-fixed, etc.)
  - MOF_Diversity_EI_GP_Matern52 removed from the main figure (recommended: its
    presence doesn't change the paper's core acquisition-function argument and
    it competes for attention with 15 other lines; full data still reported in SI)
  - all shading (mean +/- SE ribbons) removed; std/mean now reported in the SI
    table instead (all_methods_summary_stats_iter40_iter49_UNIFIED.csv)

Portability note: this is otherwise byte-identical to the original
fig3_bestq_shade_v7.py -- only `data_path` below was repointed at the copy
of figuredata_curve_best_q_UNIFIED.csv shipped alongside this script,
instead of the original hardcoded absolute path. No color/style/selection/
plotting logic changed.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="white", context="paper")
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 0. 路径
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "figuredata_curve_best_q_UNIFIED.csv")

df = pd.read_csv(data_path)

# ==========================================
# 1. 选定方法（15 个：去掉 MOF-specific，加入 Hybrid EI）
# ==========================================
SELECTED_METHODS = [
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5',
    'HYBRID_GP_Matern52_EI_ALT_gpt-5.5',
    'GP_Matern52_PI',
    'GP_Matern52_EI_BASELINE',
    'GP_Matern32_EI',
    'GP_Matern32_PI',
    'CONFRF_EI',
    'BART_like_ExtraTrees_EI',
    'LAGP_ALC_EI',
    'LAGP_NN_EI',
    'RL_Contextual_LinUCB',
    'RL_Contextual_LinUCB_Quantile90',
    'LLM_PURE_gpt-5.5',
    'LLM_PURE_gpt-4o',
    'LLM_PURE_gpt-5-mini',
    'RANDOM_SEARCH',
]

# ==========================================
# 2. 配色 / 线型 / 线宽
# ==========================================
COLOR_MAP = {
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5': '#1F5F5D',   # teal, dark  (was gold)
    'HYBRID_GP_Matern52_EI_ALT_gpt-5.5': '#4FB3AE',   # teal, light (was gold)
    'GP_Matern52_PI':          '#08306B',
    'GP_Matern52_EI_BASELINE': '#2171B5',
    'GP_Matern32_EI':          '#6BAED6',
    'GP_Matern32_PI':          '#C6DBEF',
    'CONFRF_EI':                    '#B85687',
    'BART_like_ExtraTrees_EI':      '#941555',
    'LAGP_ALC_EI':  '#C97A2B',
    'LAGP_NN_EI':   '#7A5230',
    'RL_Contextual_LinUCB':              '#D08A4A',
    'RL_Contextual_LinUCB_Quantile90':   '#A06028',
    'LLM_PURE_gpt-5.5':   '#AAC085',
    'LLM_PURE_gpt-4o':    '#82B14B',
    'LLM_PURE_gpt-5-mini':'#559415',
    'RANDOM_SEARCH': '#B4B2A9',
}

# one consistent rule: solid=GP-family(+Hybrid), dash-dot=ML/Tree, dashed=LLM-only, dotted=everything else
LINESTYLE_MAP = {
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5': '-',
    'HYBRID_GP_Matern52_EI_ALT_gpt-5.5': '-',
    'GP_Matern52_PI':                     '-',
    'GP_Matern52_EI_BASELINE':            '-',
    'GP_Matern32_EI':                     '-',
    'GP_Matern32_PI':                     '-',
    'CONFRF_EI':                          '-.',
    'BART_like_ExtraTrees_EI':            '-.',
    'LAGP_ALC_EI':                         ':',
    'LAGP_NN_EI':                          ':',
    'RL_Contextual_LinUCB':                ':',
    'RL_Contextual_LinUCB_Quantile90':     ':',
    'LLM_PURE_gpt-5.5':                  '--',
    'LLM_PURE_gpt-4o':                   '--',
    'LLM_PURE_gpt-5-mini':               '--',
    'RANDOM_SEARCH':                      ':',
}

LINEWIDTH_MAP = {m: 1.6 for m in SELECTED_METHODS}
LINEWIDTH_MAP['HYBRID_GP_Matern52_PI_ALT_gpt-5.5'] = 2.2
LINEWIDTH_MAP['HYBRID_GP_Matern52_EI_ALT_gpt-5.5'] = 2.2
LINEWIDTH_MAP['RANDOM_SEARCH'] = 1.1

LABEL_MAP = {
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5': 'GP Matérn52 + PI + GPT-5.5',
    'HYBRID_GP_Matern52_EI_ALT_gpt-5.5': 'GP Matérn52 + EI + GPT-5.5',
    'GP_Matern52_PI':                     'GP  Matérn52 + PI',
    'GP_Matern52_EI_BASELINE':            'GP  Matérn52 + EI',
    'GP_Matern32_EI':                     'GP  Matérn32 + EI',
    'GP_Matern32_PI':                     'GP  Matérn32 + PI',
    'CONFRF_EI':                          'Conformal RF + EI',
    'BART_like_ExtraTrees_EI':            'BART-like ExtraTrees + EI',
    'LAGP_ALC_EI':                         'laGP + ALC + EI',
    'LAGP_NN_EI':                          'laGP + NN + EI',
    'RL_Contextual_LinUCB':               'RL  LinUCB',
    'RL_Contextual_LinUCB_Quantile90':    'RL  LinUCB + Quantile90',
    'LLM_PURE_gpt-5.5':                  'GPT-5.5',
    'LLM_PURE_gpt-4o':                   'GPT-4o',
    'LLM_PURE_gpt-5-mini':               'GPT-5-mini',
    'RANDOM_SEARCH':                      'Random Search',
}

GROUP_HEADERS = {
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5': '── GP–LLM Hybrid ──',
    'GP_Matern52_PI':                     '── Gaussian Process (Matérn) ──',
    'CONFRF_EI':                          '── ML / Tree Ensemble ──',
    'LAGP_ALC_EI':                         '── Approximate GPs ──',
    'RL_Contextual_LinUCB':               '── Reinforcement Learning ──',
    'LLM_PURE_gpt-5.5':                  '── LLM-only ──',
    'RANDOM_SEARCH':                      '── Baseline ──',
}

LEGEND_ORDER = [
    'HYBRID_GP_Matern52_PI_ALT_gpt-5.5', 'HYBRID_GP_Matern52_EI_ALT_gpt-5.5',
    'GP_Matern52_PI', 'GP_Matern52_EI_BASELINE', 'GP_Matern32_EI', 'GP_Matern32_PI',
    'CONFRF_EI', 'BART_like_ExtraTrees_EI',
    'LAGP_ALC_EI', 'LAGP_NN_EI',
    'RL_Contextual_LinUCB', 'RL_Contextual_LinUCB_Quantile90',
    'LLM_PURE_gpt-5.5', 'LLM_PURE_gpt-4o', 'LLM_PURE_gpt-5-mini',
    'RANDOM_SEARCH',
]

DRAW_ORDER = [
    'RANDOM_SEARCH',
    'RL_Contextual_LinUCB_Quantile90', 'RL_Contextual_LinUCB',
    'LAGP_NN_EI', 'LAGP_ALC_EI',
    'GP_Matern32_PI', 'GP_Matern32_EI',
    'GP_Matern52_EI_BASELINE', 'GP_Matern52_PI',
    'BART_like_ExtraTrees_EI', 'CONFRF_EI',
    'LLM_PURE_gpt-5-mini', 'LLM_PURE_gpt-4o', 'LLM_PURE_gpt-5.5',
    'HYBRID_GP_Matern52_EI_ALT_gpt-5.5', 'HYBRID_GP_Matern52_PI_ALT_gpt-5.5',
]

# ==========================================
# 3. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(13, 7.5))

df_sel = df[(df['method'].isin(SELECTED_METHODS)) & (df['x'] <= 41)].copy()

for method in DRAW_ORDER:
    sub = df_sel[df_sel['method'] == method].sort_values('x')
    if sub.empty:
        continue
    x, mean = sub['x'].values, sub['mean'].values
    ax.plot(x, mean, color=COLOR_MAP[method], linestyle=LINESTYLE_MAP[method],
            linewidth=LINEWIDTH_MAP[method], label=LABEL_MAP.get(method, method), zorder=3)

# ── 坐标轴 ───────────────────────────────────────────────────────
ax.set_xlabel('Iteration #', fontsize=16)
ax.set_ylabel('Best $q$ so Far', fontsize=16)
ax.set_xlim(0, 41)
ax.set_ylim(0, 200000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10000))
ax.tick_params(axis='both', which='major', labelsize=14, direction='out', length=6, width=1.0)
ax.tick_params(axis='both', which='minor', direction='out', length=3, width=0.8)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# ── 版本戳 ───────────────────────────────────────────────────────
fig.text(0.005, 0.005, 'fig3_bestq_shade_v7  |  unified candidate-sampling dataset',
          fontsize=7.5, color='#999999', ha='left', va='bottom')

# ── 右轴：归一化到本图已画出的最大值 ────────────────────────────
max_plotted = df_sel['mean'].max()
ax_twin = ax.twinx()
ax_twin.set_ylim(0, 200000 / max_plotted * 100)
ax_twin.set_ylabel('% of highest plotted $q$', fontsize=14)
ax_twin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax_twin.tick_params(axis='y', which='major', labelsize=12, direction='out', length=6, width=1.0)
ax_twin.spines['top'].set_visible(False)

# ── 图例 ─────────────────────────────────────────────────────────
legend_handles = []
for method in LEGEND_ORDER:
    if method in GROUP_HEADERS:
        legend_handles.append(Line2D([0], [0], color='none', linewidth=0, label=GROUP_HEADERS[method]))
    legend_handles.append(Line2D([0], [0], color=COLOR_MAP[method], linestyle=LINESTYLE_MAP[method],
                                  linewidth=LINEWIDTH_MAP[method], label=LABEL_MAP[method]))

leg = ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.09, 0.86),
                 fontsize=11, frameon=False, handlelength=2.2, handleheight=1.2,
                 borderpad=0.5, labelspacing=0.40)
for text in leg.get_texts():
    if text.get_text().startswith('──'):
        text.set_fontsize(10); text.set_color('#666666'); text.set_fontstyle('italic')

fig.subplots_adjust(right=0.63)

for fmt in ['png', 'svg']:
    out = os.path.join(script_dir, f'fig3_bestq_shade_v7.{fmt}')
    kwargs = dict(bbox_inches='tight')
    if fmt == 'png':
        kwargs['dpi'] = 300
    fig.savefig(out, format=fmt, **kwargs)
    print(f'Saved: {out}')
