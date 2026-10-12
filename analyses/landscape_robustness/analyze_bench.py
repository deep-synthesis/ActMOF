"""
analyze_bench.py -- strategy ranking on every variant landscape.
Inputs: bench_results/worker_*.jsonl, landscape_metrics.csv, variants.csv
Outputs: strategy_by_variant.csv (per variant x method), ranking_summary.csv (per variant)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kendalltau, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
METHODS = ["GP_Matern52_PI", "GP_Matern52_EI_BASELINE", "GP_Matern32_PI", "GP_Matern32_EI",
           "CONFRF_EI", "BART_like_ExtraTrees_EI", "RANDOM_SEARCH"]
LEAD = METHODS[:6]
IT_FINAL, IT_EARLY = 39, 13


def load_results():
    rows = []
    for fp in sorted((HERE / "bench_results").glob("*.jsonl")):
        for l in fp.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                rows.append(dict(variant=d["variant"], method=d["method"], run=d["run"],
                                 best39=d["best_q"][IT_FINAL], best13=d["best_q"][IT_EARLY], n_err=d["n_err"]))
    df = pd.DataFrame(rows).drop_duplicates(["variant", "method", "run"])
    rep = (df.variant == "V0") & (df.run >= 20)
    df.loc[rep, "variant"] = "V0rep"
    return df


def main():
    df = load_results()
    lm = pd.read_csv(HERE / "landscape_metrics.csv").set_index("vid")
    var = pd.read_csv(HERE / "variants.csv")
    vids = ["V0", "V0rep"] + [v for v in var.vid if v != "V0"]
    per, summ = [], []
    ref_rank = None
    for v in vids:
        g = df[df.variant == v]
        if g.groupby("method").size().reindex(METHODS).fillna(0).min() < 20:
            continue
        mv = "V0" if v == "V0rep" else v
        thr01, qmax = lm.loc[mv, "thr_top01"], float(np.load(HERE / "maps" / f"{mv}.npy").max())
        st = g.groupby("method").agg(mean=("best39", "mean"), median=("best39", "median"),
                                     mean13=("best13", "mean")).reindex(METHODS)
        st["H01"] = g.assign(h=g.best39 >= thr01).groupby("method").h.mean().reindex(METHODS)
        st["norm_mean"] = st["mean"] / qmax
        st["rank"] = st["mean"].rank(ascending=False, method="min").astype(int)
        if v == "V0":
            ref_rank = st["rank"].copy()
        dunn = sp.posthoc_dunn(g, val_col="best39", group_col="method", p_adjust="holm")
        best_lead = st.loc[LEAD, "mean"].idxmax()
        st["p_vs_random"] = [dunn.loc[m, "RANDOM_SEARCH"] if m != "RANDOM_SEARCH" else np.nan for m in METHODS]
        st["p_vs_best"] = [dunn.loc[m, best_lead] if m != best_lead else 1.0 for m in METHODS]
        st["leading"] = [(m in LEAD) and (st.loc[m, "p_vs_random"] < 0.05) and (st.loc[m, "p_vs_best"] >= 0.05)
                         for m in METHODS]
        # secondary criterion: direct Mann-Whitney tests (one-sided vs random, Holm over 6;
        # two-sided vs the best leading strategy, Holm over 5)
        x = {m: g[g.method == m].best39.to_numpy() for m in METHODS}
        p_r = multipletests([mannwhitneyu(x[m], x["RANDOM_SEARCH"], alternative="greater").pvalue for m in LEAD], method="holm")[1]
        others = [m for m in LEAD if m != best_lead]
        p_b = dict(zip(others, multipletests([mannwhitneyu(x[m], x[best_lead]).pvalue for m in others], method="holm")[1]))
        st["mw_p_vs_random"] = list(p_r) + [np.nan]
        st["mw_p_vs_best"] = [p_b.get(m, 1.0) if m in LEAD else np.nan for m in METHODS]
        st["leading_mw"] = [(m in LEAD) and (st.loc[m, "mw_p_vs_random"] < 0.05) and (st.loc[m, "mw_p_vs_best"] >= 0.05)
                            for m in METHODS]
        kw_p = kruskal(*[g[g.method == m].best39 for m in LEAD]).pvalue
        tau7 = kendalltau(ref_rank[METHODS], st["rank"][METHODS]).statistic
        tau6 = kendalltau(ref_rank[LEAD], st["rank"][LEAD]).statistic
        for m in METHODS:
            per.append(dict(variant=v, method=m, **st.loc[m].to_dict()))
        summ.append(dict(variant=v, tau_all7=tau7, tau_lead6=tau6, kw_p_lead6=kw_p,
                         n_leading=int(st["leading"].sum()), leading_intact=bool(st.loc[LEAD, "leading"].all()),
                         n_leading_mw=int(st["leading_mw"].sum()), leading_intact_mw=bool(st.loc[LEAD, "leading_mw"].all()),
                         random_last=bool(st.loc["RANDOM_SEARCH", "rank"] == 7),
                         lead_H01_min=st.loc[LEAD, "H01"].min(), lead_H01_max=st.loc[LEAD, "H01"].max(),
                         random_H01=st.loc["RANDOM_SEARCH", "H01"],
                         lead_mean_over_random_min=(st.loc[LEAD, "mean"] / st.loc["RANDOM_SEARCH", "mean"]).min(),
                         n_err=int(g.n_err.sum())))
    pd.DataFrame(per).to_csv(HERE / "strategy_by_variant.csv", index=False)
    s = pd.DataFrame(summ); s.to_csv(HERE / "ranking_summary.csv", index=False)
    pd.set_option("display.width", 200); print(s.round(3).to_string())


if __name__ == "__main__":
    main()
