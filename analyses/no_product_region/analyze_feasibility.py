"""
analyze_feasibility.py -- CP7 tables and figures for the feasibility-aware policies (protocol Sec. 4 and 6).

    python scripts/analyze_feasibility.py --tag main

Reads results/feasibility_paired/<tag>/{runs,iterations}.csv; writes to results/feasibility_paired/:
  summary_primary.csv, summary_hits.csv, summary_secondary.csv, summary_exploratory.csv, summary_stall.csv,
  diagnostics_m1_summary.csv, dataflow_check.csv, oracle_sha256_after.txt, runs.csv, iterations.csv,
  fig_best_so_far.pdf, fig_n_first.pdf, fig_m1_diagnostics.pdf
Thresholds and q_max are used for SCORING only (never by a policy).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results"

Q1, Q01 = 42463.0, 100288.0          # SI Table S9
QMAX = 247107.0
K, M = 6120, 6101172
NRUN = 20
PRIMARY = ["B0", "M1", "M3"]
EXPLOR = ["M1_avoid", "M1_gamma0.5", "M1_gamma2"]
COL = {"B0": "#4d4d4d", "M1": "#D55E00", "M3": "#0072B2",
       "M1_avoid": "#CC79A7", "M1_gamma0.5": "#E69F00", "M1_gamma2": "#009E73"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 8.5})


def per_run(runs):
    out = []
    for (m, r), g in runs.groupby(["method", "run"]):
        g = g.reset_index(drop=True)                      # query order = file order (starts, then batches)
        pos = np.flatnonzero(g.q.values > 0)
        n_first = int(pos[0]) + 1 if len(pos) else 121
        d = dict(method=m, run=r, N_first=n_first)
        d["share_1_5"] = float((g[(g.iteration >= 1) & (g.iteration <= 5)].q > 0).mean())
        bs = g.groupby("iteration").q.max().cummax()
        for t in (0, 3, 13, 39):
            d[f"best_{t}"] = float(bs.loc[t])
        d["H1_3"] = int(d["best_3"] >= Q1)
        d["H01_13"] = int(d["best_13"] >= Q01)
        d["H01_39"] = int(d["best_39"] >= Q01)
        for t in range(0, 40):
            d[f"bs{t}"] = float(bs.loc[t])
        out.append(d)
    return pd.DataFrame(out)


def iqr(x):
    return np.percentile(x, 25), np.percentile(x, 75)


def wilcoxon_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox", method="auto").pvalue)


def primary_table(pr):
    outs = [("N_first", "N_first (experiments to first product)"),
            ("share_1_5", "share of queries with q>0, iterations 1-5"),
            ("best_3", "best-so-far q at iteration 3 (12 experiments)"),
            ("best_13", "best-so-far q at iteration 13 (42 experiments)")]
    rows = []
    piv = {k: pr.pivot(index="run", columns="method", values=k) for k, _ in outs}
    for m in ["M1", "M3"]:
        for k, lab in outs:
            a, b = piv[k]["B0"].values, piv[k][m].values
            lo0, hi0 = iqr(a); lo1, hi1 = iqr(b)
            rows.append(dict(comparison=f"{m} vs B0", outcome=lab, median_B0=np.median(a), q25_B0=lo0, q75_B0=hi0,
                             **{f"median_{m}": np.median(b), f"q25_{m}": lo1, f"q75_{m}": hi1},
                             median_paired_diff=np.median(b - a), wilcoxon_p=wilcoxon_p(b, a)))
    t = pd.DataFrame(rows)
    t["holm_p"] = multipletests(t.wilcoxon_p.values, method="holm")[1]
    return t


def hits_table(pr):
    cols = [("H1_3", "H_1%(3)"), ("H01_13", "H_0.1%(13)"), ("H01_39", "H_0.1%(39)")]
    rows = []
    for m in PRIMARY + EXPLOR:
        g = pr[pr.method == m]
        rows.append(dict(method=m, status="primary" if m in PRIMARY else "exploratory",
                         **{lab: f"{int(g[c].sum())}/{len(g)}" for c, lab in cols}))
    t = pd.DataFrame(rows)
    mc = []
    for m in ["M1", "M3"]:
        for c, lab in cols:
            a = pr[pr.method == "B0"].sort_values("run")[c].values
            b = pr[pr.method == m].sort_values("run")[c].values
            n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
            p = mcnemar([[int(((a == 1) & (b == 1)).sum()), n10], [n01, int(((a == 0) & (b == 0)).sum())]], exact=True).pvalue
            mc.append(dict(comparison=f"{m} vs B0", outcome=lab, only_B0_hit=n10, only_method_hit=n01,
                           mcnemar_exact_p=p, note="descriptive, not Holm-corrected"))
    return t, pd.DataFrame(mc)


def secondary_table(pr):
    prand = lambda n: 1 - (1 - K / M) ** n
    n50rand = np.log(0.5) / np.log(1 - K / M)
    rows = []
    for m in PRIMARY + EXPLOR:
        g = pr[pr.method == m]
        h01 = [(g[f"bs{t}"] >= Q01).mean() for t in range(40)]
        n50 = next((3 + 3 * t for t, h in enumerate(h01) if h >= 0.5), np.nan)
        reg = np.mean([(QMAX - g[f"bs{t}"]) / QMAX for t in range(40)], axis=1)      # regret curve
        rows.append(dict(method=m, status="primary" if m in PRIMARY else "exploratory",
                         best_39_median=g.best_39.median(), best_39_q25=iqr(g.best_39)[0], best_39_q75=iqr(g.best_39)[1],
                         norm_regret_39_mean=float(((QMAX - g.best_39) / QMAX).mean()),
                         EF_13=h01[13] / prand(42), EF_39=h01[39] / prand(120),
                         N50=n50, AF=(n50rand / n50) if n50 == n50 else np.nan))
    return pd.DataFrame(rows)


def exploratory_table(pr):
    rows = []
    for m in PRIMARY + EXPLOR:
        g = pr[pr.method == m]
        d = dict(method=m, status="primary" if m in PRIMARY else "exploratory (descriptive only)")
        for k in ["N_first", "share_1_5", "best_3", "best_13"]:
            lo, hi = iqr(g[k]); d[f"{k}_median"] = g[k].median(); d[f"{k}_q25"] = lo; d[f"{k}_q75"] = hi
        rows.append(d)
    return pd.DataFrame(rows)


def stall_table(runs):
    """stall = >= 10 consecutive iterations (1-39) in which no new crystalline product was found"""
    rows = []
    for (m, r), g in runs.groupby(["method", "run"]):
        newpos = g[g.iteration >= 1].groupby("iteration").q.apply(lambda s: int((s > 0).sum())).reindex(range(1, 40), fill_value=0)
        npos_after = g.groupby("iteration").q.apply(lambda s: int((s > 0).sum())).cumsum()
        longest, cur, longest_post, cur_post = 0, 0, 0, 0
        for it in range(1, 40):
            if newpos.loc[it] == 0:
                cur += 1
                longest = max(longest, cur)
                if npos_after.loc[it] > 0:
                    cur_post += 1
                    longest_post = max(longest_post, cur_post)
                else:
                    cur_post = 0
            else:
                cur = cur_post = 0
        rows.append(dict(method=m, run=r, longest_no_new_product=longest, longest_after_first_product=longest_post,
                         n_pos_final=int(npos_after.loc[39]),
                         stalled_any=longest >= 10, stalled_after_first_product=longest_post >= 10))
    d = pd.DataFrame(rows)
    s = d.groupby("method").agg(n_runs=("run", "count"), stalled_any=("stalled_any", "sum"),
                                stalled_after_first_product=("stalled_after_first_product", "sum"),
                                median_final_n_pos=("n_pos_final", "median")).reset_index()
    s["status"] = "exploratory, descriptive"
    return s, d


def dataflow_table(it):
    d = it.copy()
    d["n_neg_ok"] = True
    rows = []
    for m, g in d.groupby("method"):
        fitted = g[g.gpc_fitted]
        rows.append(dict(method=m, iterations=len(g), assertions_passed=bool(g.dataflow_ok.all()),
                         gpc_fits=len(fitted),
                         gpc_train_size_equals_queries=int((fitted.n_train == fitted.n_queries).sum()),
                         gpc_fit_failures=int(g.gpc_failed.sum()),
                         gpc_failure_rate_of_expected=float(g.gpc_failed.sum() / max(1, (g.n_pos >= 1).sum()))
                         if m.startswith("M1") else np.nan))
    return pd.DataFrame(rows)


def diag_summary(it, runs):
    m1 = it[it.method.str.startswith("M1") & it.gpc_fitted]
    rows = []
    for m, g in m1.groupby("method"):
        b = g.gpc_bounds.fillna("")
        d = dict(method=m, gpc_fits=len(g))
        for name in ["amp", "ls1", "ls2", "ls3", "ls4", "ls5"]:
            d[f"frac_{name}_at_bound"] = float(b.apply(lambda s: name in s.split(";")).mean())
        d["frac_any_at_bound"] = float((b != "").mean())
        for c in ["gpc_amp"] + [f"gpc_ls{i}" for i in range(1, 6)]:
            d[f"median_{c}"] = float(g[c].median())
        n = g.dropna(subset=["nb_acq"])
        d["iters_with_neighbourhood"] = len(n)
        d["median_nb_p"] = float(n.nb_p.median()) if len(n) else np.nan
        d["median_nb_ei"] = float(n.nb_ei.median()) if len(n) else np.nan
        d["median_nb_acq"] = float(n.nb_acq.median()) if len(n) else np.nan
        rq = runs[(runs.method == m) & runs.p_chosen.notna()]
        d["median_chosen_p"] = float(rq.p_chosen.median())
        d["median_chosen_ei"] = float(rq.ei_pos_chosen.median())
        d["median_chosen_acq"] = float(rq.acquisition_chosen.median())
        d["median_chosen_cheb"] = float(rq.cheb_to_cryst.median())
        d["frac_chosen_within_cheb2"] = float((rq.cheb_to_cryst <= 2).mean())
        # per iteration: best neighbourhood acquisition vs best acquisition among the chosen 3
        ch = rq.groupby(["run", "iteration"]).acquisition_chosen.max().rename("chosen_best")
        n2 = n.set_index(["run", "iteration"]).nb_acq.to_frame().join(ch, how="inner")
        d["frac_iters_neighbour_acq_above_chosen"] = float((n2.nb_acq > n2.chosen_best).mean()) if len(n2) else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def ci95(a):
    a = np.asarray(a)
    return a.mean(0), stats.t.ppf(0.975, a.shape[0] - 1) * a.std(0, ddof=1) / np.sqrt(a.shape[0])


def curves(pr, m):
    g = pr[pr.method == m].sort_values("run")
    return g[[f"bs{t}" for t in range(40)]].values


def fig_best(pr):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    it = np.arange(40)
    for ax, ms, title in [(axs[0], PRIMARY, "Policies of the paper"),
                          (axs[1], ["B0"] + EXPLOR, "Exploratory variants (B0 for reference)")]:
        for m in ms:
            mu, h = ci95(curves(pr, m))
            ax.plot(it, mu, color=COL[m], lw=1.6 if m in PRIMARY else 1.3, label=m.replace("_", "-"))
            ax.fill_between(it, mu - h, mu + h, color=COL[m], alpha=0.15, lw=0)
        ax.axhline(Q1, color="k", ls="--", lw=0.6); ax.axhline(Q01, color="k", ls=":", lw=0.8)
        ax.set_xlabel("iteration (experiments = 3 + 3 x iteration)"); ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ins = ax.inset_axes([0.6, 0.12, 0.36, 0.34])
        for m in ms:
            mu, h = ci95(curves(pr, m))
            ins.plot(it[:6], mu[:6], color=COL[m], lw=1.1, marker="o", ms=2)
            ins.fill_between(it[:6], (mu - h)[:6], (mu + h)[:6], color=COL[m], alpha=0.15, lw=0)
        ins.set_title("iterations 0-5", fontsize=6.5); ins.tick_params(labelsize=6)
    axs[0].set_ylabel("mean best-so-far q (95% CI, n = 20)")
    fig.tight_layout(); fig.savefig(OUT / "fig_best_so_far.pdf"); plt.close(fig)


def fig_nfirst(pr):
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.0), gridspec_kw=dict(width_ratios=[3, 2]))
    for ax, ms, title in [(axs[0], PRIMARY, "Policies of the paper"), (axs[1], ["B0", "M1_avoid"], "Exploratory")]:
        P = pr.pivot(index="run", columns="method", values="N_first")[ms]
        rng = np.random.default_rng(0)
        for r, row in P.iterrows():
            ax.plot(range(len(ms)), row.values, color="0.75", lw=0.6, zorder=1)
        for i, m in enumerate(ms):
            ax.scatter(np.full(len(P), i) + rng.uniform(-0.04, 0.04, len(P)), P[m], s=14, color=COL[m], zorder=2)
            ax.hlines(P[m].median(), i - 0.2, i + 0.2, color="k", lw=1.5, zorder=3)
        ax.set_yscale("log"); ax.set_xticks(range(len(ms))); ax.set_xticklabels([m.replace("_", "-") for m in ms])
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("N_first (experiments to first q > 0; 121 = never)")
    fig.tight_layout(); fig.savefig(OUT / "fig_n_first.pdf"); plt.close(fig)


def fig_diag(runs, it, stall_runs):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6))
    # (a) n_pos trajectories of M1
    ax = axs[0, 0]
    st = stall_runs[stall_runs.method == "M1"].set_index("run").stalled_after_first_product
    g = runs[runs.method == "M1"]
    for r, gg in g.groupby("run"):
        npos = gg.groupby("iteration").q.apply(lambda s: int((s > 0).sum())).cumsum()
        ax.plot(npos.index, npos.values, color=COL["M1"] if st.loc[r] else "0.7", lw=1.2 if st.loc[r] else 0.7)
    ax.set_xlabel("iteration"); ax.set_ylabel("crystalline products found (M1)")
    ax.set_title("(a) M1 runs; orange = stalled (>= 10 iterations without a new product)", loc="left", fontsize=7)
    # (b) chosen point p vs Chebyshev distance
    ax = axs[0, 1]
    q = runs[(runs.method == "M1") & runs.p_chosen.notna() & runs.cheb_to_cryst.notna()]
    ax.scatter(q.cheb_to_cryst + np.random.default_rng(0).uniform(-0.3, 0.3, len(q)), q.p_chosen,
               s=4, c=np.where(q.q > 0, COL["M3"], "0.6"), alpha=0.5, lw=0)
    ax.set_xlabel("Chebyshev dist. to nearest product (grid steps)"); ax.set_ylabel("p at chosen point")
    ax.set_title("(b) M1 chosen points (blue: q > 0)", loc="left", fontsize=7)
    # (c) chosen best acquisition vs best neighbourhood acquisition
    ax = axs[1, 0]
    n = it[(it.method == "M1") & it.nb_acq.notna()].set_index(["run", "iteration"]).nb_acq
    ch = g[g.p_chosen.notna()].groupby(["run", "iteration"]).acquisition_chosen.max()
    j = pd.concat([n, ch], axis=1, join="inner").dropna()
    ax.scatter(j.acquisition_chosen, j.nb_acq, s=4, color=COL["M1"], alpha=0.5, lw=0)
    lim = [1e-6, max(j.max().max(), 1)]
    ax.plot(lim, lim, "k-", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("best acquisition among the 3 chosen"); ax.set_ylabel("best acquisition within Chebyshev <= 2\nof an observed product")
    ax.set_title("(c) above the line: a near-product point scored higher", loc="left", fontsize=7)
    # (d) length scales of the GPC
    ax = axs[1, 1]
    f = it[(it.method == "M1") & it.gpc_fitted]
    data = [f[f"gpc_ls{i}"].values for i in range(1, 6)]
    ax.boxplot(data, showfliers=False, medianprops=dict(color=COL["M1"]))
    ax.axhline(10, color="k", ls=":", lw=0.7); ax.axhline(0.01, color="k", ls=":", lw=0.7)
    ax.set_yscale("log"); ax.set_xticklabels(["metal", "modulator", "solvent", "time", "temp."], fontsize=7)
    ax.set_ylabel("fitted GPC length scale (scaled)"); ax.set_title("(d) M1 GPC length scales (dotted: bounds)", loc="left", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_m1_diagnostics.pdf"); plt.close(fig)


def main(tag):
    import feasibility as F
    d = OUT / tag
    runs = pd.read_csv(d / "runs.csv")
    # iterations.csv was appended run by run; runs without a neighbourhood diagnostic have 7 fewer columns (header comes
    # from the first-written run) -> read with the full explicit column list, missing fields become NaN
    hdr = pd.read_csv(d / "iterations.csv", nrows=0).columns.tolist()
    nbc = ["nb_n", "nb_acq", "nb_p", "nb_mu", "nb_sd", "nb_ei", "nb_cheb"]
    it = pd.read_csv(d / "iterations.csv", skiprows=1, names=hdr + [c for c in nbc if c not in hdr])
    runs.to_csv(OUT / "runs.csv", index=False); it.to_csv(OUT / "iterations.csv", index=False)
    pr = per_run(runs)
    assert all((pr.groupby("method").run.count() == NRUN)), pr.groupby("method").run.count()
    pr.drop(columns=[c for c in pr.columns if c.startswith("bs")]).to_csv(OUT / "per_run_outcomes.csv", index=False)
    primary_table(pr).to_csv(OUT / "summary_primary.csv", index=False)
    ht, mc = hits_table(pr)
    ht.to_csv(OUT / "summary_hits.csv", index=False)
    mc.to_csv(OUT / "summary_hits_mcnemar.csv", index=False)
    secondary_table(pr).to_csv(OUT / "summary_secondary.csv", index=False)
    exploratory_table(pr).to_csv(OUT / "summary_exploratory.csv", index=False)
    s, sr = stall_table(runs)
    s.to_csv(OUT / "summary_stall.csv", index=False); sr.to_csv(OUT / "stall_per_run.csv", index=False)
    dataflow_table(it).to_csv(OUT / "dataflow_check.csv", index=False)
    diag_summary(it, runs).to_csv(OUT / "diagnostics_m1_summary.csv", index=False)
    h = F.oracle_sha256()
    (OUT / "oracle_sha256_after.txt").write_text("\n".join(f"{v} *{k}" for k, v in h.items()) + "\n")
    before = (OUT / "oracle_sha256_before.txt").read_text().split()
    print("oracle hash unchanged:", before[0::2] == list(h.values()))
    fig_best(pr); fig_nfirst(pr); fig_diag(runs, it, sr)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for f in ["summary_primary", "summary_hits", "summary_hits_mcnemar", "summary_secondary", "summary_stall",
              "dataflow_check", "diagnostics_m1_summary"]:
        print("\n==", f); print(pd.read_csv(OUT / f"{f}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    main(ap.parse_args().tag)
