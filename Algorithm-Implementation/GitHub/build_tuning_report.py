# ============================================================
# Combine the per-instance-size REMOVE_RATIO tuning results
# (paper_figures/tuning/E_20_5_1_mincost, E_50_5_1_mincost,
# E_75_5_1_mincost -- each produced by run_parameter_tuning.py) into:
#   - one full statistical table (CSV + printed) with every column a
#     reviewer would expect: n, mean, std, 95% CI, effect size, p-values
#   - the standard plots for this kind of study: a sensitivity plot
#     (mean +/- 95% CI vs REMOVE_RATIO, one panel per instance size),
#     grouped box plots of the objective distribution, and a served-rate
#     companion plot (since objective alone conflates cost and coverage)
#
# Usage:
#   python build_tuning_report.py
# ============================================================

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

TUNING_DIR  = os.path.join(HERE, "paper_figures", "tuning")
FIGURES_DIR = os.path.join(HERE, "paper_figures", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RUNS = [
    dict(tag="E_20_5_1_mincost", n_patients=20, iter_budget=150),
    dict(tag="E_50_5_1_mincost", n_patients=50, iter_budget=100),
    dict(tag="E_75_5_1_mincost", n_patients=75, iter_budget=25),
]

# ---- same academic plot style as generate_paper_figures.py ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})
COLOR_BAR  = "#2E5B8C"
COLOR_BEST = "#4C8577"
COLOR_SERV = "#B0413E"

# ============================================================
# BUILD THE COMBINED STATISTICAL TABLE
# ============================================================

rows = []
raw_all = []

for run in RUNS:
    out_dir = os.path.join(TUNING_DIR, run["tag"])
    summary = pd.read_csv(os.path.join(out_dir, "replications_summary.csv"))
    sig     = pd.read_csv(os.path.join(out_dir, "significance_tests.csv"))
    raw     = pd.read_csv(os.path.join(out_dir, "replications_raw.csv"))

    obj = summary[summary["metric"] == "objective"].copy()
    # Collapse to the distinct remove_ratio values actually present for
    # this run (T/C were held fixed to a single value at 50/75 patients,
    # and only the T=500,C=0.9995 row is used for the 20-patient run so
    # the comparison across sizes is apples-to-apples).
    obj["remove_ratio"] = obj["config"].str.extract(r"rr([\d.]+)_").astype(float)
    obj = obj[obj["config"].str.contains("_t500_c0.9995")]

    raw["remove_ratio"] = raw["config"].str.extract(r"rr([\d.]+)_").astype(float)
    raw = raw[raw["config"].str.contains("_t500_c0.9995")]
    raw["n_patients"]  = run["n_patients"]
    raw["iter_budget"] = run["iter_budget"]
    raw_all.append(raw)

    served_stats = raw.groupby("remove_ratio")["served"].agg(["mean", "std"])

    baseline_cfg = sig["baseline"].iloc[0] if len(sig) else None

    for _, r in obj.iterrows():
        rr = r["remove_ratio"]
        is_baseline = np.isclose(rr, 0.10)
        if is_baseline:
            sig_row = None
        else:
            match = sig[sig["comparison"] == r["config"]]
            sig_row = match.iloc[0] if len(match) else None

        rows.append(dict(
            n_patients=run["n_patients"],
            iter_budget=run["iter_budget"],
            remove_ratio=rr,
            n_seeds=int(r["n"]),
            mean_objective=r["mean"],
            std_objective=r["std"],
            median_objective=r["median"],
            ci95_low=r["ci95_low"],
            ci95_high=r["ci95_high"],
            min_objective=r["min"],
            max_objective=r["max"],
            mean_served=served_stats.loc[rr, "mean"],
            std_served=served_stats.loc[rr, "std"],
            served_rate_pct=100 * served_stats.loc[rr, "mean"] / run["n_patients"],
            cohens_dz=None if sig_row is None else sig_row["cohens_dz"],
            paired_t_pvalue=None if sig_row is None else sig_row["paired_t_pvalue"],
            wilcoxon_pvalue=None if sig_row is None else sig_row["wilcoxon_pvalue"],
            significant_at_0_05=None if sig_row is None else sig_row["significant_at_0_05"],
            is_baseline=is_baseline,
        ))

table = pd.DataFrame(rows).sort_values(["n_patients", "remove_ratio"]).reset_index(drop=True)
raw_all = pd.concat(raw_all, ignore_index=True)

table_csv = os.path.join(FIGURES_DIR, "table_remove_ratio_tuning.csv")
table.to_csv(table_csv, index=False)

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("=" * 100)
print("COMBINED REMOVE_RATIO TUNING TABLE (all instance sizes, T=500, C=0.9995)")
print("=" * 100)
print(table.to_string(index=False))
print(f"\nFull table written to: {table_csv}")

# ============================================================
# FIGURE -- SENSITIVITY: mean objective +/- 95% CI vs REMOVE_RATIO,
# one panel per instance size (the standard OFAT-style plot)
# ============================================================

sizes = [r["n_patients"] for r in RUNS]
fig, axes = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 4.2), sharey=False)

for ax, n in zip(axes, sizes):
    sub = table[table["n_patients"] == n].sort_values("remove_ratio")
    best_idx = sub["mean_objective"].idxmin()
    colors = [COLOR_BEST if i == best_idx else COLOR_BAR for i in sub.index]

    yerr_low  = sub["mean_objective"] - sub["ci95_low"]
    yerr_high = sub["ci95_high"] - sub["mean_objective"]
    ax.bar(sub["remove_ratio"].astype(str), sub["mean_objective"], color=colors, width=0.6)
    ax.errorbar(sub["remove_ratio"].astype(str), sub["mean_objective"],
                yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4, lw=1.2)

    run_meta = next(r for r in RUNS if r["n_patients"] == n)
    ax.set_title(f"{n} patients\n({run_meta['iter_budget']} iterations, n=3 seeds)", fontsize=10)
    ax.set_xlabel("REMOVE_RATIO")
    ax.set_ylabel("Mean objective (95% CI)")

fig.suptitle("Destroy-ratio sensitivity across instance sizes (T=500, cooling=0.9995)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(FIGURES_DIR, "fig7_remove_ratio_sensitivity.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig7_remove_ratio_sensitivity.pdf"))
plt.close(fig)

# ============================================================
# FIGURE -- GROUPED BOX PLOT of the objective distribution
# (n=3 seeds per box -- shown as individual points, not just quartiles,
# since 3 points can't usefully support whiskers)
# ============================================================

fig, axes = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 4.2), sharey=False)

for ax, n in zip(axes, sizes):
    sub_raw = raw_all[raw_all["n_patients"] == n]
    rrs = sorted(sub_raw["remove_ratio"].unique())
    data = [sub_raw[sub_raw["remove_ratio"] == rr]["objective"].values for rr in rrs]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BAR)
        patch.set_alpha(0.5)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i), vals, color="black", zorder=3, s=18)

    ax.set_xticks(range(1, len(rrs) + 1))
    ax.set_xticklabels([f"{rr:g}" for rr in rrs])
    run_meta = next(r for r in RUNS if r["n_patients"] == n)
    ax.set_title(f"{n} patients ({run_meta['iter_budget']} iter.)", fontsize=10)
    ax.set_xlabel("REMOVE_RATIO")
    ax.set_ylabel("Objective")

fig.suptitle("Objective distribution by destroy ratio (individual seeds shown, n=3/box)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGURES_DIR, "fig8_remove_ratio_boxplot.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig8_remove_ratio_boxplot.pdf"))
plt.close(fig)

# ============================================================
# FIGURE -- SERVED-RATE companion plot
# (objective differences can be driven by unserved-patient penalty
# rather than routing efficiency, so report coverage alongside cost)
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.25
x = np.arange(len(sizes))

rr_values = sorted(table["remove_ratio"].unique())
for i, rr in enumerate(rr_values):
    vals = [table[(table["n_patients"] == n) & (np.isclose(table["remove_ratio"], rr))]["served_rate_pct"].iloc[0]
            for n in sizes]
    ax.bar(x + (i - 1) * width, vals, width, label=f"REMOVE_RATIO={rr:g}")

ax.set_xticks(x)
ax.set_xticklabels([f"{n} patients" for n in sizes])
ax.set_ylabel("Patients served (%)")
ax.set_title("Service coverage by destroy ratio and instance size")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig9_served_rate.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig9_served_rate.pdf"))
plt.close(fig)

print(f"\nFigures written to: {FIGURES_DIR}")
print("  fig7_remove_ratio_sensitivity.png/pdf")
print("  fig8_remove_ratio_boxplot.png/pdf")
print("  fig9_served_rate.png/pdf")
