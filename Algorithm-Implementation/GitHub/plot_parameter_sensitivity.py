# ============================================================
# FIGURE -- LNS PARAMETER SENSITIVITY (one-factor-at-a-time)
#
# Re-plots the sensitivity figure from a completed run_parameter_tuning.py
# sweep (replications_summary.csv + chosen_parameters.json), in the same
# academic style as generate_paper_figures.py's other figures (consistent
# rcParams, PNG + PDF export into paper_figures/figures/) -- so the paper
# figure can be regenerated/restyled without re-running the (expensive)
# parameter grid itself.
#
# Usage:
#   python plot_parameter_sensitivity.py --tuning_dir paper_figures/tuning/E_20_5_1
# ============================================================

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

_parser = argparse.ArgumentParser()
_parser.add_argument("--tuning_dir", type=str, required=True,
                      help="Output directory from run_parameter_tuning.py -- must contain "
                           "replications_summary.csv and chosen_parameters.json.")
_parser.add_argument("--figures_dir", type=str, default=None,
                      help="Where to save the figure. Defaults to <this folder>/paper_figures/"
                           "figures, alongside generate_paper_figures.py's other figures.")
_args, _ = _parser.parse_known_args()

figures_dir = _args.figures_dir or os.path.join(HERE, "paper_figures", "figures")
os.makedirs(figures_dir, exist_ok=True)

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
COLOR      = "#2E5B8C"
BEST_COLOR = "#B0413E"

# ============================================================
# LOAD THE ALREADY-COMPUTED GRID RESULTS
# ============================================================

summary_csv = os.path.join(_args.tuning_dir, "replications_summary.csv")
chosen_path = os.path.join(_args.tuning_dir, "chosen_parameters.json")

if not os.path.exists(summary_csv):
    raise FileNotFoundError(f"{summary_csv!r} not found -- run run_parameter_tuning.py first.")
if not os.path.exists(chosen_path):
    raise FileNotFoundError(f"{chosen_path!r} not found -- run run_parameter_tuning.py first.")

agg_df = pd.read_csv(summary_csv)
with open(chosen_path) as f:
    chosen = json.load(f)

best_rr, best_t, best_c = chosen["remove_ratio"], chosen["temperature"], chosen["cooling"]

# Recover (remove_ratio, temperature, cooling) from each config's name.
# run_parameter_tuning.py names configs f"rr{rr:g}_t{t:g}_c{c:g}" -- parsing
# the name back is simpler than requiring a separate saved mapping file.
CFG_RE = re.compile(r"^rr([\d.]+)_t([\d.]+)_c([\d.]+)$")


def parse_cfg(name):
    m = CFG_RE.match(name)
    if not m:
        raise ValueError(
            f"Config name {name!r} doesn't match the rr..._t..._c... pattern "
            f"produced by run_parameter_tuning.py."
        )
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


obj_df = agg_df[agg_df["metric"] == "objective"].copy()
parsed = obj_df["config"].apply(parse_cfg)
obj_df["remove_ratio"] = parsed.apply(lambda t: t[0])
obj_df["temperature"]  = parsed.apply(lambda t: t[1])
obj_df["cooling"]      = parsed.apply(lambda t: t[2])

# ============================================================
# ONE-FACTOR-AT-A-TIME PANELS
# (mean objective +/- 95% CI vs each parameter, the other two held fixed
# at the best combination's values; a dashed line marks the chosen value)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

sweep_specs = [
    ("remove_ratio", dict(temperature=best_t, cooling=best_c), axes[0], "Destroy ratio"),
    ("temperature",  dict(remove_ratio=best_rr, cooling=best_c), axes[1], "Initial temperature"),
    ("cooling",      dict(remove_ratio=best_rr, temperature=best_t), axes[2], "Cooling rate"),
]

for param, fixed, ax, label in sweep_specs:
    sub = obj_df.copy()
    for k, v in fixed.items():
        sub = sub[np.isclose(sub[k], v)]
    sub = sub.sort_values(param)

    if len(sub) < 2:
        ax.set_title(f"{label}\n(not enough grid points with others fixed)", fontsize=9)
        continue

    yerr_low  = sub["mean"] - sub["ci95_low"]
    yerr_high = sub["ci95_high"] - sub["mean"]
    ax.errorbar(sub[param], sub["mean"], yerr=[yerr_low, yerr_high],
                fmt="o-", capsize=3, color=COLOR)
    ax.axvline(chosen[param], color=BEST_COLOR, ls="--", lw=1, alpha=0.8)

    ax.set_xlabel(label)
    ax.set_ylabel("Mean objective (95% CI)")
    fixed_str = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
    ax.set_title(f"vs. {label}\n(others fixed at best: {fixed_str})", fontsize=9)

fig.suptitle("LNS parameter sensitivity (one-factor-at-a-time)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])

png_path = os.path.join(figures_dir, "fig6_parameter_sensitivity.png")
pdf_path = os.path.join(figures_dir, "fig6_parameter_sensitivity.pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)

print(f"Figure written to:\n  {png_path}\n  {pdf_path}")
