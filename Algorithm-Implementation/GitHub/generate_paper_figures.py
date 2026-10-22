# ============================================================
# Runs the experiments behind the paper's results section and
# generates all figures from the actual output of darp_lns.py
# (no synthetic/placeholder numbers).
#
# Usage:
#   python generate_paper_figures.py
#   python generate_paper_figures.py --force        # re-run everything
#   python generate_paper_figures.py --iterations 3000
# ============================================================

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# __file__ isn't defined when this code runs inside a notebook cell (as opposed
# to a real .py file via `%run` or `python ...`), so fall back to the current
# working directory in that case.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

SCRIPT        = os.path.join(HERE, "darp_lns.py")
INSTANCE      = os.path.join(HERE, "instances", "E_20_5_1", "E_20_5_1_model_input.xlsx")
EXPERIMENTS   = os.path.join(HERE, "paper_figures", "experiments")
FIGURES_DIR   = os.path.join(HERE, "paper_figures", "figures")

_parser = argparse.ArgumentParser()
_parser.add_argument("--iterations", type=int, default=1500,
                      help="Iteration budget for every experiment run.")
_parser.add_argument("--force", action="store_true",
                      help="Re-run experiments even if results already exist.")
_args, _ = _parser.parse_known_args()   # ignore extra args (e.g. Jupyter's own flags)

# ============================================================
# CLEAN, ACADEMIC PLOT STYLE
# ============================================================

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

COLOR_STAGE = {
    "s1_1trip_naive":     "#B0413E",
    "s2_3trip_naive":     "#D98B3F",
    "s3_3trip_aware":     "#4C8577",
    "s4_3trip_aware_7am": "#2E5B8C",
}
COLOR_OUT = "#2E5B8C"
COLOR_RET = "#4C8577"

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

ABLATION = [
    dict(tag="s1_1trip_naive",     label="1 trip/vehicle\n(naive scoring)\n08:00 start",
         args=["--shifts", "1", "--day_start_hour", "8", "--naive_insertion"]),
    dict(tag="s2_3trip_naive",     label="3 trips/vehicle\n(naive scoring)\n08:00 start",
         args=["--shifts", "3", "--day_start_hour", "8", "--naive_insertion"]),
    dict(tag="s3_3trip_aware",     label="3 trips/vehicle\n(chain-aware)\n08:00 start",
         args=["--shifts", "3", "--day_start_hour", "8"]),
    dict(tag="s4_3trip_aware_7am", label="3 trips/vehicle\n(chain-aware)\n07:00 start",
         args=["--shifts", "3", "--day_start_hour", "7"]),
]
FINAL_TAG = "s4_3trip_aware_7am"   # reused for the convergence curve + Gantt chart

DAY_START_SWEEP = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
SHIFTS_SWEEP    = [1, 2, 3, 4, 5]

# ============================================================
# RUN EXPERIMENTS (skips ones already on disk unless --force)
# ============================================================

def run_experiment(tag, extra_args):
    out_dir = os.path.join(EXPERIMENTS, tag)
    summary_path = os.path.join(out_dir, "summary.json")

    if os.path.exists(summary_path) and not _args.force:
        print(f"[skip]  {tag} (already exists)")
        return out_dir

    print(f"[run ]  {tag}  ({' '.join(extra_args)})")
    cmd = [
        sys.executable, SCRIPT,
        "--instance", INSTANCE,
        "--iterations", str(_args.iterations),
        "--export_dir", out_dir,
    ] + extra_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Experiment '{tag}' failed (exit {result.returncode})")

    return out_dir


def load_summary(tag):
    with open(os.path.join(EXPERIMENTS, tag, "summary.json")) as f:
        return json.load(f)


def load_csv(tag, name):
    return pd.read_csv(os.path.join(EXPERIMENTS, tag, name))


print("=" * 70)
print("RUNNING EXPERIMENTS")
print("=" * 70)

for cfg in ABLATION:
    run_experiment(cfg["tag"], cfg["args"])

for h in DAY_START_SWEEP:
    run_experiment(f"sweep_day_start_{h}", ["--shifts", "3", "--day_start_hour", str(h)])

for s in SHIFTS_SWEEP:
    run_experiment(f"sweep_shifts_{s}", ["--shifts", str(s), "--day_start_hour", "7"])

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# FIGURE 1 -- ABLATION: objective / violation cost / violated patients
# ============================================================

print("\nBuilding Figure 1 (ablation)...")

summaries = {cfg["tag"]: load_summary(cfg["tag"]) for cfg in ABLATION}
labels    = [cfg["label"] for cfg in ABLATION]
colors    = [COLOR_STAGE[cfg["tag"]] for cfg in ABLATION]

objectives  = [summaries[cfg["tag"]]["objective"] for cfg in ABLATION]
viol_costs  = [summaries[cfg["tag"]]["violation_cost"] for cfg in ABLATION]
n_violated  = [summaries[cfg["tag"]]["num_violated_patients"] for cfg in ABLATION]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for ax, values, title, ylabel, logscale in [
    (axes[0], objectives, "Objective value", "Objective (log scale)", True),
    (axes[1], viol_costs, "Violation cost", "Violation cost (log scale)", True),
    (axes[2], n_violated, "Patients with a\nviolated time constraint", "Patients (of 20)", False),
]:
    bars = ax.bar(range(len(values)), values, color=colors, width=0.65)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    if logscale:
        ax.set_yscale("log")
    for bar, v in zip(bars, values):
        label = f"{v:,.0f}" if v >= 100 else f"{v:.0f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)

fig.suptitle("Effect of each modeling/algorithmic stage on solution quality (E_20_5_1)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGURES_DIR, "fig1_ablation.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig1_ablation.pdf"))
plt.close(fig)

# ============================================================
# FIGURE 2 -- LNS CONVERGENCE CURVE (final config)
# ============================================================

print("Building Figure 2 (convergence curve)...")

hist = load_csv(FINAL_TAG, "history.csv")
summ = load_summary(FINAL_TAG)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(hist["iteration"], hist["current_objective"], color="#B7C4CE", lw=1, label="Current solution")
ax.plot(hist["iteration"], hist["best_objective"], color=COLOR_STAGE[FINAL_TAG], lw=2, label="Best solution")

if summ["iterations_run"] < summ["iterations_budget"]:
    ax.axvline(summ["iterations_run"], color="#B0413E", ls="--", lw=1.2)
    ax.annotate("EMA stopping\ntriggered", (summ["iterations_run"], ax.get_ylim()[1]),
                textcoords="offset points", xytext=(6, -14), fontsize=9, color="#B0413E")

ax.set_yscale("log")
ax.set_xlabel("LNS iteration")
ax.set_ylabel("Objective value (log scale)")
ax.set_title("LNS convergence -- final configuration (3 trips/vehicle, 07:00 start)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig2_convergence.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig2_convergence.pdf"))
plt.close(fig)

# ============================================================
# FIGURE 3 -- VEHICLE-DAY GANTT CHART (final config)
# ============================================================

print("Building Figure 3 (vehicle-day Gantt chart)...")

sched = load_csv(FINAL_TAG, "daily_schedule.csv")
vehicles = sorted(sched["vehicle"].unique())

fig, ax = plt.subplots(figsize=(10, 0.55 * len(vehicles) + 1.5))

for i, v in enumerate(vehicles):
    trips = sched[sched["vehicle"] == v]
    for _, trip in trips.iterrows():
        ax.barh(i, trip["out_finish"] - trip["out_start"], left=trip["out_start"],
                height=0.55, color=COLOR_OUT, edgecolor="white")
        ax.barh(i, trip["ret_finish"] - trip["ret_start"], left=trip["ret_start"],
                height=0.55, color=COLOR_RET, edgecolor="white")
        mid = (trip["out_start"] + trip["ret_finish"]) / 2
        ax.annotate(f"trip {int(trip['trip'])}", (mid, i), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

ax.set_yticks(range(len(vehicles)))
ax.set_yticklabels(vehicles, fontsize=9)
ax.invert_yaxis()

# x-axis in clock-time labels
day_start_min = int(round(summ["day_start_hour"] * 60))
xt = range(0, int(sched[["out_finish", "ret_finish"]].values.max()) + 60, 60)
ax.set_xticks(list(xt))
ax.set_xticklabels([f"{(day_start_min + m)//60:02d}:{(day_start_min + m)%60:02d}" for m in xt],
                    rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Time of day")
ax.set_title("Vehicle-day schedule -- each vehicle's separate trips (final configuration)")

handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_OUT), plt.Rectangle((0, 0), 1, 1, color=COLOR_RET)]
ax.legend(handles, ["Outbound leg", "Return leg"], frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_gantt.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig3_gantt.pdf"))
plt.close(fig)

# ============================================================
# FIGURE 4 -- SENSITIVITY SWEEPS (day_start_hour, shifts)
# ============================================================

print("Building Figure 4 (sensitivity sweeps)...")

day_start_results = [load_summary(f"sweep_day_start_{h}") for h in DAY_START_SWEEP]
shifts_results    = [load_summary(f"sweep_shifts_{s}") for s in SHIFTS_SWEEP]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")

hh = [f"{int(h):02d}:{int(round((h % 1) * 60)):02d}" for h in DAY_START_SWEEP]
axes[0, 0].plot(hh, [r["objective"] for r in day_start_results], "o-", color=COLOR_STAGE["s4_3trip_aware_7am"])
axes[0, 0].set_ylabel("Objective (log scale)")
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("vs. vehicle day-start time")

axes[1, 0].plot(hh, [r["num_violated_patients"] for r in day_start_results], "o-", color="#B0413E")
axes[1, 0].set_ylabel("Violated patients (of 20)")
axes[1, 0].set_xlabel("Day-start hour")
axes[1, 0].tick_params(axis="x", rotation=45)

axes[0, 1].plot(SHIFTS_SWEEP, [r["objective"] for r in shifts_results], "o-", color=COLOR_STAGE["s4_3trip_aware_7am"])
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("vs. max trips/vehicle/day")

axes[1, 1].plot(SHIFTS_SWEEP, [r["num_violated_patients"] for r in shifts_results], "o-", color="#B0413E")
axes[1, 1].set_xlabel("Max shifts per vehicle")
axes[1, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.suptitle("Sensitivity of solution quality to day-start time and shift count", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGURES_DIR, "fig4_sensitivity.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig4_sensitivity.pdf"))
plt.close(fig)

# ============================================================
# FIGURE 5 -- VIOLATION-TYPE BREAKDOWN (baseline vs. final)
# ============================================================

print("Building Figure 5 (violation-type breakdown)...")

BASELINE_TAG = "s1_1trip_naive"
types = ["out_early_violations", "out_late_violations", "out_ride_violations", "return_delay_violations", "ret_ride_violations", "num_capacity_events"]
type_labels = ["Outbound\ntoo early", "Outbound\nlateness", "Outbound\nride-time", "Return\ndelay", "Return\nride-time", "Capacity\noverflow"]

base_summ  = load_summary(BASELINE_TAG)
final_summ = load_summary(FINAL_TAG)

base_vals  = [base_summ[t] for t in types]
final_vals = [final_summ[t] for t in types]

x = range(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline (1 trip, naive scoring, 08:00)",
       color=COLOR_STAGE[BASELINE_TAG])
ax.bar([i + width / 2 for i in x], final_vals, width, label="Final (3 trips, chain-aware, 07:00)",
       color=COLOR_STAGE[FINAL_TAG])
ax.set_xticks(list(x))
ax.set_xticklabels(type_labels, fontsize=9)
ax.set_ylabel("Number of events")
ax.set_title("Constraint violations by type: baseline vs. final configuration")
ax.legend(frameon=False, fontsize=9)

for i, (b, f) in enumerate(zip(base_vals, final_vals)):
    ax.annotate(str(b), (i - width / 2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.annotate(str(f), (i + width / 2, f), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig5_violation_breakdown.png"))
fig.savefig(os.path.join(FIGURES_DIR, "fig5_violation_breakdown.pdf"))
plt.close(fig)

print(f"\nAll figures written to: {FIGURES_DIR}")
