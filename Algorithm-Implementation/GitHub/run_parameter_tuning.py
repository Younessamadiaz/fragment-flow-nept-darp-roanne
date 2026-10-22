# ============================================================
# LNS PARAMETER TUNING -- grid search over REMOVE_RATIO / TEMPERATURE /
# COOLING, reusing run_replications.py's own multi-seed statistics engine
# (paired seeds, mean/median/95% CI, significance tests) so every grid
# point gets the same rigor as the ablation studies, not just a single
# noisy run per configuration.
#
# For each (remove_ratio, temperature, cooling) combination in the grid:
#   1. Runs --n_reps seeded replications of darp_lns.py via
#      run_replications.py (paired seeds across configs).
#   2. Picks the lowest-mean-objective combination.
#   3. Plots one-factor-at-a-time sensitivity (mean objective +/- 95% CI
#      vs each parameter, holding the other two fixed at the best-found
#      values) -- same style as generate_paper_figures.py's fig4.
#   4. Writes chosen_parameters.json for reuse when generating the final
#      results reported in the paper (that run should use a much larger
#      --n_reps, on the real test instances, NOT this tuning pass).
#
# Usage:
#   python run_parameter_tuning.py --n_reps 10
#   python run_parameter_tuning.py --instance <path> --n_reps 10 \
#       --remove_ratios "0.05,0.10,0.20" --temperatures "100,500,1000" \
#       --coolings "0.999,0.9995,0.9999"
# ============================================================

import argparse
import itertools
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- locate run_replications.py (same known-folder-first lookup it uses
# for darp_lns.py, to avoid silently picking up a stray copy elsewhere) --
_KNOWN_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(_KNOWN_PROJECT_DIR, "run_replications.py")):
    HERE = _KNOWN_PROJECT_DIR
else:
    try:
        HERE = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        HERE = os.getcwd()

RUN_REPLICATIONS = os.path.join(HERE, "run_replications.py")
if not os.path.exists(RUN_REPLICATIONS):
    raise FileNotFoundError(
        f"Can't find run_replications.py -- looked in {HERE!r}. "
        f"Run this from the VRP_paper folder."
    )

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--instance", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances", "E_20_5_1", "E_20_5_1_model_input.xlsx"),
)
_parser.add_argument("--remove_ratios", type=str, default="0.05,0.10,0.20",
                      help="Comma-separated grid values for --remove_ratio.")
_parser.add_argument("--temperatures", type=str, default="100,500,1000",
                      help="Comma-separated grid values for --temperature.")
_parser.add_argument("--coolings", type=str, default="0.999,0.9995,0.9999",
                      help="Comma-separated grid values for --cooling.")
_parser.add_argument("--n_reps", type=int, default=10,
                      help="Seeds per grid point. Lower than run_replications.py's own "
                           "default (100) on purpose -- this sweeps many configurations, "
                           "it's the tuning pass, not the final validation run.")
_parser.add_argument("--seed_start", type=int, default=0)
_parser.add_argument("--iterations", type=int, default=5000,
                      help="Passed through to darp_lns.py as --iterations (safety ceiling; "
                           "the EMA criterion below is expected to actually end each run).")
_parser.add_argument("--ema_fast_window", type=int, default=30)
_parser.add_argument("--ema_slow_window", type=int, default=150)
_parser.add_argument("--ema_eps", type=float, default=0.01)
_parser.add_argument("--ema_patience", type=int, default=2500)
_parser.add_argument("--baseline_remove_ratio", type=float, default=0.10,
                      help="darp_lns.py's own default -- added to the grid if not already "
                           "present, and used as the significance-test reference.")
_parser.add_argument("--baseline_temperature", type=float, default=500)
_parser.add_argument("--baseline_cooling", type=float, default=0.9995)
_parser.add_argument("--out_dir", type=str, default=None)
_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
_parser.add_argument("--force", action="store_true",
                      help="Passed through to run_replications.py -- re-run replications "
                           "even if a run's summary.json already exists.")
_args, _ = _parser.parse_known_args()

remove_ratios = sorted({float(x) for x in _args.remove_ratios.split(",")} | {_args.baseline_remove_ratio})
temperatures  = sorted({float(x) for x in _args.temperatures.split(",")} | {_args.baseline_temperature})
coolings      = sorted({float(x) for x in _args.coolings.split(",")} | {_args.baseline_cooling})

grid = list(itertools.product(remove_ratios, temperatures, coolings))


def cfg_name(rr, t, c):
    return f"rr{rr:g}_t{t:g}_c{c:g}"


configs   = {cfg_name(rr, t, c): f"--remove_ratio {rr} --temperature {t} --cooling {c}" for rr, t, c in grid}
param_of  = {cfg_name(rr, t, c): (rr, t, c) for rr, t, c in grid}
baseline_name = cfg_name(_args.baseline_remove_ratio, _args.baseline_temperature, _args.baseline_cooling)

instance_name = os.path.basename(os.path.dirname(_args.instance)) or "instance"
out_dir = _args.out_dir or os.path.join(HERE, "paper_figures", "tuning", instance_name)
os.makedirs(out_dir, exist_ok=True)

print(f"Grid          : {len(remove_ratios)} remove_ratio x {len(temperatures)} temperature x "
      f"{len(coolings)} cooling = {len(grid)} configurations")
print(f"Replications  : {_args.n_reps} seeds/config  ({len(grid) * _args.n_reps} total darp_lns.py runs)")
print(f"Baseline      : {baseline_name}  (used as the significance-test reference)")
print(f"Output dir    : {out_dir}\n")

cmd = [
    sys.executable, RUN_REPLICATIONS,
    "--instance", _args.instance,
    "--n_reps", str(_args.n_reps),
    "--seed_start", str(_args.seed_start),
    "--iterations", str(_args.iterations),
    "--ema_fast_window", str(_args.ema_fast_window),
    "--ema_slow_window", str(_args.ema_slow_window),
    "--ema_eps", str(_args.ema_eps),
    "--ema_patience", str(_args.ema_patience),
    "--configs", json.dumps(configs),
    "--baseline", baseline_name,
    "--out_dir", out_dir,
    "--workers", str(_args.workers),
]
if _args.force:
    cmd.append("--force")

print("Running grid via run_replications.py (this can take a while)...\n")
result = subprocess.run(cmd)
if result.returncode != 0:
    raise RuntimeError(f"run_replications.py failed (exit {result.returncode})")

# ============================================================
# LOAD RESULTS AND PICK THE BEST CONFIGURATION
# ============================================================

summary_csv = os.path.join(out_dir, "replications_summary.csv")
agg_df = pd.read_csv(summary_csv)
obj_df = agg_df[agg_df["metric"] == "objective"].copy()
obj_df["remove_ratio"] = obj_df["config"].map(lambda c: param_of[c][0])
obj_df["temperature"]  = obj_df["config"].map(lambda c: param_of[c][1])
obj_df["cooling"]      = obj_df["config"].map(lambda c: param_of[c][2])

best_row = obj_df.loc[obj_df["mean"].idxmin()]
best_rr, best_t, best_c = float(best_row["remove_ratio"]), float(best_row["temperature"]), float(best_row["cooling"])
best_name = best_row["config"]

print("\n" + "=" * 90)
print("BEST CONFIGURATION FOUND")
print("=" * 90)
print(f"  remove_ratio={best_rr:g}  temperature={best_t:g}  cooling={best_c:g}")
print(f"  mean objective = {best_row['mean']:.2f}  "
      f"(95% CI [{best_row['ci95_low']:.2f}, {best_row['ci95_high']:.2f}])")

chosen = dict(
    instance=_args.instance, n_reps=_args.n_reps,
    remove_ratio=best_rr, temperature=best_t, cooling=best_c,
    mean_objective=float(best_row["mean"]),
    ci95_low=float(best_row["ci95_low"]), ci95_high=float(best_row["ci95_high"]),
    baseline_config=baseline_name, best_config=best_name,
)
chosen_path = os.path.join(out_dir, "chosen_parameters.json")
with open(chosen_path, "w") as f:
    json.dump(chosen, f, indent=2)
print(f"\nChosen parameters written to: {chosen_path}")
print("Reuse these on the real test instances with a much larger --n_reps for the "
      "number that actually goes in the paper -- this tuning pass is not that run.")

# ============================================================
# ONE-FACTOR-AT-A-TIME SENSITIVITY PLOTS
# (mean objective +/- 95% CI vs each parameter, the other two held fixed
# at the best combination's values -- guaranteed to exist since the grid
# is a full factorial design)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sweep_specs = [
    ("remove_ratio", remove_ratios, dict(temperature=best_t, cooling=best_c), axes[0], "Destroy ratio"),
    ("temperature",  temperatures,  dict(remove_ratio=best_rr, cooling=best_c), axes[1], "Initial temperature"),
    ("cooling",      coolings,      dict(remove_ratio=best_rr, temperature=best_t), axes[2], "Cooling rate"),
]

for param, values, fixed, ax, label in sweep_specs:
    sub = obj_df.copy()
    for k, v in fixed.items():
        sub = sub[np.isclose(sub[k], v)]
    sub = sub.sort_values(param)
    if len(sub) < 2:
        ax.set_title(f"{label}\n(not enough grid points with others fixed)", fontsize=9)
        continue
    yerr_low  = sub["mean"] - sub["ci95_low"]
    yerr_high = sub["ci95_high"] - sub["mean"]
    ax.errorbar(sub[param], sub["mean"], yerr=[yerr_low, yerr_high], fmt="o-", capsize=3)
    ax.set_xlabel(label)
    ax.set_ylabel("Mean objective (95% CI)")
    fixed_str = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
    ax.set_title(f"vs. {label}\n(others fixed at best: {fixed_str})", fontsize=9)

fig.suptitle(f"LNS parameter sensitivity -- {instance_name} ({_args.n_reps} seeds/point)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig_path = os.path.join(out_dir, "parameter_sensitivity.png")
fig.savefig(fig_path, dpi=200)
plt.close(fig)

print(f"Sensitivity plot written to : {fig_path}")
print(f"Full grid results           : {summary_csv}")
print(f"Significance tests vs base. : {os.path.join(out_dir, 'significance_tests.csv')}")
