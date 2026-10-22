# ============================================================
# QUALITY-OF-SERVICE (RIDE-TIME) OBJECTIVE-COMPONENT ABLATION
#
# Compares LNS results WITH vs WITHOUT ride time (ride_penalty, weighted
# by lambda) counted in the search objective, on the SAME 35 stochastic
# size=50 instances used in run_stochastic_case_study.py -- same instance
# file, same LNS seed, same every other setting, for BOTH arms, so any
# difference is attributable only to lambda.
#
# "With QoS" reuses the size=50 results already on disk (raw_size_50.csv)
# instead of re-running -- same instance/seed/settings, so re-running would
# just reproduce identical numbers. "Without QoS" is 35 new runs with
# --lambda_override 0 (ride_penalty is still computed and reported, the
# search just stops optimizing for it -- the hard per-patient ride-time cap
# stays enforced regardless).
#
# Usage:
#   python run_qos_ablation.py
#   python run_qos_ablation.py --with_qos_csv <existing raw_size_*.csv> --out_dir <dir> --label "lean fleet"
# ============================================================

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import stats_utils

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "darp_lns.py")

_parser = argparse.ArgumentParser()
_parser.add_argument("--with_qos_csv", type=str,
                      default=os.path.join(HERE, "paper_figures", "stochastic_case_study", "raw_size_50.csv"),
                      help="Existing raw_size_*.csv to reuse as the 'with QoS' arm -- its "
                           "'instance_path' column determines which instance files the "
                           "new 'without QoS' arm solves, so both arms are guaranteed to "
                           "use identical instances.")
_parser.add_argument("--out_dir", type=str,
                      default=os.path.join(HERE, "paper_figures", "qos_ablation", "size_50"))
_parser.add_argument("--label", type=str, default="size=50",
                      help="Short label for plot titles/console output, e.g. 'lean fleet'.")
_parser.add_argument("--lambda_value", type=float, default=0.0,
                      help="The --lambda_override value to run as the NEW arm being "
                           "compared against the existing with_qos_csv baseline. Default "
                           "0.0 (the original QoS-off ablation). Use e.g. 2 to compare "
                           "against lambda=2 instead.")
_parser.add_argument("--baseline_name", type=str, default="with_qos",
                      help="Label for the existing (with_qos_csv) arm in output tables.")
_parser.add_argument("--extra_args", type=str, default=None,
                      help="Extra darp_lns.py CLI args, as a single string, e.g. "
                           "'--shifts 1 --max_shifts 1 --disable_shift_transitions', for "
                           "the NEW arm being compared against with_qos_csv. Generalizes "
                           "this script beyond the lambda/QoS ablation to any same-"
                           "instance paired comparison. Overrides --lambda_value when given "
                           "(requires --config_name too).")
_parser.add_argument("--config_name", type=str, default=None,
                      help="Label for the new arm in output tables/filenames. Required "
                           "when --extra_args is given; defaults to 'lambda_<value>' "
                           "otherwise.")
_args = _parser.parse_args()

if _args.extra_args is not None:
    if _args.config_name is None:
        raise SystemExit("--config_name is required when --extra_args is given.")
    NEW_CONFIG_NAME = _args.config_name
    EXTRA_CLI_ARGS = _args.extra_args.split()
else:
    NEW_CONFIG_NAME = _args.config_name or f"lambda_{_args.lambda_value:g}"
    EXTRA_CLI_ARGS = ["--lambda_override", str(_args.lambda_value)]

WITH_QOS_CSV = _args.with_qos_csv
OUT_DIR = _args.out_dir
RUNS_DIR = os.path.join(OUT_DIR, NEW_CONFIG_NAME)
os.makedirs(RUNS_DIR, exist_ok=True)

# ---- LNS settings: identical to the original size=50 stochastic run ----
LNS_SEED = 42
ITERATIONS = 5000
EMA_FAST, EMA_SLOW, EMA_EPS, EMA_PATIENCE = 30, 150, 0.001, 1500
TEMPERATURE, COOLING, REMOVE_RATIO = 500.0, 0.9999, 0.10
WORKERS = max(1, (os.cpu_count() or 2) - 1)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

with_qos_df = pd.read_csv(WITH_QOS_CSV).sort_values("rep").reset_index(drop=True)
print(f"Loaded {len(with_qos_df)} existing 'with QoS' results from {WITH_QOS_CSV}")


def run_without_qos(rep, instance_path):
    run_dir = os.path.join(RUNS_DIR, f"rep_{rep}")
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summ = json.load(f)
        return rep, summ, summ.get("wall_time_sec", float("nan")), "skip"

    cmd = [
        sys.executable, SCRIPT,
        "--instance", os.path.join(HERE, instance_path),
        "--seed", str(LNS_SEED),
        "--iterations", str(ITERATIONS),
        "--ema_fast_window", str(EMA_FAST),
        "--ema_slow_window", str(EMA_SLOW),
        "--ema_eps", str(EMA_EPS),
        "--ema_patience", str(EMA_PATIENCE),
        "--temperature", str(TEMPERATURE),
        "--cooling", str(COOLING),
        "--remove_ratio", str(REMOVE_RATIO),
    ] + EXTRA_CLI_ARGS + [
        "--export_dir", run_dir,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"rep={rep} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    if not os.path.exists(summary_path):
        raise RuntimeError(f"rep={rep} exited 0 but never wrote {summary_path!r}.")

    with open(summary_path) as f:
        summ = json.load(f)
    summ["wall_time_sec"] = elapsed
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=2)
    return rep, summ, elapsed, "done"


jobs = list(zip(with_qos_df["rep"], with_qos_df["instance_path"]))
print(f"Running {len(jobs)} '{NEW_CONFIG_NAME}' ({' '.join(EXTRA_CLI_ARGS)}) "
      f"replications with {WORKERS} workers...\n")

without_rows = []
completed = 0
t_start = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_without_qos, rep, path): rep for rep, path in jobs}
    for future in as_completed(futures):
        rep, summ, elapsed, status = future.result()
        row = dict(rep=rep, config=NEW_CONFIG_NAME)
        row.update(summ)
        without_rows.append(row)
        completed += 1
        tag = "[skip]" if status == "skip" else f"time={elapsed:6.1f}s"
        print(f"[{completed:3d}/{len(jobs)}] rep={rep:<3d} obj={summ['objective']:10.2f}  "
              f"ride_penalty={summ['ride_penalty']:9.2f}  served={summ['served']}/{summ['num_patients']}  "
              f"feasible={str(summ['feasible']):<5}  {tag}")

total_elapsed = time.time() - t_start
without_df = pd.DataFrame(without_rows).sort_values("rep").reset_index(drop=True)

with_qos_df = with_qos_df.copy()
with_qos_df["config"] = _args.baseline_name

METRICS = ["objective", "travel_cost", "ride_penalty", "violation_cost", "served"]
common_cols = ["rep", "config"] + [c for c in METRICS if c in with_qos_df.columns and c in without_df.columns] \
    + ["num_patients", "feasible", "wall_time_sec"]

combined = pd.concat([with_qos_df[common_cols], without_df[common_cols]], ignore_index=True)
combined_csv = os.path.join(OUT_DIR, "qos_ablation_raw.csv")
combined.to_csv(combined_csv, index=False)

# ============================================================
# PAIRED COMPARISON (paired by rep -- same instance both arms)
# ============================================================

agg_df = stats_utils.aggregate_stats(combined, "config", METRICS)
agg_csv = os.path.join(OUT_DIR, "qos_ablation_summary.csv")
agg_df.to_csv(agg_csv, index=False)

sig_rows = []
for metric in METRICS:
    sig_df = stats_utils.paired_significance(combined, "config", "rep", metric, _args.baseline_name)
    sig_df.insert(0, "metric", metric)
    sig_rows.append(sig_df)
sig_all = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
sig_csv = os.path.join(OUT_DIR, "qos_ablation_significance.csv")
sig_all.to_csv(sig_csv, index=False)

xlsx_path = os.path.join(OUT_DIR, "qos_ablation_report.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    agg_df.to_excel(writer, sheet_name="summary_stats", index=False)
    sig_all.to_excel(writer, sheet_name="significance_tests", index=False)
    combined.to_excel(writer, sheet_name="raw_paired", index=False)

# ============================================================
# PLOT -- paired objective/ride_penalty, with vs without QoS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, metric in zip(axes, ["objective", "ride_penalty"]):
    a = combined[combined["config"] == _args.baseline_name].set_index("rep")[metric].sort_index()
    b = combined[combined["config"] == NEW_CONFIG_NAME].set_index("rep")[metric].sort_index()
    for rep in a.index:
        ax.plot([0, 1], [a[rep], b[rep]], color="lightgray", lw=0.8, zorder=1)
    ax.scatter([0] * len(a), a.values, color="steelblue", zorder=2, label=_args.baseline_name)
    ax.scatter([1] * len(b), b.values, color="indianred", zorder=2, label=NEW_CONFIG_NAME)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([_args.baseline_name, NEW_CONFIG_NAME])
    ax.set_ylabel(metric)
    ax.set_title(metric)
axes[0].legend()
fig.suptitle(f"QoS (ride-time) objective-component ablation -- 35 paired instances ({_args.label})")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(OUT_DIR, "qos_ablation_paired.png")
fig.savefig(fig_path, dpi=200)
plt.close(fig)

# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n" + "=" * 90)
print(f"QoS ABLATION ({_args.label}) -- {len(jobs)} paired replications, {total_elapsed/60:.1f} min "
      f"(with-QoS side reused from {os.path.basename(WITH_QOS_CSV)}, no re-run)")
print("=" * 90)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

print(f"\n--- Descriptive stats ({_args.baseline_name} vs {NEW_CONFIG_NAME}) ---")
print(agg_df.to_string(index=False))

print("\n--- Paired significance (paired by rep = identical instance both arms) ---")
print(sig_all.to_string(index=False) if len(sig_all) else "  (no pairs)")

print(f"\nRaw paired CSV     : {combined_csv}")
print(f"Summary stats CSV  : {agg_csv}")
print(f"Significance CSV   : {sig_csv}")
print(f"Excel report       : {xlsx_path}")
print(f"Plot               : {fig_path}")
