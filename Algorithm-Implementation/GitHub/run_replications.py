# ============================================================
# Repeated-seed LNS-DARP experiment runner.
#
# Runs darp_lns.py N times per configuration (fixed instance + fixed
# algorithm parameters, varying ONLY --seed) to capture the solution-
# quality variability of the stochastic heuristic, then reports:
#   - mean and standard deviation
#   - median and interquartile range (IQR)
#   - 95% confidence interval (t-distribution)
#   - minimum and maximum
#   - runtime (wall-clock seconds per run)
#   - the best-performing configuration
#   - box plot and violin plot of the objective distribution
#   - paired statistical significance (paired t-test + Wilcoxon
#     signed-rank) and effect size (Cohen's dz + matched-pairs
#     rank-biserial correlation) against a baseline configuration
#
# Because the SAME seed list is reused for every configuration, each
# configuration's per-seed objective is naturally paired with the
# baseline's run on that same seed -- which is what makes the paired
# tests valid (not just an independent-samples comparison).
#
# Usage (default: baseline = naive insertion vs. proposed = chain-aware,
# using darp_lns.py's own --naive_insertion ablation switch):
#   python run_replications.py --n_reps 100
#
# Usage (custom configs):
#   python run_replications.py --n_reps 100 \
#       --configs "{\"1_trip\": \"--shifts 1\", \"3_trip\": \"--shifts 3\"}" \
#       --baseline 1_trip
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
import numpy as np
import pandas as pd

import stats_utils
from stats_utils import describe

# ---- locate darp_lns.py --------------------------------------------------
# Always prefer the known, verified project folder over whatever __file__ /
# os.getcwd() resolve to -- if this is pasted into a notebook cell running
# with cwd=<home dir> (or anywhere else that happens to also contain
# an older/different darp_lns.py), a cwd-based lookup can silently pick up
# the WRONG copy: it still runs (exit code 0) but may be missing features
# like --export_dir, so no summary.json ever gets written and the failure
# only shows up much later as a confusing FileNotFoundError.
_KNOWN_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(_KNOWN_PROJECT_DIR, "darp_lns.py")):
    HERE = _KNOWN_PROJECT_DIR
else:
    try:
        HERE = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        HERE = os.getcwd()

SCRIPT = os.path.join(HERE, "darp_lns.py")
if not os.path.exists(SCRIPT):
    raise FileNotFoundError(
        f"Can't find darp_lns.py -- looked in {HERE!r}. "
        f"Run this from the VRP_paper folder, or pass --out_dir explicitly."
    )
print(f"Using darp_lns.py at: {SCRIPT}")

DEFAULT_CONFIGS = {
    "baseline_naive":       "--naive_insertion",
    "proposed_chain_aware": "",
}

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--instance", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances", "E_20_5_1", "E_20_5_1_model_input.xlsx"),
    help="Path to a *_model_input.xlsx instance file.",
)
_parser.add_argument("--n_reps", type=int, default=100,
                      help="Number of replications (random seeds) PER configuration.")
_parser.add_argument("--seed_start", type=int, default=0,
                      help="First seed; seeds used are seed_start .. seed_start+n_reps-1, "
                           "the SAME seeds reused across every configuration (paired design).")
_parser.add_argument("--iterations", type=int, default=20000,
                      help="Hard ceiling on LNS iterations, passed through to darp_lns.py "
                           "as --iterations. Set high on purpose -- with --ema_patience below "
                           "this should rarely bind; the EMA stopping criterion is what's "
                           "expected to actually end each run.")
_parser.add_argument("--ema_fast_window", type=int, default=30,
                      help="Passed through to darp_lns.py --ema_fast_window.")
_parser.add_argument("--ema_slow_window", type=int, default=150,
                      help="Passed through to darp_lns.py --ema_slow_window.")
_parser.add_argument("--ema_eps", type=float, default=0.01,
                      help="Passed through to darp_lns.py --ema_eps.")
_parser.add_argument("--ema_patience", type=int, default=2500,
                      help="Passed through to darp_lns.py --ema_patience. Lower than "
                           "darp_lns.py's own built-in default (5000) -- at 5000 the "
                           "criterion can never actually trigger inside a 5000-iteration "
                           "run (not enough post-warmup iterations left to accumulate "
                           "5000 consecutive stale ones), which is why every run in testing "
                           "hit 'max iterations reached' instead of an EMA-triggered stop.")
_parser.add_argument("--out_dir", type=str, default=None,
                      help="Where to write per-run exports + the aggregate report. "
                           "Defaults to paper_figures/replications/<instance_name>.")
_parser.add_argument("--configs", type=str, default=None,
                      help="JSON object mapping config name -> extra darp_lns.py args, e.g. "
                           '\'{"1_trip": "--shifts 1", "3_trip": "--shifts 3"}\'. Defaults to '
                           "a baseline (naive insertion) vs proposed (chain-aware) ablation.")
_parser.add_argument("--baseline", type=str, default=None,
                      help="Name of the config treated as the baseline for significance tests "
                           "and effect size. Defaults to the first key in --configs.")
_parser.add_argument("--force", action="store_true",
                      help="Re-run replications even if a run's summary.json already exists.")
_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                      help="Number of (config, seed) runs to execute concurrently.")
_args, _ = _parser.parse_known_args()   # ignore extra args (e.g. Jupyter's own flags)

configs = json.loads(_args.configs) if _args.configs else dict(DEFAULT_CONFIGS)
baseline_name = _args.baseline or next(iter(configs))
if baseline_name not in configs:
    raise ValueError(f"--baseline {baseline_name!r} is not one of the configs {list(configs)}")

instance_name = os.path.basename(os.path.dirname(_args.instance)) or "instance"
out_dir = _args.out_dir or os.path.join(HERE, "paper_figures", "replications", instance_name)
runs_dir = os.path.join(out_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"   # force non-interactive backend so subprocess plt.show() never blocks

seeds = list(range(_args.seed_start, _args.seed_start + _args.n_reps))
jobs = [(cfg_name, seed) for cfg_name in configs for seed in seeds]

print(f"Instance      : {_args.instance}")
print(f"Configs       : {list(configs.keys())}  (baseline = {baseline_name!r})")
print(f"Replications  : {len(seeds)} seeds/config  (seeds {seeds[0]}..{seeds[-1]}, shared across configs)")
print(f"Iteration cap : {_args.iterations}  (safety ceiling -- actual stop is the EMA criterion below)")
print(f"EMA stopping  : fast_window={_args.ema_fast_window}  slow_window={_args.ema_slow_window}  "
      f"eps={_args.ema_eps}  patience={_args.ema_patience}")
print(f"Total runs    : {len(jobs)}")
print(f"Workers       : {_args.workers}  (of {os.cpu_count()} CPUs)")
print(f"Output dir    : {out_dir}\n")


def run_one(cfg_name, seed):
    extra_args = configs[cfg_name].split() if configs[cfg_name] else []
    run_dir = os.path.join(runs_dir, cfg_name, f"seed_{seed}")
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path) and not _args.force:
        with open(summary_path) as f:
            summ = json.load(f)
        elapsed = summ.get("wall_time_sec", float("nan"))
        status = "skip"
    else:
        cmd = [
            sys.executable, SCRIPT,
            "--instance", _args.instance,
            "--seed", str(seed),
            "--iterations", str(_args.iterations),
            "--ema_fast_window", str(_args.ema_fast_window),
            "--ema_slow_window", str(_args.ema_slow_window),
            "--ema_eps", str(_args.ema_eps),
            "--ema_patience", str(_args.ema_patience),
            "--export_dir", run_dir,
        ] + extra_args

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        elapsed = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Run config={cfg_name} seed={seed} failed (exit {result.returncode}):\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )

        if not os.path.exists(summary_path):
            raise RuntimeError(
                f"Run config={cfg_name} seed={seed} exited with code 0 but never wrote "
                f"{summary_path!r}. This usually means SCRIPT ({SCRIPT!r}) is an older/"
                f"different darp_lns.py that doesn't support --export_dir. "
                f"Last 2000 chars of its stdout:\n{result.stdout[-2000:]}"
            )

        with open(summary_path) as f:
            summ = json.load(f)
        summ["wall_time_sec"] = elapsed
        with open(summary_path, "w") as f:
            json.dump(summ, f, indent=2)
        status = "done"

    row = dict(config=cfg_name, seed=seed, wall_time_sec=elapsed)
    row.update(summ)
    return cfg_name, seed, row, status, elapsed


rows = []
t_all_start = time.time()
completed = 0

with ThreadPoolExecutor(max_workers=_args.workers) as pool:
    futures = {pool.submit(run_one, cfg_name, seed): (cfg_name, seed) for cfg_name, seed in jobs}
    for future in as_completed(futures):
        cfg_name, seed, row, status, elapsed = future.result()
        rows.append(row)
        completed += 1
        tag = "[skip]" if status == "skip" else f"time={elapsed:6.1f}s"
        print(f"[{completed:4d}/{len(jobs)}] {cfg_name:<22} seed={seed:<4d} "
              f"obj={row['objective']:10.2f}  served={row['served']}/{row['num_patients']}  "
              f"feasible={str(row['feasible']):<5}  {tag}")

rows.sort(key=lambda r: (r["config"], r["seed"]))
total_elapsed = time.time() - t_all_start

raw_df = pd.DataFrame(rows)
raw_csv = os.path.join(out_dir, "replications_raw.csv")
raw_df.to_csv(raw_csv, index=False)

# ============================================================
# PER-CONFIG DESCRIPTIVE STATISTICS
# mean/std, median/IQR, 95% CI (t-distribution), min/max, runtime,
# feasibility rate -- one row per (config, metric).
# ============================================================

METRICS = [
    "objective", "travel_cost", "ride_penalty", "violation_cost",
    "served", "num_violated_patients", "num_capacity_events",
    "wall_time_sec", "iterations_run",
]
METRICS = [m for m in METRICS if m in raw_df.columns]


agg_rows = []
for cfg_name in configs:
    sub = raw_df[raw_df["config"] == cfg_name]
    for m in METRICS:
        d = describe(sub[m])
        d.update(config=cfg_name, metric=m)
        agg_rows.append(d)
    feas = sub["feasible"].mean() if "feasible" in sub.columns else float("nan")
    agg_rows.append(dict(
        config=cfg_name, metric="feasible_rate", n=len(sub), mean=feas,
        std=float("nan"), median=float("nan"), q1=float("nan"), q3=float("nan"),
        iqr=float("nan"), ci95_low=float("nan"), ci95_high=float("nan"),
        ci95_halfwidth=float("nan"), min=float("nan"), max=float("nan"),
    ))
    # Fraction of runs where the EMA criterion actually triggered the stop
    # (vs. hitting the --iterations safety ceiling) -- confirms the EMA
    # patience setting is doing its job rather than the ceiling deciding.
    if "stop_reason" in sub.columns:
        ema_rate = sub["stop_reason"].str.contains("EMA", na=False).mean()
    else:
        ema_rate = float("nan")
    agg_rows.append(dict(
        config=cfg_name, metric="ema_stop_rate", n=len(sub), mean=ema_rate,
        std=float("nan"), median=float("nan"), q1=float("nan"), q3=float("nan"),
        iqr=float("nan"), ci95_low=float("nan"), ci95_high=float("nan"),
        ci95_halfwidth=float("nan"), min=float("nan"), max=float("nan"),
    ))

agg_df = pd.DataFrame(agg_rows)[
    ["config", "metric", "n", "mean", "std", "median", "q1", "q3", "iqr",
     "ci95_low", "ci95_high", "ci95_halfwidth", "min", "max"]
]
agg_csv = os.path.join(out_dir, "replications_summary.csv")
agg_df.to_csv(agg_csv, index=False)

# ============================================================
# BEST CONFIGURATION -- lowest mean objective across configs.
# (Feasibility rate is reported alongside, not used as a hard filter,
# since this is a penalty-based objective where an infeasible run can
# still legitimately score lower than a feasible one.)
# ============================================================

obj_by_config = agg_df[agg_df["metric"] == "objective"].set_index("config")
best_config = obj_by_config["mean"].idxmin()
best_mean_obj = obj_by_config.loc[best_config, "mean"]

# ============================================================
# PAIRED SIGNIFICANCE TESTS + EFFECT SIZE vs. BASELINE
# Seeds are shared across configs, so each config's per-seed objective
# pairs 1:1 with the baseline's run on that same seed:
#   - paired t-test + Cohen's dz  (parametric)
#   - Wilcoxon signed-rank + matched-pairs rank-biserial correlation
#     (non-parametric, robust to the objective's skew from penalties)
# ============================================================

sig_df = stats_utils.paired_significance(raw_df, "config", "seed", "objective", baseline_name)
sig_csv = os.path.join(out_dir, "significance_tests.csv")
sig_df.to_csv(sig_csv, index=False)

# ============================================================
# EXCEL REPORT -- everything in one workbook
# ============================================================

xlsx_path = os.path.join(out_dir, "replications_report.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    agg_df.to_excel(writer, sheet_name="summary_stats", index=False)
    sig_df.to_excel(writer, sheet_name="significance_tests", index=False)
    raw_df.to_excel(writer, sheet_name="raw_per_seed", index=False)

# ============================================================
# BOX + VIOLIN PLOTS -- objective distribution per configuration
# ============================================================

cfg_names = list(configs.keys())
data_by_cfg = [raw_df[raw_df["config"] == c]["objective"].values for c in cfg_names]
colors = plt.cm.tab10(np.linspace(0, 1, max(len(cfg_names), 2)))

fig, axes = plt.subplots(1, 2, figsize=(3.2 * len(cfg_names) + 3, 5))

bp = axes[0].boxplot(data_by_cfg, patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for i, vals in enumerate(data_by_cfg, start=1):
    axes[0].scatter(np.random.normal(i, 0.03, size=len(vals)), vals,
                     alpha=0.35, s=10, color="black")
axes[0].set_xticks(range(1, len(cfg_names) + 1))
axes[0].set_xticklabels(cfg_names, rotation=15, ha="right")
axes[0].set_ylabel("Objective value")
axes[0].set_title("Box plot")

vp = axes[1].violinplot(data_by_cfg, showmeans=True, showmedians=True)
for pc, c in zip(vp["bodies"], colors):
    pc.set_facecolor(c)
    pc.set_alpha(0.7)
axes[1].set_xticks(range(1, len(cfg_names) + 1))
axes[1].set_xticklabels(cfg_names, rotation=15, ha="right")
axes[1].set_title("Violin plot")

fig.suptitle(f"Objective distribution across {len(seeds)} seeds -- {instance_name}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(out_dir, "objective_box_violin.png")
fig.savefig(fig_path, dpi=200)
plt.close(fig)

# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n" + "=" * 90)
print(f"REPLICATION REPORT -- {instance_name}  "
      f"({len(seeds)} seeds/config, {len(configs)} configs, total wall time {total_elapsed/60:.1f} min)")
print("=" * 90)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

print("\n--- Descriptive statistics: objective (mean/std, median/IQR, 95% CI, min/max) ---")
print(agg_df[agg_df["metric"] == "objective"].to_string(index=False))

print("\n--- Runtime (seconds per run) ---")
print(agg_df[agg_df["metric"] == "wall_time_sec"].to_string(index=False))

if "iterations_run" in METRICS:
    print("\n--- Iterations actually run before stopping ---")
    print(agg_df[agg_df["metric"] == "iterations_run"].to_string(index=False))

print("\n--- Feasibility rate ---")
print(agg_df[agg_df["metric"] == "feasible_rate"][["config", "mean"]]
      .rename(columns={"mean": "feasible_rate"}).to_string(index=False))

print("\n--- EMA-triggered stop rate (vs. hitting the --iterations ceiling) ---")
print(agg_df[agg_df["metric"] == "ema_stop_rate"][["config", "mean"]]
      .rename(columns={"mean": "ema_stop_rate"}).to_string(index=False))

print(f"\n--- Best configuration ---")
print(f"  {best_config}  (mean objective = {best_mean_obj:,.2f})")

print(f"\n--- Significance / effect size vs. baseline ({baseline_name!r}, paired by seed) ---")
if len(sig_df):
    print(sig_df.to_string(index=False))
else:
    print("  Only one config was run -- nothing to compare against.")

print(f"\nRaw per-seed results   : {raw_csv}")
print(f"Aggregate stats CSV    : {agg_csv}")
print(f"Significance tests CSV : {sig_csv}")
print(f"Excel report           : {xlsx_path}")
print(f"Box/violin plot        : {fig_path}")
