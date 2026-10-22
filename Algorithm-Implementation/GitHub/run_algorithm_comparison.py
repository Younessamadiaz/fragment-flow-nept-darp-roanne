# ============================================================
# CROSS-ALGORITHM DARP BENCHMARK RUNNER
#
# Generalizes run_all_instances.py x run_replications.py with grouping key
# "config" -> "algorithm": drives LNS (darp_lns.py) and the four baselines
# (sa_darp.py, ts_darp.py, ga_darp.py, aco_darp.py) across every instance
# under --instances_root, N seeds each, all under the SAME wall-clock
# --time_budget_sec (the fair-comparison methodology decided up front --
# every algorithm gets equal search time per (instance, seed), not equal
# iteration count, since one GA generation costs far more compute than one
# LNS destroy/repair step or one SA relocate move).
#
# Reuses stats_utils.py for descriptive stats and paired significance tests
# (paired by seed WITHIN one instance -- pairing across instances isn't
# statistically meaningful).
# ============================================================

import argparse
import glob
import json
import os
import re
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

HERE = os.path.dirname(os.path.abspath(__file__))

ALGORITHM_SCRIPTS = {
    "lns": "darp_lns.py",
    "sa":   "sa_darp.py",
    "ts":   "ts_darp.py",
    "ga":   "ga_darp.py",
    "aco":  "aco_darp.py",
    "vns":  "vns_darp.py",
}

# ============================================================
# CONFIGURATION
# ============================================================
_parser = argparse.ArgumentParser()

_parser.add_argument("--instances", type=str, nargs="*", default=None,
                      help="Explicit list of *_model_input.xlsx paths. If omitted, "
                           "every instance under --instances_root matching --prefixes "
                           "is discovered automatically.")
_parser.add_argument("--instances_root", type=str,
                      default=os.path.join(HERE, "instances"),
                      help="Root folder to search for */*_model_input.xlsx (used when "
                           "--instances isn't given).")
_parser.add_argument("--prefixes", type=str, nargs="+", default=["E", "G"],
                      help="Instance-folder prefixes to include when auto-discovering "
                           "(E_... = real instances, G_... = generate_instances.py "
                           "synthetic instances). Folder names must match "
                           "<PREFIX>_<patients>_<vehicles>_<replicate>.")
_parser.add_argument("--algorithms", type=str, nargs="+",
                      default=list(ALGORITHM_SCRIPTS.keys()),
                      choices=list(ALGORITHM_SCRIPTS.keys()))
_parser.add_argument("--baseline", type=str, default="lns",
                      help="Algorithm treated as the baseline for significance tests.")
_parser.add_argument("--n_reps", type=int, default=20,
                      help="Seeds per (instance, algorithm). Same seed list is reused "
                           "across algorithms within an instance (paired design).")
_parser.add_argument("--seed_start", type=int, default=0)
_parser.add_argument("--time_budget_sec", type=float, default=60.0,
                      help="Wall-clock search budget given to EVERY algorithm, per "
                           "(instance, seed) run -- the primary fairness mechanism. "
                           "Consider scaling this up for larger instances.")
_parser.add_argument("--iterations_ceiling", type=int, default=10_000_000,
                      help="Safety ceiling passed as each script's own --iterations "
                           "(generations for GA, colony-iterations for ACO). Set high "
                           "on purpose -- --time_budget_sec is what's meant to bind.")
_parser.add_argument("--out_dir", type=str,
                      default=os.path.join(HERE, "paper_figures", "algorithm_comparison"))
_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
_parser.add_argument("--force", action="store_true",
                      help="Re-run (instance, algorithm, seed) jobs even if a summary.json "
                           "already exists for them.")
_parser.add_argument("--lns_remove_cap", type=int, default=None,
                      help="If given, passed through as darp_lns.py's own --remove_cap "
                           "(caps how many patients LNS removes per iteration, keeping "
                           "iteration cost bounded on larger instances). Only affects the "
                           "lns algorithm; other algorithms are unaffected.")

_args = _parser.parse_args()

os.makedirs(_args.out_dir, exist_ok=True)
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# ============================================================
# DISCOVER INSTANCES
# ============================================================

def discover_instances():
    paths = []
    for prefix in _args.prefixes:
        paths += glob.glob(os.path.join(_args.instances_root, f"{prefix}_*", "*_model_input.xlsx"))
    paths = [p for p in paths if not os.path.basename(p).startswith("~$")]

    def sort_key(path):
        name = os.path.basename(os.path.dirname(path))
        m = re.match(r"[A-Za-z]+_(\d+)_(\d+)_(\d+)", name)
        return tuple(int(x) for x in m.groups()) if m else (999, 999, 999)

    paths.sort(key=sort_key)
    return paths


instance_paths = _args.instances if _args.instances else discover_instances()
if not instance_paths:
    raise SystemExit("No instances found (check --instances / --instances_root / --prefixes).")

seeds = list(range(_args.seed_start, _args.seed_start + _args.n_reps))

print(f"Instances     : {len(instance_paths)}")
print(f"Algorithms    : {_args.algorithms}  (baseline = {_args.baseline!r})")
print(f"Seeds/config  : {len(seeds)} ({seeds[0]}..{seeds[-1]}, shared across algorithms)")
print(f"Time budget   : {_args.time_budget_sec}s per (instance, algorithm, seed) run")
print(f"Total runs    : {len(instance_paths) * len(_args.algorithms) * len(seeds)}")
print(f"Workers       : {_args.workers}")
print(f"Output dir    : {_args.out_dir}\n")

# ============================================================
# RUN ONE (instance, algorithm, seed) JOB
# ============================================================

def run_one(instance_path, instance_name, algorithm, seed):
    script = os.path.join(HERE, ALGORITHM_SCRIPTS[algorithm])
    run_dir = os.path.join(_args.out_dir, "runs", instance_name, algorithm, f"seed_{seed}")
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path) and not _args.force:
        with open(summary_path) as f:
            summ = json.load(f)
        return instance_name, algorithm, seed, summ, "skip", summ.get("wall_time_sec", float("nan"))

    cmd = [
        sys.executable, script,
        "--instance", instance_path,
        "--seed", str(seed),
        "--time_budget_sec", str(_args.time_budget_sec),
        "--iterations", str(_args.iterations_ceiling),
        "--export_dir", run_dir,
    ]
    if algorithm == "lns" and _args.lns_remove_cap is not None:
        cmd += ["--remove_cap", str(_args.lns_remove_cap)]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"Run instance={instance_name} algorithm={algorithm} seed={seed} failed "
            f"(exit {result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"Run instance={instance_name} algorithm={algorithm} seed={seed} exited 0 but "
            f"never wrote {summary_path!r}.\nLast 2000 chars of stdout:\n{result.stdout[-2000:]}"
        )

    with open(summary_path) as f:
        summ = json.load(f)
    summ["wall_time_sec"] = elapsed
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=2)

    return instance_name, algorithm, seed, summ, "done", elapsed


# ============================================================
# RUN EVERYTHING (instances sequential, jobs within an instance parallel)
# ============================================================

all_rows = []
t_all_start = time.time()

for idx, inst_path in enumerate(instance_paths, start=1):
    instance_name = os.path.basename(os.path.dirname(inst_path))
    print("=" * 72)
    print(f"[{idx}/{len(instance_paths)}] {instance_name}")
    print("=" * 72)

    jobs = [(algo, seed) for algo in _args.algorithms for seed in seeds]
    t0 = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=_args.workers) as pool:
        futures = {
            pool.submit(run_one, inst_path, instance_name, algo, seed): (algo, seed)
            for algo, seed in jobs
        }
        for future in as_completed(futures):
            inst_name, algo, seed, summ, status, elapsed = future.result()
            # summ's OWN "instance"/"algorithm"/"seed" fields (from each script's own
            # --export_dir meta) hold the full instance PATH and are otherwise redundant
            # with what we already know here -- apply summ first, then let our short,
            # authoritative instance_name/algo/seed win so grouping/pivoting downstream
            # (and the size-bucket regex) sees "E_20_5_1", not the full file path.
            row = dict(summ)
            row.update(instance=inst_name, algorithm=algo, seed=seed, wall_time_sec=elapsed)
            all_rows.append(row)
            completed += 1
            tag = "[skip]" if status == "skip" else f"time={elapsed:6.1f}s"
            print(f"  [{completed:3d}/{len(jobs)}] {algo:<6} seed={seed:<4d} "
                  f"obj={row.get('objective', float('nan')):12.2f}  "
                  f"served={row.get('served', '?')}/{row.get('num_patients', '?')}  "
                  f"feasible={str(row.get('feasible')):<5}  {tag}")

    print(f"  instance done in {(time.time() - t0) / 60:.1f} min\n")

total_elapsed = time.time() - t_all_start

# ============================================================
# AGGREGATE
# ============================================================

raw_df = pd.DataFrame(all_rows)
raw_df.sort_values(["instance", "algorithm", "seed"], inplace=True)
raw_csv = os.path.join(_args.out_dir, "results_raw.csv")
raw_df.to_csv(raw_csv, index=False)

METRICS = ["objective", "travel_cost", "ride_penalty", "violation_cost",
           "served", "num_violated_patients", "num_capacity_events", "wall_time_sec"]
METRICS = [m for m in METRICS if m in raw_df.columns]

summary_df = stats_utils.aggregate_stats(raw_df, ["instance", "algorithm"], METRICS)
summary_csv = os.path.join(_args.out_dir, "results_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# ---- per-instance paired significance tests vs. baseline algorithm ----
sig_frames = []
for instance_name, sub in raw_df.groupby("instance"):
    sig = stats_utils.paired_significance(sub, "algorithm", "seed", "objective", _args.baseline)
    if len(sig):
        sig.insert(0, "instance", instance_name)
        sig_frames.append(sig)
sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
sig_csv = os.path.join(_args.out_dir, "significance_tests.csv")
sig_df.to_csv(sig_csv, index=False)

# ---- best-known objective per instance (min over FEASIBLE runs; falls back
#      to the overall min if no run was feasible for that instance) ----
best_rows = []
for instance_name, sub in raw_df.groupby("instance"):
    feas = sub[sub["feasible"] == True] if "feasible" in sub.columns else sub
    pool = feas if len(feas) else sub
    best_row = pool.loc[pool["objective"].idxmin()]
    best_rows.append(dict(
        instance=instance_name, best_known_objective=best_row["objective"],
        best_known_algorithm=best_row["algorithm"], best_known_seed=best_row["seed"],
        any_feasible=bool(len(feas) > 0),
    ))
best_known_df = pd.DataFrame(best_rows)
best_known_csv = os.path.join(_args.out_dir, "best_known.csv")
best_known_df.to_csv(best_known_csv, index=False)

# ---- gap-to-best-known, per run ----
gap_df = raw_df.merge(best_known_df[["instance", "best_known_objective"]], on="instance", how="left")
gap_df["gap_pct"] = 100 * (gap_df["objective"] - gap_df["best_known_objective"]) / gap_df["best_known_objective"]


def _size_bucket(instance_name):
    m = re.match(r"[A-Za-z]+_(\d+)_", instance_name)
    n = int(m.group(1)) if m else 0
    if n <= 25:
        return "small (<=25)"
    if n <= 60:
        return "medium (26-60)"
    return "large (>60)"


gap_df["size_bucket"] = gap_df["instance"].apply(_size_bucket)
gap_csv = os.path.join(_args.out_dir, "gap_to_best_known.csv")
gap_df.to_csv(gap_csv, index=False)

# ---- master results table (instance x algorithm -> objective mean/std, feasibility, runtime) ----
obj_mean = summary_df[summary_df["metric"] == "objective"].pivot(index="instance", columns="algorithm", values="mean")
obj_std  = summary_df[summary_df["metric"] == "objective"].pivot(index="instance", columns="algorithm", values="std")
feas_rate = summary_df[summary_df["metric"] == "feasible_rate"].pivot(index="instance", columns="algorithm", values="mean") \
    if "feasible_rate" in summary_df["metric"].unique() else None
runtime_mean = summary_df[summary_df["metric"] == "wall_time_sec"].pivot(index="instance", columns="algorithm", values="mean")

master_path = os.path.join(_args.out_dir, "master_results_table.xlsx")
with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
    obj_mean.to_excel(writer, sheet_name="objective_mean")
    obj_std.to_excel(writer, sheet_name="objective_std")
    if feas_rate is not None:
        feas_rate.to_excel(writer, sheet_name="feasible_rate")
    runtime_mean.to_excel(writer, sheet_name="wall_time_mean")
obj_mean.to_csv(os.path.join(_args.out_dir, "master_results_table.csv"))

# ============================================================
# GROUPED BOXPLOT: gap-to-best-known, faceted by instance-size bucket
# ============================================================

buckets = ["small (<=25)", "medium (26-60)", "large (>60)"]
buckets = [b for b in buckets if b in gap_df["size_bucket"].unique()]
algos = _args.algorithms

if buckets:
    fig, axes = plt.subplots(1, len(buckets), figsize=(5 * len(buckets), 5), squeeze=False)
    for ax, bucket in zip(axes[0], buckets):
        sub = gap_df[gap_df["size_bucket"] == bucket]
        data = [sub[sub["algorithm"] == a]["gap_pct"].dropna().values for a in algos]
        ax.boxplot(data, labels=algos, showfliers=False)
        ax.set_title(bucket)
        ax.set_ylabel("Gap to best-known (%)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
    fig.suptitle("Objective gap to best-known-per-instance, by algorithm and instance scale")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(_args.out_dir, "gap_to_best_known_boxplot.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
else:
    fig_path = None

# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n" + "=" * 90)
print(f"ALGORITHM COMPARISON REPORT  ({len(instance_paths)} instances, {len(_args.algorithms)} "
      f"algorithms, {len(seeds)} seeds each, total wall time {total_elapsed / 3600:.2f} h)")
print("=" * 90)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("\n--- Mean objective, instance x algorithm ---")
print(obj_mean.to_string())

print("\n--- Mean gap to best-known (%), by algorithm ---")
print(gap_df.groupby("algorithm")["gap_pct"].mean().sort_values().to_string())

print(f"\nRaw per-run results     : {raw_csv}")
print(f"Aggregate stats CSV     : {summary_csv}")
print(f"Significance tests CSV  : {sig_csv}")
print(f"Best-known CSV          : {best_known_csv}")
print(f"Gap-to-best-known CSV   : {gap_csv}")
print(f"Master results table    : {master_path}")
if fig_path:
    print(f"Gap boxplot             : {fig_path}")
