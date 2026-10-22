# ============================================================
# STOCHASTIC CASE-STUDY EXPERIMENT
#
# Unlike run_replications.py (same instance, N different LNS seeds -- it
# measures the heuristic's OWN randomness), this script measures instance-
# level uncertainty: for each nominal size in --sizes (default 20/50/75/100
# patients), it draws --n_reps (default 35) *different* instances -- each
# with its own random number of patients, vehicles, and hospitals -- solves
# every one with darp_lns.py, and reports the mean/std of the results
# across those draws. The LNS seed itself is held FIXED across all
# replications (see --lns_seed) so the reported variability is caused by
# the stochastic instance parameters, not by the search's own randomness.
#
# Distributions (all explicit assumptions, not fitted constants -- there is
# no real data on how patient/vehicle/hospital counts vary day to day):
#   n_patients  ~ Poisson(lambda = nominal_size)
#   n_vehicles  ~ Poisson(lambda = max(3, round(0.25 * nominal_size))),
#                 then floored at ceil(nominal_size / 8) -- 0.25 matches the
#                 most generously-staffed real E_ instances (20 patients / 5
#                 vehicles); the floor matches the leanest real ratio that
#                 still worked (75 patients / 10 vehicles) so an unlucky low
#                 draw never starves the fleet below what's already been
#                 shown to work with this LNS's multi-shift growth. This
#                 makes fleet size unlikely to be *the* reason patients go
#                 unserved -- it does not, and cannot, guarantee zero drops,
#                 since time-window/ride-time/capacity limits can still
#                 leave a few patients unserved regardless of fleet size.
#   n_hospitals ~ {1: 0.45, 2: 0.35, 3: 0.20} categorical, matching the
#                 roughly 55/45 real split between 1- and 2-hospital E_
#                 instances, with a modest tail added for the new (fully
#                 synthetic -- see generate_instances.py) 3-hospital case.
#
# Usage (run one size at a time to review results incrementally):
#   python run_stochastic_case_study.py --sizes 20
#   python run_stochastic_case_study.py --sizes 50
#   python run_stochastic_case_study.py --sizes 75
#   python run_stochastic_case_study.py --sizes 100
# Each call updates the cumulative master table from every size run so far
# (found on disk under --out_dir), so results accumulate across calls.
# ============================================================

import argparse
import json
import math
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

import generate_instances as gi
import stats_utils
from stats_utils import describe

# ---- locate darp_lns.py (same robust lookup pattern as run_replications.py) ----
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
    raise FileNotFoundError(f"Can't find darp_lns.py -- looked in {HERE!r}.")

HOSPITAL_PROBS = {1: 0.45, 2: 0.35, 3: 0.20}   # see module docstring

_parser = argparse.ArgumentParser()
_parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 75, 100],
                      help="Nominal patient-count buckets to run.")
_parser.add_argument("--n_reps", type=int, default=35,
                      help="Number of stochastic instance draws PER size bucket.")
_parser.add_argument("--seed_start", type=int, default=100000,
                      help="Base seed for drawing instance parameters (n_patients/"
                           "n_vehicles/n_hospitals) and instance content; deterministic "
                           "per (bucket, replicate) so reruns are reproducible.")
_parser.add_argument("--lns_seed", type=int, default=42,
                      help="FIXED LNS seed used for every single replication, so the "
                           "measured mean/std reflects instance-parameter variability "
                           "only, not the heuristic's own seed-to-seed noise.")
_parser.add_argument("--iterations", type=int, default=5000)
_parser.add_argument("--ema_fast_window", type=int, default=30)
_parser.add_argument("--ema_slow_window", type=int, default=150)
_parser.add_argument("--ema_eps", type=float, default=0.001)
_parser.add_argument("--ema_patience", type=int, default=1500)
_parser.add_argument("--temperature", type=float, default=500.0)
_parser.add_argument("--cooling", type=float, default=0.9999)
_parser.add_argument("--remove_ratio", type=float, default=0.10)
_parser.add_argument("--time_budget_sec", type=float, default=None,
                      help="If given, passed through to darp_lns.py's own "
                           "--time_budget_sec: hard wall-clock cap per replication, "
                           "whichever stop condition (EMA/iterations/time) fires first "
                           "wins. Use this to bound the WORST case per run -- unlike "
                           "ema_patience, which only reduces the AVERAGE, a slow/stubborn "
                           "instance can still burn the full --iterations ceiling for "
                           "hours without this cap.")
_parser.add_argument("--vehicle_ratio", type=float, default=0.3,
                      help="Mean vehicles = round(vehicle_ratio * nominal_size). History: "
                           "0.25 initially -> 0.4 after size=20 showed low vehicle draws "
                           "(3-4) driving most unserved patients/infeasibility -> pulled "
                           "back to 0.3 after size=50 (at 0.4) hit 0.99 served but 0%% "
                           "feasible-rate -- user prefers trading a little service rate "
                           "back for a real chance at some feasible runs. 100%% service is "
                           "still not guaranteed regardless of this setting.")
_parser.add_argument("--vehicle_floor_ratio", type=float, default=0.15,
                      help="Hard floor on drawn vehicles = ceil(vehicle_floor_ratio * "
                           "nominal_size), so a low Poisson draw can't starve the fleet. "
                           "History: 1/8 -> 0.2 -> 0.15 alongside --vehicle_ratio.")
_parser.add_argument("--capacity", type=int, default=3,
                      help="Per-vehicle capacity Q_k, applied uniformly to all three "
                           "vehicle types (real E_ instances also use Q_k=3 for every "
                           "type -- types differ by compatibility/fleet count, not "
                           "capacity, so a uniform override stays realistic).")
_parser.add_argument("--omega", type=float, default=None,
                      help="Override Omega (max early-arrival allowance, minutes). "
                           "Default None keeps the reference file's value (45 min).")
_parser.add_argument("--psi", type=float, default=None,
                      help="Override Psi (max return-pickup delay allowance, minutes), "
                           "applied uniformly to every patient. Default None keeps the "
                           "reference file's value (35 min).")
_parser.add_argument("--sharing_rate", type=float, default=None,
                      help="If given, overrides the fitted per-activity-type sharing "
                           "eligibility rate with a single uniform P(sharing_possible=1) "
                           "for every patient -- e.g. 0.1 for mostly-exclusive-ride "
                           "instances, 0.9 for mostly-shareable ones. Default None keeps "
                           "the fitted real-data rates (5-10%% non-sharing, dialysis/"
                           "therapeutic only).")
_parser.add_argument("--service_date", type=str, default="2026-09-01")
_parser.add_argument("--reference", type=str, default=gi.DEFAULT_REFERENCE)
_parser.add_argument("--out_dir", type=str,
                      default=os.path.join(HERE, "paper_figures", "stochastic_case_study"))
_parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
_parser.add_argument("--force", action="store_true",
                      help="Re-run replications even if summary.json already exists.")
_args, _ = _parser.parse_known_args()

instances_root = os.path.join(_args.out_dir, "instances")
runs_root      = os.path.join(_args.out_dir, "runs")
os.makedirs(_args.out_dir, exist_ok=True)

print(f"Sizes         : {_args.sizes}")
print(f"Reps/size     : {_args.n_reps}")
print(f"LNS seed     : {_args.lns_seed}  (FIXED across all replications)")
print(f"Iteration cap : {_args.iterations}  (EMA stop: fast={_args.ema_fast_window} "
      f"slow={_args.ema_slow_window} eps={_args.ema_eps} patience={_args.ema_patience})")
print(f"LNS params   : temperature={_args.temperature}  cooling={_args.cooling}  "
      f"remove_ratio={_args.remove_ratio}")
print(f"Workers       : {_args.workers}  (of {os.cpu_count()} CPUs)")
print(f"Output dir    : {_args.out_dir}\n")

print(f"Loading reference tables from: {_args.reference}")
_ref = gi._load_reference(_args.reference)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"


def draw_instance_params(bucket_size, rep):
    """Deterministic per (bucket_size, rep). Returns (n_patients, n_vehicles,
    n_hospitals, content_seed)."""
    draw_seed = _args.seed_start + bucket_size * 1000 + rep
    rng = np.random.RandomState(draw_seed)

    n_patients = max(1, int(rng.poisson(bucket_size)))

    veh_mean = max(3, round(_args.vehicle_ratio * bucket_size))
    n_vehicles = max(1, int(rng.poisson(veh_mean)))
    veh_floor = max(1, math.ceil(_args.vehicle_floor_ratio * bucket_size))
    n_vehicles = max(n_vehicles, veh_floor)

    u = rng.random()
    if u < HOSPITAL_PROBS[1]:
        n_hospitals = 1
    elif u < HOSPITAL_PROBS[1] + HOSPITAL_PROBS[2]:
        n_hospitals = 2
    else:
        n_hospitals = 3

    content_seed = draw_seed + 500000
    return n_patients, n_vehicles, n_hospitals, content_seed


def run_one(bucket_size, rep):
    n_patients, n_vehicles, n_hospitals, content_seed = draw_instance_params(bucket_size, rep)

    inst_out_root = os.path.join(instances_root, f"size_{bucket_size}")
    inst_path, n_patients_actual, eta_total = gi.generate_instance(
        n_patients, n_vehicles, n_hospitals, rep, content_seed,
        _args.service_date, _ref, inst_out_root,
        capacity=_args.capacity, sharing_rate=_args.sharing_rate,
        omega_override=_args.omega, psi_override=_args.psi,
    )

    run_dir = os.path.join(runs_root, f"size_{bucket_size}", f"rep_{rep}")
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path) and not _args.force:
        with open(summary_path) as f:
            summ = json.load(f)
        elapsed = summ.get("wall_time_sec", float("nan"))
        status = "skip"
    else:
        cmd = [
            sys.executable, SCRIPT,
            "--instance", inst_path,
            "--seed", str(_args.lns_seed),
            "--iterations", str(_args.iterations),
            "--ema_fast_window", str(_args.ema_fast_window),
            "--ema_slow_window", str(_args.ema_slow_window),
            "--ema_eps", str(_args.ema_eps),
            "--ema_patience", str(_args.ema_patience),
            "--temperature", str(_args.temperature),
            "--cooling", str(_args.cooling),
            "--remove_ratio", str(_args.remove_ratio),
            "--export_dir", run_dir,
        ]
        if _args.time_budget_sec is not None:
            cmd += ["--time_budget_sec", str(_args.time_budget_sec)]
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        elapsed = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Run size={bucket_size} rep={rep} failed (exit {result.returncode}):\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        if not os.path.exists(summary_path):
            raise RuntimeError(
                f"Run size={bucket_size} rep={rep} exited 0 but never wrote {summary_path!r}."
            )
        with open(summary_path) as f:
            summ = json.load(f)
        summ["wall_time_sec"] = elapsed
        with open(summary_path, "w") as f:
            json.dump(summ, f, indent=2)
        status = "done"

    row = dict(
        bucket_size=bucket_size, rep=rep,
        n_patients_drawn=n_patients, n_vehicles_drawn=n_vehicles, n_hospitals_drawn=n_hospitals,
        n_patients_generated=n_patients_actual, n_vehicles_generated=eta_total,
        instance_path=inst_path, wall_time_sec=elapsed,
    )
    row.update(summ)
    return bucket_size, rep, row, status, elapsed


# ============================================================
# RUN ALL (bucket, rep) JOBS FOR THE REQUESTED SIZES
# ============================================================

jobs = [(b, r) for b in _args.sizes for r in range(1, _args.n_reps + 1)]
rows = []
completed = 0
t_all_start = time.time()

with ThreadPoolExecutor(max_workers=_args.workers) as pool:
    futures = {pool.submit(run_one, b, r): (b, r) for b, r in jobs}
    for future in as_completed(futures):
        b, r, row, status, elapsed = future.result()
        rows.append(row)
        completed += 1
        tag = "[skip]" if status == "skip" else f"time={elapsed:6.1f}s"
        print(f"[{completed:4d}/{len(jobs)}] size={b:<4d} rep={r:<3d} "
              f"n_pat={row['n_patients_generated']:<4d} n_veh={row['n_vehicles_generated']:<3d} "
              f"n_hosp={row['n_hospitals_drawn']:<2d} "
              f"obj={row['objective']:10.2f}  served={row['served']}/{row['num_patients']}  "
              f"feasible={str(row['feasible']):<5}  {tag}")

total_elapsed = time.time() - t_all_start
rows.sort(key=lambda r: (r["bucket_size"], r["rep"]))

# ============================================================
# PER-SIZE RAW + SUMMARY (this invocation's sizes only)
# ============================================================

METRICS = [
    "objective", "travel_cost", "ride_penalty", "violation_cost",
    "served", "num_patients", "n_patients_generated", "n_vehicles_generated",
    "n_hospitals_drawn", "wall_time_sec", "iterations_run",
]

this_run_df = pd.DataFrame(rows)
for b in _args.sizes:
    sub = this_run_df[this_run_df["bucket_size"] == b]
    sub.to_csv(os.path.join(_args.out_dir, f"raw_size_{b}.csv"), index=False)

# ============================================================
# REBUILD CUMULATIVE MASTER TABLE FROM EVERY raw_size_*.csv ON DISK
# (so calling this script size-by-size accumulates results)
# ============================================================

all_raw_paths = sorted(
    (p for p in os.listdir(_args.out_dir) if p.startswith("raw_size_") and p.endswith(".csv")),
    key=lambda p: int(p[len("raw_size_"):-len(".csv")]),
)
master_raw_df = pd.concat(
    [pd.read_csv(os.path.join(_args.out_dir, p)) for p in all_raw_paths], ignore_index=True
)
master_raw_csv = os.path.join(_args.out_dir, "raw_all_sizes.csv")
master_raw_df.to_csv(master_raw_csv, index=False)

metrics_present = [m for m in METRICS if m in master_raw_df.columns]
agg_df = stats_utils.aggregate_stats(master_raw_df, "bucket_size", metrics_present)
agg_csv = os.path.join(_args.out_dir, "summary_by_size.csv")
agg_df.to_csv(agg_csv, index=False)

# ---- one-row-per-size master table (the paste-into-the-paper table) ----
master_rows = []
for b in sorted(master_raw_df["bucket_size"].unique()):
    sub = agg_df[agg_df["bucket_size"] == b].set_index("metric")

    def get(metric, col):
        return sub.loc[metric, col] if metric in sub.index else float("nan")

    raw_sub = master_raw_df[master_raw_df["bucket_size"] == b]
    master_rows.append(dict(
        nominal_size=b, n_reps=len(raw_sub),
        n_patients_mean=get("n_patients_generated", "mean"), n_patients_std=get("n_patients_generated", "std"),
        n_vehicles_mean=get("n_vehicles_generated", "mean"), n_vehicles_std=get("n_vehicles_generated", "std"),
        n_hospitals_mean=get("n_hospitals_drawn", "mean"), n_hospitals_std=get("n_hospitals_drawn", "std"),
        objective_mean=get("objective", "mean"), objective_std=get("objective", "std"),
        served_mean=get("served", "mean"), served_std=get("served", "std"),
        served_ratio_mean=(raw_sub["served"] / raw_sub["num_patients"]).mean(),
        violation_cost_mean=get("violation_cost", "mean"), violation_cost_std=get("violation_cost", "std"),
        feasible_rate=raw_sub["feasible"].mean() if "feasible" in raw_sub.columns else float("nan"),
        wall_time_sec_mean=get("wall_time_sec", "mean"),
    ))

master_df = pd.DataFrame(master_rows)
master_csv = os.path.join(_args.out_dir, "master_table.csv")
master_xlsx = os.path.join(_args.out_dir, "master_table.xlsx")
master_df.to_csv(master_csv, index=False)
with pd.ExcelWriter(master_xlsx, engine="openpyxl") as writer:
    master_df.to_excel(writer, sheet_name="master_table", index=False)
    agg_df.to_excel(writer, sheet_name="summary_by_size", index=False)
    master_raw_df.to_excel(writer, sheet_name="raw_all_sizes", index=False)

# ============================================================
# PLOT -- objective and served-ratio vs. nominal size, with std error bars
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].errorbar(master_df["nominal_size"], master_df["objective_mean"],
                  yerr=master_df["objective_std"], fmt="o-", capsize=4, color="steelblue")
axes[0].set_xlabel("Nominal case-study size (patients)")
axes[0].set_ylabel("Objective")
axes[0].set_title("Mean objective \u00b1 std across stochastic draws")

axes[1].errorbar(master_df["nominal_size"], master_df["served_ratio_mean"],
                  yerr=master_df["served_std"] / master_df["n_patients_mean"],
                  fmt="o-", capsize=4, color="seagreen")
axes[1].set_xlabel("Nominal case-study size (patients)")
axes[1].set_ylabel("Served / total patients")
axes[1].set_title("Mean service ratio across stochastic draws")

fig.suptitle(f"Stochastic case study -- {_args.n_reps} draws/size, LNS seed fixed at {_args.lns_seed}")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig_path = os.path.join(_args.out_dir, "stochastic_case_study.png")
fig.savefig(fig_path, dpi=200)
plt.close(fig)

# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n" + "=" * 100)
print(f"STOCHASTIC CASE STUDY -- sizes {_args.sizes} run this call "
      f"({len(jobs)} runs, {total_elapsed/60:.1f} min)")
print("=" * 100)

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("\n--- This call's raw results ---")
print(this_run_df[["bucket_size", "rep", "n_patients_generated", "n_vehicles_generated",
                    "n_hospitals_drawn", "objective", "served", "num_patients",
                    "feasible", "wall_time_sec"]].to_string(index=False))

print(f"\n--- Cumulative master table (every size found under {_args.out_dir}) ---")
print(master_df.to_string(index=False))

print(f"\nPer-size raw CSVs      : {_args.out_dir}\\raw_size_<N>.csv")
print(f"Cumulative raw CSV     : {master_raw_csv}")
print(f"Per-size summary stats : {agg_csv}")
print(f"Master table CSV       : {master_csv}")
print(f"Master table Excel     : {master_xlsx}")
print(f"Plot                   : {fig_path}")
