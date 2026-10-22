# ============================================================
# MATCHED SHARING-RATE x QOS ABLATION (fixes the A4 instance-mismatch issue)
#
# Runs a genuine 2x2 paired design on the SAME 35 size-20 instances used by
# the main stochastic case study (paper_figures/stochastic_case_study/
# raw_size_20.csv), instead of the old low/high sharing sweep's independently
# regenerated instance populations (different patient/vehicle counts per
# arm -- confounded fleet ratio with sharing rate, per the A4 finding).
#
# Factors:
#   sharing in {low (rate=0.10), high (rate=0.90)}  -- via
#     darp_lns.py's new --sharing_rate_override, which overrides every
#     patient's sharing_possible flag with an i.i.d. draw at the given rate,
#     seeded deterministically from (instance path, rate) so it's IDENTICAL
#     across the with/without-QoS arms of the same instance and sharing rate.
#   qos in {with_qos (lambda=1, instance default), without_qos (lambda=0)}
#
# 35 instances x 2 sharing rates x 2 QoS arms = 140 darp_lns.py runs.
# ============================================================

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import stats_utils

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "darp_lns.py")
BASELINE_CSV = os.path.join(HERE, "paper_figures", "stochastic_case_study", "raw_size_20.csv")
_parser = argparse.ArgumentParser()
_parser.add_argument("--qos_mode", choices=["lambda0", "no_qos"], default="lambda0",
                      help="How the comparison arm removes QoS: 'lambda0' (original, "
                           "--lambda_override 0) or 'no_qos' (darp_lns.py --no_qos: "
                           "lambda=0 AND no ride/early/return-delay penalties, re-evaluated "
                           "under the full objective). The with_qos arm is always read from / "
                           "written to the original sharing_qos_matched/runs dir, so it is "
                           "never re-run.")
_parser.add_argument("--out_dir", type=str, default=None)
_args = _parser.parse_args()

BASE_OUT_DIR = os.path.join(HERE, "paper_figures", "sharing_qos_matched")
OUT_DIR = _args.out_dir or (BASE_OUT_DIR if _args.qos_mode == "lambda0"
                            else os.path.join(HERE, "paper_figures", "sharing_noqos_matched"))
WITH_RUNS_DIR = os.path.join(BASE_OUT_DIR, "runs")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

LNS_SEED = 42
ITERATIONS = 5000
EMA_FAST, EMA_SLOW, EMA_EPS, EMA_PATIENCE = 30, 150, 0.001, 1500
TEMPERATURE, COOLING, REMOVE_RATIO = 500.0, 0.9999, 0.10
WORKERS = max(1, (os.cpu_count() or 2) - 1)

SHARING_RATES = {"low": 0.10, "high": 0.90}
# arm name -> extra darp_lns.py args (empty = instance default lambda, full penalties)
if _args.qos_mode == "lambda0":
    QOS_ARMS = {"with_qos": [], "without_qos": ["--lambda_override", "0.0"]}
else:
    QOS_ARMS = {"with_qos": [], "no_qos": ["--no_qos"]}
NEW_ARM = [q for q in QOS_ARMS if q != "with_qos"][0]

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

df = pd.read_csv(BASELINE_CSV).sort_values("rep").reset_index(drop=True)
print(f"Loaded {len(df)} baseline size-20 instances from {BASELINE_CSV}")

jobs = []
for _, row in df.iterrows():
    rep = int(row["rep"])
    inst = os.path.join(HERE, row["instance_path"])
    for sname, srate in SHARING_RATES.items():
        for qname, lam in QOS_ARMS.items():
            jobs.append((rep, inst, sname, srate, qname, lam))

print(f"Total jobs: {len(jobs)} (35 instances x {len(SHARING_RATES)} sharing rates x "
      f"{len(QOS_ARMS)} QoS arms), {WORKERS} workers\n")


def run_one(job):
    rep, inst, sname, srate, qname, lam = job
    base = WITH_RUNS_DIR if qname == "with_qos" else RUNS_DIR
    run_dir = os.path.join(base, sname, qname, f"rep_{rep}")
    summary_path = os.path.join(run_dir, "summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summ = json.load(f)
        return dict(rep=rep, sharing=sname, sharing_rate=srate, qos=qname,
                     instance_path=inst, status="skip", **summ)

    cmd = [
        sys.executable, SCRIPT,
        "--instance", inst,
        "--seed", str(LNS_SEED),
        "--iterations", str(ITERATIONS),
        "--ema_fast_window", str(EMA_FAST),
        "--ema_slow_window", str(EMA_SLOW),
        "--ema_eps", str(EMA_EPS),
        "--ema_patience", str(EMA_PATIENCE),
        "--temperature", str(TEMPERATURE),
        "--cooling", str(COOLING),
        "--remove_ratio", str(REMOVE_RATIO),
        "--sharing_rate_override", str(srate),
        "--export_dir", run_dir,
    ]
    cmd += lam

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"job rep={rep} sharing={sname} qos={qname} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    if not os.path.exists(summary_path):
        raise RuntimeError(f"job rep={rep} sharing={sname} qos={qname} exited 0 but never wrote summary.json")

    with open(summary_path) as f:
        summ = json.load(f)
    summ["wall_time_sec"] = elapsed
    with open(summary_path, "w") as f:
        json.dump(summ, f, indent=2)

    return dict(rep=rep, sharing=sname, sharing_rate=srate, qos=qname,
                instance_path=inst, status="done", **summ)


rows = []
t_start = time.time()
completed = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(run_one, j): j for j in jobs}
    for fut in as_completed(futures):
        r = fut.result()
        rows.append(r)
        completed += 1
        elapsed = time.time() - t_start
        print(f"[{completed:3d}/{len(jobs)}] elapsed={elapsed:7.0f}s  "
              f"sharing={r['sharing']:<4s} qos={r['qos']:<11s} rep={r['rep']:<3d} "
              f"obj={r['objective']:10.2f}  served={r['served']}/{r['num_patients']}  "
              f"({r['status']})")

raw_df = pd.DataFrame(rows).sort_values(["sharing", "qos", "rep"]).reset_index(drop=True)
raw_df.to_csv(os.path.join(OUT_DIR, "raw_matched.csv"), index=False)
print(f"\nRaw results written to {os.path.join(OUT_DIR, 'raw_matched.csv')}")

metrics = ["objective", "travel_cost", "ride_penalty", "violation_cost", "served"]

for sname in SHARING_RATES:
    sub = raw_df[raw_df["sharing"] == sname].copy()

    agg = stats_utils.aggregate_stats(sub, "qos", metrics)
    agg_path = os.path.join(OUT_DIR, f"summary_{sname}.csv")
    agg.to_csv(agg_path, index=False)

    sig_rows = [stats_utils.paired_significance(sub, "qos", "rep", m, baseline="with_qos")
                .assign(metric=m) for m in metrics]
    sig_df = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    sig_path = os.path.join(OUT_DIR, f"significance_{sname}.csv")
    sig_df.to_csv(sig_path, index=False)

    print(f"\n{'='*90}\n{sname.upper()} SHARING (rate={SHARING_RATES[sname]:g}) -- summary\n{'='*90}")
    print(agg.to_string(index=False))
    print(f"\n{sname.upper()} SHARING -- paired significance ({NEW_ARM} vs with_qos baseline)")
    print(sig_df.to_string(index=False))
    print(f"\nWritten: {agg_path}\n         {sig_path}")

print("\nDone.")
