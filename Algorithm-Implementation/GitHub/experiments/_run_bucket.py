import json
import os
import subprocess
import sys

bucket_file, time_budget, n_reps, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

with open(bucket_file) as f:
    instances = json.load(f)

cmd = [
    sys.executable, os.path.join(ROOT, "run_algorithm_comparison.py"),
    "--instances", *instances,
    "--algorithms", "lns", "sa", "ts", "vns",
    "--n_reps", n_reps,
    "--time_budget_sec", time_budget,
    "--out_dir", out_dir,
    "--workers", "3",
    "--lns_remove_cap", "4",
    "--force",
]
sys.exit(subprocess.call(cmd))
