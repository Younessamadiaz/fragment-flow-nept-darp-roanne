#!/bin/bash
# Reproduces Table 4 (LNS vs. exact optimum on five small instances).
# Run from the repository root.
set -e
OUT=paper_figures/exact_validation
for name in G_5_3_1 G_6_4_1 G_7_4_1 G_8_5_1 G_9_5_1; do
  inst="instances_exact/$name/${name}_model_input.xlsx"
  mkdir -p "$OUT/$name"
  # Exact model (Gurobi). 400 s limit; G_7/G_8/G_9 were re-solved with 1800 s to prove optimality.
  python exact_mip.py --instance "$inst" --solver gurobi --time_limit 1800 \
    --mip_gap 0.0001 --shifts 5 --export_dir "$OUT/$name/exact_trans5_long" \
    > "$OUT/$name/exact_trans5_long.log" 2>&1
  # LNS
  python darp_lns.py --instance "$inst" --seed 42 --iterations 30000 \
    --disable_shift_transitions --export_dir "$OUT/$name/lns_noshift" \
    > "$OUT/$name/lns_noshift.log" 2>&1
done
