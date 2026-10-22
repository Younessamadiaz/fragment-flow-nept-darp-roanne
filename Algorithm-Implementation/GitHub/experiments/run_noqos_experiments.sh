#!/bin/bash
# Reproduces Table 8 (full-QoS ablation, size 20) and Table 11 (sharing x QoS).
# The "without QoS" arm runs darp_lns.py --no_qos: lambda = 0 and no penalty on
# excess ride time, early arrival, or return delay during the search; the best
# solution is re-evaluated under the full objective before reporting.
# The "with QoS" arms are read from the existing case-study and sharing results.
# Run from the repository root.
set -e
export MPLBACKEND=Agg

python run_qos_ablation.py \
  --with_qos_csv paper_figures/stochastic_case_study/raw_size_20.csv \
  --out_dir paper_figures/noqos_ablation/size_20 \
  --label "size=20, full QoS removed" \
  --extra_args=--no_qos --config_name no_qos

python run_sharing_qos_matched.py --qos_mode no_qos

python make_tables_8_10.py
python compute_sharing_interaction.py
