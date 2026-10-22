#!/bin/bash
# Reproduces Table 9 (value of fragment connectivity, size 20): the same instances,
# seeds, and number of shifts, with all intra- and inter-shift transitions disabled
# (darp_lns.py --disable_transitions). The "with transitions" arm is read from
# paper_figures/stochastic_case_study/raw_size_20.csv. Run from the repository root.
set -e
export MPLBACKEND=Agg

python run_qos_ablation.py \
  --with_qos_csv paper_figures/stochastic_case_study/raw_size_20.csv \
  --out_dir paper_figures/transitions_ablation/size_20 \
  --label "size=20, transitions off" --baseline_name transitions_on \
  --extra_args=--disable_transitions --config_name transitions_off
