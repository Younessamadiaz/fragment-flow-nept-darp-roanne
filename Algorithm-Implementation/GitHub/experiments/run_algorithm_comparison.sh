#!/bin/bash
# Reproduces Tables 5-6 and Fig. 4 (LNS vs. SA, TS, VNS; 30 instances x 3 seeds x 300 s).
# Run from the repository root.
set -e
for b in small medium large; do
  python experiments/_run_bucket.py experiments/_bucket_$b.json 300 3 paper_figures/algorithm_comparison/$b
done
