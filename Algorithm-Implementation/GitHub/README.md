# Fragment-flow multi-shift DARP for Non-Emergency Patient Transportation

Code and data for:

> Khorshidvand, B., Amadiaz, Y., Haddad, H., Gilani Larimi, N., Nait Sidi Moh, A., Guitouni, A.
> *Fragment Connectivity in Non-Emergency Patient Transportation: A Multi-Shift Dial-a-Ride
> Framework with Quality of Service.*

## Contents

| Path | What it is |
|---|---|
| `darp_core.py` | Instance loader, solution representation, evaluation (objective, QoS violations, transitions). Shared by all algorithms. |
| `darp_lns.py` | Proposed Large Neighborhood Search (LNS) with intra-shift and inter-shift transitions (see note below) |
| `exact_mip.py` | Exact MILP model (Pyomo; Gurobi or CBC) |
| `sa_darp.py`, `ts_darp.py`, `vns_darp.py` | Benchmark metaheuristics (SA, TS, VNS) |
| `generate_instances.py` | Synthetic Roanne instance generator (uses `instances/E_20_5_2` as the reference template) |
| `run_*.py`, `stats_utils.py` | Experiment drivers and statistics (Friedman, Wilcoxon + Holm) |
| `make_tables_8_10.py`, `compute_sharing_interaction.py`, `make_supplement_tables.py` | Build Tables 8 and 11, the QoS × sharing interaction tests (Sec. 6.5), and supplement Tables S3–S6 and S8 from the result CSVs |
| `build_tuning_report.py`, `plot_parameter_sensitivity.py`, `generate_paper_figures.py` | Figures |
| `experiments/` | Instance lists and shell scripts for the benchmark, exact-validation, no-QoS, and transitions-off runs |
| `instances/` | 30 algorithm-comparison instances + reference template |
| `instances_exact/` | 5 exact-validation instances |
| `paper_figures/` | Results behind every table, plus the 140 case-study instances |

Instance names follow `<prefix>_<patients>_<vehicles>_<index>`.

### Note on the solution method

The paper's method is an LNS with simulated-annealing acceptance and a single random-removal
destroy operator (`darp_lns.py`). The script also contains an optional adaptive
operator-selection layer (`--adaptive`: roulette-wheel choice among random, worst, and related
removal), which is **off by default and was not used in any reported experiment**.

LNS settings used in the paper:

| Experiment | Destroy ratio ρ | Removal cap | Cooling α | θ₀ | Stopping rule |
|---|---|---|---|---|---|
| Algorithm comparison (Tables 5–6, S2) | 0.20 (script default) | 4 | 0.9995 (script default) | 500 | 300 s per run |
| Case study and ablations (Tables 7–11, S3–S8) | 0.10 | – | 0.9999 | 500 | 5,000 iterations or EMA stagnation (ε = 0.001, 1,500 iterations) |

All result values in the CSVs are in raw units: `travel_cost` and `objective` in cost units
(divide by 0.875 for time-equivalent minutes), `ride_penalty` in minutes, and `violation_cost`
equal to 100 × ξ (divide by 100 for violation minutes plus excess-passenger counts, as reported
in the paper).

## Setup

```bash
pip install -r requirements.txt
```

`exact_mip.py` needs Gurobi (academic licence) or CBC. Run every command from the repository root.

## Reproducing the paper

| Paper | Command | Results in |
|---|---|---|
| Sec. 5.2, Table S1, Figs S6–S7 (tuning) | `python run_parameter_tuning.py --instance instances/E_20_5_1/E_20_5_1_model_input.xlsx` (repeat for the 50/75 tuning instances), then `python build_tuning_report.py` | `paper_figures/tuning/` |
| Table 4 (exact validation) | `bash experiments/run_exact_validation.sh` | `paper_figures/exact_validation/` |
| Tables 5–6 (benchmark) | `bash experiments/run_algorithm_comparison.sh` | `paper_figures/algorithm_comparison/` |
| Table 7 (case study) | `python run_stochastic_case_study.py --sizes 20 --vehicle_ratio 0.25` (sizes 50/75/100 with ratios 0.4/0.3/0.3) | `paper_figures/stochastic_case_study/` |
| Table 8 (full-QoS ablation) | `bash experiments/run_noqos_experiments.sh` (first part) | `paper_figures/noqos_ablation/` |
| Sec. 6.2 ride-time-only ablation (λ = 0) | `python run_qos_ablation.py --with_qos_csv paper_figures/stochastic_case_study/raw_size_20.csv --out_dir paper_figures/qos_ablation/size_20 --label size=20` (and `raw_size_50.csv` → `size_50`) | `paper_figures/qos_ablation/` |
| Table S7 (λ = 2) | `python run_qos_ablation.py --with_qos_csv paper_figures/stochastic_case_study/raw_size_20.csv --out_dir paper_figures/qos_ablation/size_20_lambda2 --lambda_value 2` | `paper_figures/qos_ablation/size_20_lambda2/` |
| Table 9 (fragment connectivity: transitions on vs. off) | `bash experiments/run_transitions_ablation.sh` | `paper_figures/transitions_ablation/` |
| Table 10 (single vs. multi-shift) | `python run_qos_ablation.py --with_qos_csv paper_figures/stochastic_case_study/raw_size_20.csv --out_dir paper_figures/shift_ablation/size_20 --config_name single_shift --extra_args="--shifts 1 --max_shifts 1 --disable_shift_transitions"` | `paper_figures/shift_ablation/` |
| Table 11 (sharing × QoS) and Sec. 6.5 interaction tests | `bash experiments/run_noqos_experiments.sh` (second part) | `paper_figures/sharing_noqos_matched/` |
| Tables S3–S6, S8 | `python make_supplement_tables.py` | `supplement_tables/` |

`--extra_args` must be written with `=` (as above) because its value starts with `--`.

The ablation scripts reuse the case-study instances listed in the `instance_path` column of
`raw_size_*.csv`, so they run on exactly the instances reported in the paper. The "with QoS" arm of Table 8 and the
"with transitions" arm of Table 9 are read from `raw_size_20.csv` rather than re-run. Because the repository stores result CSVs but not
per-run folders, `run_sharing_qos_matched.py` re-runs the "with QoS" arm of Table 11 (70 runs)
before the "without QoS" arm. `paper_figures/sharing_qos_matched/` holds the earlier λ = 0
version of the sharing experiment, whose "with QoS" arm is the one reported in Table 11.

Runs are long: a single size-50 case-study instance takes 30–55 minutes on a 4-core laptop CPU.

## Licence

TODO
