"""QoS x sharing-eligibility interaction tests reported in Section 6.5.

For each instance and sharing setting, computes the change caused by removing
the QoS layer (no_qos minus with_qos) and tests, with a two-sided Wilcoxon
signed-rank test, whether that change differs between high and low sharing
eligibility. Also prints the aggregate per-served-patient values.
Reads paper_figures/sharing_noqos_matched/raw_matched.csv.
"""
import os

import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
CONV, GAMMA = 0.875, 100.0  # cost units per minute; violation penalty

raw = pd.read_csv(os.path.join(HERE, "paper_figures", "sharing_noqos_matched", "raw_matched.csv"))


def arm(sharing, qos):
    return raw[(raw.sharing == sharing) & (raw.qos == qos)].set_index("rep").sort_index()


arms = {(s, q): arm(s, q) for s in ("low", "high") for q in ("with_qos", "no_qos")}

print("Aggregate values per served patient (minutes)")
for s in ("low", "high"):
    for q in ("with_qos", "no_qos"):
        d = arms[(s, q)]
        n = d.served.sum()
        print(f"  {s:<4} {q:<8}  travel {d.travel_cost.sum() / CONV / n:6.2f}  "
              f"ride {d.ride_penalty.sum() / n:6.2f}  violation {d.violation_cost.sum() / GAMMA / n:6.2f}  "
              f"served ratio {(d.served / d.num_patients).mean():.2f}")

print("\nInteraction tests: change from removing QoS, high vs. low eligibility")
ride_per_served = {
    s: arms[(s, "no_qos")].ride_penalty / arms[(s, "no_qos")].served
    - arms[(s, "with_qos")].ride_penalty / arms[(s, "with_qos")].served
    for s in ("low", "high")
}
print(f"  ride time per served patient: low {ride_per_served['low'].mean():.2f}, "
      f"high {ride_per_served['high'].mean():.2f} min, "
      f"Wilcoxon p = {wilcoxon(ride_per_served['high'], ride_per_served['low']).pvalue:.4g}")
for metric in ("ride_penalty", "travel_cost", "violation_cost", "served"):
    diff = {s: arms[(s, "no_qos")][metric] - arms[(s, "with_qos")][metric] for s in ("low", "high")}
    print(f"  {metric:<15}: Wilcoxon p = {wilcoxon(diff['high'], diff['low']).pvalue:.4g}")
