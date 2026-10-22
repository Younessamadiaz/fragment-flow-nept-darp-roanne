# Builds LaTeX rows for Table 8 (size-20 full-QoS ablation) and Table 10
# (sharing x full-QoS) from the no-QoS experiment outputs.
# Units: objective and travel cost / 0.875 -> time-equivalent minutes;
# ride_penalty is already minutes; violation_cost / VIOLATION_PENALTY -> xi.
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "paper_figures")
CONV, GAMMA = 0.875, 100.0
ROWS = [("Objective value", "objective", CONV), ("Travel time", "travel_cost", CONV),
        ("Ride time", "ride_penalty", 1.0), ("Violation", "violation_cost", GAMMA),
        ("Served patients", "served", 1.0)]


def fmt(x):
    return f"{x:,.2f}"


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def block(summary_csv, sig_csv, base, new):
    agg = pd.read_csv(summary_csv)
    sig = pd.read_csv(sig_csv)
    grp = "config" if "config" in agg.columns else "qos"
    out = []
    for label, metric, div in ROWS:
        a = agg[(agg[grp] == base) & (agg.metric == metric)].iloc[0]
        b = agg[(agg[grp] == new) & (agg.metric == metric)].iloc[0]
        pct = 100 * (b["mean"] - a["mean"]) / a["mean"] if a["mean"] else float("nan")
        p = sig[sig.metric == metric]["wilcoxon_pvalue"].iloc[0]
        med = (a["median"] / div, b["median"] / div)
        out.append(dict(label=label, cells=[fmt(a["mean"] / div), fmt(a["std"] / div),
                                            fmt(b["mean"] / div), fmt(b["std"] / div),
                                            f"${pct:+.2f}\\%$", fmt_p(p)], median=med))
    return out


def show(title, rows):
    print(f"\n% ---- {title} ----")
    for r in rows:
        print(f"{r['label']} & " + " & ".join(r["cells"]) + " \\\\")
    print("% medians (with, without): " +
          "; ".join(f"{r['label']}: {r['median'][0]:.2f} / {r['median'][1]:.2f}" for r in rows))


d8 = os.path.join(PF, "noqos_ablation", "size_20")
show("Table 8 (size 20)", block(os.path.join(d8, "qos_ablation_summary.csv"),
                                os.path.join(d8, "qos_ablation_significance.csv"),
                                "with_qos", "no_qos"))

d10 = os.path.join(PF, "sharing_noqos_matched")
if os.path.exists(os.path.join(d10, "summary_high.csv")):
    low = block(os.path.join(d10, "summary_low.csv"), os.path.join(d10, "significance_low.csv"),
                "with_qos", "no_qos")
    high = block(os.path.join(d10, "summary_high.csv"), os.path.join(d10, "significance_high.csv"),
                 "with_qos", "no_qos")
    print("\n% ---- Table 10 (low | high) ----")
    for l, h in zip(low, high):
        print(f"{l['label']}\n& " + " & ".join(l["cells"]) + "\n& " + " & ".join(h["cells"]) + " \\\\")
    show("Table 10 low (medians)", low)
    show("Table 10 high (medians)", high)
else:
    print("\nTable 10 outputs not ready yet.")
