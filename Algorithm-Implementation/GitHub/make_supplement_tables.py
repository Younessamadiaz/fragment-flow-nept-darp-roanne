# Generates the data-driven LaTeX tables of the online supplement:
#   Tables S3-S6 (per-replication case-study results, violation as xi)
#   Table S8     (per-instance full-QoS ablation, size 20)
# Units: objective and travel cost / 0.875 -> time-equivalent minutes;
# ride time already in minutes; violation_cost / 100 -> xi (minutes plus
# excess-passenger counts, before weighting by gamma).
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "paper_figures")
OUT = os.path.join(HERE, "supplement_tables")
CONV, GAMMA = 0.875, 100.0
os.makedirs(OUT, exist_ok=True)

HEAD = ("Rep & Patients & Vehicles & Hosp. & Objective & Travel time & Ride time "
        "& Violation ($\\xi$) & Served (\\%) & Time (min) \\\\")


def per_rep_table(size, label, caption):
    d = pd.read_csv(os.path.join(PF, "stochastic_case_study", f"raw_size_{size}.csv")).sort_values("rep")
    lines = [
        "\\begingroup", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{0.90}", "",
        "\\begin{longtable}{rrrrrrrrrr}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}} \\\\", "\\toprule", HEAD, "\\midrule",
        "\\endfirsthead", "",
        "\\multicolumn{10}{c}{\\tablename\\ \\thetable{} -- continued} \\\\",
        "\\toprule", HEAD, "\\midrule", "\\endhead", "",
        "\\midrule", "\\multicolumn{10}{r}{continued on next page} \\\\", "\\endfoot", "",
        "\\bottomrule", "\\endlastfoot", "",
    ]
    for _, r in d.iterrows():
        lines.append(
            f"{int(r.rep)} & {int(r.num_patients)} & {int(r.n_vehicles_generated)} & "
            f"{int(r.n_hospitals_drawn)} & {r.objective / CONV:.2f} & {r.travel_cost / CONV:.2f} & "
            f"{r.ride_penalty:.2f} & {r.violation_cost / GAMMA:.2f} & "
            f"{100 * r.served / r.num_patients:.1f} & {r.wall_time_sec / 60:.1f} \\\\")
    lines += ["", "\\end{longtable}", "", "\\endgroup"]
    path = os.path.join(OUT, f"table_reps_{size}.tex")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return d


CAP_NOTE = {
    20: "", 50: "", 75: "",
    100: (" Replications 2, 3, 5, 6, 7, and 10 were solved without the two-hour search limit; "
          "all others were solved under it. Recorded times include instance loading and "
          "post-search reporting and can therefore exceed 120 minutes for capped replications."),
}
for size, lab in [(20, "tab:full-20"), (50, "tab:full-50"), (75, "tab:full-75"), (100, "tab:full-100")]:
    per_rep_table(size, lab,
                  f"Full per-replication results, nominal size $n={size}$ (35 stochastic draws)."
                  + CAP_NOTE[size])


# ---- Table S8: per-instance full-QoS ablation (size 20) ----
paired = pd.read_csv(os.path.join(PF, "noqos_ablation", "size_20", "qos_ablation_raw.csv"))
base = paired[paired.config == "with_qos"].set_index("rep").sort_index()
noq = paired[paired.config == "no_qos"].set_index("rep").sort_index()

H8 = ("& & \\multicolumn{2}{c}{\\textbf{Served}} & \\multicolumn{2}{c}{\\textbf{Travel}} "
      "& \\multicolumn{2}{c}{\\textbf{Ride}} & \\multicolumn{2}{c}{\\textbf{Violation}} \\\\\n"
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
      "Rep & Patients & With & Without & With & Without & With & Without & With & Without \\\\")
lines = [
    "\\begingroup", "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}",
    "\\renewcommand{\\arraystretch}{0.90}", "",
    "\\begin{longtable}{rrrrrrrrrr}",
    "\\caption{Per-instance results of the full-QoS ablation, nominal size $n=20$. "
    "``With'' is the case-study configuration; ``Without'' sets $\\lambda=0$ and removes the "
    "penalties on excess ride time, early arrival, and return delay from the search objective, "
    "with solutions re-evaluated under the full objective. Travel, ride time, and violation "
    "($\\xi$) are in minutes per served patient.}",
    "\\label{tab:noqos-per-instance} \\\\", "\\toprule", H8, "\\midrule", "\\endfirsthead", "",
    "\\multicolumn{10}{c}{\\tablename\\ \\thetable{} -- continued} \\\\",
    "\\toprule", H8, "\\midrule", "\\endhead", "",
    "\\midrule", "\\multicolumn{10}{r}{continued on next page} \\\\", "\\endfoot", "",
    "\\bottomrule", "\\endlastfoot", "",
]
for rep in base.index:
    a, b = base.loc[rep], noq.loc[rep]

    def per(x, col, div):
        return x[col] / div / x["served"] if x["served"] else float("nan")

    lines.append(
        f"{rep} & {int(a.num_patients)} & {int(a.served)} & {int(b.served)} & "
        f"{per(a, 'travel_cost', CONV):.2f} & {per(b, 'travel_cost', CONV):.2f} & "
        f"{per(a, 'ride_penalty', 1):.2f} & {per(b, 'ride_penalty', 1):.2f} & "
        f"{per(a, 'violation_cost', GAMMA):.2f} & {per(b, 'violation_cost', GAMMA):.2f} \\\\")
lines += ["", "\\end{longtable}", "", "\\endgroup"]
open(os.path.join(OUT, "table_noqos_per_instance.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("written to", OUT)
