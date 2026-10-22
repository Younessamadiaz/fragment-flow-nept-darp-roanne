# ============================================================
# SYNTHETIC DARP INSTANCE GENERATOR
#
# Produces *_model_input.xlsx files matching the exact schema of the real
# "darp_roanne_E_instances" instances, at new scales/fleet sizes, so the
# algorithm comparison isn't limited to the 11 real instances (20/50/75
# patients only).
#
# Every generation rule below was fitted from the 11 real instances (see
# PHASE0_INSTANCE_CONSTANTS.md) -- not guessed. Static/structural sheets
# (global_parameters, requirement_vehicle_compatibili, sharing_rule_summary,
# math_parameter_status, model_dictionary, hospital locations, per-hospital
# demand templates) are read VERBATIM from a real reference instance rather
# than hardcoded, so they can never silently drift from the ground truth.
#
# Known assumptions (flagged, not hidden -- see PHASE0_INSTANCE_CONSTANTS.md
# "Open items"): the vehicle_type_summary eta_k split formula and the
# per-activity-type parameter distributions were fitted from a limited
# number of real instances/fleet sizes and are extrapolated here to new
# scales.
#
# Generated instances use a "G_" prefix (not "E_") so real and synthetic
# instances are never silently mixed by folder-discovery code that doesn't
# explicitly ask for both.
# ============================================================

import argparse
import ast
import math
import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REFERENCE = os.path.join(
    HERE, "instances", "E_20_5_2", "E_20_5_2_model_input.xlsx"
)   # 2-hospital instance -- gives us BOTH hospitals' demand templates in one file

DEPOT_LATLON = (46.050000, 4.070000)
SPEED_KMH    = 42.0     # fitted exactly (see PHASE0_INSTANCE_CONSTANTS.md)
COST_PER_KM  = 1.25     # fitted exactly
HOME_RADIUS_DEG_MAX = 0.16   # matches observed patient-home scatter around the depot

# ---- per-activity-type parameter tables, fitted from all 11 real instances ----

DELTA_P_FIXED = {"dialysis": 25, "oncology": 25, "other": 35, "therapeutic": 35}
# 'external' Delta_p isn't a single constant in the real data (45-55, empirical spread) --
# sampled from the observed distribution instead of fixed.
DELTA_P_EXTERNAL_CHOICES = [45, 47, 48, 50, 52, 54, 55]
DELTA_P_EXTERNAL_WEIGHTS = [11, 1, 2, 1, 1, 1, 3]

TAU_P_CHOICES = {
    "dialysis":    [240],
    "external":    [20, 25, 30],
    "oncology":    list(range(150, 181, 5)),
    "other":       list(range(60, 121, 5)),
    "therapeutic": list(range(60, 121, 5)),
}

# requirement category proportions per activity_type, fitted from all 11 instances
REQUIREMENT_PROPORTIONS = {
    "dialysis":    {"vsl_required": 0.718, "prescription_medicale": 0.282},
    "external":    {"standard_assis": 0.70, "prescription_medicale": 0.30},
    "oncology":    {"vsl_required": 0.50, "prescription_medicale": 0.292, "standard_assis": 0.208},
    "other":       {"standard_assis": 0.620, "prescription_medicale": 0.216,
                     "vsl_required": 0.129, "premium": 0.035},
    "therapeutic": {"vsl_required": 0.452, "prescription_medicale": 0.387, "standard_assis": 0.161},
}

# non-sharing rate per activity_type (sharing_possible=0 with this probability);
# activity types not listed here are always sharing_possible=1, matching real data
NON_SHARING_RATE = {"dialysis": 0.05, "therapeutic": 0.10}


def _load_reference(reference_path):
    xls = pd.ExcelFile(reference_path)
    demand = pd.read_excel(xls, "medical_demand_plan")
    return dict(
        global_parameters=pd.read_excel(xls, "global_parameters"),
        requirement_vehicle_compatibili=pd.read_excel(xls, "requirement_vehicle_compatibili"),
        sharing_rule_summary=pd.read_excel(xls, "sharing_rule_summary"),
        math_parameter_status=pd.read_excel(xls, "math_parameter_status"),
        model_dictionary=pd.read_excel(xls, "model_dictionary"),
        hospitals=pd.read_excel(xls, "hospitals"),
        demand_h1=demand[demand["hospital_id"] == "H1"].reset_index(drop=True),
        demand_h2=demand[demand["hospital_id"] == "H2"].reset_index(drop=True),
    )


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _largest_remainder_apportion(base_counts, target_total):
    base = np.asarray(base_counts, dtype=float)
    raw = base / base.sum() * target_total
    floor_alloc = np.floor(raw).astype(int)
    remainder = int(target_total - floor_alloc.sum())
    frac = raw - floor_alloc
    order = np.argsort(-frac)
    alloc = floor_alloc.copy()
    for i in order[:remainder]:
        alloc[i] += 1
    return alloc


def _weighted_choice(rng, choices, weights):
    total = sum(weights)
    r = rng.random() * total
    cum = 0.0
    for c, w in zip(choices, weights):
        cum += w
        if r <= cum:
            return c
    return choices[-1]


def _sample_requirement(rng, activity_type):
    props = REQUIREMENT_PROPORTIONS[activity_type]
    return _weighted_choice(rng, list(props.keys()), list(props.values()))


def _sample_tau(rng, activity_type):
    return rng.choice(TAU_P_CHOICES[activity_type])


def _sample_delta(rng, activity_type):
    if activity_type == "external":
        return _weighted_choice(rng, DELTA_P_EXTERNAL_CHOICES, DELTA_P_EXTERNAL_WEIGHTS)
    return DELTA_P_FIXED[activity_type]


def _sample_sharing(rng, activity_type, sharing_rate=None):
    """sharing_rate, if given, overrides the fitted per-activity-type
    NON_SHARING_RATE with a single uniform P(sharing_possible=1) applied to
    every patient regardless of activity type -- used to sweep the overall
    ride-sharing-eligible proportion for the sharing-vs-QoS experiment,
    since the fitted rates (5-10% non-sharing, only for dialysis/
    therapeutic) don't vary enough on their own to sweep."""
    if sharing_rate is not None:
        return 1 if rng.random() < sharing_rate else 0
    non_sharing_rate = NON_SHARING_RATE.get(activity_type, 0.0)
    return 0 if rng.random() < non_sharing_rate else 1


def _eta_split(total_vehicles):
    """vtc_premium=1 (constant), taxi=ceil((n-1)/2), vsl=floor((n-1)/2) --
    fitted EXACTLY from the two observed real fleet totals (5, 10).
    Extrapolated here to other totals -- an explicit assumption, see
    PHASE0_INSTANCE_CONSTANTS.md."""
    n = max(1, total_vehicles)
    vtc = 1
    remaining = n - vtc
    taxi = math.ceil(remaining / 2)
    vsl  = math.floor(remaining / 2)
    return dict(taxi_conventionne=taxi, vsl=vsl, vtc_premium=vtc)


def generate_instance(target_n_patients, target_n_vehicles, n_hospitals, replicate,
                       seed, service_date, ref, out_root, capacity=3, sharing_rate=None,
                       omega_override=None, psi_override=None):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    instance_name = f"G_{target_n_patients}_{target_n_vehicles}_{replicate}"
    out_dir = os.path.join(out_root, instance_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- demand plan (apportion target_n_patients across activity-type rows) ----
    if n_hospitals == 1:
        demand_rows = ref["demand_h1"].copy()
        hospitals_df = ref["hospitals"][ref["hospitals"]["hospital_id"] == "H1"].reset_index(drop=True)
    elif n_hospitals == 2:
        demand_rows = pd.concat([ref["demand_h1"], ref["demand_h2"]], ignore_index=True)
        hospitals_df = ref["hospitals"].reset_index(drop=True)
    elif n_hospitals == 3:
        # No real 3-hospital data exists in the source dataset (only H1/H2 are
        # observed). H3 is a SYNTHETIC third hospital -- its demand template is
        # cloned from H2's (same activity-type mix/proportions) and its location
        # is H2 mirrored across the depot, so it sits on the opposite side of the
        # existing cluster rather than on top of it. Flagged, not hidden: this is
        # an extrapolation for the stochastic case study, not a fitted constant.
        demand_h3 = ref["demand_h2"].copy()
        demand_h3["hospital_id"] = "H3"
        demand_rows = pd.concat([ref["demand_h1"], ref["demand_h2"], demand_h3], ignore_index=True)

        h2_row = ref["hospitals"][ref["hospitals"]["hospital_id"] == "H2"].iloc[0]
        h3_row = dict(
            hospital_id="H3",
            hospital_name="Synthetic Hospital 3 (mirrored, no real data)",
            city=h2_row["city"],
            lat=2 * DEPOT_LATLON[0] - h2_row["lat"],
            lon=2 * DEPOT_LATLON[1] - h2_row["lon"],
        )
        hospitals_df = pd.concat(
            [ref["hospitals"].reset_index(drop=True), pd.DataFrame([h3_row])], ignore_index=True
        )
    else:
        raise ValueError("n_hospitals must be 1, 2, or 3 (H3 is synthetic -- see comment above)")

    alloc = _largest_remainder_apportion(demand_rows["base_n_patients"].values, target_n_patients)
    demand_rows = demand_rows.copy()
    demand_rows["n_patients"] = alloc
    demand_rows["selected_in_instance"] = 1

    # ---- patients ----
    patients = []
    pid_counter = 1
    for _, row in demand_rows.iterrows():
        h = row["hospital_id"]
        act = row["activity_type"]
        n = int(row["n_patients"])
        appt_times = ast.literal_eval(row["appointment_times"]) if isinstance(row["appointment_times"], str) \
            else row["appointment_times"]
        for _ in range(n):
            patient_id = f"P{pid_counter}"
            pid_counter += 1
            a_p_hhmm = rng.choice(appt_times)
            hh, mm = a_p_hhmm.split(":")
            a_p = int(hh) * 60 + int(mm)
            patients.append(dict(
                patient_id=patient_id, hospital_id=h, activity_type=act,
                requirement=_sample_requirement(rng, act),
                a_p=a_p, a_p_hhmm=a_p_hhmm,
                a_p_datetime=f"{service_date} {a_p_hhmm}:00",
                tau_p=int(_sample_tau(rng, act)),
                Psi_p=int(psi_override) if psi_override is not None
                      else int(ref["global_parameters"].set_index("symbol")["value"]["Psi"]),
                Delta_p=int(_sample_delta(rng, act)),
                sharing_possible=_sample_sharing(rng, act, sharing_rate=sharing_rate),
            ))

    n_patients_actual = len(patients)

    # ---- points (depot, hospitals, patient homes) ----
    points = [dict(point_id="D0", point_type="depot", lat=DEPOT_LATLON[0], lon=DEPOT_LATLON[1])]
    for _, hrow in hospitals_df.iterrows():
        points.append(dict(point_id=hrow["hospital_id"], point_type="hospital",
                            lat=hrow["lat"], lon=hrow["lon"]))

    home_point_of = {}
    for p in patients:
        radius = HOME_RADIUS_DEG_MAX * math.sqrt(rng.random())
        angle = rng.random() * 2 * math.pi
        lat = DEPOT_LATLON[0] + radius * math.cos(angle)
        lon = DEPOT_LATLON[1] + radius * math.sin(angle)
        home_id = f"HOME_{p['patient_id']}"
        home_point_of[p["patient_id"]] = home_id
        points.append(dict(point_id=home_id, point_type="patient_home", lat=lat, lon=lon))
    for p in patients:
        p["home_point_id"] = home_point_of[p["patient_id"]]

    points_df = pd.DataFrame(points)

    # ---- arcs (fully connected directed graph) ----
    n_points = len(points_df)
    coords = {r["point_id"]: (r["lat"], r["lon"]) for _, r in points_df.iterrows()}
    ptype  = {r["point_id"]: r["point_type"] for _, r in points_df.iterrows()}

    arc_rows = []
    dist_matrix = {i: {} for i in coords}
    for i in coords:
        for j in coords:
            if i == j:
                dist_matrix[i][j] = 0.0
                continue
            d = round(_haversine_km(*coords[i], *coords[j]), 5)
            dist_matrix[i][j] = d
            arc_rows.append(dict(
                i=i, j=j, origin_point_type=ptype[i], destination_point_type=ptype[j],
                t_ij=round(d / SPEED_KMH * 60, 2), c_ij=round(d * COST_PER_KM, 2), distance_km=d,
            ))
    arc_df = pd.DataFrame(arc_rows)

    dm_rows = []
    for i in coords:
        row = {"origin_point_id": i}
        row.update({j: dist_matrix[i][j] for j in coords})
        dm_rows.append(row)
    distance_matrix_df = pd.DataFrame(dm_rows)

    # ---- patient_parameters ----
    patient_df = pd.DataFrame(patients)[[
        "patient_id", "home_point_id", "hospital_id", "activity_type", "requirement",
        "a_p", "a_p_hhmm", "a_p_datetime", "tau_p", "Psi_p", "Delta_p", "sharing_possible",
    ]]

    # ---- appointment_plan ----
    appointment_df = pd.DataFrame([
        dict(
            appointment_id=f"A{idx + 1:04d}", patient_id=p["patient_id"], hospital_id=p["hospital_id"],
            activity_type=p["activity_type"], requirement=p["requirement"],
            appointment_minute_a_p=p["a_p"], appointment_time_hhmm=p["a_p_hhmm"],
            appointment_datetime=p["a_p_datetime"], treatment_duration_tau_p=p["tau_p"],
            sharing_possible=p["sharing_possible"],
        )
        for idx, p in enumerate(patients)
    ])

    # ---- requests + alpha_pvk_phi ----
    compat_lookup = ref["requirement_vehicle_compatibili"].set_index("requirement")
    request_rows = []
    alpha_rows = []
    for idx, p in enumerate(patients):
        appt_id = f"A{idx + 1:04d}"
        r_out = f"R{2 * idx + 1:04d}"
        r_ret = f"R{2 * idx + 2:04d}"
        home = p["home_point_id"]
        hosp = p["hospital_id"]

        request_rows.append(dict(
            request_id=r_out, patient_id=p["patient_id"], appointment_id=appt_id,
            service_phase_phi="O", trip_type="outbound",
            origin_point_id=home, destination_point_id=hosp, hospital_id=hosp,
            activity_type=p["activity_type"], requirement=p["requirement"],
            paired_request_id=r_ret, sharing_possible=p["sharing_possible"],
        ))
        request_rows.append(dict(
            request_id=r_ret, patient_id=p["patient_id"], appointment_id=appt_id,
            service_phase_phi="B", trip_type="return",
            origin_point_id=hosp, destination_point_id=home, hospital_id=hosp,
            activity_type=p["activity_type"], requirement=p["requirement"],
            paired_request_id=r_out, sharing_possible=p["sharing_possible"],
        ))

        compat_row = compat_lookup.loc[p["requirement"]]
        for rid, phase, trip in ((r_out, "O", "outbound"), (r_ret, "B", "return")):
            alpha_rows.append(dict(
                request_id=rid, patient_id=p["patient_id"], service_phase_phi=phase, trip_type=trip,
                requirement=p["requirement"],
                alpha_taxi_conventionne=int(compat_row["taxi_conventionne"]),
                alpha_vsl=int(compat_row["vsl"]),
                alpha_vtc_premium=int(compat_row["vtc_premium"]),
            ))

    requests_df = pd.DataFrame(request_rows)
    alpha_df    = pd.DataFrame(alpha_rows)

    # ---- vehicle_type_summary ----
    eta = _eta_split(target_n_vehicles)
    vtype_df = pd.DataFrame([
        dict(vehicle_type_k="taxi_conventionne", Q_k=capacity, eta_k=eta["taxi_conventionne"]),
        dict(vehicle_type_k="vsl", Q_k=capacity, eta_k=eta["vsl"]),
        dict(vehicle_type_k="vtc_premium", Q_k=capacity, eta_k=eta["vtc_premium"]),
    ])
    generated_eta_total = int(vtype_df["eta_k"].sum())

    # ---- metadata ----
    metadata_df = pd.DataFrame([
        dict(field="instance_id", value=instance_name),
        dict(field="category", value=f"G/{target_n_patients}/{target_n_vehicles}/{replicate}"),
        dict(field="size", value="G"),
        dict(field="service_date", value=service_date),
        dict(field="seed", value=seed),
        dict(field="target_n_patients", value=target_n_patients),
        dict(field="generated_n_patients", value=n_patients_actual),
        dict(field="target_n_vehicles", value=target_n_vehicles),
        dict(field="generated_eta_total", value=generated_eta_total),
        dict(field="target_n_hospitals", value=n_hospitals),
        dict(field="generated_n_hospitals", value=n_hospitals),
        dict(field="n_requests", value=2 * n_patients_actual),
        dict(field="n_outbound_requests", value=n_patients_actual),
        dict(field="n_return_requests", value=n_patients_actual),
        dict(field="n_points", value=n_points),
        dict(field="n_arcs", value=len(arc_df)),
    ])

    # ---- write workbook ----
    out_path = os.path.join(out_dir, f"{instance_name}_model_input.xlsx")
    global_parameters_out = ref["global_parameters"].copy()
    if omega_override is not None:
        global_parameters_out.loc[global_parameters_out["symbol"] == "Omega", "value"] = omega_override
    if psi_override is not None:
        global_parameters_out.loc[global_parameters_out["symbol"] == "Psi", "value"] = psi_override
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        patient_df.to_excel(writer, sheet_name="patient_parameters", index=False)
        global_parameters_out.to_excel(writer, sheet_name="global_parameters", index=False)
        vtype_df.to_excel(writer, sheet_name="vehicle_type_summary", index=False)
        arc_df.to_excel(writer, sheet_name="arc_parameters", index=False)
        distance_matrix_df.to_excel(writer, sheet_name="distance_matrix_km", index=False)
        alpha_df.to_excel(writer, sheet_name="alpha_pvk_phi", index=False)
        hospitals_df.to_excel(writer, sheet_name="hospitals", index=False)
        points_df.to_excel(writer, sheet_name="points", index=False)
        demand_rows.to_excel(writer, sheet_name="medical_demand_plan", index=False)
        appointment_df.to_excel(writer, sheet_name="appointment_plan", index=False)
        requests_df.to_excel(writer, sheet_name="requests", index=False)
        ref["requirement_vehicle_compatibili"].to_excel(writer, sheet_name="requirement_vehicle_compatibili", index=False)
        ref["sharing_rule_summary"].to_excel(writer, sheet_name="sharing_rule_summary", index=False)
        ref["math_parameter_status"].to_excel(writer, sheet_name="math_parameter_status", index=False)
        ref["model_dictionary"].to_excel(writer, sheet_name="model_dictionary", index=False)

    return out_path, n_patients_actual, generated_eta_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 30, 40, 60, 100, 125, 150])
    parser.add_argument("--vehicle_counts", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--hospital_counts", type=int, nargs="+", default=[1])
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--service_date", type=str, default="2026-09-01")
    parser.add_argument("--reference", type=str, default=DEFAULT_REFERENCE)
    parser.add_argument("--out_root", type=str,
                         default=os.path.join(HERE, "instances"))
    parser.add_argument("--capacity", type=int, default=3,
                         help="Per-vehicle capacity Q_k, applied uniformly to all three "
                              "vehicle types. Default 3 matches the real E_ instances, "
                              "which use Q_k=3 for every type (types differ by "
                              "compatibility/fleet count, not capacity).")
    parser.add_argument("--omega", type=float, default=None,
                         help="Override Omega (max early-arrival allowance, minutes). "
                              "Default None keeps the reference file's value (45 min).")
    parser.add_argument("--psi", type=float, default=None,
                         help="Override Psi (max return-pickup delay allowance, minutes), "
                              "applied uniformly to every patient. Default None keeps the "
                              "reference file's value (35 min).")
    args = parser.parse_args()

    print(f"Loading reference tables from: {args.reference}")
    ref = _load_reference(args.reference)

    seed = args.seed_start
    generated = []
    for n_patients in args.sizes:
        for n_vehicles in args.vehicle_counts:
            for n_hospitals in args.hospital_counts:
                for rep in range(1, args.replicates + 1):
                    path, n_actual, eta_total = generate_instance(
                        n_patients, n_vehicles, n_hospitals, rep, seed,
                        args.service_date, ref, args.out_root, capacity=args.capacity,
                        omega_override=args.omega, psi_override=args.psi,
                    )
                    print(f"  generated {os.path.basename(os.path.dirname(path))}: "
                          f"{n_actual} patients, {eta_total} vehicles, {n_hospitals} hospital(s) "
                          f"(seed={seed})")
                    generated.append(path)
                    seed += 1

    print(f"\nGenerated {len(generated)} instance(s) under: {args.out_root}")


if __name__ == "__main__":
    main()
