# ============================================================
# HEALTHCARE DIAL-A-RIDE PROBLEM -- SHARED CORE
#
# Instance loading, the Solution representation, all constraint/evaluation
# logic, and the destroy/repair primitives, factored out of darp_lns.py so
# every search algorithm (LNS, SA, TS, GA, ACO) optimizes over the exact
# same feasible-solution space and objective -- the comparison between them
# is then genuinely about search strategy, not five slightly different
# reimplementations of the DARP model.
#
# NOT CONCURRENCY-SAFE: instance state lives in MODULE-LEVEL GLOBALS, set by
# load_instance(). Every algorithm script runs as its own subprocess/process
# (one instance loaded per process), so this is safe in practice -- but this
# module must never be imported into a multithreaded or multi-instance
# driver without redesigning it around an explicit instance object instead.
# ============================================================

import copy
import json
import math
import os
import random

import numpy as np
import pandas as pd

from dataclasses import dataclass

DEPOT = 'DEPOT'

VIOLATION_PENALTY = 100
UNSERVED_PENALTY  = 1000

# Multiplier on the patient-QoS soft-constraint penalties only (excess ride
# time beyond Delta_p, early arrival beyond OMEGA, return delay beyond PSI).
# Lateness past the appointment, capacity and sharing keep the full
# VIOLATION_PENALTY regardless. 1.0 = original behavior; darp_lns.py's
# --no_qos sets it to 0 for the search and back to 1 before reporting.
QOS_PENALTY_SCALE = 1.0

# When False, no intra-shift or inter-shift transition is ever activated: every
# fragment starts and ends at the depot (depot-based baseline). True = original
# behavior; darp_lns.py's --disable_transitions sets it to False.
TRANSITIONS_ENABLED = True

# Per-activity-type QoS (ride-time) weight, used only when darp_lns.py's
# --weighted_qos flag is set. Grounded in the model's own existing Delta_p
# (hard ride-time cap) ordering, not arbitrary: dialysis/oncology already
# get the tightest caps (25 min) in generate_instances.py's fitted
# DELTA_P_FIXED, other/therapeutic looser (35 min), external loosest
# (45-55 min) -- these weights track that same necessity ordering.
PATIENT_TYPE_WEIGHT = {
    "dialysis": 2.0,
    "oncology": 1.75,
    "therapeutic": 1.25,
    "other": 1.0,
    "external": 0.75,
}

# ============================================================
# INSTANCE STATE -- populated by load_instance(); everything below reads
# these as late-bound module globals (resolved at call time, not def time).
# ============================================================

PATIENTS      = None
HOSPITALS     = None
VEHICLE_TYPES = None
VEHICLES      = None
vehicle_type  = None
CAPACITY      = None
hospital      = None
appointment   = None
tau           = None
DELTA_P       = None
PSI_P         = None
service       = None
arc_time      = None
arc_cost      = None
compatO       = None
compatR       = None
LAMBDA        = None
OMEGA         = None
SHARING_POSSIBLE = None
ACTIVITY_TYPE     = None
PATIENT_QOS_WEIGHT = None   # per-patient ride_penalty multiplier; all-1.0 unless darp_lns.py's --weighted_qos sets it

DAY_START_MINUTES  = None
DAY_LENGTH_MINUTES = None
DAY_START_HOUR     = None

NUM_PATIENTS  = None
NUM_HOSPITALS = None
NUM_TYPES     = None

PHYSICAL_VEHICLES  = None
SHIFTS_PER_VEHICLE = None
MAX_SHIFTS         = None
SLOTS_OF_VEHICLE   = None

# ============================================================
# TIME FORMAT
# ============================================================

def fmt(minutes):
    total_min = DAY_START_MINUTES + int(minutes)
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"

def _slot_label(slot):
    """'vsl_V1::S2' -> 'vsl_V1 (trip 2)', for readable output."""
    v, k = slot.split("::S")
    return f"{v} (trip {k})"

# ============================================================
# LOAD INSTANCE FROM EXCEL (*_model_input.xlsx)
# ============================================================

def load_excel_instance(path, day_start_buffer_hours, day_end_buffer_hours):
    xls = pd.ExcelFile(path)

    patient_df = pd.read_excel(xls, 'patient_parameters')
    global_s   = pd.read_excel(xls, 'global_parameters').set_index('symbol')['value']
    vtype_df   = pd.read_excel(xls, 'vehicle_type_summary')
    arc_df     = pd.read_excel(xls, 'arc_parameters')
    alpha_df   = pd.read_excel(xls, 'alpha_pvk_phi')
    hosp_df    = pd.read_excel(xls, 'hospitals')
    points_df  = pd.read_excel(xls, 'points')

    patients      = list(patient_df['patient_id'])
    hospitals     = list(hosp_df['hospital_id'])
    vehicle_types = list(vtype_df['vehicle_type_k'])

    day_start_raw = int(patient_df['a_p'].min()) - int(round(day_start_buffer_hours * 60))
    return_deadlines_raw = patient_df['a_p'] + patient_df['tau_p'] + patient_df['Psi_p']
    day_end_raw = int(return_deadlines_raw.max()) + int(round(day_end_buffer_hours * 60))
    day_length_minutes = day_end_raw - day_start_raw

    vehicles     = []
    vehicle_type = {}
    capacity     = {}
    for _, row in vtype_df.iterrows():
        vt = row['vehicle_type_k']
        for i in range(int(row['eta_k'])):
            v = f"{vt}_V{i + 1}"
            vehicles.append(v)
            vehicle_type[v] = vt
            capacity[v] = int(row['Q_k'])

    hospital         = dict(zip(patient_df['patient_id'], patient_df['hospital_id']))
    sharing_possible = dict(zip(patient_df['patient_id'], patient_df['sharing_possible'].astype(int)))
    activity_type    = dict(zip(patient_df['patient_id'], patient_df['activity_type']))
    appointment = {
        p: a - day_start_raw
        for p, a in zip(patient_df['patient_id'], patient_df['a_p'])
    }
    tau     = dict(zip(patient_df['patient_id'], patient_df['tau_p']))
    delta_p = dict(zip(patient_df['patient_id'], patient_df['Delta_p']))
    psi_p   = dict(zip(patient_df['patient_id'], patient_df['Psi_p']))

    s_time  = float(global_s['s'])
    service = {DEPOT: 0.0}
    for p in patients:
        service[p] = s_time
    for h in hospitals:
        service[h] = s_time

    # node label (as used in routes) -> point id (as used in the arc table)
    depot_point = points_df.loc[points_df['point_type'] == 'depot', 'point_id'].iloc[0]
    node_to_point = {DEPOT: depot_point}
    for h in hospitals:
        node_to_point[h] = h
    for p, home in zip(patient_df['patient_id'], patient_df['home_point_id']):
        node_to_point[p] = home

    time_table = {}
    cost_table = {}
    for _, row in arc_df.iterrows():
        time_table[(row['i'], row['j'])] = float(row['t_ij'])
        cost_table[(row['i'], row['j'])] = float(row['c_ij'])

    def arc_time(a, b):
        if a == b:
            return 0.0
        return time_table[(node_to_point[a], node_to_point[b])]

    def arc_cost(a, b):
        if a == b:
            return 0.0
        return cost_table[(node_to_point[a], node_to_point[b])]

    # phase-specific (outbound 'O' / return 'B') vehicle-type compatibility,
    # already resolved per patient in the instance file
    compatO = {}
    compatR = {}
    for _, row in alpha_df.iterrows():
        p     = row['patient_id']
        table = compatO if row['service_phase_phi'] == 'O' else compatR
        for vt in vehicle_types:
            ok = int(row[f'alpha_{vt}'])
            for v in vehicles:
                if vehicle_type[v] == vt:
                    table[p, v] = ok

    lam   = float(global_s['lambda'])
    omega = float(global_s['Omega'])

    return dict(
        patients=patients, hospitals=hospitals,
        vehicle_types=vehicle_types, vehicles=vehicles,
        vehicle_type=vehicle_type, capacity=capacity,
        hospital=hospital, appointment=appointment, tau=tau,
        delta_p=delta_p, psi_p=psi_p, service=service, sharing_possible=sharing_possible,
        activity_type=activity_type,
        arc_time=arc_time, arc_cost=arc_cost,
        compatO=compatO, compatR=compatR,
        lam=lam, omega=omega,
        day_start_minutes=day_start_raw, day_length_minutes=day_length_minutes,
    )

def load_instance(path, shifts=2, max_shifts=12,
                   day_start_buffer_hours=1.0, day_end_buffer_hours=2.0):
    """Loads a *_model_input.xlsx instance and sets ALL module globals used
    by the rest of darp_core.py, expanding each physical vehicle into its
    starting routing slots. Returns an info dict (counts, day-window
    strings) for the caller to print -- this function itself never prints."""
    global PATIENTS, HOSPITALS, VEHICLE_TYPES, VEHICLES, vehicle_type, CAPACITY
    global hospital, appointment, tau, DELTA_P, PSI_P, service, arc_time, arc_cost
    global compatO, compatR, LAMBDA, OMEGA, SHARING_POSSIBLE, ACTIVITY_TYPE, PATIENT_QOS_WEIGHT
    global DAY_START_MINUTES, DAY_LENGTH_MINUTES, DAY_START_HOUR
    global NUM_PATIENTS, NUM_HOSPITALS, NUM_TYPES
    global PHYSICAL_VEHICLES, SHIFTS_PER_VEHICLE, MAX_SHIFTS, SLOTS_OF_VEHICLE

    _inst = load_excel_instance(path, day_start_buffer_hours, day_end_buffer_hours)

    PATIENTS      = _inst['patients']
    HOSPITALS     = _inst['hospitals']
    VEHICLE_TYPES = _inst['vehicle_types']
    VEHICLES      = _inst['vehicles']
    vehicle_type  = _inst['vehicle_type']
    CAPACITY      = _inst['capacity']
    hospital      = _inst['hospital']
    SHARING_POSSIBLE = _inst['sharing_possible']
    ACTIVITY_TYPE     = _inst['activity_type']
    PATIENT_QOS_WEIGHT = {p: 1.0 for p in _inst['patients']}
    appointment   = _inst['appointment']
    tau           = _inst['tau']
    DELTA_P       = _inst['delta_p']
    PSI_P         = _inst['psi_p']
    service       = _inst['service']
    arc_time      = _inst['arc_time']
    arc_cost      = _inst['arc_cost']
    compatO       = _inst['compatO']
    compatR       = _inst['compatR']
    LAMBDA        = _inst['lam']
    OMEGA         = _inst['omega']

    DAY_START_MINUTES  = _inst['day_start_minutes']
    DAY_LENGTH_MINUTES = _inst['day_length_minutes']
    DAY_START_HOUR     = DAY_START_MINUTES / 60.0

    NUM_PATIENTS  = len(PATIENTS)
    NUM_HOSPITALS = len(HOSPITALS)
    NUM_TYPES     = len(VEHICLE_TYPES)

    PHYSICAL_VEHICLES = VEHICLES
    SHIFTS_PER_VEHICLE = shifts
    MAX_SHIFTS         = max_shifts

    SLOTS_OF_VEHICLE = {}
    _slot_vehicle_type = {}
    _slot_capacity     = {}
    _slot_compatO      = {}
    _slot_compatR      = {}

    VEHICLES = []
    for v in PHYSICAL_VEHICLES:
        slots = [f"{v}::S{k + 1}" for k in range(SHIFTS_PER_VEHICLE)]
        SLOTS_OF_VEHICLE[v] = slots
        for slot in slots:
            VEHICLES.append(slot)
            _slot_vehicle_type[slot] = vehicle_type[v]
            _slot_capacity[slot]     = CAPACITY[v]
            for p in PATIENTS:
                _slot_compatO[p, slot] = compatO[p, v]
                _slot_compatR[p, slot] = compatR[p, v]

    vehicle_type = _slot_vehicle_type
    CAPACITY     = _slot_capacity
    compatO      = _slot_compatO
    compatR      = _slot_compatR

    fleet_counts = {
        t: sum(1 for v in PHYSICAL_VEHICLES if _slot_vehicle_type[SLOTS_OF_VEHICLE[v][0]] == t)
        for t in VEHICLE_TYPES
    }

    return dict(
        num_patients=NUM_PATIENTS, num_hospitals=NUM_HOSPITALS,
        num_physical_vehicles=len(PHYSICAL_VEHICLES), fleet_counts=fleet_counts,
        num_routing_slots=len(VEHICLES),
        day_window_start=fmt(0), day_window_end=fmt(DAY_LENGTH_MINUTES),
        day_start_buffer_hours=day_start_buffer_hours,
        day_end_buffer_hours=day_end_buffer_hours,
        shifts_start=SHIFTS_PER_VEHICLE, shifts_max=MAX_SHIFTS,
    )

def grow_vehicle_slot(pv):
    """Add one new empty shift slot to physical vehicle pv (copying its
    capacity/type/compatibility from an existing sibling slot) and return
    its id."""
    sibling = SLOTS_OF_VEHICLE[pv][0]
    k = len(SLOTS_OF_VEHICLE[pv]) + 1
    new_slot = f"{pv}::S{k}"

    SLOTS_OF_VEHICLE[pv].append(new_slot)
    VEHICLES.append(new_slot)
    vehicle_type[new_slot] = vehicle_type[sibling]
    CAPACITY[new_slot]     = CAPACITY[sibling]
    for p in PATIENTS:
        compatO[p, new_slot] = compatO[p, sibling]
        compatR[p, new_slot] = compatR[p, sibling]

    return new_slot

def ensure_growth_room(*solutions):
    """For every physical vehicle whose current LAST slot is already in use
    (in the first solution passed -- normally `current`), grow one more
    empty slot, and patch every solution passed in with an empty route for
    it, so all of them stay consistent with the (now larger) global slot
    list. Call this before destroy()/repair() each iteration, and once
    before the search loop starts."""
    if not solutions:
        return

    reference = solutions[0]

    for pv in PHYSICAL_VEHICLES:
        if len(SLOTS_OF_VEHICLE[pv]) >= MAX_SHIFTS:
            continue

        last_slot = SLOTS_OF_VEHICLE[pv][-1]
        in_use = (
            any(n in PATIENTS for n in reference.outbound_routes[last_slot])
            or any(n in PATIENTS for n in reference.return_routes[last_slot])
        )
        if not in_use:
            continue

        new_slot = grow_vehicle_slot(pv)
        for sol in solutions:
            sol.outbound_routes[new_slot] = [DEPOT, DEPOT]
            sol.return_routes[new_slot]   = [DEPOT, DEPOT]

# ============================================================
# CLEAN ROUTE
# ============================================================

def clean_route(route):
    cleaned = [route[0]]
    for n in route[1:]:
        if n in HOSPITALS and cleaned[-1] == n:
            continue
        cleaned.append(n)
    return cleaned

# ============================================================
# REMOVE REDUNDANT HOSPITALS
# ============================================================

def remove_redundant_hospitals(route):
    if len(route) <= 2:
        return route
    cleaned = [route[0]]
    for i in range(1, len(route) - 1):
        prev = cleaned[-1]
        curr = route[i]
        if prev in HOSPITALS and curr in HOSPITALS:
            continue
        cleaned.append(curr)
    cleaned.append(route[-1])
    return cleaned

# ============================================================
# ROUTE STRUCTURE VALIDATION
# ============================================================

def route_structure_ok(route, phase):
    meaningful = [n for n in route if n != DEPOT]
    if not meaningful:
        return True
    if phase == 'OUT':
        if meaningful[0] in HOSPITALS:
            return False
    else:
        if meaningful[0] in PATIENTS:
            return False
    return True

# ============================================================
# PRECEDENCE VALIDATION
# ============================================================

def valid_order(inner, phase):
    patient_positions  = {}
    hospital_positions = {}
    for i, n in enumerate(inner):
        if n in PATIENTS:
            patient_positions[n] = i
        elif n in HOSPITALS:
            hospital_positions.setdefault(n, []).append(i)

    for p in patient_positions:
        h = hospital[p]
        if h not in hospital_positions:
            return False
        p_pos = patient_positions[p]
        if phase == 'OUT':
            if not any(hp > p_pos for hp in hospital_positions[h]):
                return False
        else:
            if not any(hp < p_pos for hp in hospital_positions[h]):
                return False
    return True

# ============================================================
# STRICT CONSISTENCY VALIDATION
# ============================================================

def route_consistency_ok(route, phase):
    positions = {}
    for i, n in enumerate(route):
        positions.setdefault(n, []).append(i)

    patients_in_route = [n for n in route if n in PATIENTS]

    if phase == 'OUT':
        for p in patients_in_route:
            h = hospital[p]
            if h not in positions:
                return False
            p_pos = positions[p][0]
            if not any(hp > p_pos for hp in positions[h]):
                return False
    else:
        for p in patients_in_route:
            h = hospital[p]
            if h not in positions:
                return False
            p_pos = positions[p][0]
            if not any(hp < p_pos for hp in positions[h]):
                return False
    return True

# ============================================================
# ROUTE CAPACITY VALIDATION
# ============================================================

def route_capacity_ok(route, vehicle, phase):
    cap = CAPACITY[vehicle]
    load = 0
    onboard = set()
    route_patients = [n for n in route if n in PATIENTS]

    for i in range(len(route) - 1):
        n2 = route[i + 1]

        if phase == 'OUT':
            if n2 in PATIENTS:
                load += 1
                onboard.add(n2)
                if load > cap:
                    return False
                if load > 1 and any(SHARING_POSSIBLE[p] == 0 for p in onboard):
                    return False
            elif n2 in HOSPITALS:
                delivered = [p for p in onboard if hospital[p] == n2]
                for p in delivered:
                    onboard.discard(p)
                load -= len(delivered)

        else:  # RET
            if n2 in HOSPITALS:
                pts_here = [p for p in route_patients if hospital[p] == n2]
                new_pts = [p for p in pts_here if p not in onboard]
                onboard.update(new_pts)
                load += len(new_pts)
                if load > cap:
                    return False
                if load > 1 and any(SHARING_POSSIBLE[p] == 0 for p in onboard):
                    return False
            elif n2 in PATIENTS:
                if n2 in onboard:
                    onboard.discard(n2)
                    load -= 1

    return True

# ============================================================
# ROUTE ON-TIME VALIDATION
# ============================================================

def route_ontime_ok(route, vehicle, phase, start_time=0.0):
    if phase != 'OUT':
        return True

    route = clean_route(route)
    route = remove_redundant_hospitals(route)

    if not route_structure_ok(route, phase) or not route_consistency_ok(route, phase):
        return True   # structural problems are rejected elsewhere; not our concern here

    t = float(start_time)
    pickups = {}

    for i in range(len(route) - 1):
        n1 = route[i]
        n2 = route[i + 1]

        t += service[n1]
        t += arc_time(n1, n2)

        if n2 in PATIENTS:
            pickups[n2] = t

        elif n2 in HOSPITALS:
            pts_here = [p for p in pickups if hospital[p] == n2]
            if pts_here:
                earliest = max(appointment[p] - OMEGA for p in pts_here)
                if t < earliest:
                    t = earliest
            for p in pts_here:
                if t > appointment[p]:
                    return False
                del pickups[p]

    return True

# ============================================================
# SOLUTION CLASS
# ============================================================

@dataclass
class Solution:
    outbound_routes: dict
    return_routes:   dict
    transitions:     dict
    shift_transitions: dict

    travel_cost:    float
    ride_penalty:   float
    violation_cost: float

    objective: float
    feasible:  bool

    num_served: int
    unserved:   list

# ============================================================
# OUTBOUND PRE-PICKUP WAIT
# ============================================================

def _out_run_wait(route, start_idx, t0):
    """How long to wait at route[start_idx] (a point where the outbound
    vehicle is empty, about to start a new pickup/drop run, currently at
    ABSOLUTE time t0) before departing, to avoid arriving needlessly early
    at an upcoming hospital.

    evaluate_route's node times were previously always ASAP: a vehicle
    departed the instant it could, so any hospital reachable long before a
    patient's appointment got flagged as an early-arrival violation even
    though the vehicle could simply have left later. Waiting HERE -- before
    anyone in the run is picked up -- delays pickup and hospital arrival by
    the same amount, so ride time is unaffected (unlike waiting mid-ride,
    with a passenger already onboard, which would inflate it). This mirrors
    the exact MIP, where node visit times are free variables bounded only
    below by cumulative travel -- i.e. waiting anywhere is already free
    there; ASAP-only propagation was an LNS-only limitation.

    Returns the wait needed to satisfy the current run's own patients'
    earliest-arrival bound (appointment-OMEGA), capped so it never pushes
    ANY hospital arrival for the rest of the route (this run or later ones,
    since the wait propagates downstream through cumulative time) past its
    hard appointment deadline."""
    rel_t       = 0.0
    onboard_sim = set()
    lo_need     = 0.0
    hi_allow    = None
    run_open    = True   # still accumulating lo_need for the CURRENT run

    for i in range(start_idx, len(route) - 1):
        n1, n2 = route[i], route[i + 1]
        rel_t += service[n1] + arc_time(n1, n2)

        if n2 in PATIENTS:
            onboard_sim.add(n2)
        elif n2 in HOSPITALS:
            pts_here = [p for p in onboard_sim if hospital[p] == n2]
            for p in pts_here:
                if run_open:
                    lo_need = max(lo_need, appointment[p] - OMEGA - t0 - rel_t)
                cap = appointment[p] - t0 - rel_t
                hi_allow = cap if hi_allow is None else min(hi_allow, cap)
                onboard_sim.discard(p)
            if not onboard_sim:
                run_open = False   # current run's patients are all accounted
                                    # for; keep scanning (hi_allow only) so a
                                    # later run's deadline is never breached

    if hi_allow is None:
        return 0.0
    return max(0.0, min(lo_need, hi_allow))

# ============================================================
# ROUTE EVALUATION
# ============================================================

def evaluate_route(route, vehicle, phase, start_time=0.0):
    route = clean_route(route)
    route = remove_redundant_hospitals(route)

    if not route_structure_ok(route, phase):
        return False, 1e9, 1e9, 1e9, {}
    if not route_consistency_ok(route, phase):
        return False, 1e9, 1e9, 1e9, {}

    t            = float(start_time)
    load         = 0
    pickups      = {}
    feasible     = True
    travel_cost  = 0.0
    ride_penalty = 0.0
    viol_cost    = 0.0
    node_times   = {route[0]: t}

    route_patients = [n for n in route if n in PATIENTS]
    cap = CAPACITY[vehicle]

    for i in range(len(route) - 1):
        n1     = route[i]
        n2     = route[i + 1]

        if phase == 'OUT' and load == 0:
            t += _out_run_wait(route, i, t)
            node_times[n1] = t

        t += service[n1]
        t += arc_time(n1, n2)
        arrival_time = t
        travel_cost += arc_cost(n1, n2)

        # ====================================================
        # OUTBOUND
        # ====================================================
        if phase == 'OUT':

            if n2 in PATIENTS:
                load += 1
                pickups[n2] = arrival_time
                if load > cap:
                    feasible   = False
                    viol_cost += VIOLATION_PENALTY * (load - cap)
                if load > 1 and any(SHARING_POSSIBLE[p] == 0 for p in pickups):
                    feasible   = False
                    viol_cost += VIOLATION_PENALTY * (load - 1)
                node_times[n2] = t

            elif n2 in HOSPITALS:
                # Time-window wait: vehicle must arrive in [appt-OMEGA, appt]
                pts_here = [p for p in pickups if hospital[p] == n2]
                if pts_here:
                    earliest = max(appointment[p] - OMEGA for p in pts_here)
                    if t < earliest:
                        t = earliest

                delivered = []
                for p in list(pickups):
                    if hospital[p] == n2:
                        ride          = arrival_time - pickups[p]
                        ride_penalty += PATIENT_QOS_WEIGHT[p] * ride
                        if ride > DELTA_P[p]:
                            feasible   = False
                            viol_cost += QOS_PENALTY_SCALE * VIOLATION_PENALTY * (ride - DELTA_P[p])
                        # Soft time-window check: physically arriving more than
                        # OMEGA minutes before the appointment is penalized (like
                        # ride-time/lateness), even though the wait above still
                        # absorbs it for downstream timing -- it's a time-window
                        # constraint, not a hard one like capacity.
                        if arrival_time < appointment[p] - OMEGA:
                            feasible   = False
                            viol_cost += QOS_PENALTY_SCALE * VIOLATION_PENALTY * (appointment[p] - OMEGA - arrival_time)
                        if t > appointment[p]:
                            feasible   = False
                            viol_cost += VIOLATION_PENALTY * (t - appointment[p])
                        delivered.append(p)
                        load -= 1
                for p in delivered:
                    del pickups[p]

                node_times[n2] = t      # after waiting

            else:   # DEPOT
                node_times[n2] = t

        # ====================================================
        # RETURN
        # ====================================================
        else:

            if n2 in HOSPITALS:
                pts_here = [p for p in route_patients if hospital[p] == n2 and p not in pickups]
                for p in pts_here:
                    ready = appointment[p] + tau[p]
                    if t < ready:
                        t = ready
                for p in pts_here:
                    ready = appointment[p] + tau[p]
                    if t > ready + PSI_P[p]:
                        feasible   = False
                        viol_cost += QOS_PENALTY_SCALE * VIOLATION_PENALTY * (t - (ready + PSI_P[p]))
                    pickups[p] = t
                    load += 1
                    if load > cap:
                        feasible   = False
                        viol_cost += VIOLATION_PENALTY * (load - cap)
                    if load > 1 and any(SHARING_POSSIBLE[q] == 0 for q in pickups):
                        feasible   = False
                        viol_cost += VIOLATION_PENALTY * (load - 1)

                node_times[n2] = t      # after waiting

            elif n2 in PATIENTS:
                if n2 in pickups:
                    ride          = arrival_time - pickups[n2]
                    ride_penalty += PATIENT_QOS_WEIGHT[n2] * ride
                    if ride > DELTA_P[n2]:
                        feasible   = False
                        viol_cost += QOS_PENALTY_SCALE * VIOLATION_PENALTY * (ride - DELTA_P[n2])
                    load -= 1
                    del pickups[n2]
                node_times[n2] = t

            else:   # DEPOT
                node_times[n2] = t

    return feasible, travel_cost, ride_penalty, viol_cost, node_times

# ============================================================
# ROUTE WITHIN DAY VALIDATION
# ============================================================

def route_within_day_ok(route, vehicle, phase, start_time=0.0):
    _, _, _, _, node_times = evaluate_route(route, vehicle, phase, start_time)
    if not node_times:
        return True   # structural problems are rejected elsewhere; not our concern here
    return max(node_times.values()) <= DAY_LENGTH_MINUTES

# ============================================================
# ROUTE SCORE  (single-route cost for insertion comparison)
# ============================================================

def route_score(route, vehicle, phase, start_time=0.0):
    _, c, rp, viol, _ = evaluate_route(route, vehicle, phase, start_time)
    return c + LAMBDA * rp + viol

# ============================================================
# TRANSITION HELPERS
# ============================================================

def _last_outbound_hospital(route):
    """Last hospital in the outbound route (immediately before the final DEPOT)."""
    for node in reversed(route[:-1]):
        if node in HOSPITALS:
            return node
    return None

def _first_return_hospital(route):
    """First hospital in the return route (immediately after the initial DEPOT)."""
    for node in route[1:]:
        if node in HOSPITALS:
            return node
    return None

def _last_node(route):
    """Last meaningful (non-DEPOT) node in the route, of any type, or None if empty."""
    meaningful = [n for n in route if n != DEPOT]
    return meaningful[-1] if meaningful else None

def _first_node(route):
    """First meaningful (non-DEPOT) node in the route, of any type, or None if empty."""
    meaningful = [n for n in route if n != DEPOT]
    return meaningful[0] if meaningful else None

# ============================================================
# ADVANCE SHIFT POSITION
# ============================================================

def _advance_shift_position(out_routes_v, ret_routes_v, out_ntimes, ret_ntimes,
                             prev_last_node, prev_last_ntimes):
    last_ret = _last_node(ret_routes_v)
    if last_ret is not None:
        return last_ret, ret_ntimes
    last_out = _last_node(out_routes_v)
    if last_out is not None:
        return last_out, out_ntimes
    return prev_last_node, prev_last_ntimes

# ============================================================
# INTER-SHIFT TRANSITION START TIME
# ============================================================

def _shift_transition_start_time(shift_trans, prev_last_node, prev_last_ntimes, actual_first, expected_phase):
    if shift_trans is None or prev_last_node is None or prev_last_ntimes is None:
        return None

    exp_prev_last, exp_next_first, exp_phase = shift_trans
    if exp_phase != expected_phase or exp_prev_last != prev_last_node or exp_next_first != actual_first:
        return None
    if prev_last_node not in prev_last_ntimes:
        return None

    t_prev_last    = prev_last_ntimes[prev_last_node]
    departure_prev = t_prev_last + service[prev_last_node]
    direct_arrival = departure_prev + arc_time(prev_last_node, actual_first)
    d_depot_first  = arc_time(DEPOT, actual_first)
    return max(0.0, direct_arrival - d_depot_first)

# ============================================================
# COMPUTE SLOT START TIMES
# ============================================================

def compute_slot_start_times(out_routes, ret_routes, transitions, shift_transitions):
    out_start = {}
    ret_start = {}

    for pv in PHYSICAL_VEHICLES:
        vehicle_free_time = 0.0
        prev_last_node   = None
        prev_last_ntimes = None

        for v in SLOTS_OF_VEHICLE[pv]:
            s_trans = shift_transitions.get(v, None)

            actual_first = _first_node(out_routes[v])
            s_start = _shift_transition_start_time(
                s_trans, prev_last_node, prev_last_ntimes, actual_first, expected_phase='OUT'
            )
            out_start_time = vehicle_free_time if s_start is None else s_start
            out_start[v] = out_start_time

            _, _, _, _, out_ntimes = evaluate_route(out_routes[v], v, 'OUT', out_start_time)
            out_finish = out_ntimes.get(DEPOT, out_start_time)

            start_time = out_finish
            trans = transitions.get(v, None)
            if trans is not None:
                h_out, h_ret = trans
                last_out  = _last_outbound_hospital(out_routes[v])
                first_ret = _first_return_hospital(ret_routes[v])
                if h_out == last_out and h_ret == first_ret and h_out in out_ntimes:
                    t_h_out         = out_ntimes[h_out]
                    departure_h_out = t_h_out + service[h_out]
                    transition_arr  = departure_h_out + arc_time(h_out, h_ret)
                    d_depot_hret    = arc_time(DEPOT, h_ret)
                    start_time      = max(0.0, transition_arr - d_depot_hret)
                else:
                    start_time = out_finish
            else:
                actual_first_ret = _first_node(ret_routes[v])
                s_start_ret = _shift_transition_start_time(
                    s_trans, prev_last_node, prev_last_ntimes, actual_first_ret, expected_phase='RET'
                )
                if s_start_ret is not None:
                    start_time = s_start_ret

            ret_start[v] = start_time

            _, _, _, _, ret_ntimes = evaluate_route(ret_routes[v], v, 'RET', start_time)
            vehicle_free_time = ret_ntimes.get(DEPOT, start_time)

            prev_last_node, prev_last_ntimes = _advance_shift_position(
                out_routes[v], ret_routes[v], out_ntimes, ret_ntimes, prev_last_node, prev_last_ntimes
            )

    return out_start, ret_start

# ============================================================
# TRY ADD SHIFT TRANSITIONS
# ============================================================

def _shift_entry_point(out_route, ret_route):
    """The first real (non-DEPOT) node this shift needs to reach, and which phase
    it belongs to. Normally the outbound route's first stop -- but for a
    return-only shift (no outbound patients at all, which turns out to be
    common: a vehicle drops several batches early, then makes separate return
    sweeps later) it's the return route's first stop instead."""
    first_out = _first_node(out_route)
    if first_out is not None:
        return 'OUT', first_out
    first_ret = _first_node(ret_route)
    if first_ret is not None:
        return 'RET', first_ret
    return None, None

def try_add_shift_transitions(out_routes, ret_routes, current_shift_transitions=None):
    if not TRANSITIONS_ENABLED:
        return {}
    shift_transitions = {} if current_shift_transitions is None else copy.deepcopy(current_shift_transitions)

    for pv in PHYSICAL_VEHICLES:
        slots = SLOTS_OF_VEHICLE[pv]
        for k in range(len(slots) - 1):
            v_prev = slots[k]
            v_next = slots[k + 1]

            prev_last     = _last_node(ret_routes[v_prev]) or _last_node(out_routes[v_prev])
            phase, next_first = _shift_entry_point(out_routes[v_next], ret_routes[v_next])

            if prev_last is None or next_first is None:
                shift_transitions.pop(v_next, None)
                continue

            saving = (
                arc_cost(prev_last, DEPOT)
                + arc_cost(DEPOT, next_first)
                - arc_cost(prev_last, next_first)
            )

            if saving > 1e-6:
                shift_transitions[v_next] = (prev_last, next_first, phase)
            else:
                shift_transitions.pop(v_next, None)

    return shift_transitions

# ============================================================
# TRY ADD TRANSITIONS
# ============================================================

def try_add_transitions(out_routes, ret_routes, current_transitions=None):
    if not TRANSITIONS_ENABLED:
        return {}
    transitions = {} if current_transitions is None else copy.deepcopy(current_transitions)

    for v in VEHICLES:
        has_outbound = any(n in PATIENTS for n in out_routes[v])
        has_return   = any(n in PATIENTS for n in ret_routes[v])

        if not (has_outbound and has_return):
            transitions.pop(v, None)
            continue

        h_out = _last_outbound_hospital(out_routes[v])
        h_ret = _first_return_hospital(ret_routes[v])

        if h_out is None or h_ret is None:
            transitions.pop(v, None)
            continue

        saving = (
            arc_cost(h_out, DEPOT)
            + arc_cost(DEPOT, h_ret)
            - arc_cost(h_out, h_ret)
        )

        if saving > 1e-6:
            transitions[v] = (h_out, h_ret)
        else:
            transitions.pop(v, None)

    return transitions

# ============================================================
# SOLUTION EVALUATION
# ============================================================

def evaluate_solution(out_routes, ret_routes, transitions=None, shift_transitions=None):
    if transitions is None:
        transitions = {}
    if shift_transitions is None:
        shift_transitions = {}

    feasible     = True
    total_travel = 0.0
    total_ride   = 0.0
    total_viol   = 0.0

    # A patient counts as served only if BOTH legs are actually routed --
    # mirrors the exact MIP's single u[p] (one binary ties outbound and
    # return together). repair() enforces this invariant when it inserts
    # patients, so in practice a patient is never present in only one leg.
    out_present = set()
    ret_present = set()
    for v in VEHICLES:
        for p in out_routes[v]:
            if p in PATIENTS:
                out_present.add(p)
        for p in ret_routes[v]:
            if p in PATIENTS:
                ret_present.add(p)
    served = out_present & ret_present
    unserved = [p for p in PATIENTS if p not in served]

    for pv in PHYSICAL_VEHICLES:
        vehicle_free_time = 0.0
        prev_last_node   = None
        prev_last_ntimes = None

        for v in SLOTS_OF_VEHICLE[pv]:
            s_trans          = shift_transitions.get(v, None)
            actual_first     = _first_node(out_routes[v])
            travel_adj_shift = 0.0
            s_start = _shift_transition_start_time(
                s_trans, prev_last_node, prev_last_ntimes, actual_first, expected_phase='OUT'
            )
            out_start_time = vehicle_free_time if s_start is None else s_start
            if s_start is not None:
                travel_adj_shift = (
                    - arc_cost(prev_last_node, DEPOT)
                    - arc_cost(DEPOT, actual_first)
                    + arc_cost(prev_last_node, actual_first)
                )

            ok, c, rp, viol, out_ntimes = evaluate_route(out_routes[v], v, 'OUT', out_start_time)
            total_travel += c + travel_adj_shift
            total_ride   += rp
            total_viol   += viol
            if not ok:
                feasible = False

            out_finish = out_ntimes.get(DEPOT, out_start_time)
            start_time = out_finish
            travel_adj = 0.0

            trans = transitions.get(v, None)
            if trans is not None:
                h_out, h_ret = trans

                last_out  = _last_outbound_hospital(out_routes[v])
                first_ret = _first_return_hospital(ret_routes[v])

                if h_out == last_out and h_ret == first_ret and h_out in out_ntimes:
                    t_h_out          = out_ntimes[h_out]
                    departure_h_out  = t_h_out + service[h_out]
                    transition_arr   = departure_h_out + arc_time(h_out, h_ret)

                    d_depot_hret = arc_time(DEPOT, h_ret)
                    start_time   = max(0.0, transition_arr - d_depot_hret)

                    travel_adj = (
                        - arc_cost(h_out, DEPOT)    # outbound last leg (not driven)
                        - arc_cost(DEPOT, h_ret)    # return first leg (not driven)
                        + arc_cost(h_out, h_ret)    # transition arc
                    )
                else:
                    # Transition is stale / inconsistent; ignore it
                    start_time = out_finish
            else:
                actual_first_ret = _first_node(ret_routes[v])
                s_start_ret = _shift_transition_start_time(
                    s_trans, prev_last_node, prev_last_ntimes, actual_first_ret, expected_phase='RET'
                )
                if s_start_ret is not None:
                    start_time = s_start_ret
                    travel_adj = (
                        - arc_cost(prev_last_node, DEPOT)
                        - arc_cost(DEPOT, actual_first_ret)
                        + arc_cost(prev_last_node, actual_first_ret)
                    )

            ok, c, rp, viol, ret_ntimes = evaluate_route(ret_routes[v], v, 'RET', start_time)
            total_travel += c + travel_adj
            total_ride   += rp
            total_viol   += viol
            if not ok:
                feasible = False

            vehicle_free_time = ret_ntimes.get(DEPOT, start_time)

            prev_last_node, prev_last_ntimes = _advance_shift_position(
                out_routes[v], ret_routes[v], out_ntimes, ret_ntimes, prev_last_node, prev_last_ntimes
            )

    objective = (
        total_travel
        + LAMBDA * total_ride
        + total_viol
        + UNSERVED_PENALTY * len(unserved)
    )

    return Solution(
        outbound_routes=copy.deepcopy(out_routes),
        return_routes=copy.deepcopy(ret_routes),
        transitions=copy.deepcopy(transitions),
        shift_transitions=copy.deepcopy(shift_transitions),
        travel_cost=total_travel,
        ride_penalty=total_ride,
        violation_cost=total_viol,
        objective=objective,
        feasible=feasible,
        num_served=len(served),
        unserved=unserved,
    )

# ============================================================
# EXPLAIN ROUTE
# ============================================================

def explain_route(route, vehicle, phase, start_time=0.0):
    route = clean_route(route)
    route = remove_redundant_hospitals(route)

    records   = []
    cap_events = []

    if not route_structure_ok(route, phase) or not route_consistency_ok(route, phase):
        return records, cap_events, {}

    t     = float(start_time)
    load  = 0
    pickups = {}
    node_times = {route[0]: t}
    route_patients = [n for n in route if n in PATIENTS]
    cap = CAPACITY[vehicle]

    for i in range(len(route) - 1):
        n1 = route[i]
        n2 = route[i + 1]

        if phase == 'OUT' and load == 0:
            t += _out_run_wait(route, i, t)
            node_times[n1] = t

        t += service[n1]
        t += arc_time(n1, n2)
        arrival_time = t

        if phase == 'OUT':

            if n2 in PATIENTS:
                load += 1
                pickups[n2] = arrival_time
                if load > cap:
                    cap_events.append(dict(
                        vehicle=vehicle, phase='OUT', node=n2, time=arrival_time,
                        load=load, capacity=cap, excess=load - cap,
                    ))
                node_times[n2] = t

            elif n2 in HOSPITALS:
                pts_here = [p for p in pickups if hospital[p] == n2]
                if pts_here:
                    earliest = max(appointment[p] - OMEGA for p in pts_here)
                    if t < earliest:
                        t = earliest

                delivered = []
                for p in list(pickups):
                    if hospital[p] == n2:
                        ride = arrival_time - pickups[p]
                        records.append(dict(
                            patient=p, vehicle_out=vehicle,
                            home_pickup=pickups[p], hospital_arrival=arrival_time,
                            appointment=appointment[p],
                            out_ride_time=ride, out_ride_limit=DELTA_P[p],
                            out_ride_violation=ride > DELTA_P[p],
                            early_minutes=max(0.0, appointment[p] - OMEGA - arrival_time),
                            out_early_violation=arrival_time < appointment[p] - OMEGA,
                            late_minutes=max(0.0, t - appointment[p]),
                            out_late_violation=t > appointment[p],
                        ))
                        delivered.append(p)
                        load -= 1
                for p in delivered:
                    del pickups[p]

                node_times[n2] = t

            else:
                node_times[n2] = t

        else:  # RET

            if n2 in HOSPITALS:
                pts_here = [p for p in route_patients if hospital[p] == n2 and p not in pickups]
                for p in pts_here:
                    ready = appointment[p] + tau[p]
                    if t < ready:
                        t = ready
                for p in pts_here:
                    load += 1
                    if load > cap:
                        cap_events.append(dict(
                            vehicle=vehicle, phase='RET', node=n2, time=t,
                            load=load, capacity=cap, excess=load - cap,
                        ))
                    pickups[p] = t

                node_times[n2] = t

            elif n2 in PATIENTS:
                if n2 in pickups:
                    ready = appointment[n2] + tau[n2]
                    delay = pickups[n2] - ready
                    ride  = arrival_time - pickups[n2]
                    records.append(dict(
                        patient=n2, vehicle_ret=vehicle,
                        treatment_ready=ready, return_pickup=pickups[n2],
                        return_delay=delay, return_delay_limit=PSI_P[n2],
                        return_delay_violation=delay > PSI_P[n2],
                        home_dropoff=arrival_time,
                        ret_ride_time=ride, ret_ride_limit=DELTA_P[n2],
                        ret_ride_violation=ride > DELTA_P[n2],
                    ))
                    load -= 1
                    del pickups[n2]
                node_times[n2] = t

            else:
                node_times[n2] = t

    return records, cap_events, node_times

# ============================================================
# BUILD PER-PATIENT DIAGNOSTICS TABLE
# ============================================================

def build_diagnostics(solution):
    out_records = []
    ret_records = []
    cap_events  = []

    for pv in PHYSICAL_VEHICLES:
        vehicle_free_time = 0.0
        prev_last_node   = None
        prev_last_ntimes = None

        for v in SLOTS_OF_VEHICLE[pv]:
            s_trans = solution.shift_transitions.get(v, None)

            actual_first = _first_node(solution.outbound_routes[v])
            s_start = _shift_transition_start_time(
                s_trans, prev_last_node, prev_last_ntimes, actual_first, expected_phase='OUT'
            )
            out_start_time = vehicle_free_time if s_start is None else s_start

            recs, caps, out_ntimes = explain_route(solution.outbound_routes[v], v, 'OUT', out_start_time)
            out_records += recs
            cap_events  += caps

            out_finish = out_ntimes.get(DEPOT, out_start_time)
            start_time = out_finish

            trans = solution.transitions.get(v, None)
            if trans is not None:
                h_out, h_ret = trans
                last_out  = _last_outbound_hospital(solution.outbound_routes[v])
                first_ret = _first_return_hospital(solution.return_routes[v])
                if h_out == last_out and h_ret == first_ret and h_out in out_ntimes:
                    t_h_out         = out_ntimes[h_out]
                    departure_h_out = t_h_out + service[h_out]
                    transition_arr  = departure_h_out + arc_time(h_out, h_ret)
                    d_depot_hret    = arc_time(DEPOT, h_ret)
                    start_time      = max(0.0, transition_arr - d_depot_hret)
            else:
                actual_first_ret = _first_node(solution.return_routes[v])
                s_start_ret = _shift_transition_start_time(
                    s_trans, prev_last_node, prev_last_ntimes, actual_first_ret, expected_phase='RET'
                )
                if s_start_ret is not None:
                    start_time = s_start_ret

            recs, caps, ret_ntimes = explain_route(solution.return_routes[v], v, 'RET', start_time)
            ret_records += recs
            cap_events  += caps

            vehicle_free_time = ret_ntimes.get(DEPOT, start_time)

            prev_last_node, prev_last_ntimes = _advance_shift_position(
                solution.outbound_routes[v], solution.return_routes[v], out_ntimes, ret_ntimes,
                prev_last_node, prev_last_ntimes
            )

    out_df = pd.DataFrame(out_records)
    ret_df = pd.DataFrame(ret_records)

    table = pd.DataFrame({'patient': PATIENTS})
    table = table.merge(out_df, on='patient', how='left')
    table = table.merge(ret_df, on='patient', how='left')

    for col in ('vehicle_out', 'vehicle_ret'):
        if col in table.columns:
            table[col] = table[col].apply(lambda s: _slot_label(s) if pd.notna(s) else s)

    for col in ('home_pickup', 'hospital_arrival', 'appointment',
                'treatment_ready', 'return_pickup', 'home_dropoff'):
        if col in table.columns:
            table[col + '_hhmm'] = table[col].apply(lambda x: fmt(x) if pd.notna(x) else None)

    table['any_violation'] = (
        table.get('out_ride_violation', False).fillna(False)
        | table.get('out_early_violation', False).fillna(False)
        | table.get('out_late_violation', False).fillna(False)
        | table.get('return_delay_violation', False).fillna(False)
        | table.get('ret_ride_violation', False).fillna(False)
    )

    cap_df = pd.DataFrame(cap_events)
    if len(cap_df) > 0:
        cap_df['vehicle'] = cap_df['vehicle'].apply(_slot_label)

    return table, cap_df

# ============================================================
# BUILD DAILY SCHEDULE
# ============================================================

def build_daily_schedule(solution):
    schedule = {}

    for pv in PHYSICAL_VEHICLES:
        vehicle_free_time = 0.0
        prev_last_node   = None
        prev_last_ntimes = None
        trips = []

        for v in SLOTS_OF_VEHICLE[pv]:
            out_route = solution.outbound_routes[v]
            ret_route = solution.return_routes[v]

            has_out = any(n in PATIENTS for n in out_route)
            has_ret = any(n in PATIENTS for n in ret_route)

            s_trans = solution.shift_transitions.get(v, None)

            actual_first = _first_node(out_route)
            s_start = _shift_transition_start_time(
                s_trans, prev_last_node, prev_last_ntimes, actual_first, expected_phase='OUT'
            )
            out_start = vehicle_free_time if s_start is None else s_start
            shift_transition = (prev_last_node, actual_first) if s_start is not None else None

            _, _, _, _, out_ntimes = evaluate_route(out_route, v, 'OUT', out_start)
            out_finish = out_ntimes.get(DEPOT, out_start)

            start_time = out_finish
            trans = solution.transitions.get(v, None)
            h_out = h_ret = None
            if trans is not None:
                h_out, h_ret = trans
                last_out  = _last_outbound_hospital(out_route)
                first_ret = _first_return_hospital(ret_route)
                if h_out == last_out and h_ret == first_ret and h_out in out_ntimes:
                    t_h_out         = out_ntimes[h_out]
                    departure_h_out = t_h_out + service[h_out]
                    transition_arr  = departure_h_out + arc_time(h_out, h_ret)
                    d_depot_hret    = arc_time(DEPOT, h_ret)
                    start_time      = max(0.0, transition_arr - d_depot_hret)
                else:
                    trans = None
                    start_time = out_finish
            else:
                actual_first_ret = _first_node(ret_route)
                s_start_ret = _shift_transition_start_time(
                    s_trans, prev_last_node, prev_last_ntimes, actual_first_ret, expected_phase='RET'
                )
                if s_start_ret is not None:
                    start_time = s_start_ret
                    shift_transition = (prev_last_node, actual_first_ret)

            _, _, _, _, ret_ntimes = evaluate_route(ret_route, v, 'RET', start_time)
            ret_finish = ret_ntimes.get(DEPOT, start_time)

            if has_out or has_ret:
                trips.append(dict(
                    slot=v, out_route=out_route, ret_route=ret_route,
                    out_start=out_start, out_finish=out_finish,
                    ret_start=start_time, ret_finish=ret_finish,
                    transition=(h_out, h_ret) if trans is not None else None,
                    shift_transition=shift_transition,
                    has_out=has_out, has_ret=has_ret,
                ))

            vehicle_free_time = ret_finish

            prev_last_node, prev_last_ntimes = _advance_shift_position(
                out_route, ret_route, out_ntimes, ret_ntimes, prev_last_node, prev_last_ntimes
            )

        if trips:
            schedule[pv] = trips

    return schedule

# ============================================================
# INITIAL ROUTES
# ============================================================

def initial_routes():
    out_routes = {v: [DEPOT, DEPOT] for v in VEHICLES}
    ret_routes = {v: [DEPOT, DEPOT] for v in VEHICLES}
    out_load   = {v: 0 for v in VEHICLES}
    ret_load   = {v: 0 for v in VEHICLES}

    for p in PATIENTS:
        eligible = [v for v in VEHICLES if compatO[p, v] == 1]
        if eligible:
            v = min(eligible, key=lambda x: out_load[x])
            h = hospital[p]
            out_routes[v] = out_routes[v][:-1] + [p, h, DEPOT]
            out_load[v] += 1

    for p in PATIENTS:
        eligible = [v for v in VEHICLES if compatR[p, v] == 1]
        if eligible:
            v = min(eligible, key=lambda x: ret_load[x])
            h = hospital[p]
            ret_routes[v] = ret_routes[v][:-1] + [h, p, DEPOT]
            ret_load[v] += 1

    return out_routes, ret_routes

def randomized_initial_routes(rng=random):
    """Same greedy least-loaded-slot construction as initial_routes(), but
    patients are considered in a shuffled order each call -- used by GA/ACO
    to build a diverse initial population instead of one fixed construction."""
    out_routes = {v: [DEPOT, DEPOT] for v in VEHICLES}
    ret_routes = {v: [DEPOT, DEPOT] for v in VEHICLES}
    out_load   = {v: 0 for v in VEHICLES}
    ret_load   = {v: 0 for v in VEHICLES}

    out_order = list(PATIENTS)
    ret_order = list(PATIENTS)
    rng.shuffle(out_order)
    rng.shuffle(ret_order)

    for p in out_order:
        eligible = [v for v in VEHICLES if compatO[p, v] == 1]
        if eligible:
            v = min(eligible, key=lambda x: out_load[x])
            h = hospital[p]
            out_routes[v] = out_routes[v][:-1] + [p, h, DEPOT]
            out_load[v] += 1

    for p in ret_order:
        eligible = [v for v in VEHICLES if compatR[p, v] == 1]
        if eligible:
            v = min(eligible, key=lambda x: ret_load[x])
            h = hospital[p]
            ret_routes[v] = ret_routes[v][:-1] + [h, p, DEPOT]
            ret_load[v] += 1

    return out_routes, ret_routes

# ============================================================
# REMOVE PATIENT
# ============================================================

def patient_slot(routes, p):
    """The vehicle slot currently containing patient p in this routes dict
    (outbound OR return, whichever is passed), or None if unassigned."""
    for v in VEHICLES:
        if p in routes[v]:
            return v
    return None

def remove_patient(route, p):
    h = hospital[p]
    new_route = [n for n in route if n != p]
    remaining = [x for x in new_route if x in PATIENTS]
    still_uses_h = any(hospital[q] == h for q in remaining)
    if not still_uses_h:
        new_route = [n for n in new_route if n != h]
    new_route = clean_route(new_route)
    new_route = remove_redundant_hospitals(new_route)
    return new_route

# ============================================================
# DESTROY
# ============================================================

def remove_specific_patients(solution, patients_to_remove):
    """Same per-patient removal logic as destroy() (removes each patient from
    whatever slot(s) it's in via remove_patient(), clears stale transitions
    for touched slots) but for a CALLER-CHOSEN list of patients instead of a
    random sample -- used by ts_darp.py (specific tabu-neighborhood
    candidates) and ga_darp.py (specific duplicated/missing patients from a
    crossover child) where the random sampling in destroy() doesn't apply."""
    out_routes  = copy.deepcopy(solution.outbound_routes)
    ret_routes  = copy.deepcopy(solution.return_routes)
    transitions = copy.deepcopy(solution.transitions)
    shift_transitions = copy.deepcopy(solution.shift_transitions)

    modified = set()
    for p in patients_to_remove:
        for v in VEHICLES:
            if p in out_routes[v]:
                out_routes[v] = remove_patient(out_routes[v], p)
                modified.add(v)
            if p in ret_routes[v]:
                ret_routes[v] = remove_patient(ret_routes[v], p)
                modified.add(v)

    for v in modified:
        transitions.pop(v, None)
        shift_transitions.pop(v, None)

    return out_routes, ret_routes, list(patients_to_remove), transitions, shift_transitions

def destroy(solution, remove_count):
    """Removes up to `remove_count` random SERVED patients from both out/ret
    routes; clears stale transitions for touched slots. `remove_count` is a
    parameter (not a global) so callers can use a large-neighborhood LNS-
    style removal or a small 1-2 patient removal (e.g. GA mutation)."""
    out_routes  = copy.deepcopy(solution.outbound_routes)
    ret_routes  = copy.deepcopy(solution.return_routes)
    transitions = copy.deepcopy(solution.transitions)
    shift_transitions = copy.deepcopy(solution.shift_transitions)

    served  = [p for p in PATIENTS if p not in solution.unserved]
    removed = random.sample(served, min(remove_count, len(served)))

    # Always give every currently-unserved patient another shot at insertion
    # this iteration too -- otherwise, once repair() declines an insertion
    # (a legitimate outcome now that it compares cost against
    # UNSERVED_PENALTY), that patient is never sampled by destroy() again
    # (it only draws from `served`), leaving them permanently unserved even
    # if a later perturbation elsewhere in the routes would free up room.
    removed = removed + list(solution.unserved)

    modified = set()

    for p in removed:
        for v in VEHICLES:
            if p in out_routes[v]:
                out_routes[v] = remove_patient(out_routes[v], p)
                modified.add(v)
            if p in ret_routes[v]:
                ret_routes[v] = remove_patient(ret_routes[v], p)
                modified.add(v)

    # Clear transitions for vehicles whose routes changed
    for v in modified:
        transitions.pop(v, None)
        shift_transitions.pop(v, None)

    return out_routes, ret_routes, removed, transitions, shift_transitions

def destroy_worst(solution, remove_count, worst_fraction=0.5, rng=random):
    """'Worst removal' destroy operator -- identifies served patients with
    the largest ride-time/timing violations (via build_diagnostics) and
    removes the worst `remove_count` of them. Candidates are drawn from a
    slightly larger pool (remove_count / worst_fraction) and then randomly
    sampled down, so the operator doesn't attack the exact same patients
    every single time it's selected."""
    served = [p for p in PATIENTS if p not in solution.unserved]
    if not served:
        return remove_specific_patients(solution, [])

    diagnostics, _ = build_diagnostics(solution)
    diagnostics = diagnostics.set_index('patient')

    def badness(p):
        if p not in diagnostics.index:
            return 0.0
        row = diagnostics.loc[p]
        score = 0.0
        if bool(row.get('out_ride_violation', False)):
            score += 1000.0 + float(row.get('out_ride_time', 0.0) or 0.0)
        if bool(row.get('ret_ride_violation', False)):
            score += 1000.0 + float(row.get('ret_ride_time', 0.0) or 0.0)
        if bool(row.get('out_early_violation', False)):
            score += 1000.0 + float(row.get('early_minutes', 0.0) or 0.0)
        if bool(row.get('out_late_violation', False)):
            score += 1000.0 + float(row.get('late_minutes', 0.0) or 0.0)
        if bool(row.get('return_delay_violation', False)):
            score += 1000.0 + float(row.get('return_delay', 0.0) or 0.0)
        return score

    ranked = sorted(served, key=badness, reverse=True)
    pool_size = min(len(ranked), max(remove_count, int(remove_count / max(worst_fraction, 1e-6))))
    pool = ranked[:pool_size]
    chosen = rng.sample(pool, min(remove_count, len(pool)))
    # Always give unserved patients another shot too (see destroy()).
    chosen = chosen + list(solution.unserved)
    return remove_specific_patients(solution, chosen)

def destroy_related(solution, remove_count, rng=random):
    """'Related/Shaw removal' destroy operator -- picks a random served
    patient as a seed, then removes the remove_count-1 OTHER served
    patients most 'related' to it (same hospital preferred, close
    appointment time), forming a coherent cluster to jointly re-optimize
    instead of remove_count unrelated random patients."""
    served = [p for p in PATIENTS if p not in solution.unserved]
    # Always give unserved patients another shot too (see destroy()).
    if not served:
        return remove_specific_patients(solution, list(solution.unserved))
    if len(served) <= remove_count:
        return remove_specific_patients(solution, served + list(solution.unserved))

    seed_patient = rng.choice(served)

    def relatedness(p):
        same_hospital_penalty = 0.0 if hospital[p] == hospital[seed_patient] else 500.0
        time_gap = abs(appointment[p] - appointment[seed_patient])
        return same_hospital_penalty + time_gap

    others = [p for p in served if p != seed_patient]
    others.sort(key=relatedness)
    chosen = [seed_patient] + others[:remove_count - 1] + list(solution.unserved)
    return remove_specific_patients(solution, chosen)

# ============================================================
# BEST INSERTION
# ============================================================

def best_insertion(patient, routes, phase, compat, start_times, candidate_vehicles=None):
    """Brute-force cheapest feasible insertion position across compatible
    vehicle slots. `candidate_vehicles`, if given, restricts the search to
    that subset instead of all of VEHICLES (default None = search everything,
    IDENTICAL to the original behavior) -- used by aco_darp.py to score a
    single (patient, slot) pair without re-scanning every vehicle.
    Returns (best_vehicle, best_route, best_score) -- best_score is the raw
    route_score() of the winning insertion (None if no feasible insertion was
    found), used by aco_darp.py as the desirability signal. Existing callers
    that only want the first two values can unpack with a trailing `_`."""
    best_score   = 1e18
    best_vehicle = None
    best_route   = None
    h = hospital[patient]

    search_space = VEHICLES if candidate_vehicles is None else candidate_vehicles

    for v in search_space:
        if compat[patient, v] == 0:
            continue
        inner = routes[v][1:-1]
        v_start = start_times.get(v, 0.0)

        if phase == 'OUT':
            n = len(inner)
            for p_pos in range(n + 1):
                inner_p = inner[:p_pos] + [patient] + inner[p_pos:]
                for h_pos in range(p_pos + 1, len(inner_p) + 1):
                    trial_inner = inner_p[:h_pos] + [h] + inner_p[h_pos:]
                    if not valid_order(trial_inner, 'OUT'):
                        continue
                    trial = [DEPOT] + trial_inner + [DEPOT]
                    trial = clean_route(trial)
                    trial = remove_redundant_hospitals(trial)
                    if not route_structure_ok(trial, 'OUT'):
                        continue
                    if not route_consistency_ok(trial, 'OUT'):
                        continue
                    if not route_capacity_ok(trial, v, 'OUT'):
                        continue
                    if not route_ontime_ok(trial, v, 'OUT', v_start):
                        continue
                    if not route_within_day_ok(trial, v, 'OUT', v_start):
                        continue
                    s = route_score(trial, v, 'OUT', v_start)
                    if s < best_score:
                        best_score   = s
                        best_vehicle = v
                        best_route   = trial

        else:
            n = len(inner)
            for h_pos in range(n + 1):
                inner_h = inner[:h_pos] + [h] + inner[h_pos:]
                for p_pos in range(h_pos + 1, len(inner_h) + 1):
                    trial_inner = inner_h[:p_pos] + [patient] + inner_h[p_pos:]
                    if not valid_order(trial_inner, 'RET'):
                        continue
                    trial = [DEPOT] + trial_inner + [DEPOT]
                    trial = clean_route(trial)
                    trial = remove_redundant_hospitals(trial)
                    if not route_structure_ok(trial, 'RET'):
                        continue
                    if not route_consistency_ok(trial, 'RET'):
                        continue
                    if not route_capacity_ok(trial, v, 'RET'):
                        continue
                    if not route_within_day_ok(trial, v, 'RET', v_start):
                        continue
                    s = route_score(trial, v, 'RET', v_start)
                    if s < best_score:
                        best_score   = s
                        best_vehicle = v
                        best_route   = trial

    return best_vehicle, best_route, (None if best_vehicle is None else best_score)

# ============================================================
# REPAIR
# ============================================================

def repair(out_routes, ret_routes, removed, transitions, shift_transitions, naive_insertion=False,
           max_candidate_vehicles=None, disable_shift_transitions=False):
    """max_candidate_vehicles, if given, caps best_insertion's per-patient search
    to a random sample of that many COMPATIBLE slots instead of scanning every
    one -- a standard candidate-list / granular-neighborhood technique for
    keeping per-iteration cost bounded as instance size (and the number of
    dynamically-grown shift slots) grows. Default None = unrestricted, exactly
    the original exhaustive search -- existing callers are unaffected unless
    they opt in.

    A removed patient is only re-inserted if BOTH legs find a feasible slot
    AND their combined marginal cost is no worse than UNSERVED_PENALTY --
    mirrors the exact MIP's single u[p] binary, which ties outbound and
    return together (serve both or neither). Either leg failing, or the
    combined cost exceeding UNSERVED_PENALTY, leaves the patient out of BOTH
    routes for this pass, so a patient is never left served on only one leg."""
    for p in removed:
        # Recompute each slot's real chained start time before every single
        # insertion decision, so a vehicle's later shifts are correctly
        # scored as starting later than its first shift (not as if every
        # slot were equally available at time 0).
        if naive_insertion:
            out_start, ret_start = {}, {}   # ablation: every slot scores as if start_time=0.0
        else:
            out_start, ret_start = compute_slot_start_times(out_routes, ret_routes, transitions, shift_transitions)

        out_candidates = ret_candidates = None
        if max_candidate_vehicles is not None:
            out_pool = [v for v in VEHICLES if compatO[p, v] == 1]
            ret_pool = [v for v in VEHICLES if compatR[p, v] == 1]
            out_candidates = out_pool if len(out_pool) <= max_candidate_vehicles \
                else random.sample(out_pool, max_candidate_vehicles)
            ret_candidates = ret_pool if len(ret_pool) <= max_candidate_vehicles \
                else random.sample(ret_pool, max_candidate_vehicles)

        out_v, out_r, out_score = best_insertion(p, out_routes, 'OUT', compatO, out_start, candidate_vehicles=out_candidates)
        ret_v, ret_r, ret_score = best_insertion(p, ret_routes, 'RET', compatR, ret_start, candidate_vehicles=ret_candidates)

        if out_v is None or ret_v is None:
            continue   # can't serve both legs -> patient stays unserved this pass

        out_baseline = route_score(out_routes[out_v], out_v, 'OUT', out_start.get(out_v, 0.0))
        ret_baseline = route_score(ret_routes[ret_v], ret_v, 'RET', ret_start.get(ret_v, 0.0))
        combined_marginal_cost = (out_score - out_baseline) + (ret_score - ret_baseline)

        if combined_marginal_cost <= UNSERVED_PENALTY:
            out_routes[out_v] = out_r
            ret_routes[ret_v] = ret_r
        # else: leave unserved this pass -- cheaper to pay UNSERVED_PENALTY once

    # Re-evaluate which vehicles benefit from a direct transition (within a
    # shift, and -- unless disable_shift_transitions -- between consecutive
    # shifts of the same physical vehicle too). The inter-shift case isn't
    # modeled in the exact MIP benchmark, so this flag lets LNS be run
    # restricted to exactly what the exact model represents, for an
    # apples-to-apples comparison.
    transitions = try_add_transitions(out_routes, ret_routes, transitions)
    if not disable_shift_transitions:
        shift_transitions = try_add_shift_transitions(out_routes, ret_routes, shift_transitions)

    return out_routes, ret_routes, transitions, shift_transitions

# ============================================================
# CONVENIENCE WRAPPERS SHARED ACROSS ALL 5 ALGORITHMS
# ============================================================

def finalize_solution(out_routes, ret_routes, transitions=None, shift_transitions=None):
    """try_add_transitions -> try_add_shift_transitions -> evaluate_solution
    in one call, so every algorithm's 'I just mutated routes, now give me a
    scored Solution' step is one line instead of three repeated everywhere."""
    transitions       = try_add_transitions(out_routes, ret_routes, transitions)
    shift_transitions = try_add_shift_transitions(out_routes, ret_routes, shift_transitions)
    return evaluate_solution(out_routes, ret_routes, transitions, shift_transitions)

def is_better(candidate, incumbent):
    """The 'best' bookkeeping rule darp_lns.py's own loop uses inline
    (lower objective wins -- UNSERVED_PENALTY is already folded into
    objective, so this also prefers serving more patients in practice).
    Factored out so all 5 algorithms report 'best' identically."""
    return candidate.objective < incumbent.objective

def metropolis_accept(delta, temperature):
    """Shared Metropolis acceptance rule (darp_lns.py's own LNS loop and
    sa_darp.py's small-step local search both use this identical formula)."""
    return delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-6))

# ============================================================
# MOVE-CANDIDATE GENERATORS (shared by SA / TS local-search neighborhoods)
# ============================================================

def _served_patients(out_routes):
    return [p for p in PATIENTS if any(p in out_routes[v] for v in VEHICLES)]

def random_relocate_candidates(out_routes, ret_routes, k, rng=random):
    """Up to k random served patients, each a candidate for a single-patient
    relocate move (remove + best_insertion elsewhere)."""
    served = _served_patients(out_routes)
    if not served:
        return []
    return rng.sample(served, min(k, len(served)))

def random_swap_candidates(out_routes, ret_routes, k, rng=random):
    """Up to k random (patientA, patientB) DISTINCT-pair candidates for a
    two-patient swap move (remove both, reinsert both via best_insertion)."""
    served = _served_patients(out_routes)
    pairs = []
    if len(served) < 2:
        return pairs
    for _ in range(k):
        a, b = rng.sample(served, 2)
        pairs.append((a, b))
    return pairs

# ============================================================
# TIME-BUDGET / STOPPING CRITERION
# ============================================================

def check_time_budget(t_start, time_budget_sec):
    """time_budget_sec is None -> never stop on time (used when a script
    doesn't pass --time_budget_sec, preserving iteration/EMA-only behavior)."""
    if time_budget_sec is None:
        return False
    import time as _time
    return (_time.time() - t_start) >= time_budget_sec

class StoppingCriterion:
    """Shared stop-condition helper for the four baselines (which have no
    prior EMA-based stopping to preserve). Stops on whichever of
    time_budget_sec / max_iterations / EMA-plateau fires first. Any of the
    three may be disabled by passing None (time_budget_sec, max_iterations)
    or leaving ema_patience at its default (always enabled once past
    ema_slow_window iterations -- pass ema_patience=None to disable EMA
    entirely and rely only on time/iteration limits)."""

    def __init__(self, time_budget_sec=None, max_iterations=None,
                 ema_fast_window=30, ema_slow_window=150, ema_eps=0.01, ema_patience=5000):
        import time as _time
        self._time = _time
        self.time_budget_sec = time_budget_sec
        self.max_iterations  = max_iterations
        self.alpha_fast = 2.0 / (ema_fast_window + 1)
        self.alpha_slow = 2.0 / (ema_slow_window + 1)
        self.ema_eps      = ema_eps
        self.ema_patience = ema_patience
        self.warmup       = ema_slow_window
        self.ema_fast = None
        self.ema_slow = None
        self.stale_checks = 0
        self.t_start = None

    def start(self):
        self.t_start = self._time.time()
        return self

    def step(self, it, objective):
        """Call once per iteration/generation/colony-round. Returns
        (stop: bool, reason: str | None)."""
        if self.max_iterations is not None and it >= self.max_iterations:
            return True, "max iterations reached"

        if self.time_budget_sec is not None and (self._time.time() - self.t_start) >= self.time_budget_sec:
            return True, f"time budget reached ({self.time_budget_sec:.1f}s)"

        if self.ema_patience is not None:
            self.ema_fast = objective if self.ema_fast is None else self.alpha_fast * objective + (1 - self.alpha_fast) * self.ema_fast
            self.ema_slow = objective if self.ema_slow is None else self.alpha_slow * objective + (1 - self.alpha_slow) * self.ema_slow
            if it > self.warmup:
                rel_gap = (self.ema_slow - self.ema_fast) / self.ema_slow if self.ema_slow != 0 else 0.0
                self.stale_checks = self.stale_checks + 1 if rel_gap < self.ema_eps else 0
                if self.stale_checks >= self.ema_patience:
                    return True, (
                        f"EMA trend flattened (gap={rel_gap:.4%} < {self.ema_eps:.2%} "
                        f"for {self.ema_patience} consecutive iterations)"
                    )

        return False, None

# ============================================================
# EXPORT RESULTS (shared by all 5 CLIs -- guarantees identical schema)
# ============================================================

def export_run(export_dir, best_solution, history_rows, meta):
    """Writes history.csv, diagnostics.csv, capacity_violations.csv,
    daily_schedule.csv, summary.json -- the same set of files/columns
    darp_lns.py's own --export_dir block writes. `history_rows` is a list
    of dicts (columns = whatever the caller chooses to log per iteration/
    generation/colony-round). `meta` carries run-level fields (instance,
    seed, algorithm, stop_reason, iterations_run, wall_time_sec, ...) that
    get merged into summary.json alongside the Solution-derived fields.
    Returns the daily_schedule dict in case the caller wants to print it."""
    os.makedirs(export_dir, exist_ok=True)

    pd.DataFrame(history_rows).to_csv(os.path.join(export_dir, 'history.csv'), index=False)

    diagnostics_table, capacity_violations = build_diagnostics(best_solution)
    diagnostics_table.to_csv(os.path.join(export_dir, 'diagnostics.csv'), index=False)
    capacity_violations.to_csv(os.path.join(export_dir, 'capacity_violations.csv'), index=False)

    daily_schedule = build_daily_schedule(best_solution)
    schedule_rows = []
    for pv, trips in daily_schedule.items():
        for i, trip in enumerate(trips, start=1):
            schedule_rows.append(dict(
                vehicle=pv, trip=i,
                out_start=trip['out_start'], out_finish=trip['out_finish'],
                ret_start=trip['ret_start'], ret_finish=trip['ret_finish'],
                out_start_hhmm=fmt(trip['out_start']), out_finish_hhmm=fmt(trip['out_finish']),
                ret_start_hhmm=fmt(trip['ret_start']), ret_finish_hhmm=fmt(trip['ret_finish']),
                has_transition=trip['transition'] is not None,
                has_shift_transition=trip['shift_transition'] is not None,
            ))
    pd.DataFrame(schedule_rows).to_csv(os.path.join(export_dir, 'daily_schedule.csv'), index=False)

    violated = diagnostics_table[diagnostics_table['any_violation'] == True]

    summary = dict(
        objective=best_solution.objective, travel_cost=best_solution.travel_cost,
        ride_penalty=best_solution.ride_penalty, violation_cost=best_solution.violation_cost,
        served=best_solution.num_served, num_patients=NUM_PATIENTS,
        feasible=best_solution.feasible, num_transitions=len(best_solution.transitions),
        num_shift_transitions=len(best_solution.shift_transitions),
        num_violated_patients=int(len(violated)), num_capacity_events=int(len(capacity_violations)),
        out_early_violations=int(diagnostics_table.get('out_early_violation', pd.Series(dtype=bool)).fillna(False).sum()),
        out_late_violations=int(diagnostics_table.get('out_late_violation', pd.Series(dtype=bool)).fillna(False).sum()),
        out_ride_violations=int(diagnostics_table.get('out_ride_violation', pd.Series(dtype=bool)).fillna(False).sum()),
        return_delay_violations=int(diagnostics_table.get('return_delay_violation', pd.Series(dtype=bool)).fillna(False).sum()),
        ret_ride_violations=int(diagnostics_table.get('ret_ride_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    )
    summary.update(meta)

    with open(os.path.join(export_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return daily_schedule
