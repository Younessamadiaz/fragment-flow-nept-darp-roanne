# ============================================================
# HEALTHCARE DIAL-A-RIDE PROBLEM
# Bi-Level LNS-DARP with Outbound-to-Return Transitions
#
# Thin driver: instance loading, the Solution representation, all
# constraint/evaluation logic, and the destroy/repair primitives live in
# darp_core.py (shared with the SA/TS/GA/ACO baselines) -- this file is
# just the LNS-specific search loop (large-neighborhood destroy/repair
# with Metropolis/EMA-based acceptance) plus reporting/export.
# ============================================================

import argparse
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import darp_core as dc

# ============================================================
# CONFIGURATION
# ============================================================
_parser = argparse.ArgumentParser()

_parser.add_argument(
    "--instance", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances", "E_20_5_1", "E_20_5_1_model_input.xlsx"),
    help="Path to a *_model_input.xlsx instance file.",
)

_parser.add_argument("--seed",       type=int, default=42)
_parser.add_argument("--iterations", type=int, default=5000)
_parser.add_argument("--shifts",     type=int, default=2,
                      help="STARTING number of sequential trips (an outbound leg plus "
                           "a return leg) each physical vehicle has available. The "
                           "search grows this on demand -- a vehicle only gets a new "
                           "trip slot once its current last one actually gets used --"
                           " up to --max_shifts, so you don't have to guess the right "
                           "ceiling up front.")
_parser.add_argument("--max_shifts", type=int, default=12,
                      help="Safety cap on how many trips a vehicle may grow to in one "
                           "day. Generous by design -- it should rarely if ever bind; "
                           "it just prevents runaway growth.")
_parser.add_argument("--day_start_buffer_hours", type=float, default=1.0,
                      help="The single daily shift begins this many hours before the "
                           "EARLIEST appointment in the instance (auto-derived from the "
                           "data, not a fixed clock time) -- enough slack to actually "
                           "use the Omega early-arrival window instead of already "
                           "starting behind schedule.")
_parser.add_argument("--day_end_buffer_hours", type=float, default=2.0,
                      help="The single daily shift ends this many hours after the "
                           "LATEST return-pickup deadline in the instance (appointment "
                           "+ treatment duration + Psi_p, auto-derived from the data). "
                           "No trip may extend past this -- it's a hard constraint, "
                           "same as capacity.")
_parser.add_argument("--naive_insertion", action="store_true",
                      help="Ablation switch: score candidate insertions as if every "
                           "shift slot starts at time 0, ignoring real shift-chaining "
                           "(reproduces the pre-fix search blind spot for comparison).")
_parser.add_argument("--disable_shift_transitions", action="store_true",
                      help="Restrict LNS to within-shift depot-skip transitions only, "
                           "disabling the inter-shift ones (chaining consecutive shifts "
                           "of the same vehicle without a depot stop in between). Use "
                           "this to match exact_mip.py's modeling scope exactly, for an "
                           "apples-to-apples comparison -- exact_mip.py doesn't represent "
                           "inter-shift transitions, so with this flag LNS's search space "
                           "matches it precisely. Default off = full LNS capability.")
_parser.add_argument("--max_candidate_vehicles", type=int, default=None,
                      help="If given, best_insertion searches only a random sample of "
                           "this many compatible vehicle slots per removed patient "
                           "instead of scanning every one -- a standard candidate-list/"
                           "granular-neighborhood technique that keeps per-iteration cost "
                           "bounded as instance size and shift-slot count grow. Default "
                           "None = unrestricted exhaustive search (original behavior).")
_parser.add_argument("--remove_ratio", type=float, default=0.20,
                      help="Fraction of patients removed per destroy() call "
                           "(REMOVE_COUNT = max(2, int(NUM_PATIENTS*remove_ratio))). "
                           "Default 0.20 matches original behavior. Larger values give "
                           "each iteration a bigger neighborhood to jointly restructure, "
                           "which can help escape local optima where several patients' "
                           "slots need to move together to free room for one more --  "
                           "at the cost of a more expensive repair() per iteration.")
_parser.add_argument("--remove_cap", type=int, default=None,
                      help="If given, caps REMOVE_COUNT (normally "
                           "max(2, int(NUM_PATIENTS*remove_ratio)), unbounded) at this "
                           "absolute number of patients per iteration -- keeps each "
                           "iteration's insertion-search cost from growing linearly with "
                           "instance size, so larger instances get proportionally more "
                           "iterations in a fixed time budget without sacrificing "
                           "per-patient insertion search quality (unlike "
                           "--max_candidate_vehicles). Default None = unbounded "
                           "(original behavior).")
_parser.add_argument("--adaptive", action="store_true",
                      help="Optional adaptive operator selection, not used in the reported experiments: instead of always using uniform "
                           "random removal, choose each iteration's destroy operator "
                           "from {random, worst-cost, related/Shaw removal} via "
                           "roulette-wheel selection over adaptively-updated weights "
                           "(reward = finding a new global best > improving current > "
                           "accepted-but-worse > rejected). Default off = exactly the "
                           "original single-operator behavior.")
_parser.add_argument("--adaptive_reaction", type=float, default=0.2,
                      help="Weight update reaction factor (0-1): how much each "
                           "operator's weight moves toward its latest reward vs. its "
                           "prior smoothed value.")
_parser.add_argument("--temperature", type=float, default=500.0,
                      help="Initial simulated-annealing temperature for Metropolis "
                           "acceptance of worsening moves. Default 500 matches original "
                           "behavior.")
_parser.add_argument("--cooling", type=float, default=0.9995,
                      help="Per-iteration multiplicative cooling rate applied to the "
                           "temperature. Default 0.9995 matches original behavior.")
_parser.add_argument("--weighted_qos", action="store_true",
                      help="Weight ride_penalty per-patient by activity type instead of "
                           "counting every patient's ride time equally, using "
                           "dc.PATIENT_TYPE_WEIGHT (dialysis 2.0, oncology 1.75, "
                           "therapeutic 1.25, other 1.0, external 0.75 -- grounded in the "
                           "model's own existing Delta_p ride-time-cap ordering). Default "
                           "off = every patient weight 1.0, original behavior unchanged.")
_parser.add_argument("--lambda_override", type=float, default=None,
                      help="If given, replaces the instance file's own lambda (the "
                           "ride-time/quality-of-service weight in the objective, "
                           "route_score = travel_cost + lambda*ride_penalty + viol) "
                           "with this value, e.g. 0 to stop the SEARCH from optimizing "
                           "ride time. Note ride_penalty is still computed and reported "
                           "either way, and the hard per-patient ride-time cap (Delta_p) "
                           "is a separate constraint that stays enforced regardless of "
                           "this flag. Default None = use the instance file's own value "
                           "unchanged.")
_parser.add_argument("--disable_transitions", action="store_true",
                      help="Depot-based baseline: never activate intra-shift or inter-shift "
                           "transitions, so every outbound and return fragment starts and "
                           "ends at the depot. The number of shifts and all other settings "
                           "are unchanged.")
_parser.add_argument("--no_qos", action="store_true",
                      help="Remove the whole patient-QoS layer from the SEARCH objective: "
                           "lambda=0 AND zero penalty on excess ride time (Delta_p), early "
                           "arrival (OMEGA) and return delay (PSI). Lateness past the "
                           "appointment, capacity, sharing and unserved penalties are kept, "
                           "so the search becomes a pure cost-minimizing DARP. The best "
                           "solution is re-evaluated under the ORIGINAL lambda and full "
                           "penalties before reporting/export, so objective, ride time and "
                           "violation_cost are directly comparable with a normal run.")
_parser.add_argument("--sharing_rate_override", type=float, default=None,
                      help="If given, replaces every patient's sharing_possible flag "
                           "(loaded from the instance file's own per-activity-type rates) "
                           "with an i.i.d. draw at this single probability (1=shareable). "
                           "Used to run a matched sharing-rate ablation on the SAME "
                           "instance files used elsewhere (e.g. the main stochastic case "
                           "study), instead of regenerating a fresh instance population at "
                           "the target rate -- avoids confounding the sharing-rate effect "
                           "with a different random draw of patients/fleet. Seeded "
                           "deterministically from (--instance path, this rate), so the "
                           "override is IDENTICAL regardless of --lambda_override -- a "
                           "with-QoS/without-QoS pair on the same instance and rate always "
                           "gets the same sharing assignment, keeping that comparison "
                           "paired on everything except lambda. Default None = use the "
                           "instance file's own sharing_possible values unchanged.")
_parser.add_argument("--export_dir", type=str, default=None,
                      help="If given, write the per-iteration objective history and "
                           "final result tables (CSV/JSON) to this directory for later "
                           "analysis/plotting.")
_parser.add_argument("--time_budget_sec", type=float, default=None,
                      help="If given, stop the moment this many wall-clock seconds have "
                           "elapsed since the search loop started, regardless of the EMA/"
                           "iteration criteria below -- whichever stop condition fires "
                           "first wins. Omitted (default None) leaves behavior exactly as "
                           "before this flag existed. Used to give every baseline "
                           "algorithm (SA/TS/GA/ACO) the same wall-clock search budget "
                           "for a fair comparison.")

# ---- EMA stopping criterion ----
_parser.add_argument("--ema_fast_window", type=int,   default=30,
                      help="Effective window size for the fast EMA (alpha = 2/(W+1)).")
_parser.add_argument("--ema_slow_window", type=int,   default=150,
                      help="Effective window size for the slow EMA (alpha = 2/(W+1)).")
_parser.add_argument("--ema_eps",         type=float, default=0.001,
                      help="Minimum relative gap between slow and fast EMA to reset patience.")
_parser.add_argument("--ema_patience",    type=int,   default=1500,
                      help="Consecutive iterations below ema_eps before stopping early.")

_args, _ = _parser.parse_known_args()

# ============================================================
# PARAMETERS
# ============================================================

ITERATIONS = _args.iterations
SEED       = _args.seed

EMA_FAST_WINDOW = _args.ema_fast_window
EMA_SLOW_WINDOW = _args.ema_slow_window
EMA_EPS         = _args.ema_eps
EMA_PATIENCE    = _args.ema_patience

ALPHA_FAST = 2.0 / (EMA_FAST_WINDOW + 1)
ALPHA_SLOW = 2.0 / (EMA_SLOW_WINDOW + 1)

# ============================================================
# IMPORTANT PARAMETERS
# ============================================================

REMOVE_RATIO = _args.remove_ratio

TEMPERATURE = _args.temperature
COOLING     = _args.cooling

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

DEPOT = dc.DEPOT

# ============================================================
# LOAD INSTANCE
# ============================================================

print(f"\nLoading instance from: {_args.instance}")
_info = dc.load_instance(
    _args.instance, shifts=_args.shifts, max_shifts=_args.max_shifts,
    day_start_buffer_hours=_args.day_start_buffer_hours,
    day_end_buffer_hours=_args.day_end_buffer_hours,
)

if _args.lambda_override is not None:
    print(f"Overriding instance lambda ({dc.LAMBDA:g}) with --lambda_override "
          f"{_args.lambda_override:g}")
    dc.LAMBDA = _args.lambda_override

if _args.disable_transitions:
    print("--disable_transitions: depot-based baseline (no intra- or inter-shift transitions)")
    dc.TRANSITIONS_ENABLED = False

_REPORT_LAMBDA = dc.LAMBDA
if _args.no_qos:
    print(f"--no_qos: search uses lambda=0 and QoS penalty scale 0 "
          f"(reporting re-evaluates under lambda={_REPORT_LAMBDA:g}, full penalties)")
    dc.LAMBDA = 0.0
    dc.QOS_PENALTY_SCALE = 0.0

if _args.sharing_rate_override is not None:
    import hashlib
    _seed_material = f"{_args.instance}|{_args.sharing_rate_override:.6f}".encode()
    _override_seed = int(hashlib.sha256(_seed_material).hexdigest(), 16) % (2**32)
    _sharing_rng = np.random.RandomState(_override_seed)
    dc.SHARING_POSSIBLE = {
        p: (1 if _sharing_rng.random() < _args.sharing_rate_override else 0)
        for p in sorted(dc.PATIENTS)
    }
    _n_shareable = sum(dc.SHARING_POSSIBLE.values())
    print(f"Overriding sharing_possible with rate={_args.sharing_rate_override:g} "
          f"(seed={_override_seed}, {_n_shareable}/{len(dc.SHARING_POSSIBLE)} patients "
          f"shareable) -- deterministic in (instance, rate), independent of --lambda_override")

if _args.weighted_qos:
    dc.PATIENT_QOS_WEIGHT = {
        p: dc.PATIENT_TYPE_WEIGHT.get(dc.ACTIVITY_TYPE[p], 1.0) for p in dc.PATIENTS
    }
    print(f"Weighted QoS enabled: {dc.PATIENT_TYPE_WEIGHT}")

NUM_PATIENTS  = dc.NUM_PATIENTS
NUM_HOSPITALS = dc.NUM_HOSPITALS
NUM_TYPES     = dc.NUM_TYPES

REMOVE_COUNT = max(2, int(NUM_PATIENTS * REMOVE_RATIO))
if _args.remove_cap is not None:
    REMOVE_COUNT = min(REMOVE_COUNT, _args.remove_cap)

print(f"Day window: {dc.fmt(0)} -> {dc.fmt(dc.DAY_LENGTH_MINUTES)}  "
      f"(starts {_args.day_start_buffer_hours:g}h before earliest appointment, "
      f"ends {_args.day_end_buffer_hours:g}h after latest return-pickup deadline)")

PHYSICAL_VEHICLES  = dc.PHYSICAL_VEHICLES
SHIFTS_PER_VEHICLE = dc.SHIFTS_PER_VEHICLE
MAX_SHIFTS         = dc.MAX_SHIFTS

_fleet_counts_str = ', '.join(f'{t}: {_info["fleet_counts"][t]}' for t in dc.VEHICLE_TYPES)
print(f"Patients  : {NUM_PATIENTS}")
print(f"Hospitals : {NUM_HOSPITALS}")
print(f"Vehicles  : {len(PHYSICAL_VEHICLES)}  ({_fleet_counts_str})")
print(f"Shifts    : start at {SHIFTS_PER_VEHICLE}, grow on demand up to {MAX_SHIFTS} trips/vehicle/day  "
      f"({len(dc.VEHICLES)} routing slots initially)")

# ============================================================
# INITIAL SOLUTION
# ============================================================

print("\nBuilding initial solution...")

out_routes, ret_routes = dc.initial_routes()
transitions       = dc.try_add_transitions(out_routes, ret_routes)
shift_transitions = {} if _args.disable_shift_transitions else dc.try_add_shift_transitions(out_routes, ret_routes)

current = dc.evaluate_solution(out_routes, ret_routes, transitions, shift_transitions)
best    = copy.deepcopy(current)

# Give every vehicle whose starting slot(s) are already full room to grow
# before the search even begins.
dc.ensure_growth_room(current, best)

print(f"Initial objective : {best.objective:.2f}")
print(f"Initial served    : {best.num_served}/{NUM_PATIENTS}")
print(f"Initial transitions : {len(best.transitions)}")
print(f"Initial shift transitions : {len(best.shift_transitions)}")

# ============================================================
# LNS
# ============================================================

print("\nRunning LNS...\n")

DESTROY_OPERATORS = {
    "random":  dc.destroy,
    "worst":   dc.destroy_worst,
    "related": dc.destroy_related,
}
REWARD_NEW_BEST      = 33.0
REWARD_IMPROVED      = 9.0
REWARD_ACCEPTED_WORSE = 1.0
operator_weights = {name: 1.0 for name in DESTROY_OPERATORS}
operator_uses    = {name: 0 for name in DESTROY_OPERATORS}

def _select_operator():
    names   = list(DESTROY_OPERATORS.keys())
    weights = [operator_weights[n] for n in names]
    return random.choices(names, weights=weights, k=1)[0]

temperature = TEMPERATURE
t_start     = time.time()

# ---- EMA stopping criterion state ----
ema_fast     = None
ema_slow     = None
stale_checks = 0
stop_reason  = "max iterations reached"
warmup       = EMA_SLOW_WINDOW    # let the slow EMA settle before checking

it = 0
history = []   # list of dicts: iteration, current_objective, best_objective, best_violation_cost, best_feasible

while it < ITERATIONS:

    dc.ensure_growth_room(current, best)

    if _args.adaptive:
        op_name = _select_operator()
        out_d, ret_d, removed, transitions_d, shift_transitions_d = DESTROY_OPERATORS[op_name](current, REMOVE_COUNT)
        operator_uses[op_name] += 1
    else:
        out_d, ret_d, removed, transitions_d, shift_transitions_d = dc.destroy(current, REMOVE_COUNT)

    out_r, ret_r, transitions_r, shift_transitions_r = dc.repair(
        out_d, ret_d, removed, transitions_d, shift_transitions_d,
        naive_insertion=_args.naive_insertion,
        max_candidate_vehicles=_args.max_candidate_vehicles,
        disable_shift_transitions=_args.disable_shift_transitions,
    )

    candidate = dc.evaluate_solution(out_r, ret_r, transitions_r, shift_transitions_r)

    delta = candidate.objective - current.objective
    accepted = dc.metropolis_accept(delta, temperature)

    if _args.adaptive:
        if candidate.objective < best.objective:
            reward = REWARD_NEW_BEST
        elif accepted and delta < 0:
            reward = REWARD_IMPROVED
        elif accepted:
            reward = REWARD_ACCEPTED_WORSE
        else:
            reward = 0.0
        r = _args.adaptive_reaction
        operator_weights[op_name] = max(0.1, (1 - r) * operator_weights[op_name] + r * reward)

    if accepted:
        current = candidate

    if candidate.objective < best.objective:
        best = copy.deepcopy(candidate)

    temperature *= COOLING
    it += 1

    history.append(dict(
        iteration=it, current_objective=current.objective, best_objective=best.objective,
        best_violation_cost=best.violation_cost, best_feasible=best.feasible,
    ))

    # ---- EMA update (O(1) per iteration, no buffer needed) ----
    O = current.objective
    ema_fast = O if ema_fast is None else ALPHA_FAST * O + (1 - ALPHA_FAST) * ema_fast
    ema_slow = O if ema_slow is None else ALPHA_SLOW * O + (1 - ALPHA_SLOW) * ema_slow

    if it > warmup:
        # positive gap = fast average still below slow average = still improving
        rel_gap = (ema_slow - ema_fast) / ema_slow if ema_slow != 0 else 0.0
        stale_checks = stale_checks + 1 if rel_gap < EMA_EPS else 0
        if stale_checks >= EMA_PATIENCE:
            stop_reason = (
                f"EMA trend flattened "
                f"(fast W={EMA_FAST_WINDOW}, slow W={EMA_SLOW_WINDOW}, "
                f"gap={rel_gap:.4%} < {EMA_EPS:.2%} for {EMA_PATIENCE} consecutive iterations)"
            )
            if (it % 100) != 0:
                elapsed = time.time() - t_start
                print(
                    f"Iter {it:4d} | "
                    f"Obj = {best.objective:10.2f} | "
                    f"Served = {best.num_served:3d}/{NUM_PATIENTS} | "
                    f"Viol = {best.violation_cost:10.2f} | "
                    f"Trans = {len(best.transitions):2d} | "
                    f"Feasible = {best.feasible} | "
                    f"Time = {elapsed:6.1f}s"
                )
            break

    if dc.check_time_budget(t_start, _args.time_budget_sec):
        stop_reason = f"time budget reached ({_args.time_budget_sec:.1f}s)"
        if (it % 100) != 0:
            elapsed = time.time() - t_start
            print(
                f"Iter {it:4d} | "
                f"Obj = {best.objective:10.2f} | "
                f"Served = {best.num_served:3d}/{NUM_PATIENTS} | "
                f"Viol = {best.violation_cost:10.2f} | "
                f"Trans = {len(best.transitions):2d} | "
                f"Feasible = {best.feasible} | "
                f"Time = {elapsed:6.1f}s"
            )
        break

    if it % 100 == 0:
        elapsed = time.time() - t_start
        print(
            f"Iter {it:4d} | "
            f"Obj = {best.objective:10.2f} | "
            f"Served = {best.num_served:3d}/{NUM_PATIENTS} | "
            f"Viol = {best.violation_cost:10.2f} | "
            f"Trans = {len(best.transitions):2d} | "
            f"Feasible = {best.feasible} | "
            f"Time = {elapsed:6.1f}s"
        )

print(f"\nStopped after {it} iterations: {stop_reason}")
if _args.adaptive:
    print("Operator usage / final weights:")
    for name in DESTROY_OPERATORS:
        print(f"  {name:<8} used={operator_uses[name]:5d}  weight={operator_weights[name]:.2f}")

if _args.no_qos:
    _search_objective = best.objective
    dc.LAMBDA = _REPORT_LAMBDA
    dc.QOS_PENALTY_SCALE = 1.0
    best = dc.evaluate_solution(best.outbound_routes, best.return_routes,
                                best.transitions, best.shift_transitions)
    print(f"--no_qos: re-evaluated best under full objective: "
          f"{_search_objective:.2f} (search) -> {best.objective:.2f} (reported)")

# ============================================================
# EXTRACT ASSIGNMENTS
# ============================================================

out_assignments = {}
ret_assignments = {}

for v in dc.VEHICLES:
    for n in best.outbound_routes[v]:
        if n in dc.PATIENTS:
            out_assignments[n] = v
    for n in best.return_routes[v]:
        if n in dc.PATIENTS:
            ret_assignments[n] = v

# ============================================================
# FINAL RESULTS
# ============================================================

SEP = "=" * 80

print("\n")
print(SEP)
print("FINAL SOLUTION")
print(SEP)
print(f"Objective       : {best.objective:.2f}")
print(f"Travel cost     : {best.travel_cost:.2f}")
print(f"Ride penalty    : {best.ride_penalty:.2f}")
print(f"Violation cost  : {best.violation_cost:.2f}")
print(f"Served patients : {best.num_served}/{NUM_PATIENTS}")
print(f"Unserved        : {len(best.unserved)}")
print(f"Feasible        : {best.feasible}")
print(f"Transitions     : {len(best.transitions)}")
_slot_counts = sorted(len(dc.SLOTS_OF_VEHICLE[pv]) for pv in PHYSICAL_VEHICLES)
print(f"Shifts grown to : min {_slot_counts[0]}, max {_slot_counts[-1]} "
      f"(started at {SHIFTS_PER_VEHICLE}, capped at {MAX_SHIFTS})")

# ============================================================
# OUTBOUND ASSIGNMENTS
# ============================================================

print("\n")
print(SEP)
print("OUTBOUND ASSIGNMENTS (UPPER LEVEL)")
print(SEP)
print(f"\n{'Patient':<12}{'Hospital':<12}{'Vehicle':<24}")
print("-" * 48)
for p in sorted(dc.PATIENTS):
    veh = out_assignments.get(p)
    veh = dc._slot_label(veh) if veh else "UNASSIGNED"
    print(f"{p:<12}{dc.hospital[p]:<12}{veh:<24}")

# ============================================================
# RETURN ASSIGNMENTS
# ============================================================

print("\n")
print(SEP)
print("RETURN ASSIGNMENTS (UPPER LEVEL)")
print(SEP)
print(f"\n{'Patient':<12}{'Hospital':<12}{'Vehicle':<24}")
print("-" * 48)
for p in sorted(dc.PATIENTS):
    veh = ret_assignments.get(p)
    veh = dc._slot_label(veh) if veh else "UNASSIGNED"
    print(f"{p:<12}{dc.hospital[p]:<12}{veh:<24}")

# ============================================================
# TRANSITIONS
# ============================================================

print("\n")
print(SEP)
print("OUTBOUND-TO-RETURN TRANSITIONS")
print(SEP)
if best.transitions:
    print(f"\n{'Vehicle trip':<24}{'From (h_out)':<16}{'To (h_ret)':<16}{'Distance Saved':>16}")
    print("-" * 72)
    for v, (h_out, h_ret) in sorted(best.transitions.items()):
        saved = (
            dc.arc_cost(h_out, DEPOT)
            + dc.arc_cost(DEPOT, h_ret)
            - dc.arc_cost(h_out, h_ret)
        )
        print(f"{dc._slot_label(v):<24}{h_out:<16}{h_ret:<16}{saved:>16.2f}")
else:
    print("\n  No direct transitions used.")

# ============================================================
# INTER-SHIFT TRANSITIONS
# ============================================================

print("\n")
print(SEP)
print("INTER-SHIFT TRANSITIONS (skip the depot between trips)")
print(SEP)
if best.shift_transitions:
    print(f"\n{'Vehicle trip':<24}{'From (last stop)':<18}{'To (next stop)':<18}{'Into':<6}{'Distance Saved':>16}")
    print("-" * 82)
    for v, (prev_last, next_first, phase) in sorted(best.shift_transitions.items()):
        saved = (
            dc.arc_cost(prev_last, DEPOT)
            + dc.arc_cost(DEPOT, next_first)
            - dc.arc_cost(prev_last, next_first)
        )
        print(f"{dc._slot_label(v):<24}{prev_last:<18}{next_first:<18}{phase:<6}{saved:>16.2f}")
else:
    print("\n  No inter-shift transitions used.")

# ============================================================
# FULL DAILY ROUTES (per physical vehicle, all its trips in order)
# ============================================================

print("\n")
print(SEP)
print("FULL DAILY ROUTES (one physical vehicle may run several trips)")
print(SEP)

daily_schedule = dc.build_daily_schedule(best)

for pv in PHYSICAL_VEHICLES:
    trips = daily_schedule.get(pv)
    if not trips:
        continue

    print(f"\n{pv}  ({len(trips)} trip(s) today):")

    for i, trip in enumerate(trips, start=1):
        out_route = trip['out_route']
        ret_route = trip['ret_route']

        if trip['shift_transition']:
            prev_last, next_first = trip['shift_transition']
            print(f"  (drives straight from {prev_last} into trip {i} -- no depot stop in between)")

        if trip['transition']:
            h_out, h_ret = trip['transition']
            out_physical = out_route[:-1] if out_route[-1] == DEPOT else out_route
            ret_physical = ret_route[1:] if ret_route[0] == DEPOT else ret_route
            full = ' -> '.join(out_physical) + f'  ~~[{h_out}->{h_ret}]~~>  ' + ' -> '.join(ret_physical)
            print(f"  Trip {i} [{dc.fmt(trip['out_start'])}-{dc.fmt(trip['ret_finish'])}, transition {h_out}->{h_ret}]:")
            print(f"    {full}")

        elif trip['has_out'] and trip['has_ret']:
            full = ' -> '.join(out_route) + '  -->  ' + ' -> '.join(ret_route)
            print(f"  Trip {i} [out {dc.fmt(trip['out_start'])}-{dc.fmt(trip['out_finish'])}, "
                  f"ret {dc.fmt(trip['ret_start'])}-{dc.fmt(trip['ret_finish'])}]:")
            print(f"    {full}")

        elif trip['has_out']:
            print(f"  Trip {i} [outbound only, {dc.fmt(trip['out_start'])}-{dc.fmt(trip['out_finish'])}]:")
            print(f"    {' -> '.join(out_route)}")

        else:
            print(f"  Trip {i} [return only, {dc.fmt(trip['ret_start'])}-{dc.fmt(trip['ret_finish'])}]:")
            print(f"    {' -> '.join(ret_route)}")

# ============================================================
# PER-PATIENT TIMING TABLE
# ============================================================

diagnostics_table, capacity_violations = dc.build_diagnostics(best)

print("\n")
print(SEP)
print("PER-PATIENT TIMING TABLE")
print(SEP)
_display_cols = [
    'patient', 'vehicle_out', 'home_pickup_hhmm', 'hospital_arrival_hhmm',
    'appointment_hhmm', 'out_ride_time', 'out_ride_violation',
    'early_minutes', 'out_early_violation', 'out_late_violation',
    'vehicle_ret', 'treatment_ready_hhmm', 'return_pickup_hhmm', 'return_delay',
    'return_delay_violation', 'home_dropoff_hhmm', 'ret_ride_time', 'ret_ride_violation',
]
_display_cols = [c for c in _display_cols if c in diagnostics_table.columns]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n" + diagnostics_table[_display_cols].to_string(index=False))

# ============================================================
# CONSTRAINT VIOLATIONS
# ============================================================

print("\n")
print(SEP)
print("CONSTRAINT VIOLATIONS")
print(SEP)

violated = diagnostics_table[diagnostics_table['any_violation'] == True]
if len(violated) == 0:
    print("\n  No time-window / ride-time / return-delay violations.")
else:
    print(f"\n  {len(violated)} patient(s) with a violated time constraint:\n")
    print(violated[_display_cols].to_string(index=False))

if len(capacity_violations) == 0:
    print("\n  No vehicle-capacity violations.")
else:
    print(f"\n  {len(capacity_violations)} vehicle-capacity overflow event(s):\n")
    capacity_violations['time_hhmm'] = capacity_violations['time'].apply(dc.fmt)
    print(capacity_violations[['vehicle', 'phase', 'node', 'time_hhmm', 'load', 'capacity', 'excess']].to_string(index=False))

# ============================================================
# EXPORT RESULTS  (for later analysis / figure generation)
# ============================================================

if _args.export_dir:
    meta = dict(
        instance=_args.instance, seed=SEED, iterations_run=it, iterations_budget=ITERATIONS,
        shifts_start=SHIFTS_PER_VEHICLE, shifts_max=MAX_SHIFTS,
        shifts_grown_to={pv: len(dc.SLOTS_OF_VEHICLE[pv]) for pv in PHYSICAL_VEHICLES},
        day_start_hour=dc.DAY_START_HOUR, day_length_minutes=dc.DAY_LENGTH_MINUTES,
        naive_insertion=_args.naive_insertion,
        stop_reason=stop_reason,
        algorithm="lns",
        lambda_used=dc.LAMBDA,
        lambda_override=_args.lambda_override,
        disable_transitions=_args.disable_transitions,
        no_qos=_args.no_qos,
        search_objective=_search_objective if _args.no_qos else None,
        sharing_rate_override=_args.sharing_rate_override,
        weighted_qos=_args.weighted_qos,
        remove_ratio=REMOVE_RATIO,
        remove_cap=_args.remove_cap,
        adaptive=_args.adaptive,
        operator_uses=operator_uses if _args.adaptive else None,
        operator_final_weights=operator_weights if _args.adaptive else None,
    )
    dc.export_run(_args.export_dir, best, history, meta)
    print(f"\nExported results to: {_args.export_dir}")

# ============================================================
# PLOTS  (simple, from this run's own results -- no extra runs)
# ============================================================

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Plots go into --export_dir (unique per run) when given, instead of two
# fixed filenames in the current working directory -- otherwise every
# concurrent LNS process (across every instance/seed in a comparison run)
# races to overwrite the SAME two files, which under a synced folder (e.g.
# OneDrive-backed Desktop) causes real file-lock contention that gets worse
# the more runs have already executed in the same session.
_plot_dir = _args.export_dir if _args.export_dir else "."
_convergence_path = os.path.join(_plot_dir, "convergence.png")
_violations_path  = os.path.join(_plot_dir, "violations.png")
if _args.export_dir:
    os.makedirs(_args.export_dir, exist_ok=True)

# ---- Convergence curve ----
hist_df = pd.DataFrame(history)

plt.figure(figsize=(8, 5))
plt.plot(hist_df['iteration'], hist_df['current_objective'], color='lightgray', lw=1, label='Current solution')
plt.plot(hist_df['iteration'], hist_df['best_objective'], color='steelblue', lw=2, label='Best solution')
plt.yscale('log')
plt.xlabel('LNS iteration')
plt.ylabel('Objective (log scale)')
plt.title('LNS convergence')
plt.legend()
plt.tight_layout()
plt.savefig(_convergence_path, dpi=150)
plt.show()

# ---- Violation counts by type ----
violation_counts = {
    'Outbound\ntoo early': int(diagnostics_table.get('out_early_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    'Outbound\nlateness':  int(diagnostics_table.get('out_late_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    'Outbound\nride-time': int(diagnostics_table.get('out_ride_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    'Return\ndelay':       int(diagnostics_table.get('return_delay_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    'Return\nride-time':   int(diagnostics_table.get('ret_ride_violation', pd.Series(dtype=bool)).fillna(False).sum()),
    'Capacity\noverflow':  int(len(capacity_violations)),
}

plt.figure(figsize=(8, 5))
bars = plt.bar(violation_counts.keys(), violation_counts.values(), color='indianred')
plt.ylabel('Number of events')
plt.title('Constraint violations by type')
for bar, v in zip(bars, violation_counts.values()):
    plt.annotate(str(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 textcoords='offset points', xytext=(0, 3), ha='center')
plt.tight_layout()
plt.savefig(_violations_path, dpi=150)
plt.show()
