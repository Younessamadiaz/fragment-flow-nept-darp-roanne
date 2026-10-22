# ============================================================
# HEALTHCARE DIAL-A-RIDE PROBLEM -- VARIABLE NEIGHBORHOOD SEARCH BASELINE
#
# Basic VNS (Mladenovic & Hansen): a systematic sequence of neighborhoods
# N_1 .. N_kmax of increasing perturbation strength, rather than SA's
# random single move or LNS's fixed-size large neighborhood.
#
#   - Shaking: N_k = "remove k random patients, reinsert via best_insertion"
#     (dc.destroy(current, k) + dc.repair) -- the neighborhood SIZE k is
#     what's systematic here, unlike SA/TS's random 1-2 patient moves.
#   - Local search: first-improvement descent from the shaken solution,
#     repeatedly relocating a sampled patient if it improves the solution,
#     until no improving move is found or a step cap is hit.
#   - If local search improved on `current`, accept it and reset k=1;
#     otherwise move to the next (larger) neighborhood k+1, wrapping back
#     to 1 after k_max.
#
# VNS is one of the most widely used and competitive metaheuristics in the
# DARP literature specifically (systematic neighborhood shaking maps well
# onto ride-sharing route perturbation). Built entirely on darp_core.py's
# shared instance/constraint/evaluation/repair primitives.
# ============================================================

import argparse
import os
import copy
import random
import time

import numpy as np

import darp_core as dc

# ============================================================
# CONFIGURATION
# ============================================================
_parser = argparse.ArgumentParser()

_parser.add_argument(
    "--instance", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances", "E_20_5_1", "E_20_5_1_model_input.xlsx"),
)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--iterations", type=int, default=100000,
                      help="Safety ceiling. --time_budget_sec is the intended primary "
                           "stop condition for cross-algorithm comparison runs.")
_parser.add_argument("--shifts", type=int, default=2)
_parser.add_argument("--max_shifts", type=int, default=12)
_parser.add_argument("--day_start_buffer_hours", type=float, default=1.0)
_parser.add_argument("--day_end_buffer_hours", type=float, default=2.0)
_parser.add_argument("--export_dir", type=str, default=None)
_parser.add_argument("--time_budget_sec", type=float, default=None)

_parser.add_argument("--k_max", type=int, default=5,
                      help="Number of neighborhood structures N_1..N_k_max. Shaking at "
                           "N_k removes k random served patients and reinserts them.")
_parser.add_argument("--local_search_iters", type=int, default=25,
                      help="Max first-improvement descent steps per shake.")
_parser.add_argument("--local_search_sample", type=int, default=6,
                      help="Candidate patients sampled per descent step (first "
                           "improving relocate found among the sample is taken).")

_parser.add_argument("--ema_fast_window", type=int,   default=30)
_parser.add_argument("--ema_slow_window", type=int,   default=150)
_parser.add_argument("--ema_eps",         type=float, default=0.001)
_parser.add_argument("--ema_patience",    type=int,   default=1500)

_args, _ = _parser.parse_known_args()

random.seed(_args.seed)
np.random.seed(_args.seed)

# ============================================================
# LOAD INSTANCE
# ============================================================

print(f"\n[VNS] Loading instance from: {_args.instance}")
_info = dc.load_instance(
    _args.instance, shifts=_args.shifts, max_shifts=_args.max_shifts,
    day_start_buffer_hours=_args.day_start_buffer_hours,
    day_end_buffer_hours=_args.day_end_buffer_hours,
)
print(f"[VNS] Patients={dc.NUM_PATIENTS}  Hospitals={dc.NUM_HOSPITALS}  "
      f"Vehicles={_info['num_physical_vehicles']}  Slots={_info['num_routing_slots']}  "
      f"Day window: {_info['day_window_start']} -> {_info['day_window_end']}")

# ============================================================
# SHAKING + LOCAL SEARCH
# ============================================================

def shake(solution, k):
    out_d, ret_d, removed, tr_d, str_d = dc.destroy(solution, k)
    out_r, ret_r, tr_r, str_r = dc.repair(out_d, ret_d, removed, tr_d, str_d, naive_insertion=False)
    return dc.evaluate_solution(out_r, ret_r, tr_r, str_r)

def local_search(solution, max_iters, sample_size):
    current = solution
    for _ in range(max_iters):
        served = [p for p in dc.PATIENTS if p not in current.unserved]
        if not served:
            break
        candidates = random.sample(served, min(sample_size, len(served)))
        improved_this_step = False
        for p in candidates:
            out_r, ret_r, removed, tr_r, str_r = dc.remove_specific_patients(current, [p])
            out_r, ret_r, tr_r, str_r = dc.repair(out_r, ret_r, removed, tr_r, str_r, naive_insertion=False)
            candidate = dc.evaluate_solution(out_r, ret_r, tr_r, str_r)
            if dc.is_better(candidate, current):
                current = candidate
                improved_this_step = True
                break   # first improvement: restart descent from the new current
        if not improved_this_step:
            break
    return current

# ============================================================
# INITIAL SOLUTION
# ============================================================

print("[VNS] Building initial solution...")
out_routes, ret_routes = dc.initial_routes()
transitions       = dc.try_add_transitions(out_routes, ret_routes)
shift_transitions = dc.try_add_shift_transitions(out_routes, ret_routes)

current = dc.evaluate_solution(out_routes, ret_routes, transitions, shift_transitions)
best    = copy.deepcopy(current)
dc.ensure_growth_room(current, best)

print(f"[VNS] Initial objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}")

# ============================================================
# VNS LOOP
# ============================================================

stopper = dc.StoppingCriterion(
    time_budget_sec=_args.time_budget_sec, max_iterations=_args.iterations,
    ema_fast_window=_args.ema_fast_window, ema_slow_window=_args.ema_slow_window,
    ema_eps=_args.ema_eps, ema_patience=_args.ema_patience,
).start()

history = []
it = 0
k = 1
stop_reason = "max iterations reached"

print("\n[VNS] Running...\n")

while True:
    dc.ensure_growth_room(current, best)

    shaken = shake(current, k)
    candidate = local_search(shaken, _args.local_search_iters, _args.local_search_sample)

    if dc.is_better(candidate, current):
        current = candidate
        k = 1
    else:
        k += 1
        if k > _args.k_max:
            k = 1

    if dc.is_better(current, best):
        best = copy.deepcopy(current)

    it += 1
    history.append(dict(
        iteration=it, current_objective=current.objective, best_objective=best.objective,
        best_violation_cost=best.violation_cost, best_feasible=best.feasible,
    ))

    stop, reason = stopper.step(it, current.objective)

    if it % 200 == 0 or stop:
        elapsed = time.time() - stopper.t_start
        print(f"Iter {it:6d} | Obj = {best.objective:10.2f} | Served = {best.num_served:3d}/{dc.NUM_PATIENTS} | "
              f"Viol = {best.violation_cost:10.2f} | k = {k}/{_args.k_max} | Time = {elapsed:6.1f}s")

    if stop:
        stop_reason = reason
        break

print(f"\n[VNS] Stopped after {it} iterations: {stop_reason}")
print(f"[VNS] Final objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}  feasible={best.feasible}")

# ============================================================
# EXPORT
# ============================================================

if _args.export_dir:
    meta = dict(
        instance=_args.instance, seed=_args.seed, algorithm="vns",
        iterations_run=it, iterations_budget=_args.iterations,
        shifts_start=_args.shifts, shifts_max=_args.max_shifts,
        shifts_grown_to={pv: len(dc.SLOTS_OF_VEHICLE[pv]) for pv in dc.PHYSICAL_VEHICLES},
        day_start_hour=dc.DAY_START_HOUR, day_length_minutes=dc.DAY_LENGTH_MINUTES,
        stop_reason=stop_reason,
        k_max=_args.k_max, local_search_iters=_args.local_search_iters,
        local_search_sample=_args.local_search_sample, time_budget_sec=_args.time_budget_sec,
    )
    dc.export_run(_args.export_dir, best, history, meta)
    print(f"[VNS] Exported results to: {_args.export_dir}")
