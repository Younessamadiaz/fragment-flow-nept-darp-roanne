# ============================================================
# HEALTHCARE DIAL-A-RIDE PROBLEM -- SIMULATED ANNEALING BASELINE
#
# darp_lns.py's own LNS loop already does Metropolis acceptance + geometric
# cooling around a LARGE-neighborhood destroy/repair move (~20% of patients
# per iteration). For this to be a meaningfully different baseline rather
# than a relabeled copy of the same algorithm, the move here is deliberately
# SMALL-step: remove just 1-2 random patients (single relocate / two-patient
# swap) and reinsert via the same best_insertion used everywhere else, then
# accept/reject with an independently-tuned Metropolis schedule.
#
# Built entirely on darp_core.py's shared instance/constraint/evaluation/
# destroy/repair primitives -- it searches the exact same feasible-solution
# space and objective as darp_lns.py and the other baselines.
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
_parser.add_argument("--iterations", type=int, default=200000,
                      help="Safety ceiling on iterations. Moves here are much cheaper "
                           "than LNS's large-neighborhood destroy/repair, so this can "
                           "be much higher -- --time_budget_sec is the intended primary "
                           "stop condition for cross-algorithm comparison runs.")
_parser.add_argument("--shifts", type=int, default=2)
_parser.add_argument("--max_shifts", type=int, default=12)
_parser.add_argument("--day_start_buffer_hours", type=float, default=1.0)
_parser.add_argument("--day_end_buffer_hours", type=float, default=2.0)
_parser.add_argument("--export_dir", type=str, default=None)
_parser.add_argument("--time_budget_sec", type=float, default=None,
                      help="Primary stop condition for fair cross-algorithm comparison: "
                           "stop once this many wall-clock seconds have elapsed since the "
                           "search loop started (whichever of time/iterations/EMA fires "
                           "first wins). Timer excludes initial-solution construction, "
                           "matching darp_lns.py's own convention.")

_parser.add_argument("--temperature", type=float, default=50.0,
                      help="Initial SA temperature. Independently tuned from LNS's own "
                           "T0=500 -- single/double-patient move deltas are typically much "
                           "smaller in magnitude than LNS's large-neighborhood deltas, so "
                           "a much smaller starting temperature is needed for the "
                           "Metropolis criterion to behave sensibly.")
_parser.add_argument("--cooling", type=float, default=0.99995,
                      help="Geometric cooling rate per iteration.")
_parser.add_argument("--move_size_max", type=int, default=2,
                      help="Each iteration removes a random 1..move_size_max served "
                           "patients (uniformly) and reinserts them via best_insertion -- "
                           "1 = single-patient relocate, 2 = two-patient swap-like move.")

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

print(f"\n[SA] Loading instance from: {_args.instance}")
_info = dc.load_instance(
    _args.instance, shifts=_args.shifts, max_shifts=_args.max_shifts,
    day_start_buffer_hours=_args.day_start_buffer_hours,
    day_end_buffer_hours=_args.day_end_buffer_hours,
)
print(f"[SA] Patients={dc.NUM_PATIENTS}  Hospitals={dc.NUM_HOSPITALS}  "
      f"Vehicles={_info['num_physical_vehicles']}  Slots={_info['num_routing_slots']}  "
      f"Day window: {_info['day_window_start']} -> {_info['day_window_end']}")

# ============================================================
# INITIAL SOLUTION
# ============================================================

print("[SA] Building initial solution...")
out_routes, ret_routes = dc.initial_routes()
transitions       = dc.try_add_transitions(out_routes, ret_routes)
shift_transitions = dc.try_add_shift_transitions(out_routes, ret_routes)

current = dc.evaluate_solution(out_routes, ret_routes, transitions, shift_transitions)
best    = copy.deepcopy(current)
dc.ensure_growth_room(current, best)

print(f"[SA] Initial objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}")

# ============================================================
# SIMULATED ANNEALING LOOP (small-step local search)
# ============================================================

stopper = dc.StoppingCriterion(
    time_budget_sec=_args.time_budget_sec, max_iterations=_args.iterations,
    ema_fast_window=_args.ema_fast_window, ema_slow_window=_args.ema_slow_window,
    ema_eps=_args.ema_eps, ema_patience=_args.ema_patience,
).start()

temperature = _args.temperature
history = []
it = 0
stop_reason = "max iterations reached"

print("\n[SA] Running...\n")
t_report = time.time()

while True:
    dc.ensure_growth_room(current, best)

    move_size = random.randint(1, max(1, _args.move_size_max))
    out_d, ret_d, removed, transitions_d, shift_transitions_d = dc.destroy(current, move_size)
    out_r, ret_r, transitions_r, shift_transitions_r = dc.repair(
        out_d, ret_d, removed, transitions_d, shift_transitions_d, naive_insertion=False
    )
    candidate = dc.evaluate_solution(out_r, ret_r, transitions_r, shift_transitions_r)

    delta = candidate.objective - current.objective
    if dc.metropolis_accept(delta, temperature):
        current = candidate
    if dc.is_better(candidate, best):
        best = copy.deepcopy(candidate)

    temperature *= _args.cooling
    it += 1

    history.append(dict(
        iteration=it, current_objective=current.objective, best_objective=best.objective,
        best_violation_cost=best.violation_cost, best_feasible=best.feasible,
    ))

    stop, reason = stopper.step(it, current.objective)

    if it % 500 == 0 or stop:
        elapsed = time.time() - stopper.t_start
        print(f"Iter {it:6d} | Obj = {best.objective:10.2f} | Served = {best.num_served:3d}/{dc.NUM_PATIENTS} | "
              f"Viol = {best.violation_cost:10.2f} | T = {temperature:8.3f} | Time = {elapsed:6.1f}s")

    if stop:
        stop_reason = reason
        break

print(f"\n[SA] Stopped after {it} iterations: {stop_reason}")
print(f"[SA] Final objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}  feasible={best.feasible}")

# ============================================================
# EXPORT
# ============================================================

if _args.export_dir:
    meta = dict(
        instance=_args.instance, seed=_args.seed, algorithm="sa",
        iterations_run=it, iterations_budget=_args.iterations,
        shifts_start=_args.shifts, shifts_max=_args.max_shifts,
        shifts_grown_to={pv: len(dc.SLOTS_OF_VEHICLE[pv]) for pv in dc.PHYSICAL_VEHICLES},
        day_start_hour=dc.DAY_START_HOUR, day_length_minutes=dc.DAY_LENGTH_MINUTES,
        stop_reason=stop_reason,
        temperature0=_args.temperature, cooling=_args.cooling, move_size_max=_args.move_size_max,
        time_budget_sec=_args.time_budget_sec,
    )
    dc.export_run(_args.export_dir, best, history, meta)
    print(f"[SA] Exported results to: {_args.export_dir}")
