# ============================================================
# HEALTHCARE DIAL-A-RIDE PROBLEM -- TABU SEARCH BASELINE
#
# Same small-step neighborhood as sa_darp.py (remove 1-2 random patients,
# reinsert via best_insertion), but instead of Metropolis acceptance around
# a single candidate: each iteration SAMPLES several candidate moves, keeps
# the best one that is not tabu (or is tabu but beats the global best --
# aspiration), and marks the moved patient(s)' vacated slot(s) tabu for a
# fixed tenure to discourage immediately reversing the move.
#
# Built entirely on darp_core.py's shared instance/constraint/evaluation/
# repair primitives.
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
_parser.add_argument("--iterations", type=int, default=50000,
                      help="Safety ceiling. --time_budget_sec is the intended primary "
                           "stop condition for cross-algorithm comparison runs.")
_parser.add_argument("--shifts", type=int, default=2)
_parser.add_argument("--max_shifts", type=int, default=12)
_parser.add_argument("--day_start_buffer_hours", type=float, default=1.0)
_parser.add_argument("--day_end_buffer_hours", type=float, default=2.0)
_parser.add_argument("--export_dir", type=str, default=None)
_parser.add_argument("--time_budget_sec", type=float, default=None)

_parser.add_argument("--neighborhood_sample", type=int, default=15,
                      help="Number of candidate moves sampled and evaluated per "
                           "iteration; the best admissible (non-tabu, or tabu-but-"
                           "improves-global-best) one is applied.")
_parser.add_argument("--move_size_max", type=int, default=2,
                      help="Each candidate move removes 1..move_size_max random "
                           "served patients and reinserts them via best_insertion.")
_parser.add_argument("--tabu_tenure", type=int, default=15,
                      help="Iterations a (patient, vacated_slot) pair stays tabu -- "
                           "i.e. how long that patient may not be reinserted back into "
                           "the exact slot it was just moved out of.")

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

print(f"\n[TS] Loading instance from: {_args.instance}")
_info = dc.load_instance(
    _args.instance, shifts=_args.shifts, max_shifts=_args.max_shifts,
    day_start_buffer_hours=_args.day_start_buffer_hours,
    day_end_buffer_hours=_args.day_end_buffer_hours,
)
print(f"[TS] Patients={dc.NUM_PATIENTS}  Hospitals={dc.NUM_HOSPITALS}  "
      f"Vehicles={_info['num_physical_vehicles']}  Slots={_info['num_routing_slots']}  "
      f"Day window: {_info['day_window_start']} -> {_info['day_window_end']}")

# ============================================================
# INITIAL SOLUTION
# ============================================================

print("[TS] Building initial solution...")
out_routes, ret_routes = dc.initial_routes()
transitions       = dc.try_add_transitions(out_routes, ret_routes)
shift_transitions = dc.try_add_shift_transitions(out_routes, ret_routes)

current = dc.evaluate_solution(out_routes, ret_routes, transitions, shift_transitions)
best    = copy.deepcopy(current)
dc.ensure_growth_room(current, best)

print(f"[TS] Initial objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}")

# ============================================================
# TABU SEARCH LOOP
# ============================================================

stopper = dc.StoppingCriterion(
    time_budget_sec=_args.time_budget_sec, max_iterations=_args.iterations,
    ema_fast_window=_args.ema_fast_window, ema_slow_window=_args.ema_slow_window,
    ema_eps=_args.ema_eps, ema_patience=_args.ema_patience,
).start()

tabu_until = {}   # (patient, slot) -> iteration after which it's allowed again
history = []
it = 0
stop_reason = "max iterations reached"

print("\n[TS] Running...\n")

while True:
    dc.ensure_growth_room(current, best)

    served = [p for p in dc.PATIENTS if p not in current.unserved]
    admissible = []   # (objective, candidate_solution, moved_patients, old_out_slot, old_ret_slot)

    for _ in range(_args.neighborhood_sample):
        if not served:
            break
        move_size = random.randint(1, max(1, _args.move_size_max))
        if len(served) < move_size:
            move_size = len(served)
        patients = random.sample(served, move_size)

        old_out = {p: dc.patient_slot(current.outbound_routes, p) for p in patients}
        old_ret = {p: dc.patient_slot(current.return_routes, p) for p in patients}

        out_r, ret_r, removed, transitions_r, shift_transitions_r = dc.remove_specific_patients(current, patients)
        out_r, ret_r, transitions_r, shift_transitions_r = dc.repair(
            out_r, ret_r, removed, transitions_r, shift_transitions_r, naive_insertion=False
        )
        candidate = dc.evaluate_solution(out_r, ret_r, transitions_r, shift_transitions_r)

        is_tabu = False
        for p in patients:
            new_out = dc.patient_slot(candidate.outbound_routes, p)
            new_ret = dc.patient_slot(candidate.return_routes, p)
            if tabu_until.get((p, new_out), -1) > it or tabu_until.get((p, new_ret), -1) > it:
                is_tabu = True
                break

        aspiration = dc.is_better(candidate, best)
        if is_tabu and not aspiration:
            continue

        admissible.append((candidate.objective, candidate, patients, old_out, old_ret))

    it += 1

    if admissible:
        _, candidate, moved_patients, old_out, old_ret = min(admissible, key=lambda c: c[0])
        current = candidate
        for p in moved_patients:
            if old_out[p] is not None:
                tabu_until[(p, old_out[p])] = it + _args.tabu_tenure
            if old_ret[p] is not None:
                tabu_until[(p, old_ret[p])] = it + _args.tabu_tenure
        if dc.is_better(current, best):
            best = copy.deepcopy(current)

    history.append(dict(
        iteration=it, current_objective=current.objective, best_objective=best.objective,
        best_violation_cost=best.violation_cost, best_feasible=best.feasible,
    ))

    stop, reason = stopper.step(it, current.objective)

    if it % 200 == 0 or stop:
        elapsed = time.time() - stopper.t_start
        print(f"Iter {it:6d} | Obj = {best.objective:10.2f} | Served = {best.num_served:3d}/{dc.NUM_PATIENTS} | "
              f"Viol = {best.violation_cost:10.2f} | Tabu = {len(tabu_until):4d} | Time = {elapsed:6.1f}s")

    if stop:
        stop_reason = reason
        break

print(f"\n[TS] Stopped after {it} iterations: {stop_reason}")
print(f"[TS] Final objective={best.objective:.2f}  served={best.num_served}/{dc.NUM_PATIENTS}  feasible={best.feasible}")

# ============================================================
# EXPORT
# ============================================================

if _args.export_dir:
    meta = dict(
        instance=_args.instance, seed=_args.seed, algorithm="ts",
        iterations_run=it, iterations_budget=_args.iterations,
        shifts_start=_args.shifts, shifts_max=_args.max_shifts,
        shifts_grown_to={pv: len(dc.SLOTS_OF_VEHICLE[pv]) for pv in dc.PHYSICAL_VEHICLES},
        day_start_hour=dc.DAY_START_HOUR, day_length_minutes=dc.DAY_LENGTH_MINUTES,
        stop_reason=stop_reason,
        neighborhood_sample=_args.neighborhood_sample, move_size_max=_args.move_size_max,
        tabu_tenure=_args.tabu_tenure, time_budget_sec=_args.time_budget_sec,
    )
    dc.export_run(_args.export_dir, best, history, meta)
    print(f"[TS] Exported results to: {_args.export_dir}")
