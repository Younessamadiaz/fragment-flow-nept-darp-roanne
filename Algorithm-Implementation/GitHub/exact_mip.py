# ============================================================
# HEALTHCARE DARP - EXACT MIP MODEL (Pyomo + Gurobi/CBC)
#
# Adapted from the user's original toy-instance model to (a) load real
# *_model_input.xlsx instances via the same loader darp_lns.py/darp_core.py
# use, so results are directly comparable to the heuristics, and (b) use
# the SAME penalized objective as LNS instead of a mix of hard constraints,
# so "gap to optimal" is a comparison of the same objective function, not
# two different problems.
#
# Hard constraints (kept exactly as in the original model):
#   - Vehicle capacity (both phases)
#   - Outbound: arrival at hospital must not be later than the appointment
#     (upper bound only -- no early-arrival/Omega lower bound; arriving
#     early just means an implicit free wait, never a violation)
#   - Return: pickup cannot happen before treatment actually ends (ready
#     time) -- a physical necessity, not a preference
#
# Soft constraints (penalized in the objective, same constants as LNS):
#   - Ride-time excess beyond Delta_p, both phases: 100 x minutes over
#   - Return-delay excess beyond ready+Psi_p: 100 x minutes over (replaces
#     the original model's hard upper bound on the return window)
#   - Unserved patients: 1000 each. A single binary u[p] ties BOTH phases
#     together (a patient is either fully served or fully unserved) --
#     simpler and more physically meaningful than tracking outbound/return
#     service independently.
#
# MULTI-SHIFT: each physical vehicle is expanded into a fixed number of
# shift SLOTS (--shifts per vehicle, same slot-naming/expansion as LNS's
# darp_core.load_instance), each with its own outbound+return trip. Unlike
# LNS, slot count here is static (a MIP can't grow slots mid-solve), so
# --shifts should be set generously (>= the max shift count LNS actually
# used) for the comparison to be meaningful.
#
# Three things are enforced that a naive per-slot copy of the single-trip
# model would miss, and that LNS enforces via its route evaluator:
#   - A slot's own return trip cannot start before its own outbound trip
#     actually finished (previously absent: tO and tR for the same vehicle
#     had no timing relationship at all -- the old single-trip model let a
#     vehicle's outbound and return happen in parallel, which isn't
#     physically possible for one vehicle).
#   - A physical vehicle's slots run strictly in order: slot i+1 cannot
#     start until slot i is fully wrapped up (back at the depot).
#   - Every node's visit time is capped at DAY_LENGTH (previously only
#     bounded by a generic BigM, i.e. effectively unbounded) -- matching
#     LNS's own hard within-day constraint.
#
# TRANSITIONS: a slot may skip the depot between its own outbound close and
# its own return open, when both happen at the SAME hospital (h->h, zero
# direct-arc cost since it's the same physical stop) -- mirrors LNS's own
# within-shift "transition" move (try_add_transitions). This model does
# NOT represent LNS's INTER-shift transitions (skipping the depot between
# two CONSECUTIVE shifts of the same vehicle) -- that is a documented,
# deliberate scope limitation, not an oversight; see the project's
# exact-vs-LNS comparison notes for why (a general node-to-node inter-
# shift mechanism was prototyped and found to need substantially more
# machinery to get right than this within-shift-only version).
# ============================================================

import argparse
import math
import os

from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.common.tempfiles import TempfileManager

import darp_core as dc

# Windows + some solver plugins (observed with CBC) can mis-join a relative
# "." with the OS temp path when writing/reading solution files. Pointing
# Pyomo's temp directory at an explicit, simple path avoids that.
TempfileManager.tempdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mip_tmp")
os.makedirs(TempfileManager.tempdir, exist_ok=True)

VIOLATION_PENALTY = 100
UNSERVED_PENALTY  = 1000

# ============================================================
# CLI
# ============================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--instance", type=str, required=True)
_parser.add_argument("--solver", type=str, default="gurobi",
                      help="Pyomo solver name -- 'gurobi' for the real runs, "
                           "'cbc' for open-source structural verification.")
_parser.add_argument("--time_limit", type=float, default=1800,
                      help="Solver time limit in seconds. DARP MIPs are hard -- "
                           "even small instances can need much longer than the "
                           "300s used for the original toy example.")
_parser.add_argument("--mip_gap", type=float, default=0.0001)
_parser.add_argument("--shifts", type=int, default=3,
                      help="Fixed number of shift slots per physical vehicle "
                           "(each slot = one outbound+return trip). Static, "
                           "unlike LNS's on-demand growth -- set this to at "
                           "least the max shift count LNS actually used on "
                           "this instance, or the comparison isn't apples-to-"
                           "apples.")
_parser.add_argument("--day_start_buffer_hours", type=float, default=1.0)
_parser.add_argument("--day_end_buffer_hours", type=float, default=2.0)
_parser.add_argument("--export_dir", type=str, default=None,
                      help="If given, write a per-patient solution_detail.json "
                           "(vehicle/times/violation breakdown for each leg) "
                           "here, alongside the usual summary.json, for direct "
                           "comparison against a heuristic's routing choices.")
_args = _parser.parse_args()

# ============================================================
# INSTANCE DATA (loaded via darp_core.load_instance, the SAME shift-slot
# expansion LNS uses -- VEHICLES below are shift slots, e.g.
# "vsl_V1::S1", "vsl_V1::S2", not physical vehicles)
# ============================================================

dc.load_instance(
    _args.instance, shifts=_args.shifts, max_shifts=_args.shifts,
    day_start_buffer_hours=_args.day_start_buffer_hours,
    day_end_buffer_hours=_args.day_end_buffer_hours,
)

PATIENTS  = dc.PATIENTS
VEHICLES  = dc.VEHICLES   # shift slots
HOSPITALS = dc.HOSPITALS
DEPOT     = dc.DEPOT

phys_hosp   = dc.hospital
appointment = dc.appointment
tau         = dc.tau
DELTA_P     = dc.DELTA_P
PSI_P       = dc.PSI_P
service     = dc.service
arc_time    = dc.arc_time
arc_cost    = dc.arc_cost
compatO     = dc.compatO
compatR     = dc.compatR
capacity    = dc.CAPACITY
LAMBDA      = dc.LAMBDA
OMEGA       = dc.OMEGA
DAY_LENGTH  = dc.DAY_LENGTH_MINUTES

print(f"Loaded {_args.instance}")
print(f"  Patients={len(PATIENTS)}  Hospitals={len(HOSPITALS)}  "
      f"Physical vehicles={len(dc.PHYSICAL_VEHICLES)}  "
      f"Shift slots={len(VEHICLES)} ({_args.shifts} shifts/vehicle)")

BigM = DAY_LENGTH + 200.0

# ============================================================
# PREPROCESSING (arc filters -- same structure as the original model)
# ============================================================

O_NODES = [DEPOT] + PATIENTS + HOSPITALS
R_NODES = [DEPOT] + HOSPITALS + PATIENTS

def valid_o_arc(i, j):
    if i == j or (i == DEPOT and j == DEPOT):
        return False
    if i in PATIENTS and j in HOSPITALS and phys_hosp[i] != j:
        return False
    if i in PATIENTS and j == DEPOT:
        return False
    return True

def valid_r_arc(i, j):
    if i == j or (i == DEPOT and j == DEPOT):
        return False
    if i in HOSPITALS and j in PATIENTS and phys_hosp[j] != i:
        return False
    if i == DEPOT and j in PATIENTS:
        return False
    return True

O_ARCS = [(i, j) for i in O_NODES for j in O_NODES if valid_o_arc(i, j)]
R_ARCS = [(i, j) for i in R_NODES for j in R_NODES if valid_r_arc(i, j)]

def dc_cost(a, b):
    return 0.0 if a == b else arc_cost(a, b)

def dc_time(a, b):
    return 0.0 if a == b else arc_time(a, b)

O_succ = {n: [j for (i, j) in O_ARCS if i == n] for n in O_NODES}
O_pred = {n: [i for (i, j) in O_ARCS if j == n] for n in O_NODES}
R_succ = {n: [j for (i, j) in R_ARCS if i == n] for n in R_NODES}
R_pred = {n: [i for (i, j) in R_ARCS if j == n] for n in R_NODES}

# ============================================================
# PYOMO MODEL
# ============================================================

model = ConcreteModel()

model.P  = Set(initialize=PATIENTS)
model.H  = Set(initialize=HOSPITALS)
model.K  = Set(initialize=VEHICLES)
model.OA = Set(initialize=O_ARCS,  dimen=2)
model.RA = Set(initialize=R_ARCS,  dimen=2)
model.ON = Set(initialize=O_NODES)
model.RN = Set(initialize=R_NODES)

model.xO = Var(model.OA, model.K, within=Binary)
model.xR = Var(model.RA, model.K, within=Binary)

# Bounded by DAY_LENGTH (not a generic BigM) -- a real within-day cap,
# matching LNS's hard route_within_day_ok check.
model.tO = Var(model.ON, model.K, within=NonNegativeReals, bounds=(0, DAY_LENGTH))
model.tR = Var(model.RN, model.K, within=NonNegativeReals, bounds=(0, DAY_LENGTH))

# When this slot's vehicle is actually free to start (0 for a slot's own
# first use; for slot i+1 of the same physical vehicle, chained below to
# slot i's return-to-depot time via cChain).
model.SlotStart = Var(model.K, within=NonNegativeReals, bounds=(0, DAY_LENGTH))

# Ride-time contributions (lower-bounded definition; upper bound is now SOFT)
model.rO = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))
model.rR = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))

# Soft-constraint excess variables
model.eO = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))  # outbound ride-time excess over Delta_p
model.eR = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))  # return ride-time excess over Delta_p
model.dR = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))  # return-delay excess over ready+Psi_p
model.fO = Var(model.P, model.K, within=NonNegativeReals, bounds=(0, BigM))  # outbound early-arrival excess before appt-Omega

# Unserved indicator -- ties outbound and return service together
model.u = Var(model.P, within=Binary)

# 1 if slot k transitions directly from its own outbound trip into its own
# return trip at hospital h (skipping the depot round-trip in between) --
# mirrors LNS's own within-shift "transition" move (try_add_transitions).
# Modeled per-hospital (not just per-slot) so it generalizes beyond the
# single-hospital instances this comparison currently uses.
model.yTrans = Var(model.H, model.K, within=Binary)

# ---- Objective ----
model.obj = Objective(
    expr=(
        sum(dc_cost(i, j) * model.xO[i, j, k] for (i, j) in O_ARCS for k in VEHICLES)
      + sum(dc_cost(i, j) * model.xR[i, j, k] for (i, j) in R_ARCS for k in VEHICLES)
      + LAMBDA * sum(model.rO[p, k] + model.rR[p, k] for p in PATIENTS for k in VEHICLES)
      + VIOLATION_PENALTY * sum(model.eO[p, k] + model.eR[p, k] + model.dR[p, k] + model.fO[p, k]
                                for p in PATIENTS for k in VEHICLES)
      + UNSERVED_PENALTY * sum(model.u[p] for p in PATIENTS)
    ),
    sense=minimize
)

# ============================================================
# OUTBOUND CONSTRAINTS
# ============================================================

# C-O1 (SOFT): each patient served at most once; u[p]=1 means unserved
model.cO1 = ConstraintList()
for p in PATIENTS:
    model.cO1.add(
        sum(model.xO[p, j, k] for j in O_succ[p] for k in VEHICLES) == 1 - model.u[p]
    )

model.cO2 = ConstraintList()
for h in HOSPITALS:
    for k in VEHICLES:
        model.cO2.add(sum(model.xO[h, j, k] for j in O_succ[h]) <= 1)

model.cO3 = ConstraintList()
for node in PATIENTS:
    for k in VEHICLES:
        model.cO3.add(
            sum(model.xO[i, node, k] for i in O_pred[node])
         == sum(model.xO[node, j, k] for j in O_succ[node])
        )
# Hospitals get a relaxed balance (see yTrans below): if this hospital is
# where the slot transitions directly into its own return trip, the
# outbound side is allowed one more arrival than departure (it never
# departs again within O_ARCS -- it continues via the transition instead).
for h in HOSPITALS:
    for k in VEHICLES:
        model.cO3.add(
            sum(model.xO[i, h, k] for i in O_pred[h])
         == sum(model.xO[h, j, k] for j in O_succ[h]) + model.yTrans[h, k]
        )

model.cO4 = ConstraintList()
for k in VEHICLES:
    model.cO4.add(sum(model.xO[DEPOT, j, k] for j in O_succ[DEPOT]) <= 1)

# C-O4b: if this slot's outbound trip departs the depot, it must also
# return to it -- UNLESS it instead closes via a direct transition into its
# own return trip (yTrans), in which case it's expected to depart without
# ever returning to the depot at all.
model.cO4b = ConstraintList()
for k in VEHICLES:
    model.cO4b.add(
        sum(model.xO[i, DEPOT, k] for i in O_pred[DEPOT])
     == sum(model.xO[DEPOT, j, k] for j in O_succ[DEPOT])
      - sum(model.yTrans[h, k] for h in HOSPITALS)
    )

model.cO5 = ConstraintList()
for p in PATIENTS:
    for k in VEHICLES:
        if compatO[p, k] == 0:
            model.cO5.add(sum(model.xO[p, j, k] for j in O_succ[p]) == 0)

# C-O7 (time propagation, using real travel TIME -- not cost). Departure
# from the depot is gated by SlotStart[k] (0 for a slot's first use, or
# chained after the same physical vehicle's previous slot -- see cChain)
# instead of a bare constant, so a vehicle's slots can't overlap in time.
model.cO7 = ConstraintList()
for (i, j) in O_ARCS:
    for k in VEHICLES:
        if i == DEPOT:
            model.cO7.add(
                model.tO[j, k] >= model.SlotStart[k] + dc_time(i, j) - BigM * (1 - model.xO[i, j, k])
            )
        else:
            model.cO7.add(
                model.tO[j, k] >= model.tO[i, k] + service[i] + dc_time(i, j)
                                 - BigM * (1 - model.xO[i, j, k])
            )

# C-O9: outbound arrival must not be later than the appointment (HARD --
# never miss an appointment). Arriving more than Omega minutes early is
# allowed but SOFT-penalized via fO (matches LNS's own early-arrival rule).
model.cO9 = ConstraintList()
for p in PATIENTS:
    h  = phys_hosp[p]
    hi = float(appointment[p])
    lo = float(appointment[p] - OMEGA)
    for k in VEHICLES:
        yOpk = sum(model.xO[p, j, k] for j in O_succ[p])
        model.cO9.add(model.tO[h, k] <= hi + BigM * (1 - yOpk))
        model.cO9.add(model.fO[p, k] >= lo - model.tO[h, k] - BigM * (1 - yOpk))

# C-O10: patient visited before their hospital (outbound)
model.cO10 = ConstraintList()
for p in PATIENTS:
    h = phys_hosp[p]
    for k in VEHICLES:
        yOpk = sum(model.xO[p, j, k] for j in O_succ[p])
        model.cO10.add(model.tO[h, k] >= model.tO[p, k] + service[p] - BigM * (1 - yOpk))

# C-O11 (HARD): vehicle capacity
model.cO11 = ConstraintList()
for k in VEHICLES:
    model.cO11.add(sum(model.xO[p, j, k] for p in PATIENTS for j in O_succ[p]) <= capacity[k])

# C-O12 (ride-time definition, now SOFT via excess variable eO)
model.cO12 = ConstraintList()
for p in PATIENTS:
    h = phys_hosp[p]
    for k in VEHICLES:
        yOpk = sum(model.xO[p, j, k] for j in O_succ[p])
        model.cO12.add(model.rO[p, k] >= model.tO[h, k] - model.tO[p, k] - BigM * (1 - yOpk))
        model.cO12.add(model.eO[p, k] >= model.rO[p, k] - DELTA_P[p] - BigM * (1 - yOpk))

# ============================================================
# RETURN CONSTRAINTS
# ============================================================

# C-R1 (SOFT): tied to the SAME u[p] as outbound
model.cR1 = ConstraintList()
for p in PATIENTS:
    model.cR1.add(
        sum(model.xR[p, j, k] for j in R_succ[p] for k in VEHICLES) == 1 - model.u[p]
    )

model.cR2 = ConstraintList()
for h in HOSPITALS:
    for k in VEHICLES:
        model.cR2.add(sum(model.xR[h, j, k] for j in R_succ[h]) <= 1)

model.cR3 = ConstraintList()
for node in PATIENTS:
    for k in VEHICLES:
        model.cR3.add(
            sum(model.xR[i, node, k] for i in R_pred[node])
         == sum(model.xR[node, j, k] for j in R_succ[node])
        )
# Hospitals get a relaxed balance, mirroring cO3: if this hospital is where
# the slot transitions directly in from its own outbound trip, the return
# side is allowed one more departure than arrival (it never arrived via
# R_ARCS -- it continued in via the transition instead).
for h in HOSPITALS:
    for k in VEHICLES:
        model.cR3.add(
            sum(model.xR[h, j, k] for j in R_succ[h])
         == sum(model.xR[i, h, k] for i in R_pred[h]) + model.yTrans[h, k]
        )

model.cR4 = ConstraintList()
for k in VEHICLES:
    model.cR4.add(sum(model.xR[DEPOT, j, k] for j in R_succ[DEPOT]) <= 1)

# C-R4b: same depot round-trip requirement as C-O4b, for the return leg --
# UNLESS this slot's return instead starts via a direct transition from its
# own outbound trip (yTrans), in which case it's expected to never have
# departed the depot at all for this leg.
model.cR4b = ConstraintList()
for k in VEHICLES:
    model.cR4b.add(
        sum(model.xR[i, DEPOT, k] for i in R_pred[DEPOT])
     == sum(model.xR[DEPOT, j, k] for j in R_succ[DEPOT])
      + sum(model.yTrans[h, k] for h in HOSPITALS)
    )

# A slot can transition through at most one hospital (it's a single
# physical vehicle continuing in place, not multiple simultaneous handoffs).
model.cTransCap = ConstraintList()
for k in VEHICLES:
    model.cTransCap.add(sum(model.yTrans[h, k] for h in HOSPITALS) <= 1)

# C-TRANSTIME: when a transition is used, the return leg at h can't start
# until the outbound leg's own service at h is done -- no travel time in
# between, since it's the same physical stop continuing directly into
# return-phase pickups (mirrors LNS's try_add_transitions timing exactly:
# departure_h_out = t_h_out + service[h_out], then arc_time(h_out,h_ret)=0
# since h_out == h_ret here).
model.cTransTime = ConstraintList()
for h in HOSPITALS:
    for k in VEHICLES:
        model.cTransTime.add(
            model.tR[h, k] >= model.tO[h, k] + service[h] - BigM * (1 - model.yTrans[h, k])
        )

model.cR5 = ConstraintList()
for p in PATIENTS:
    for k in VEHICLES:
        if compatR[p, k] == 0:
            model.cR5.add(sum(model.xR[p, j, k] for j in R_succ[p]) == 0)

# C-R7 (time propagation for the return leg). Departure from the depot is
# gated by tO[DEPOT,k] -- this slot's OWN outbound trip's return-to-depot
# time -- instead of a bare constant, so a slot's return trip can no longer
# start before its own outbound trip actually finished (previously
# unconstrained: tO and tR for the same k had no timing relationship).
model.cR7 = ConstraintList()
for (i, j) in R_ARCS:
    for k in VEHICLES:
        if i == DEPOT:
            model.cR7.add(
                model.tR[j, k] >= model.tO[DEPOT, k] + dc_time(i, j) - BigM * (1 - model.xR[i, j, k])
            )
        else:
            model.cR7.add(
                model.tR[j, k] >= model.tR[i, k] + service[i] + dc_time(i, j)
                                 - BigM * (1 - model.xR[i, j, k])
            )

# C-R9 (HARD lower bound only): cannot pick up before treatment ends.
# Upper bound (Psi window) is now SOFT via dR.
model.cR9 = ConstraintList()
for p in PATIENTS:
    h     = phys_hosp[p]
    ready = float(appointment[p] + tau[p])
    for k in VEHICLES:
        yRpk = sum(model.xR[p, j, k] for j in R_succ[p])
        model.cR9.add(model.tR[h, k] >= ready - BigM * (1 - yRpk))
        model.cR9.add(model.dR[p, k] >= model.tR[h, k] - (ready + PSI_P[p]) - BigM * (1 - yRpk))

# C-R10: hospital visited before patient (return)
model.cR10 = ConstraintList()
for p in PATIENTS:
    h = phys_hosp[p]
    for k in VEHICLES:
        yRpk = sum(model.xR[p, j, k] for j in R_succ[p])
        model.cR10.add(model.tR[p, k] >= model.tR[h, k] + service[h] - BigM * (1 - yRpk))

# C-R11 (HARD): vehicle capacity (return)
model.cR11 = ConstraintList()
for k in VEHICLES:
    model.cR11.add(sum(model.xR[p, j, k] for p in PATIENTS for j in R_succ[p]) <= capacity[k])

# C-R12 (ride-time definition, now SOFT via excess variable eR)
model.cR12 = ConstraintList()
for p in PATIENTS:
    h = phys_hosp[p]
    for k in VEHICLES:
        yRpk = sum(model.xR[p, j, k] for j in R_succ[p])
        model.cR12.add(model.rR[p, k] >= model.tR[p, k] - model.tR[h, k] - BigM * (1 - yRpk))
        model.cR12.add(model.eR[p, k] >= model.rR[p, k] - DELTA_P[p] - BigM * (1 - yRpk))

# ============================================================
# MULTI-SHIFT CHAINING
# ============================================================

# C-CHAIN: a physical vehicle's shift slots run strictly in the fixed order
# LNS itself uses (S1, S2, ... -- no free reordering) -- slot i+1 cannot
# start until slot i is fully wrapped up (back at the depot after its own
# return trip). Mirrors LNS's compute_slot_start_times chaining.
model.cChain = ConstraintList()
for pv in dc.PHYSICAL_VEHICLES:
    slots = dc.SLOTS_OF_VEHICLE[pv]
    for prev_slot, next_slot in zip(slots, slots[1:]):
        model.cChain.add(model.SlotStart[next_slot] >= model.tR[DEPOT, prev_slot])

# C-SLOTFLOOR: tO[DEPOT,k] gates this slot's OWN return leg (via cR7's
# DEPOT-origin case) regardless of whether this slot's outbound is used.
# Without this, a slot with an entirely unused outbound leg gets
# tO[DEPOT,k] with no lower bound from cO7 (nothing forces it above 0),
# so its return leg could start as if the vehicle had been free since time
# 0 -- silently skipping the cChain constraint that was supposed to make it
# wait for the PREVIOUS slot to finish. Forcing tO[DEPOT,k] >= SlotStart[k]
# unconditionally closes that loophole: an outbound-unused slot's return
# now correctly inherits the real chained availability time.
model.cSlotFloor = ConstraintList()
for k in VEHICLES:
    model.cSlotFloor.add(model.tO[DEPOT, k] >= model.SlotStart[k])

# C-SLOTFLOOR2: the mirror-image bug -- tR[DEPOT,k] gates the NEXT slot's
# start via cChain, regardless of whether THIS slot's return leg is used.
# Without this, a slot with an entirely unused return leg gets
# tR[DEPOT,k] with no lower bound at all (cR7 only fires for used arcs),
# so the next physical-vehicle slot could start as if this whole slot --
# including its outbound, which may be very much in use -- had already
# finished at time 0. Forcing tR[DEPOT,k] >= tO[DEPOT,k] unconditionally
# means an unused return leg simply inherits this slot's own outbound
# finish time instead of resetting to zero.
model.cSlotFloor2 = ConstraintList()
for k in VEHICLES:
    model.cSlotFloor2.add(model.tR[DEPOT, k] >= model.tO[DEPOT, k])

# ============================================================
# SOLVE
# ============================================================

if _args.solver == "cbc":
    # solver_io='nl' avoids a Windows path-handling bug in the LP+.soln
    # interface (observed with pyomo 6.10.1 + CBC on Windows: the solution
    # file path gets mangled, e.g. ".\C:\...\tmp.pyomo.soln", so the solve
    # succeeds but Pyomo can never read the result back). Only affects the
    # open-source CBC structural-check path -- 'gurobi' is unaffected.
    solver = SolverFactory('cbc', solver_io='nl')
    solver.options['sec']  = _args.time_limit
    solver.options['ratio'] = _args.mip_gap
else:
    solver = SolverFactory(_args.solver)
    solver.options['TimeLimit'] = _args.time_limit
    solver.options['MIPGap']    = _args.mip_gap

print(f"\nSolving exact MIP with {_args.solver} (time_limit={_args.time_limit}s, gap={_args.mip_gap})...")
results = solver.solve(model, tee=True)

status = results.solver.status
term   = results.solver.termination_condition
print(f"\nSolver status: {status}   Termination: {term}")

# On a time-limit stop with a feasible incumbent already found, Gurobi
# reports status='aborted' (not 'ok') even though a usable solution exists
# -- checking status==ok alone would discard that solution. Instead, check
# whether the model actually has values loaded (only infeasible/unbounded/
# error terminations leave it empty).
try:
    value(model.obj)
    has_solution = True
except Exception:
    has_solution = False

if not has_solution or term in (TerminationCondition.infeasible,
                                 TerminationCondition.infeasibleOrUnbounded,
                                 TerminationCondition.unbounded):
    raise SystemExit(f"No feasible solution found (status={status}, termination={term}).")

# ============================================================
# RESULTS
# ============================================================

total_travel = (
    sum(dc_cost(i, j) * value(model.xO[i, j, k]) for (i, j) in O_ARCS for k in VEHICLES)
  + sum(dc_cost(i, j) * value(model.xR[i, j, k]) for (i, j) in R_ARCS for k in VEHICLES)
)
total_ride = sum(value(model.rO[p, k]) + value(model.rR[p, k]) for p in PATIENTS for k in VEHICLES)
total_viol = sum(
    value(model.eO[p, k]) + value(model.eR[p, k]) + value(model.dR[p, k]) + value(model.fO[p, k])
    for p in PATIENTS for k in VEHICLES
)
num_unserved = sum(value(model.u[p]) for p in PATIENTS)
obj_val = value(model.obj)

print("\n" + "=" * 70)
print("FINAL SOLUTION (same objective definition as LNS)")
print("=" * 70)
print(f"  Objective        : {obj_val:.4f}")
print(f"  Travel cost      : {total_travel:.4f}")
print(f"  Ride time (raw)  : {total_ride:.4f}")
print(f"  Ride cost (lam*r): {LAMBDA * total_ride:.4f}")
print(f"  Violation cost   : {VIOLATION_PENALTY * total_viol:.4f}")
print(f"  Unserved         : {num_unserved:.0f} / {len(PATIENTS)}")
print(f"  Optimal (proven) : {term == TerminationCondition.optimal}")

used_trans = [(h, k) for h in HOSPITALS for k in VEHICLES if value(model.yTrans[h, k]) > 0.5]
print(f"  Within-shift transitions used: {len(used_trans)}  {used_trans}")

# Emit a summary.json compatible with the heuristics' schema, for direct comparison
import json
summary = dict(
    instance=_args.instance, algorithm="exact_mip", solver=_args.solver,
    objective=obj_val, travel_cost=total_travel, ride_penalty=total_ride,
    violation_cost=VIOLATION_PENALTY * total_viol,
    served=len(PATIENTS) - int(round(num_unserved)), num_patients=len(PATIENTS),
    feasible=(total_viol < 1e-4 and num_unserved < 1e-4),
    proven_optimal=(term == TerminationCondition.optimal),
    termination_condition=str(term), time_limit_sec=_args.time_limit,
)
print("\nSummary JSON:")
print(json.dumps(summary, indent=2))

if _args.export_dir:
    os.makedirs(_args.export_dir, exist_ok=True)

    with open(os.path.join(_args.export_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def _assigned_vehicle(xvar, arcs, p):
        for (i, j) in arcs:
            if i == p or j == p:
                for k in VEHICLES:
                    if value(xvar[i, j, k]) > 0.5:
                        return k
        return None

    detail = {}
    for p in PATIENTS:
        served_p = value(model.u[p]) < 0.5
        row = dict(served=served_p)
        if served_p:
            h = phys_hosp[p]
            out_v = _assigned_vehicle(model.xO, O_ARCS, p)
            ret_v = _assigned_vehicle(model.xR, R_ARCS, p)
            row.update(
                out_vehicle=out_v, ret_vehicle=ret_v,
                out_pickup_t=value(model.tO[p, out_v]) if out_v else None,
                out_hosp_arrival_t=value(model.tO[h, out_v]) if out_v else None,
                ret_hosp_pickup_t=value(model.tR[h, ret_v]) if ret_v else None,
                ret_dropoff_t=value(model.tR[p, ret_v]) if ret_v else None,
                appointment_t=float(appointment[p]),
                ready_t=float(appointment[p] + tau[p]),
                out_ride_excess=value(model.eO[p, out_v]) if out_v else None,
                out_early_excess=value(model.fO[p, out_v]) if out_v else None,
                ret_ride_excess=value(model.eR[p, ret_v]) if ret_v else None,
                ret_delay_excess=value(model.dR[p, ret_v]) if ret_v else None,
            )
        detail[p] = row

    with open(os.path.join(_args.export_dir, "solution_detail.json"), "w") as f:
        json.dump(detail, f, indent=2)
    print(f"\nExported per-patient solution detail to: {_args.export_dir}/solution_detail.json")
