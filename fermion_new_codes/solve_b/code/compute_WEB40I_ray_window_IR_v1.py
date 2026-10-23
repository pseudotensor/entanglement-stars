#!/usr/bin/env python3
"""Only the missing X1 infrared panels, using reviewed own R2C8--9.

Pilot: eight log-q nodes, eight channels, both endpoint settings, two
half-inner diagnostics. Twelve-node continuation needs a complete pilot
and its actual outer receipt; no automatic continuation or ODE retry.
"""
from __future__ import annotations
import time
START = time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import socket

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

RSS_LIMIT_KIB = 512 * 1024
AS_LIMIT_BYTES = 1536 * 1024**2
TOTAL_WALL_LIMIT = 900.0
TOTAL_SOLVE_LIMIT = 4096
TOTAL_RHS_LIMIT = 50_000_000
PHASE_WALL = 117.0  # Graceful snapshot before the 119 s outer watchdog.
PHASE_SOLVES = 512
PHASE_RHS = 5_000_000
CHARGED_WALL = 0.0
CHARGED_SOLVES = 0
CHARGED_RHS = 0
SOLVES = 0
RHS_CALLS = 0
SETTINGS = (
    {"name": "first", "outer_order": 10, "outer_multiplier": 1,
     "rtol": 1e-8, "atol": 1e-10},
    {"name": "refined", "outer_order": 12, "outer_multiplier": 2,
     "rtol": 2e-9, "atol": 2e-11},
)


class BudgetExhausted(Exception):
    pass


def check_budget(*_args):
    elapsed = time.monotonic() - START
    if elapsed >= PHASE_WALL or CHARGED_WALL + elapsed >= TOTAL_WALL_LIMIT - 1:
        raise BudgetExhausted("wall cap; no automatic continuation")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= RSS_LIMIT_KIB:
        raise BudgetExhausted("peak RSS cap")
    if RHS_CALLS >= PHASE_RHS or CHARGED_RHS + RHS_CALLS >= TOTAL_RHS_LIMIT:
        raise BudgetExhausted("RHS cap")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, result):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["resources"] = {"phase_wall_seconds": time.monotonic() - START,
        "process_user_cpu_seconds": usage.ru_utime, "process_system_cpu_seconds": usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss, "new_ode_solves_started": SOLVES,
        "new_rhs_evaluations": RHS_CALLS}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def inner_endpoint(k, q, z):
    minus = np.zeros(9, dtype=complex)
    plus = np.zeros(9, dtype=complex)
    minus[0] = 1
    def at(v, n):
        return v[n] if n >= 0 else 0j
    for n in range(1, 9):
        minus[n] = (2 * (n - 2) * at(minus, n - 2)
                    - (n - 4) * at(minus, n - 4)
                    - 2 * k * at(plus, n - 1) + 2 * k * at(plus, n - 3)) / n
        plus[n] = (2 * (n - 2) * at(plus, n - 2)
                   - (n - 4) * at(plus, n - 4)
                   - 2 * k * at(minus, n - 1) + 2 * k * at(minus, n - 3)) / (n - 4j * q)
    dm = np.polynomial.polynomial.polyval(z, minus)
    ep = np.polynomial.polynomial.polyval(z, plus)
    flux = float(abs(ep)**2 - abs(dm)**2)
    if not flux < 0:
        raise ValueError("wrong signed inner flux")
    norm = math.sqrt(-flux)
    rho = 1 / (1 - z*z)
    xi = math.log(z*z / (1 - z*z))
    return dm / norm, ep * np.exp(-2j * q * (rho + xi)) / norm, flux


def outer_column(k, q, sign, order, rho):
    d = np.zeros(order + 1, dtype=complex)
    e = np.zeros(order + 1, dtype=complex)
    bs = np.ones(order + 1)
    d[0] = 1
    for j in range(1, order + 1):
        bs[j] = bs[j - 1] * (j - 1.5) / j
    for n in range(1, order + 1):
        ep = e[n - 2] if n >= 2 else 0j
        e[n] = ((n - 1) * e[n - 1] - (n - 2) * ep
                - k * np.dot(bs[:n], d[n - 1::-1])) / (2j * sign * q)
        d[n] = ((n - 1) * d[n - 1] + k * np.dot(bs[:n + 1], e[n::-1])) / n
    ds = np.polynomial.polynomial.polyval(1 / rho, d)
    es = np.polynomial.polynomial.polyval(1 / rho, e)
    flux = float(sign * (abs(ds)**2 - abs(es)**2))
    if not sign * flux > 0:
        raise ValueError("wrong signed outer flux")
    norm = math.sqrt(sign * flux)
    return ds / norm, es / norm, flux


def transmission(k, q, setting, z=1e-4):
    global SOLVES, RHS_CALLS
    check_budget()
    if SOLVES >= PHASE_SOLVES or CHARGED_SOLVES + SOLVES >= TOTAL_SOLVE_LIMIT:
        raise BudgetExhausted("ODE solve cap")
    SOLVES += 1
    xi0 = math.log(z*z / (1 - z*z))
    rho = setting["outer_multiplier"] * max(80, 40 * k*k / q)
    xi1 = math.log(rho - 1)
    a0, b0, inner_flux = inner_endpoint(k, q, z)
    before = RHS_CALLS
    def rhs(xi, state):
        global RHS_CALLS
        if RHS_CALLS >= PHASE_RHS or CHARGED_RHS + RHS_CALLS >= TOTAL_RHS_LIMIT:
            raise BudgetExhausted("RHS cap")
        RHS_CALLS += 1
        if RHS_CALLS % 128 == 1:
            check_budget()
        t = math.exp(xi)
        phase = 2 * q * (1 + t + xi)
        oscillation = complex(math.cos(phase), math.sin(phase))
        coupling = -k * math.sqrt(t / (1 + t))
        return np.array((coupling * oscillation * state[1],
                         coupling * oscillation.conjugate() * state[0]))
    sol = solve_ivp(rhs, (xi0, xi1), np.array((a0, b0)), method="DOP853",
                    rtol=setting["rtol"], atol=setting["atol"], t_eval=[xi1])
    if not sol.success:
        raise ValueError("DOP853: " + sol.message)
    final = sol.y[:, -1]
    dm, em, fm = outer_column(k, q, -1, setting["outer_order"], rho)
    dp, ep, fp = outer_column(k, q, 1, setting["outer_order"], rho)
    phase = 2 * q * (rho + xi1)
    oscillation = complex(math.cos(phase), math.sin(phase))
    columns = np.array(((dm, ep * oscillation), (em * oscillation.conjugate(), dp)))
    ain, aout = np.linalg.solve(columns, final)
    gamma = float(1 / abs(ain)**2)
    reflection = float(abs(aout / ain)**2)
    return {"k": k, "q": q, "inner_z": z, "inner_order": 8, "outer_rho": rho,
        "gamma": gamma, "reflection": reflection,
        "gamma_range_defect": max(0, -gamma, gamma - 1),
        "flux_defect": gamma + reflection - 1,
        "inner_raw_flux": inner_flux, "outer_raw_flux_minus": fm, "outer_raw_flux_plus": fp,
        "final_component_abs": [float(abs(x)) for x in final],
        "incoming_abs": float(abs(ain)), "outgoing_abs": float(abs(aout)),
        "endpoint_column_condition": float(np.linalg.cond(columns)),
        "relative_matching_residual": float(np.linalg.norm(columns @ np.array((ain, aout)) - final) / np.linalg.norm(final)),
        "ode_evaluations": int(sol.nfev), "counted_rhs": RHS_CALLS - before}


def duration(q, onset, convention):
    b = (2 * q / 64)**(2 / 3)
    fd = b if convention == "gap" else b / (1 - b)
    fc = float(onset["gap"])
    if fd >= fc:
        return 0.0
    def extra(f):
        z = f / (math.sqrt(1 + f) + 1)
        return 6*z + 3*z*z + 2*z*z*z/3 - 2*math.log1p(z/2)
    return 2*math.log(fc/fd) + extra(fc) - extra(fd)


def integrands(row, onsets):
    gamma = row["gamma"]
    if not 0 <= gamma <= 1:
        return None
    q, k = row["q"], row["k"]
    p = gamma / (math.exp(4*math.pi*q) + 1)
    ent = 0 if p in (0, 1) else -p*math.log(p) - (1-p)*math.log1p(-p)
    weight = 32*k/math.pi * row["weight_dq"]
    return [{"criterion": o["criterion"], "convention": cv, "D": duration(q, o, cv),
        "weighted_S_N_rsE": [weight*duration(q,o,cv)*ent, weight*duration(q,o,cv)*p,
                              weight*duration(q,o,cv)*q*p]}
        for o in onsets for cv in ("gap", "lapse")]


def aggregate(rb, onsets, nx):
    output = []
    for onset in onsets:
        for cv in ("gap", "lapse"):
            channels = []
            for k in range(1, 9):
                rows = [x for x in rb["rows"] if x.get("k")==k and x.get("integrands") is not None]
                vals = [next(i["weighted_S_N_rsE"] for i in x["integrands"]
                             if i["criterion"]==onset["criterion"] and i["convention"]==cv) for x in rows]
                channels.append({"k": k, "completed_nodes": len(rows), "expected_nodes": nx,
                    "complete": len(rows)==nx, "S_N_rsE": [math.fsum(x[j] for x in vals) for j in range(3)]})
            output.append({"criterion": onset["criterion"], "convention": cv, "channels": channels,
                "cumulative": [{"k_cap": cap, "complete": all(x["complete"] for x in channels[:cap]),
                    "S_N_rsE": [math.fsum(x["S_N_rsE"][j] for x in channels[:cap]) for j in range(3)]}
                    for cap in (1,2,4,6,8)]})
    return output


def main():
    global PHASE_WALL, PHASE_SOLVES, PHASE_RHS, CHARGED_WALL, CHARGED_SOLVES, CHARGED_RHS
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("pilot8", "refined12"), required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--stage1-execution", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--previous-execution", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("saved output exists; no ODE repetition")
    stage1 = json.loads(args.stage1.read_text())
    execution = json.loads(args.stage1_execution.read_text())
    assert stage1["complete"] and execution["actual_outer_returncode"]==0
    assert sha(args.stage1)==execution["output_sha256"]
    assert sha(args.stage1)=="1bf75ab20dfbe657e81d671b2fc03edbdc3e45455b065e98f8119145c5fc1ebb"
    CHARGED_WALL = execution["outer_wall_seconds"]
    previous_identity = None
    if args.phase=="refined12":
        assert args.previous and args.previous_execution
        prev = json.loads(args.previous.read_text())
        receipt = json.loads(args.previous_execution.read_text())
        assert prev["complete"] and receipt["actual_outer_returncode"]==0
        assert prev["phase"]=="pilot8" and prev["nx"]==8
        assert len(prev["readbacks"])==2
        for expected_setting, rb in zip(SETTINGS,prev["readbacks"]):
            assert rb["complete"] and rb["settings"]==expected_setting
            assert rb["expected_rows"]==64 and len(rb["rows"])==64
            assert sorted((r["k"],r["quadrature_index"]) for r in rb["rows"])==[
                (k,j) for k in range(1,9) for j in range(8)]
        assert len(prev["inner_diagnostics"])==2
        assert sorted(r["k"] for r in prev["inner_diagnostics"])==[1,8]
        assert all(r["inner_z"]==5e-5 and r["q"]==prev["q_nodes"][0]
                   for r in prev["inner_diagnostics"])
        assert prev["resources"]["new_ode_solves_started"]==130
        assert prev["charged_prior"]["solves"]==0 and prev["charged_prior"]["rhs"]==0
        assert prev["charged_prior"]["wall_seconds"]==CHARGED_WALL
        assert sha(args.previous)==receipt["output_sha256"]
        assert prev["source_sha256"]==sha(Path(__file__))
        CHARGED_WALL += receipt["outer_wall_seconds"]
        CHARGED_SOLVES = prev["resources"]["new_ode_solves_started"]
        CHARGED_RHS = prev["resources"]["new_rhs_evaluations"]
        PHASE_WALL = TOTAL_WALL_LIMIT - CHARGED_WALL - 2
        PHASE_SOLVES = TOTAL_SOLVE_LIMIT - CHARGED_SOLVES
        PHASE_RHS = TOTAL_RHS_LIMIT - CHARGED_RHS
        previous_identity = {"path": str(args.previous), "sha256": sha(args.previous)}
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT_BYTES, AS_LIMIT_BYTES))
    signal.signal(signal.SIGALRM, check_budget)
    signal.alarm(max(1, math.ceil(PHASE_WALL - (time.monotonic()-START))))
    nx = 8 if args.phase=="pilot8" else 12
    node, w = leggauss(nx)
    lo, hi = math.log(1e-8), math.log(1e-3)
    qs = np.exp((hi+lo)/2 + (hi-lo)/2*node)
    weights = w*(hi-lo)/2*qs
    onsets = stage1["scalar_80"]["onsets"]
    result = {"task": "X1 required missing IR integral", "phase": args.phase,
        "source_sha256": sha(Path(__file__)), "stage1_sha256": sha(args.stage1),
        "previous": previous_identity, "host": socket.gethostname(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "nice": os.getpriority(os.PRIO_PROCESS,0),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "thread_environment": {k:os.environ[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        "limits": {"phase_wall_seconds":PHASE_WALL,"registered_pilot_wall_ceiling_seconds":120,
            "phase_solves":PHASE_SOLVES,"phase_rhs":PHASE_RHS,
            "peak_rss_kib":RSS_LIMIT_KIB,"address_space_bytes":AS_LIMIT_BYTES,
            "aggregate_wall_seconds":900,"aggregate_solves":4096,"aggregate_rhs":50_000_000},
        "charged_prior": {"wall_seconds":CHARGED_WALL,"solves":CHARGED_SOLVES,"rhs":CHARGED_RHS},
        "q_interval":[1e-8,1e-3],"nx":nx,"q_nodes":qs.tolist(),"weight_dq":weights.tolist(),
        "scope":"Finite k<=8 missing IR panel only; empirical endpoint/mesh checks, no angular-tail or physical-history clearance.",
        "complete":False,"readbacks":[],"inner_diagnostics":[],"reason":"running"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)
    try:
        for setting in SETTINGS:
            rb = {"settings":setting,"rows":[],"expected_rows":nx*8,"complete":False}
            result["readbacks"].append(rb)
            for k in range(1,9):
                for index,(q,weight) in enumerate(zip(qs,weights)):
                    result["active_attempt"] = {"purpose":"quadrature", "k":k,
                        "q":float(q), "setting":setting["name"], "index":index,
                        "state":"started"}
                    row = transmission(k,float(q),setting)
                    row.update({"quadrature_index":index,"weight_dq":float(weight)})
                    row["integrands"] = integrands(row,onsets)
                    rb["rows"].append(row)
                    result["active_attempt"]["state"] = "completed"
                    atomic_json(args.output,result)
            rb["aggregates"] = aggregate(rb,onsets,nx)
            rb["complete"] = len(rb["rows"])==nx*8 and all(x["integrands"] is not None for x in rb["rows"])
        if args.phase=="pilot8":
            for k in (1,8):
                result["active_attempt"] = {"purpose":"half-inner diagnostic", "k":k,
                    "q":float(qs[0]), "setting":"refined", "state":"started"}
                row = transmission(k,float(qs[0]),SETTINGS[1],z=5e-5)
                base = next(x for x in result["readbacks"][1]["rows"] if x["k"]==k and x["quadrature_index"]==0)
                row.update({"base_row_index":result["readbacks"][1]["rows"].index(base),
                    "gamma_minus_base":row["gamma"]-base["gamma"],
                    "gamma_relative_change":None if base["gamma"]==0 else (row["gamma"]-base["gamma"])/base["gamma"]})
                result["inner_diagnostics"].append(row)
                result["active_attempt"]["state"] = "completed"
                atomic_json(args.output,result)
        result["complete"] = all(x["complete"] for x in result["readbacks"])
        result["reason"] = "prescribed phase completed; no automatic next phase"
    except Exception as error:
        result["reason"] = type(error).__name__+": "+str(error)
    finally:
        signal.alarm(0)
        for rb in result["readbacks"]:
            rb["aggregates"] = aggregate(rb,onsets,nx)
        atomic_json(args.output,result)
        print(json.dumps({"phase":args.phase,"complete":result["complete"],"reason":result["reason"],"resources":result["resources"]}))


if __name__=="__main__":
    main()
