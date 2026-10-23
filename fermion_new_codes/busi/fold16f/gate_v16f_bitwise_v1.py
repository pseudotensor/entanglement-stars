#!/usr/bin/env python
"""gate_v16f_bitwise_v1.py (busi, analyst; ReviewA_s4fail_v1 sec. 2, inbox #444): the gate of the fold exit, box CPU.
Mode rows: every chosen row of a fields npz (its stored eps_tot, J_tot) is solved by engine_fields_v16d and engine_fields_v16f in one process,
on the ivp route and on the collocation route (production's, with its ivp fallback), warm start off, at the npz's own g, ell, collar policy and
support_rel, rtol 1e-12 / atol 1e-16. Per solve, BITWISE: the failure string; N_sites, v_sites, W_nodes (bytes); the averaged site fields of
site_fields_kernel_average (bytes); every non-callable info entry (arrays by md5 of their bytes, recursively), with only the wall-clock keys
dropped ("timing", "seconds" and keys ending in "_s"). PASS = every solve of every row equal.
Mode fail: the five reproduced failures of F256_s4_fieldfail_T050_v1 (S(lambda) = S(0.4375) + lambda (S(0.4375) - S(0.375)), lambda 2.00,
2.02, 2.04, 2.06, 2.08): v16d must fail (the downward stretch), v16f must pass with a fold exit, and v16f's fields must agree with the plain
r-integration (v16d with the switch level 1e-14 U_scale) to the note's max |dN| 1.9e-8 and max |dv| 1.0e-9 (the note's 1.0e-9 is the
rounded maximum 1.048e-9, so the line is <= 1.05e-9), on both routes.
    python gate_v16f_bitwise_v1.py rows --code DIR --v16d PATH --v16f PATH --fields NPZ [--last N] --tag T --out JSON
    python gate_v16f_bitwise_v1.py fail --code DIR --v16d PATH --v16f PATH --fields F256_s4_NPZ --tag T --out JSON
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

DROPPED = set()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def san(v, key=""):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k == "src" or callable(x):
                continue
            if k in ("timing", "seconds") or (isinstance(k, str) and k.endswith("_s")):
                DROPPED.add(k)
                continue
            out[str(k)] = san(x, k)
        return out
    if isinstance(v, (list, tuple)):
        return [san(x) for x in v]
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return [san(x) for x in v.tolist()]
        return hashlib.md5(np.ascontiguousarray(v).tobytes()).hexdigest() + str(v.shape) + str(v.dtype)
    if isinstance(v, (np.floating, float)):
        return float(v).hex()
    if isinstance(v, (np.integer,)):
        return int(v)
    if callable(v):
        return "<callable>"
    return v if isinstance(v, (str, int, bool, type(None))) else repr(v)


def bhash(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [bhash(y) for y in x]
    x = np.asarray(x)
    if x.dtype == object:
        return [bhash(y) for y in x.tolist()]
    return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest() + str(x.shape) + str(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "fail"])
    ap.add_argument("--code", required=True); ap.add_argument("--v16d", required=True); ap.add_argument("--v16f", required=True)
    ap.add_argument("--fields", required=True); ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for k in ("QI2_FIELDS_ROUTE", "QI2_FIELDS_WARM"):
        os.environ.pop(k, None)
    sys.path.insert(0, a.code)
    FD = load("engine_fields_v16d_gate", a.v16d); FF = load("engine_fields_v16f_gate", a.v16f)
    md5 = {n: hashlib.md5(open(p, "rb").read()).hexdigest() for n, p in (("v16d", a.v16d), ("v16f", a.v16f))}
    assert md5["v16d"].startswith("7a69faad"), md5
    z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(z["R"], float); g = float(z["g"]); ell = z["ell"].item()
    try:
        solver = json.loads(str(z["solver"]))
    except Exception:
        solver = {}
    cp = solver.get("collar_policy", "flag"); sr = float(solver.get("support_rel", 1e-14))
    T = np.asarray(z["T"], float)
    head = {"tag": a.tag, "mode": a.mode, "fields": a.fields, "md5": md5, "sites": len(R), "g": g, "ell": float(ell), "collar_policy": cp,
            "support_rel": sr, "threads": os.environ.get("OMP_NUM_THREADS"), "numpy": np.__version__, "python": sys.version.split()[0]}
    print(json.dumps(head), flush=True)

    def solve(mod, route, eps, J, ul=None):
        mod._RD2["route"] = route; mod._RD2["warm"] = False; mod._RD2["last"] = None
        orig = mod.integrate_UL
        if ul is not None:
            mod.integrate_UL = ul
        try:
            Nn, vn, Wn, info = mod.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy=cp, support_rel=sr)
        finally:
            mod.integrate_UL = orig
            mod._RD2["route"] = "ivp"
        out = {"failed": info.get("failed"), "N": Nn, "v": vn, "W": Wn, "info": info}
        if not info.get("failed"):
            out["avg"] = mod.site_fields_kernel_average(info, R, ell)
        return out

    res = {"head": head, "rows": [], "fail_cases": []}
    if a.mode == "rows":
        idx = list(range(len(T))) if a.last <= 0 else list(range(max(0, len(T) - a.last), len(T)))
        n_eq = 0; n_all = 0; t0 = time.time()
        for i in idx:
            eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
            row = {"i": i, "T": float(T[i])}
            for route in ("ivp", "collocation"):
                D = solve(FD, route, eps, J); F = solve(FF, route, eps, J)
                chk = {"failed": D["failed"] == F["failed"], "N": bhash(D["N"]) == bhash(F["N"]), "v": bhash(D["v"]) == bhash(F["v"]),
                       "W": bhash(D["W"]) == bhash(F["W"]), "avg": bhash(D.get("avg")) == bhash(F.get("avg")),
                       "info": json.dumps(san(D["info"]), sort_keys=True) == json.dumps(san(F["info"]), sort_keys=True)}
                ok = all(chk.values())
                n_all += 1; n_eq += int(ok)
                fr = F["info"].get("fields_route") or {}
                row[route] = {"bitwise": ok, "checks": chk, "failed": D["failed"],
                              "route_taken": fr.get("route", "ivp" if route == "ivp" else "collocation"),
                              "fold_exit": bool(((F["info"].get("gauge") or {}).get("fold_exits")))}
            res["rows"].append(row)
            print(json.dumps({"i": i, "T": float(T[i]), "ivp": row["ivp"]["bitwise"], "colloc": row["collocation"]["bitwise"],
                              "colloc_route": row["collocation"]["route_taken"], "failed": row["ivp"]["failed"]}), flush=True)
        res["summary"] = {"solves": n_all, "bitwise_equal": n_eq, "PASS": n_eq == n_all, "rows": len(idx), "seconds": time.time() - t0,
                          "dropped_keys": sorted(DROPPED)}
    else:
        ti = {round(float(t) / 256.0, 4): k for k, t in enumerate(T)}
        i1, i0 = ti[0.4375], ti[0.375]
        E = np.asarray(z["eps_tot"], float); Jt = np.asarray(z["J_tot"], float)
        ul14 = functools.partial(FD.integrate_UL, eps_switch=1e-14)
        allok = True
        for lam in (2.00, 2.02, 2.04, 2.06, 2.08):
            eps = E[i1] + lam * (E[i1] - E[i0]); J = Jt[i1] + lam * (Jt[i1] - Jt[i0])
            c = {"lambda": lam}
            for route in ("ivp", "collocation"):
                D = solve(FD, route, eps, J); F = solve(FF, route, eps, J); Rf = solve(FD, route, eps, J, ul=ul14)
                fx = (F["info"].get("gauge") or {}).get("fold_exits")
                e = {"v16d_failed": D["failed"], "v16f_failed": F["failed"], "ref_failed": Rf["failed"], "fold_exits": fx}
                if not F["failed"] and not Rf["failed"]:
                    e["max_dN"] = float(np.max(np.abs(F["N"] - Rf["N"]))); e["max_dv"] = float(np.max(np.abs(F["v"] - Rf["v"])))
                    e["N_min"] = float(np.min(F["N"])); e["regime_max"] = float(F["info"]["regime"]["max_ratio"])
                    e["gauge_label"] = F["info"]["gauge"]["label"][:140]
                e["PASS"] = bool(D["failed"] and D["failed"].startswith("the downward stretch") and not F["failed"] and fx
                                 and e.get("max_dN", 1.0) <= 1.9e-8 and e.get("max_dv", 1.0) <= 1.05e-9)
                allok &= e["PASS"]
                c[route] = e
            res["fail_cases"].append(c)
            print(json.dumps(c, default=str)[:900], flush=True)
        res["summary"] = {"PASS": bool(allok), "cases": 5}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res["summary"]), flush=True)


if __name__ == "__main__":
    main()
