#!/usr/bin/env python
"""rd2b_diag_v1.py (busi, analyst; PERF RD-2b step 1): per stored record, the production solve's (v16c) branch structure: the centre
series, the segments' signs, the zeros, the stretches, the gauge label, and why v16d's collocation route falls back.
usage: python rd2b_diag_v1.py CODE_DIR V16D FIELDS_NPZ [--recs i,j,...|--all]"""
import argparse, importlib.util, json, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("code"); ap.add_argument("v16d"); ap.add_argument("npz"); ap.add_argument("--recs", default=None); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.code)
import engine_fields_v16 as F, engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_kcache_v1 as KC
spec = importlib.util.spec_from_file_location("engine_fields_v16d", a.v16d); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
KC.install([F4, F12, F14, F, D])
z = np.load(a.npz, allow_pickle=True)
R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"]); sv = json.loads(str(z["solver"]))
cp = sv.get("collar_policy", "flag"); sr = float(sv.get("support_rel", 1e-14))
idx = range(len(z["T"])) if a.all else [int(x) for x in a.recs.split(",")]
D.rd2_configure("collocation", "off")
for i in idx:
    eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
    t0 = time.time(); Nn, vn, Wn, info = F.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, collar_policy=cp, support_rel=sr); t1 = time.time()
    if info.get("failed"):
        print("rec %d T %g: v16c failed: %s" % (i, z["T"][i], info["failed"])); continue
    ga = info["gauge"]; cc = info["ivp"]["series"]
    segsign = None
    st, out = D._rd2_collocation(eps, J, g, R, ell, 1e-12, 1e-16, cp, sr, 1.0 / 64.0)
    print("rec %d T %g: %.2f s nfev %d segs %d | rho0 %.3e rho2 %.3e a %.3e b %.3e U0 %.3e | zeros %s stretches %d near_tang %d U_scale %.3e | gauge %s | min U/Uref %.3e dLb %s | regime %s | E %.12g Nmin %.6f | colloc: %s"
          % (i, z["T"][i], t1 - t0, info["ivp"]["nfev"], info["ivp"]["segments"], cc["rho0"], cc["rho2"], cc["a"], cc["b"], info["ivp"]["U0"],
             [(round(q["r"], 4), q["kind"], "%.2e" % q["U_prime"]) for q in ga["zeros"]], ga["stretches"], ga["near_tangencies"], ga["U_scale"],
             ga["label"][:120], ga["min_U_over_U_ref"], ga.get("delta_L_bound_two_resolutions"), info["regime"]["label"][:60], info["E_collar"], info["N_min"],
             st if st == "ok" else out), flush=True)
