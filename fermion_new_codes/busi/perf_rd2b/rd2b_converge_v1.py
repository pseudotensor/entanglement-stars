#!/usr/bin/env python
"""rd2b_converge_v1.py (busi, analyst; PERF RD-2b, supporting evidence): the negative-centre collocation's discretization at the production
setting (p 8, 2 panels per ell, grading K 6) against refined settings (p 10 and 12, 4 panels per ell, K 4 and 10) and against v16c at
rtol 1e-14 (P14), per record: E rel, the dense N, v on rfine, the averaged site fields, r_*.
usage: python rd2b_converge_v1.py CODE16C V16E FIELDS_NPZ --recs i,j,..."""
import argparse, importlib.util, json, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("code"); ap.add_argument("v16e"); ap.add_argument("npz"); ap.add_argument("--recs", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code)
import engine_fields_v16 as F, engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_kcache_v1 as KC
spec = importlib.util.spec_from_file_location("engine_fields_v16e", a.v16e); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
KC.install([F4, F12, F14, F, E])
z = np.load(a.npz, allow_pickle=True)
R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"]); sv = json.loads(str(z["solver"]))
cp = sv.get("collar_policy", "flag"); sr = float(sv.get("support_rel", 1e-14))
E._RD2["route"] = "collocation"
VAR = [("base p8 ppe2 K6", 8, 2, 6), ("p10", 10, 2, 6), ("p12", 12, 2, 6), ("ppe4", 8, 4, 6), ("K4", 8, 2, 4), ("K10", 8, 2, 10), ("p12 ppe4 K10", 12, 4, 10)]
for i in [int(x) for x in a.recs.split(",")]:
    eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
    E._RD2["route"] = "ivp"
    _, _, _, I14 = F.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-14, atol=1e-18, collar_policy=cp, support_rel=sr)
    _, _, _, I12 = F.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, collar_policy=cp, support_rel=sr)
    rf = I14["rfine"]
    out = {}
    for name, p, ppe, K in VAR:
        E._RD2["route"] = "collocation"; E._RD2["p"] = p; E._RD2["ppe"] = ppe; E._RD2["flat_K"] = K
        Ns, vs, Wn, info = E.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, collar_policy=cp, support_rel=sr)
        out[name] = info
    E._RD2["p"] = 8; E._RD2["ppe"] = 2; E._RD2["flat_K"] = 6
    base = out["base p8 ppe2 K6"]
    print("rec %d T %g: route %s, r_* %s (P14 %s, P12 %s)" % (i, z["T"][i], base["fields_route"].get("branch", "intact")[:30], base["fields_route"].get("r_star"),
          (I14["gauge"]["zeros"] or [{}])[0].get("r"), (I12["gauge"]["zeros"] or [{}])[0].get("r")))
    def d(X, Y):
        return "E %.1e N %.1e v %.1e W %.1e U %.1e" % (abs(X["E_collar"] - Y["E_collar"]) / abs(Y["E_collar"]), np.max(np.abs(X["N_dense"](rf) - Y["N_dense"](rf))),
                                               np.max(np.abs(X["v_dense"](rf) - Y["v_dense"](rf))), np.max(np.abs(X["W_field"](rf) - Y["W_field"](rf))),
                                               np.max(np.abs(X["U_dense"](rf) - Y["U_dense"](rf))))
    for name in out:
        print("   %-14s vs P14: %s | vs base: %s | Newton %s, panels %s" % (name, d(out[name], I14), d(out[name], base), out[name]["fields_route"].get("newton_iterations"), out[name]["fields_route"].get("panels")))
    print("   P12            vs P14: %s" % d(I12, I14), flush=True)
