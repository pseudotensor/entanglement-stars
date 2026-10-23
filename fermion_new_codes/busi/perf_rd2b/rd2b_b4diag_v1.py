#!/usr/bin/env python
"""rd2b_b4diag_v1.py (busi, analyst; PERF RD-2b, the B4 line at one record): where the site-field differences of Ce, P12, P14 and a refined
collocation (4 panels per ell, K 10) sit, with W and r_* there.  usage: CODE16C V16E NPZ REC"""
import json, sys, importlib.util
import numpy as np
code, v16e, npz, rec = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, code)
import engine_fields_v16 as F, engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_kcache_v1 as KC
spec = importlib.util.spec_from_file_location("engine_fields_v16e", v16e); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
KC.install([F4, F12, F14, F, E])
z = np.load(npz, allow_pickle=True); R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"]); sv = json.loads(str(z["solver"]))
kw = dict(collar_policy=sv.get("collar_policy", "flag"), support_rel=float(sv.get("support_rel", 1e-14)))
eps = np.asarray(z["eps_tot"][rec], float); J = np.asarray(z["J_tot"][rec], float)
out = {}
out["P12"] = F.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, **kw)
out["P14"] = F.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-14, atol=1e-18, **kw)
E._RD2["route"] = "collocation"
out["Ce"] = E.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, **kw)
E._RD2["ppe"] = 4; E._RD2["flat_K"] = 10
out["Cr"] = E.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-12, atol=1e-16, **kw)
rs = out["Ce"][3]["fields_route"].get("r_star") or 0.0
print("rec %d: r_* Ce %s Cr %s P14 %s" % (rec, rs, out["Cr"][3]["fields_route"].get("r_star"), (out["P14"][3]["gauge"]["zeros"] or [{}])[0].get("r")))
Wc = out["Ce"][3]["W_field"](R)
for a, b in (("Ce", "P14"), ("P12", "P14"), ("Ce", "Cr"), ("P14", "Cr"), ("P12", "Cr")):
    dv = np.abs(out[a][1] - out[b][1]); k = int(np.argmax(dv))
    dW = np.abs(out[a][3]["W_field"](R) - out[b][3]["W_field"](R))
    print("%s - %s: max |dv| %.2e at R %.4f (W there %.3e, v %.3e; R - r_* %.3e); max |dW| %.2e at R %.4f; max |dN| %.2e; E rel %.1e" % (
        a, b, dv[k], R[k], Wc[k], out[a][1][k], R[k] - rs, dW.max(), R[int(np.argmax(dW))], np.max(np.abs(out[a][0] - out[b][0])),
        abs(out[a][3]["E_collar"] - out[b][3]["E_collar"]) / abs(out[b][3]["E_collar"])))
i0 = np.where(Wc > 0)[0]
print("W first positive site R %.4f (W %.3e), last nonpositive before it R %.4f (W %.3e)" % (R[i0[0]], Wc[i0[0]], R[i0[0] - 1] if i0[0] else -1, Wc[i0[0] - 1] if i0[0] else 0))
