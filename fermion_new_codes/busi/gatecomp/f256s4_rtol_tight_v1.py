import sys, os, json, numpy as np
sys.path.insert(0, sys.argv[1]); os.environ["QI2_FIELDS_ROUTE"] = "ivp"
import engine_fields_v16 as FLD
z = np.load(sys.argv[2], allow_pickle=True)
R = z["R"]; ell = int(z["ell"]); g = float(z["g"]); T = np.round(z["T"] / 256.0, 4).tolist()
E = z["eps_tot"]; J = z["J_tot"]; i1, i0 = T.index(0.4375), T.index(0.375)
for lam in (2.0, 2.06):
    e = E[i1] + lam * (E[i1] - E[i0]); j = J[i1] + lam * (J[i1] - J[i0])
    for rt, at in ((1e-13, 1e-17), (1e-14, 1e-18), (1e-11, 1e-15), (1e-9, 1e-13)):
        info = FLD.solve_constraints_ivp(e, j, g, R, ell, rtol=rt, atol=at, collar_policy="flag", support_rel=1e-14)[3]
        print(json.dumps({"lam": lam, "rtol": rt, "atol": at, "failed": info.get("failed")}), flush=True)
