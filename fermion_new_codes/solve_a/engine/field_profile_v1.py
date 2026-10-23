"""field_profile_v1.py (the lane's analyst builder; engine spec 42.1 (3) (ii)): one field solve at 2048 sites on the cost run's stored sources
(record T/r_s 0.0625), timed and profiled (cProfile, top entries by own time), engine_fields_v17 with the engine's tolerances and site averaging."""
import sys, time, cProfile, pstats, io, numpy as np
sys.path.insert(0, "<internal>/code")
import engine_fields_v17 as F
Z = np.load("<internal>/collapse_rs32_cost_ell2_fields.npz", allow_pickle=True)
print([k for k in Z.files])
k = 1; eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float); R = np.asarray(Z["R"], float)
def one():
    out = F.solve_constraints_ivp(eps, J, 0.25, R, 2.0, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    Na, va, nr = F.site_fields_kernel_average(out[3], R, 2.0)
    return out, Na, va
t0 = time.time(); out, Na, va = one(); t1 = time.time()
print("solve %.3f s; ivp nfev %s; failed %s" % (t1 - t0, (out[3].get("ivp") or {}).get("nfev"), out[3].get("failed")))
pr = cProfile.Profile(); pr.enable(); one(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14); print(s.getvalue()[:4000])
