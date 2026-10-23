"""field_kcache_test_v1.py (the lane's analyst builder; engine spec 42.1 (3) (ii)): the cached kernels against the uncached solve, bitwise, on
the cost run's stored sources at 3 records and one perturbed source (a Newton-like trial), with walls."""
import sys, time, numpy as np
sys.path.insert(0, "<internal>/code")
import engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_v16 as F16, engine_fields_v17 as F
import engine_fields_kcache_v1 as KC
Z = np.load("<internal>/collapse_rs32_cost_ell2_fields.npz", allow_pickle=True); R = np.asarray(Z["R"], float)
def run(eps, J):
    out = F.solve_constraints_ivp(eps, J, 0.25, R, 2.0, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    Na, va, nr = F.site_fields_kernel_average(out[3], R, 2.0)
    return np.concatenate([out[0], out[1], Na, va])
cases = [(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float)) for k in range(min(3, len(Z["T"])))]
rng = np.random.default_rng(1); e0, j0 = cases[-1]
cases.append((e0 * (1 + 1e-6 * rng.standard_normal(len(e0))), j0 * (1 + 1e-6 * rng.standard_normal(len(j0)))))
ref = []
for e, j in cases:
    t0 = time.time(); ref.append(run(e, j)); print("uncached %.3f s" % (time.time() - t0))
KC.install([F4, F12, F14, F16, F])
for i, (e, j) in enumerate(cases):
    for rep in range(2):
        t0 = time.time(); x = run(e, j); w = time.time() - t0
        print("case %d rep %d cached %.3f s; bitwise equal %s; max |diff| %.1e; stats %s" % (i, rep, w, np.array_equal(x, ref[i]), np.max(np.abs(x - ref[i])), KC.STATS))
