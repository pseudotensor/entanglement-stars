#!/usr/bin/env python
"""field_numba_test_v1.py (solve builder 5; REVIEWA_gpu_economics_v1 sec. 4, the measurement before the compiled integrator): one field solve
of the engine (engine_fields_v17.solve_constraints_ivp with the engine's tolerances, collar policy flag, then the kernel average onto the
sites; the kernel cache installed as --field-kernel-cache does) on the r_s/a 32 cost run's stored sources at T/r_s 0.0625, timed 5 times
after a warm call; a cProfile split of 1 solve; then the same solve with the right-hand side's source evaluations (the quintic splines
rho_ell(r), J_ell(r), called twice per right-hand-side call) replaced by a numba de Boor evaluation of the same knots and coefficients, under
the same adaptive integrator (scipy's DOP853 with the same rtol, atol and events), timed the same way, with the largest differences of the
site fields N, v against the plain solve. Usage: field_numba_test_v1.py <fields.npz> <T/r_s> <out.json> (box, qi2 env)."""
import cProfile, json, pstats, sys, time
import numpy as np
sys.path.insert(0, "<internal>/code")
import engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_v16 as F16, engine_fields_v17 as F17
import engine_fields_kcache_v1 as KC
import numba

KC.install([F4, F12, F14, F16, F17])
fz, Tq, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
Z = np.load(fz, allow_pickle=True)
rs = 32.0; g = 0.25; ell = 2
k = int(np.argmin(np.abs(Z["T"] / rs - Tq))); R = np.asarray(Z["R"], float)
eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)


def solve():
    Nn, vn, Wn, info = F17.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        raise RuntimeError(info["failed"])
    Na, va, _ = F17.site_fields_kernel_average(info, R, ell)
    return Na, va, info


def timed(n=5):
    w = []
    for _ in range(n):
        t0 = time.perf_counter(); r_ = solve(); w.append(time.perf_counter() - t0)
    return w, r_


solve()                                   # warm: the kernel cache filled
w0, (N0, v0, info0) = timed()
nfev = sum(int(s[4]) for s in info0.get("segments", [])) if isinstance(info0.get("segments"), list) else info0.get("ivp", {}).get("nfev")
pr = cProfile.Profile(); pr.enable(); solve(); pr.disable()
st = pstats.Stats(pr); tot = st.total_tt
split = {}
for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
    key = None
    if name in ("rhs_pos", "rhs_neg", "rhs_s", "Uprime"):
        key = "rhs_" + name
    elif name == "__call__" and "_bsplines" in fn:
        key = "BSpline.__call__"
    elif name == "solve_ivp":
        key = "solve_ivp"
    elif name == "integrate_UL_v17":
        key = "integrate_UL_v17"
    elif name == "site_fields_kernel_average":
        key = "site_fields_kernel_average"
    elif name == "__init__" and "engine_fields_v12" in fn:
        key = "SmoothedSources5.__init__"
    elif name == "<lambda>" and "engine_fields_v17" in fn:
        key = "solW_lambda"
    if key:
        e = split.setdefault(key, {"calls": 0, "cumulative_s": 0.0, "own_s": 0.0})
        e["calls"] += nc; e["cumulative_s"] += ct; e["own_s"] += tt


@numba.njit(cache=False)
def deboor(t, c, k, x):
    n = c.shape[0]
    lo = k; hi = n
    if x < t[k]:
        l = k
    elif x >= t[n]:
        l = n - 1
    else:
        l = k
        while lo < hi:                    # the largest l in [k, n-1] with t[l] <= x
            mid = (lo + hi) // 2
            if t[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        l = lo - 1
        if l > n - 1:
            l = n - 1
    d = np.empty(k + 1)
    for j in range(k + 1):
        d[j] = c[j + l - k]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            i = j + l - k
            den = t[i + k + 1 - r] - t[i]
            al = (x - t[i]) / den if den != 0.0 else 0.0
            d[j] = (1.0 - al) * d[j - 1] + al * d[j]
    return d[k]


class FastSpline:
    def __init__(self, b):
        self.b = b; self.t = np.ascontiguousarray(b.t, float); self.c = np.ascontiguousarray(b.c, float); self.k = int(b.k)

    def __call__(self, x, *a, **kw):
        if isinstance(x, float) and not a and not kw:
            return deboor(self.t, self.c, self.k, x)
        return self.b(x, *a, **kw)


Orig = F17.SmoothedSources5 if hasattr(F17, "SmoothedSources5") else F12.SmoothedSources5


class NbSources(Orig):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rho = FastSpline(self.rho); self.J = FastSpline(self.J)


mods = [m for m in (F17, F16, F12) if getattr(m, "SmoothedSources5", None) is Orig]
for m in mods:
    m.SmoothedSources5 = NbSources
solve()                                   # warm: the JIT compiled
xs = np.linspace(-5.0, 300.0, 2001)
src_chk = NbSources(eps, J, R, ell, 400.0)
spl_diff = max(abs(src_chk.rho(float(x)) - float(src_chk.rho.b(x))) for x in xs)
w1, (N1, v1, info1) = timed()
for m in mods:
    m.SmoothedSources5 = Orig
r = {"record_T_over_rs": float(Z["T"][k] / rs), "plain_walls_s": w0, "numba_rhs_walls_s": w1,
     "plain_median_s": float(np.median(w0)), "numba_median_s": float(np.median(w1)), "gain": float(np.median(w0) / np.median(w1)),
     "max_abs_dN": float(np.max(np.abs(N1 - N0))), "max_abs_dv": float(np.max(np.abs(v1 - v0))), "spline_eval_max_abs_diff": float(spl_diff),
     "profile_total_s": tot, "profile_split": split, "numba": numba.__version__, "numpy": np.__version__}
print(json.dumps(r, indent=1)); json.dump(r, open(out, "w"), indent=1)
