#!/usr/bin/env python
"""cv_riemann_sympy.py (busi lane, correctness reading of CURVATURE_readout_v1): own symbolic derivation of
  (1) the orthonormal slice-frame Riemann components of ds^2 = -N^2 dT^2 + (dr - v dT)^2 + r^2 dOmega^2 with N(T, r), v(T, r),
      frame e0 = (d_T + v d_r)/N, e1 = d_r, e2 = d_theta/r, e3 = d_phi/(r sin theta); the frame norm (sum over all abcd of R_abcd^2) and the
      Kretschmann scalar by brute force over all 256 index orderings (no multiplicity bookkeeping);
  (2) the radial null geodesics in Hamiltonian form (H = g^{mu nu} k_mu k_nu / 2; a different route from the delivery's Christoffel check):
      dr/dT, the slice-frame energy eps = -n^mu k_mu, d ln eps / dT along the ray, and dT/dlambda;
  (3) the Einstein tensor's constraint components G_nn, G_nr, and the inversion of the Ricci identities used by the delivery's second route.
Each result is compared with the delivery's printed formulas (typed in from CURVATURE_readout_v1.md sec. 1). Light symbolic work only.
"""
import json
import sys

import sympy as sp

T, r, th, ph = sp.symbols("T r theta phi", real=True)
X = [T, r, th, ph]
N = sp.Function("N")(T, r)
v = sp.Function("v")(T, r)

# covariant metric
g = sp.zeros(4, 4)
g[0, 0] = -N**2 + v**2
g[0, 1] = g[1, 0] = -v
g[1, 1] = 1
g[2, 2] = r**2
g[3, 3] = r**2 * sp.sin(th)**2
ginv = sp.simplify(g.inv())

# Christoffels Gamma^a_{bc}
Gam = [[[sp.simplify(sum(ginv[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d])) for d in range(4)) / 2)
         for c in range(4)] for b in range(4)] for a in range(4)]


def riem_up(a, b, c, d):
    """R^a_{bcd} = d_c Gam^a_{bd} - d_d Gam^a_{bc} + Gam^a_{ce} Gam^e_{bd} - Gam^a_{de} Gam^e_{bc}"""
    e = sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
    e += sum(Gam[a][c][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][c] for k in range(4))
    return e


# full covariant tensor, all 256 entries (brute force; spherical symmetry makes most vanish)
Rcov = {}
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                Rcov[(a, b, c, d)] = sp.simplify(sum(g[a, k] * riem_up(k, b, c, d) for k in range(4)))

# tetrad E[A, mu] = e_A^mu
E = sp.zeros(4, 4)
E[0, 0] = 1 / N
E[0, 1] = v / N
E[1, 1] = 1
E[2, 2] = 1 / r
E[3, 3] = 1 / (r * sp.sin(th))
eta = sp.diag(-1, 1, 1, 1)
assert sp.simplify(E * g * E.T - eta) == sp.zeros(4, 4), "tetrad is not orthonormal"

Rf = {}
for A in range(4):
    for B in range(4):
        for C in range(4):
            for D in range(4):
                tot = 0
                for a in range(4):
                    if E[A, a] == 0:
                        continue
                    for b in range(4):
                        if E[B, b] == 0:
                            continue
                        for c in range(4):
                            if E[C, c] == 0:
                                continue
                            for d in range(4):
                                if E[D, d] == 0:
                                    continue
                                tot += E[A, a] * E[B, b] * E[C, c] * E[D, d] * Rcov[(a, b, c, d)]
                Rf[(A, B, C, D)] = sp.simplify(tot)

norm2_bf = sp.simplify(sum(x**2 for x in Rf.values()))
kret_bf = sp.simplify(sum(x**2 * (-1) ** ([A, B, C, D].count(0)) for (A, B, C, D), x in Rf.items()))

# the delivery's printed formulas (CURVATURE_readout_v1 sec. 1), in plain symbols
Nt, Nr, Nrr, vt, vr, vrr, vtr, Ns, vs, rs = sp.symbols("N_t N_r N_rr v_t v_r v_rr v_tr N v r")
deliv = {
    (0, 1, 0, 1): (Ns * Nrr - vs * vrr - vr**2 - vtr) / Ns**2 + (Nr * vs * vr + vr * Nt) / Ns**3,
    (0, 2, 0, 2): (Ns**2 * Nr - Ns * vs * vr - Ns * vt + Nr * vs**2 + vs * Nt) / (Ns**3 * rs),
    (0, 2, 1, 2): vs * Nr / (Ns**2 * rs),
    (1, 2, 1, 2): vs * vr / (Ns**2 * rs),
    (2, 3, 2, 3): (vs / (rs * Ns))**2,
}
rep = [(sp.Derivative(N, T, r), sp.Symbol("N_tr")), (sp.Derivative(N, r, T), sp.Symbol("N_tr")), (sp.Derivative(v, T, r), vtr), (sp.Derivative(v, r, T), vtr),
       (sp.Derivative(N, (r, 2)), Nrr), (sp.Derivative(v, (r, 2)), vrr), (sp.Derivative(N, T), Nt), (sp.Derivative(N, r), Nr),
       (sp.Derivative(v, T), vt), (sp.Derivative(v, r), vr)]


def plain(e):
    for a_, b_ in rep:
        e = e.subs(a_, b_)
    return sp.simplify(e.subs({N: Ns, v: vs}).subs(r, rs))


report = {"components": {}, "checks": {}}
ok_all = True
for key, mine in Rf.items():
    A, B, C, D = key
    if A < B and C < D and (A, B) <= (C, D) and mine != 0:
        report["components"]["R%d%d%d%d" % key] = str(plain(mine))
for key, dv in deliv.items():
    diff = sp.simplify(plain(Rf[key]) - dv)
    report["checks"]["R%d%d%d%d minus delivery" % key] = str(diff)
    ok_all &= (diff == 0)
# duplicates by symmetry and the vanishing ones
report["checks"]["R0303 - R0202"] = str(sp.simplify(Rf[(0, 3, 0, 3)] - Rf[(0, 2, 0, 2)]))
report["checks"]["R0313 - R0212"] = str(sp.simplify(Rf[(0, 3, 1, 3)] - Rf[(0, 2, 1, 2)]))
report["checks"]["R1313 - R1212"] = str(sp.simplify(Rf[(1, 3, 1, 3)] - Rf[(1, 2, 1, 2)]))
report["checks"]["R0123"] = str(Rf[(0, 1, 2, 3)]); report["checks"]["R0213"] = str(Rf[(0, 2, 1, 3)]); report["checks"]["R0102"] = str(Rf[(0, 1, 0, 2)])
# norm and Kretschmann against the delivery's bookkeeping
R0101, R0202, R0212, R1212, R2323 = (Rf[k] for k in ((0, 1, 0, 1), (0, 2, 0, 2), (0, 2, 1, 2), (1, 2, 1, 2), (2, 3, 2, 3)))
norm2_deliv = 4 * (R0101**2 + 2 * R0202**2 + 2 * R1212**2 + R2323**2) + 16 * R0212**2
kret_deliv = 4 * (R0101**2 + 2 * R0202**2 + 2 * R1212**2 + R2323**2) - 16 * R0212**2
report["checks"]["norm^2 brute force minus delivery bookkeeping"] = str(sp.simplify(norm2_bf - norm2_deliv))
report["checks"]["Kretschmann brute force minus delivery bookkeeping"] = str(sp.simplify(kret_bf - kret_deliv))
ok_all &= sp.simplify(norm2_bf - norm2_deliv) == 0 and sp.simplify(kret_bf - kret_deliv) == 0

# K_t, K_r and the Gauss relations; Einstein constraint components
Kt = v / (r * N); Kr = sp.diff(v, r) / N
report["checks"]["R2323 - K_t^2"] = str(sp.simplify(R2323 - Kt**2))
report["checks"]["R1212 - K_r K_t"] = str(sp.simplify(R1212 - Kr * Kt))
Ric = {}
for (A, B) in [(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)]:
    Ric[(A, B)] = sp.simplify(sum(eta[C, C] * Rf[(C, A, C, B)] for C in range(4)))
Rs_ = sp.simplify(sum(eta[A, A] * Ric[(A, A)] for A in range(4)))
G00 = sp.simplify(Ric[(0, 0)] + Rs_ / 2)
G01 = sp.simplify(Ric[(0, 1)])
report["checks"]["G_nn - (K_t^2 + 2 K_r K_t)"] = str(sp.simplify(G00 - (Kt**2 + 2 * Kr * Kt)))
report["checks"]["G_nr - 2 v N_r/(N^2 r)"] = str(sp.simplify(G01 - 2 * v * sp.diff(N, r) / (N**2 * r)))
# inversion used by the delivery's second route: given Ric_AB = 8 pi g (T_AB - eta_AB T/2) with T = diag(rho, Sr, St, St) and T01, and R2323:
rho, Sr, St, T01, Kt2, pg = sp.symbols("rho S_r S_t T01 Kt2 pg")
Ric00 = pg * (rho + (-rho + Sr + 2 * St) / 2); Ric11 = pg * (Sr - (-rho + Sr + 2 * St) / 2); Ric22 = pg * (St - (-rho + Sr + 2 * St) / 2); Ric01 = pg * T01
a0101, a0202, a0212, a1212 = sp.symbols("a0101 a0202 a0212 a1212")
sol = sp.solve([sp.Eq(a0101 + 2 * a0202, Ric00), sp.Eq(-a0101 + 2 * a1212, Ric11), sp.Eq(-a0202 + a1212 + Kt2, Ric22), sp.Eq(2 * a0212, Ric01)],
               [a0101, a0202, a0212, a1212], dict=True)[0]
report["second_route_inversion"] = {str(k): str(sp.simplify(val)) for k, val in sol.items()}
report["checks"]["second route R1212 - (pg rho - Kt2)/2"] = str(sp.simplify(sol[a1212] - (pg * rho - Kt2) / 2))
report["checks"]["second route R0202 - (pg Sr + Kt2)/2"] = str(sp.simplify(sol[a0202] - (pg * Sr + Kt2) / 2))
report["checks"]["second route R0101 - (pg (rho - Sr + 2 St)/2 - Kt2)"] = str(sp.simplify(sol[a0101] - (pg * (rho - Sr + 2 * St) / 2 - Kt2)))
report["checks"]["second route R0212 - pg T01/2"] = str(sp.simplify(sol[a0212] - pg * T01 / 2))
# consistency of the four Ricci equations (the system is overdetermined only if Ric11 is used; check the 4x4 solve is exact)

# (2) radial null geodesics in Hamiltonian form: H = (1/2) g^{mu nu} k_mu k_nu; dx/dlam = dH/dk, dk/dlam = -dH/dx
kT, kr = sp.symbols("k_T k_r")
H = (ginv[0, 0] * kT**2 + 2 * ginv[0, 1] * kT * kr + ginv[1, 1] * kr**2) / 2
dT_dlam = sp.diff(H, kT); dr_dlam = sp.diff(H, kr)
dkT_dlam = -sp.diff(H, T); dkr_dlam = -sp.diff(H, r)
# null condition solved for k_T: (k_T + v k_r)^2 = N^2 k_r^2 -> k_T = -v k_r + s' N k_r
sgn = sp.symbols("s")   # s = +1 outgoing, -1 ingoing (to be identified)
eps = -(E[0, 0] * kT + E[0, 1] * kr)     # eps = -n^mu k_mu
ray_report = {}
for s_val in (1, -1):
    kT_null = -v * kr - s_val * N * kr      # choose the root so that dr/dT = v + s N (checked below)
    drdT = sp.simplify((dr_dlam / dT_dlam).subs(kT, kT_null))
    ray_report["s=%+d: dr/dT - (v + s N)" % s_val] = str(sp.simplify(drdT - (v + s_val * N)))
    eps_s = sp.simplify(eps.subs(kT, kT_null))
    ray_report["s=%+d: eps in terms of k_r" % s_val] = str(eps_s)
    # d eps/dT along the ray: eps = eps(T, r(T), k_T(T), k_r(T)); use the Hamiltonian flow for k and dT/dlam
    deps_dlam = (sp.diff(eps, T) * dT_dlam + sp.diff(eps, r) * dr_dlam + sp.diff(eps, kT) * dkT_dlam + sp.diff(eps, kr) * dkr_dlam)
    dlneps_dT = sp.simplify((deps_dlam / dT_dlam / eps).subs(kT, kT_null))
    ray_report["s=%+d: d ln eps/dT + (v' + s N')" % s_val] = str(sp.simplify(dlneps_dT + (sp.diff(v, r) + s_val * sp.diff(N, r))))
    dTdlam_over_eps = sp.simplify((dT_dlam / eps).subs(kT, kT_null))
    ray_report["s=%+d: dT/dlambda - eps/N" % s_val] = str(sp.simplify(dTdlam_over_eps - 1 / N))
report["rays"] = ray_report
report["all_component_and_norm_checks_pass"] = bool(ok_all)
print(json.dumps(report, indent=1))
out = sys.argv[1] if len(sys.argv) > 1 else "cv_riemann_sympy.json"
json.dump(report, open(out, "w"), indent=1)
print("wrote", out)
