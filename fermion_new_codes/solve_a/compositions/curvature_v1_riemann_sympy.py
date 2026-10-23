"""curvature_v1_riemann_sympy.py (solve lane, CURVATURE_readout_v1): the slice-frame orthonormal Riemann tensor of
    ds^2 = -N^2 dt^2 + (dr - v dt)^2 + r^2 dOmega^2      (the engine's dictionary; reviewer B singularity_v1 sec. (2))
with N(t, r), v(t, r) generic. Frame: e0 = n = (d_t + v d_r)/N (slice normal), e1 = d_r, e2 = d_theta/r, e3 = d_phi/(r sin theta).
Prints the independent orthonormal components in terms of N, v and their derivatives, the frame norm
sqrt(sum_{abcd} R_abcd^2) (reviewer B's script R), the Kretschmann scalar, and checks:
  (1) R_{2323} = (v/(rN))^2 = 2GM/r^3 (reviewer A sec. (1)(ii)); R_{1212} = K_r K_t with K_r = v'/N, K_t = v/(rN);
  (2) along a radial null geodesic with dr/dt = v + s N, the slice-frame energy eps = -n.k obeys d ln eps/dt = -(v' + s N')
      (reviewer A sec. (2)(b)) and dt/dlambda = eps/N.
Writes the component expressions (srepr) to riemann_components_v1.json beside itself (or argv[1]) for the data script.
Light symbolic work only."""
import json
import sys

import sympy as sp

t, r, th, ph = sp.symbols("t r theta phi", real=True)
X = [t, r, th, ph]
N = sp.Function("N")(t, r)
v = sp.Function("v")(t, r)
g = sp.zeros(4, 4)
g[0, 0] = -N**2 + v**2
g[0, 1] = g[1, 0] = -v
g[1, 1] = 1
g[2, 2] = r**2
g[3, 3] = r**2 * sp.sin(th)**2
gi = sp.simplify(g.inv())
Gam = [[[sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d])) for d in range(4)) / 2)
         for c in range(4)] for b in range(4)] for a in range(4)]


def Rup(a, b, c, d):  # R^a_{bcd}
    e = sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
    e += sum(Gam[a][c][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][c] for k in range(4))
    return e


Rdn = {}
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                if a < b and c < d and (a, b) <= (c, d):
                    Rdn[(a, b, c, d)] = sp.simplify(sum(g[a, k] * Rup(k, b, c, d) for k in range(4)))


def Rfull(a, b, c, d):
    s = 1
    if a == b or c == d:
        return 0
    if a > b:
        a, b, s = b, a, -s
    if c > d:
        c, d, s = d, c, -s
    if (a, b) > (c, d):
        a, b, c, d = c, d, a, b
    return s * Rdn[(a, b, c, d)]


E = sp.zeros(4, 4)  # E[A, mu] = e_A^mu
E[0, 0] = 1 / N
E[0, 1] = v / N
E[1, 1] = 1
E[2, 2] = 1 / r
E[3, 3] = 1 / (r * sp.sin(th))
eta = sp.diag(-1, 1, 1, 1)
chk = sp.simplify(E * g * E.T - eta)
assert chk == sp.zeros(4, 4), chk

# orthonormal components needed (spherical symmetry): 0101, 0202, 0212, 1212, 2323 (0303 = 0202, 0313 = 0212, 1313 = 1212)
def Rhat(A, B, C, D):
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
                    tot += E[A, a] * E[B, b] * E[C, c] * E[D, d] * Rfull(a, b, c, d)
    return sp.simplify(tot)


names = {"0101": (0, 1, 0, 1), "0202": (0, 2, 0, 2), "0212": (0, 2, 1, 2), "1212": (1, 2, 1, 2), "2323": (2, 3, 2, 3),
         "0303": (0, 3, 0, 3), "0313": (0, 3, 1, 3), "1313": (1, 3, 1, 3), "0123": (0, 1, 2, 3), "0213": (0, 2, 1, 3)}
comp = {k: Rhat(*ix) for k, ix in names.items()}
for k in comp:
    print("R_hat%s =" % k, comp[k])
assert sp.simplify(comp["0303"] - comp["0202"]) == 0 and sp.simplify(comp["0313"] - comp["0212"]) == 0
assert sp.simplify(comp["1313"] - comp["1212"]) == 0 and comp["0123"] == 0 and comp["0213"] == 0

Nr, vr = sp.diff(N, r), sp.diff(v, r)
Kr, Kt = vr / N, v / (r * N)
print("check R2323 - K_t^2:", sp.simplify(comp["2323"] - Kt**2))
print("check R1212 - K_r K_t:", sp.simplify(comp["1212"] - Kr * Kt))

# frame norm: all index orderings. Each independent pair-component appears with multiplicity:
# R_0101 type (pairs (01),(01)): 4 orderings; R_0212 type (pairs (02),(12)): 8 orderings.
R0101, R0202, R0212, R1212, R2323 = (comp[k] for k in ("0101", "0202", "0212", "1212", "2323"))
norm2 = 4 * (R0101**2 + 2 * R0202**2 + 2 * R1212**2 + R2323**2) + 8 * (2 * R0212**2)
# Kretschmann: R_abcd R^abcd with eta: sign (-1)^{number of 0 indices}
kret = 4 * (R0101**2 + 2 * R0202**2 + 2 * R1212**2 + R2323**2) - 8 * (2 * R0212**2)
print("frame norm^2 =", norm2)

# brute-force check of the frame norm multiplicity on the orthonormal tensor
def Rh(A, B, C, D):
    key = None
    s = 1
    if A == B or C == D:
        return 0
    if A > B:
        A, B, s = B, A, -s
    if C > D:
        C, D, s = D, C, -s
    if (A, B) > (C, D):
        A, B, C, D = C, D, A, B
    tag = "%d%d%d%d" % (A, B, C, D)
    return s * comp[tag] if tag in comp else None


bf = 0
bfK = 0
for A in range(4):
    for B in range(4):
        for C in range(4):
            for D in range(4):
                x = Rh(A, B, C, D)
                if x is None:
                    x = Rhat(A, B, C, D)
                bf += x**2
                bfK += x**2 * (-1) ** ([A, B, C, D].count(0))
assert sp.simplify(bf - norm2) == 0, "frame norm multiplicity"
assert sp.simplify(bfK - kret) == 0, "Kretschmann multiplicity"
print("frame-norm and Kretschmann multiplicities verified by brute force")

# Einstein tensor in the frame, for the constraint-row comparison (reviewer A (2)(a) second route, constraint components only)
Ric = {}
for (A, B) in [(0, 0), (0, 1), (1, 1), (2, 2)]:
    Ric[(A, B)] = sp.simplify(sum(eta[C, C] * Rhat(C, A, C, B) for C in range(4)))
Rs = sp.simplify(sum(eta[A, A] * Ric[(A, A)] for A in range(3)) + eta[3, 3] * Ric[(2, 2)])
G00 = sp.simplify(Ric[(0, 0)] + Rs / 2)  # G_ab = R_ab - (1/2) eta_ab R, eta_00 = -1
G01 = sp.simplify(Ric[(0, 1)])
print("G_nn =", G00)
print("check G_nn - (K_t^2 + 2 K_r K_t):", sp.simplify(G00 - (Kt**2 + 2 * Kr * Kt)))
print("G_nr =", G01)

# ray check: k^t = eps/N, k^r = (v + s N) k^t; eps = -n_mu k^mu with n_mu = (-N, 0, 0, 0)
s_ = sp.symbols("s")
kt = sp.Function("kt")(t)
# geodesic eq for k^t along the ray, with r = r(t) on the ray: dk^t/dlambda = -Gam^t_{ab} k^a k^b, and d/dlambda = k^t d/dt
rt = sp.Function("rr")(t)
sub = {r: rt}
kvec = [1, v + s_ * N, 0, 0]  # k^mu / k^t
acc_t = -sum(Gam[0][a][b] * kvec[a] * kvec[b] for a in range(4) for b in range(4))  # (dk^t/dlambda)/(k^t)^2
# d ln k^t / dt = acc_t (since dk^t/dlambda = k^t dk^t/dt)
# eps = N k^t  =>  d ln eps/dt = d ln N/dt|_ray + d ln k^t/dt, with d/dt|_ray = d_t + (v + sN) d_r
dlnN = (sp.diff(N, t) + (v + s_ * N) * sp.diff(N, r)) / N
dlneps = sp.simplify(dlnN + acc_t)
for sv in (1, -1):
    print("s = %+d: d ln eps/dt =" % sv, sp.simplify(dlneps.subs(s_, sv)), "; minus reviewer A's -(v' + s N'):",
          sp.simplify(dlneps.subs(s_, sv) + (sp.diff(v, r) + sv * sp.diff(N, r))))
# also: the ray's second geodesic component must be consistent with null + dr/dt (radial null curves are geodesics in 1+1)

# export expressions in plain symbols
Nt_, Nr_, Nrr_, Ntr_, vt_, vr_, vrr_, vtr_, N_, v_, r_ = sp.symbols("N_t N_r N_rr N_tr v_t v_r v_rr v_tr N v r")
rep = [(sp.Derivative(N, t, r), Ntr_), (sp.Derivative(N, r, t), Ntr_), (sp.Derivative(v, t, r), vtr_), (sp.Derivative(v, r, t), vtr_),
       (sp.Derivative(N, (r, 2)), Nrr_), (sp.Derivative(v, (r, 2)), vrr_), (sp.Derivative(N, t), Nt_), (sp.Derivative(N, r), Nr_),
       (sp.Derivative(v, t), vt_), (sp.Derivative(v, r), vr_)]
out = {}
for k in ("0101", "0202", "0212", "1212", "2323"):
    e = comp[k]
    for a_, b_ in rep:
        e = e.subs(a_, b_)
    e = e.subs({N: N_, v: v_}).subs(r, r_)
    out["R" + k] = sp.srepr(sp.simplify(e))
    print("export R%s =" % k, sp.simplify(e))
for nm, e in (("G_nn", G00), ("G_nr", G01)):
    for a_, b_ in rep:
        e = e.subs(a_, b_)
    e = e.subs({N: N_, v: v_}).subs(r, r_)
    out[nm] = sp.srepr(sp.simplify(e))
    print("export %s =" % nm, sp.simplify(e))
path = sys.argv[1] if len(sys.argv) > 1 else "riemann_components_v1.json"
json.dump(out, open(path, "w"), indent=1)
print("wrote", path)
