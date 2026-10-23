#!/usr/bin/env python
"""Independent recomputation of the NONCONS_universal_v1 check C1c (points G2, G4).

Written from the definitions only (VERTEX_stress_v1 eq. (1) for the on-site converter,
NONCONS_universal_v1 sec. 1 for the model, (A') for the shell). Shares no code with
busi/code/noncons/. Exact sympy arithmetic throughout; a = 1, m = 3/5.

Three routes to |A|_F^2:
  R1  explicit positive-band spinors u_s = (E+h) e_s / sqrt(2E(E+m)) and the pair formula
      A[t1,t2;s1,s2] = 4 sum_mu eta_mu conj(v1^T M^mu v2)[t1t2] (u1^T M^mu u2)[s1s2]
  R2  basis-free projector trace: |A|_F^2 = 16 sum_{mu,nu} eta_mu eta_nu
          conj(T_L[mu,nu]) T_H[mu,nu],  T[mu,nu] = tr(M^mu^T P1 conj(M^nu) P2^T),
      with P = (1 + h/E)/2 (only 1/E radicals; no spinor choice at all)
  R3  explicit single-site Fock space (8 fermionic modes: 4 heavy, 4 light components),
      W_0 = sum_mu eta_mu Phi_L^mu^dag Phi_H^mu + h.c. with Phi^mu = sum_ab M_ab c_a c_b,
      matrix element <0| b_L(v2) b_L(v1) W_0 b_H(u1)^dag b_H(u2)^dag |0>; this checks the
      factor 4 and the antisymmetrisation of the pair formula (V_s <f|V|i> = <f_0|W_0|i_0>).
Jacobian: variables (p1, p2, k1) in R^9, k2 = p1 + p2 - k1; 4x9 Jacobian of
(E_H(p1), E_H(p2), E_L(k1), E_L(k2)); row i multiplied by E_i > 0 (rank-preserving) makes
every entry rational (sin x cos), so the rank is exact over Q. Also: rank of the 3-energy
map restricted to the shell tangent space (ker dG, G = E_L1 + E_L2 - E_H1 - E_H2).
"""
import itertools
import sys

import sympy as sp

I = sp.I
m = sp.Rational(3, 5)

# ---------- Dirac matrices (component order: upper up, upper down, lower up, lower down)
s0 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])


def kron(A, B):
    return sp.Matrix(sp.kronecker_product(A, B))


alpha = [kron(sx, sx), kron(sx, sy), kron(sx, sz)]
beta = kron(sz, s0)
M = [kron(sz, sy), kron(sy, sz), I * kron(sy, s0), -kron(sy, sx)]
eta = [1, -1, -1, -1]
for Mu in M:
    assert sp.simplify(Mu + Mu.T) == sp.zeros(4, 4)  # all antisymmetric


# ---------- exact trig data: a momentum component is given as (sin, cos) rational pair
class Ang:
    def __init__(self, s, c):
        self.s, self.c = sp.sympify(s), sp.sympify(c)
        assert sp.simplify(self.s**2 + self.c**2 - 1) == 0

    def __neg__(self):
        return Ang(-self.s, self.c)

    def __add__(self, o):
        return Ang(self.s * o.c + self.c * o.s, self.c * o.c - self.s * o.s)

    def __sub__(self, o):
        return self + (-o)


Z = Ang(0, 1)


def A_(s):  # arcsin(s) with |arcsin| < pi/2
    s = sp.Rational(s)
    return Ang(s, sp.sqrt(1 - s**2))


def vadd(p, q):
    return [x + y for x, y in zip(p, q)]


def vsub(p, q):
    return [x - y for x, y in zip(p, q)]


def h0(mass, p):
    return mass * beta + sum((alpha[i] * p[i].s for i in range(3)), sp.zeros(4, 4))


def energy(mass, p):
    return sp.sqrt(mass**2 + sum(x.s**2 for x in p))


def spinors(mass, p):
    E = energy(mass, p)
    h = h0(mass, p)
    us = []
    for s in range(2):
        e = sp.zeros(4, 1)
        e[s] = 1
        u = (E * sp.eye(4) + h) * e / sp.sqrt(2 * E * (E + mass))
        us.append(sp.simplify(u))
    # exact checks: eigenvector with +E, orthonormal
    for u in us:
        assert sp.simplify(h * u - E * u) == sp.zeros(4, 1)
    G = sp.Matrix(2, 2, lambda i, j: sp.simplify((us[i].H * us[j])[0]))
    assert G == sp.eye(2), G
    return us


def projector(mass, p):
    E = energy(mass, p)
    return (sp.eye(4) + h0(mass, p) / E) / 2


def amp_R1(p1, p2, k1, k2):
    u1, u2 = spinors(m, p1), spinors(m, p2)
    v1, v2 = spinors(0, k1), spinors(0, k2)
    A = sp.zeros(4, 4)  # rows (t1,t2), cols (s1,s2)
    for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
        val = 0
        for mu in range(4):
            JL = (v1[t1].T * M[mu] * v2[t2])[0]
            JH = (u1[s1].T * M[mu] * u2[s2])[0]
            val += eta[mu] * sp.conjugate(JL) * JH
        A[2 * t1 + t2, 2 * s1 + s2] = sp.simplify(4 * val)
    F2 = sum(sp.expand(x * sp.conjugate(x)) for x in A)
    return A, sp.radsimp(sp.simplify(F2))


def amp_R2(p1, p2, k1, k2):
    P1, P2 = projector(m, p1), projector(m, p2)
    Q1, Q2 = projector(0, k1), projector(0, k2)
    tot = 0
    for mu in range(4):
        for nu in range(4):
            TH = (M[mu].T * P1 * M[nu].conjugate() * P2.T).trace()
            TL = (M[mu].T * Q1 * M[nu].conjugate() * Q2.T).trace()
            tot += eta[mu] * eta[nu] * sp.conjugate(TL) * TH
    return sp.radsimp(sp.simplify(16 * tot))


# ---------- R3: explicit fermionic Fock space, exact (dict of sorted mode tuples)
# modes 0..3 heavy components, 4..7 light components
def apply_c(state, j):
    out = {}
    for occ, amp in state.items():
        if j in occ:
            k = occ.index(j)
            new = occ[:k] + occ[k + 1:]
            out[new] = out.get(new, 0) + (-1)**k * amp
    return out


def apply_cd(state, j):
    out = {}
    for occ, amp in state.items():
        if j not in occ:
            k = sum(1 for x in occ if x < j)
            new = tuple(sorted(occ + (j,)))
            out[new] = out.get(new, 0) + (-1)**k * amp
    return out


def add(a, b, c=1):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + c * v
    return out


def W0(state):
    out = {}
    for mu in range(4):
        # Phi_H^mu = sum_ab M_ab c_a c_b  (act c_b first)
        phiH = {}
        for a_ in range(4):
            for b_ in range(4):
                if M[mu][a_, b_] != 0:
                    phiH = add(phiH, apply_c(apply_c(state, b_), a_), M[mu][a_, b_])
        # Phi_L^mu^dag = sum_ab conj(M_ab) c_b^dag c_a^dag (act c_a^dag first)
        res = {}
        for a_ in range(4):
            for b_ in range(4):
                if M[mu][a_, b_] != 0:
                    res = add(res, apply_cd(apply_cd(phiH, 4 + a_), 4 + b_),
                              sp.conjugate(M[mu][a_, b_]))
        out = add(out, res, eta[mu])
    return out  # the h.c. term annihilates heavy->light matrix elements


def bdag(state, phi, off):
    out = {}
    for a_ in range(4):
        if phi[a_] != 0:
            out = add(out, apply_cd(state, off + a_), phi[a_])
    return out


def amp_R3(p1, p2, k1, k2):
    u1, u2 = spinors(m, p1), spinors(m, p2)
    v1, v2 = spinors(0, k1), spinors(0, k2)
    vac = {(): 1}
    A = sp.zeros(4, 4)
    for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
        ini = bdag(bdag(vac, u2[s2], 0), u1[s1], 0)  # b1^dag b2^dag |0>
        fin = bdag(bdag(vac, v2[t2], 4), v1[t1], 4)
        Wi = W0(ini)
        val = sum(sp.conjugate(c) * Wi.get(k, 0) for k, c in fin.items())
        A[2 * t1 + t2, 2 * s1 + s2] = sp.simplify(val)
    return A


def jacobian(p1, p2, k1, k2):
    # d/dp_j E(p) = sin(p_j) cos(p_j) / E(p); scaled row = E * grad (rational)
    def g(p):
        return [x.s * x.c for x in p]
    zero = [0, 0, 0]
    # variables (p1, p2, k1); k2 = p1 + p2 - k1
    gk2 = g(k2)
    rows = [
        g(p1) + zero + zero,
        zero + g(p2) + zero,
        zero + zero + g(k1),
        gk2 + gk2 + [-x for x in gk2],
    ]
    J = sp.Matrix(rows)
    return J


def report(name, p1, p2, k1, k2, out):
    w = lambda *a: print(*a, file=out, flush=True)
    w(f"===== {name}")
    for lab, p in (("p1", p1), ("p2", p2), ("k1", k1), ("k2", k2)):
        w(f"  {lab}: sin = {[x.s for x in p]}, cos = {[x.c for x in p]}")
    cons = vsub(vadd(p1, p2), vadd(k1, k2))
    assert all(x.s == 0 and x.c == 1 for x in cons)
    w("  momentum conservation p1+p2 = k1+k2: exact (sin 0, cos 1 componentwise)")
    E = [energy(m, p1), energy(m, p2), energy(0, k1), energy(0, k2)]
    E = [sp.radsimp(sp.simplify(e)) for e in E]
    w(f"  energies E_H(p1), E_H(p2), E_L(k1), E_L(k2) = {E}  ~ {[sp.N(e, 12) for e in E]}")
    w(f"  energy balance E_H1+E_H2-E_L1-E_L2 = {sp.simplify(E[0] + E[1] - E[2] - E[3])}")
    w(f"  all in I = (m, 1/a) = (3/5, 1): {all(sp.Rational(3, 5) < e < 1 for e in E)}")
    w(f"  valley-0 cube |p_j| < pi/2 (all cos > 0): "
      f"{all(x.c > 0 for p in (p1, p2, k1, k2) for x in p)}")
    A1, F1 = amp_R1(p1, p2, k1, k2)
    w(f"  R1 A (rows t1t2, cols s1s2) = {A1.tolist()}")
    w(f"  R1 |A|_F^2 = {F1}  ~ {sp.N(F1, 20)}")
    F2 = amp_R2(p1, p2, k1, k2)
    w(f"  R2 |A|_F^2 (projector trace) = {F2}  ~ {sp.N(F2, 20)}")
    A3 = amp_R3(p1, p2, k1, k2)
    d13 = sp.simplify(A3 - A1)
    w(f"  R3 Fock-space A minus R1 A = zero matrix: {d13 == sp.zeros(4, 4)}")
    w(f"  R1 == R2 exactly: {sp.simplify(F1 - F2) == 0}")
    if sp.simplify(F1 - F2) == 0 and not F1.is_rational:
        w(f"  minimal polynomial of |A|_F^2: {sp.minimal_polynomial(F1, sp.Symbol('x'))}")
    J = jacobian(p1, p2, k1, k2)
    w(f"  scaled Jacobian (rows E_i * dE_i; cols p1,p2,k1) = {J.tolist()}")
    r = J.rank()
    w(f"  exact rank (over Q) of 4-energy Jacobian on momentum-conserving manifold = {r}")
    # nonzero 4x4 minors as an explicit certificate
    for cols in itertools.combinations(range(9), 4):
        d = J[:, list(cols)].det()
        if d != 0:
            w(f"  certificate: minor on columns {cols} = {d}")
            break
    # rank of 3-energy map on shell tangent space ker dG
    Jtrue = sp.Matrix([[J[i, j] / E[i] for j in range(9)] for i in range(4)])
    dG = Jtrue[2, :] + Jtrue[3, :] - Jtrue[0, :] - Jtrue[1, :]
    ker = dG.nullspace()
    T = sp.Matrix.hstack(*ker)
    J3 = sp.simplify(Jtrue[:3, :] * T)
    r3 = sp.Matrix(J3).rank(simplify=True)
    w(f"  dG nonzero: {any(sp.simplify(x) != 0 for x in dG)}; "
      f"rank of (E_H1, E_H2, E_L1) on shell tangent space (dim {len(ker)}) = {r3}")
    # alternative reading: no conservation (12 free variables) - trivially 4, reported
    return F1, r, E


def main():
    out = sys.stdout
    t35, t513 = A_("3/5"), A_("5/13")
    G2 = ([t35, Z, Z], [Z, Z, t35], [t35, t35, Z], [Z, -t35, t35])
    G4 = ([t513, Z, Z], [Z, Z, t35], [t513, t35, Z], [Z, -t35, t35])
    F2, r2, E2 = report("G2", *G2, out)
    F4, r4, E4 = report("G4", *G4, out)
    # control: threshold point (p = 0 heavy) should give zero tensor per VERTEX_force_event row 1
    print("===== verdict", file=out)
    c1 = (F2 == 16)
    c2 = (r2 == 4 and r4 == 4)
    print(f"  |A|_F^2(G2) == 16 exactly: {c1}  ({F2})", file=out)
    print(f"  rank 4 at G2 and G4: {c2}  ({r2}, {r4})", file=out)
    print(f"  |A|_F^2(G4) = {F4} ~ {sp.N(F4, 15)}", file=out)
    print(f"  OVERALL: {'PASS' if (c1 and c2) else 'FAIL'}", file=out)


if __name__ == "__main__":
    main()
