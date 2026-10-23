#!/usr/bin/env python
"""C1c: hypothesis (A') for Theorem 2 with ALL energies inside I = (3/5, 1): exact (sympy) amplitude tensors and the
EXACT rank of the energy-map Jacobian on the shell. Same conventions as c1_amplitude_exact_v1.py (a = 1, ma = 3/5).
th = asin 3/5 (cos 4/5), ph = asin 4/5 (cos 3/5), ps = asin 5/13 (cos 12/13).
 G2: heavy (th,0,0),(0,0,th) -> light (th,th,0),(0,-th,th); all four energies 3 sqrt2/5 = 0.8485; total momentum (th,0,th).
 G4: heavy (ps,0,0),(0,0,th) -> light (ps,th,0),(0,-th,th); energies sqrt(2146)/65 = 0.7127 and 3 sqrt2/5 each side,
     total momentum (ps,0,th); unequal energies, all in I.
 G3 (boundary, informational): heavy (th,0,0),(0,ph,0) -> light (0,ph,th),(th,0,-th); energies 0.8485 and 1.
For each point: exact tensor A, |A|_F^2, S (filtered leg sum, kappa(P) not included), and the exact rank of the
4 x 9 Jacobian of F(p1,p2,k1) = (E_H(p1), E_H(p2), E_L(k1), E_H(p1)+E_H(p2)-E_L(k1)-E_L(p1+p2-k1)) at the point.
Rank 4 means the three energies (eps1, eps2, eps3) vary over an open set of R^3 along the shell near the point.
"""
import sympy as sp, itertools

I2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]]); sy = sp.Matrix([[0, -sp.I], [sp.I, 0]]); sz = sp.Matrix([[1, 0], [0, -1]])
def kron(A, B):
    return sp.Matrix(sp.BlockMatrix([[A[i, j] * B for j in range(A.cols)] for i in range(A.rows)]).as_explicit())
alpha = [kron(sx, sx), kron(sx, sy), kron(sx, sz)]; beta = kron(sz, I2)
M = [kron(sz, sy), kron(sy, sz), sp.I * kron(sy, I2), -kron(sy, sx)]; eta = [1, -1, -1, -1]
m_H = sp.Rational(3, 5)
def h(p, m): return m * beta + sum((alpha[i] * sp.sin(p[i]) for i in range(3)), sp.zeros(4, 4))
def spinors(p, m):
    H = h(p, m); E = sp.sqrt(m**2 + sum(sp.sin(p[i])**2 for i in range(3))); cols = []
    for s in range(2):
        e = sp.zeros(4, 1); e[s] = 1; cols.append((E * sp.eye(4) + H) * e / sp.sqrt(2 * E * (E + m)))
    u = sp.Matrix.hstack(*cols)
    assert sp.simplify(H * u - E * u) == sp.zeros(4, 2) and sp.simplify(u.H * u - I2) == sp.zeros(2, 2)
    return u, E
def amplitude(pH1, pH2, kL1, kL2):
    uH1, EH1 = spinors(pH1, m_H); uH2, EH2 = spinors(pH2, m_H); uL1, EL1 = spinors(kL1, 0); uL2, EL2 = spinors(kL2, 0)
    assert sp.simplify(EH1 + EH2 - EL1 - EL2) == 0, "off shell"
    JH = [uH1.T * M[mu] * uH2 for mu in range(4)]; JL = [uL1.T * M[mu] * uL2 for mu in range(4)]
    A = sp.zeros(4, 4)
    for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
        A[2*t1+t2, 2*s1+s2] = sp.simplify(4 * sum(eta[mu] * sp.conjugate(JL[mu][t1, t2]) * JH[mu][s1, s2] for mu in range(4)))
    return A, (EH1, EH2, EL1, EL2)
def fb(p, b): return sp.prod([(1 + (-1)**b[j] * sp.cos(p[j])) / 2 for j in range(3)])
def S_of(legs):
    return sp.nsimplify(sp.simplify(sum(sp.prod([fb(p, b) for p in legs]) for b in itertools.product((0, 1), repeat=3))))
def exact_jacobian_rank(pH1, pH2, kL1):
    x = sp.symbols('x0:9', real=True)
    p1 = x[0:3]; p2 = x[3:6]; k1 = x[6:9]; k2 = [p1[j] + p2[j] - k1[j] for j in range(3)]
    EH = lambda v: sp.sqrt(m_H**2 + sum(sp.sin(v[j])**2 for j in range(3)))
    EL = lambda v: sp.sqrt(sum(sp.sin(v[j])**2 for j in range(3)))
    F = sp.Matrix([EH(p1), EH(p2), EL(k1), EH(p1) + EH(p2) - EL(k1) - EL(k2)])
    Jm = F.jacobian(sp.Matrix(x))
    subs = {x[j]: v for j, v in enumerate(list(pH1) + list(pH2) + list(kL1))}
    Jv = Jm.subs(subs).applyfunc(lambda z: sp.nsimplify(sp.simplify(z)))
    return Jv.rank(simplify=True), Jv

th = sp.asin(sp.Rational(3, 5)); ph = sp.asin(sp.Rational(4, 5)); ps = sp.asin(sp.Rational(5, 13)); Z = sp.Integer(0)
points = {
    "G2_all_energies_3sqrt2_over_5": ((th, Z, Z), (Z, Z, th), (th, th, Z), (Z, -th, th)),
    "G4_unequal_energies_in_I": ((ps, Z, Z), (Z, Z, th), (ps, th, Z), (Z, -th, th)),
    "G3_boundary_energy_1": ((th, Z, Z), (Z, ph, Z), (Z, ph, th), (th, Z, -th)),
}
for name, (pH1, pH2, kL1, kL2) in points.items():
    A, Es = amplitude(pH1, pH2, kL1, kL2)
    frob2 = sp.nsimplify(sp.simplify(sum(sp.Abs(A[i, j])**2 for i in range(4) for j in range(4))))
    S = S_of([pH1, pH2, kL1, kL2]); rank, Jv = exact_jacobian_rank(pH1, pH2, kL1)
    print("=== %s ===" % name)
    print("energies (EH1,EH2,EL1,EL2) =", [sp.nsimplify(sp.simplify(E)) for E in Es], " floats", [round(float(E), 6) for E in Es])
    print("momentum conservation:", [sp.simplify(pH1[j] + pH2[j] - kL1[j] - kL2[j]) for j in range(3)])
    sp.pprint(A.applyfunc(lambda z: sp.nsimplify(sp.simplify(z))))
    print("|A|_F^2 exact =", frob2, " float %.6f" % float(frob2), ";  S =", S, " float %.6f" % float(S))
    print("exact rank of the 4x9 energy/shell Jacobian =", rank)
    print()
