#!/usr/bin/env python
"""C1: exact (sympy) stripped conversion amplitudes at rational on-shell points, ma = 3/5, a = 1.

Conventions = VERTEX_stress_v1 (1)-(4), VERTEX_force_event_v1 (1)-(5):
  alpha_i = tau_x x sigma_i, beta = tau_z x I2, h_A(p) = m_A beta + alpha . d(p), d_i = sin p_i,
  M^0 = tau_z x sigma_y, M^1 = tau_y x sigma_z, M^2 = i tau_y x I2, M^3 = -tau_y x sigma_x,
  eta = diag(1,-1,-1,-1),  A_{t1t2;s1s2} = 4 sum_mu eta_mu conj(J_L^mu)_{t1t2} (J_H^mu)_{s1s2},
  J_A^mu = u_{A1}^T M^mu u_{A2}  (positive-band spinors, u^dagger u = I2),
  filtered at total momentum P: A_F = kappa(P) S A,  S = sum_b prod_l f_b(p_l),
  f_b(p) = prod_j (1 + (-1)^{b_j} cos p_j)/2,  kappa(0) = 1.
Points (all positive-band, exact on shell):
  P1 (hypothesis A, total momentum 0): heavy +-p, p = (asin 3/5, 0, 0); light +-k, k = (asin 3/5, asin 3/5, 0);
     E_H^2 = 9/25 + 9/25 = 18/25 = E_L^2.
  P2 (hypothesis A, total momentum 0, eps = 1): heavy +-p, p = (asin 4/5, 0, 0); light +-k, k = (asin 3/5, asin 4/5, 0).
  P3 (hypothesis A', zero relative momentum, total momentum 2p): heavy p, p with p = (asin 3/5, 0, 0) (distinct spins);
     light p +- kappa, kappa = (0, asin 3/5, 0); E_L^2 = 9/25 + 9/25 = 18/25 = E_H^2 each.
  P0 (control, known zero): same-valley threshold, heavy at rest, light +-(asin 3/5, 0, 0) (VERTEX_force_event table row 1).
Outputs: for each point the exact tensor A (4x4 in (t1t2),(s1s2)), its squared Frobenius norm, S, and S^2 |A|^2.
"""
import sympy as sp

I2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]]); sy = sp.Matrix([[0, -sp.I], [sp.I, 0]]); sz = sp.Matrix([[1, 0], [0, -1]])
tx, ty, tz = sx, sy, sz
def kron(A, B):
    return sp.Matrix(sp.BlockMatrix([[A[i, j] * B for j in range(A.cols)] for i in range(A.rows)]).as_explicit())
alpha = [kron(tx, sx), kron(tx, sy), kron(tx, sz)]
beta = kron(tz, I2)
M = [kron(tz, sy), kron(ty, sz), sp.I * kron(ty, I2), -kron(ty, sx)]
eta = [1, -1, -1, -1]
m_H = sp.Rational(3, 5)

def h(p, m):
    return m * beta + sum((alpha[i] * sp.sin(p[i]) for i in range(3)), sp.zeros(4, 4))

def spinors(p, m):
    """positive-band spinors u_s = (E + h) e_s / sqrt(2E(E+m)), s = 0,1; returns 4x2 matrix and E."""
    H = h(p, m)
    E = sp.sqrt(m**2 + sum(sp.sin(p[i])**2 for i in range(3)))
    cols = []
    for s in range(2):
        e = sp.zeros(4, 1); e[s] = 1
        v = (E * sp.eye(4) + H) * e
        cols.append(sp.simplify(v / sp.sqrt(2 * E * (E + m))))
    u = sp.Matrix.hstack(*cols)
    # checks: eigenvector, orthonormal
    assert sp.simplify(H * u - E * u) == sp.zeros(4, 2), "not an eigenvector"
    assert sp.simplify(u.H * u - I2) == sp.zeros(2, 2), "not orthonormal"
    return u, E

def J(u1, u2):
    return [sp.simplify(u1.T * M[mu] * u2) for mu in range(4)]  # 2x2 in (s1,s2)

def amplitude(pH1, pH2, kL1, kL2):
    uH1, EH1 = spinors(pH1, m_H); uH2, EH2 = spinors(pH2, m_H)
    uL1, EL1 = spinors(kL1, 0); uL2, EL2 = spinors(kL2, 0)
    assert sp.simplify(EH1 + EH2 - EL1 - EL2) == 0, "off shell"
    JH = J(uH1, uH2); JL = J(uL1, uL2)
    A = sp.zeros(4, 4)  # rows (t1,t2), cols (s1,s2)
    for t1 in range(2):
        for t2 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    A[2 * t1 + t2, 2 * s1 + s2] = sp.simplify(4 * sum(eta[mu] * sp.conjugate(JL[mu][t1, t2]) * JH[mu][s1, s2] for mu in range(4)))
    return A, (EH1, EH2, EL1, EL2)

def fb(p, b):
    return sp.prod([(1 + (-1)**b[j] * sp.cos(p[j])) / 2 for j in range(3)])

def S_of(legs):
    tot = 0
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                tot += sp.prod([fb(p, (bx, by, bz)) for p in legs])
    return sp.nsimplify(sp.simplify(tot))

a35 = sp.asin(sp.Rational(3, 5)); a45 = sp.asin(sp.Rational(4, 5))
Z = sp.Integer(0)
points = {
    "P0_threshold_control": ((Z, Z, Z), (Z, Z, Z), (a35, Z, Z), (-a35, Z, Z)),
    "P1_P0_shell_eps_3sqrt2_over_5": ((a35, Z, Z), (-a35, Z, Z), (a35, a35, Z), (-a35, -a35, Z)),
    "P2_P0_shell_eps_1": ((a45, Z, Z), (-a45, Z, Z), (a35, a45, Z), (-a35, -a45, Z)),
    "P3_boosted_zero_relative": ((a35, Z, Z), (a35, Z, Z), (a35, a35, Z), (a35, -a35, Z)),
}
for name, (pH1, pH2, kL1, kL2) in points.items():
    A, Es = amplitude(pH1, pH2, kL1, kL2)
    frob2 = sp.nsimplify(sp.simplify(sum(sp.Abs(A[i, j])**2 for i in range(4) for j in range(4))))
    S = S_of([pH1, pH2, kL1, kL2])
    print("=== %s ===" % name)
    print("energies (EH1, EH2, EL1, EL2) =", [sp.nsimplify(sp.simplify(E)) for E in Es])
    print("A (rows t1t2 = 00,01,10,11; cols s1s2 = 00,01,10,11):")
    sp.pprint(A.applyfunc(lambda z: sp.nsimplify(sp.simplify(z))))
    print("|A|_F^2 =", frob2, " (float %.6f)" % float(frob2))
    print("S (filtered leg-filter sum) =", S, " (float %.6f)" % float(S))
    print("S^2 |A|_F^2 =", sp.nsimplify(S**2 * frob2), " (float %.6f)" % float(S**2 * frob2))
    print()
print("hypothesis (A) on-site: nonzero at P1 and P2:", end=" ")
print(all(sp.simplify(sum(sp.Abs(amplitude(*points[k])[0][i, j])**2 for i in range(4) for j in range(4))) != 0 for k in ("P1_P0_shell_eps_3sqrt2_over_5", "P2_P0_shell_eps_1")))
