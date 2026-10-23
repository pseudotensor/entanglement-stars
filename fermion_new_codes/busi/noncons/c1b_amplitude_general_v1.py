#!/usr/bin/env python
"""C1b: hypothesis (A') for Theorem 2: a general 2->2 on-shell point with UNEQUAL light energies, exact (sympy),
plus (informational) a numeric scan of the zero-relative-momentum family (p,p)->(p+kappa,p-kappa), where C1's
point P3 gave an exactly zero tensor, and the rank of the energy map on the shell at the general point.
Same conventions as c1_amplitude_exact_v1.py (a = 1, ma = 3/5).

General point G1 (exact): heavy p1 = (asin 3/5, 0, 0), p2 = (0, asin 3/5, 0), E_H = 3 sqrt2/5 each;
  light k1 = (asin 3/5, asin 3/5, asin t), k2 = (0, 0, -asin t), t = 9 sqrt2/20;
  E_L(k1) = sqrt(18/25 + 81/200) = 3 sqrt2/4, E_L(k2) = 9 sqrt2/20, sum = 6 sqrt2/5 = E_H(p1) + E_H(p2).
  Total momentum (asin 3/5, asin 3/5, 0) both sides. All |k_j| < pi/2 (valley 0).
"""
import sympy as sp, numpy as np, itertools

I2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]]); sy = sp.Matrix([[0, -sp.I], [sp.I, 0]]); sz = sp.Matrix([[1, 0], [0, -1]])
def kron(A, B):
    return sp.Matrix(sp.BlockMatrix([[A[i, j] * B for j in range(A.cols)] for i in range(A.rows)]).as_explicit())
alpha = [kron(sx, sx), kron(sx, sy), kron(sx, sz)]
beta = kron(sz, I2)
M = [kron(sz, sy), kron(sy, sz), sp.I * kron(sy, I2), -kron(sy, sx)]
eta = [1, -1, -1, -1]
m_H = sp.Rational(3, 5)

def h(p, m):
    return m * beta + sum((alpha[i] * sp.sin(p[i]) for i in range(3)), sp.zeros(4, 4))
def spinors(p, m):
    H = h(p, m); E = sp.sqrt(m**2 + sum(sp.sin(p[i])**2 for i in range(3)))
    cols = []
    for s in range(2):
        e = sp.zeros(4, 1); e[s] = 1
        cols.append((E * sp.eye(4) + H) * e / sp.sqrt(2 * E * (E + m)))
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

a35 = sp.asin(sp.Rational(3, 5)); Z = sp.Integer(0); t = 9 * sp.sqrt(2) / 20
pH1 = (a35, Z, Z); pH2 = (Z, a35, Z); kL1 = (a35, a35, sp.asin(t)); kL2 = (Z, Z, -sp.asin(t))
A, Es = amplitude(pH1, pH2, kL1, kL2)
print("=== G1 general 2->2 point ===")
print("energies (EH1,EH2,EL1,EL2) =", [sp.nsimplify(sp.simplify(E)) for E in Es])
print("total momentum check:", [sp.simplify(pH1[j] + pH2[j] - kL1[j] - kL2[j]) for j in range(3)])
An = A.applyfunc(lambda z: sp.nsimplify(sp.simplify(z)))
sp.pprint(An)
frob2 = sp.nsimplify(sp.simplify(sum(sp.Abs(A[i, j])**2 for i in range(4) for j in range(4))))
print("|A|_F^2 exact =", frob2, " float = %.10f" % float(frob2))
print("hypothesis (A') on-site at G1: nonzero =", frob2 != 0)

# --- numeric part: zero-relative-momentum family and Jacobian rank ---
def spin_np(p, m):
    s = np.sin(p); H = m*np.array(beta.tolist(), dtype=complex) + sum(np.array(alpha[i].tolist(), dtype=complex)*s[i] for i in range(3))
    E = np.sqrt(m*m + (s*s).sum()); U = np.zeros((4, 2), complex)
    for k in range(2):
        e = np.zeros(4); e[k] = 1; v = (E*np.eye(4) + H) @ e; U[:, k] = v/np.sqrt(2*E*(E+m))
    return U, E
Mn = [np.array(Mi.tolist(), dtype=complex) for Mi in M]
def amp_np(p1, p2, k1, k2, m=0.6):
    u1, E1 = spin_np(p1, m); u2, E2 = spin_np(p2, m); v1, F1 = spin_np(k1, 0.0); v2, F2 = spin_np(k2, 0.0)
    JH = [u1.T @ Mn[mu] @ u2 for mu in range(4)]; JL = [v1.T @ Mn[mu] @ v2 for mu in range(4)]
    A = np.zeros((4, 4), complex)
    for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
        A[2*t1+t2, 2*s1+s2] = 4*sum(eta[mu]*np.conj(JL[mu][t1, t2])*JH[mu][s1, s2] for mu in range(4))
    return A, E1+E2-F1-F2
def EL(k): return np.sqrt((np.sin(k)**2).sum())
def EH(p, m=0.6): return np.sqrt(m*m + (np.sin(p)**2).sum())
print("\n=== zero-relative-momentum family (p,p) -> (p+kappa, p-kappa), numeric, m = 0.6 ===")
rng = np.random.default_rng(1)
p = np.array([np.arcsin(0.6), 0.0, 0.0])
for trial in range(5):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    # solve |kappa| along direction n for the shell: EL(p+r n)+EL(p-r n) = 2 EH(p)
    f = lambda r: EL(p + r*n) + EL(p - r*n) - 2*EH(p)
    lo, hi = 0.0, 1.4
    assert f(lo) < 0 < f(hi) or f(lo) > 0 > f(hi), "bracket"
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(lo)*f(mid) <= 0: hi = mid
        else: lo = mid
    r = 0.5*(lo+hi); A, off = amp_np(p, p, p + r*n, p - r*n)
    print("dir %s  r=%.6f  offshell=%.1e  EL1=%.4f EL2=%.4f  |A|_F=%.3e" % (np.round(n, 3), r, off, EL(p+r*n), EL(p-r*n), np.linalg.norm(A)))
print("\n=== Jacobian rank of (E_H(p1), E_H(p2), E_L(k1)) restricted to the shell at G1 (numeric) ===")
x0 = np.array([float(v) for v in list(pH1)+list(pH2)+list(kL1)])
def F(x):
    p1, p2, k1 = x[0:3], x[3:6], x[6:9]; k2 = p1 + p2 - k1
    return np.array([EH(p1), EH(p2), EL(k1), EH(p1)+EH(p2)-EL(k1)-EL(k2)])
Jm = np.zeros((4, 9)); hstep = 1e-6
for j in range(9):
    e = np.zeros(9); e[j] = hstep; Jm[:, j] = (F(x0+e) - F(x0-e))/(2*hstep)
print("F(x0) =", np.round(F(x0), 12))
print("rank of full 4x9 Jacobian (energies and shell constraint) =", np.linalg.matrix_rank(Jm, tol=1e-8), " singular values:", np.round(np.linalg.svd(Jm, compute_uv=False), 6))
