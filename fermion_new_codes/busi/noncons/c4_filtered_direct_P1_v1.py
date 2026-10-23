#!/usr/bin/env python
"""C4: the filtered converter's amplitude at P1 computed DIRECTLY from the filtered W_0 (VERTEX_stress_v1 (2)), no
factorization: the leg filters F_b are applied as explicit 27-point position-space stencils to the plane-wave orbitals,
the pair operators Phi_{A,b,x}^mu = (F_b c_A)_x^T M^mu (F_b c_A)_x act by Wick contraction on explicit two-particle
Slater states, both orderings O_{b,xy} and O_{b,yx} and the 1/2 are kept, and the Gram weights g_r are kept as
symbols on |r|_inf <= 2 (125 symbols). The only inputs not expanded are the translation sum over x (which gives the
crystal-momentum delta and is stripped exactly as for the on-site amplitude) and, at the end, sum_r g_r = 1.
PASS iff the result equals S * A entrywise with S = 265761/500000 and A the on-site tensor of check C1 at P1.
Part B (numeric, a = 1, L = 8 torus): the same literal real-space construction with two explicit stand-in weight sets
(delta_{r,0}; and a normalized even 7-point set), at the admissible C2 momenta with total momentum P != 0, compared
with kappa(P) S A / V_s, kappa(P) = sum_r g_r e^{i P.r} evaluated from the same weights. Tests the P != 0 factor.
Conventions as in c1_amplitude_exact_v1.py; (T_j u)_x = u_{x+e_j}; F_b = prod_j (1 + (-1)^{b_j}(T_j + T_j^dagger)/2)/2.
"""
import sympy as sp, numpy as np, itertools

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
    for s_ in range(2):
        e = sp.zeros(4, 1); e[s_] = 1; cols.append((E * sp.eye(4) + H) * e / sp.sqrt(2 * E * (E + m)))
    u = sp.Matrix.hstack(*cols)
    assert sp.simplify(H * u - E * u) == sp.zeros(4, 2) and sp.simplify(u.H * u - I2) == sp.zeros(2, 2)
    return u, E

# --- stencil of F_b (position space), exact ---
def wcoef(bj, sj):
    return sp.Rational(1, 2) if sj == 0 else sp.Rational((-1)**bj, 4)
def stencil(b):
    return {s_: sp.prod([wcoef(b[j], s_[j]) for j in range(3)]) for s_ in itertools.product((-1, 0, 1), repeat=3)}
def phase(p, s_):   # e^{i p.s} exact for p with rational cos/sin
    return sp.prod([(sp.cos(p[j]) + sp.I * sp.sin(p[j]))**s_[j] for j in range(3)])
def filtered_orbital_at_origin(p, u, b):
    """(F_b phi)(y) with phi(x) = e^{i p.x} u / sqrt(V_s), evaluated at y = 0 and multiplied by sqrt(V_s):
       explicit 27-term sum  sum_s c_b(s) e^{i p.s} u  (no closed-form filter symbol used)."""
    st = stencil(b)
    return sum((st[s_] * phase(p, s_) for s_ in st), sp.Integer(0)) * u

def pair_contraction(v1, v2, mu):
    """<0| (F c)^T M^mu (F c) |phi1 phi2> at a site = -2 (F phi1)^T M^mu (F phi2) (CAR; M antisymmetric)."""
    return -2 * (v1.T * M[mu] * v2)[0, 0]

th = sp.asin(sp.Rational(3, 5)); Z = sp.Integer(0)
pH1 = (th, Z, Z); pH2 = (-th, Z, Z); kL1 = (th, th, Z); kL2 = (-th, -th, Z)
uH1, EH1 = spinors(pH1, m_H); uH2, EH2 = spinors(pH2, m_H); uL1, EL1 = spinors(kL1, 0); uL2, EL2 = spinors(kL2, 0)
assert sp.simplify(EH1 + EH2 - EL1 - EL2) == 0
Pin = [pH1[j] + pH2[j] for j in range(3)]; Pout = [kL1[j] + kL2[j] for j in range(3)]
assert all(sp.simplify(Pin[j] - Pout[j]) == 0 for j in range(3))
gsyms = {r: sp.Symbol('g_%d_%d_%d' % tuple(x + 2 for x in r), real=True) for r in itertools.product(range(-2, 3), repeat=3)}

# Matrix element of sum_x W_x between |i> = HH plane waves and <f| = LL plane waves, stripped of V_s delta_P.
# W_x = (1/2) sum_{b,y} g_{x-y} (O_{b,xy} + O_{b,yx} + h.c.),  O_{b,xy} = sum_mu eta_mu Phi_{L,b,x}^{mu dagger} Phi_{H,b,y}^mu.
# Position dependence of each pair contraction is the plane-wave phase e^{i P.y} (heavy at y) and e^{-i P.x} (light at x);
# with y = x - r the x-sum gives V_s delta_P and leaves the explicit r-sum sum_r g_r e^{-i P.r} (O_{b,xy}) and
# sum_r g_r e^{+i P.r} (O_{b,yx}); at P = 0 both are sum_r g_r, kept symbolic.
result = sp.zeros(4, 4)   # rows (t1,t2) light spins, cols (s1,s2) heavy spins
for b in itertools.product((0, 1), repeat=3):
    FH1 = filtered_orbital_at_origin(pH1, uH1, b); FH2 = filtered_orbital_at_origin(pH2, uH2, b)
    FL1 = filtered_orbital_at_origin(kL1, uL1, b); FL2 = filtered_orbital_at_origin(kL2, uL2, b)
    for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
        acc = sp.Integer(0)
        for mu in range(4):
            heavy = pair_contraction(FH1[:, s1], FH2[:, s2], mu)
            light = sp.conjugate(pair_contraction(FL1[:, t1], FL2[:, t2], mu))
            acc += eta[mu] * light * heavy
        rsum_xy = sum((gsyms[r] * phase(tuple(-x for x in Pin), r) for r in gsyms), sp.Integer(0))   # e^{-i P.r}
        rsum_yx = sum((gsyms[r] * phase(tuple(Pin), r) for r in gsyms), sp.Integer(0))              # e^{+i P.r}
        result[2 * t1 + t2, 2 * s1 + s2] += sp.Rational(1, 2) * (rsum_xy + rsum_yx) * acc
Gamma = sum(gsyms.values())
result = result.applyfunc(lambda z: sp.simplify(sp.expand(z)))
# reference: S * A with A from the on-site formula (C1) and S from the leg-filter formula
def fb(p, b): return sp.prod([(1 + (-1)**b[j] * sp.cos(p[j])) / 2 for j in range(3)])
S = sp.nsimplify(sp.simplify(sum(sp.prod([fb(q, b) for q in (pH1, pH2, kL1, kL2)]) for b in itertools.product((0, 1), repeat=3))))
JH = [uH1.T * M[mu] * uH2 for mu in range(4)]; JL = [uL1.T * M[mu] * uL2 for mu in range(4)]
A = sp.zeros(4, 4)
for t1, t2, s1, s2 in itertools.product(range(2), repeat=4):
    A[2*t1+t2, 2*s1+s2] = sp.simplify(4 * sum(eta[mu] * sp.conjugate(JL[mu][t1, t2]) * JH[mu][s1, s2] for mu in range(4)))
print("S (leg-filter formula) =", S)
print("direct result / Gamma, Gamma = sum_r g_r (should equal S*A entrywise):")
quot = result.applyfunc(lambda z: sp.simplify(z / Gamma))
sp.pprint(quot.applyfunc(sp.nsimplify))
diff = (quot - S * A).applyfunc(lambda z: sp.simplify(z))
print("entrywise difference (direct/Gamma - S*A):", diff)
print("direct result depends on g_r only through Gamma:", all(sp.simplify(sp.diff(result[i, j], g) - sp.diff(Gamma, g) * quot[i, j]) == 0 for i in range(4) for j in range(4) for g in list(gsyms.values())[::31]))
print("PASS (direct == S*A entrywise with sum_r g_r = 1):", diff == sp.zeros(4, 4))

# ---------------- Part B: literal real-space torus construction at P != 0 (numeric) ----------------
L = 8; m = 1 / np.sqrt(2.0); pi = np.pi
Mn = [np.array(Mi.tolist(), dtype=complex) for Mi in M]
alpn = [np.array(a.tolist(), dtype=complex) for a in alpha]; betn = np.array(beta.tolist(), dtype=complex)
def pos_band(p, mass):
    w, v = np.linalg.eigh(mass * betn + sum(alpn[i] * np.sin(p[i]) for i in range(3))); return w[2:], v[:, 2:]
sites = np.array(list(itertools.product(range(L), repeat=3)))
def planewave(p, spinor): return np.exp(1j * (sites @ p))[:, None] * spinor[None, :] / np.sqrt(L**3)
idx = {tuple(x): n for n, x in enumerate(sites)}
def shift(phi, s_):   # (T^s phi)(x) = phi(x + s) on the torus
    out = np.empty_like(phi)
    for n, x in enumerate(sites): out[n] = phi[idx[tuple((x + np.array(s_)) % L)]]
    return out
def apply_F(phi, b):
    out = np.zeros_like(phi)
    for s_ in itertools.product((-1, 0, 1), repeat=3):
        c = np.prod([0.5 if s_[j] == 0 else (-1)**b[j] / 4 for j in range(3)]); out += c * shift(phi, s_)
    return out
def pairfield(f1, f2, mu):   # -2 (F phi1)(x)^T M (F phi2)(x) for all x
    return -2 * np.einsum('xa,ab,xb->x', f1, Mn[mu], f2)
def V_direct(phiH, phiL, g):   # g: dict r -> weight; returns <f| sum_x W_x |i>
    tot = 0.0
    for b in itertools.product((0, 1), repeat=3):
        FH = [apply_F(phiH[0], b), apply_F(phiH[1], b)]; FL = [apply_F(phiL[0], b), apply_F(phiL[1], b)]
        for mu in range(4):
            Hy = pairfield(FH[0], FH[1], mu); Lx = np.conj(pairfield(FL[0], FL[1], mu))
            for r, gr in g.items():
                Hy_at_x_minus_r = shift(Hy, tuple(-x for x in r))   # heavy pair at y = x - r
                Lx_at_x_minus_r = shift(Lx, tuple(-x for x in r))   # light pair at y = x - r (the O_{b,yx} term)
                tot += 0.5 * gr * eta[mu] * (np.sum(Lx * Hy_at_x_minus_r) + np.sum(Lx_at_x_minus_r * Hy))
    return tot
p1 = np.array([pi/4, 0, 0]); p2 = np.array([pi/4, pi/4, 0]); k1 = np.array([pi/4, pi/2, 0]); k2 = np.array([pi/4, -pi/4, 0]); P = p1 + p2
EH1n, UH1n = pos_band(p1, m); EH2n, UH2n = pos_band(p2, m); EL1n, UL1n = pos_band(k1, 0.0); EL2n, UL2n = pos_band(k2, 0.0)
phiH = [planewave(p1, UH1n[:, 0]), planewave(p2, UH2n[:, 1])]; phiL = [planewave(k1, UL1n[:, 0]), planewave(k2, UL2n[:, 1])]
def fbn(p, b): return np.prod([(1 + (-1)**b[j] * np.cos(p[j])) / 2 for j in range(3)])
Sn = sum(np.prod([fbn(q, b) for q in (p1, p2, k1, k2)]) for b in itertools.product((0, 1), repeat=3))
JHn = [UH1n[:, 0].T @ Mn[mu] @ UH2n[:, 1] for mu in range(4)]; JLn = [UL1n[:, 0].T @ Mn[mu] @ UL2n[:, 1] for mu in range(4)]
An = 4 * sum(eta[mu] * np.conj(JLn[mu]) * JHn[mu] for mu in range(4))
weights = {"delta": {(0, 0, 0): 1.0},
           "7pt": {(0, 0, 0): 0.4, (1, 0, 0): 0.1, (-1, 0, 0): 0.1, (0, 1, 0): 0.1, (0, -1, 0): 0.1, (0, 0, 1): 0.1, (0, 0, -1): 0.1}}
for name, g in weights.items():
    kappaP = sum(gr * np.exp(1j * (P @ np.array(r))) for r, gr in g.items())
    direct = V_direct(phiH, phiL, g); fact = kappaP * Sn * An / L**3
    print("Part B [%s]: kappa(P) = %.6f%+.6fj  direct V_fi = %.9e%+.9ej  factorized kappa S A/V_s = %.9e%+.9ej  |diff| = %.2e" % (name, kappaP.real, kappaP.imag, direct.real, direct.imag, fact.real, fact.imag, abs(direct - fact)))
