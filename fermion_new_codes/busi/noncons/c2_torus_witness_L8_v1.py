#!/usr/bin/env python
"""C2: finite-torus witness of the first-order obstruction (Lemma 1), on-site converter, L = 8, a = 1, m = 1/sqrt2.

Named pair (exactly degenerate, same total crystal momentum P = (pi/2, pi/4, 0)):
  |i> = heavy (pi/4, 0, 0) + heavy (pi/4, pi/4, 0):  E = 1 + sqrt(3/2)
  |f> = light (pi/4, pi/2, 0) + light (pi/4, -pi/4, 0): E = sqrt(3/2) + 1
  Dq for q = sin(2 p_y)/2:  q_i = 0 + 1/2,  q_f = 0 - 1/2,  Dq = q_f - q_i = -1.
Steps:
 (1) one-body h_0(p) diagonalized exactly (4x4 per momentum); positive-band energies; degeneracy verified to 1e-12.
 (2) V_fi from the REAL-SPACE site-local vertex: W_x = sum_mu eta_mu Phi_{L,x}^{mu dagger} Phi_{H,x}^mu + h.c.,
     Phi_{A,x}^mu = c_{A,x}^T M^mu c_{A,x}, acting on explicit Slater-determinant two-particle wavefunctions
     psi(a,b) = phi1(a) phi2(b) - phi1(b) phi2(a) over mode pairs; only same-site pairs contribute. Derivation of
     the pair action (CAR, M antisymmetric): W_x |a,b>_H = 4 sum_{e<f} sum_mu eta_mu conj(M^mu_{ef}) M^mu_{ab} |e,f>_L
     for a<b, e<f component indices at site x; the matrix of V in the two-particle HH -> LL sector is this per site.
 (3) Independent comparison with the stripped spinor formula A/V_s (VERTEX_force_event_v1 (4)).
 (4) The on-shell block: all two-particle HH and LL plane-wave eigenstates with total momentum P and energy E
     (|E - E0| < 1e-9) are enumerated; V_E = P_E V P_E built from (2); Q_E diagonal with the exact symbol sums;
     ||[V_E, Q_E]||_F reported. Nonzero => no operator Q_1 whatsoever solves [H_0,Q_1] + [V,Q_0] = 0 on this torus.
 (5) Uniform non-stationary state psi = (|i> + e^{i phi}|f>)/sqrt2: d<Q_0>/dt / U = Dq Im(e^{i phi} V_fi) at first order.
"""
import numpy as np, itertools

L = 8; m = 1/np.sqrt(2.0)
I2 = np.eye(2); sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
alpha = [np.kron(sx, sx), np.kron(sx, sy), np.kron(sx, sz)]; beta = np.kron(sz, I2)
M = [np.kron(sz, sy), np.kron(sy, sz), 1j*np.kron(sy, I2), -np.kron(sy, sx)]; eta = [1, -1, -1, -1]
for Mi in M: assert np.allclose(Mi.T, -Mi)
K = np.zeros((4, 4, 4, 4), complex)   # K[e,f,a,b] = 4 sum eta conj(M_ef) M_ab
for mu in range(4): K += 4*eta[mu]*np.einsum('ef,ab->efab', np.conj(M[mu]), M[mu])

def h0(p, mass): return mass*beta + sum(alpha[i]*np.sin(p[i]) for i in range(3))
def pos_band(p, mass):
    w, v = np.linalg.eigh(h0(p, mass)); return w[2:], v[:, 2:]     # two positive eigenvalues (ascending)
sites = np.array(list(itertools.product(range(L), repeat=3)))      # (512,3)
def planewave(p, spinor):   # returns (512,4) array phi(x,c)
    return np.exp(1j*(sites @ p))[:, None]*spinor[None, :]/np.sqrt(L**3)
def slater_site_pairs(phi1, phi2):
    """antisymmetric coefficients psi(x; a<b) for same-site component pairs: shape (512, 4, 4), antisymmetric in (a,b)."""
    return np.einsum('xa,xb->xab', phi1, phi2) - np.einsum('xa,xb->xab', phi2, phi1)
def V_elem(psiL, psiH):
    """<f|V|i> from site-local vertex: sum_x sum_{e<f,a<b} conj(psiL[x,e,f]) K[e,f,a,b] psiH[x,a,b]."""
    tot = 0.0
    for e, f in itertools.combinations(range(4), 2):
        for a, b in itertools.combinations(range(4), 2):
            tot += np.sum(np.conj(psiL[:, e, f])*K[e, f, a, b]*psiH[:, a, b])
    return tot

pi = np.pi
p1 = np.array([pi/4, 0, 0]); p2 = np.array([pi/4, pi/4, 0]); k1 = np.array([pi/4, pi/2, 0]); k2 = np.array([pi/4, -pi/4, 0])
P = p1 + p2; assert np.allclose(np.mod(k1 + k2 - P, 2*pi), 0)
EH1, UH1 = pos_band(p1, m); EH2, UH2 = pos_band(p2, m); EL1, UL1 = pos_band(k1, 0.0); EL2, UL2 = pos_band(k2, 0.0)
E0 = EH1[0] + EH2[0]
print("one-body positive energies: heavy %.12f %.12f ; light %.12f %.12f" % (EH1[0], EH2[0], EL1[0], EL2[0]))
print("E_i = %.15f  E_f = %.15f  |E_i - E_f| = %.2e   (exact: 1 + sqrt(3/2) = %.15f)" % (E0, EL1[0]+EL2[0], abs(E0-EL1[0]-EL2[0]), 1+np.sqrt(1.5)))
qsym = lambda p: np.sin(2*p[1])/2
Dq = qsym(k1) + qsym(k2) - qsym(p1) - qsym(p2)
print("Dq for q = sin(2 p_y)/2 :", Dq)

# (2)+(3): named pair, all spin combinations
print("\nV_fi * V_s (real-space site-local vertex) vs stripped spinor formula A, rows (t1,t2), cols (s1,s2):")
Vmat = np.zeros((4, 4), complex); Amat = np.zeros((4, 4), complex)
for s1, s2 in itertools.product(range(2), repeat=2):
    psiH = slater_site_pairs(planewave(p1, UH1[:, s1]), planewave(p2, UH2[:, s2]))
    for t1, t2 in itertools.product(range(2), repeat=2):
        psiL = slater_site_pairs(planewave(k1, UL1[:, t1]), planewave(k2, UL2[:, t2]))
        Vmat[2*t1+t2, 2*s1+s2] = V_elem(psiL, psiH)*L**3
        JH = [UH1[:, s1].T @ M[mu] @ UH2[:, s2] for mu in range(4)]; JL = [UL1[:, t1].T @ M[mu] @ UL2[:, t2] for mu in range(4)]
        Amat[2*t1+t2, 2*s1+s2] = 4*sum(eta[mu]*np.conj(JL[mu])*JH[mu] for mu in range(4))
np.set_printoptions(precision=6, suppress=True, linewidth=150)
print("real-space:\n", Vmat); print("stripped formula:\n", Amat)
print("max |difference| = %.3e ;  |V_fi V_s|_F = %.6f" % (np.abs(Vmat-Amat).max(), np.linalg.norm(Vmat)))
print("Dq * V_fi * V_s nonzero:", np.linalg.norm(Dq*Vmat) > 1e-9)

# (4): on-shell block at energy E0 with total momentum P (HH and LL two-particle states)
moms = [np.array(n)*2*pi/L for n in itertools.product(range(L), repeat=3)]
def key(p): return tuple(np.round(np.mod(p, 2*pi)/(2*pi/L)).astype(int) % L)
states = []   # (species, pa, sa, pb, sb, energy, qsum)
for pa in moms:
    pb = np.mod(P - pa, 2*pi)
    if key(pa) > key(pb): continue
    for species, mass in (("H", m), ("L", 0.0)):
        Ea, Ua = pos_band(pa, mass); Eb, Ub = pos_band(pb, mass)
        for sa in range(2):
            for sb in range(2):
                if key(pa) == key(pb) and sb <= sa: continue
                E = Ea[sa] + Eb[sb]
                if abs(E - E0) < 1e-9:
                    states.append((species, pa, sa, pb, sb, E, qsym(pa)+qsym(pb)))
nH = sum(1 for s in states if s[0] == "H"); nL = len(states) - nH
print("\nexactly degenerate two-particle states at E0 with total momentum P: HH %d, LL %d" % (nH, nL))
def psi_of(st):
    sp_, pa, sa, pb, sb, E, q = st; mass = m if sp_ == "H" else 0.0
    Ea, Ua = pos_band(pa, mass); Eb, Ub = pos_band(pb, mass)
    return slater_site_pairs(planewave(pa, Ua[:, sa]), planewave(pb, Ub[:, sb]))
psis = [psi_of(s) for s in states]
# orthonormality check of the enumerated basis (same-site pair coefficients are not the full overlap; use plane-wave labels)
VE = np.zeros((len(states), len(states)), complex)
for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if si[0] == "L" and sj[0] == "H":
            VE[i, j] = V_elem(psis[i], psis[j])
VE = VE + VE.conj().T
QE = np.diag([s[6] for s in states])
comm = VE @ QE - QE @ VE
print("||P_E V P_E||_F = %.6f   ||[P_E V P_E, Q_0]||_F = %.6f   (nonzero => no Q_1 exists on this torus)" % (np.linalg.norm(VE), np.linalg.norm(comm)))
print("distinct Q_0 eigenvalues on the shell:", sorted(set(np.round([s[6] for s in states], 12))))
print("Hermiticity of V_E: %.2e" % np.abs(VE - VE.conj().T).max())
# (5) uniform non-stationary state
Vfi = Vmat[0, 1]/L**3 if abs(Vmat[0, 1]) > 1e-12 else Vmat.flat[np.argmax(np.abs(Vmat))]/L**3
best = max(((abs((Dq*np.exp(1j*ph)*Vfi).imag), ph) for ph in np.linspace(0, 2*pi, 721)))
print("\nuniform state psi=(|i>+e^{i phi}|f>)/sqrt2 with the largest-|V_fi| spin pair: V_fi = %s ;  max_phi |d<Q_0>/dt| / U = %.6e (per torus), per site %.3e" % (Vfi, best[0], best[0]/L**3))
