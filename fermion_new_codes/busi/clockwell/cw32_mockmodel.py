"""cw32: busi's own independent builder for the no-flow lattice mock (library; imports no icloud code).

Model (MOCK_unitary_lead_run_spec_v1.md sec. 1; MOCK_filtered_clockwell_spec_v1.md sec. 2):
  modes 8x + 4A + c, A = 0 heavy, 1 light; c = upper-up, upper-down, lower-up, lower-down; a = 1; hbar = 1.
  one-body: h[x,x] = m_A N_x beta, h[x,x+1] = -i (N_x + N_{x+1}) alpha / 4 (+ h.c.), beta = diag(1,1,-1,-1),
            alpha = sigma_x (x) sigma_x, m_H = float(3/5) (represented mass), m_L = 0.
  on-site converter: W_x = O_x + O_x^dagger, O_x = sum_{P,Q} K_PQ l^dag_a l^dag_b h_d h_c (P = (a,b), Q = (c,d) in
            PAIR_ORDER), V = sum_x N_x W_x.
  filtered converter (endpoint-copy halo): d_{b,j,c} = sum_x L[b,j,x] c_{x,c}, L = delta_{xj}/2 + (-1)^b (delta_{x,j-1}
            + delta_{x,j+1})/4 (fields zero outside the chain), O_jk = sum_b sum_{P,Q} K_PQ d^dag_{L,b,j,a} d^dag_{L,b,j,b'}
            d_{H,b,k,d} d_{H,b,k,c}; F = sum_{j,k=-1..n} G_{j-k} (N_j + N_k)/2 O_jk; V = F + F^dag; N_{-1} = N_0 = 1/2,
            N_n = N_{n-1} = 1; G_0, G_1, G_2 the admitted exact overlap coefficients.
  H = dGamma(h) + g V on the complete two-particle sector, basis |m1 m2> = c^dag_m1 c^dag_m2 |0>, m1 < m2.
"""
from fractions import Fraction as Fr
import numpy as np
import itertools

PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
K = np.array([(4, 0, 0, 0, 0, -4), (0, -8, 0, 0, 0, 0), (0, 0, -4, -4, 0, 0),
              (0, 0, -4, -4, 0, 0), (0, 0, 0, 0, -8, 0), (-4, 0, 0, 0, 0, 4)], dtype=float)
G = {0: float(Fr(388627261303, 549873254400)), 1: float(Fr(456290907571, 3115948441600)), 2: float(Fr(180865117, 983983718400))}
BETA = np.diag([1.0, 1.0, -1.0, -1.0])
SX = np.array([[0, 1], [1, 0]], dtype=float)
ALPHA = np.kron(SX, SX)
MH = 3 / 5  # float(3/5): the represented mass

def mode(x, A, c):
    return 8 * x + 4 * A + c

def basis(n):
    """two-particle basis ordered by occupation bitmask value (as the saved row_keys)"""
    M = 8 * n
    pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
    pairs.sort(key=lambda p: (1 << p[0]) | (1 << p[1]))
    return pairs

def profile(n):
    return [0.5] + [1.0] * (n - 1)

def one_body(n, N=None):
    N = N or profile(n)
    M = 8 * n
    h = np.zeros((M, M), dtype=complex)
    for x in range(n):
        for A, m in ((0, MH), (1, 0.0)):
            for c in range(4):
                for c2 in range(4):
                    h[mode(x, A, c), mode(x, A, c2)] += m * N[x] * BETA[c, c2]
            if x + 1 < n:
                for c in range(4):
                    for c2 in range(4):
                        v = -1j * (N[x] + N[x + 1]) * ALPHA[c, c2] / 4
                        h[mode(x, A, c), mode(x + 1, A, c2)] += v
                        h[mode(x + 1, A, c2), mode(x, A, c)] += np.conj(v)
    return h

def onsite_quartic(n, N=None):
    """dict (p, q, s, r) -> coefficient of l^dag_p l^dag_q h_s h_r in F (forward part of V)"""
    N = N or profile(n)
    C = {}
    for x in range(n):
        for P, (a, b) in enumerate(PAIR_ORDER):
            for Q, (c, d) in enumerate(PAIR_ORDER):
                if K[P, Q]:
                    key = (mode(x, 1, a), mode(x, 1, b), mode(x, 0, d), mode(x, 0, c))
                    C[key] = C.get(key, 0.0) + N[x] * K[P, Q]
    return C

def stencil(n, b, j):
    """{x: L[b,j,x]} over physical sites"""
    out = {}
    for x in range(n):
        v = (0.5 if x == j else 0.0) + ((-1) ** b) * ((1.0 if x == j - 1 else 0.0) + (1.0 if x == j + 1 else 0.0)) / 4
        if v:
            out[x] = v
    return out

def filtered_quartic(n, N=None):
    Nphys = N or profile(n)
    Nall = {j: (Nphys[0] if j == -1 else (Nphys[n - 1] if j == n else Nphys[j])) for j in range(-1, n + 1)}
    C = {}
    for j in range(-1, n + 1):
        for k in range(-1, n + 1):
            gr = G.get(abs(j - k))
            if gr is None:
                continue
            w = gr * (Nall[j] + Nall[k]) / 2
            for b in (0, 1):
                Lj = stencil(n, b, j); Lk = stencil(n, b, k)
                if not Lj or not Lk:
                    continue
                for P, (a, a2) in enumerate(PAIR_ORDER):
                    for Q, (c, d) in enumerate(PAIR_ORDER):
                        kpq = K[P, Q]
                        if not kpq:
                            continue
                        for x1, v1 in Lj.items():
                            for x2, v2 in Lj.items():
                                for y1, u1 in Lk.items():
                                    for y2, u2 in Lk.items():
                                        key = (mode(x1, 1, a), mode(x2, 1, a2), mode(y2, 0, d), mode(y1, 0, c))
                                        C[key] = C.get(key, 0.0) + w * kpq * v1 * v2 * u1 * u2
    return C

def assemble(n, g, form, N=None, keep=None):
    """dense two-particle H; keep(term_modes) -> bool filters monomials (protection); returns H, basis"""
    B = basis(n); idx = {p: i for i, p in enumerate(B)}; D = len(B)
    h = one_body(n, N)
    if keep is not None:
        for i in range(h.shape[0]):
            for j in range(h.shape[0]):
                if h[i, j] != 0 and not keep((i, j)):
                    h[i, j] = 0
    H = np.zeros((D, D), dtype=complex)
    for (a, b2), I in idx.items():
        for (c, d), J in idx.items():
            H[I, J] = (h[a, c] if b2 == d else 0) - (h[a, d] if b2 == c else 0) - (h[b2, c] if a == d else 0) + (h[b2, d] if a == c else 0)
    if g:
        C = onsite_quartic(n, N) if form == 'onsite' else filtered_quartic(n, N)
        F = np.zeros((D, D), dtype=complex)
        for (p, q, s, r), cf in C.items():
            if p == q or r == s or cf == 0:
                continue
            if keep is not None and not keep((p, q, s, r)):
                continue
            I = idx[(min(p, q), max(p, q))]; sg = 1.0 if p < q else -1.0
            # h_s h_r |n1 n2> = (delta_{r n1} delta_{s n2} - delta_{r n2} delta_{s n1}) |0>
            if r < s:
                F[I, idx[(r, s)]] += sg * cf
            else:
                F[I, idx[(s, r)]] -= sg * cf
        H += g * (F + F.conj().T)
    return H, B

def propagate(H, psi, t):
    w, V = np.linalg.eigh((H + H.conj().T) / 2)
    return V @ (np.exp(-1j * w * t) * (V.conj().T @ psi))

def initial_states(B):
    idx = {p: i for i, p in enumerate(B)}; D = len(B)
    pure = np.zeros(D, dtype=complex); pure[idx[(0, 1)]] = 1
    Dw = np.zeros(D, dtype=complex); Dw[idx[(0, 3)]] = 1 / np.sqrt(2); Dw[idx[(1, 2)]] = -1 / np.sqrt(2)
    reg = np.vstack([pure / np.sqrt(2), Dw / np.sqrt(2)])
    return pure, reg

def reduce_rho(vecs, B, keep):
    keep = set(keep); kc, tc, ent = {}, {}, []
    for i, (m1, m2) in enumerate(B):
        a = tuple(m for m in (m1, m2) if m in keep); t = tuple(m for m in (m1, m2) if m not in keep)
        sign = -1.0 if (m1 not in keep and m2 in keep) else 1.0
        kc.setdefault(a, len(kc)); tc.setdefault(t, len(tc)); ent.append((i, kc[a], tc[t], sign))
    R = vecs.shape[0]
    Cm = np.zeros((R * len(kc), len(tc)), dtype=complex)
    for i, a, t, s in ent:
        for r in range(R):
            Cm[r * len(kc) + a, t] += s * vecs[r, i]
    return Cm @ Cm.conj().T, len(kc)

def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2); w = w[w > 1e-300]
    return float(-(w * np.log(w)).sum())

def D_RK(reg, B, out_modes, n):
    Km = [m for m in range(8 * n) if m not in set(out_modes)]
    rho, dk = reduce_rho(reg, B, Km)
    rR = np.array([[np.trace(rho[r*dk:(r+1)*dk, s*dk:(s+1)*dk]) for s in range(2)] for r in range(2)])
    rK = rho[:dk, :dk] + rho[dk:, dk:]
    X = rho - np.kron(rR, rK)
    return 0.5 * float(np.abs(np.linalg.eigvalsh((X + X.conj().T) / 2)).sum())

def p_species(psi, B):
    sp = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in B])
    w = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
    return float(w[sp == 0].sum()), float(w[sp == 2].sum()), float(w[sp == 1].sum())
