"""Second reader (clockwell_r2), #492: own builder of the no-flow lattice mock, written from the specs
(MOCK_unitary_lead_run_spec_v1 sec. 1, MOCK_filtered_clockwell_spec_v1 sec. 2, MOCK_n5_plan_v2 sec. 3).
Modes m = 8x + 4A + c (A = 0 heavy, 1 light; c = upper-up, upper-down, lower-up, lower-down).
Basis of the N = 2 sector: |m1 m2> = c^dag_m1 c^dag_m2 |0>, m1 < m2, ordered by the bitmask (1<<m1)|(1<<m2).
Free part: h[x,x] = m_A N_x beta, h[x,x+1] = -i (N_x + N_{x+1}) alpha / 4, beta = diag(1,1,-1,-1), alpha = sigma_x (x) sigma_x.
Converter (N = 2 sector, exact): a pair-field bilinear is a sum of outer products of two-particle pair states:
  on-site  O_x  = sum_{s,s'} K[s,s'] |L-pair s at x><H-pair s' at x|,     V = sum_x N_x (O_x + O_x^dag)
  filtered O_jk = sum_{b=0,1} sum_{s,s'} K[s,s'] |L-pair s of d_{b,j}><H-pair s' of d_{b,k}|,
           d_{b,j,a}^dag = sum_x L[b,j,x] c^dag_{x,a},  L[b,j,x] = delta_xj/2 + (-1)^b (delta_{x,j-1} + delta_{x,j+1})/4,
           W_j = (1/2) sum_k G_{j-k} (O_jk + O_kj + h.c.),  V = sum_{j=-1..n} N_j W_j, N_-1 = N_0, N_n = N_{n-1} (endpoint-copy halo)
K (6 x 6, pair order 01,02,03,12,13,23) is read from icloud's saved n = 3 on-site Hamiltonians (g = 1 minus g = 0, site 1 block)
and validated by reproducing every saved n = 3 Hamiltonian (r2_51).
A generic scalar field is supported (float/complex numpy, or exact via the 'exact' path with Fractions) for the dark-subspace rank.
"""
import numpy as np
from fractions import Fraction as Fr

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
G_EXACT = {0: Fr(388627261303, 549873254400), 1: Fr(456290907571, 3115948441600), 2: Fr(180865117, 983983718400)}
ALPHA = np.zeros((4, 4)); ALPHA[0, 3] = ALPHA[1, 2] = ALPHA[2, 1] = ALPHA[3, 0] = 1.0
BETA = np.diag([1.0, 1.0, -1.0, -1.0])

def basis(n):
    M = 8 * n
    pairs = sorted(((1 << a) | (1 << b), a, b) for a in range(M) for b in range(a + 1, M))
    idx = {(a, b): i for i, (_, a, b) in enumerate(pairs)}
    return [(a, b) for _, a, b in pairs], idx

def profile(n):
    return [0.5] + [1.0] * (n - 1)

def one_body(n, mH=0.6, prof=None, exact=False):
    """h as dict {(p,q): value}; exact -> (re Fraction, im Fraction)"""
    N = prof or profile(n); h = {}
    one = Fr(1) if exact else 1.0
    for x in range(n):
        Nx = Fr(N[x]).limit_denominator(4) if exact else N[x]
        for A, m in ((0, (Fr(3, 5) if exact else mH)), (1, 0)):
            for c in range(4):
                v = m * Nx * (1 if c < 2 else -1)
                if v != 0: h[(8 * x + 4 * A + c, 8 * x + 4 * A + c)] = (v, Fr(0)) if exact else complex(v)
    for x in range(n - 1):
        Ns = (Fr(N[x]).limit_denominator(4) + Fr(N[x + 1]).limit_denominator(4)) if exact else N[x] + N[x + 1]
        for A in (0, 1):
            for c in range(4):
                for c2 in range(4):
                    if ALPHA[c, c2] == 0: continue
                    p, q = 8 * x + 4 * A + c, 8 * (x + 1) + 4 * A + c2
                    if exact:
                        h[(p, q)] = (Fr(0), -Ns / 4); h[(q, p)] = (Fr(0), Ns / 4)
                    else:
                        h[(p, q)] = -1j * Ns / 4; h[(q, p)] = 1j * Ns / 4
    return h

def dgamma(n, h, exact=False):
    """second quantization of h on the N = 2 sector; returns dict {(i,j): value} (row i, col j)"""
    B, idx = basis(n); out = {}
    by_col = {}
    for (p, q), v in h.items(): by_col.setdefault(q, []).append((p, v))
    for j, (m1, m2) in enumerate(B):
        for (keep, mv) in ((m2, m1), (m1, m2)):  # apply c^dag_p c_mv on |m1 m2>
            s_ann = 1 if mv == m1 else -1          # c_m1 |m1 m2> = +|m2>, c_m2 |m1 m2> = -|m1>
            for p, v in by_col.get(mv, []):
                if p == keep: continue
                s_cr = 1 if p < keep else -1       # c^dag_p |keep> -> canonical order
                i = idx[(min(p, keep), max(p, keep))]
                s = s_ann * s_cr
                if exact:
                    cur = out.get((i, j), (Fr(0), Fr(0))); out[(i, j)] = (cur[0] + s * v[0], cur[1] + s * v[1])
                else:
                    out[(i, j)] = out.get((i, j), 0) + s * v
    return out

def pair_vec(n, A, sp, weights, exact=False):
    """two-particle vector d_a^dag d_b^dag |0>, sp = (a,b), d_c^dag = sum_x weights[x] c^dag_{x,A,c}; dict {index: amp}"""
    B, idx = basis(n); a, b = sp; v = {}
    for x, wx in weights.items():
        for y, wy in weights.items():
            m, m2 = 8 * x + 4 * A + a, 8 * y + 4 * A + b
            if m == m2: continue
            s = 1 if m < m2 else -1
            i = idx[(min(m, m2), max(m, m2))]
            v[i] = v.get(i, 0) + s * wx * wy
    return v

def stencil(n, b, j, exact=False):
    half = Fr(1, 2) if exact else 0.5; q = Fr(1, 4) if exact else 0.25
    w = {}
    for x in range(n):
        val = (half if x == j else 0) + (-1) ** b * ((q if x == j - 1 else 0) + (q if x == j + 1 else 0))
        if val != 0: w[x] = val
    return w

def converter_terms(n, form, K, prof=None, exact=False):
    """list of (coef, Lvec, Hvec) with V = sum coef (|L><H| + |H><L|) on the N = 2 sector (K real)."""
    N = prof or profile(n); terms = []
    if form == "onsite":
        for x in range(n):
            Nx = Fr(N[x]).limit_denominator(4) if exact else N[x]
            Lv = [pair_vec(n, 1, s, {x: 1}, exact) for s in PAIRS]; Hv = [pair_vec(n, 0, s, {x: 1}, exact) for s in PAIRS]
            for si in range(6):
                for sj in range(6):
                    if K[si][sj] != 0: terms.append((Nx * K[si][sj], Lv[si], Hv[sj]))
        return terms
    Nh = {j: (Fr(N[j]).limit_denominator(4) if exact else N[j]) for j in range(n)}
    Nh[-1] = Nh[0]; Nh[n] = Nh[n - 1]
    for j in range(-1, n + 1):
        for k in range(-1, n + 1):
            r = abs(j - k)
            if r > 2: continue
            Gr = G_EXACT[r] if exact else float(G_EXACT[r])
            wgt = Gr * (Nh[j] + Nh[k]) / 2        # sum_j N_j W_j = sum_{j,k} G (N_j+N_k)/2 (O_jk + h.c.)
            for b in (0, 1):
                sj, sk = stencil(n, b, j, exact), stencil(n, b, k, exact)
                if not sj or not sk: continue
                Lv = [pair_vec(n, 1, s, sj, exact) for s in PAIRS]; Hv = [pair_vec(n, 0, s, sk, exact) for s in PAIRS]
                for si in range(6):
                    for sj_ in range(6):
                        if K[si][sj_] != 0: terms.append((wgt * K[si][sj_], Lv[si], Hv[sj_]))
    return terms

def hamiltonian(n, form, g, K, prof=None, mH=0.6):
    """dense complex128 H (dimension C(8n,2))"""
    B, idx = basis(n); D = len(B)
    H = np.zeros((D, D), complex)
    for (i, j), v in dgamma(n, one_body(n, mH, prof)).items(): H[i, j] += v
    if g != 0:
        V = np.zeros((D, D), complex)
        for c, Lv, Hv in converter_terms(n, form, K, prof):
            l = np.zeros(D); hh = np.zeros(D)
            for i, a in Lv.items(): l[i] = a
            for i, a in Hv.items(): hh[i] = a
            V += c * np.outer(l, hh)
        H += g * (V + V.conj().T)
    return H
