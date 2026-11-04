"""vaclaw_w2_v3: full-flow second-gradient vacuum-polarization counterterm W_2 (route 1, solve), per ReviewB_w2fulla_v1 section 1.

CHAIN (eqs (2)-(4)), heavy chain channel k, mass m, sites r_n = n (a = 1), h[u] = (1/2){N, h0} + (1/2){v, Q}:
    d(r, p) = (sin p, k/r, m),  eps = |d|,  n = d/eps,  n_xi = (I - n n^T) d_xi / eps,  q = sin(2p)/2,
    A_n = int dp/2pi  q^2 |n_p|^2 / (8 eps),   B_n = int q q' n_p . n_r / (8 eps),   C_n = int q'^2 |n_r|^2 / (8 eps),
    F_n = A_n v'_n^2 - 2 B_n v_n v'_n + C_n v_n^2,     W_2[N, v] = - sum_n F_n / N_n.
  n_r carries the mass vector M = (k/r, m) (magnitude and direction); it is KEPT. EXACT in the flow amplitude at second-gradient
  order (a scalar tilt changes neither the band eigenvectors nor the 2 N eps gap); degree one: W_2(lam N, lam v) = lam W_2.
  Discretize first: v'_n = (v_{n+1} - v_{n-1})/2 (one-sided at the ends), d_r = (0, -k/r_n^2, 0) analytic; dW2 is the exact derivative of
  this discrete functional (adjoint difference operator). p-quadrature: periodic trapezoid (npts nodes).
  Identity with v1/v2: A = v1.S11, C = v1.S22, B = -v1.S12 (algebraically; checked numerically), so v3 = v2 on the chain.
  Full reference functional with preparation u_star = (N*, v*) (P_star = P0 for homogeneous preparation):
    W[u] = sum_x (N_x Tr(P0 e_x) + v_x Tr(P0 p_x)) + W_2[u] - DW_2[u_star] . u,   r = D_u W.
  For u_star = (1, 0), DW_2[u_star] = 0 (F is quadratic in v).
SCOPE: the low-k core is OUTSIDE control: eta_M = k/(k^2 + m^2 r^2) > 0.05 (r < 50a at ma = 0.2, all k) [reviewer B sec 2]; the chain check
  gave A/A_2 = 0.682 on a flow reaching r = 0 (k = 1) and 0.990 -> 0.9994 (quarter law) on flows cut below 32a. eta_M(r) is reported
  by eta_M(ns, k, m); the module does not zero W_2 there (that would be a different subtraction).
CUBIC (eq (5)): W_2 = - sum_x d_a v_i d_b v_j kappa_{ai,bj} / N, kappa_{ai,bj} = (1/4) int_BZ d^3p/(2pi)^3 q_i q_j / eps (d_{p_a} n . d_{p_b} n),
  n = (sin p1, sin p2, sin p3, m)/eps (no position dependence), kappa(w, m) = kappa(0, m) exactly at this order. Periodic grid,
  central differences, exact discrete derivative.
API (adapter-compatible with v2): A2(N, v), W2(N, v), dW2(N, v) -> (dW2/dN, dW2/dv), W_full(N, v, mom, u_star=None), r_full(...),
  coeffs(ns, k, m, npts) -> (A, B, C), eta_M(ns, k, m); cubic: kappa_cubic(m, npts), W2_cubic(N, v3, kap), dW2_cubic(N, v3, kap).
"""
import numpy as np

_CACHE = {}


def coeffs(ns, k=1.0, m=0.2, npts=4096):
    key = (ns, float(k), float(m), npts)
    if key in _CACHE:
        return _CACHE[key]
    r = np.arange(1, ns + 1, dtype=float)
    p = -np.pi + 2 * np.pi * np.arange(npts) / npts
    s, cp = np.sin(p), np.cos(p); q = np.sin(2 * p) / 2; qp = np.cos(2 * p)
    mu = k / r; mup = -k / r ** 2
    eps = np.sqrt(s[None, :] ** 2 + mu[:, None] ** 2 + m ** 2)
    n1, n2, n3 = s[None, :] / eps, mu[:, None] / eps, m / eps
    # d_p = (cos p, 0, 0), d_r = (0, mu', 0); n_xi = (d_xi - n (n . d_xi)) / eps
    ndp = n1 * cp[None, :]; ndr = n2 * mup[:, None]
    np1, np2, np3 = (cp[None, :] - n1 * ndp) / eps, (-n2 * ndp) / eps, (-n3 * ndp) / eps
    nr1, nr2, nr3 = (-n1 * ndr) / eps, (mup[:, None] - n2 * ndr) / eps, (-n3 * ndr) / eps
    w = 1.0 / (8 * eps) / npts
    A = ((q ** 2)[None, :] * (np1 ** 2 + np2 ** 2 + np3 ** 2) * w).sum(1)
    B = ((q * qp)[None, :] * (np1 * nr1 + np2 * nr2 + np3 * nr3) * w).sum(1)
    C = ((qp ** 2)[None, :] * (nr1 ** 2 + nr2 ** 2 + nr3 ** 2) * w).sum(1)
    _CACHE[key] = (A, B, C)
    return _CACHE[key]


def eta_M(ns, k=1.0, m=0.2):
    r = np.arange(1, ns + 1, dtype=float)
    return k / (k ** 2 + (m * r) ** 2)


def _D(v):
    vp = np.empty_like(v); vp[1:-1] = (v[2:] - v[:-2]) / 2; vp[0] = v[1] - v[0]; vp[-1] = v[-1] - v[-2]
    return vp


def _DT(g):
    out = np.zeros_like(g)
    out[2:] += g[1:-1] / 2; out[:-2] -= g[1:-1] / 2
    out[1] += g[0]; out[0] -= g[0]; out[-1] += g[-1]; out[-2] -= g[-1]
    return out


def F(v, k=1.0, m=0.2, npts=4096):
    v = np.asarray(v, float); A, B, C = coeffs(len(v), k, m, npts); vp = _D(v)
    return A * vp ** 2 - 2 * B * v * vp + C * v ** 2


def A2(N, v, k=1.0, m=0.2, npts=4096):
    return float((F(v, k, m, npts) / np.asarray(N, float)).sum())


def W2(N, v, k=1.0, m=0.2, npts=4096):
    return -A2(N, v, k, m, npts)


def dW2(N, v, k=1.0, m=0.2, npts=4096):
    v = np.asarray(v, float); N = np.asarray(N, float); A, B, C = coeffs(len(v), k, m, npts); vp = _D(v)
    Fn = A * vp ** 2 - 2 * B * v * vp + C * v ** 2
    dN = Fn / N ** 2                                        # dW2/dN = +F/N^2
    gv = -((2 * C * v - 2 * B * vp) / N + _DT((2 * A * vp - 2 * B * v) / N))   # dW2/dv (exact discrete)
    return dN, gv


def W_full(N, v, mom, u_star=None, k=1.0, m=0.2, npts=4096):
    e0, p0 = mom
    W = float(np.dot(N, e0) + np.dot(v, p0)) + W2(N, v, k, m, npts)
    if u_star is not None:
        sN, sv = dW2(u_star[0], u_star[1], k, m, npts); W -= float(np.dot(sN, N) + np.dot(sv, v))
    return W


def r_full(N, v, mom, u_star=None, k=1.0, m=0.2, npts=4096):
    e0, p0 = mom; gN, gv = dW2(N, v, k, m, npts)
    rN, rv = np.asarray(e0) + gN, np.asarray(p0) + gv
    if u_star is not None:
        sN, sv = dW2(u_star[0], u_star[1], k, m, npts); rN = rN - sN; rv = rv - sv
    return rN, rv


# ---------------------------------------------------------------- cubic (eq (5))
def kappa_cubic(m=0.2, npts=64):
    p = -np.pi + 2 * np.pi * np.arange(npts) / npts
    P = np.meshgrid(p, p, p, indexing='ij')
    s = [np.sin(x) for x in P]; c = [np.cos(x) for x in P]; q = [np.sin(2 * x) / 2 for x in P]
    eps = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + m ** 2)
    n = [s[0] / eps, s[1] / eps, s[2] / eps, m / eps]
    dn = []                                              # d_{p_a} n (4-vector) for a = 0, 1, 2
    for a in range(3):
        da = [c[a] if j == a else 0.0 for j in range(4)]
        nd = n[a] * c[a]
        dn.append([(da[j] - n[j] * nd) / eps for j in range(4)])
    kap = np.zeros((3, 3, 3, 3))
    w = 0.25 / eps / npts ** 3
    for a in range(3):
        for b in range(3):
            g = sum(dn[a][j] * dn[b][j] for j in range(4)) * w
            for i in range(3):
                for j in range(3):
                    kap[a, i, b, j] = float((q[i] * q[j] * g).sum())
    return kap


def _cd(f, ax):
    return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / 2


def W2_cubic(N, v3, kap):
    """N (nx, ny, nz); v3 (3, nx, ny, nz) periodic; central differences."""
    G = np.array([[_cd(v3[i], a) for i in range(3)] for a in range(3)])          # G[a, i] = d_a v_i
    dens = np.einsum('aibj,ai...,bj...->...', kap, G, G)
    return -float((dens / N).sum())


def dW2_cubic(N, v3, kap):
    G = np.array([[_cd(v3[i], a) for i in range(3)] for a in range(3)])
    dens = np.einsum('aibj,ai...,bj...->...', kap, G, G)
    dN = dens / N ** 2
    H = 2 * np.einsum('aibj,bj...->ai...', kap, G) / N                               # d dens/d G_ai / N  (kap symmetric in (ai)<->(bj))
    gv = np.zeros_like(v3)
    for i in range(3):
        for a in range(3):
            gv[i] += _cd(H[a, i], a)       # dW/dv_i = -sum_a D_a^T H_ai = +sum_a D_a H_ai (periodic: D^T = -D)
    return dN, gv
