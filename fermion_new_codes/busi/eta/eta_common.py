"""eta_common.py -- shared routines of the busi lane's planar-coefficient (eta) computation.

Conventions, binding for every script in this directory:
  * lattice spacing a = 1; natural logarithms (nats);
  * one 4-component sine-Dirac field, h(k) = m beta + sum_i alpha_i sin k_i, occupied projector
    P(k) = (1 - h/eps)/2, eps = sqrt(m^2 + sum_i sin^2 k_i);
  * eta(n, m) is the one-cut entanglement entropy of the flat sea across one planar face with unit
    normal n, per unit area a^2; eta_sphere is its average over the unit sphere;
  * complete elliptic integrals are in the PARAMETER convention and are evaluated through the
    arithmetic-geometric mean with an explicit complementary parameter c = 1 - m,
    K(1 - c) = pi / (2 AGM(1, sqrt(c))), so that parameters near 1 lose no precision.
Importing this module imports numpy and scipy: set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS before importing it (every script here does so from its --threads argument).
"""
import math
import time
import numpy as np
from scipy.special import xlogy, expit
from scipy.linalg import eigvalsh, toeplitz
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

TWO_PI = 2.0 * math.pi
LOG2 = math.log(2.0)

# Clifford matrices in the chiral representation: alpha_i = diag(sigma_i, -sigma_i), beta = offdiag(1, 1).
SIGMA = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
ALPHA4 = np.zeros((3, 4, 4), dtype=complex)
for _i in range(3):
    ALPHA4[_i, :2, :2] = SIGMA[_i]
    ALPHA4[_i, 2:, 2:] = -SIGMA[_i]
BETA4 = np.zeros((4, 4), dtype=complex)
BETA4[:2, 2:] = np.eye(2)
BETA4[2:, :2] = np.eye(2)

# Reference values used by the tests (targets to reproduce, not inputs to the computation).
TARGET_ETA100 = 0.7865833865531381
TARGET_ETA111 = 0.92338180595796
TARGET_B100 = -0.4175180912397777

# The solve lane's J7 directional values (input 8): a cross-check of 22 directions, never an input.
SOLVE_LANE_J7 = {
    (1, 0, 0): 0.78658329, (1, 1, 0): 0.86917979, (1, 1, 1): 0.92338188, (2, 1, 0): 0.88596155,
    (2, 1, 1): 0.95198209, (2, 2, 1): 0.93349039, (3, 1, 0): 0.87335590, (3, 1, 1): 0.93604504,
    (3, 2, 0): 0.88768464, (3, 2, 1): 0.94517150, (3, 2, 2): 0.94912313, (3, 3, 1): 0.92335251,
    (3, 3, 2): 0.93681401, (4, 1, 0): 0.86017279, (4, 1, 1): 0.91620079, (4, 2, 1): 0.93845788,
    (4, 3, 0): 0.88535333, (4, 3, 1): 0.93324066, (4, 3, 2): 0.94910514, (4, 3, 3): 0.94477236,
    (4, 4, 1): 0.91442901, (4, 4, 3): 0.93583892,
}
# Values printed by the web return P23 (idea file, not a source): information only.
P23_INFO = {(1, 1, 0): 0.8691793924069138, (2, 1, 1): 0.9519805871089785, (17, 10, 8): 0.9536558380095366}


# ----------------------------------------------------------------------------------------------
# elementary pieces
# ----------------------------------------------------------------------------------------------
def agm(a, b, rel=1e-16, maxit=64):
    """Arithmetic-geometric mean of a and b, elementwise."""
    a = np.array(a, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    for _ in range(maxit):
        an = 0.5 * (a + b)
        bn = np.sqrt(a * b)
        a, b = an, bn
        if np.all(np.abs(a - b) <= rel * np.abs(a)):
            break
    return 0.5 * (a + b)


def ellipK_compl(c):
    """Complete elliptic integral K(m), parameter m = 1 - c, from its complement c in (0, 1]."""
    return math.pi / (2.0 * agm(1.0, np.sqrt(c)))


def hF(x):
    """h_F(x) = log(1 + e^-x) + x/(1 + e^x): entropy of one single-particle entanglement level x."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-x)) + x * expit(-x)


def binary_entropy_sum(lam):
    """sum_i h_2(lam_i) with lam clipped to [0, 1] (clipping removes only round-off outside)."""
    c = np.clip(np.asarray(lam, dtype=float), 0.0, 1.0)
    return float(np.sum(-xlogy(c, c) - xlogy(1.0 - c, 1.0 - c)))


def next_pow2(n):
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


# ----------------------------------------------------------------------------------------------
# exact chain formulas (double precision; the mpmath versions are in eta_exact.py)
# ----------------------------------------------------------------------------------------------
def eps_axial(M):
    """Ladder spacing of the [100] chain at transverse gap M: pi K(q)/K(t), q = M^2/(1+M^2), t = 1/(1+M^2)."""
    M2 = float(M) ** 2
    t = 1.0 / (1.0 + M2)
    q = M2 / (1.0 + M2)
    # K(q) has complement t, K(t) has complement q; K(q)/K(t) = AGM(1, sqrt q)/AGM(1, sqrt t)
    return math.pi * float(agm(1.0, math.sqrt(q))) / float(agm(1.0, math.sqrt(t)))


def s4_axial(M, ladder=False, cut=45.0):
    """One-cut 4-component entropy of the [100] chain at transverse gap M = sqrt(m^2 + sin^2 ky + sin^2 kz)."""
    M = float(M)
    if M <= 0.0:
        raise ValueError("s4_axial needs M > 0")
    if ladder:
        e = eps_axial(M)
        j = np.arange(int(math.ceil(cut / e)) + 1)
        return 4.0 * float(np.sum(hF((2 * j + 1) * e)))
    s = M * M
    t = 1.0 / (1.0 + s)
    q = s / (1.0 + s)
    Kt = float(ellipK_compl(q))
    Kq = float(ellipK_compl(t))
    return (math.log(16.0 * (1.0 + s) ** 2 / s) + (4.0 / math.pi) * Kt * Kq * (1.0 - s) / (1.0 + s)) / 6.0


def rho100(s):
    """Transverse density of s = sin^2 ky + sin^2 kz on (0, 2): (2/pi^2) K(s(2 - s)); complement (1 - s)^2."""
    s = np.asarray(s, dtype=float)
    return 2.0 / math.pi ** 2 * ellipK_compl(np.maximum((1.0 - s) ** 2, 1e-300))


def eps_111(z):
    """Ladder spacing of the reduced [111] chain: pi K(1 - k^2)/K(k^2), k = (u - v)/(u + v)."""
    u = math.sqrt((3.0 + z) / 2.0)
    v = math.sqrt(max(3.0 - z, 0.0) / 2.0)
    k = (u - v) / (u + v)
    k2c = 4.0 * u * v / (u + v) ** 2          # 1 - k^2 without cancellation
    return math.pi * float(agm(1.0, math.sqrt(k2c))) / float(agm(1.0, k))


def F111(z, cut=45.0):
    """One-cut 4-component entropy of the [111] chain at z = |1 + e^{2i phi} + e^{2i psi}| in [0, 3)."""
    z = float(z)
    if z <= 0.0:
        return 2.0 * LOG2
    if z >= 3.0:
        return math.inf
    e = eps_111(z)
    j = np.arange(1, int(math.ceil(cut / e)) + 1)
    return 2.0 * LOG2 + 4.0 * float(np.sum(hF(j * e)))


def z111_from_phases(c):
    """z for the chain sin(theta + c_1), sin(theta + c_2), sin(theta + c_3): |sum_i e^{2 i c_i}|."""
    c = np.asarray(c, dtype=float)
    return float(abs(np.sum(np.exp(2j * c))))


# ----------------------------------------------------------------------------------------------
# integer geometry of the rational plane
# ----------------------------------------------------------------------------------------------
def xgcd(a, b):
    """g, x, y with x a + y b = g = gcd(|a|, |b|) >= 0."""
    old_r, r = abs(int(a)), abs(int(b))
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    sa = 1 if a >= 0 else -1
    sb = 1 if b >= 0 else -1
    return old_r, old_s * sa, old_t * sb


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def is_primitive(p):
    return math.gcd(math.gcd(abs(int(p[0])), abs(int(p[1]))), abs(int(p[2]))) == 1


def unimodular_rows(p):
    """Integer rows r, s with det(p; r; s) = 1 for a primitive p (extended Euclid, no search)."""
    a, b, c = (int(v) for v in p)
    if not is_primitive((a, b, c)):
        raise ValueError("normal must be primitive")
    g, x, y = xgcd(a, b)                    # x a + y b = g
    if g == 0:                              # a = b = 0, c = +-1
        r = (1, 0, 0)
        s = (0, c, 0)
    else:
        one, A, B = xgcd(g, c)              # A g + B c = 1
        if one != 1:
            raise ArithmeticError("gcd failure")
        r = (-y, x, 0)
        s = (-B * (a // g), -B * (b // g), A)
    if det3((a, b, c), r, s) != 1:
        raise ArithmeticError("integer completion failed")
    return r, s


def gram_int(p, r, s):
    """|p|^2 times the Gram matrix of the reciprocal transverse basis (exact integers)."""
    pp = _dot(p, p)

    def gij(u, v):
        return pp * _dot(u, v) - _dot(u, p) * _dot(v, p)

    return gij(r, r), gij(r, s), gij(s, s)


def reduce_rows(p, r, s):
    """Lagrange-reduce the transverse basis by SL(2, Z) operations that keep det(p; r; s) = 1."""
    p = tuple(int(v) for v in p)
    r = [int(v) for v in r]
    s = [int(v) for v in s]
    for _ in range(100000):
        g11, g12, g22 = gram_int(p, r, s)
        if g11 > g22:
            r, s = s, [-v for v in r]
            continue
        mu = (2 * g12 + g11) // (2 * g11)   # nearest integer to g12/g11
        if mu == 0:
            break
        s = [sv - mu * rv for sv, rv in zip(s, r)]
    g11, g12, g22 = gram_int(p, r, s)
    if not (g11 <= g22 and 2 * abs(g12) <= g11):
        raise ArithmeticError("reduction did not terminate in the reduced domain")
    if det3(p, r, s) != 1:
        raise ArithmeticError("determinant lost in reduction")
    return tuple(r), tuple(s)


def plane_basis(p):
    """Reduced unimodular rows (r, s) and the float Gram matrix G (det G = 1/|p|^2) for primitive p."""
    p = tuple(int(v) for v in p)
    r, s = unimodular_rows(p)
    r, s = reduce_rows(p, r, s)
    g11, g12, g22 = gram_int(p, r, s)
    pp = _dot(p, p)
    G = np.array([[g11, g12], [g12, g22]], dtype=float) / pp
    return r, s, G


def direction_orbits(pmax):
    """Primitive (i, j, k) with pmax >= i >= j >= k >= 0, sorted by |p| then lexicographically."""
    out = []
    for i in range(1, pmax + 1):
        for j in range(i + 1):
            for k in range(j + 1):
                if math.gcd(math.gcd(i, j), k) == 1:
                    out.append((i, j, k))
    out.sort(key=lambda q: (q[0] ** 2 + q[1] ** 2 + q[2] ** 2, q))
    return out


def cubic_images(p):
    """All signed permutations of p (6, 8, 12, 24 or 48 distinct integer vectors)."""
    from itertools import permutations, product
    out = set()
    for perm in set(permutations(tuple(int(v) for v in p))):
        for sg in product((-1, 1), repeat=3):
            out.add(tuple(sg[i] * perm[i] for i in range(3)))
    return sorted(out)


def canonical_orbit(p):
    q = tuple(sorted((abs(int(v)) for v in p), reverse=True))
    return q


# ----------------------------------------------------------------------------------------------
# the chain at one transverse momentum
# ----------------------------------------------------------------------------------------------
def chain_phases(r, s, phi, psi):
    """c_i = r_i phi + s_i psi: the chain symbol is sin(p_i theta + c_i)."""
    return np.array([r[0] * phi + s[0] * psi, r[1] * phi + s[1] * psi, r[2] * phi + s[2] * psi], dtype=float)


def chain_gap(p, r, s, phi, psi, mass):
    """Exact chain gap Delta = min over theta of sqrt(m^2 + sum_i sin^2(p_i theta + c_i))."""
    pv = np.asarray(p, dtype=float)
    c = chain_phases(r, s, phi, psi)
    g = int(np.max(np.abs(pv)))
    n = max(4096, 256 * g)
    th = TWO_PI * np.arange(n) / n
    e2 = mass * mass + np.sum(np.sin(np.outer(th, pv) + c[None, :]) ** 2, axis=1)
    k = int(np.argmin(e2))

    def f(t):
        return mass * mass + float(np.sum(np.sin(pv * t + c) ** 2))

    res = minimize_scalar(f, bounds=(th[k] - TWO_PI / n, th[k] + TWO_PI / n), method='bounded',
                          options={'xatol': 1e-12})
    return math.sqrt(max(min(float(res.fun), float(e2[k])), 0.0))


def hermitian_eigvals(C, device='cpu', gpu_min_n=1024):
    """Eigenvalues of a Hermitian matrix; the GPU path is used for n >= gpu_min_n when device == 'cuda'."""
    n = C.shape[0]
    if device == 'cuda' and n >= gpu_min_n:
        import torch
        t = torch.from_numpy(C).to('cuda')
        lam = torch.linalg.eigvalsh(t).cpu().numpy()
        del t
        return lam
    return eigvalsh(C, overwrite_a=True, check_finite=False, driver='evr')


def slab_entropy(p, r, s, phi, psi, mass, L, ntheta=None, device='cpu', force4=False):
    """One-cut 4-component entropy from the slab of L layers of the infinite chain (2 cuts).

    m = 0 (and not force4): the 2L x 2L Weyl slab; its 2-cut entropy equals the 4-component 1-cut entropy.
    m > 0 or force4: the 4L x 4L slab; half its entropy is returned.
    The deficit from the interaction of the 2 cuts is O(e^{-2 Delta L/|p|}); choose L accordingly.
    """
    pv = np.asarray(p, dtype=float)
    c = chain_phases(r, s, phi, psi)
    L = int(L)
    if ntheta is None:
        ntheta = max(8192, next_pow2(4 * L))
    th = TWO_PI * np.arange(ntheta) / ntheta
    d = np.sin(np.outer(th, pv) + c[None, :])              # (ntheta, 3)
    eps = np.sqrt(mass * mass + np.sum(d * d, axis=1))
    if mass == 0.0 and not force4:
        Q = np.einsum('ki,iab->kab', d / eps[:, None], SIGMA)
        P = 0.5 * (np.eye(2, dtype=complex)[None, :, :] - Q)
        nc = 2
    else:
        H = np.einsum('ki,iab->kab', d, ALPHA4) + mass * BETA4[None, :, :]
        P = 0.5 * (np.eye(4, dtype=complex)[None, :, :] - H / eps[:, None, None])
        nc = 4
    cf = np.fft.ifft(P, axis=0)                              # Fourier coefficients c_n, n = 0..ntheta-1
    col = cf[:L]                                             # c_{j - 0}
    row = cf[(-np.arange(L)) % ntheta]                       # c_{0 - j'}
    C = np.empty((L, nc, L, nc), dtype=complex)
    for a in range(nc):
        for b in range(nc):
            C[:, a, :, b] = toeplitz(col[:, a, b], row[:, a, b])
    C = C.reshape(L * nc, L * nc)
    lam = hermitian_eigvals(C, device=device)
    del C
    S_slab = binary_entropy_sum(lam)
    return S_slab if nc == 2 else 0.5 * S_slab


# ----------------------------------------------------------------------------------------------
# the node model, its exact integral, the transverse grid
# ----------------------------------------------------------------------------------------------
def node_model(G, y, phi, psi):
    """pi-periodic model of the chain gap squared: positive except at the corner lattice, = q^T G q + y + O(q^4)."""
    g11, g12, g22 = G[0, 0], G[0, 1], G[1, 1]
    s1, c1 = np.sin(phi), np.cos(phi)
    s2, c2 = np.sin(psi), np.cos(psi)
    return g11 * s1 * s1 + 2.0 * g12 * s1 * c1 * s2 * c2 + g22 * s2 * s2 + y


def reference_log_integral(G, y):
    """R = (1/pi^2) int_0^pi int_0^pi log(node_model) dphi dpsi: inner integral in closed form, outer by quad."""
    g11, g12, g22 = float(G[0, 0]), float(G[0, 1]), float(G[1, 1])

    def inner(phi):
        s = math.sin(phi)
        c = math.cos(phi)
        a0 = g11 * s * s + y
        alpha = a0 + 0.5 * g22
        disc = a0 * a0 + a0 * g22 - (g12 * s * c) ** 2       # alpha^2 - rho^2 without cancellation
        return math.log(0.5 * (alpha + math.sqrt(max(disc, 0.0))))

    val, err = quad(inner, 0.0, math.pi, limit=400, epsabs=1e-14, epsrel=1e-14)
    return val / math.pi, err / math.pi


def grid_sizes(G, h0, level):
    """Midpoint grid sizes on the pi-zone: physical spacing about h0/2^level in both reduced directions."""
    N1 = max(1, int(math.ceil(math.pi * math.sqrt(G[0, 0]) / h0))) << level
    N2 = max(1, int(math.ceil(math.pi * math.sqrt(G[1, 1]) / h0))) << level
    return N1, N2


def half_grid(N1, N2):
    """Grid points with their inversion partners removed: list of (i, j, weight); weights sum to N1 N2."""
    pts = []
    for i in range(N1):
        for j in range(N2):
            ip, jp = N1 - 1 - i, N2 - 1 - j
            if (i, j) < (ip, jp):
                pts.append((i, j, 2))
            elif (i, j) == (ip, jp):
                pts.append((i, j, 1))
    return pts


def grid_arrays(N1, N2):
    pts = half_grid(N1, N2)
    phis = np.array([math.pi * (i + 0.5) / N1 for i, j, w in pts])
    psis = np.array([math.pi * (j + 0.5) / N2 for i, j, w in pts])
    wts = np.array([float(w) for i, j, w in pts])
    return phis, psis, wts


# ----------------------------------------------------------------------------------------------
# cost model (calibrate T_EIG_1000 on the machine)
# ----------------------------------------------------------------------------------------------
T_EIG_1000 = 1.5     # seconds per complex Hermitian eigenvalue problem of size 1000, 1 thread


def eig_seconds(n, device='cpu'):
    n = np.asarray(n, dtype=float)
    if device == 'cuda':
        return np.where(n >= 1024, 0.3 + 3.0 * (n / 8000.0) ** 3, T_EIG_1000 * (n / 1000.0) ** 3 + 2e-3)
    return T_EIG_1000 * (n / 1000.0) ** 3 + 2e-3


def estimate_case(p, mass, level, h0, cL, Lmin, device='cpu'):
    """Model cost of one (p, mass, level) case from the grid and the node-model gaps."""
    r, s, G = plane_basis(p)
    pn = math.sqrt(sum(v * v for v in p))
    y = mass * mass
    N1, N2 = grid_sizes(G, h0, level)
    phis, psis, wts = grid_arrays(N1, N2)
    D = node_model(G, y, phis, psis)
    Delta = np.sqrt(np.maximum(D, 1e-300))
    L = Lmin + np.ceil(cL * pn / Delta).astype(int)
    nc = 2 if mass == 0.0 else 4
    n = nc * L
    secs = float(np.sum(eig_seconds(n, device))) + 0.004 * len(phis)
    return dict(N1=N1, N2=N2, n_points=int(len(phis)), n_max=int(n.max()), seconds=secs,
                mem_gb=float(3.0 * 16.0 * float(n.max()) ** 2 / 1e9))


def now():
    return time.strftime('%H:%M:%S')
