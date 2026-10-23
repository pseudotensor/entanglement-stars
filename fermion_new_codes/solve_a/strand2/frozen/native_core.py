"""Infrastructure for the frozen native source action test (integ65 section 4).

Independent of the open source definitions: lattices with twisted Dirac stencils, the isolated exterior
stiffness K, fixed-number Fock sectors, a small operator-expression algebra with analytic
(Daleckii-Krein) derivatives and adjoints, the operator bridge
    C_x(S) = e_x(S) - k L_{S_x}([K(S - b)]_x),   L_A(X) = (AX + XA)/2,   Q_b = k 1^T K (S - b),
its Newton solution, the adjoint multiplier J^* Lambda = -k b_vec (x) rho (b_vec = K 1), the sigma
family C_sigma = e - k L_S K(S - sigma 1), Q_sigma = k sigma 1^T K (S - sigma 1), R = d_sigma H_sigma,
the Ward residual sum_x Tr Lambda_x (i[D, e_x(S)] - De_x(S)[i[D, S]]) and the energy-block test P_E R P_E.
"""
import itertools

import numpy as np
from scipy import integrate, special

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ------------------------------------------------------------ exterior stiffness
def G_inf(r):
    """Isolated cubic-lattice Green function int cos(theta.r)/(2 sum(1 - cos theta_j)) d^3theta/(2pi)^3,
    via G(r) = int_0^inf prod_j ive(|r_j|, 2t) dt."""
    r = [abs(int(v)) for v in r]
    f = lambda t: float(np.prod([special.ive(n, 2.0 * t) for n in r]))
    a, _ = integrate.quad(f, 0.0, 40.0, limit=400, epsabs=1e-15, epsrel=1e-14)
    b, _ = integrate.quad(f, 40.0, 4000.0, limit=400, epsabs=1e-15, epsrel=1e-14)
    # tail beyond T: prod ive ~ (4 pi t)^-3/2 (1 - sum(4 n^2 - 1)/(16 t)), integrated analytically
    T = 4000.0
    c = sum(4 * n * n - 1 for n in r) / 16.0
    tail = (4 * np.pi) ** -1.5 * (2 * T ** -0.5 - c * (2.0 / 3.0) * T ** -1.5)
    return a + b + tail


def K_matrix(positions):
    """K = (G_inf restricted to the sites)^-1, the isolated Schur complement of the exterior."""
    n = len(positions)
    cache = {}
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            key = tuple(sorted(abs(a - b) for a, b in zip(positions[i], positions[j])))
            if key not in cache:
                cache[key] = G_inf(key)
            G[i, j] = cache[key]
    return np.linalg.inv(G), G


# ------------------------------------------------------------------- lattices
def lattice(name, theta=(0.0, 0.0, 0.0), ncomp=2):
    """Positions (for K), one-body D(theta) (uniform-gauge twist theta_j / L_j per link), beta, mode->site."""
    if name == "ring4":
        positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        nb = {x: [((x + 1) % 4, 0, +1), ((x - 1) % 4, 0, -1)] for x in range(4)}
        alphas, beta, Ls = [SX], SZ, [4]
    elif name == "plaq":
        positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        idx = lambda x, y: (x % 2) + 2 * (y % 2)
        nb = {}
        for y in range(2):
            for x in range(2):
                nb[idx(x, y)] = [(idx(x + 1, y), 0, +1), (idx(x - 1, y), 0, -1),
                                 (idx(x, y + 1), 1, +1), (idx(x, y - 1), 1, -1)]
        alphas, beta, Ls = [SX, SY], SZ, [2, 2]
    elif name == "cube":
        positions = [(x0, x1, x2) for x2 in (0, 1) for x1 in (0, 1) for x0 in (0, 1)]
        pidx = {p: i for i, p in enumerate(positions)}
        nb = {}
        for p in positions:
            lst = []
            for j in range(3):
                for sgn in (1, -1):
                    q = list(p)
                    q[j] += sgn
                    if tuple(q) in pidx:
                        lst.append((pidx[tuple(q)], j, sgn))
            nb[pidx[p]] = lst
        alphas, beta, Ls = [SX, SY, SZ], None, None
    else:
        raise ValueError(name)
    n = len(positions)
    D = np.zeros((n * ncomp, n * ncomp), dtype=complex)
    for x in range(n):
        for y, j, sgn in nb[x]:
            ph = np.exp(1j * sgn * theta[j] / Ls[j]) if Ls is not None else 1.0
            D[x * ncomp:(x + 1) * ncomp, y * ncomp:(y + 1) * ncomp] += sgn * ph * alphas[j] / 2j
    B = None if beta is None else np.kron(np.eye(n), beta)
    return {"name": name, "positions": positions, "n": n, "ncomp": ncomp, "D": D, "beta": B,
            "site_of_mode": np.repeat(np.arange(n), ncomp), "L": Ls}


# ---------------------------------------------------------------- Fock sectors
class Sector:
    """Fixed-number sector of nmodes fermionic modes; basis = sorted occupied tuples."""

    def __init__(self, nmodes, N):
        self.nmodes, self.N = nmodes, N
        self.basis = list(itertools.combinations(range(nmodes), N))
        self.index = {b: i for i, b in enumerate(self.basis)}
        self.d = len(self.basis)
        self._hop = {}

    def hop(self, i, j):
        """Matrix of c_i^dag c_j in the sector."""
        key = (i, j)
        if key in self._hop:
            return self._hop[key]
        M = np.zeros((self.d, self.d))
        for col, occ in enumerate(self.basis):
            if j not in occ:
                continue
            rest = [o for o in occ if o != j]
            if i in rest:
                continue
            sign = (-1) ** sum(1 for o in rest if min(i, j) < o < max(i, j))
            new = tuple(sorted(rest + [i]))
            M[self.index[new], col] = sign
        self._hop[key] = M
        return M

    def bilinear(self, h):
        """sum_ij h_ij c_i^dag c_j."""
        out = np.zeros((self.d, self.d), dtype=complex)
        for i, j in zip(*np.nonzero(np.abs(h) > 0)):
            out += h[i, j] * self.hop(i, j)
        return out

    def number(self, i):
        return np.diag([float(i in occ) for occ in self.basis])

    def occupations(self):
        return np.array([[float(i in occ) for i in range(self.nmodes)] for occ in self.basis])


# ------------------------------------------------- Hermitian parameterization
def herm_basis(d):
    """Orthonormal (under Re Tr) real basis of d x d Hermitian matrices, in herm_to_vec coordinate order."""
    return [vec_to_herm(v, d) for v in np.eye(d * d)]


def herm_to_vec(X):
    d = X.shape[0]
    iu = np.triu_indices(d, 1)
    return np.concatenate([np.real(np.diag(X)), np.sqrt(2) * np.real(X[iu]), -np.sqrt(2) * np.imag(X[iu])])


def vec_to_herm(v, d):
    iu = np.triu_indices(d, 1)
    m = len(iu[0])
    X = np.diag(v[:d]).astype(complex)
    X[iu] = (v[d:d + m] - 1j * v[d + m:]) / np.sqrt(2)
    X[(iu[1], iu[0])] = np.conj(X[iu])
    return X


def pack(S):
    return np.concatenate([herm_to_vec(s) for s in S])


def unpack(v, n, d):
    return [vec_to_herm(v[x * d * d:(x + 1) * d * d], d) for x in range(n)]


def herm(A):
    return 0.5 * (A + A.conj().T)


# --------------------------------------------- operator expressions with derivatives
class MatFun:
    """Cached spectral data of the rulers for powers S_y^p and their Daleckii-Krein derivatives."""

    def __init__(self, S):
        self.S = S
        self.eig = [np.linalg.eigh(s) for s in S]
        if min(w.min() for w, _ in self.eig) <= 0:
            raise ValueError("ruler not positive")
        self._pow = {}

    def power(self, y, p):
        key = (y, p)
        if key not in self._pow:
            w, V = self.eig[y]
            self._pow[key] = (V * w ** p) @ V.conj().T
        return self._pow[key]

    def dd(self, y, p):
        """Divided-difference matrix of t -> t^p at the eigenvalues of S_y."""
        w, _ = self.eig[y]
        Wi, Wj = np.meshgrid(w, w, indexing='ij')
        diff = Wi - Wj
        close = np.abs(diff) < 1e-10 * np.maximum(1.0, np.abs(Wi))
        num = Wi ** p - Wj ** p
        der = p * (0.5 * (Wi + Wj)) ** (p - 1)
        return np.where(close, der, num / np.where(close, 1.0, diff))

    def dpower(self, y, p, X):
        w, V = self.eig[y]
        return V @ (self.dd(y, p) * (V.conj().T @ X @ V)) @ V.conj().T


class Expr:
    """Sum of terms coef * F_1 F_2 ... with F = ('S', y, p) (ruler power S_y^p) or ('M', matrix)."""

    def __init__(self, terms=None):
        self.terms = terms or []

    def add(self, coef, *factors):
        self.terms.append((coef, list(factors)))
        return self

    def _factor(self, mf, f):
        return mf.power(f[1], f[2]) if f[0] == 'S' else f[1]

    def value(self, mf):
        d = mf.S[0].shape[0]
        out = np.zeros((d, d), dtype=complex)
        for coef, fs in self.terms:
            P = np.eye(d, dtype=complex)
            for f in fs:
                P = P @ self._factor(mf, f)
            out += coef * P
        return out

    def deriv(self, mf, X):
        """De[X] for a direction X = list of Hermitian matrices (one per ruler)."""
        d = mf.S[0].shape[0]
        out = np.zeros((d, d), dtype=complex)
        for coef, fs in self.terms:
            mats = [self._factor(mf, f) for f in fs]
            for i, f in enumerate(fs):
                if f[0] != 'S':
                    continue
                L = np.eye(d, dtype=complex)
                for A in mats[:i]:
                    L = L @ A
                Rm = np.eye(d, dtype=complex)
                for A in mats[i + 1:]:
                    Rm = Rm @ A
                out += coef * L @ mf.dpower(f[1], f[2], X[f[1]]) @ Rm
        return out

    def adjoint(self, mf, Lam, n):
        """De^*[Lam] per ruler under Re Tr pairing: Re Tr(Lam De[X]) = sum_y Re Tr(De^*_y[Lam] X_y)."""
        d = mf.S[0].shape[0]
        out = [np.zeros((d, d), dtype=complex) for _ in range(n)]
        for coef, fs in self.terms:
            mats = [self._factor(mf, f) for f in fs]
            for i, f in enumerate(fs):
                if f[0] != 'S':
                    continue
                Z = np.eye(d, dtype=complex)
                for A in mats[i + 1:]:
                    Z = Z @ A
                Z = Z @ Lam
                for A in mats[:i]:
                    Z = Z @ A
                # Tr(Lam L Df[X] R) = Tr(Z Df[X]) = Tr(Df[Z] X) with Z = R Lam L (real symmetric divided differences)
                out[f[1]] += coef * mf.dpower(f[1], f[2], Z)
        return [herm(o) for o in out]


# ------------------------------------------------------------------- bridge
def jordan(A, X):
    return 0.5 * (A @ X + X @ A)


class Bridge:
    """C_x(S) = e_x(S) - k L_{S_x}([K(S - b)]_x) on a sector of dimension d with n rulers."""

    def __init__(self, exprs, K, k, d):
        self.e, self.K, self.k, self.n, self.d = exprs, np.asarray(K, float), float(k), len(exprs), d
        self.I = np.eye(d, dtype=complex)

    def w(self, S, b):
        return [sum(self.K[x, y] * (S[y] - b * self.I) for y in range(self.n)) for x in range(self.n)]

    def C(self, S, b=1.0, mf=None):
        mf = mf or MatFun(S)
        w = self.w(S, b)
        return [self.e[x].value(mf) - self.k * jordan(S[x], w[x]) for x in range(self.n)]

    def J(self, S, X, b=1.0, mf=None):
        mf = mf or MatFun(S)
        w = self.w(S, b)
        KX = [sum(self.K[x, y] * X[y] for y in range(self.n)) for x in range(self.n)]
        return [self.e[x].deriv(mf, X) - self.k * jordan(w[x], X[x]) - self.k * jordan(S[x], KX[x])
                for x in range(self.n)]

    def Jadj(self, S, Lam, b=1.0, mf=None):
        """(J^* Lam)_x = (De^* Lam)_x - k L_{w_x}(Lam_x) - k (K Y)_x, Y_x = L_{S_x}(Lam_x)."""
        mf = mf or MatFun(S)
        w = self.w(S, b)
        out = [np.zeros((self.d, self.d), dtype=complex) for _ in range(self.n)]
        for x in range(self.n):
            adj = self.e[x].adjoint(mf, Lam[x], self.n)
            for y in range(self.n):
                out[y] += adj[y]
        Y = [jordan(S[x], Lam[x]) for x in range(self.n)]
        return [out[x] - self.k * jordan(w[x], Lam[x]) - self.k * sum(self.K[x, y] * Y[y] for y in range(self.n))
                for x in range(self.n)]

    def J_dense(self, S, b=1.0, adjoint=False):
        mf = MatFun(S)
        nd = self.n * self.d * self.d
        B = herm_basis(self.d)
        cols = []
        for x in range(self.n):
            for E in B:
                X = [np.zeros((self.d, self.d), dtype=complex) for _ in range(self.n)]
                X[x] = E
                cols.append(pack([herm(c) for c in (self.Jadj(S, X, b, mf) if adjoint else self.J(S, X, b, mf))]))
        return np.array(cols).T.reshape(nd, nd)

    def solve(self, b=1.0, S0=None, tol=1e-13, maxit=40):
        S = [s.copy() for s in S0] if S0 is not None else [self.I.copy() for _ in range(self.n)]
        hist = []
        for _ in range(maxit):
            Cv = pack([herm(c) for c in self.C(S, b)])
            hist.append(float(np.linalg.norm(Cv)))
            if hist[-1] < tol:
                break
            Jm = self.J_dense(S, b)
            dv = np.linalg.solve(Jm, -Cv)
            step = 1.0
            while True:
                Sn = [herm(s + step * dx) for s, dx in zip(S, unpack(dv, self.n, self.d))]
                if min(np.linalg.eigvalsh(s).min() for s in Sn) > 0 or step < 1e-6:
                    break
                step *= 0.5
            S = Sn
        Jm = self.J_dense(S, b)
        sv = np.linalg.svd(Jm, compute_uv=False)
        return S, {"residuals": hist, "final_residual": hist[-1], "J_sigma_min": float(sv[-1]),
                   "J_cond": float(sv[0] / sv[-1]), "min_eig_S": float(min(np.linalg.eigvalsh(s).min() for s in S))}

    def Q(self, S, b=1.0):
        return self.k * sum(self.K[x, y] * (S[y] - b * self.I) for x in range(self.n) for y in range(self.n))


class SigmaBridge(Bridge):
    """C_sigma(S) = e(S) - k L_S K(S - sigma 1), Q_sigma = k sigma 1^T K (S - sigma 1)  (FRESH1136 eq. 18)."""

    def H_sigma(self, sigma, S0=None):
        S, info = self.solve(b=sigma, S0=S0)
        return sigma * self.Q(S, b=sigma), S, info


def solve_krylov(bridge, b=1.0, S0=None, tol=1e-12, maxit=30, gmres_tol=1e-13):
    """Newton-GMRES for large sectors (matrix-free J); returns S, info."""
    from scipy.sparse.linalg import LinearOperator, gmres
    n, d = bridge.n, bridge.d
    S = [s.copy() for s in S0] if S0 is not None else [bridge.I.copy() for _ in range(n)]
    hist, kry = [], []
    for _ in range(maxit):
        Cv = pack([herm(c) for c in bridge.C(S, b)])
        hist.append(float(np.linalg.norm(Cv)))
        if hist[-1] < tol:
            break
        mf = MatFun(S)
        op = LinearOperator((n * d * d, n * d * d), dtype=float,
                            matvec=lambda v: pack([herm(c) for c in bridge.J(S, unpack(v, n, d), b, mf)]))
        dv, code = gmres(op, -Cv, rtol=gmres_tol, atol=0.0, restart=200, maxiter=2000)
        kry.append(int(code))
        S = [herm(s + dx) for s, dx in zip(S, unpack(dv, n, d))]
    return S, {"residuals": hist, "final_residual": hist[-1], "gmres_codes": kry,
               "min_eig_S": float(min(np.linalg.eigvalsh(s).min() for s in S))}


def adjoint_krylov(bridge, S, rho, b=1.0, gmres_tol=1e-14):
    """Matrix-free J^* Lambda = -k b_vec (x) rho."""
    from scipy.sparse.linalg import LinearOperator, gmres
    n, d = bridge.n, bridge.d
    mf = MatFun(S)
    bvec = bridge.K @ np.ones(n)
    rhs = pack([herm(-bridge.k * bvec[x] * rho) for x in range(n)])
    op = LinearOperator((n * d * d, n * d * d), dtype=float,
                        matvec=lambda v: pack([herm(c) for c in bridge.Jadj(S, unpack(v, n, d), b, mf)]))
    lam, code = gmres(op, rhs, rtol=gmres_tol, atol=0.0, restart=200, maxiter=4000)
    Lam = unpack(lam, n, d)
    resid = float(np.linalg.norm(pack([herm(c) for c in bridge.Jadj(S, Lam, b, mf)]) - rhs))
    return Lam, resid, int(code)


def sigma_min_estimate(bridge, S, b=1.0, iters=8, seed=0):
    """Smallest singular value of J by inverse iteration on J^T J with GMRES solves (matrix-free)."""
    from scipy.sparse.linalg import LinearOperator, gmres
    n, d = bridge.n, bridge.d
    mf = MatFun(S)
    N = n * d * d
    J = LinearOperator((N, N), dtype=float, matvec=lambda v: pack([herm(c) for c in bridge.J(S, unpack(v, n, d), b, mf)]))
    Jt = LinearOperator((N, N), dtype=float, matvec=lambda v: pack([herm(c) for c in bridge.Jadj(S, unpack(v, n, d), b, mf)]))
    from scipy.sparse.linalg import eigsh

    def inv_jtj(v):
        y, _ = gmres(Jt, v, rtol=1e-12, atol=0.0, restart=200, maxiter=4000)
        x, _ = gmres(J, y, rtol=1e-12, atol=0.0, restart=200, maxiter=4000)
        return x

    A = LinearOperator((N, N), dtype=float, matvec=inv_jtj)
    v0 = np.random.default_rng(seed).normal(size=N)
    lam = eigsh(A, k=1, which='LA', v0=v0, tol=1e-10, maxiter=200 * iters, return_eigenvectors=False)
    return float(1.0 / np.sqrt(lam[0]))


def dispersion_fit(P, E, order=8):
    """Fit E(P) = E0 + P^2/(2 I) + c4 P^4 + ... (even powers up to order) by least squares; returns E0, I, c4."""
    P, E = np.asarray(P, float), np.asarray(E, float)
    A = np.stack([P ** p for p in range(0, order + 1, 2)], axis=1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return float(coef[0]), float(1.0 / (2.0 * coef[1])), float(coef[2])


def self_energy_coefficients(k_values, P, E_of_kP, E_free_P):
    """F_0, F_2 from k (E_k(P) - E_free(P)) = F_0 + F_2 P^2 + ..., using the largest k (and the k-trend)."""
    out = []
    for k, E in zip(k_values, E_of_kP):
        y = k * (np.asarray(E) - np.asarray(E_free_P))
        A = np.stack([np.ones_like(P), np.asarray(P) ** 2, np.asarray(P) ** 4], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append({"k": float(k), "F0": float(coef[0]), "F2": float(coef[1])})
    return out


def adjoint_multiplier(bridge, S, rho, b=1.0):
    """Solve J^* Lambda = -k b_vec (x) rho, b_vec = K 1 (dense)."""
    bvec = bridge.K @ np.ones(bridge.n)
    rhs = pack([herm(-bridge.k * bvec[x] * rho) for x in range(bridge.n)])
    Ja = bridge.J_dense(S, b, adjoint=True)
    lam = np.linalg.solve(Ja, rhs)
    Lam = unpack(lam, bridge.n, bridge.d)
    resid = float(np.linalg.norm(pack([herm(c) for c in bridge.Jadj(S, Lam, b)]) - rhs))
    return Lam, resid


def ward_residual(bridge, S, Lam, Dgen):
    """sum_x Tr Lam_x (i[D, e_x(S)] - De_x(S)[i[D, S]]) and the same without the feedback term."""
    mf = MatFun(S)
    X = [1j * (Dgen @ s - s @ Dgen) for s in S]
    full, nofb = 0.0, 0.0
    for x in range(bridge.n):
        ex = bridge.e[x].value(mf)
        comm = 1j * (Dgen @ ex - ex @ Dgen)
        fb = bridge.e[x].deriv(mf, X)
        full += np.trace(Lam[x] @ (comm - fb))
        nofb += np.trace(Lam[x] @ comm)
    return complex(full), complex(nofb)


def energy_blocks(H, tol=1e-9):
    w, V = np.linalg.eigh(herm(H))
    blocks, cur = [], [0]
    for j in range(1, len(w)):
        if w[j] - w[cur[-1]] < tol:
            cur.append(j)
        else:
            blocks.append(cur)
            cur = [j]
    blocks.append(cur)
    return w, V, blocks


def mass_lock_test(H, R, tol=1e-9):
    """max over energy blocks of the entries of P_E R P_E (including degenerate off-diagonal entries)."""
    w, V, blocks = energy_blocks(H, tol)
    worst, per = 0.0, []
    Rt = V.conj().T @ herm(R) @ V
    for bl in blocks:
        blk = Rt[np.ix_(bl, bl)]
        per.append({"E": float(w[bl[0]]), "deg": len(bl), "max_abs": float(np.abs(blk).max())})
        worst = max(worst, float(np.abs(blk).max()))
    return worst, per
