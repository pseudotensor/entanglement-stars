"""items34_lib.py (busi, reviewb5 items 3, 4, 9; ReviewB_n12chk_v1.md sec. 5).

Library for the projected-model background (item 3) and its finite-amplitude admissibility (item 4) on N3's ball parent.
Definitions (see common/items34_interface.txt):
  parent  : ball R_b = 8 a of the cubic lattice, no hopping across its surface (n3_background_v1.BallParent), a = 1.
  kernel  : the engine's point Gaussian G_ell(z) = (2 pi ell^2)^(-3/2) exp(-|z|^2 / 2 ell^2), ell = 2 (N3's kernel).
  tests   : rho_ell(y_mu), J_ell,i(y_mu) at y_mu = d n, |y_mu| <= R_b; vertices A = 1/2{diag G(y_mu - x), h_0}, 1/2{diag G, Q_i}.
  laws    : rho_g = [(tr S)^2 - tr S^2 - b tr(A^T A)] / (16 pi G N^2),  J_g,j = -(1/8 pi G) sum_i d_i[T_ij / N],
            S = sym grad v, A = antisym grad v, (grad v)_ij = d_i v_j, T = S - (tr S) I + b A.  (Eq. dm-density, c = 0, a1 = -1, a2 = 1.)
  trial   : delta u = sum_j c_j e_a psi(y - z_j), psi(z) = (1 - |z|^2/w^2)_+^5, a in {N, vx, vy, vz}, z_j = the test points.
Rounding: U = 2^-53; gamma(k) = k U / (1 - k U) (Higham's gamma_k). Every 'enclosure' below adds explicit rounding allowances."""
import numpy as np
import scipy.sparse as sp
import sympy as S

U_RND = 2.0 ** -53


def gam(k):
    k = float(k)
    return k * U_RND / (1.0 - k * U_RND)


# ============================================================================ geometry, kernel, tests
def test_points(d, Rb):
    n = int(np.floor(Rb / d + 1e-12))
    g = np.arange(-n, n + 1) * float(d)
    Y = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return Y[np.sqrt((Y ** 2).sum(1)) <= Rb + 1e-9]


def gauss3(Z, ell):
    r2 = (np.asarray(Z, float) ** 2).sum(-1)
    return (2 * np.pi * ell ** 2) ** -1.5 * np.exp(-r2 / (2 * ell ** 2))


def test_weights(Y, X, ell):
    """(n_pts, n_sites): G_ell(y_mu - x)."""
    return gauss3(Y[:, None, :] - X[None, :, :], ell)


def vertex_list(P, Wt):
    """vertices in test order mu = 4 q + tau, tau = 0 energy, 1..3 momentum components (sparse csr, Hermitian)."""
    V = []
    for q in range(Wt.shape[0]):
        D = sp.diags(np.repeat(Wt[q], 4))
        V.append((0.5 * (D @ P.h + P.h @ D)).tocsr())
        for i in range(3):
            V.append((0.5 * (D @ P.Q[i] + P.Q[i] @ D)).tocsr())
    return V


def pattern_matrix(V, n):
    """the union sparsity pattern of the vertices and the (nnz_pat x n_T) matrix of their entries on it."""
    pat = V[0].copy().astype(np.complex128)
    pat.data[:] = 1.0
    for A in V[1:]:
        B = A.copy(); B.data[:] = 1.0
        pat = pat + B
    pat = pat.tocsr(); pat.sum_duplicates(); pat.sort_indices()
    rows, cols = pat.nonzero()
    key = rows.astype(np.int64) * n + cols
    order = np.argsort(key); key = key[order]
    rows, cols = rows[order], cols[order]
    ri, ci, vals = [], [], []
    for mu, A in enumerate(V):
        Ac = A.tocoo()
        k = Ac.row.astype(np.int64) * n + Ac.col
        pos = np.searchsorted(key, k)
        assert np.all(key[pos] == k)
        ri.append(pos); ci.append(np.full(len(pos), mu)); vals.append(Ac.data)
    Vp = sp.csr_matrix((np.concatenate(vals), (np.concatenate(ri), np.concatenate(ci))), shape=(len(key), len(V)))
    return rows, cols, Vp


def gram(Vp):
    """Gamma_mu,nu = Tr(A_mu A_nu) = sum_ab A_mu[a,b] conj(A_nu[a,b]) (Hermitian A): real symmetric; plus the entrywise rounding radius
    gamma(k) (|Vp|^T |Vp|), k the largest number of nonzero terms in a column pair (bounded by the column nnz)."""
    G = (Vp.conj().T @ Vp).toarray()
    Ab = abs(Vp)
    Gabs = (Ab.T @ Ab).toarray()
    kmax = int(np.diff(Vp.tocsc().indptr).max()) + 2
    rad = gam(2 * kmax) * Gabs.real + np.abs(G.imag)          # the imaginary part is rounding; its size is moved into the radius
    return np.real(G), rad


def op_from_coef(rows, cols, Vp, y, n):
    """sum_mu y_mu A_mu as csr (y real or complex)."""
    vals = Vp @ y
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def inf_norm(A):
    """max absolute row sum of a sparse/dense matrix (an upper bound of ||A||_2 for Hermitian A), with rounding allowance."""
    if sp.issparse(A):
        A = A.tocsr()
        rs = np.asarray(abs(A).sum(axis=1)).ravel()
        k = int(np.diff(A.indptr).max()) if A.nnz else 1
    else:
        rs = np.abs(A).sum(axis=1); k = A.shape[1]
    return float(rs.max()) * (1.0 + gam(k + 2))


# ============================================================================ the variation basis: psi and its Gaussian back-average
def psi_jets(Z, w):
    """psi(z) = (1 - s)^5, s = |z|^2/w^2 (0 for s >= 1): value (n,), gradient (n,3), Hessian (n,3,3)."""
    Z = np.asarray(Z, float)
    s = (Z ** 2).sum(-1) / w ** 2
    inside = s < 1.0
    t = np.where(inside, 1.0 - s, 0.0)
    val = t ** 5
    grad = (-10.0 / w ** 2) * (t ** 4)[..., None] * Z
    hess = (-10.0 / w ** 2) * (t ** 4)[..., None, None] * np.eye(3) + (80.0 / w ** 4) * (t ** 3)[..., None, None] * Z[..., :, None] * Z[..., None, :]
    return val, grad, hess


def psi_backavg(dist, w, ell, kernels, nq=96):
    """(G_ell * psi)(s) at the distances s: int_0^w 4 pi R^2 psi(R) K_0(s, R) dR, Gauss-Legendre on [0, w] (psi a polynomial in R there)."""
    xq, wq = np.polynomial.legendre.leggauss(nq)
    R = 0.5 * w * (xq + 1.0); wR = 0.5 * w * wq
    ps = (1.0 - R ** 2 / w ** 2) ** 5
    flat = np.asarray(dist, float).ravel()
    out = np.empty_like(flat)
    ch = 200000
    for i in range(0, len(flat), ch):
        K0, _ = kernels(flat[i:i + ch], R, ell)
        out[i:i + ch] = K0 @ (4 * np.pi * R ** 2 * ps * wR)
    return out.reshape(np.shape(dist))


# ============================================================================ the laws as functions of the jets (sympy)
JET_NAMES = (["N"] + ["dN%d" % i for i in range(3)] + ["dv%d%d" % (i, j) for i in range(3) for j in range(3)]
             + ["ddv%d%d%d" % (k, i, j) for k in range(3) for i in range(k, 3) for j in range(3)])
NJ = len(JET_NAMES)   # 1 + 3 + 9 + 18 = 31


def jet_index(name):
    return JET_NAMES.index(name)


def ddv_name(k, i, j):
    a, b = (k, i) if k <= i else (i, k)
    return "ddv%d%d%d" % (a, b, j)


def law_exprs(bsym=None):
    """sympy expressions (without the constants 1/(16 pi G), -1/(8 pi G)): f_rho, f_J[3] in the jet symbols and b."""
    sy = {n: S.Symbol(n, real=True) for n in JET_NAMES}
    b = S.Symbol("b", real=True) if bsym is None else bsym
    N = sy["N"]
    dN = [sy["dN%d" % i] for i in range(3)]
    gv = S.Matrix(3, 3, lambda i, j: sy["dv%d%d" % (i, j)])
    Sm = (gv + gv.T) / 2; Am = (gv - gv.T) / 2
    f_rho = (Sm.trace() ** 2 - (Sm * Sm).trace() - b * (Am.T * Am).trace()) / N ** 2
    dd = lambda k, i, j: sy[ddv_name(k, i, j)]      # d_k d_i v_j
    T = Sm - Sm.trace() * S.eye(3) + b * Am
    f_J = []
    for j in range(3):
        acc = 0
        for i in range(3):
            # d_i T_ij
            dS = (dd(i, i, j) + dd(i, j, i)) / 2
            dtr = sum(dd(i, k, k) for k in range(3)) if i == j else 0
            dA = (dd(i, i, j) - dd(i, j, i)) / 2
            dT = dS - dtr + b * dA
            acc += dT / N - T[i, j] * dN[i] / N ** 2
        f_J.append(acc)
    return sy, b, f_rho, f_J


class Laws:
    """numerical laws, gradients (numpy) and interval Hessians (mpmath iv) at given b."""

    def __init__(self, b, G):
        self.bval = float(b); self.G = float(G)
        sy, bs, f_rho, f_J = law_exprs()
        syms = [sy[n] for n in JET_NAMES]
        fs = [f_rho] + f_J
        fs = [f.subs(bs, S.Rational(int(round(b * 2)), 2)) for f in fs]
        self.fs = fs; self.syms = syms
        self.f_num = [S.lambdify(syms, f, "numpy") for f in fs]
        self.g_num = [S.lambdify(syms, [S.diff(f, s) for s in syms], "numpy") for f in fs]
        self.used = [[k for k, s in enumerate(syms) if f.has(s)] for f in fs]
        self.h_iv = []
        for f, us in zip(fs, self.used):
            us_s = [syms[k] for k in us]
            H = [[S.diff(f, a, c) for c in us_s] for a in us_s]
            from mpmath import iv as _iv
            self.h_iv.append(S.lambdify(us_s, H, [{"mpf": _iv.mpf, "pi": _iv.pi}, "mpmath"]))
        self.const = [1.0 / (16 * np.pi * self.G)] + [-1.0 / (8 * np.pi * self.G)] * 3

    def values(self, J):
        """J: (n_pts, NJ) -> (n_pts, 4) the law values rho_g, J_g,1..3 (constants included)."""
        cols = [J[:, k] for k in range(NJ)]
        return np.stack([c * np.asarray(f(*cols), float) * np.ones(J.shape[0]) for c, f in zip(self.const, self.f_num)], axis=1)

    def grads(self, J):
        """(n_pts, 4, NJ)."""
        cols = [J[:, k] for k in range(NJ)]
        out = np.zeros((J.shape[0], 4, NJ))
        for t, (c, g) in enumerate(zip(self.const, self.g_num)):
            gg = g(*cols)
            for k in range(NJ):
                out[:, t, k] = c * np.asarray(gg[k], float) * np.ones(J.shape[0])
        return out

    def remainder_bound(self, jet0, beta, tau):
        """rigorous bound of |f(jet0 + delta) - f(jet0) - grad f . delta| over |delta_k| <= beta_k, law tau (constant included):
        1/2 sum_ab sup|H_ab| beta_a beta_b, the Hessian enclosed by interval arithmetic on the box jet0 + [-beta, beta]."""
        from mpmath import iv, mpf
        us = self.used[tau]
        box = [iv.mpf([mpf(float(jet0[k])) - mpf(float(beta[k])), mpf(float(jet0[k])) + mpf(float(beta[k]))]) for k in us]
        H = self.h_iv[tau](*box)
        tot = iv.mpf(0)
        for a in range(len(us)):
            for c in range(len(us)):
                h = H[a][c]
                if not isinstance(h, type(iv.mpf(0))):
                    h = iv.mpf(h)
                mag = max(abs(h.a), abs(h.b))
                tot += mag * iv.mpf(float(beta[us[a]])) * iv.mpf(float(beta[us[c]]))
        cst = iv.mpf(1) / (16 * iv.pi * iv.mpf(self.G)) if tau == 0 else iv.mpf(1) / (8 * iv.pi * iv.mpf(self.G))
        val = tot * cst / 2
        return float(val.b)


# ============================================================================ jets of the radial background (finite differences)
def radial_field_fns(info):
    Nf = lambda r: np.asarray(info["N_dense"](np.abs(r)), float)
    vf = info["v_dense"]
    return Nf, vf


def background_jets(Y, info, h=0.05):
    """jets of N0(y) = N(|y|), v0(y) = v(|y|) y/|y| at the points Y by 4th-order central differences (step h) in 3D."""
    Nf, vf = radial_field_fns(info)

    def N3(P):
        return Nf(np.sqrt((P ** 2).sum(-1)))

    def V3(P):
        r = np.sqrt((P ** 2).sum(-1))
        vr = np.asarray(vf(r), float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(r[..., None] > 0, (vr / np.where(r > 0, r, 1.0))[..., None] * P, 0.0)
        return out
    e = np.eye(3)
    c4 = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]

    def D1(F, P, i):
        return sum(w * F(P + s * h * e[i]) for s, w in c4) / h

    J = np.zeros((len(Y), NJ))
    J[:, 0] = N3(Y)
    for i in range(3):
        J[:, jet_index("dN%d" % i)] = D1(N3, Y, i)
        dvi = D1(V3, Y, i)
        for j in range(3):
            J[:, jet_index("dv%d%d" % (i, j))] = dvi[:, j]
    for k in range(3):
        for i in range(k, 3):
            Fk = lambda P, i=i: D1(V3, P, i)
            dd = D1(Fk, Y, k)
            for j in range(3):
                J[:, jet_index("ddv%d%d%d" % (k, i, j))] = dd[:, j]
    return J


def basis_jets(Y, Zc, w):
    """Jb[q, j, :]: the jets at test point q of basis element j = 4 p + a (a = 0: N, 1..3: v_x, v_y, v_z) centred at Zc[p]."""
    nq, npnt = len(Y), len(Zc)
    val, grad, hess = psi_jets(Y[:, None, :] - Zc[None, :, :], w)        # (nq, np), (nq, np, 3), (nq, np, 3, 3)
    Jb = np.zeros((nq, 4 * npnt, NJ))
    Jb[:, 0::4, 0] = val
    for i in range(3):
        Jb[:, 0::4, jet_index("dN%d" % i)] = grad[..., i]
    for a in range(3):
        sl = slice(1 + a, None, 4)
        for i in range(3):
            Jb[:, sl, jet_index("dv%d%d" % (i, a))] = grad[..., i]
        for k in range(3):
            for i in range(k, 3):
                Jb[:, sl, jet_index("ddv%d%d%d" % (k, i, a))] = hess[..., k, i]
    return Jb


def basis_site_fields(Xs, Zc, w, ell, kernels):
    """F[p, x] = (W^dag psi_p)(x) = (G_ell * psi)(|x - z_p|) on the sites."""
    dist = np.sqrt(((Xs[None, :, :] - Zc[:, None, :]) ** 2).sum(-1))
    return psi_backavg(dist, w, ell, kernels)


def h_of_basis(P, F, j):
    """h[phi_j] = 1/2{diag F_p, h_0} (a = 0) or 1/2{diag F_p, Q_a} (a = 1..3), sparse."""
    p, a = divmod(j, 4)
    D = sp.diags(np.repeat(F[p], 4))
    A = P.h if a == 0 else P.Q[a - 1]
    return (0.5 * (D @ A + A @ D)).tocsr()


def h_of_field(P, F, c):
    """h[sum_j c_j phi_j] (exact linearity of h in the fields)."""
    c = np.asarray(c, float).reshape(-1, 4)
    Nd = sp.diags(np.repeat(F.T @ c[:, 0], 4))
    H = 0.5 * (Nd @ P.h + P.h @ Nd)
    for a in range(3):
        Vd = sp.diags(np.repeat(F.T @ c[:, 1 + a], 4))
        H = H + 0.5 * (Vd @ P.Q[a] + P.Q[a] @ Vd)
    return H.tocsr()


# ============================================================================ rigorous symmetric eigen-enclosure (Kahan/Parlett residual theorem)
def eig_enclosure_dense(A, lam, X, A_rad_F=0.0, torch=None):
    """for a Hermitian A (exact = computed A + E, ||E||_F <= A_rad_F) and computed (lam, X) of eigh: every eigenvalue of the exact A lies
    within r of the paired lam_i, r = (||A X - X diag lam||_F + rounding) / sqrt(1 - ||X^* X - I||_F - rounding) + A_rad_F
    (Kahan 1967; Parlett, The Symmetric Eigenvalue Problem, Thm 11.5.1, with X square of full rank). Returns r and its parts."""
    n = A.shape[0]
    if torch is not None:
        At = torch.from_numpy(np.ascontiguousarray(A)); Xt = torch.from_numpy(np.ascontiguousarray(X))
        Rm = (At @ Xt - Xt * torch.from_numpy(lam).to(Xt.dtype)[None, :])
        resF = float(torch.linalg.norm(Rm))
        G = Xt.conj().T @ Xt
        G -= torch.eye(n, dtype=G.dtype)
        orthF = float(torch.linalg.norm(G))
        del Rm, G, At, Xt
    else:
        resF = float(np.linalg.norm(A @ X - X * lam[None, :]))
        orthF = float(np.linalg.norm(X.conj().T @ X - np.eye(n)))
    AF = float(np.linalg.norm(A)); XF = float(np.linalg.norm(X))
    res_rnd = gam(n + 2) * AF * XF + gam(2) * XF * float(np.abs(lam).max())
    orth_rnd = gam(n + 2) * XF * XF
    resU = (resF + res_rnd) * (1 + gam(n * n))
    orthU = (orthF + orth_rnd + 1.0 * gam(n)) * (1 + gam(n * n))
    if orthU >= 1:
        return {"r": float("inf"), "resF": resF, "orthF": orthF}
    r = resU / np.sqrt(1.0 - orthU) * (1 + 1e-15) + A_rad_F
    return {"r": float(r), "resF_upper": resU, "orthF_upper": orthU, "A_rad_F": A_rad_F}


def eig_enclosure_sparseH(Hs, lam, X, torch):
    """the same with a sparse Hermitian H (computes H X by sparse-dense products in column blocks)."""
    n = Hs.shape[0]
    Hs = Hs.tocsr()
    k = int(np.diff(Hs.indptr).max())
    resF2 = 0.0
    for i0 in range(0, n, 1024):
        Xb = X[:, i0:i0 + 1024]
        Rb = Hs @ Xb - Xb * lam[None, i0:i0 + 1024]
        resF2 += float(np.sum(np.abs(Rb) ** 2))
    resF = np.sqrt(resF2)
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    G = Xt.conj().T @ Xt
    G -= torch.eye(n, dtype=G.dtype)
    orthF = float(torch.linalg.norm(G)); del G, Xt
    HF = float(np.sqrt(np.sum(np.abs(Hs.data) ** 2))); XF = float(np.linalg.norm(X))
    res_rnd = gam(k + 2) * HF * XF + gam(2) * XF * float(np.abs(lam).max())
    orth_rnd = gam(n + 2) * XF * XF
    resU = (resF + res_rnd) * (1 + gam(n * n))
    orthU = (orthF + orth_rnd + gam(n)) * (1 + gam(n * n))
    r = resU / np.sqrt(1.0 - orthU) * (1 + 1e-15)
    return {"r": float(r), "resF_upper": float(resU), "orthF_upper": float(orthU)}


# ============================================================================ extended-precision Gram (x87 long double, u = 2^-64)
U_LD = float(np.finfo(np.longdouble).eps) / 2.0


def gam_ld(k):
    k = float(k)
    return k * U_LD / (1.0 - k * U_LD)


class PreciseGram:
    """Gamma for the test order mu = 4 q + tau is exactly block diagonal in tau (energy, momentum 1..3: the spin traces vanish), with
    Gamma_tau = Phi M_tau Phi^T, Phi = the stored double test weights (exact in long double), M_tau = (K + diag(K 1))/2,
    K[x, y] = sum_{a in x, b in y} |A_ab|^2 for A = h_0 (tau = 0) or Q_tau. Built in long double with an entrywise rounding radius;
    lambda_min certified per block by a long-double Cholesky of Gamma_tau - c I and Higham's backward error (Thm 10.3:
    R^T R = A' + dA, |dA| <= gamma_{n+1} |R^T||R|, so lambda_min(A') >= -gamma_{n+1} ||R||_F^2); solves by the long-double factor."""

    def __init__(self, P, Wt):
        nsite = Wt.shape[1]
        B = sp.kron(sp.identity(nsite, format="csr"), np.ones((1, 4))).tocsr()
        Phi = Wt.astype(np.longdouble)
        self.n = Wt.shape[0]
        self.blocks = []
        for A in [P.h] + list(P.Q):
            A2 = A.copy().tocsr(); A2.data = np.abs(A2.data) ** 2
            K = (B @ A2 @ B.T).toarray().astype(np.longdouble)
            M = 0.5 * (K + np.diag(K.sum(1)))
            T1 = Phi @ M
            Gt = T1 @ Phi.T
            Gt = 0.5 * (Gt + Gt.T)
            Gabs = (np.abs(Phi) @ np.abs(M)) @ np.abs(Phi).T
            rad = gam_ld(2 * nsite + 4) * Gabs + 4 * U_LD * Gabs     # products and sums; the |A_ab|^2 squares (one rounding) included
            self.blocks.append({"G": Gt, "rad": rad})

    def certify(self, frac=0.9):
        out = []
        for blk in self.blocks:
            G = blk["G"]; n = G.shape[0]
            lf = float(np.linalg.eigvalsh(G.astype(np.float64))[0])
            c = np.longdouble(frac * lf) if lf > 0 else np.longdouble(0)
            A = G - c * np.eye(n, dtype=np.longdouble)
            R, ok = chol_ld(A)
            radF = float(np.sqrt(np.sum(blk["rad"].astype(np.float64) ** 2)))
            if ok:
                RF2 = float(np.sum((R.astype(np.float64)) ** 2))
                lo = float(c) - gam_ld(n + 1) * RF2 * (1 + 1e-6) - U_LD * float(np.abs(np.diag(G)).max() + abs(c)) - radF
            else:
                lo = -np.inf
            blk.update({"R": R if ok else None, "lam_float": lf, "lam_lo": lo, "radF": radF, "chol_ok": ok, "shift": float(c)})
            out.append({"lam_float": lf, "lam_lo": lo, "radF": radF, "chol_ok": ok, "shift": float(c)})
        self.lam_lo = min(o["lam_lo"] for o in out)
        return out

    def dense_double(self):
        nT = 4 * self.n
        G = np.zeros((nT, nT))
        for t, blk in enumerate(self.blocks):
            G[t::4, t::4] = blk["G"].astype(np.float64)
        return G

    def solve(self, X):
        """Y = Gamma^-1 X (X double, (nT,) or (nT, k)) by block Cholesky in long double with two refinement steps, returned as double
        Yd, with a rigorous columnwise bound of the exact residual |X - Gamma_exact Yd|_2 (computed for the double Yd in long double, with
        its rounding and the radius of Gamma). The realized operator's error is then <= that / sqrt(lambda_lo) (HS norm, since
        ||sum_mu A_mu dy_mu||_HS^2 = dy^T Gamma dy = r^T Gamma^-1 r)."""
        X = np.asarray(X, float)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        Yd = np.zeros(X.shape)
        res2 = np.zeros(X.shape[1])
        for t, blk in enumerate(self.blocks):
            if "R0" not in blk:
                blk["R0"], _ = chol_ld(blk["G"])
            G = blk["G"]; Rhs = X[t::4].astype(np.longdouble)
            Yt = chol_solve_ld(blk["R0"], Rhs)
            for _ in range(2):
                r = Rhs - G @ Yt
                Yt = Yt + chol_solve_ld(blk["R0"], r)
            Ytd = Yt.astype(np.float64)
            Yl = Ytd.astype(np.longdouble)
            r = Rhs - G @ Yl
            rnd = gam_ld(self.n + 2) * (np.abs(G) @ np.abs(Yl) + np.abs(Rhs))
            col = np.sqrt(np.sum(((np.abs(r) + rnd + blk["rad"] @ np.abs(Yl)).astype(np.float64)) ** 2, axis=0)) * (1 + 1e-12)
            res2 += col ** 2
            Yd[t::4] = Ytd
        res = np.sqrt(res2)
        return (Yd[:, 0], res[0]) if one else (Yd, res)


def chol_ld(A):
    """upper Cholesky factor R (A = R^T R) in long double, vectorized by columns; (R, ok)."""
    n = A.shape[0]
    R = np.zeros_like(A)
    W = A.copy()
    for j in range(n):
        s = W[j, j]
        if not s > 0:
            return R, False
        rjj = np.sqrt(s)
        R[j, j] = rjj
        if j + 1 < n:
            row = W[j, j + 1:] / rjj
            R[j, j + 1:] = row
            W[j + 1:, j + 1:] -= np.outer(row, row)
    return R, True


def chol_solve_ld(R, B, shift=None):
    """solve R^T R Y = B in long double (B (n, k))."""
    n = R.shape[0]
    Z = np.zeros_like(B)
    for i in range(n):                     # R^T Z = B (lower triangular)
        Z[i] = (B[i] - R[:i, i] @ Z[:i]) / R[i, i]
    Y = np.zeros_like(B)
    for i in range(n - 1, -1, -1):         # R Y = Z
        Y[i] = (Z[i] - R[i, i + 1:] @ Y[i + 1:]) / R[i, i]
    return Y
