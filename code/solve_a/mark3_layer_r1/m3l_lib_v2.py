# m3l_lib_v1: structured engine for the mark-3 root-layer test (route 1), manifest v2 + v3 changes + mark3zz.
# Written fresh for this test; no code reused from mark3_r1 (definitions differ).
import numpy as np, math
import numba as nb
from scipy.special import ellipj, ellipk
import scipy.sparse as sp

G_NEWTON = 0.25
A = 1.0

# ---------------------------------------------------------------- chain operators (band storage: list of diagonals)
# A Hermitian banded matrix of dim n, half-bandwidth b is stored as D[d] (d=0..b), D[d][i] = H[i, i+d] (length n-d).

def heavy_band(ns, k, m, Nb, vb, a=A):
    """Full 2-component heavy chain, interleaved index 2x+s (s=0 up, 1 down). h = 1/2{N,h0} + 1/2{v,Q}."""
    n = 2*ns; r = np.arange(1, ns+1)*a
    D = [np.zeros(n-d, complex) for d in range(5)]
    D[0][0::2] = m*Nb; D[0][1::2] = -m*Nb
    D[1][0::2] = -1j*k/r*Nb                      # (up_x, down_x): sigma_2 k/r -> -i k/r
    c = 0.5*(Nb[:-1]+Nb[1:])/(2j*a)               # sigma_1 p between x, x+1
    D[3][0::2][:ns-1] = c                         # (up_x, down_{x+1}) offset 3
    D[1][1::2][:ns-1] = c                         # (down_x, up_{x+1}) offset 1
    q = 0.5*(vb[:-2]+vb[2:])/(4j*a)
    D[4][0::2][:ns-2] = q; D[4][1::2][:ns-2] = q  # offset 4 same spin
    return D

def light_block_band(ns, k, Nb, vb, a=A, gamma=+1):
    """Gamma = S sigma_2 block of the massless chain (pentadiagonal), basis e_n (x) chi_{gamma S_n}."""
    r = np.arange(1, ns+1)*a; S = (-1.0)**np.arange(1, ns+1)
    D = [np.zeros(ns-d, complex) for d in range(3)]
    D[0][:] = gamma*S*k*Nb/r
    D[1][:] = -gamma*S[:-1]*(Nb[:-1]+Nb[1:])/(4*a)
    D[2][:] = -1j*(vb[:-2]+vb[2:])/(8*a)
    return D

def light_full_band(ns, k, Nb, vb, a=A):
    return heavy_band(ns, k, 0.0, Nb, vb, a)

def band_to_dense(D):
    n = len(D[0]); H = np.zeros((n, n), complex)
    for d, x in enumerate(D):
        idx = np.arange(n-d)
        H[idx, idx+d] = x
        if d: H[idx+d, idx] = np.conj(x)
    return H

def band_to_blocks(D, b):
    """Block-tridiagonal form with blocks of size b (n divisible by b). Returns Ad[nb,b,b], Au[nb-1,b,b] (A_{i,i+1})."""
    H = None
    n = len(D[0]); nbk = n//b
    Ad = np.zeros((nbk, b, b), complex); Au = np.zeros((max(nbk-1, 1), b, b), complex)
    for d, x in enumerate(D):
        for i0 in range(n-d):
            pass
    # vectorized fill
    for d, x in enumerate(D):
        i = np.arange(n-d); j = i+d
        bi, ri = divmod(i, b); bj, rj = divmod(j, b)
        same = bi == bj
        Ad[bi[same], ri[same], rj[same]] = x[same]
        Ad[bi[same], rj[same], ri[same]] = np.conj(x[same])
        nxt = bj == bi+1
        Au[bi[nxt], ri[nxt], rj[nxt]] = x[nxt]
    return Ad, Au

def band_norm_bound(D):
    """Row-sum bound on ||H|| (Gershgorin-type, valid for the 2-norm via ||H||_2 <= sqrt(||H||_1 ||H||_inf))."""
    n = len(D[0]); s = np.abs(D[0]).astype(float).copy()
    for d in range(1, len(D)):
        s[:n-d] += np.abs(D[d]); s[d:] += np.abs(D[d])
    return s.max()

def gershgorin_interval(D):
    n = len(D[0]); c = D[0].real.copy(); rad = np.zeros(n)
    for d in range(1, len(D)):
        rad[:n-d] += np.abs(D[d]); rad[d:] += np.abs(D[d])
    return (c-rad).min(), (c+rad).max()

# ---------------------------------------------------------------- numba: inertia and selected inversion
@nb.njit(cache=True)
def _inv(M):
    return np.linalg.inv(M)

@nb.njit(cache=True)
def inertia(Ad, Au, shift):
    """Number of negative eigenvalues of H - shift (Haynsworth on block LDL^H). Returns (count, min |pivot eig|)."""
    nbk, b, _ = Ad.shape
    S = Ad[0].copy()
    for t in range(b): S[t, t] -= shift
    cnt = 0; mn = 1e300
    for i in range(nbk):
        Sh = 0.5*(S + S.conj().T)
        ev = np.linalg.eigvalsh(Sh)
        for t in range(b):
            if ev[t] < 0: cnt += 1
            if abs(ev[t]) < mn: mn = abs(ev[t])
        if i+1 < nbk:
            X = np.linalg.solve(Sh, Au[i])
            S = Ad[i+1] - Au[i].conj().T @ X
            for t in range(b): S[t, t] -= shift
    return cnt, mn

@nb.njit(cache=True, parallel=True)
def selinv_multi(Ad, Au, zs):
    """For each complex shift z: diagonal blocks G_ii, first and second upper off-diagonal blocks of (H - z)^{-1}
    by the recursive Green's function method."""
    nbk, b, _ = Ad.shape; nz = zs.shape[0]
    Gd = np.zeros((nz, nbk, b, b), np.complex128)
    G1 = np.zeros((nz, max(nbk-1, 1), b, b), np.complex128)
    G2 = np.zeros((nz, max(nbk-2, 1), b, b), np.complex128)
    for iz in nb.prange(nz):
        z = zs[iz]
        gL = np.zeros((nbk, b, b), np.complex128); gR = np.zeros((nbk, b, b), np.complex128)
        M = Ad[0].copy()
        for t in range(b): M[t, t] -= z
        gL[0] = np.linalg.inv(M)
        for i in range(1, nbk):
            M = Ad[i] - Au[i-1].conj().T @ gL[i-1] @ Au[i-1]
            for t in range(b): M[t, t] -= z
            gL[i] = np.linalg.inv(M)
        M = Ad[nbk-1].copy()
        for t in range(b): M[t, t] -= z
        gR[nbk-1] = np.linalg.inv(M)
        for i in range(nbk-2, -1, -1):
            M = Ad[i] - Au[i] @ gR[i+1] @ Au[i].conj().T
            for t in range(b): M[t, t] -= z
            gR[i] = np.linalg.inv(M)
        for i in range(nbk):
            M = Ad[i].copy()
            for t in range(b): M[t, t] -= z
            if i > 0: M -= Au[i-1].conj().T @ gL[i-1] @ Au[i-1]
            if i < nbk-1: M -= Au[i] @ gR[i+1] @ Au[i].conj().T
            Gd[iz, i] = np.linalg.inv(M)
        for i in range(nbk-1):
            G1[iz, i] = -Gd[iz, i] @ Au[i] @ gR[i+1]
        for i in range(nbk-2):
            G2[iz, i] = -G1[iz, i] @ Au[i+1] @ gR[i+2]
    return Gd, G1, G2

def blocks_to_band(Gd, G1, G2, b, nband):
    """Convert block entries (upper part) to diagonals d=0..nband (nband <= 3b-1). Returns list of arrays (upper)."""
    nbk = Gd.shape[0]; n = nbk*b
    out = []
    for d in range(nband+1):
        i = np.arange(n-d); j = i+d
        bi, ri = divmod(i, b); bj, rj = divmod(j, b)
        x = np.zeros(n-d, complex)
        s0 = bj == bi; s1 = bj == bi+1; s2 = bj == bi+2
        x[s0] = Gd[bi[s0], ri[s0], rj[s0]]
        x[s1] = G1[bi[s1], ri[s1], rj[s1]]
        if s2.any(): x[s2] = G2[bi[s2], ri[s2], rj[s2]]
        out.append(x)
    return out

# ---------------------------------------------------------------- Zolotarev sign approximation
def zolotarev(kk, n):
    """sign(x) ~ x sum_l b_l/(x^2 + c_l) on [-1,-kk] U [kk,1]. Returns (b, c, max error on the set)."""
    kp2 = 1 - kk*kk
    Kp = ellipk(kp2)
    l = np.arange(1, 2*n)
    sn, cn, dn, _ = ellipj(l*Kp/(2*n), kp2)
    c = kk*kk*sn**2/cn**2
    codd = c[0::2]; ceven = c[1::2]   # c_1,c_3,...,c_{2n-1} (n of them); c_2..c_{2n-2} (n-1)
    bl = np.empty(n)
    for j in range(n):
        num = np.prod(ceven - codd[j]); den = np.prod(np.delete(codd, j) - codd[j])
        bl[j] = num/den
    xs = np.concatenate([np.geomspace(kk, 1, 20000)])
    rr = xs*np.sum(bl[None, :]/(xs[:, None]**2 + codd[None, :]), axis=1)
    D = 2/(rr.max()+rr.min())
    bl *= D; rr *= D
    return bl, codd, np.abs(rr-1).max()

def zolo_degree(kk, eps=1e-14):
    for n in range(4, 200):
        if 4*math.exp(-math.pi**2*n/math.log(4/kk)) < eps: return n
    return 200

# ---------------------------------------------------------------- projector band via Zolotarev + RGF
def rank_gap(Ad, Au, nocc, lo, hi, tol=1e-15, it=200):
    """Bisection by inertia for lambda_nocc (largest occupied) and lambda_{nocc+1}. lo/hi bracket all eigenvalues."""
    def kth(kidx):  # smallest x with count(x) >= kidx  -> lambda_kidx
        a_, b_ = lo, hi
        for _ in range(it):
            mid = 0.5*(a_+b_)
            c, _m = inertia(Ad, Au, mid)
            if c >= kidx: b_ = mid
            else: a_ = mid
            if b_-a_ < tol*max(1.0, abs(mid)): break
        return a_, b_
    l1 = kth(nocc); l2 = kth(nocc+1)
    return l1, l2   # enclosures [a,b] of lambda_nocc and lambda_{nocc+1}

def projector_band(D, b, nocc, lo, hi, nband, eps=1e-14, gapinfo=None):
    """Band entries (offsets 0..nband) of the projector onto the lowest nocc levels of H (single symmetry block).
    Certified via inertia: lambda_nocc < mu < lambda_{nocc+1}, distance >= dmin. Rational error <= errZ on spectrum."""
    Ad, Au = band_to_blocks(D, b)
    if gapinfo is None:
        (a1, b1), (a2, b2) = rank_gap(Ad, Au, nocc, lo, hi)
    else:
        (a1, b1), (a2, b2) = gapinfo
    mu = 0.5*(b1+a2); dmin = min(mu-b1, a2-mu)
    if dmin <= 0: raise RuntimeError("rank gap not resolved")
    E = max(hi-mu, mu-lo)*1.0001
    kk = dmin/E
    nz = zolo_degree(kk, eps)
    bl, cl, errZ = zolotarev(kk, nz)
    s = np.sqrt(cl)*E
    zs = mu + 1j*s
    Gd, G1, G2 = selinv_multi(Ad, Au, zs.astype(np.complex128))
    # sign(H-mu) = sum_l bl/E * (H-mu)[(H-mu)^2 + (E s_l')^2]^{-1} ... with x=(H-mu)/E: x/(x^2+c) = (E/2)[(H-mu-iEsq)^{-1} + (H-mu+iEsq)^{-1}]
    # (H - z)^{-1} with z = mu + i s ; (H - zbar)^{-1} = [(H - z)^{-1}]^dagger
    n = len(D[0]); Sg = [np.zeros(n-d, complex) for d in range(nband+1)]
    Gup = [blocks_to_band(Gd[l], G1[l], G2[l], b, nband) for l in range(len(zs))]
    for l in range(len(zs)):
        coef = bl[l]*E/2*(1.0/E)  # x/(x^2+c) with x = (H-mu)/E -> (H-mu)/E / ((H-mu)^2/E^2 + c) = E (H-mu)/((H-mu)^2 + E^2 c)
        # E (H-mu)/((H-mu)^2+s^2) = (E/2)[(H-mu-is)^{-1} + (H-mu+is)^{-1}],  so coef = bl * E/2 * (1/E)?? -> see below
        pass
    # Correct assembly: sign ~ sum_l bl_l * x/(x^2+c_l), x/(x^2+c_l) = E(H-mu)/((H-mu)^2 + s_l^2) = (E/2)[R(z_l) + R(z_l)^dag]
    for l in range(len(zs)):
        for d in range(nband+1):
            up = Gup[l][d]                         # R(z)_{i,i+d}
            # R^dag_{i,i+d} = conj(R_{i+d,i}); need lower entries of R(z): R(z)_{i+d,i}. For Hermitian H, R(z)^T = R(z) only if H symmetric.
            # Use (H - z)^{-1} lower entry = [(H - zbar)^{-1}]^dagger upper ... compute separately below.
            pass
    # lower entries: R(z)_{i+d,i} = conj(R(zbar)_{i,i+d}); R(zbar) upper from a second RGF at conj shifts
    Gd2, G12, G22 = selinv_multi(Ad, Au, np.conj(zs).astype(np.complex128))
    for l in range(len(zs)):
        Gl = blocks_to_band(Gd2[l], G12[l], G22[l], b, nband)
        for d in range(nband+1):
            up = Gup[l][d]                 # R(z)_{i,i+d}
            dag = Gl[d]                    # R(z)^dag_{i,i+d} = conj(R(z)_{i+d,i}) = R(zbar)_{i,i+d}
            Sg[d] += bl[l]*E/2*(up + dag)
    P = [(-0.5*Sg[d]) for d in range(nband+1)]
    P[0] = P[0] + 0.5
    info = dict(mu=mu, dmin=dmin, E=E, kk=kk, nz=nz, errZ=errZ, lam_occ=(a1, b1), lam_emp=(a2, b2))
    return P, info

# ---------------------------------------------------------------- dense reference (for checks only)
def dense_projector(D, nocc):
    H = band_to_dense(D); w, V = np.linalg.eigh(H)
    return V[:, :nocc] @ V[:, :nocc].conj().T, w

# ---------------------------------------------------------------- kernels and rows
def K0(y, r, sg):
    y = np.asarray(y, float); r = np.asarray(r, float)
    t2 = 2*y*r/sg**2
    base = np.exp(-(y-r)**2/(2*sg**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        val = r/(math.sqrt(2*math.pi)*sg*y)*base*(-np.expm1(-t2))
    lim = r/(math.sqrt(2*math.pi)*sg)*np.exp(-r**2/(2*sg**2))*2*r/sg**2
    return np.where(y > 0, val, lim)

def K1(y, r, sg):
    y = np.asarray(y, float); r = np.asarray(r, float)
    t = 2*y*r/sg**2
    base = np.exp(-(y-r)**2/(2*sg**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        brk = (1+np.exp(-t)) - (-np.expm1(-t))/(t/2)
        ser = t**2/6 - t**3/12 + t**4/40
        brk = np.where(t < 1e-3, ser, brk)
        val = r/(math.sqrt(2*math.pi)*sg*y)*base*brk
    return np.where(y > 0, val, 0.0)

class Grid:
    def __init__(self, nsites, sg, hd, a=A, ext=10.0):
        self.a = a; self.sg = sg; self.hd = hd
        self.r = np.arange(1, nsites+1)*a
        self.V = 4*math.pi*self.r**2*a
        I = int(round((self.r[-1] + ext*sg)/hd))
        self.y = np.arange(I+1)*hd
        c = np.ones(I+1); c[0] = c[-1] = 0.5
        self.c = c
        self.w = 4*math.pi*self.y**2*hd*c
        rows = []; cols = []; v0 = []; v1 = []
        span = int(math.ceil(ext*sg/a))+1
        for n, rn in enumerate(self.r):
            i0 = max(0, int((rn - ext*sg)/hd)); i1 = min(I, int((rn + ext*sg)/hd)+1)
            yy = self.y[i0:i1+1]
            rows.append(np.arange(i0, i1+1)); cols.append(np.full(len(yy), n))
            v0.append(a*K0(yy, rn, sg)); v1.append(a*K1(yy, rn, sg))
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        self.W0 = sp.csr_matrix((np.concatenate(v0), (rows, cols)), shape=(I+1, nsites))
        self.W1 = sp.csr_matrix((np.concatenate(v1), (rows, cols)), shape=(I+1, nsites))
        self.W0T = self.W0.T.tocsr(); self.W1T = self.W1.T.tocsr()
    def smooth(self, rho, J):
        return self.W0 @ rho, self.W1 @ J
    def back(self, Nd, vd):
        return (self.W0T @ (self.w*Nd))/self.V, (self.W1T @ (self.w*vd))/self.V

@nb.njit(cache=True)
def rows_forward(y, hd, rho, J, G):
    I = y.shape[0]-1
    M = np.zeros(I+1); w = np.zeros(I+1); s = np.zeros(I+1); Dd = np.ones(I+1)
    for i in range(I):
        yi1 = y[i+1]
        rhs = M[i] + 0.5*hd*s[i]
        c = 0.5*hd*4*math.pi*yi1*yi1
        x = max(M[i], 0.0)
        for it in range(60):
            ww = -math.sqrt(max(2*G*x/yi1, 0.0))
            F = x - rhs - c*(rho[i+1] + ww*J[i+1])
            dw = 0.0 if x <= 0 else ww/(2*x)
            dF = 1 - c*J[i+1]*dw
            dx = F/dF
            x -= dx
            if abs(dx) <= 1e-15*max(abs(x), 1e-300): break
        M[i+1] = x
        w[i+1] = -math.sqrt(max(2*G*x/yi1, 0.0))
        s[i+1] = 4*math.pi*yi1*yi1*(rho[i+1] + w[i+1]*J[i+1])
    return M, w, s

@nb.njit(cache=True)
def rows_adjoint(y, hd, J, M, w, alpha, beta, G):
    """Gradient of Q = sum_i alpha_i M_i + beta_i s_i w.r.t. rho_i and J_i (grid), through the implicit recursion."""
    I = y.shape[0]-1
    dsdM = np.zeros(I+1)
    for i in range(1, I+1):
        if M[i] > 0: dsdM[i] = 4*math.pi*y[i]*y[i]*J[i]*w[i]/(2*M[i])
    gr = np.zeros(I+1); gJ = np.zeros(I+1)
    lam = alpha[I] + beta[I]*dsdM[I]          # dQ/dM_I (total, M_I last)
    # Q's direct dependence on rho_I,J_I via s_I
    gr[I] += beta[I]*4*math.pi*y[I]**2; gJ[I] += beta[I]*4*math.pi*y[I]**2*w[I]
    for i in range(I-1, -1, -1):
        # M_{i+1} = M_i + hd/2 (s_i + s_{i+1}); D_{i+1} = 1 - hd/2 dsdM_{i+1}
        D1 = 1 - 0.5*hd*dsdM[i+1]
        mu = lam/D1                            # multiplier of the step residual
        # rho_{i+1}, J_{i+1} through step i (right end)
        gr[i+1] += mu*0.5*hd*4*math.pi*y[i+1]**2
        gJ[i+1] += mu*0.5*hd*4*math.pi*y[i+1]**2*w[i+1]
        # rho_i, J_i through step i (left end)
        gr[i] += mu*0.5*hd*4*math.pi*y[i]**2
        gJ[i] += mu*0.5*hd*4*math.pi*y[i]**2*w[i]
        lam = alpha[i] + beta[i]*dsdM[i] + mu*(1 + 0.5*hd*dsdM[i])
        gr[i] += beta[i]*4*math.pi*y[i]**2; gJ[i] += beta[i]*4*math.pi*y[i]**2*w[i]
    return gr, gJ

def fields_from_sources(grid, rho_s, J_s, G=G_NEWTON):
    rl, Jl = grid.smooth(rho_s, J_s)
    M, w, s = rows_forward(grid.y, grid.hd, rl, Jl, G)
    I = len(grid.y)-1
    al = np.zeros(I+1); al[I] = 1.0; be = np.zeros(I+1)
    gr, gJ = rows_adjoint(grid.y, grid.hd, Jl, M, w, al, be, G)
    with np.errstate(divide='ignore', invalid='ignore'):
        Nd = np.where(grid.w > 0, gr/grid.w, 0.0); vd = np.where(grid.w > 0, gJ/grid.w, 0.0)
    Nd[0] = (4*Nd[1]-Nd[2])/3; vd[0] = 0.0
    Nb, vb = grid.back(Nd, vd)
    return dict(rl=rl, Jl=Jl, M=M, w=w, s=s, Nd=Nd, vd=vd, Nb=Nb, vb=vb, E=M[-1])

def site_grad(grid, f, alpha, beta, G=G_NEWTON):
    """dQ/drho_n, dQ/dJ_n (site sources) for Q = sum alpha_i M_i + beta_i s_i."""
    gr, gJ = rows_adjoint(grid.y, grid.hd, f['Jl'], f['M'], f['w'], alpha, beta, G)
    return grid.W0T @ gr, grid.W1T @ gJ

# ---------------------------------------------------------------- root
def root_seed(grid, f, G=G_NEWTON):
    """Outermost... all sign changes of g(y) = 2 G M(y) - y on the grid, cubic Hermite with M' = s (rows' derivative)."""
    y = grid.y; M = f['M']; s = f['s']
    g = 2*G*M - y
    idx = np.nonzero((g[:-1] > 0) & (g[1:] <= 0) | (g[:-1] < 0) & (g[1:] >= 0))[0]
    return idx, g

def hermite_root(grid, f, i, G=G_NEWTON):
    y = grid.y; M = f['M']; s = f['s']; h = grid.hd
    def Mh(x):
        t = (x - y[i])/h
        h00 = 2*t**3-3*t**2+1; h10 = t**3-2*t**2+t; h01 = -2*t**3+3*t**2; h11 = t**3-t**2
        return h00*M[i]+h10*h*s[i]+h01*M[i+1]+h11*h*s[i+1]
    def dMh(x):
        t = (x - y[i])/h
        return ((6*t**2-6*t)*M[i]+(3*t**2-4*t+1)*h*s[i]+(-6*t**2+6*t)*M[i+1]+(3*t**2-2*t)*h*s[i+1])/h
    a_, b_ = y[i], y[i+1]
    ga = 2*G*Mh(a_)-a_
    for _ in range(200):
        m = 0.5*(a_+b_); gm = 2*G*Mh(m)-m
        if (gm > 0) == (ga > 0): a_, ga = m, gm
        else: b_ = m
        if b_-a_ < 1e-14*m: break
    R = 0.5*(a_+b_)
    return R, Mh(R), dMh(R)

def hermite_weights(grid, i, x):
    """alpha, beta seeds such that M_interp(x) = sum alpha M + beta s."""
    h = grid.hd; t = (x - grid.y[i])/h
    al = {i: 2*t**3-3*t**2+1, i+1: -2*t**3+3*t**2}
    be = {i: h*(t**3-2*t**2+t), i+1: h*(t**3-t**2)}
    return al, be

# ---------------------------------------------------------------- GPU panel propagation (Chebyshev)
import torch
from scipy.special import jv

def to_dev_band(D, dev):
    return [torch.as_tensor(np.ascontiguousarray(x), dtype=torch.complex128, device=dev).unsqueeze(1) for x in D]

def band_matvec(Dt, Z, out=None):
    Y = Dt[0]*Z
    for d in range(1, len(Dt)):
        Y[:-d] += Dt[d]*Z[d:]
        Y[d:] += Dt[d].conj()*Z[:-d]
    return Y

def cheb_degree(tau_s, tol=1e-16):
    x = abs(tau_s)
    for K in range(int(x)+2, int(x)+400):
        if abs(jv(K, x)) < tol*1e-2 and abs(jv(K+1, x)) < tol*1e-2: return K
    return int(x)+400

def cheb_expm(Dt, Z, tau, lo, hi, tol=1e-16):
    """exp(-i tau H) Z by Chebyshev on [lo,hi] (an enclosure of spec H). Returns (Z', K, tail bound)."""
    c = 0.5*(hi+lo); s = 0.5*(hi-lo)*1.000001
    x = tau*s
    K = cheb_degree(x, tol)
    Dn = [((Dt[0]-c)/s)] + [d/s for d in Dt[1:]]
    T0 = Z; T1 = band_matvec(Dn, Z)
    acc = jv(0, x)*T0 + 2*((-1j)**1)*jv(1, x)*T1
    for kk in range(2, K+1):
        T2 = 2*band_matvec(Dn, T1) - T0
        acc = acc + 2*((-1j)**kk)*jv(kk, x)*T2
        T0, T1 = T1, T2
    tail = 2*sum(abs(jv(kk, x)) for kk in range(K+1, K+40))
    return acc*np.exp(-1j*tau*c), K, tail

def panel_band(Z, nband):
    """C_{i,i+d} = sum_k Z_ik conj(Z_{i+d,k}), d=0..nband, returned as numpy."""
    out = []
    for d in range(nband+1):
        if d == 0: x = (Z*Z.conj()).real.sum(1).to(torch.complex128)
        else: x = (Z[:-d]*Z[d:].conj()).sum(1)
        out.append(x.cpu().numpy())
    return out

# ================================================================ v2 optimized kernels (override v1 names)
@nb.njit(cache=True, inline='always')
def _sinv(M):
    n = M.shape[0]
    Aug = np.zeros((n, 2*n), np.complex128)
    for i in range(n):
        for j in range(n): Aug[i, j] = M[i, j]
        Aug[i, n+i] = 1.0
    for c in range(n):
        p = c; best = abs(Aug[c, c])
        for rr in range(c+1, n):
            if abs(Aug[rr, c]) > best: best = abs(Aug[rr, c]); p = rr
        if p != c:
            for j in range(2*n):
                tmp = Aug[c, j]; Aug[c, j] = Aug[p, j]; Aug[p, j] = tmp
        piv = Aug[c, c]
        for j in range(2*n): Aug[c, j] /= piv
        for rr in range(n):
            if rr != c:
                fct = Aug[rr, c]
                if fct != 0:
                    for j in range(2*n): Aug[rr, j] -= fct*Aug[c, j]
    out = np.empty((n, n), np.complex128)
    for i in range(n):
        for j in range(n): out[i, j] = Aug[i, n+j]
    return out

@nb.njit(cache=True, inline='always')
def _mm(X, Y):
    n = X.shape[0]; m = Y.shape[1]; kk = X.shape[1]
    out = np.zeros((n, m), np.complex128)
    for i in range(n):
        for j in range(m):
            s = 0j
            for t in range(kk): s += X[i, t]*Y[t, j]
            out[i, j] = s
    return out

@nb.njit(cache=True, inline='always')
def _ct(X):
    n, m = X.shape
    out = np.empty((m, n), np.complex128)
    for i in range(n):
        for j in range(m): out[j, i] = np.conj(X[i, j])
    return out

@nb.njit(cache=True, parallel=True)
def selinv_multi2(Ad, Au, zs):
    """Upper and lower band blocks of (H - z)^{-1} by RGF: Gd, G1 (i,i+1), G2 (i,i+2), L1 (i+1,i), L2 (i+2,i)."""
    nbk, b, _ = Ad.shape; nz = zs.shape[0]
    Gd = np.zeros((nz, nbk, b, b), np.complex128)
    G1 = np.zeros((nz, nbk-1, b, b), np.complex128); G2 = np.zeros((nz, nbk-2, b, b), np.complex128)
    L1 = np.zeros((nz, nbk-1, b, b), np.complex128); L2 = np.zeros((nz, nbk-2, b, b), np.complex128)
    for iz in nb.prange(nz):
        z = zs[iz]
        gL = np.zeros((nbk, b, b), np.complex128); gR = np.zeros((nbk, b, b), np.complex128)
        M = Ad[0].copy()
        for t in range(b): M[t, t] -= z
        gL[0] = _sinv(M)
        for i in range(1, nbk):
            M = Ad[i] - _mm(_mm(_ct(Au[i-1]), gL[i-1]), Au[i-1])
            for t in range(b): M[t, t] -= z
            gL[i] = _sinv(M)
        M = Ad[nbk-1].copy()
        for t in range(b): M[t, t] -= z
        gR[nbk-1] = _sinv(M)
        for i in range(nbk-2, -1, -1):
            M = Ad[i] - _mm(_mm(Au[i], gR[i+1]), _ct(Au[i]))
            for t in range(b): M[t, t] -= z
            gR[i] = _sinv(M)
        for i in range(nbk):
            M = Ad[i].copy()
            for t in range(b): M[t, t] -= z
            if i > 0: M -= _mm(_mm(_ct(Au[i-1]), gL[i-1]), Au[i-1])
            if i < nbk-1: M -= _mm(_mm(Au[i], gR[i+1]), _ct(Au[i]))
            Gd[iz, i] = _sinv(M)
        for i in range(nbk-1):
            G1[iz, i] = -_mm(_mm(Gd[iz, i], Au[i]), gR[i+1])
            L1[iz, i] = -_mm(_mm(gR[i+1], _ct(Au[i])), Gd[iz, i])
        for i in range(nbk-2):
            G2[iz, i] = -_mm(_mm(G1[iz, i], Au[i+1]), gR[i+2])
            L2[iz, i] = -_mm(_mm(gR[i+2], _ct(Au[i+1])), L1[iz, i])
    return Gd, G1, G2, L1, L2

@nb.njit(cache=True)
def _assemble_sign(Gd, G1, G2, L1, L2, coef, b, nband):
    """S_{i,i+d} = sum_l coef_l [R_l(i,i+d) + conj(R_l(i+d,i))], d=0..nband, packed as (nband+1, n)."""
    nz, nbk = Gd.shape[0], Gd.shape[1]; n = nbk*b
    S = np.zeros((nband+1, n), np.complex128)
    for l in range(nz):
        cl = coef[l]
        for d in range(nband+1):
            for i in range(n-d):
                j = i+d; bi = i//b; ri = i % b; bj = j//b; rj = j % b
                if bj == bi:
                    up = Gd[l, bi, ri, rj]; lo = Gd[l, bi, rj, ri]
                elif bj == bi+1:
                    up = G1[l, bi, ri, rj]; lo = L1[l, bi, rj, ri]
                else:
                    up = G2[l, bi, ri, rj]; lo = L2[l, bi, rj, ri]
                S[d, i] += cl*(up + np.conj(lo))
    return S

def zolo_cached(kk, eps=1e-14, _cache={}):
    nz = zolo_degree(kk, eps)
    key = (round(math.log(kk), 3), nz)
    if key not in _cache:
        _cache[key] = zolotarev(math.exp(key[0]), nz)
    return _cache[key], math.exp(key[0])

def projector_band2(D, b, nocc, lo, hi, nband, eps=1e-14, gapinfo=None, mu_fixed=None, zolo=None):
    """As projector_band; mu_fixed/zolo allow evaluating the same rational function at a nearby H (for Pdot)."""
    Ad, Au = band_to_blocks(D, b)
    if gapinfo is None:
        gapinfo = rank_gap(Ad, Au, nocc, lo, hi)
    (a1, b1), (a2, b2) = gapinfo
    if mu_fixed is None:
        mu = 0.5*(b1+a2); dmin = min(mu-b1, a2-mu)
        if dmin <= 0: raise RuntimeError("rank gap not resolved")
        E = max(hi-mu, mu-lo)*1.0001
        (bl, cl, errZ), kk = zolo_cached(dmin*0.999/E, eps)
    else:
        mu, E, bl, cl, errZ, kk = mu_fixed
        dmin = min(mu-b1, a2-mu)
    s = np.sqrt(cl)*E
    zs = (mu + 1j*s).astype(np.complex128)
    Gd, G1, G2, L1, L2 = selinv_multi2(Ad, Au, zs)
    S = _assemble_sign(Gd, G1, G2, L1, L2, (bl*E/2).astype(np.complex128), b, nband)
    P = [-0.5*S[d][:len(D[0])-d] for d in range(nband+1)]
    P[0] = P[0] + 0.5
    info = dict(mu=mu, dmin=dmin, E=E, kk=kk, nz=len(bl), errZ=errZ, gapinfo=gapinfo, rat=(mu, E, bl, cl, errZ, kk))
    return P, info

def band_matvec2(Dt, Z, Y):
    torch.mul(Z, Dt[0], out=Y)
    for d in range(1, len(Dt)):
        Y[:-d].addcmul_(Dt[d], Z[d:])
        Y[d:].addcmul_(Dt[d].conj(), Z[:-d])
    return Y

def cheb_expm2(Dt, Z, tau, lo, hi, tol=1e-16):
    c = 0.5*(hi+lo); s = 0.5*(hi-lo)*1.000001
    x = tau*s; K = cheb_degree(x, tol)
    Dn = [((Dt[0]-c)/s)] + [d/s for d in Dt[1:]]
    T0 = Z.clone(); T1 = torch.empty_like(Z); T2 = torch.empty_like(Z)
    band_matvec2(Dn, T0, T1)
    acc = T0*jv(0, x)
    acc.add_(T1, alpha=complex(2*((-1j)**1)*jv(1, x)))
    for kk in range(2, K+1):
        band_matvec2(Dn, T1, T2)
        T2.mul_(2).sub_(T0)
        acc.add_(T2, alpha=complex(2*((-1j)**kk)*jv(kk, x)))
        T0, T1, T2 = T1, T2, T0
    tail = 2*sum(abs(jv(kk, x)) for kk in range(K+1, K+40))
    acc.mul_(complex(np.exp(-1j*tau*c)))
    del T0, T1, T2
    return acc, K, tail

def panel_band2(Z, nband, chunk=1024):
    n = Z.shape[0]; out = [torch.zeros(n-d, dtype=torch.complex128, device=Z.device) for d in range(nband+1)]
    for c0 in range(0, Z.shape[1], chunk):
        Zc = Z[:, c0:c0+chunk]; Zcc = Zc.conj()
        for d in range(nband+1):
            out[d] += torch.einsum('ik,ik->i', Zc[:n-d], Zcc[d:])
    return [o.cpu().numpy() for o in out]
