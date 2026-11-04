# m3r2_linalg_v1.py -- block-tridiagonal Green's functions (RGF), spectral-projector bands by a
# sign-function quadrature, its tangent (Eq. dm-contact route), block Sturm inertia counts, and
# the Chebyshev propagator exp(-i dt h) applied to orbital blocks (exact to roundoff).
import numpy as np
import numba as nb
from scipy.special import jv


@nb.njit(cache=True)
def _band_to_blocks(HB, b):
    bw1, dim = HB.shape
    nbk = dim // b
    D = np.zeros((nbk, b, b), np.complex128)
    U = np.zeros((nbk - 1, b, b), np.complex128)
    for d in range(bw1):
        for i in range(dim - d):
            j = i + d
            K = i // b; p = i - K * b
            L = j // b; q = j - L * b
            val = HB[d, i]
            if K == L:
                D[K, p, q] = val
                D[K, q, p] = np.conj(val)
            else:
                U[K, p, q] = val
    return D, U


def band_to_blocks(HB, b):
    """Block-tridiagonal blocks (b x b) of a Hermitian band matrix with bandwidth <= b."""
    bw1, dim = HB.shape
    assert bw1 - 1 <= b and dim % b == 0
    return _band_to_blocks(np.ascontiguousarray(HB.astype(np.complex128)), b)


@nb.njit(cache=True, inline='always')
def _inv(M):
    """Small complex matrix inverse, Gauss-Jordan with partial pivoting."""
    n = M.shape[0]
    A = M.copy()
    X = np.eye(n).astype(np.complex128)
    for c in range(n):
        p = c
        best = abs(A[c, c])
        for r in range(c + 1, n):
            if abs(A[r, c]) > best:
                best = abs(A[r, c]); p = r
        if p != c:
            for k in range(n):
                t = A[c, k]; A[c, k] = A[p, k]; A[p, k] = t
                t = X[c, k]; X[c, k] = X[p, k]; X[p, k] = t
        piv = A[c, c]
        for k in range(n):
            A[c, k] /= piv
            X[c, k] /= piv
        for r in range(n):
            if r != c:
                f = A[r, c]
                if f != 0:
                    for k in range(n):
                        A[r, k] -= f * A[c, k]
                        X[r, k] -= f * X[c, k]
    return X


@nb.njit(cache=True, inline='always')
def _mm(A, B):
    n = A.shape[0]; m = B.shape[1]; l = A.shape[1]
    C = np.zeros((n, m), np.complex128)
    for i in range(n):
        for k in range(l):
            a = A[i, k]
            if a != 0:
                for j in range(m):
                    C[i, j] += a * B[k, j]
    return C


@nb.njit(cache=True, inline='always')
def _dag(A):
    n, m = A.shape
    C = np.empty((m, n), np.complex128)
    for i in range(n):
        for j in range(m):
            C[j, i] = np.conj(A[i, j])
    return C


@nb.njit(cache=True)
def _rgf_one(z, D, U):
    """Diagonal blocks GD[K], upper GU[K] = G_{K,K+1}, lower GLo[K] = G_{K+1,K}, plus gL, gR."""
    nbk, b, _ = D.shape
    I = np.eye(b).astype(np.complex128)
    gL = np.zeros((nbk, b, b), np.complex128)
    gR = np.zeros((nbk, b, b), np.complex128)
    gL[0] = _inv(z * I - D[0])
    for K in range(1, nbk):
        Um = U[K - 1]
        gL[K] = _inv(z * I - D[K] - _mm(_dag(Um), _mm(gL[K - 1], Um)))
    gR[nbk - 1] = _inv(z * I - D[nbk - 1])
    for K in range(nbk - 2, -1, -1):
        Uk = U[K]
        gR[K] = _inv(z * I - D[K] - _mm(Uk, _mm(gR[K + 1], _dag(Uk))))
    GD = np.zeros((nbk, b, b), np.complex128)
    GU = np.zeros((nbk - 1, b, b), np.complex128)
    GLo = np.zeros((nbk - 1, b, b), np.complex128)
    for K in range(nbk):
        Y = z * I - D[K]
        if K > 0:
            Y = Y - _mm(_dag(U[K - 1]), _mm(gL[K - 1], U[K - 1]))
        if K < nbk - 1:
            Y = Y - _mm(U[K], _mm(gR[K + 1], _dag(U[K])))
        GD[K] = _inv(Y)
    for K in range(nbk - 1):
        GU[K] = _mm(GD[K], _mm(U[K], gR[K + 1]))
        GLo[K] = _mm(gR[K + 1], _mm(_dag(U[K]), GD[K]))
    return GD, GU, GLo, gL, gR


@nb.njit(parallel=True, cache=True)
def sign_band_rgf(D, U, zs, cs, lin_c, mu):
    """S ~ sign(h - mu) on the band blocks: S = sum_p c_p (G(z_p) + G(z_p)^dag) + lin_c (h - mu).
    Returns SD (nbk,b,b) and SU (nbk-1,b,b)."""
    nbk, b, _ = D.shape
    npole = zs.shape[0]
    SDp = np.zeros((npole, nbk, b, b), np.complex128)
    SUp = np.zeros((npole, nbk - 1, b, b), np.complex128)
    for p in nb.prange(npole):
        GD, GU, GLo, gL, gR = _rgf_one(zs[p], D, U)
        c = cs[p]
        for K in range(nbk):
            SDp[p, K] = c * (GD[K] + _dag(GD[K]))
        for K in range(nbk - 1):
            SUp[p, K] = c * (GU[K] + _dag(GLo[K]))
    SD = np.zeros((nbk, b, b), np.complex128)
    SU = np.zeros((nbk - 1, b, b), np.complex128)
    for p in range(npole):
        SD += SDp[p]
        SU += SUp[p]
    I = np.eye(b).astype(np.complex128)
    for K in range(nbk):
        SD[K] += lin_c * (D[K] - mu * I)
    for K in range(nbk - 1):
        SU[K] += lin_c * U[K]
    return SD, SU


@nb.njit(cache=True)
def _rgf_tangent_one(z, D, U, dD, dU):
    """Primal and tangent (direction dh) of the band blocks of G = (z - h)^{-1}:
    dG = G dh G. Returns GD, GU, GLo, dGD, dGU, dGLo."""
    nbk, b, _ = D.shape
    I = np.eye(b).astype(np.complex128)
    gL = np.zeros((nbk, b, b), np.complex128)
    gR = np.zeros((nbk, b, b), np.complex128)
    dgL = np.zeros((nbk, b, b), np.complex128)
    dgR = np.zeros((nbk, b, b), np.complex128)
    gL[0] = _inv(z * I - D[0])
    dgL[0] = _mm(gL[0], _mm(dD[0], gL[0]))
    for K in range(1, nbk):
        Um = U[K - 1]; dUm = dU[K - 1]
        Y = z * I - D[K] - _mm(_dag(Um), _mm(gL[K - 1], Um))
        dY = -dD[K] - _mm(_dag(dUm), _mm(gL[K - 1], Um)) - _mm(_dag(Um), _mm(dgL[K - 1], Um)) \
             - _mm(_dag(Um), _mm(gL[K - 1], dUm))
        gL[K] = _inv(Y)
        dgL[K] = -_mm(gL[K], _mm(dY, gL[K]))
    gR[nbk - 1] = _inv(z * I - D[nbk - 1])
    dgR[nbk - 1] = _mm(gR[nbk - 1], _mm(dD[nbk - 1], gR[nbk - 1]))
    for K in range(nbk - 2, -1, -1):
        Uk = U[K]; dUk = dU[K]
        Y = z * I - D[K] - _mm(Uk, _mm(gR[K + 1], _dag(Uk)))
        dY = -dD[K] - _mm(dUk, _mm(gR[K + 1], _dag(Uk))) - _mm(Uk, _mm(dgR[K + 1], _dag(Uk))) \
             - _mm(Uk, _mm(gR[K + 1], _dag(dUk)))
        gR[K] = _inv(Y)
        dgR[K] = -_mm(gR[K], _mm(dY, gR[K]))
    GD = np.zeros((nbk, b, b), np.complex128)
    dGD = np.zeros((nbk, b, b), np.complex128)
    for K in range(nbk):
        Y = z * I - D[K]
        dY = -dD[K]
        if K > 0:
            Um = U[K - 1]; dUm = dU[K - 1]
            Y = Y - _mm(_dag(Um), _mm(gL[K - 1], Um))
            dY = dY - _mm(_dag(dUm), _mm(gL[K - 1], Um)) - _mm(_dag(Um), _mm(dgL[K - 1], Um)) \
                 - _mm(_dag(Um), _mm(gL[K - 1], dUm))
        if K < nbk - 1:
            Uk = U[K]; dUk = dU[K]
            Y = Y - _mm(Uk, _mm(gR[K + 1], _dag(Uk)))
            dY = dY - _mm(dUk, _mm(gR[K + 1], _dag(Uk))) - _mm(Uk, _mm(dgR[K + 1], _dag(Uk))) \
                 - _mm(Uk, _mm(gR[K + 1], _dag(dUk)))
        GD[K] = _inv(Y)
        dGD[K] = -_mm(GD[K], _mm(dY, GD[K]))
    GU = np.zeros((nbk - 1, b, b), np.complex128)
    GLo = np.zeros((nbk - 1, b, b), np.complex128)
    dGU = np.zeros((nbk - 1, b, b), np.complex128)
    dGLo = np.zeros((nbk - 1, b, b), np.complex128)
    for K in range(nbk - 1):
        GU[K] = _mm(GD[K], _mm(U[K], gR[K + 1]))
        dGU[K] = _mm(dGD[K], _mm(U[K], gR[K + 1])) + _mm(GD[K], _mm(dU[K], gR[K + 1])) \
                 + _mm(GD[K], _mm(U[K], dgR[K + 1]))
        GLo[K] = _mm(gR[K + 1], _mm(_dag(U[K]), GD[K]))
        dGLo[K] = _mm(dgR[K + 1], _mm(_dag(U[K]), GD[K])) + _mm(gR[K + 1], _mm(_dag(dU[K]), GD[K])) \
                  + _mm(gR[K + 1], _mm(_dag(U[K]), dGD[K]))
    return GD, GU, GLo, dGD, dGU, dGLo


@nb.njit(parallel=True, cache=True)
def sign_band_tangent_rgf(D, U, dD, dU, zs, cs, lin_c):
    """Tangent of sign_band_rgf along dh (fixed mu, fixed occupations): dS band blocks."""
    nbk, b, _ = D.shape
    npole = zs.shape[0]
    SDp = np.zeros((npole, nbk, b, b), np.complex128)
    SUp = np.zeros((npole, nbk - 1, b, b), np.complex128)
    for p in nb.prange(npole):
        GD, GU, GLo, dGD, dGU, dGLo = _rgf_tangent_one(zs[p], D, U, dD, dU)
        c = cs[p]
        for K in range(nbk):
            SDp[p, K] = c * (dGD[K] + _dag(dGD[K]))
        for K in range(nbk - 1):
            SUp[p, K] = c * (dGU[K] + _dag(dGLo[K]))
    SD = np.zeros((nbk, b, b), np.complex128)
    SU = np.zeros((nbk - 1, b, b), np.complex128)
    for p in range(npole):
        SD += SDp[p]
        SU += SUp[p]
    for K in range(nbk):
        SD[K] += lin_c * dD[K]
    for K in range(nbk - 1):
        SU[K] += lin_c * dU[K]
    return SD, SU


@nb.njit(parallel=True, cache=True)
def sign_window_rgf(D, U, zs, cs, lin_c, mu, K0, K1):
    """Full block window [K0, K1] (inclusive) of S = sum c (G + G^dag) + lin_c (h - mu)."""
    nbk, b, _ = D.shape
    nw = K1 - K0 + 1
    npole = zs.shape[0]
    Wp = np.zeros((npole, nw * b, nw * b), np.complex128)
    for p in nb.prange(npole):
        GD, GU, GLo, gL, gR = _rgf_one(zs[p], D, U)
        Gw = np.zeros((nw, nw, b, b), np.complex128)
        for K in range(K0, K1 + 1):
            Gw[K - K0, K - K0] = GD[K]
            # to the right: G_{K,L+1} = G_{K,L} U_L gR[L+1]; to the left: G_{L+1,K} = gR[L+1] U_L^dag G_{L,K}
            for L in range(K, K1):
                Gw[K - K0, L + 1 - K0] = _mm(Gw[K - K0, L - K0], _mm(U[L], gR[L + 1]))
                Gw[L + 1 - K0, K - K0] = _mm(gR[L + 1], _mm(_dag(U[L]), Gw[L - K0, K - K0]))
        c = cs[p]
        for I1 in range(nw):
            for J1 in range(nw):
                for x in range(b):
                    for y in range(b):
                        g = Gw[I1, J1, x, y]
                        gd = np.conj(Gw[J1, I1, y, x])
                        Wp[p, I1 * b + x, J1 * b + y] = c * (g + gd)
    W = np.zeros((nw * b, nw * b), np.complex128)
    for p in range(npole):
        W += Wp[p]
    for K in range(K0, K1 + 1):
        for x in range(b):
            for y in range(b):
                v = D[K, x, y]
                if x == y:
                    v = v - mu
                W[(K - K0) * b + x, (K - K0) * b + y] += lin_c * v
        if K < K1:
            for x in range(b):
                for y in range(b):
                    W[(K - K0) * b + x, (K + 1 - K0) * b + y] += lin_c * U[K, x, y]
                    W[(K + 1 - K0) * b + y, (K - K0) * b + x] += lin_c * np.conj(U[K, x, y])
    return W


def sign_quadrature(gmin, wmax, ds=0.25, tail=39.0):
    """Poles z = mu + i t_p, t_p = exp(s_p), trapezoid in s on [ln gmin - tail, ln wmax + tail]:
    sign(x) = (2/pi) int_0^inf x/(x^2+t^2) dt = -(1/pi) int (G(mu+it)+G^dag) dt.  Error ~ exp(-pi^2/ds)
    plus endpoint values ~ exp(-tail)."""
    s0 = np.log(gmin) - tail
    s1 = np.log(wmax) + tail
    n = int(np.ceil((s1 - s0) / ds)) + 1
    s = np.linspace(s0, s1, n)
    h = s[1] - s[0]
    t = np.exp(s)
    w = np.full(n, h)
    w[0] *= 0.5; w[-1] *= 0.5
    cs = -(1.0 / np.pi) * w * t           # coefficient of (G + G^dag) for that pole
    # tails: int_0^{t0} x/(x^2+t^2) dt ~ t0/x (negligible at exp(-tail)), int_{t1}^inf ~ x/t1: linear term
    lin_c = (2.0 / np.pi) / t[-1]
    return t, cs, lin_c


@nb.njit(cache=True)
def blocks_to_band(SD, SU, bw):
    """Convert band blocks (diagonal + first upper) to band storage (bw <= b)."""
    nbk, b, _ = SD.shape
    dim = nbk * b
    B = np.zeros((bw + 1, dim), np.complex128)
    for d in range(bw + 1):
        for i in range(dim - d):
            j = i + d
            K = i // b; p = i - K * b
            L = j // b; q = j - L * b
            if K == L:
                B[d, i] = SD[K, p, q]
            elif L == K + 1:
                B[d, i] = SU[K, p, q]
    return B


@nb.njit(cache=True)
def _neg_count_herm(S):
    """Number of negative eigenvalues of a small Hermitian matrix via LDL^T (Sylvester), with
    tiny-pivot replacement."""
    n = S.shape[0]
    A = S.copy()
    cnt = 0
    tiny = 1e-300
    for c in range(n):
        piv = A[c, c].real
        if abs(piv) < tiny:
            piv = -tiny
        if piv < 0:
            cnt += 1
        # Hermitian Schur complement: A[r,k] -= A[r,c] A[c,k] / piv
        for r in range(c + 1, n):
            f = A[r, c] / piv
            for k in range(c + 1, n):
                A[r, k] -= f * A[c, k]
    return cnt


@nb.njit(cache=True)
def sturm_count(D, U, sigma):
    """# eigenvalues of the block-tridiagonal Hermitian h below sigma (block LDL, Sylvester)."""
    nbk, b, _ = D.shape
    I = np.eye(b).astype(np.complex128)
    S = D[0] - sigma * I
    cnt = _neg_count_herm(S)
    for K in range(1, nbk):
        Um = U[K - 1]
        Si = _inv(S)
        S = D[K] - sigma * I - _mm(_dag(Um), _mm(Si, Um))
        # symmetrize against roundoff
        S = 0.5 * (S + _dag(S))
        cnt += _neg_count_herm(S)
    return cnt


def eig_by_index(D, U, idx, lo, hi, tol=1e-14, maxit=200):
    """The idx-th eigenvalue (0-based, ascending) of h by bisection on Sturm counts in [lo, hi]."""
    a, b = lo, hi
    it = 0
    while b - a > tol * max(1.0, abs(a) + abs(b)) and it < maxit:
        c = 0.5 * (a + b)
        if sturm_count(D, U, c) > idx:
            b = c
        else:
            a = c
        it += 1
    return 0.5 * (a + b)


# ---------------- Chebyshev propagation ----------------
def cheb_coeffs(dt, c, r, tol=1e-18, kmax=4000):
    x = dt * r
    k = np.arange(kmax)
    J = jv(k, x)
    # last index where |J| > tol (super-exponential decay beyond k ~ x)
    big = np.nonzero(np.abs(J) > tol)[0]
    K = int(big[-1]) + 3
    K = max(K, 3)
    a = 2.0 * ((-1j) ** k[:K]) * J[:K]
    a[0] = J[0]
    phase = np.exp(-1j * dt * c)
    return a.astype(np.complex128), phase, K


@nb.njit(cache=True, inline='always')
def _hx_block(HB, X, Y, c, r):
    """Y = ((h - c)/r) X for a column block X (dim, cb), h Hermitian band (bw+1, dim)."""
    bw1, dim = HB.shape
    cb = X.shape[1]
    for i in range(dim):
        for q in range(cb):
            Y[i, q] = (HB[0, i].real - c) * X[i, q]
        for d in range(1, bw1):
            if i + d < dim:
                hij = HB[d, i]
                if hij != 0:
                    for q in range(cb):
                        Y[i, q] += hij * X[i + d, q]
            if i - d >= 0:
                hji = np.conj(HB[d, i - d])
                if hji != 0:
                    for q in range(cb):
                        Y[i, q] += hji * X[i - d, q]
        for q in range(cb):
            Y[i, q] /= r


@nb.njit(parallel=True, cache=True)
def cheb_propagate(HB, Psi, a, phase, c, r, cb):
    """Return exp(-i dt h) Psi with Chebyshev coefficients a (already for dt), column blocks of cb."""
    dim, norb = Psi.shape
    K = a.shape[0]
    out = np.empty_like(Psi)
    nblk = (norb + cb - 1) // cb
    for bi in nb.prange(nblk):
        j0 = bi * cb
        j1 = min(norb, j0 + cb)
        w = j1 - j0
        T0 = np.empty((dim, w), np.complex128)
        T1 = np.empty((dim, w), np.complex128)
        T2 = np.empty((dim, w), np.complex128)
        acc = np.empty((dim, w), np.complex128)
        for i in range(dim):
            for q in range(w):
                T0[i, q] = Psi[i, j0 + q]
        _hx_block(HB, T0, T1, c, r)
        for i in range(dim):
            for q in range(w):
                acc[i, q] = a[0] * T0[i, q] + a[1] * T1[i, q]
        for k in range(2, K):
            _hx_block(HB, T1, T2, c, r)
            ak = a[k]
            for i in range(dim):
                for q in range(w):
                    t2 = 2.0 * T2[i, q] - T0[i, q]
                    T2[i, q] = t2
                    acc[i, q] += ak * t2
            T0, T1, T2 = T1, T2, T0
        for i in range(dim):
            for q in range(w):
                out[i, j0 + q] = phase * acc[i, q]
    return out


def propagate(HB, Psi, dt, cb=4, pad=1.02):
    lo, hi = _gersh(HB)
    c = 0.5 * (hi + lo)
    r = 0.5 * (hi - lo) * pad
    a, phase, K = cheb_coeffs(dt, c, r)
    return cheb_propagate(HB, np.ascontiguousarray(Psi), a, phase, c, r, cb), K


def _gersh(HB):
    bw1, dim = HB.shape
    rad = np.zeros(dim)
    for d in range(1, bw1):
        av = np.abs(HB[d, :dim - d])
        rad[:dim - d] += av
        rad[d:] += av
    diag = HB[0].real
    return float(np.min(diag - rad)), float(np.max(diag + rad))
