# m3r2_rows_v1.py -- manifest Secs. 4-5 (v3 Sec. 4 quadrature, v3 Sec. 6(a) tails):
# partial-wave Gaussian kernels K_0 (scalar) and K_1 (vector), the forward smoothing W_l, its
# exact discrete adjoint W_l^dagger (with the analytic tail beyond the dense grid, exterior branch
# N = 1, v = -sqrt(2 G M_I / y)), the discrete rows (trapezoid forward recursion for M with the
# implicit w, scalar Newton), the adjoint clock N_i = dE/d(varpi_i rho_i) and v_i = w_i N_i.
import numpy as np
import numba as nb
import scipy.sparse as sp
from scipy.special import erfc

SQ2PI = np.sqrt(2.0 * np.pi)


def K0(y, r, sig):
    y = np.asarray(y, float); r = np.asarray(r, float)
    out = np.empty(np.broadcast(y, r).shape)
    yy, rr = np.broadcast_arrays(y, r)
    pos = yy > 0
    x = yy[pos] * rr[pos] / sig**2
    out[pos] = rr[pos] / (SQ2PI * sig * yy[pos]) * np.exp(-(yy[pos] - rr[pos])**2 / (2 * sig**2)) * (-np.expm1(-2 * x))
    out[~pos] = 2 * rr[~pos]**2 / (SQ2PI * sig**3) * np.exp(-rr[~pos]**2 / (2 * sig**2))
    return out


def _g_small(x):
    # cosh x - sinh(x)/x = sum_{n>=1} x^{2n} 2n/(2n+1)!
    x2 = x * x
    s = np.zeros_like(x)
    term_coef = [1/3, 1/30, 1/840, 1/45360, 1/3991680, 1/518918400, 1/93405312000, 1/22230464256000]
    p = x2.copy()
    for c in term_coef:
        s += c * p
        p = p * x2
    return s


def K1(y, r, sig):
    y = np.asarray(y, float); r = np.asarray(r, float)
    yy, rr = np.broadcast_arrays(y, r)
    out = np.zeros(yy.shape)
    pos = yy > 0
    yp, rp = yy[pos], rr[pos]
    x = yp * rp / sig**2
    pref = rp / (SQ2PI * sig * yp)
    res = np.empty(x.shape)
    sm = x < 1.0
    # small x: 2 e^{-(y^2+r^2)/2s^2} g(x)
    res[sm] = 2.0 * np.exp(-(yp[sm]**2 + rp[sm]**2) / (2 * sig**2)) * _g_small(x[sm])
    lg = ~sm
    e2 = np.exp(-2 * x[lg])
    res[lg] = np.exp(-(yp[lg] - rp[lg])**2 / (2 * sig**2)) * ((1 + e2) - (1 - e2) / x[lg])
    out[pos] = pref * res
    return out


class Kernel:
    """Dense grid y_i = i h_d, i = 0..I, varpi_i = 4 pi y_i^2 h_d c_i (trapezoid c_0 = c_I = 1/2);
    sites r_n = n a, n = 1..ns; V_n = 4 pi r_n^2 a."""

    def __init__(self, ns, a, hd, sig, yend, cut=14.0):
        self.ns, self.a, self.hd, self.sig = ns, a, hd, sig
        I = int(round(yend / hd))
        self.I = I
        self.y = hd * np.arange(I + 1)
        c = np.ones(I + 1); c[0] = 0.5; c[-1] = 0.5
        self.varpi = 4 * np.pi * self.y**2 * hd * c
        self.r = a * np.arange(1, ns + 1, dtype=float)
        self.V = 4 * np.pi * self.r**2 * a
        rows, cols, v0, v1 = [], [], [], []
        span = cut * sig
        for n in range(ns):
            rn = self.r[n]
            i0 = max(0, int(np.floor((rn - span) / hd)))
            i1 = min(I, int(np.ceil((rn + span) / hd)))
            ii = np.arange(i0, i1 + 1)
            rows.append(ii); cols.append(np.full(ii.size, n))
            v0.append(a * K0(self.y[ii], rn, sig)); v1.append(a * K1(self.y[ii], rn, sig))
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        self.W0 = sp.csr_matrix((np.concatenate(v0), (rows, cols)), shape=(I + 1, ns))
        self.W1 = sp.csr_matrix((np.concatenate(v1), (rows, cols)), shape=(I + 1, ns))
        # adjoints: (W^dag F)_n = V_n^{-1} sum_i varpi_i W[i,n] F_i  (+ tail)
        Dv = sp.diags(self.varpi)
        self.W0T = (self.W0.T @ Dv).tocsr()
        self.W1T = (self.W1.T @ Dv).tocsr()
        self._tails()

    def _tails(self):
        """T0_n = a int_Y^inf 4 pi y^2 K_0(y,r_n) dy (closed form, erfc);
        tau1_n = a int_Y^inf 4 pi y^{3/2} K_1(y,r_n) dy (Gauss-Legendre to machine precision)."""
        Y = self.y[-1]; s = self.sig; a = self.a
        r = self.r
        def mom1(c):  # int_Y^inf y e^{-(y-c)^2/2s^2} dy
            z = (Y - c)
            return s**2 * np.exp(-z**2 / (2 * s**2)) + c * s * np.sqrt(np.pi / 2) * erfc(z / (np.sqrt(2) * s))
        T0 = a * 4 * np.pi * r / (SQ2PI * s) * (mom1(r) - mom1(-r))
        self.T0 = np.where(Y - r > 40 * s, 0.0, T0)
        xg, wg = np.polynomial.legendre.leggauss(64)
        tau1 = np.zeros(self.ns)
        near = np.nonzero(Y - r < 16 * s)[0]
        npan = 24
        edges = np.linspace(Y, Y + 20 * s, npan + 1)
        for n in near:
            tot = 0.0
            for p in range(npan):
                lo, hi = edges[p], edges[p + 1]
                yy = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
                tot += 0.5 * (hi - lo) * np.sum(wg * 4 * np.pi * yy**1.5 * K1(yy, r[n], s))
            tau1[n] = a * tot
        self.tau1 = tau1

    def fwd(self, rho, J):
        return self.W0 @ rho, self.W1 @ J

    def adj(self, N, v, MI, G):
        """Back-averaged site fields incl. the exterior tail: N_ext = 1, v_ext = -sqrt(2 G MI / y)."""
        Nb = (self.W0T @ N + self.T0 * 1.0) / self.V
        vb = (self.W1T @ v - np.sqrt(2 * G * max(MI, 0.0)) * self.tau1) / self.V
        return Nb, vb

    def tail_terms(self, rho, J):
        return float(self.T0 @ rho), float(self.tau1 @ J)


@nb.njit(cache=True)
def rows_forward(y, hd, rhol, Jl, G, tail_rho, tail_tau1J):
    """M_i (forward trapezoid, implicit w by scalar Newton to 1e-14), w_i, s_i, E and flags."""
    n = y.shape[0]
    M = np.zeros(n); w = np.zeros(n); s = np.zeros(n)
    flag = 0
    for i in range(n - 1):
        yi1 = y[i + 1]
        a4 = 4.0 * np.pi * yi1 * yi1
        Mg = M[i] + hd * s[i]
        if Mg <= 0:
            Mg = M[i] + 0.5 * hd * (s[i] + a4 * rhol[i + 1])
        for it in range(100):
            if Mg <= 0:
                flag |= 1
                Mg = 0.0
                ww = 0.0
                ds = 0.0
            else:
                ww = -np.sqrt(2.0 * G * Mg / yi1)
                ds = a4 * Jl[i + 1] * ww / (2.0 * Mg)
            si1 = a4 * (rhol[i + 1] + ww * Jl[i + 1])
            F = Mg - M[i] - 0.5 * hd * (s[i] + si1)
            Fp = 1.0 - 0.5 * hd * ds
            dM = -F / Fp
            Mg += dM
            if abs(dM) <= 1e-15 * max(abs(Mg), 1e-300):
                break
        if Mg <= 0:
            flag |= 1
            ww = 0.0
        else:
            ww = -np.sqrt(2.0 * G * Mg / yi1)
        M[i + 1] = Mg
        w[i + 1] = ww
        s[i + 1] = a4 * (rhol[i + 1] + ww * Jl[i + 1])
    MI = M[n - 1]
    E = MI + tail_rho - np.sqrt(2.0 * G * max(MI, 0.0)) * tail_tau1J
    return M, w, s, E, flag


@nb.njit(cache=True)
def rows_adjoint(y, hd, M, w, Jl, G, tail_tau1J):
    """lambda recursion; N_i = (lambda_i + lambda_{i+1})/2, N_I = lambda_I, N_0 from N'(0)=0; v = w N."""
    n = y.shape[0]
    I = n - 1
    sp_ = np.zeros(n)
    for j in range(1, n):
        if M[j] > 0:
            sp_[j] = 4.0 * np.pi * y[j] * y[j] * Jl[j] * w[j] / (2.0 * M[j])
    lam = np.zeros(n + 1)
    MI = M[I]
    Tp = 0.0
    if MI > 0:
        Tp = -tail_tau1J * np.sqrt(2.0 * G) / (2.0 * np.sqrt(MI))
    lam[I] = (1.0 + Tp) / (1.0 - 0.5 * hd * sp_[I])
    for j in range(I - 1, 0, -1):
        lam[j] = lam[j + 1] * (1.0 + 0.5 * hd * sp_[j]) / (1.0 - 0.5 * hd * sp_[j])
    N = np.zeros(n)
    for i in range(1, I):
        N[i] = 0.5 * (lam[i] + lam[i + 1])
    N[I] = lam[I]
    N[0] = (4.0 * N[1] - N[2]) / 3.0
    v = w * N
    return N, v, lam


@nb.njit(cache=True)
def cut_adjoint(y, hd, M, w, Jl, c):
    """dM_c/d(varpi_i rho_i) and dM_c/d(varpi_i J_i) for i <= c (adjoint of the forward recursion
    truncated at node c): returns (Ac_rho, Ac_J) on the full grid (zero beyond c)."""
    n = y.shape[0]
    sp_ = np.zeros(n)
    for j in range(1, c + 1):
        if M[j] > 0:
            sp_[j] = 4.0 * np.pi * y[j] * y[j] * Jl[j] * w[j] / (2.0 * M[j])
    lam = np.zeros(n + 1)
    lam[c] = 1.0 / (1.0 - 0.5 * hd * sp_[c])
    for j in range(c - 1, 0, -1):
        lam[j] = lam[j + 1] * (1.0 + 0.5 * hd * sp_[j]) / (1.0 - 0.5 * hd * sp_[j])
    Ar = np.zeros(n)
    for i in range(1, c):
        Ar[i] = 0.5 * (lam[i] + lam[i + 1])
    Ar[c] = 0.5 * lam[c]          # node c enters only F_c: (hd/2) 4pi y^2 lam_c / varpi_c with full varpi
    AJ = Ar * w
    return Ar, AJ, lam


def hermite_eval(y0, h, f0, f1, d0, d1, t):
    """Cubic Hermite on [y0, y0+h] at y0 + t h (t in [0,1]); returns value and derivative."""
    h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t; h01 = -2*t**3 + 3*t**2; h11 = t**3 - t**2
    val = h00*f0 + h10*h*d0 + h01*f1 + h11*h*d1
    dh00 = 6*t**2 - 6*t; dh10 = 3*t**2 - 4*t + 1; dh01 = -6*t**2 + 6*t; dh11 = 3*t**2 - 2*t
    der = (dh00*f0 + dh10*h*d0 + dh01*f1 + dh11*h*d1) / h
    return val, der


def dense_roots(y, M, s, G):
    """All zeros of g(y) = 2 G M(y) - y on the cubic Hermite interpolant of M (slopes s = M').
    Returns list of (root, t-index c, t, M'(root), sign of crossing)."""
    g = 2 * G * M - y
    out = []
    for c in range(1, len(y) - 1):
        if g[c] == 0 or (g[c] > 0) != (g[c + 1] > 0):
            h = y[c + 1] - y[c]
            lo, hi = 0.0, 1.0
            glo = g[c]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                Mv, _d = hermite_eval(y[c], h, M[c], M[c + 1], s[c], s[c + 1], mid)
                gm = 2 * G * Mv - (y[c] + mid * h)
                if (gm > 0) == (glo > 0):
                    lo, glo = mid, gm
                else:
                    hi = mid
                if hi - lo < 1e-16:
                    break
            t = 0.5 * (lo + hi)
            Mv, Md = hermite_eval(y[c], h, M[c], M[c + 1], s[c], s[c + 1], t)
            out.append((y[c] + t * h, c, t, Md, 1 if g[c] > 0 else -1, Mv))
    return out
