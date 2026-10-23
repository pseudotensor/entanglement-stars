#!/usr/bin/env python
"""banded_eigh_gpu_v1.py (busi lane, reviewer A redesign prototype, REVIEWA_algorithm_redesign_v1 proposal R1):
an O(n^2) eigensolver for the engines' clocked chain h_m (complex Hermitian, half-bandwidth 4 in the interleaved
site/component index) in place of the dense O(n^3) torch.linalg.eigh.

Method (numba.cuda; no dense n x n object is ever formed):
  (1) eigenvalue isolation by multisection on Sylvester's inertia: 8 threads per eigenvalue evaluate the count of
      eigenvalues below 8 interior points of the bracket (the count is the number of negative pivots of the no-pivoting
      LDL^H of h - sigma on the band, a 4-column window held in registers; a pivot below pivmin is replaced by -pivmin as
      LAPACK's dstebz does, i.e. the counted matrix is perturbed by at most pivmin); the bracket shrinks 9x per step; once
      it is narrower than iso_rel (relative) the counts at its ends test that it holds exactly one eigenvalue, else the
      search continues to the rounding width (a cluster);
  (2) the eigenpair by inverse iteration with Rayleigh-quotient refinement, one thread per eigenvalue: a register-blocked
      banded LU with partial pivoting (the 5 x 9 active window in registers on a zero-padded matrix so that every loop has
      constant bounds; the factors streamed to a per-thread global scratch in LAPACK's gbtrf layout), a pseudo-random
      start vector, the solve, the Rayleigh quotient, then further rounds (refactor at the refined shift, solve, Rayleigh
      quotient) while the shift still moves by more than rq_tol * ||h||_G (cubic convergence; 2 rounds for well separated
      levels, 3 to 4 for pairs closer than about 1e-6 ||h||);
  (3) the particle-hole lift: for the massive chain C = S K anticommutes with h_m (S = sum_n (-1)^n sigma_y), so only the
      n_mat/2 positive eigenpairs are computed; the negative half is e -> -e, u -> S conj(u), exact;
  (4) clusters (consecutive eigenvalues closer than cluster_tol * ||h||_G) re-orthonormalized by a Rayleigh-Ritz step, and
      one banded Lowdin polish U <- U (I - (G - I)/2) on the energy-banded Gram matrix (|i - j| <= width), which takes the
      inverse iteration's eps ||h||/gap non-orthogonality to its square.
The accuracy class is that of a backward-stable eigensolver: eigenvalues to a few eps ||h||, eigenvectors with residuals
eps ||h|| and the same freedom inside near-degenerate pairs as the dense solver. It is not bitwise with cusolver.

API: eigh_banded(band, n, device, ...) -> (e, U, info); band_from_csr(h) for the 5 upper diagonals (band[0] the diagonal,
band[m][j] = h[j, j + m]); solve_shifted(band, n, sigs, B) the batched shifted banded solves (the Sternheimer route, R3).
"""
import math
import time

import numpy as np
import numba
from numba import cuda
import torch

KL = 4            # half-bandwidth of the clocked chain in the interleaved index
LDAB = 13         # 2 KL + KU + 1 rows of the gbtrf band layout
NW = 9            # the active window's columns (KL + KU + 1)
PAD = 16          # zero-padded columns beyond n (constant-bound loops)
MS = 8            # threads per eigenvalue in the multisection
TPB = 64          # threads per block


def band_from_csr(h):
    """the 5 upper diagonals of a Hermitian scipy sparse matrix with half-bandwidth 4: band[m, j] = h[j, j + m] (band[0] real)."""
    n = h.shape[0]
    band = np.zeros((KL + 1, n), dtype=np.complex128)
    for m in range(KL + 1):
        d = np.asarray(h.diagonal(m)).ravel()
        band[m, :n - m] = d
    band[0] = band[0].real
    return band


def gershgorin_band(band):
    n = band.shape[1]
    s = np.abs(band[0].real).copy()
    for m in range(1, KL + 1):
        a = np.abs(band[m, :n - m])
        s[:n - m] += a
        s[m:] += a
    return float(s.max())


def _padded(band, n):
    bp = np.zeros((KL + 1, n + PAD), dtype=np.complex128)
    bp[:, :n] = band[:, :n]
    bp[0, n:] = 1.0          # the padded block is the identity (decoupled; harmless pivots)
    return np.ascontiguousarray(bp)


@cuda.jit(device=True, inline=True)
def _csq(z):
    return z.real * z.real + z.imag * z.imag


@cuda.jit(device=True)
def _sturm_count(band, n, sigma, pivmin):
    """the number of eigenvalues of the band matrix (columns < n) below sigma: negative pivots of the no-pivoting LDL^H, the 4-column
    window in registers (p_km = L[(j - k) + m, j - k], d_k = d_{j-k}); the next column's entries loaded one iteration ahead."""
    zero = complex(0.0, 0.0)
    p11 = zero; p12 = zero; p13 = zero; p14 = zero
    p21 = zero; p22 = zero; p23 = zero; p24 = zero
    p31 = zero; p32 = zero; p33 = zero; p34 = zero
    p41 = zero; p42 = zero; p43 = zero; p44 = zero
    d1 = 0.0; d2 = 0.0; d3 = 0.0; d4 = 0.0
    cnt = 0
    b0n = band[0, 0].real; b1n = band[1, 0]; b2n = band[2, 0]; b3n = band[3, 0]; b4n = band[4, 0]
    for j in range(n):
        b0 = b0n; b1 = b1n.conjugate(); b2 = b2n.conjugate(); b3 = b3n.conjugate(); b4 = b4n.conjugate()
        b0n = band[0, j + 1].real; b1n = band[1, j + 1]; b2n = band[2, j + 1]; b3n = band[3, j + 1]; b4n = band[4, j + 1]
        dj = b0 - sigma - _csq(p11) * d1 - _csq(p22) * d2 - _csq(p33) * d3 - _csq(p44) * d4
        if abs(dj) < pivmin:
            dj = -pivmin
        if dj < 0.0:
            cnt += 1
        rd = 1.0 / dj
        c1 = p11.conjugate() * d1; c2 = p22.conjugate() * d2; c3 = p33.conjugate() * d3
        l1 = (b1 - p12 * c1 - p23 * c2 - p34 * c3) * rd
        l2 = (b2 - p13 * c1 - p24 * c2) * rd
        l3 = (b3 - p14 * c1) * rd
        l4 = b4 * rd
        p41 = p31; p42 = p32; p43 = p33; p44 = p34
        p31 = p21; p32 = p22; p33 = p23; p34 = p24
        p21 = p11; p22 = p12; p23 = p13; p24 = p14
        p11 = l1; p12 = l2; p13 = l3; p14 = l4
        d4 = d3; d3 = d2; d2 = d1; d1 = dj
    return cnt


@cuda.jit
def _msect_kernel(band, n, kidx, lo, hi, nsteps, pivmin, tol_rel, iso_rel, out_lo, out_hi, out_iso, out_steps):
    """MS threads per eigenvalue (TPB // MS eigenvalues per block): multisection of the bracket of eigenvalue kidx[e] on the counts at
    MS interior points, the isolation test (the counts at the ends) once the bracket is narrower than iso_rel; the whole block iterates
    in lock step (the barriers), finished groups idle."""
    sh = cuda.shared.array(TPB, numba.int32)
    flag = cuda.shared.array(1, numba.int32)
    tid = cuda.threadIdx.x
    g = tid // MS
    q = tid - g * MS
    e = cuda.blockIdx.x * (TPB // MS) + g
    T = kidx.shape[0]
    active = e < T
    k = 0; a = 0.0; b = 1.0
    if active:
        k = kidx[e]; a = lo[e]; b = hi[e]
    done = not active
    iso = 0
    steps = 0
    for it in range(nsteps):
        if tid == 0:
            flag[0] = 0
        cuda.syncthreads()
        if not done:
            flag[0] = 1
        cuda.syncthreads()
        if flag[0] == 0:
            break
        isophase = False
        pt = 0.0
        if not done:
            w = b - a
            sc = max(abs(a), abs(b))
            if w <= tol_rel * sc:
                done = True
            else:
                isophase = (iso == 0) and (w <= iso_rel * sc)
                if isophase:
                    if q == 0:
                        pt = a
                    elif q == 1:
                        pt = b
                    else:
                        pt = a + (q - 1) * w / (MS - 1)
                else:
                    pt = a + (q + 1) * w / (MS + 1)
                sh[tid] = _sturm_count(band, n, pt, pivmin)
                steps += 1
        cuda.syncthreads()
        if not done:
            base = g * MS
            if isophase:
                ca = sh[base]; cb = sh[base + 1]
                na = a; nb = b
                for r in range(2, MS):
                    p_r = a + (r - 1) * (b - a) / (MS - 1)
                    if sh[base + r] > k:
                        if p_r < nb:
                            nb = p_r
                    else:
                        if p_r > na:
                            na = p_r
                if ca == k and cb == k + 1:
                    iso = 1
                    done = True
                else:
                    iso = -1
                a = na; b = nb
            else:
                na = a; nb = b
                for r in range(MS):
                    p_r = a + (r + 1) * (b - a) / (MS + 1)
                    if sh[base + r] > k:
                        if p_r < nb:
                            nb = p_r
                    else:
                        if p_r > na:
                            na = p_r
                a = na; b = nb
        cuda.syncthreads()
    if active and q == 0:
        out_lo[e] = a
        out_hi[e] = b
        out_iso[e] = 1 if iso == 1 else 0
        out_steps[e] = steps


def _lu_source():
    """the fully unrolled register-resident banded LU sweep as source text (45 window entries a_rc, 5 right-hand-side entries y_r):
    per column: the pivot among rows 0..4 of column 0, the swap, the elimination of rows 1..4 over columns 1..8, the stores of row 0
    (U, 9 entries) and the multipliers (L, 4 entries) to W and of y0 to X, the window shift and the load of the next row from the
    padded band (entries beyond n are the identity's, zero off the diagonal)."""
    L = []
    A = lambda r, c: "a%d%d" % (r, c)
    L.append("def _lu_sweep(band, n, npad, sigma, W, piv, X, t, pivmin):")
    for r in range(5):
        for c in range(NW):
            if r == c:
                L.append("    %s = complex(band[0, %d].real - sigma, 0.0)" % (A(r, c), r))
            elif r < c:
                L.append("    %s = band[%d, %d]" % (A(r, c), c - r, r) if c - r <= KL else "    %s = complex(0.0, 0.0)" % A(r, c))
            else:
                L.append("    %s = band[%d, %d].conjugate()" % (A(r, c), r - c, c) if r - c <= KL else "    %s = complex(0.0, 0.0)" % A(r, c))
    for r in range(5):
        L.append("    y%d = X[%d, t]" % (r, r))
    L.append("    for j in range(npad - %d):" % NW)
    L.append("        p = 0")
    L.append("        amax = abs(a00)")
    for r in range(1, 5):
        L.append("        tq = abs(%s)" % A(r, 0))
        L.append("        if tq > amax:")
        L.append("            amax = tq")
        L.append("            p = %d" % r)
    L.append("        piv[j, t] = j + p")
    for r in range(1, 5):
        L.append("        %s p == %d:" % ("if" if r == 1 else "elif", r))
        for c in range(NW):
            L.append("            tq = %s; %s = %s; %s = tq" % (A(0, c), A(0, c), A(r, c), A(r, c)))
        L.append("            tq = y0; y0 = y%d; y%d = tq" % (r, r))
    L.append("        d = a00")
    L.append("        if abs(d) < pivmin:")
    L.append("            d = complex(pivmin, 0.0)")
    L.append("            a00 = d")
    L.append("        rd = 1.0 / d")
    for r in range(1, 5):
        L.append("        l%d = %s * rd" % (r, A(r, 0)))
        for c in range(1, NW):
            L.append("        %s -= l%d * %s" % (A(r, c), r, A(0, c)))
        L.append("        y%d -= l%d * y0" % (r, r))
    for c in range(NW):
        L.append("        W[%d, j + %d, t] = %s" % (8 - c, c, A(0, c)))
    for r in range(1, 5):
        L.append("        W[%d, j, t] = l%d" % (8 + r, r))
    L.append("        X[j, t] = y0")
    for r in range(4):
        for c in range(NW - 1):
            L.append("        %s = %s" % (A(r, c), A(r + 1, c + 1)))
        L.append("        %s = complex(0.0, 0.0)" % A(r, NW - 1))
        L.append("        y%d = y%d" % (r, r + 1))
    L.append("        i5 = j + 5")
    for c in range(4):
        L.append("        a4%d = band[%d, j + %d].conjugate()" % (c, 4 - c, 1 + c))
    L.append("        if i5 < n:")
    L.append("            a44 = complex(band[0, i5].real - sigma, 0.0)")
    L.append("        else:")
    L.append("            a44 = complex(1.0, 0.0)")
    for c in range(5, NW):
        L.append("        a4%d = band[%d, i5]" % (c, c - 4))
    L.append("        y4 = X[i5, t]")
    L.append("    for j in range(npad - %d, npad):" % NW)
    L.append("        piv[j, t] = j")
    L.append("        W[8, j, t] = complex(1.0, 0.0)")
    L.append("        X[j, t] = complex(0.0, 0.0)")
    return "\n".join(L) + "\n"


_ns = {"complex": complex, "abs": abs, "range": range}
exec(_lu_source(), _ns)
_lu_sweep = cuda.jit(device=True)(_ns["_lu_sweep"])


@cuda.jit(device=True)
def _back_solve(n, npad, W, X, t):
    """U X = Y with the streamed U (row j at W[8 - c, j + c], c = 0..8); the next row's U entries loaded one column ahead."""
    j = n - 1
    u0 = W[8, j, t]; u1 = W[7, j + 1, t]; u2 = W[6, j + 2, t]; u3 = W[5, j + 3, t]; u4 = W[4, j + 4, t]
    u5 = W[3, j + 5, t]; u6 = W[2, j + 6, t]; u7 = W[1, j + 7, t]; u8 = W[0, j + 8, t]
    for j in range(n - 1, -1, -1):
        c0 = u0; c1 = u1; c2 = u2; c3 = u3; c4 = u4; c5 = u5; c6 = u6; c7 = u7; c8 = u8
        jn = j - 1 if j > 0 else 0
        u0 = W[8, jn, t]; u1 = W[7, jn + 1, t]; u2 = W[6, jn + 2, t]; u3 = W[5, jn + 3, t]; u4 = W[4, jn + 4, t]
        u5 = W[3, jn + 5, t]; u6 = W[2, jn + 6, t]; u7 = W[1, jn + 7, t]; u8 = W[0, jn + 8, t]
        acc = X[j, t] - c1 * X[j + 1, t] - c2 * X[j + 2, t] - c3 * X[j + 3, t] - c4 * X[j + 4, t] - c5 * X[j + 5, t] - c6 * X[j + 6, t] - c7 * X[j + 7, t] - c8 * X[j + 8, t]
        X[j, t] = acc / c0


@cuda.jit(device=True)
def _band_lu_solve(band, n, npad, sigma, W, piv, X, t, pivmin, A, Y):
    """factor (band - sigma) = P L U (the unrolled register sweep, the right-hand side in X carried through the forward elimination)
    and back-substitute; A and Y unused (kept for the callers' signature)."""
    _lu_sweep(band, n, npad, sigma, W, piv, X, t, pivmin)
    _back_solve(n, npad, W, X, t)


@cuda.jit(device=True)
def _band_resolve(n, npad, W, piv, X, t):
    """solve with the stored factors for the right-hand side in X[:, t] (zero beyond n); the multipliers loaded one column ahead."""
    m1 = W[9, 0, t]; m2 = W[10, 0, t]; m3 = W[11, 0, t]; m4 = W[12, 0, t]; pn = piv[0, t]
    for j in range(npad - NW):
        p = pn; l1 = m1; l2 = m2; l3 = m3; l4 = m4
        m1 = W[9, j + 1, t]; m2 = W[10, j + 1, t]; m3 = W[11, j + 1, t]; m4 = W[12, j + 1, t]; pn = piv[j + 1, t]
        if p != j:
            tmp = X[j, t]; X[j, t] = X[p, t]; X[p, t] = tmp
        xj = X[j, t]
        X[j + 1, t] -= l1 * xj
        X[j + 2, t] -= l2 * xj
        X[j + 3, t] -= l3 * xj
        X[j + 4, t] -= l4 * xj
    _back_solve(n, npad, W, X, t)


@cuda.jit(device=True)
def _normalize(X, n, t):
    nrm = 0.0
    for j in range(n):
        nrm += _csq(X[j, t])
    nrm = 1.0 / math.sqrt(nrm)
    for j in range(n):
        X[j, t] = X[j, t] * nrm


@cuda.jit(device=True)
def _rayleigh(band, n, X, t):
    """the Rayleigh quotient x^H h x of the unit vector in X[:, t] and the residual norm ||h x - lam x|| (two sweeps)."""
    acc = 0.0
    for j in range(n):
        hx = band[0, j] * X[j, t]
        for m in range(1, KL + 1):
            hx += band[m, j] * X[j + m, t]
            if j - m >= 0:
                hx += band[m, j - m].conjugate() * X[j - m, t]
        acc += (X[j, t].conjugate() * hx).real
    rn = 0.0
    for j in range(n):
        hx = band[0, j] * X[j, t]
        for m in range(1, KL + 1):
            hx += band[m, j] * X[j + m, t]
            if j - m >= 0:
                hx += band[m, j - m].conjugate() * X[j - m, t]
        rn += _csq(hx - acc * X[j, t])
    return acc, math.sqrt(rn)


@cuda.jit
def _invit_kernel(band, n, npad, sig, W, piv, X, nextra_max, res_tol, pivmin, seed, out_lam, out_rounds, out_res):
    """per thread t: the eigenpair at the shift sig[t]: one banded factorization, the solve from a pseudo-random start, a re-solve with
    the stored factors, the Rayleigh quotient and the residual; further re-solves (up to nextra_max) while the residual exceeds res_tol
    (a close pair: each re-solve multiplies the neighbour's contamination by |lam - sig| / gap)."""
    t = cuda.grid(1)
    if t >= sig.shape[0]:
        return
    A = cuda.local.array((5, NW), numba.complex128)
    Y = cuda.local.array(5, numba.complex128)
    s = sig[t]
    for j in range(npad):
        if j < n:
            X[j, t] = complex(math.sin(0.7 * j + 1.3 * t + seed), math.cos(1.1 * j - 0.9 * t + 0.5 * seed))
        else:
            X[j, t] = complex(0.0, 0.0)
    _band_lu_solve(band, n, npad, s, W, piv, X, t, pivmin, A, Y)
    _normalize(X, n, t)
    _band_resolve(n, npad, W, piv, X, t)
    _normalize(X, n, t)
    lam, rn = _rayleigh(band, n, X, t)
    nr = 2
    for r in range(nextra_max):
        if rn <= res_tol:
            break
        _band_resolve(n, npad, W, piv, X, t)
        _normalize(X, n, t)
        lam, rn = _rayleigh(band, n, X, t)
        nr += 1
    out_lam[t] = lam
    out_rounds[t] = nr
    out_res[t] = rn


@cuda.jit
def _solve_kernel(band, n, npad, sig, W, piv, X, pivmin):
    """per thread t: X[:, t] <- (band - sig[t])^{-1} X[:, t] (the given right-hand side, zero beyond n)."""
    t = cuda.grid(1)
    if t >= sig.shape[0]:
        return
    A = cuda.local.array((5, NW), numba.complex128)
    Y = cuda.local.array(5, numba.complex128)
    _band_lu_solve(band, n, npad, sig[t], W, piv, X, t, pivmin, A, Y)


def band_matvec(band_t, V):
    """h @ V for the band (torch tensors on one device; band_t[m, j] = h[j, j + m]), V (n, k)."""
    n = V.shape[0]
    out = band_t[0][:, None] * V
    for m in range(1, KL + 1):
        a = band_t[m, :n - m][:, None]
        out[:n - m] += a * V[m:]
        out[m:] += a.conj() * V[:n - m]
    return out


def ph_lift(U):
    """S conj(U) with S = sum_n (-1)^n sigma_y on the interleaved index: (S x)_{2i} = -i (-1)^i x_{2i+1}, (S x)_{2i+1} = i (-1)^i x_{2i}."""
    n = U.shape[0]
    i = torch.arange(n // 2, device=U.device)
    sgn = torch.where(i % 2 == 0, 1.0, -1.0).to(torch.complex128)
    V = torch.empty_like(U)
    Uc = U.conj()
    V[0::2] = (-1j) * sgn[:, None] * Uc[1::2]
    V[1::2] = (1j) * sgn[:, None] * Uc[0::2]
    return V


def lowdin_polish(U, width=48):
    """one banded Lowdin step: U <- U (I - (G - I)/2) with G = U^H U restricted to |i - j| <= width (the vectors are energy ordered and
    the non-orthogonality of inverse iteration lives on near neighbours); the overlap error goes to its square."""
    n, T = U.shape
    out = U.clone()
    for off in range(1, width + 1):
        g = (U[:, :T - off].conj() * U[:, off:]).sum(dim=0)          # G[i, i + off] = u_i^H u_{i+off}
        out[:, :T - off] -= 0.5 * g.conj()[None, :] * U[:, off:]     # column i: -(1/2) G[i+off, i] u_{i+off} = -(1/2) conj(g_i) u_{i+off}
        out[:, off:] -= 0.5 * g[None, :] * U[:, :T - off]            # column i + off: -(1/2) G[i, i+off] u_i
    nrm = torch.linalg.norm(out, dim=0)
    return out / nrm[None, :]


def solve_shifted(band, n, sigs, B, batch=2048, pivmin_rel=64.0):
    """X[:, t] = (band - sigs[t])^{-1} B[:, t] by the banded LU with partial pivoting, one thread per column. B a torch tensor on the
    CUDA device (n, T) complex128 (the result is a torch tensor on the same device, no host round trip) or a numpy array (then numpy).
    A shift at an eigenvalue gives the pivmin-regularized solve (the caller projects the singular direction out)."""
    eps = float(np.finfo(float).eps)
    G = gershgorin_band(band)
    pivmin = pivmin_rel * eps * G
    bp = _padded(band, n); npad = n + PAD
    bd = cuda.to_device(bp)
    T = B.shape[1]
    on_device = hasattr(B, "device") and str(getattr(B, "device")).startswith("cuda")
    if on_device:
        out = torch.empty((n, T), dtype=torch.complex128, device=B.device)
    else:
        out = np.empty((n, T), dtype=np.complex128)
    for b0 in range(0, T, batch):
        b1 = min(T, b0 + batch); Tb = b1 - b0
        W = cuda.device_array((LDAB, npad, Tb), np.complex128)
        piv = cuda.device_array((npad, Tb), np.int32)
        if on_device:
            Xt = torch.zeros((npad, Tb), dtype=torch.complex128, device=B.device)
            Xt[:n] = B[:, b0:b1]
            torch.cuda.synchronize(B.device)
            X = cuda.as_cuda_array(Xt)
        else:
            Xh = np.zeros((npad, Tb), dtype=np.complex128); Xh[:n] = B[:, b0:b1]
            X = cuda.to_device(Xh)
        d_sig = cuda.to_device(np.ascontiguousarray(np.asarray(sigs[b0:b1], dtype=np.float64)))
        _solve_kernel[(Tb + TPB - 1) // TPB, TPB](bd, n, npad, d_sig, W, piv, X, pivmin)
        cuda.synchronize()
        if on_device:
            out[:, b0:b1] = Xt[:n]
        else:
            out[:, b0:b1] = X.copy_to_host()[:n]
        del W, piv, X
    return out


def eigh_banded(band, n, device, half="ph", nsteps=40, nextra_max=4, batch=2048, cluster_tol=1e-9, pivmin_rel=64.0, iso_rel=1e-12, res_tol_rel=200.0,
                polish=16, timing=None, seed=0.0):
    """eigenpairs of the banded Hermitian matrix. half="ph": the n/2 positive eigenpairs by the kernels and the negative half by the
    particle-hole lift (massive chain); half="all": every eigenpair by the kernels (no symmetry assumed). Returns (e, U, info)."""
    tm = {} if timing is None else timing
    t0 = time.time()
    eps = float(np.finfo(float).eps)
    G = gershgorin_band(band)
    pivmin = pivmin_rel * eps * G
    bp = _padded(band, n); npad = n + PAD
    bd = cuda.to_device(bp)
    if half == "ph":
        kidx = np.arange(n // 2, n, dtype=np.int64)
        lo0 = np.zeros(len(kidx)); hi0 = np.full(len(kidx), G)
    else:
        kidx = np.arange(n, dtype=np.int64)
        lo0 = np.full(len(kidx), -G); hi0 = np.full(len(kidx), G)
    T = len(kidx)
    d_k = cuda.to_device(kidx); d_lo = cuda.to_device(lo0); d_hi = cuda.to_device(hi0)
    o_lo = cuda.device_array(T, np.float64); o_hi = cuda.device_array(T, np.float64); o_iso = cuda.device_array(T, np.int32); o_st = cuda.device_array(T, np.int32)
    cuda.synchronize(); t1 = time.time(); tm["setup_s"] = t1 - t0
    nblk = (T + (TPB // MS) - 1) // (TPB // MS)
    _msect_kernel[nblk, TPB](bd, n, d_k, d_lo, d_hi, nsteps, pivmin, 4.0 * eps, iso_rel, o_lo, o_hi, o_iso, o_st)
    cuda.synchronize(); t2 = time.time(); tm["bisect_s"] = t2 - t1
    lo = o_lo.copy_to_host(); hi = o_hi.copy_to_host(); iso = o_iso.copy_to_host(); st = o_st.copy_to_host()
    sig = 0.5 * (lo + hi)
    dev = torch.device(device)
    Up = torch.empty((n, T), dtype=torch.complex128, device=dev)
    lam_all = np.empty(T); rounds = np.empty(T, np.int32); res_k = np.empty(T)
    for b0 in range(0, T, batch):
        b1 = min(T, b0 + batch)
        Tb = b1 - b0
        W = cuda.device_array((LDAB, npad, Tb), np.complex128)
        piv = cuda.device_array((npad, Tb), np.int32)
        X = cuda.device_array((npad, Tb), np.complex128)
        o_lam = cuda.device_array(Tb, np.float64); o_r = cuda.device_array(Tb, np.int32); o_res = cuda.device_array(Tb, np.float64)
        d_sig = cuda.to_device(np.ascontiguousarray(sig[b0:b1]))
        _invit_kernel[(Tb + TPB - 1) // TPB, TPB](bd, n, npad, d_sig, W, piv, X, nextra_max, res_tol_rel * eps * G, pivmin, float(seed), o_lam, o_r, o_res)
        cuda.synchronize()
        Up[:, b0:b1] = torch.as_tensor(X, device=dev)[:n]          # zero-copy view of the device array, then a device copy
        lam_all[b0:b1] = o_lam.copy_to_host(); rounds[b0:b1] = o_r.copy_to_host(); res_k[b0:b1] = o_res.copy_to_host()
        del W, piv, X, o_lam, o_r, o_res
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t3 = time.time(); tm["invit_s"] = t3 - t2
    band_t = torch.as_tensor(np.ascontiguousarray(band[:, :n]), device=dev)
    lam = torch.as_tensor(lam_all, device=dev)
    n_cl = 0; cl_sizes = []
    j = 0
    while j < T - 1:
        k = j
        while k + 1 < T and (sig[k + 1] - sig[k]) < cluster_tol * G:
            k += 1
        if k > j:
            idx = torch.arange(j, k + 1, device=dev)
            B = Up[:, idx]
            Q, _R = torch.linalg.qr(B)
            Hs = Q.conj().T @ band_matvec(band_t, Q)
            Hs = 0.5 * (Hs + Hs.conj().T)
            w, V = torch.linalg.eigh(Hs)
            Up[:, idx] = Q @ V
            lam[idx] = w.real
            n_cl += 1; cl_sizes.append(k - j + 1)
        j = k + 1
    orth_before = float((Up[:, :-1].conj() * Up[:, 1:]).sum(dim=0).abs().max())
    if polish:
        Up = lowdin_polish(Up, width=int(polish))
        lam = (Up.conj() * band_matvec(band_t, Up)).real.sum(dim=0)
    HU = band_matvec(band_t, Up)
    res = torch.linalg.norm(HU - Up * lam[None, :], dim=0)
    if half == "ph":
        Un = ph_lift(Up)
        e = torch.cat([-lam.flip(0), lam])
        U = torch.cat([Un.flip(1), Up], dim=1)
    else:
        e = lam; U = Up
    order = torch.argsort(e)
    e = e[order]; U = U[:, order]
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t4 = time.time(); tm["post_s"] = t4 - t3; tm["total_s"] = t4 - t0
    info = {"gershgorin": G, "pivmin": pivmin, "bracket_max_rel": float(np.max((hi - lo) / np.maximum(np.abs(sig), 1e-300))), "isolated_fraction": float(np.mean(iso)),
            "count_sweeps_mean": float(np.mean(st)), "count_sweeps_max": int(np.max(st)), "rounds_mean": float(np.mean(rounds)), "rounds_max": int(np.max(rounds)),
            "solves_extra": int(np.sum(rounds - 2)), "kernel_residual_max": float(np.max(res_k)), "kernel_residual_max_rel": float(np.max(res_k)) / G,
            "residual_max": float(res.max()), "residual_max_rel": float(res.max()) / G, "clusters": n_cl, "cluster_sizes": cl_sizes, "n_threads": T,
            "neighbour_overlap_before_polish": orth_before, "shift_vs_rayleigh_max": float(np.max(np.abs(sig - lam_all)))}
    return e, U, info
