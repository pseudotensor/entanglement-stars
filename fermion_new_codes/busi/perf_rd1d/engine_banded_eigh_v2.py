#!/usr/bin/env python
"""engine_banded_eigh_v2.py (busi, analyst; PERF RD-1d; built by build_engine_banded_eigh_v2.py over engine_banded_eigh_v1, md5 73ee46fa) [rd1d memsafe]
v2 = v1 with a memory-safe dense route: _dense(hmT) runs v1's torch.linalg.eigh(hmT); only when that raises torch.cuda.OutOfMemoryError [rd1d memsafe]
does it (i) empty the caching allocator and retry, then (ii) park the device's other resident tensors (strided CUDA tensors reachable [rd1d memsafe]
from gc, by storage, >= 64 MiB, largest first) on the host byte for byte, retry after each, and restore every parked storage byte for [rd1d memsafe]
byte before returning (the same call on the same input: v1's dense result). Nothing else changes; MEMSAFE counts the events, and [rd1d memsafe]
stats_snapshot carries "memsafe" only when one happened. The flat member F256_s4's OOM (22 Oct) is the case. v1's docstring follows. [rd1d memsafe]
engine_banded_eigh_v1.py (busi, analyst; PERF RD-1 + RD-3, REVIEWA_algorithm_redesign_v1 sec. 1 and 3): the production module of the banded
O(n^2) GPU eigensolver of the clocked chain h_m (complex Hermitian, half-bandwidth 4 in the interleaved site/component index) and of the
shifted banded solves of the exact J.v (the Sternheimer route), for the Newton engine engine_collapse_v36nq_rd13 (--eigh banded,
--jvp sternheimer). Built from the redesign prototype banded_eigh_gpu_v1.py (md5 17453a40): the CUDA kernels (_sturm_count, _msect_kernel,
the unrolled register LU _lu_sweep, _back_solve, _band_resolve, _normalize, _rayleigh, _invit_kernel, _solve_kernel) and band_matvec,
ph_lift and lowdin_polish are the prototype's text unchanged; what is new here is the device handling (every device buffer is a torch
tensor handed to the kernels by __cuda_array_interface__, so the memory comes from torch's caching allocator on the engine's device and
numba allocates nothing), the guards and fallbacks below, and the engine entry points eigT_banded (the drop-in for engine_collapse_v5.eigT:
a dense torch matrix in, (e, U) out in the same layout) and sternheimer_X (the J.v's X matrix).

Method (numba.cuda; the spec's sec. 1.2): (1) eigenvalue isolation by multisection on Sylvester's inertia (8 threads per eigenvalue, a
pivot below pivmin = 64 eps ||h||_G replaced by -pivmin as LAPACK's dstebz does); (2) the eigenpair by inverse iteration with a
register-blocked banded LU with partial pivoting, a pseudo-random start, a re-solve with the stored factors, the Rayleigh quotient and the
residual, further re-solves (at most 4) while the residual exceeds 200 eps ||h||_G; (3) the particle-hole lift u_-E = S conj(u_E) for the
massive chain (only the n/2 positive eigenpairs computed); (4) consecutive eigenvalues closer than 1e-9 ||h||_G re-orthonormalized by a
Rayleigh-Ritz step on their span, one banded Lowdin polish U <- U (I - (G - I)/2) on |i - j| <= 16 and, beyond the spec, on every pair with |lam_i - lam_j| < 1e-3 ||h||_G
(LAPACK dstein's reorthogonalization criterion; lowdin_polish_window), the Rayleigh quotients, the sort.

Guards of eigT_banded (the conservative choices where the spec is silent; each failed guard is counted in STATS and named in the log):
  G1 the band is exact: every entry of the dense input outside |i - j| <= 4 is exactly 0.0 (the nonzero count of the whole matrix equals
     that of its 9 central diagonals), else the dense torch.linalg.eigh is used for that call (fallback "offband");
  G2 the band is read from the LOWER triangle (band[m][j] = conj(h[j + m, j]); the diagonal's real part), which is what torch.linalg.eigh
     (UPLO "L") reads;
  G3 particle-hole symmetry: ||h S conj(x) + S conj(h x)|| <= 1e-13 ||h||_G ||x|| on a fixed pseudo-random x; if it fails (the dust
     preparation's well hamiltonian h_m + U_well is not particle-hole symmetric) every eigenpair is computed by the kernels (half "all"),
     no lift (counted "no_ph");
  G4 in the lifted mode the smallest positive eigenvalue exceeds 1e-8 ||h||_G (the bracket [0, G] then holds exactly the n/2 upper
     eigenvalues and the lifted pair is separated far above the cluster tolerance); else the dense route (fallback "near_zero");
  G5 after the polish: every residual ||h u - lam u|| <= 1e-11 ||h||_G (the computed half), every neighbour overlap |u_i^H u_j|, 1 <= j - i
     <= 4, <= 1e-10, everything finite; else the dense route (fallback "residual" / "overlap" / "nonfinite").
A dense fallback returns exactly what torch.linalg.eigh returns (engine_collapse_v5.eigT's code).

Guards of sternheimer_X (per channel; a failed guard returns None and the caller runs the eigenbasis formula, counted):
  S1 no occupied-unoccupied pair with |e_o - e_u| < 1e-12 (the eigenbasis route drops those pairs; the solve cannot) and none closer than
     1e-9 ||h||_G (fallback "occ_unocc_gap");
  S2 no two occupied levels closer than 1e-10 ||h||_G (fallback "occ_occ_gap"); and the near-degenerate occupied partners are deflated:
     besides u_o itself, every occupied u_o' with |e_o' - e_o| < 1e-3 ||h||_G is projected out of b_o before the solve and out of y_o after
     it. This is exact (the pair's two components c_{o'o} and c_{oo'} = -conj(c_{o'o}) cancel in the projector change, so removing both
     changes nothing analytically), and it removes the solve's ill-conditioned direction: without it the rounding of the cancelling pairs
     grows as eps ||h|| / gap^2 (W2 slice kappa 1, 2048 sites: the sea holds 349 occupied pairs closer than 1e-6 ||h||_G; the density
     response against the eigenbasis formula is 1.1e-8 relative with u_o alone deflated, 6.2e-9 / 3.2e-10 / 1.35e-10 / 1.35e-11 with the
     partners within 1e-6 / 1e-4 / 1e-3 / 1e-2 ||h||_G deflated, 9e-12 with the full occupied projection; diag_partner_v1). More than 1024
     partners on one side is refused (fallback "occ_cluster");
  S3 the solve's relative residual ||(h - e_o) y - b||_F/||b||_F <= 1e-9 and finite (fallback "solve_residual");
  S4 a device out-of-memory error anywhere in the route (fallback "oom"; solve_shifted first halves its batch down to 64 columns).

API: eigh_banded(band, n, device, ...) -> (e, U, info) (the prototype's signature); eigT_banded(hmT) -> (e, U); band_from_dense_lower(hmT);
band_from_csr(h); solve_shifted(band, n, sigs, B) (torch B on the device); sternheimer_X(band, eo, Uo, B) -> (Y or None, info);
warmup(device) (the kernels' JIT compile on a 64-site chain); STATS (the process's counters).
"""
import math
import time

import numpy as np
import numba
from numba import cuda
import torch

VERSION = "engine_banded_eigh_v2"   # rd1d memsafe

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


def lowdin_polish_window(U, lam, width=16, window=0.0, max_off=4096):
    """lowdin_polish (|i - j| <= width) extended by every pair farther apart in index whose eigenvalues differ by less than window: one
    first-order Lowdin step U <- U (I - (G - I)/2) on that sparsity pattern of the Gram matrix G = U^H U, then the columns normalized.
    window = 1e-3 ||h||_G is LAPACK dstein's reorthogonalization criterion (ORTOL); inverse iteration leaves |u_i^H u_j| ~ eps ||h|| / gap,
    so on dense spectra the pairs at |i - j| > 16 can carry 1e-12 (ISO_128 kappa 10: 6e-12 at |i - j| 40, gap 2.4e-3) and the index band
    alone does not reach them."""
    n, T = U.shape
    out = U.clone()
    for off in range(1, min(width, T - 1) + 1):
        g = (U[:, :T - off].conj() * U[:, off:]).sum(dim=0)
        out[:, :T - off] -= 0.5 * g.conj()[None, :] * U[:, off:]
        out[:, off:] -= 0.5 * g[None, :] * U[:, :T - off]
    n_extra = 0
    if window > 0.0:
        for off in range(width + 1, min(max_off, T - 1) + 1):
            sel = torch.nonzero((lam[off:] - lam[:-off]).abs() < window).flatten()
            if sel.numel() == 0:
                break
            j_ = sel + off
            g = (U[:, sel].conj() * U[:, j_]).sum(dim=0)
            out[:, sel] -= 0.5 * g.conj()[None, :] * U[:, j_]
            out[:, j_] -= 0.5 * g[None, :] * U[:, sel]
            n_extra += int(sel.numel())
    nrm = torch.linalg.norm(out, dim=0)
    return out / nrm[None, :], n_extra


STATS = {"calls": 0, "banded": 0, "banded_all": 0, "dense_fallback": 0, "fallback_why": {}, "seconds": 0.0, "guard_seconds": 0.0,
         "kernel_seconds": 0.0, "post_seconds": 0.0, "max_residual_rel": 0.0, "max_overlap": 0.0, "clusters": 0, "isolated_min": 1.0,
         "st_channels": 0, "st_fallback": 0, "st_fallback_why": {}, "st_seconds": 0.0, "st_max_residual_rel": 0.0, "warmup_s": None,
         "device": None, "devnum_patch": None}
_DEV = {"idx": None}


def _bind(dev):
    """numba's current device = the torch device's index (once per process; both use the device's primary context)."""
    dev = torch.device(dev)
    idx = 0 if dev.index is None else int(dev.index)
    if _DEV["idx"] != idx:
        torch.zeros(1, device=dev)                                    # torch's primary context exists and is current
        _patch_devnum()
        cuda.select_device(idx)
        _DEV["idx"] = idx
        STATS["device"] = "cuda:%d (%s)" % (idx, torch.cuda.get_device_name(dev))
    return dev


def _patch_devnum():
    """numba 0.61 (ctypes driver) binds the driver's versioned entry points first; on a driver that exports cuCtxGetDevice_v2 (CUDA 13: an
    extra CUcontext argument) numba's one-argument call returns a wrong device (h100-f: 8 for ordinal 0) and every numba call on torch's
    context fails (IndexError in the device list). With a context current (torch's), numba's cuCtxGetDevice is compared with the library's
    plain cuCtxGetDevice symbol; when, and only when, they differ, the Driver instance's cuCtxGetDevice is replaced by the plain symbol
    (numba's error check kept). On the box they agree and nothing is patched."""
    if STATS.get("devnum_patch") is not None:
        return
    from ctypes import byref, c_int
    from numba.cuda.cudadrv import driver as _drv, drvapi as _api
    D = _drv.driver
    if _drv.USE_NV_BINDING:
        STATS["devnum_patch"] = "none (NVIDIA binding in use)"
        return
    a = _api.cu_device(); D.cuCtxGetDevice(byref(a))
    raw = D.lib.cuCtxGetDevice
    b = c_int(-1); rc = raw(byref(b))
    if rc != 0 or int(a.value) == int(b.value):
        STATS["devnum_patch"] = "none (numba's cuCtxGetDevice agrees: %d)" % int(a.value) if rc == 0 else "none (plain symbol rc %d)" % rc
        return

    def cuCtxGetDevice_plain(ptr):
        r = raw(ptr)
        if r != 0:
            raise _drv.CudaAPIError(r, "cuCtxGetDevice")
    D.cuCtxGetDevice = cuCtxGetDevice_plain
    STATS["devnum_patch"] = "numba's cuCtxGetDevice returned %d, the plain symbol %d: replaced by the plain symbol" % (int(a.value), int(b.value))


def _ca(t):
    """the torch tensor as a numba device array (zero copy)."""
    return cuda.as_cuda_array(t)


def _fail(stats_key, why_key, why):
    STATS[stats_key] += 1
    d = STATS[why_key]
    d[why] = d.get(why, 0) + 1


def _free_bytes(dev):
    """bytes the next allocations can use: the driver's free memory plus torch's cached but unallocated memory."""
    free, _tot = torch.cuda.mem_get_info(dev)
    return int(free) + int(torch.cuda.memory_reserved(dev)) - int(torch.cuda.memory_allocated(dev))


def solve_shifted(band, n, sigs, B, batch=None, pivmin_rel=64.0):
    """X[:, t] = (band - sigs[t])^{-1} B[:, t] by the banded LU with partial pivoting, one thread per column (the prototype's routine with
    the buffers as torch tensors on B's device). B: torch (n, T) complex128 on a CUDA device; returns a torch tensor on that device. A
    shift at an eigenvalue gives the pivmin-regularized solve (the caller projects the singular direction out).
    The batch (columns per launch) is at most 2048 and at most what half the device's free memory holds (the per-column scratch is
    (13 + 1) (n + 16) complex128 entries); on an out-of-memory error the cache is emptied and the batch halved (down to 64). Every column is
    one thread's own arithmetic, so the result does not depend on the batch, bit for bit."""
    dev = _bind(B.device)
    eps = float(np.finfo(float).eps)
    G = gershgorin_band(band)
    pivmin = pivmin_rel * eps * G
    npad = n + PAD
    bd_t = torch.as_tensor(_padded(band, n), device=dev)
    T = B.shape[1]
    out = torch.empty((n, T), dtype=torch.complex128, device=dev)
    sig_all = torch.as_tensor(np.ascontiguousarray(np.asarray(sigs, dtype=np.float64)), device=dev)
    if batch is None:
        per_col = (LDAB + 1) * npad * 16 + npad * 4
        batch = int(max(64, min(2048, (_free_bytes(dev) // 2) // per_col)))
    b0 = 0
    while b0 < T:
        b1 = min(T, b0 + batch); Tb = b1 - b0
        try:
            W = torch.empty((LDAB, npad, Tb), dtype=torch.complex128, device=dev)
            piv = torch.empty((npad, Tb), dtype=torch.int32, device=dev)
            Xt = torch.zeros((npad, Tb), dtype=torch.complex128, device=dev)
        except torch.cuda.OutOfMemoryError:
            W = piv = Xt = None
            torch.cuda.empty_cache()
            if batch <= 64:
                raise
            batch = max(64, batch // 2); STATS["oom_retries"] = STATS.get("oom_retries", 0) + 1
            continue
        Xt[:n] = B[:, b0:b1]
        sg = sig_all[b0:b1].contiguous()
        torch.cuda.synchronize(dev)
        _solve_kernel[(Tb + TPB - 1) // TPB, TPB](_ca(bd_t), n, npad, _ca(sg), _ca(W), _ca(piv), _ca(Xt), pivmin)
        cuda.synchronize()
        out[:, b0:b1] = Xt[:n]
        del W, piv, Xt
        b0 = b1
    STATS["st_batch_min"] = min(STATS.get("st_batch_min") or batch, batch)
    return out


def eigh_banded(band, n, device, half="ph", nsteps=40, nextra_max=4, batch=2048, cluster_tol=1e-9, pivmin_rel=64.0, iso_rel=1e-12, res_tol_rel=200.0,
                polish=16, polish_window_rel=1e-3, timing=None, seed=0.0):
    """eigenpairs of the banded Hermitian matrix (the prototype's routine, the same parameters and steps; the buffers as torch tensors; the
    Lowdin polish over |i - j| <= polish plus every pair with |lam_i - lam_j| < polish_window_rel ||h||_G, lowdin_polish_window).
    half="ph": the n/2 positive eigenpairs by the kernels and the negative half by the particle-hole lift (massive chain); half="all": every
    eigenpair by the kernels (no symmetry assumed). Returns (e, U, info)."""
    tm = {} if timing is None else timing
    t0 = time.time()
    dev = _bind(device)
    eps = float(np.finfo(float).eps)
    G = gershgorin_band(band)
    pivmin = pivmin_rel * eps * G
    npad = n + PAD
    bd_t = torch.as_tensor(_padded(band, n), device=dev)
    if half == "ph":
        kidx = np.arange(n // 2, n, dtype=np.int64)
        lo0 = np.zeros(len(kidx)); hi0 = np.full(len(kidx), G)
    else:
        kidx = np.arange(n, dtype=np.int64)
        lo0 = np.full(len(kidx), -G); hi0 = np.full(len(kidx), G)
    T = len(kidx)
    d_k = torch.as_tensor(kidx, device=dev); d_lo = torch.as_tensor(lo0, device=dev); d_hi = torch.as_tensor(hi0, device=dev)
    o_lo = torch.empty(T, dtype=torch.float64, device=dev); o_hi = torch.empty(T, dtype=torch.float64, device=dev)
    o_iso = torch.empty(T, dtype=torch.int32, device=dev); o_st = torch.empty(T, dtype=torch.int32, device=dev)
    torch.cuda.synchronize(dev); t1 = time.time(); tm["setup_s"] = t1 - t0
    nblk = (T + (TPB // MS) - 1) // (TPB // MS)
    _msect_kernel[nblk, TPB](_ca(bd_t), n, _ca(d_k), _ca(d_lo), _ca(d_hi), nsteps, pivmin, 4.0 * eps, iso_rel, _ca(o_lo), _ca(o_hi), _ca(o_iso), _ca(o_st))
    cuda.synchronize(); t2 = time.time(); tm["bisect_s"] = t2 - t1
    lo = o_lo.cpu().numpy(); hi = o_hi.cpu().numpy(); iso = o_iso.cpu().numpy(); st = o_st.cpu().numpy()
    sig = 0.5 * (lo + hi)
    Up = torch.empty((n, T), dtype=torch.complex128, device=dev)
    lam_all = np.empty(T); rounds = np.empty(T, np.int32); res_k = np.empty(T)
    for b0 in range(0, T, batch):
        b1 = min(T, b0 + batch)
        Tb = b1 - b0
        W = torch.empty((LDAB, npad, Tb), dtype=torch.complex128, device=dev)
        piv = torch.empty((npad, Tb), dtype=torch.int32, device=dev)
        X = torch.empty((npad, Tb), dtype=torch.complex128, device=dev)
        o_lam = torch.empty(Tb, dtype=torch.float64, device=dev); o_r = torch.empty(Tb, dtype=torch.int32, device=dev); o_res = torch.empty(Tb, dtype=torch.float64, device=dev)
        d_sig = torch.as_tensor(np.ascontiguousarray(sig[b0:b1]), device=dev)
        torch.cuda.synchronize(dev)
        _invit_kernel[(Tb + TPB - 1) // TPB, TPB](_ca(bd_t), n, npad, _ca(d_sig), _ca(W), _ca(piv), _ca(X), nextra_max, res_tol_rel * eps * G, pivmin, float(seed),
                                                  _ca(o_lam), _ca(o_r), _ca(o_res))
        cuda.synchronize()
        Up[:, b0:b1] = X[:n]
        lam_all[b0:b1] = o_lam.cpu().numpy(); rounds[b0:b1] = o_r.cpu().numpy(); res_k[b0:b1] = o_res.cpu().numpy()
        del W, piv, X, o_lam, o_r, o_res
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
    n_extra = 0
    if polish:
        Up, n_extra = lowdin_polish_window(Up, lam, width=int(polish), window=polish_window_rel * G)
        lam = (Up.conj() * band_matvec(band_t, Up)).real.sum(dim=0)
    HU = band_matvec(band_t, Up)
    res = torch.linalg.norm(HU - Up * lam[None, :], dim=0)
    del HU
    orth_after = 0.0                                                  # G5: the neighbour overlaps after the polish, 1 <= j - i <= 4
    for off in range(1, min(5, T)):
        orth_after = max(orth_after, float((Up[:, :T - off].conj() * Up[:, off:]).sum(dim=0).abs().max()))
    if half == "ph":
        Un = ph_lift(Up)
        e = torch.cat([-lam.flip(0), lam])
        U = torch.cat([Un.flip(1), Up], dim=1)
        del Un
    else:
        e = lam; U = Up
    del Up
    order = torch.argsort(e)
    e = e[order]; U = U[:, order]
    torch.cuda.synchronize(dev)
    t4 = time.time(); tm["post_s"] = t4 - t3; tm["total_s"] = t4 - t0
    info = {"gershgorin": G, "pivmin": pivmin, "bracket_max_rel": float(np.max((hi - lo) / np.maximum(np.abs(sig), 1e-300))), "isolated_fraction": float(np.mean(iso)),
            "count_sweeps_mean": float(np.mean(st)), "count_sweeps_max": int(np.max(st)), "rounds_mean": float(np.mean(rounds)), "rounds_max": int(np.max(rounds)),
            "solves_extra": int(np.sum(rounds - 2)), "kernel_residual_max": float(np.max(res_k)), "kernel_residual_max_rel": float(np.max(res_k)) / G,
            "residual_max": float(res.max()), "residual_max_rel": float(res.max()) / G, "clusters": n_cl, "cluster_sizes": cl_sizes, "n_threads": T,
            "neighbour_overlap_before_polish": orth_before, "neighbour_overlap_after_polish": orth_after, "shift_vs_rayleigh_max": float(np.max(np.abs(sig - lam_all))),
            "lam_min_computed": float(lam.min()), "half": half, "polish_extra_pairs": n_extra}
    return e, U, info


# ---------------------------------------------------------------------------------------------------------------- the engine entry points
def band_from_dense_lower(hmT):
    """the 5 upper diagonals of the Hermitian matrix as torch.linalg.eigh sees it (its lower triangle): band[m, j] = conj(h[j + m, j]),
    band[0] = Re diag (numpy, host)."""
    n = hmT.shape[0]
    band = np.zeros((KL + 1, n), dtype=np.complex128)
    band[0] = torch.diagonal(hmT, 0).real.cpu().numpy()
    for m in range(1, KL + 1):
        band[m, :n - m] = torch.diagonal(hmT, -m).conj().cpu().numpy()
    return band


def offband_nonzero(hmT):
    """the number of nonzero entries of hmT outside |i - j| <= KL (exact: the nonzero count of the whole matrix minus that of its 2 KL + 1
    central diagonals; one reduction pass, no full-size temporary)."""
    tot = int(torch.count_nonzero(hmT))
    bandc = sum(int(torch.count_nonzero(torch.diagonal(hmT, k))) for k in range(-KL, KL + 1))
    return tot - bandc


def _band_apply_np(band, x):
    n = band.shape[1]
    y = band[0].real * x
    for m in range(1, KL + 1):
        y[:n - m] += band[m, :n - m] * x[m:]
        y[m:] += np.conj(band[m, :n - m]) * x[:n - m]
    return y


def _S_np(x):
    """S = sum_n (-1)^n sigma_y on the interleaved index (ph_lift's S, before the conjugation)."""
    i = np.arange(x.shape[0] // 2)
    sgn = np.where(i % 2 == 0, 1.0, -1.0)
    y = np.empty_like(x)
    y[0::2] = (-1j) * sgn * x[1::2]
    y[1::2] = (1j) * sgn * x[0::2]
    return y


def ph_defect(band):
    """||h S conj(x) + S conj(h x)|| / (||h||_G ||x||) on a fixed pseudo-random x (0 for an exactly particle-hole symmetric band)."""
    n = band.shape[1]
    rng = np.random.default_rng(12345)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a = _band_apply_np(band, _S_np(np.conj(x)))
    b = _S_np(np.conj(_band_apply_np(band, x)))
    G = gershgorin_band(band)
    return float(np.linalg.norm(a + b) / (max(G, 1e-300) * np.linalg.norm(x)))


MEMSAFE = {"events": 0, "ok_after_empty_cache": 0, "ok_after_park": 0, "failed": 0, "retries": 0, "tensors_parked": 0, "bytes_parked": 0,   # rd1d memsafe
           "park_s": 0.0, "restore_s": 0.0, "seconds": 0.0, "max_parked_bytes_one_call": 0}   # rd1d memsafe
PARK_MIN_BYTES = 64 << 20   # rd1d memsafe: storages below 64 MiB are never parked


class MemsafeRestoreFailed(BaseException):   # rd1d memsafe: a parked storage that cannot come back stops the process (not catchable by except Exception)
    pass   # rd1d memsafe


def _park_candidates(hmT):   # rd1d memsafe: the device storages that may leave: (nbytes, storage), largest first
    import gc   # rd1d memsafe
    dev = hmT.device   # rd1d memsafe
    keep = hmT.untyped_storage().data_ptr()   # rd1d memsafe
    seen = {}   # rd1d memsafe
    for o in gc.get_objects():   # rd1d memsafe
        try:   # rd1d memsafe
            if not isinstance(o, torch.Tensor) or o.layout != torch.strided or o.device != dev:   # rd1d memsafe
                continue   # rd1d memsafe
            s = o.untyped_storage()   # rd1d memsafe
            p = s.data_ptr(); nb = s.nbytes()   # rd1d memsafe
        except Exception:   # rd1d memsafe: a tensor without storage (sparse, meta, functional): never parked
            continue   # rd1d memsafe
        if p == 0 or p == keep or nb < PARK_MIN_BYTES or p in seen or not s.resizable():   # rd1d memsafe
            continue   # rd1d memsafe
        seen[p] = (nb, s)   # rd1d memsafe
    return sorted(seen.values(), key=lambda q: -q[0])   # rd1d memsafe


def _park(s, nb):   # rd1d memsafe: the storage's bytes to a pageable host buffer, then its device memory released
    h = torch.empty(nb, dtype=torch.uint8, device="cpu")   # rd1d memsafe
    v = torch.empty(0, dtype=torch.uint8, device=s.device).set_(s, 0, (nb,), (1,))   # rd1d memsafe: a byte view of the whole storage
    h.copy_(v)   # rd1d memsafe: synchronous (pageable)
    del v   # rd1d memsafe
    s.resize_(0)   # rd1d memsafe
    return h   # rd1d memsafe


def _unpark(s, h):   # rd1d memsafe: the storage reallocated at its size and its bytes copied back
    nb = h.numel()   # rd1d memsafe
    s.resize_(nb)   # rd1d memsafe
    v = torch.empty(0, dtype=torch.uint8, device=s.device).set_(s, 0, (nb,), (1,))   # rd1d memsafe
    v.copy_(h)   # rd1d memsafe
    del v   # rd1d memsafe


def _dense_memsafe(hmT, err, log=None):   # rd1d memsafe: v1's dense call after empty_cache, then with the resident storages parked one by one
    t0 = time.time()   # rd1d memsafe
    dev = hmT.device   # rd1d memsafe
    MEMSAFE["events"] += 1   # rd1d memsafe
    free0 = _free_bytes(dev)   # rd1d memsafe
    import gc   # rd1d memsafe
    torch.cuda.synchronize(dev); gc.collect(); torch.cuda.empty_cache()   # rd1d memsafe: unreachable tensors (reference cycles) freed first, never parked
    MEMSAFE["retries"] += 1   # rd1d memsafe
    try:   # rd1d memsafe
        e, U = torch.linalg.eigh(hmT)   # rd1d memsafe: engine_collapse_v5.eigT's code
        MEMSAFE["ok_after_empty_cache"] += 1; MEMSAFE["seconds"] += time.time() - t0   # rd1d memsafe
        if log is not None:   # rd1d memsafe
            log("%s: dense eigh out of memory (%.2f GiB free): done after emptying the allocator's cache" % (VERSION, free0 / 2**30))   # rd1d memsafe
        return e, U   # rd1d memsafe
    except torch.cuda.OutOfMemoryError as ex2:   # rd1d memsafe
        err = ex2   # rd1d memsafe
    parked = []   # rd1d memsafe
    out = None   # rd1d memsafe
    cands = _park_candidates(hmT)   # rd1d memsafe
    try:   # rd1d memsafe
        for nb, s in cands:   # rd1d memsafe
            t_p = time.time()   # rd1d memsafe
            torch.cuda.synchronize(dev)   # rd1d memsafe: no queued work reads the storage while it leaves
            parked.append((s, _park(s, nb)))   # rd1d memsafe
            torch.cuda.empty_cache()   # rd1d memsafe
            MEMSAFE["park_s"] += time.time() - t_p   # rd1d memsafe
            MEMSAFE["retries"] += 1   # rd1d memsafe
            try:   # rd1d memsafe
                out = torch.linalg.eigh(hmT)   # rd1d memsafe: engine_collapse_v5.eigT's code
                break   # rd1d memsafe
            except torch.cuda.OutOfMemoryError as ex3:   # rd1d memsafe
                err = ex3   # rd1d memsafe
    finally:   # rd1d memsafe: every parked storage back, byte for byte, before anything else runs
        t_r = time.time()   # rd1d memsafe
        torch.cuda.synchronize(dev)   # rd1d memsafe
        bad = None   # rd1d memsafe
        for s, h in reversed(parked):   # rd1d memsafe
            try:   # rd1d memsafe
                _unpark(s, h)   # rd1d memsafe
            except torch.cuda.OutOfMemoryError as ex4:   # rd1d memsafe: the result and the residents do not fit together
                bad = ex4   # rd1d memsafe
                break   # rd1d memsafe
        if bad is not None:   # rd1d memsafe: give the result's memory back and restore the rest; then v1's outcome (the error)
            out = None   # rd1d memsafe
            for attempt in range(2):   # rd1d memsafe
                gc.collect(); torch.cuda.empty_cache()   # rd1d memsafe
                left = 0   # rd1d memsafe
                for s, h in reversed(parked):   # rd1d memsafe
                    if s.nbytes() == 0:   # rd1d memsafe
                        try:   # rd1d memsafe
                            _unpark(s, h)   # rd1d memsafe
                        except torch.cuda.OutOfMemoryError:   # rd1d memsafe
                            left += 1   # rd1d memsafe
                if left == 0:   # rd1d memsafe
                    break   # rd1d memsafe
            if left:   # rd1d memsafe: never continue with a resident tensor emptied: a BaseException, which the engine's broad handlers do not catch
                raise MemsafeRestoreFailed("%s: %d parked storages could not be restored; the process stops" % (VERSION, left))   # rd1d memsafe
            err = bad   # rd1d memsafe
        torch.cuda.synchronize(dev)   # rd1d memsafe
        MEMSAFE["restore_s"] += time.time() - t_r   # rd1d memsafe
    nbp = sum(h.numel() for _s, h in parked); npk = len(parked)   # rd1d memsafe
    MEMSAFE["tensors_parked"] += len(parked); MEMSAFE["bytes_parked"] += nbp   # rd1d memsafe
    MEMSAFE["max_parked_bytes_one_call"] = max(MEMSAFE["max_parked_bytes_one_call"], nbp)   # rd1d memsafe
    MEMSAFE["seconds"] += time.time() - t0   # rd1d memsafe
    del parked   # rd1d memsafe
    if out is None:   # rd1d memsafe
        MEMSAFE["failed"] += 1   # rd1d memsafe
        if log is not None:   # rd1d memsafe
            log("%s: dense eigh out of memory (%.2f GiB free): %d of %d storages parked (%.2f GiB) did not make room; restored; raising" % (   # rd1d memsafe
                VERSION, free0 / 2**30, npk, len(cands), nbp / 2**30))   # rd1d memsafe
        raise err   # rd1d memsafe
    MEMSAFE["ok_after_park"] += 1   # rd1d memsafe
    if log is not None:   # rd1d memsafe
        log("%s: dense eigh out of memory (%.2f GiB free): done with %.2f GiB of resident storages parked on the host and restored (%.1f s)" % (   # rd1d memsafe
            VERSION, free0 / 2**30, nbp / 2**30, time.time() - t0))   # rd1d memsafe
    return out   # rd1d memsafe


def _dense(hmT, log=None):   # rd1d memsafe
    try:   # rd1d memsafe
        e, U = torch.linalg.eigh(hmT)                                    # engine_collapse_v5.eigT's code [rd1d memsafe]
    except torch.cuda.OutOfMemoryError as ex:   # rd1d memsafe: only then the memory-safe path
        e, U = _dense_memsafe(hmT, ex, log)   # rd1d memsafe
    return e.real, U


def eigT_banded(hmT, ph_tol=1e-13, near_zero_rel=1e-8, res_rel_max=1e-11, overlap_max=1e-10, log=None):
    """the drop-in for engine_collapse_v5.eigT: (e, U) of the dense Hermitian torch matrix hmT (n x n, complex128, on a CUDA device), e real
    ascending, U's columns the eigenvectors; by the banded route under the guards G1 to G5 (the module docstring), else the dense route."""
    t0 = time.time()
    STATS["calls"] += 1
    n = hmT.shape[0]
    why = None
    if hmT.device.type != "cuda" or hmT.dtype != torch.complex128 or n < 32 or n % 2:
        why = "input"
    else:
        if offband_nonzero(hmT) != 0:
            why = "offband"
    if why is None:
        band = band_from_dense_lower(hmT)
        G = gershgorin_band(band)
        half = "ph" if ph_defect(band) <= ph_tol else "all"
        STATS["guard_seconds"] += time.time() - t0
        tm = {}
        try:
            e, U, info = eigh_banded(band, n, hmT.device, half=half, timing=tm)
        except Exception as ex:                                       # a kernel or allocation error: the dense route (named)
            e = U = None; info = {}; why = "error: %s" % str(ex)[:80]
            torch.cuda.empty_cache()
        if why is None:
            STATS["kernel_seconds"] += tm.get("bisect_s", 0.0) + tm.get("invit_s", 0.0); STATS["post_seconds"] += tm.get("post_s", 0.0)
            if not (np.isfinite(info["residual_max"]) and bool(torch.isfinite(e).all())):
                why = "nonfinite"
            elif half == "ph" and info["lam_min_computed"] <= near_zero_rel * G:
                why = "near_zero"
            elif info["residual_max_rel"] > res_rel_max:
                why = "residual"
            elif info["neighbour_overlap_after_polish"] > overlap_max:
                why = "overlap"
        if why is None:
            STATS["banded"] += 1
            if half == "all":
                STATS["banded_all"] += 1
            STATS["max_residual_rel"] = max(STATS["max_residual_rel"], info["residual_max_rel"])
            STATS["max_overlap"] = max(STATS["max_overlap"], info["neighbour_overlap_after_polish"])
            STATS["clusters"] += info["clusters"]; STATS["isolated_min"] = min(STATS["isolated_min"], info["isolated_fraction"])
            STATS["seconds"] += time.time() - t0
            return e, U
        del e, U
    _fail("dense_fallback", "fallback_why", why)
    if log is not None:
        log("%s: the dense route for this call (%s)" % (VERSION, why))   # rd1d memsafe
    out = _dense(hmT, log)   # rd1d memsafe
    STATS["seconds"] += time.time() - t0
    return out


def sternheimer_X(band, eo, eu, Uo, B, occ_unocc_rel=1e-9, occ_occ_rel=1e-10, partner_rel=1e-3, partner_max=1024, res_rel_max=1e-9):
    """the J.v's X by shifted banded solves: B = -dh U_o (n x n_o, torch on the device; U_o's columns in ascending energy), each column deflated of
    its own u_o and its near-degenerate occupied partners (S2), y_o = (h - e_o)^{-1} b_o (the pivmin-regularized banded solve), y_o deflated
    the same way. Y U_o^H + U_o Y^H equals X U_o^H + U_o X^H of the eigenbasis formula
    (X = U_u M, M = U_u^H dh U_o / (e_o - e_u)) exactly: the occupied-occupied components of Y cancel pairwise. Returns (Y, info) or (None,
    info) on a failed guard S1 to S3 (the module docstring): the caller then runs the eigenbasis formula."""
    t0 = time.time()
    STATS["st_channels"] += 1
    G = gershgorin_band(band)
    n = band.shape[1]
    eo_np = np.sort(eo.detach().cpu().numpy().astype(np.float64)); eu_np = np.sort(eu.detach().cpu().numpy().astype(np.float64))
    info = {"gershgorin": G}
    why = None
    if len(eu_np) and len(eo_np):
        lo = np.searchsorted(eu_np, eo_np - 1e-12, side="right"); hi = np.searchsorted(eu_np, eo_np + 1e-12, side="left")
        info["pairs_below_1e-12"] = int(np.sum(hi - lo))
        k = np.clip(np.searchsorted(eu_np, eo_np), 1, len(eu_np) - 1) if len(eu_np) > 1 else np.zeros(len(eo_np), int)
        dmin = float(np.min(np.minimum(np.abs(eu_np[k] - eo_np), np.abs(eu_np[k - 1] - eo_np)))) if len(eu_np) > 1 else float(np.min(np.abs(eu_np[0] - eo_np)))
        info["min_occ_unocc_gap"] = dmin
        if info["pairs_below_1e-12"] or dmin < occ_unocc_rel * G:
            why = "occ_unocc_gap"
    if why is None and len(eo_np) > 1 and not bool(np.all(np.diff(eo.detach().cpu().numpy()) >= 0)):
        why = "unsorted"                                              # U_o's columns must be in ascending energy (the partner search)
    if why is None and len(eo_np) > 1:
        info["min_occ_occ_gap"] = float(np.min(np.diff(eo_np)))
        if info["min_occ_occ_gap"] < occ_occ_rel * G:
            why = "occ_occ_gap"
    Y = None
    offs = []
    if why is None:                                                   # the near-degenerate occupied partners (eo ascending: neighbours in index)
        eo_t = eo.detach().to(torch.float64)
        for off in range(1, partner_max + 2):
            if off >= eo_t.shape[0]:
                break
            sel = torch.nonzero((eo_t[off:] - eo_t[:-off]).abs() < partner_rel * G).flatten()
            if sel.numel() == 0:
                break
            if off > partner_max:
                why = "occ_cluster"; break
            offs.append((off, sel))
        info["partner_offsets"] = len(offs); info["partner_pairs"] = int(sum(int(q[1].numel()) for q in offs))

    def deflate(Z):
        c = (Uo.conj() * Z).sum(dim=0)
        Z = Z - Uo * c[None, :]
        for off, sel in offs:
            i_, j_ = sel, sel + off
            ci = (Uo[:, j_].conj() * Z[:, i_]).sum(dim=0); cj = (Uo[:, i_].conj() * Z[:, j_]).sum(dim=0)
            Z[:, i_] -= Uo[:, j_] * ci[None, :]; Z[:, j_] -= Uo[:, i_] * cj[None, :]
        return Z
    if why is None:
        try:
            B = deflate(B)
            Y = solve_shifted(band, n, eo.detach().cpu().numpy().astype(np.float64), B)
            Y = deflate(Y)
        except torch.cuda.OutOfMemoryError:                           # the eigenbasis formula then runs for this channel (counted "oom")
            Y = None; why = "oom"
            torch.cuda.empty_cache()
    if Y is not None:
        band_t = torch.as_tensor(np.ascontiguousarray(band), device=B.device)
        try:
            resid = band_matvec(band_t, Y) - Y * eo[None, :].to(torch.float64).to(Y.dtype) - B
            nb = float(torch.linalg.norm(B))
            rr = float(torch.linalg.norm(resid)) / nb if nb > 0 else 0.0
            del resid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); rr = float("inf"); why = "oom"
        info["solve_residual_rel"] = rr
        if why == "oom":
            Y = None
        elif not (np.isfinite(rr) and bool(torch.isfinite(Y).all())) or rr > res_rel_max:
            why = "solve_residual"; Y = None
        else:
            STATS["st_max_residual_rel"] = max(STATS["st_max_residual_rel"], rr)
    if why is not None:
        _fail("st_fallback", "st_fallback_why", why)
        info["fallback"] = why
    STATS["st_seconds"] += time.time() - t0
    return Y, info


def warmup(device):
    """the kernels' JIT compile (the multisection, the inverse iteration, the shifted solve) on a 64-site free massive chain; returns seconds."""
    t0 = time.time()
    dev = _bind(device)
    ns = 64; n = 2 * ns
    rng = np.random.default_rng(7)
    band = np.zeros((KL + 1, n), dtype=np.complex128)
    band[0] = rng.standard_normal(n)
    for m in range(1, KL + 1):
        band[m, :n - m] = rng.standard_normal(n - m) + 1j * rng.standard_normal(n - m)
    e, U, info = eigh_banded(band, n, dev, half="all")
    B = torch.as_tensor(rng.standard_normal((n, 4)) + 0j, device=dev)
    solve_shifted(band, n, np.asarray(e[:4].cpu().numpy()) + 0.5, B)
    torch.cuda.synchronize(dev)
    STATS["warmup_s"] = time.time() - t0
    return STATS["warmup_s"]


def stats_snapshot(reset=False):
    """a copy of STATS (the record's counters); reset=True zeroes the per-record counters (warmup_s and device kept)."""
    snap = {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()}
    if MEMSAFE["events"]:   # rd1d memsafe: only when the memory-safe path acted
        snap["memsafe"] = dict(MEMSAFE)   # rd1d memsafe
        if reset:   # rd1d memsafe
            for k in MEMSAFE:   # rd1d memsafe
                MEMSAFE[k] = 0.0 if isinstance(MEMSAFE[k], float) else 0   # rd1d memsafe
    if reset:
        for k, v in STATS.items():
            if k in ("warmup_s", "device", "devnum_patch"):
                continue
            STATS[k] = {} if isinstance(v, dict) else (1.0 if k == "isolated_min" else (0.0 if isinstance(v, float) else 0))
    return snap
