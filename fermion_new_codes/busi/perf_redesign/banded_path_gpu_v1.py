#!/usr/bin/env python
"""banded_path_gpu_v1.py (busi lane, reviewer A redesign prototype R4): inertia counts and eigenvalue brackets of a banded Hermitian
PATH h(s) = band0 + s dband (the two-field engine's straight record path, in the S = +1 sector block), batched over (s, shift) pairs on
the GPU: one thread per count, MS threads per bracketed eigenvalue (the multisection of banded_eigh_gpu_v1). Used to tabulate the gap
functions d_j(s) = min |spec(H_+(s)) - a_j(s)| of the sector walk's anchors in one launch, in place of the walk's per-piece Cholesky
ladders on K(s)^2 - d^2."""
import numpy as np
import numba
from numba import cuda

import banded_eigh_gpu_v1 as BE

MS = BE.MS
TPB = BE.TPB


@cuda.jit(device=True, inline=True)
def _csq(z):
    return z.real * z.real + z.imag * z.imag


@cuda.jit(device=True)
def _sturm_count_path(band0, dband, s, n, sigma, pivmin):
    """the count of eigenvalues of band0 + s dband below sigma (the no-pivoting LDL^H on the band, the window in registers)."""
    zero = complex(0.0, 0.0)
    p11 = zero; p12 = zero; p13 = zero; p14 = zero
    p21 = zero; p22 = zero; p23 = zero; p24 = zero
    p31 = zero; p32 = zero; p33 = zero; p34 = zero
    p41 = zero; p42 = zero; p43 = zero; p44 = zero
    d1 = 0.0; d2 = 0.0; d3 = 0.0; d4 = 0.0
    cnt = 0
    for j in range(n):
        b0 = band0[0, j].real + s * dband[0, j].real
        b1 = (band0[1, j] + s * dband[1, j]).conjugate(); b2 = (band0[2, j] + s * dband[2, j]).conjugate()
        b3 = (band0[3, j] + s * dband[3, j]).conjugate(); b4 = (band0[4, j] + s * dband[4, j]).conjugate()
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
def _count_path_kernel(band0, dband, n, s_arr, sig_arr, pivmin, out):
    t = cuda.grid(1)
    if t >= s_arr.shape[0]:
        return
    out[t] = _sturm_count_path(band0, dband, s_arr[t], n, sig_arr[t], pivmin)


@cuda.jit
def _msect_path_kernel(band0, dband, n, s_arr, kidx, lo, hi, nsteps, pivmin, tol_rel, out_lo, out_hi):
    """MS threads per task e: the bracket of eigenvalue kidx[e] of band0 + s_arr[e] dband, multisection to the relative width tol_rel."""
    sh = cuda.shared.array(TPB, numba.int32)
    flag = cuda.shared.array(1, numba.int32)
    tid = cuda.threadIdx.x
    g = tid // MS
    q = tid - g * MS
    e = cuda.blockIdx.x * (TPB // MS) + g
    T = kidx.shape[0]
    active = e < T
    k = 0; a = 0.0; b = 1.0; s = 0.0
    if active:
        k = kidx[e]; a = lo[e]; b = hi[e]; s = s_arr[e]
    done = not active
    for it in range(nsteps):
        if tid == 0:
            flag[0] = 0
        cuda.syncthreads()
        if not done:
            flag[0] = 1
        cuda.syncthreads()
        if flag[0] == 0:
            break
        pt = 0.0
        if not done:
            w = b - a
            sc = max(abs(a), abs(b))
            if w <= tol_rel * sc:
                done = True
            else:
                pt = a + (q + 1) * w / (MS + 1)
                sh[tid] = _sturm_count_path(band0, dband, s, n, pt, pivmin)
        cuda.syncthreads()
        if not done:
            base = g * MS
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


def counts_on_path(band0, dband, n, s_arr, sig_arr, pivmin):
    b0 = cuda.to_device(np.ascontiguousarray(band0)); b1 = cuda.to_device(np.ascontiguousarray(dband))
    T = len(s_arr)
    d_s = cuda.to_device(np.ascontiguousarray(np.asarray(s_arr, float))); d_sig = cuda.to_device(np.ascontiguousarray(np.asarray(sig_arr, float)))
    out = cuda.device_array(T, np.int32)
    _count_path_kernel[(T + TPB - 1) // TPB, TPB](b0, b1, n, d_s, d_sig, pivmin, out)
    cuda.synchronize()
    return out.copy_to_host()


def brackets_on_path(band0, dband, n, s_arr, kidx, lo, hi, pivmin, nsteps=24, tol_rel=1e-13):
    b0 = cuda.to_device(np.ascontiguousarray(band0)); b1 = cuda.to_device(np.ascontiguousarray(dband))
    T = len(kidx)
    d_s = cuda.to_device(np.ascontiguousarray(np.asarray(s_arr, float))); d_k = cuda.to_device(np.ascontiguousarray(np.asarray(kidx, np.int64)))
    d_lo = cuda.to_device(np.ascontiguousarray(np.asarray(lo, float))); d_hi = cuda.to_device(np.ascontiguousarray(np.asarray(hi, float)))
    o_lo = cuda.device_array(T, np.float64); o_hi = cuda.device_array(T, np.float64)
    nblk = (T + (TPB // MS) - 1) // (TPB // MS)
    _msect_path_kernel[nblk, TPB](b0, b1, n, d_s, d_k, d_lo, d_hi, nsteps, pivmin, tol_rel, o_lo, o_hi)
    cuda.synchronize()
    return o_lo.copy_to_host(), o_hi.copy_to_host()
