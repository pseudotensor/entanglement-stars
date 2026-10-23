#!/usr/bin/env python
"""build_engine_twofield_v9c6.py (busi lane, PERF tracker row RD-4; reviewer A algorithm redesign final4/PERF/REVIEWA_algorithm_redesign_v1.md sec. 4):
engine_twofield_v9c6.py over engine_twofield_v9c5.py (md5 7ca18d7193b86f255897d8e4ea72163e) in the lane's patch form: exact replacements, each
OLD occurring once in v9c5; every line of v9c6 not in v9c5 carries "[v9c6 perf]"; --reverse restores v9c5 byte for byte; the source md5 is
checked; an existing output is never overwritten.

The walk (light_sector_walk_v9) tests at every probe s and level d whether K_j(s)^2 - d^2 is positive definite (the banded Cholesky ladder).
That verdict is a statement about the gap function g_j(s) = min |spec H_+(s) - a_j(s)| (spec K_j is the two sectors' spectra, |spec K_-| =
|spec K_+|, K_+ = H_+ - a_j): with F = 64 eps G_K^2 the ladder's own rounding floor, the Cholesky passes when g^2 > d^2 + F and fails when
g^2 < d^2 - F. Options (consumed by argparse actions into module dicts; the parsed namespace and the json args stay v9c5's):

  --light-walk ladder|tabulated (default ladder: every statement executed is v9c5's, apart from one function call per walk and one test per probe).
      tabulated: per walk segment the S = +1 blocks of h at the segment's ends (SYM-C2's _symc_plus_host; the off-sector block exactly zero
      required) as band arrays (half-bandwidth 2 on the light, so the half-bandwidth-2 sweep), and each anchor's count check at both ends (the
      host inertia count at a_j equals the boundary's index + 1; otherwise that anchor keeps the ladder). Then on an s grid (--light-walk-grid,
      default 2048 points) for every anchor the inertia count of H_+(s) at a_j(s) and the two adjacent levels bracketed by multisection to 1e-12
      (one numba.cuda launch each on the engine's device; numba's CPU threads when the engine runs on the CPU): the gap functions with an error
      budget of 1e-10 on every level, checked at both ends against the engine's own full spectra (each tabulated level within the budget of a
      level there), otherwise that anchor's table is not used. Every ladder verdict of the piece loop is then decided in three tiers, the loop
      itself (the step rule, the halving, the doubling check, the crossing phase with its reopening Cholesky) unchanged:
        1. the table: the Weyl (Lipschitz) bound between the two grid points around the probe, |g(s) - g(s_k)| <= L |s - s_k|, L the smaller
           Gershgorin bound of dK and dK_+: PASS when the lower bound g_lo has g_lo^2 > d^2 + 4F, FAIL when the upper bound g_hi has
           g_hi^2 < d^2 - 4F;
        2. the inertia count at the probe itself (the band LDL^H count of H_+(s) on the host, numba): PASS when no level lies within
           sqrt(d^2 + 4F) + 1e-10 of the anchor, FAIL when one lies within sqrt(d^2 - 4F) - 1e-10;
        3. otherwise the ladder's own Cholesky (band_upper_v36 of Ks @ Ks, pd_banded_v36), built only when needed.
      The margins (4F, four times the ladder's own floor, and the 1e-10 budget) make the tiers' verdicts the Cholesky's wherever the
      Cholesky's error is within F: so the pieces, the failure probes, the crossing windows and everything downstream are the ladder
      route's. The record's cholesky counts are the Cholesky factorizations actually done (tier 3 and the crossing phase's reopening tests);
      the tier counts are printed on the "v9c6 perf (the run)" line.
  --light-walk-grid N: the grid points per segment (default 2048).
  --light-walk-table-after N: a segment's table is built after N verdicts of the segment (default 0: at its first verdict, the spec's form;
      a larger N leaves short segments to tiers 2 and 3, for a device where the launch costs more than the verdicts).
  --light-walk-audit (a test): every tier-1/tier-2 verdict is also factored by the Cholesky and the disagreements counted; the walk follows
      the tiers' verdicts.
  --perf-sym-extend-sector (the v9c5b item of PERF_SYMC_build_v1): the primitive's chord extension V9cSpectra._extend (s outside [0, 1])
      takes SYM-C1's sector eigh _symc_eig_light in place of eigT(clocked_gpu); default off.

Scope: light_sector_walk_v9 only (the record route's light_walk_v36 is unchanged). H3's B1 anchor pool is not used for the tabulated verdicts.

Usage. python build_engine_twofield_v9c6.py engine_twofield_v9c5.py [engine_twofield_v9c6.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c6.py restored.py [--against engine_twofield_v9c5.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c6 perf]"
SOURCE_MD5 = "7ca18d7193b86f255897d8e4ea72163e"   # engine_twofield_v9c5.py

BLOCK = r'''# >>> v9c6 perf: PERF RD-4 (reviewer A algorithm redesign v1 sec. 4): the light sector walk's ladder verdicts from the gap functions tabulated on an s grid, the probe's inertia count, the Cholesky last; --light-walk ladder (the default) runs v9c5's statements
import argparse as _rd4_argparse
import os as _rd4_os
import threading as _rd4_threading

V9C6_VERSION = "engine_twofield_v9c6"
_RD4 = {"walk": "ladder", "grid": 2048, "table_after": 0, "audit": False, "device": None, "threads": 1}
_RD4_STATS = {}
_RD4_LAST = {}
_RD4_LOCK = _rd4_threading.Lock()
_RD4_NB = {}
_RD4_ERR = 1e-10
_RD4_TOL = 1e-12
_RD4_MS = 8
_RD4_TPB = 64
_RD4_KL = 4
_RD4_CUDA13 = ("cuCtxGetDevice", "cuCtxSynchronize", "cuEventElapsedTime", "cuStreamGetCtx", "cuMemAdvise", "cuMemPrefetchAsync")


def _rd4_inc(k, n=1):
    # diagnostic counters only (no numerical effect)
    with _RD4_LOCK:
        _RD4_STATS[k] = _RD4_STATS.get(k, 0) + n
        _RD4_LAST[k] = _RD4_LAST.get(k, 0) + n


def _rd4_max(k, x):
    with _RD4_LOCK:
        _RD4_STATS[k] = max(_RD4_STATS.get(k, 0.0), float(x))
        _RD4_LAST[k] = max(_RD4_LAST.get(k, 0.0), float(x))


class _Rd4Action(_rd4_argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _RD4[self.dest] = True if self.nargs == 0 else values
        # Deliberately do not modify namespace: output args stay v9c5's.


def _rd4_add_arguments(ap):
    ap.add_argument("--light-walk", dest="walk", choices=["ladder", "tabulated"], action=_Rd4Action, default=_rd4_argparse.SUPPRESS,
                    help="RD-4: the sector walk's ladder verdicts: ladder (the default, v9c5's Cholesky ladder) or tabulated (the gap functions on an s grid with the Weyl bound, then the inertia count at the probe, then the Cholesky)")
    ap.add_argument("--light-walk-grid", dest="grid", type=int, action=_Rd4Action, default=_rd4_argparse.SUPPRESS,
                    help="RD-4: the grid points per walk segment of the tabulated gap functions (default 2048)")
    ap.add_argument("--light-walk-table-after", dest="table_after", type=int, action=_Rd4Action, default=_rd4_argparse.SUPPRESS,
                    help="RD-4: a walk segment's grid table is built after this many verdicts of the segment (default 0: at its first verdict); before it the probe count and the Cholesky decide")
    ap.add_argument("--light-walk-audit", dest="audit", nargs=0, action=_Rd4Action, default=_rd4_argparse.SUPPRESS,
                    help="RD-4 test: every verdict the table or the probe count decides is also factored by the Cholesky and the disagreements counted")
    ap.add_argument("--perf-sym-extend-sector", dest="extend", nargs=0, action=_SymcAction, default=_rd4_argparse.SUPPRESS,
                    help="v9c5b: the primitive's chord extension V9cSpectra._extend by SYM-C1's sector eigh; default off")


def _rd4_patch_driver():
    # numba 0.61's ctypes binding prefers the _v2 symbols; the CUDA 13 driver gives some of these names new signatures (h100-f), so the base symbols are used for them
    from numba.cuda.cudadrv import driver as nbd
    if getattr(nbd.Driver, "_rd4_patched", False):
        return
    orig = nbd.Driver._find_api

    def find_api(self, fname):
        if fname in _RD4_CUDA13:
            try:
                return getattr(self.lib, fname)
            except AttributeError:
                pass
        return orig(self, fname)
    nbd.Driver._find_api = find_api
    nbd.Driver._rd4_patched = True


def _rd4_sturm_py(band0, dband, s, n, sigma, pivmin):
    # the count of eigenvalues of band0 + s dband below sigma (no-pivoting LDL^H on the band, half-bandwidth <= 4; banded_path_gpu_v1's sweep)
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
        dj = (b0 - sigma - (p11.real * p11.real + p11.imag * p11.imag) * d1 - (p22.real * p22.real + p22.imag * p22.imag) * d2
              - (p33.real * p33.real + p33.imag * p33.imag) * d3 - (p44.real * p44.real + p44.imag * p44.imag) * d4)
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


def _rd4_sturm2_py(band0, dband, s, n, sigma, pivmin):
    # _rd4_sturm_py for half-bandwidth <= 2 (the light H_+: rows 3 and 4 of the band exactly zero), the same recurrence without the zero terms
    zero = complex(0.0, 0.0)
    p11 = zero; p12 = zero; p22 = zero
    d1 = 0.0; d2 = 0.0
    cnt = 0
    for j in range(n):
        b0 = band0[0, j].real + s * dband[0, j].real
        b1 = (band0[1, j] + s * dband[1, j]).conjugate(); b2 = (band0[2, j] + s * dband[2, j]).conjugate()
        dj = b0 - sigma - (p11.real * p11.real + p11.imag * p11.imag) * d1 - (p22.real * p22.real + p22.imag * p22.imag) * d2
        if abs(dj) < pivmin:
            dj = -pivmin
        if dj < 0.0:
            cnt += 1
        rd = 1.0 / dj
        l1 = (b1 - p12 * (p11.conjugate() * d1)) * rd
        l2 = b2 * rd
        p22 = p12
        p11 = l1; p12 = l2
        d2 = d1; d1 = dj
    return cnt


def _rd4_kernels():
    # numba compiled once per process: the host count (and the host table when the engine runs on the CPU), the device table on the engine's GPU
    if _RD4_NB:
        return _RD4_NB
    import warnings
    import numba
    from numba import njit, prange
    try:
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        warnings.filterwarnings("ignore", message="The TBB threading layer")
    except Exception:
        pass
    try:
        numba.set_num_threads(max(1, min(int(numba.config.NUMBA_NUM_THREADS), int(_RD4["threads"]))))
    except Exception:
        pass
    sturm_cpu = njit(cache=False)(_rd4_sturm_py)
    sturm2_cpu = njit(cache=False)(_rd4_sturm2_py)

    def counts_cpu_py(band0, dband, n, s_arr, sig_arr, pivmin, two, out):
        for t in prange(s_arr.shape[0]):
            if two:
                out[t] = sturm2_cpu(band0, dband, s_arr[t], n, sig_arr[t], pivmin)
            else:
                out[t] = sturm_cpu(band0, dband, s_arr[t], n, sig_arr[t], pivmin)

    def bisect_cpu_py(band0, dband, n, s_arr, kidx, lo, hi, pivmin, tol, nmax, two, out_lo, out_hi):
        for e in prange(kidx.shape[0]):
            a_ = lo[e]; b_ = hi[e]; k = kidx[e]; s = s_arr[e]
            for it in range(nmax):
                if b_ - a_ <= tol:
                    break
                mid = 0.5 * (a_ + b_)
                if two:
                    c = sturm2_cpu(band0, dband, s, n, mid, pivmin)
                else:
                    c = sturm_cpu(band0, dband, s, n, mid, pivmin)
                if c > k:
                    b_ = mid
                else:
                    a_ = mid
            out_lo[e] = a_; out_hi[e] = b_
    nb = {"sturm_cpu": sturm_cpu, "sturm2_cpu": sturm2_cpu, "counts_cpu": njit(parallel=True, cache=False)(counts_cpu_py), "bisect_cpu": njit(parallel=True, cache=False)(bisect_cpu_py), "gpu": None}
    if _RD4["device"] is not None:
        _rd4_patch_driver()
        from numba import cuda
        cuda.select_device(int(_RD4["device"]))
        sturm_dev = cuda.jit(device=True)(_rd4_sturm_py)
        sturm2_dev = cuda.jit(device=True)(_rd4_sturm2_py)
        MS = _RD4_MS; TPB = _RD4_TPB

        def count_k_py(band0, dband, n, s_arr, sig_arr, pivmin, two, out):
            t = cuda.grid(1)
            if t >= s_arr.shape[0]:
                return
            if two:
                out[t] = sturm2_dev(band0, dband, s_arr[t], n, sig_arr[t], pivmin)
            else:
                out[t] = sturm_dev(band0, dband, s_arr[t], n, sig_arr[t], pivmin)

        def msect_k_py(band0, dband, n, s_arr, kidx, lo, hi, nsteps, pivmin, tol, two, out_lo, out_hi):
            sh = cuda.shared.array(TPB, numba.int32)
            flag = cuda.shared.array(1, numba.int32)
            tid = cuda.threadIdx.x
            g = tid // MS
            q = tid - g * MS
            e = cuda.blockIdx.x * (TPB // MS) + g
            T = kidx.shape[0]
            active = e < T
            k = 0; a_ = 0.0; b_ = 1.0; s = 0.0
            if active:
                k = kidx[e]; a_ = lo[e]; b_ = hi[e]; s = s_arr[e]
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
                if not done:
                    w = b_ - a_
                    if w <= tol:
                        done = True
                    else:
                        pt = a_ + (q + 1) * w / (MS + 1)
                        if two:
                            sh[tid] = sturm2_dev(band0, dband, s, n, pt, pivmin)
                        else:
                            sh[tid] = sturm_dev(band0, dband, s, n, pt, pivmin)
                cuda.syncthreads()
                if not done:
                    base = g * MS
                    na = a_; nbb = b_
                    for r in range(MS):
                        p_r = a_ + (r + 1) * (b_ - a_) / (MS + 1)
                        if sh[base + r] > k:
                            if p_r < nbb:
                                nbb = p_r
                        else:
                            if p_r > na:
                                na = p_r
                    a_ = na; b_ = nbb
                cuda.syncthreads()
            if active and q == 0:
                out_lo[e] = a_
                out_hi[e] = b_
        nb["gpu"] = {"cuda": cuda, "count": cuda.jit(count_k_py), "msect": cuda.jit(msect_k_py)}
    _RD4_NB.update(nb)
    return _RD4_NB


def _rd4_band(Hp):
    # the 5 upper diagonals band[m, j] = Hp[j, j + m] (band[0] real) of a Hermitian host CSR of half-bandwidth <= 4; None when wider
    co = Hp.tocoo(); off = co.col - co.row
    if off.size and int(np.max(np.abs(off))) > _RD4_KL:
        return None
    n = Hp.shape[0]; band = np.zeros((_RD4_KL + 1, n), dtype=np.complex128)
    for m_ in range(_RD4_KL + 1):
        band[m_, :n - m_] = np.asarray(Hp.diagonal(m_)).ravel()
    band[0] = band[0].real
    return np.ascontiguousarray(band)


def _rd4_gersh(band, shift=0.0):
    n = band.shape[1]; s = np.abs(band[0].real + shift)
    for m_ in range(1, _RD4_KL + 1):
        a_ = np.abs(band[m_, :n - m_]); s[:n - m_] += a_; s[m_:] += a_
    return float(s.max()) if n else 0.0


def _rd4_two(B0, dB):
    # half-bandwidth <= 2: the band rows 3 and 4 exactly zero
    return not (np.any(B0[3:] != 0) or np.any(dB[3:] != 0))


def _rd4_counts(B0, dB, s_arr, sig_arr, pivmin, two):
    nb = _rd4_kernels(); n = int(B0.shape[1]); T = int(len(s_arr))
    s_arr = np.ascontiguousarray(np.asarray(s_arr, float)); sig_arr = np.ascontiguousarray(np.asarray(sig_arr, float))
    if nb["gpu"] is not None:
        cuda = nb["gpu"]["cuda"]
        b0 = cuda.to_device(B0); b1 = cuda.to_device(dB); d_s = cuda.to_device(s_arr); d_sig = cuda.to_device(sig_arr)
        out = cuda.device_array(T, np.int32)
        nb["gpu"]["count"][(T + _RD4_TPB - 1) // _RD4_TPB, _RD4_TPB](b0, b1, n, d_s, d_sig, float(pivmin), bool(two), out)
        cuda.synchronize()
        return out.copy_to_host().astype(np.int64)
    out = np.zeros(T, np.int64)
    nb["counts_cpu"](B0, dB, n, s_arr, sig_arr, float(pivmin), bool(two), out)
    return out


def _rd4_brackets(B0, dB, s_arr, kidx, lo, hi, pivmin, two):
    nb = _rd4_kernels(); n = int(B0.shape[1]); T = int(len(kidx))
    s_arr = np.ascontiguousarray(np.asarray(s_arr, float)); kidx = np.ascontiguousarray(np.asarray(kidx, np.int64))
    lo = np.ascontiguousarray(np.asarray(lo, float)); hi = np.ascontiguousarray(np.asarray(hi, float))
    if nb["gpu"] is not None:
        cuda = nb["gpu"]["cuda"]
        b0 = cuda.to_device(B0); b1 = cuda.to_device(dB); d_s = cuda.to_device(s_arr); d_k = cuda.to_device(kidx)
        d_lo = cuda.to_device(lo); d_hi = cuda.to_device(hi)
        o_lo = cuda.device_array(T, np.float64); o_hi = cuda.device_array(T, np.float64)
        nblk = (T + (_RD4_TPB // _RD4_MS) - 1) // (_RD4_TPB // _RD4_MS)
        nb["gpu"]["msect"][nblk, _RD4_TPB](b0, b1, n, d_s, d_k, d_lo, d_hi, 32, float(pivmin), float(_RD4_TOL), bool(two), o_lo, o_hi)
        cuda.synchronize()
        return o_lo.copy_to_host(), o_hi.copy_to_host()
    o_lo = np.zeros(T); o_hi = np.zeros(T)
    nb["bisect_cpu"](B0, dB, n, s_arr, kidx, lo, hi, float(pivmin), float(_RD4_TOL), 80, bool(two), o_lo, o_hi)
    return o_lo, o_hi


def _rd4_gap_table(B0, dB, a0, a1, tg, Gb, two):
    # per grid point (rows) and anchor (columns): the count below a_j(t), the gap's lower and upper bounds, the two adjacent levels
    n = int(B0.shape[1]); m = len(a0); G = len(tg)
    pivmin = 64.0 * float(np.finfo(float).eps) * Gb
    S_all = np.repeat(tg, m); A_all = (a0[None, :] + tg[:, None] * (a1 - a0)[None, :]).ravel()
    M = int(S_all.size)
    cnt = _rd4_counts(B0, dB, S_all, A_all, pivmin, two)
    vb = cnt >= 1; va = cnt <= n - 1
    kidx = np.concatenate([np.where(vb, cnt - 1, 0), np.where(va, cnt, 0)])
    lo = np.concatenate([np.full(M, -Gb), A_all]); hi = np.concatenate([A_all, np.full(M, Gb)])
    blo, bhi = _rd4_brackets(B0, dB, np.concatenate([S_all, S_all]), kidx, lo, hi, pivmin, two)
    lb_lo, lb_hi, la_lo, la_hi = blo[:M], bhi[:M], blo[M:], bhi[M:]
    glo = np.minimum(np.where(vb, A_all - lb_hi, np.inf), np.where(va, la_lo - A_all, np.inf)) - _RD4_ERR
    ghi = np.minimum(np.where(vb, A_all - lb_lo, np.inf), np.where(va, la_hi - A_all, np.inf)) + _RD4_ERR
    lam_b = np.where(vb, 0.5 * (lb_lo + lb_hi), -np.inf); lam_a = np.where(va, 0.5 * (la_lo + la_hi), np.inf)
    _rd4_max("bracket_width_max", float(max(np.max(np.where(vb, lb_hi - lb_lo, 0.0)), np.max(np.where(va, la_hi - la_lo, 0.0)))))
    return cnt.reshape(G, m), glo.reshape(G, m), ghi.reshape(G, m), lam_b.reshape(G, m), lam_a.reshape(G, m)


def _rd4_near(ev, x):
    # the distance from x to the nearest entry of the sorted array ev
    i = int(np.searchsorted(ev, x)); best = np.inf
    for q in (i - 1, i):
        if 0 <= q < len(ev):
            best = min(best, abs(float(ev[q]) - float(x)))
    return best


def _rd4_prepare(h_seg, h1, seg, e_seg, e1, cache):
    # one walk segment (the anchors seg["a0"] -> seg["a1"] over the segment's parameter t in [0, 1]): the S = +1 band arrays of h at the ends and the anchors'
    # count check at both ends (the host count at a_j equals the boundary's index + 1); an anchor that fails keeps the ladder. The grid table is built later
    # (_rd4_table_build), at the segment's first verdict or after --light-walk-table-after verdicts.
    t_ = time.time(); m = len(seg["bnds"]); tab = {"ok": [False] * m, "tok": [False] * m, "built": False, "verdicts": 0}
    try:
        if "Hp1" not in cache:
            cache["Hp1"] = _symc_plus_host(h1.tocsr())
        Hp1, o1 = cache["Hp1"]
        Hp0, o0 = _symc_plus_host(h_seg.tocsr())
        if o0 != 0.0 or o1 != 0.0:
            _rd4_inc("segment_offblock"); return tab
        B0 = _rd4_band(Hp0); B1 = _rd4_band(Hp1)
        if B0 is None or B1 is None:
            _rd4_inc("segment_band_wide"); return tab
        dB = np.ascontiguousarray(B1 - B0)
        Gb = (_rd4_gersh(B0) + _rd4_gersh(dB)) * (1.0 + 1e-12) + 1e-300
        a0 = np.asarray(seg["a0"], float); a1 = np.asarray(seg["a1"], float)
        two = _rd4_two(B0, dB); pv = 64.0 * float(np.finfo(float).eps) * Gb; n = int(B0.shape[1])
        f = _rd4_kernels()["sturm2_cpu" if two else "sturm_cpu"]
        ok = []; L = []
        for j in range(m):
            b = int(seg["bnds"][j])
            L.append(min(float(seg["nG"][j]), _rd4_gersh(dB, -(a1[j] - a0[j]))))
            good = bool(int(f(B0, dB, 0.0, n, float(a0[j]), pv)) == b + 1 and int(f(B0, dB, 1.0, n, float(a1[j]), pv)) == b + 1)
            ok.append(good)
            if not good:
                _rd4_inc("segment_count_check_fail")
        tab.update({"ok": ok, "L": L, "B0": B0, "dB": dB, "a0": a0, "a1": a1, "n": n, "Gb": Gb, "two": two, "pivmin": pv, "f": f, "bnds": [int(b) for b in seg["bnds"]],
                    "e_seg": np.sort(np.asarray(e_seg, float)), "e1": np.sort(np.asarray(e1, float))})
        _rd4_inc("segments")
    except Exception as exc:
        _rd4_inc("segment_errors"); tab = {"ok": [False] * m, "tok": [False] * m, "built": True, "verdicts": 0}
        if _RD4_STATS.get("segment_errors", 0) <= 3:
            print("  v9c6 perf: the segment's band arrays failed (%s: %s); this segment keeps the ladder" % (type(exc).__name__, exc), flush=True)
    _rd4_inc("prepare_s", time.time() - t_)
    return tab


def _rd4_table_build(tab):
    # the gap functions on the grid for the anchors that passed the count check, with the level check at both ends: each tabulated level within the budget of a level
    # of the engine's own full spectrum there (sorted, both sectors: inside a numerically degenerate block the engine's sector labels follow its columns, not the sorted e)
    t_ = time.time(); tab["built"] = True
    try:
        G = int(_RD4["grid"]); tg = np.linspace(0.0, 1.0, G)
        cnt, glo, ghi, lb, la = _rd4_gap_table(tab["B0"], tab["dB"], tab["a0"], tab["a1"], tg, tab["Gb"], tab["two"])
        tok = []
        for j, b in enumerate(tab["bnds"]):
            dev_ = max(_rd4_near(tab["e_seg"], lb[0, j]), _rd4_near(tab["e_seg"], la[0, j]), _rd4_near(tab["e1"], lb[-1, j]), _rd4_near(tab["e1"], la[-1, j]))
            good = bool(tab["ok"][j] and int(cnt[0, j]) == b + 1 and int(cnt[-1, j]) == b + 1 and dev_ <= _RD4_ERR)
            _rd4_max("table_end_dev_max", dev_ if np.isfinite(dev_) else 1e300)
            tok.append(good)
            if tab["ok"][j] and not good:
                _rd4_inc("table_end_check_fail")
        tab.update({"tok": tok, "t": tg, "glo": glo, "ghi": ghi, "G": G})
        _rd4_inc("tables"); _rd4_inc("table_points", G * len(tok))
    except Exception as exc:
        _rd4_inc("table_errors")
        if _RD4_STATS.get("table_errors", 0) <= 3:
            print("  v9c6 perf: the gap table failed (%s: %s); this segment keeps the probe count" % (type(exc).__name__, exc), flush=True)
    _rd4_inc("table_s", time.time() - t_)


def _rd4_decide(tab, j, t1, d, F):
    # (verdict, tier) of the ladder's question "K_j(t1)^2 - d^2 positive definite": the table (tier 1), the count at the probe (tier 2); (None, None): the Cholesky decides
    ok = tab.get("ok") or []
    if j >= len(ok) or not ok[j]:
        return None, None
    m4 = 4.0 * F; d2 = d * d
    if tab["tok"][j]:
        tg = tab["t"]; G = tab["G"]
        k = int(t1 * (G - 1))
        k = 0 if k < 0 else (G - 2 if k > G - 2 else k)
        while k > 0 and tg[k] > t1:
            k -= 1
        while k < G - 2 and tg[k + 1] < t1:
            k += 1
        L = tab["L"][j]; dl = max(t1 - tg[k], 0.0); dr = max(tg[k + 1] - t1, 0.0)
        glo = max(tab["glo"][k, j] - L * dl, tab["glo"][k + 1, j] - L * dr)
        ghi = min(tab["ghi"][k, j] + L * dl, tab["ghi"][k + 1, j] + L * dr)
        if glo > 0.0 and d2 + m4 < glo * glo:
            return True, "table"
        if d2 - m4 > ghi * ghi:
            return False, "table"
    f = tab["f"]
    a = float(tab["a0"][j] + t1 * (tab["a1"][j] - tab["a0"][j]))
    B0 = tab["B0"]; dB = tab["dB"]; n = tab["n"]; pv = tab["pivmin"]
    r_out = math.sqrt(d2 + m4) + _RD4_ERR
    if f(B0, dB, t1, n, a + r_out, pv) == f(B0, dB, t1, n, a - r_out, pv):
        return True, "count"
    r2 = d2 - m4
    if r2 > 0.0:
        r_in = math.sqrt(r2) - _RD4_ERR
        if r_in > 0.0 and f(B0, dB, t1, n, a + r_in, pv) != f(B0, dB, t1, n, a - r_in, pv):
            return False, "count"
    return None, None


def _rd4_ladder(seg, j, t1):
    # the ladder of light_sector_walk_v9 at the probe t1 for anchor j (the same d sequence: d, halving to d_eff, the doubling check); returns (d_new or None, Cholesky calls)
    tab = seg.get("tab") or {"ok": [], "built": True, "verdicts": 0}
    if not tab["built"] and tab["verdicts"] >= int(_RD4["table_after"]) and any(tab["ok"]):
        _rd4_table_build(tab)
    F = seg["F"][j]; d_eff = seg["d_eff"][j]; d0 = seg["d"][j]
    st = {"ab": None, "calls": 0}

    def chol(d):
        if st["ab"] is None:
            Ks = (seg["K0"][j] + t1 * seg["dK"][j]).tocsr()
            st["ab"], _u = band_upper_v36(Ks @ Ks)
        return pd_banded_v36(st["ab"], d)

    def verdict(d):
        tab["verdicts"] = tab.get("verdicts", 0) + 1
        v, tier = _rd4_decide(tab, j, t1, d, F)
        if v is None:
            _rd4_inc("verdict_cholesky"); st["calls"] += 1
            return chol(d)
        _rd4_inc("verdict_" + tier)
        if _RD4["audit"]:
            _rd4_inc("audit_cholesky")
            if chol(d) != v:
                _rd4_inc("audit_disagree_" + tier)
        return v
    d_new = None; d_try = d0
    while d_try >= d_eff:
        if verdict(d_try):
            d_new = d_try
            break
        d_try *= 0.5
    if d_new is not None and d_new == d0 and d_new * 2.0 < 1.0:
        if verdict(2.0 * d_new):
            d_new = 2.0 * d_new
    return d_new, st["calls"]


def _rd4_walk_start():
    if _RD4["walk"] != "tabulated":
        return None
    with _RD4_LOCK:
        _RD4_LAST.clear()
    _rd4_inc("walks")
    return {}


def _rd4_selfcheck():
    # the kernels compiled and checked on small random band paths (half-bandwidth 2 and 4) against numpy's eigvalsh: counts exact, levels to 1e-11
    worst = 0.0
    for wide in (False, True):
        rng = np.random.default_rng(12345); n = 64
        B0 = np.zeros((_RD4_KL + 1, n), np.complex128); dB = np.zeros_like(B0)
        for m_ in range(5 if wide else 3):
            B0[m_, :n - m_] = rng.normal(size=n - m_) + (1j * rng.normal(size=n - m_) if m_ else 0.0)
            dB[m_, :n - m_] = 0.1 * (rng.normal(size=n - m_) + (1j * rng.normal(size=n - m_) if m_ else 0.0))
        B0[0] = B0[0].real; dB[0] = dB[0].real
        B0 = np.ascontiguousarray(B0); dB = np.ascontiguousarray(dB)
        two = _rd4_two(B0, dB)
        if two == wide:
            raise RuntimeError("v9c6 perf self-check: the bandwidth test")

        def dense(s):
            H = np.zeros((n, n), np.complex128)
            for m_ in range(_RD4_KL + 1):
                v_ = B0[m_, :n - m_] + s * dB[m_, :n - m_]
                H += np.diag(v_, m_)
                if m_:
                    H += np.diag(v_.conj(), -m_)
            return H
        tg = np.linspace(0.0, 1.0, 5); a0 = np.array([-0.3, 0.2]); a1 = np.array([-0.1, 0.4])
        Gb = (_rd4_gersh(B0) + _rd4_gersh(dB)) * (1.0 + 1e-12)
        cnt, glo, ghi, lb, la = _rd4_gap_table(B0, dB, a0, a1, tg, Gb, two)
        f = _rd4_kernels()["sturm2_cpu" if two else "sturm_cpu"]; g = _RD4_NB["sturm_cpu"]
        for k, s in enumerate(tg):
            ev = np.linalg.eigvalsh(dense(s))
            for j in range(2):
                a = a0[j] + s * (a1[j] - a0[j]); c = int(np.sum(ev < a))
                if c != int(cnt[k, j]):
                    raise RuntimeError("v9c6 perf self-check: count %d against %d (half-bandwidth %d)" % (int(cnt[k, j]), c, 4 if wide else 2))
                worst = max(worst, abs(lb[k, j] - ev[c - 1]), abs(la[k, j] - ev[c]))
                pv = 64.0 * float(np.finfo(float).eps) * Gb
                hc = int(f(B0, dB, float(s), n, float(a), pv)); hg = int(g(B0, dB, float(s), n, float(a), pv))
                if hc != c or hg != c:
                    raise RuntimeError("v9c6 perf self-check: host counts %d %d against %d" % (hc, hg, c))
    if not (worst <= 1e-11):
        raise RuntimeError("v9c6 perf self-check: levels off by %.3e" % worst)
    return worst


def rd4_set(a):
    if _SYMC.get("extend"):
        print("v9c6 perf: %s; --perf-sym-extend-sector on (the chord extension by the sector eigh)" % V9C6_VERSION, flush=True)
    if _RD4["walk"] != "tabulated":
        return
    if not sector_v9(a):
        raise SystemExit("v9c6 perf: --light-walk tabulated acts on the light sector walk: --cut-transport sector is required")
    if int(_RD4["grid"]) < 16:
        raise SystemExit("v9c6 perf: --light-walk-grid must be at least 16")
    if int(_RD4["table_after"]) < 0:
        raise SystemExit("v9c6 perf: --light-walk-table-after must be nonnegative")
    dev = BK.get("dev")
    _RD4["device"] = (int(dev.index or 0) if (dev is not None and getattr(dev, "type", "cpu") == "cuda") else None)
    try:
        _RD4["threads"] = int(_rd4_os.environ.get("OMP_NUM_THREADS") or 1)
    except ValueError:
        _RD4["threads"] = 1
    t_ = time.time()
    try:
        worst = _rd4_selfcheck()
    except Exception as exc:
        raise SystemExit("v9c6 perf: --light-walk tabulated needs numba (numba.cuda on a GPU run); the kernels failed their self-check (%s: %s)" % (type(exc).__name__, exc))
    print("v9c6 perf: %s; --light-walk tabulated (grid %d, the table after %d verdicts of a segment, on %s, host count threads %d, error budget %.0e, bracket width %.0e; audit %s); kernels compiled and checked in %.1f s (levels to %.1e)" % (
        V9C6_VERSION, int(_RD4["grid"]), int(_RD4["table_after"]), ("cuda:%d" % _RD4["device"]) if _RD4["device"] is not None else "the CPU", int(_RD4["threads"]), _RD4_ERR, _RD4_TOL,
        "on" if _RD4["audit"] else "off", time.time() - t_, worst), flush=True)


def rd4_close():
    if (_RD4["walk"] == "tabulated" or _SYMC.get("extend")) and (_RD4_STATS or _SYMC.get("extend")):
        print("  v9c6 perf (the run): %s" % ", ".join("%s %s" % (k, (("%.3g" % v) if isinstance(v, float) else v)) for k, v in sorted(_RD4_STATS.items())), flush=True)
# <<< v9c6 perf'''

OLD_BLOCK_ANCHOR = "# <<< v9c5 perf  # [v9c5 perf]\n"

OLD_WALK_START = '    pieces = []; crossings = []; nchol = 0; dmin_all = None; cross_len = 0.0; segments = 0; n_probe_cross = 0; end_resolved = set()   # v9 sector (review): the boundaries a crossing phase resolved at s = 1\n'
WALK_START_INS = '    rd4_c = _rd4_walk_start()\n'

OLD_LADDER = ('        if _H3["anchor_workers"]:  # [v9c4 perf]\n'
              '            jobs = [  # [v9c4 perf]\n'
              '                (  # [v9c4 perf]\n'
              '                    seg["K0"][j], seg["dK"][j], t1,  # [v9c4 perf]\n'
              '                    seg["d"][j], seg["d_eff"][j],  # [v9c4 perf]\n')
NEW_LADDER_HEAD = ('        if rd4_c is not None:\n'
                   '            if seg.get("tab") is None:\n'
                   '                seg["tab"] = _rd4_prepare(h_seg, h1, seg, e_seg, e1_np, rd4_c)\n'
                   '            for j in others:\n'
                   '                d_new, calls = _rd4_ladder(seg, j, t1)\n'
                   '                nchol += calls\n'
                   '                if d_new is None:\n'
                   '                    newly.append(j)\n'
                   '                    continue\n'
                   '                new_d[j] = d_new\n')
NEW_LADDER_ELIF = '        elif _H3["anchor_workers"]:  # [v9c4 perf]\n'

OLD_EXTEND = '            e, U = eigT_(gpu_(ch, N, v)); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1  # v9c prim\n'
EXTEND_HEAD = '            if _SYMC.get("extend"):\n                e, U = _symc_eig_light(ch, N, v); p.eigh_n = int(getattr(p, "eigh_n", 0)) + 1\n            else:\n'


class BuildStop(Exception):
    pass


class Rep(object):
    def __init__(self, name, old, new):
        self.name = name; self.old = old; self.new = new


def mark(text):
    out = []
    for line in text.split("\n"):
        out.append(line if (line.strip() == "" or MARK in line) else line + "  # " + MARK)
    return "\n".join(out)


def indent4(t):
    return "\n".join(("    " + x) if x.strip() else x for x in t.rstrip("\n").split("\n")) + "\n"


def after(old, ins):
    return old + mark(ins.rstrip("\n")) + "\n"


def branch(old, head):
    # head's last line is "else:"; the original statement(s) re-indented under it, marked (the reverse restores them verbatim)
    return mark(head.rstrip("\n")) + "\n" + mark(indent4(old).rstrip("\n")) + "\n"


def replacements():
    M = "  # " + MARK
    rest = OLD_LADDER.split("\n", 1)[1]
    return [
        Rep("P1 block", OLD_BLOCK_ANCHOR, OLD_BLOCK_ANCHOR + "\n\n" + mark(BLOCK) + "\n"),
        Rep("P2 args", "    _symc_add_arguments(ap)  # [v9c5 perf]\n", "    _symc_add_arguments(ap)  # [v9c5 perf]\n    _rd4_add_arguments(ap)" + M + "\n"),
        Rep("P3 set", "    symc_set(a)  # [v9c5 perf]\n", "    symc_set(a)  # [v9c5 perf]\n    rd4_set(a)" + M + "\n"),
        Rep("P3 close", "        symc_close()  # [v9c5 perf]\n", "        rd4_close()" + M + "\n        symc_close()  # [v9c5 perf]\n"),
        Rep("R4 walk start", OLD_WALK_START, after(OLD_WALK_START, WALK_START_INS)),
        Rep("R4 ladder", OLD_LADDER, mark(NEW_LADDER_HEAD.rstrip("\n")) + "\n" + NEW_LADDER_ELIF.rstrip("\n") + M + "\n" + rest),
        Rep("v9c5b extend", OLD_EXTEND, branch(OLD_EXTEND, EXTEND_HEAD)),
    ]


def precheck(text, reps, reverse=False):
    return [(r, text.count(r.new if reverse else r.old)) for r in reps if text.count(r.new if reverse else r.old) != 1]


def apply(text, reps, reverse=False):
    bad = precheck(text, reps, reverse)
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (r.name, "NEW" if reverse else "OLD", n) for r, n in bad))
    for r in (reversed(reps) if reverse else reps):
        old, new = (r.new, r.old) if reverse else (r.old, r.new)
        text = text.replace(old, new, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n")); bad = []
    for q, line in enumerate(out.split("\n"), 1):
        if line.strip() == "" or line in old:
            continue
        if MARK not in line:
            bad.append((q, line))
    return bad


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def write_new(path, text):
    if os.path.exists(path):
        raise BuildStop("the output %s exists; it is never overwritten" % path)
    with open(path, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c6 over engine_twofield_v9c5 (exact replacements; fails closed)")
    ap.add_argument("source"); ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    src = read(a.source); reps = replacements()
    try:
        if a.reverse:
            back = apply(src, reps, reverse=True)
            out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
            write_new(out, back)
            print("reverse: %s (md5 %s) -> %s (md5 %s)" % (a.source, md5(src), out, md5(back)))
            if a.against:
                ok = read(a.against) == back
                print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ok else "DIFFERS"))
                return 0 if ok else 2
            return 0
        if md5(src) != SOURCE_MD5:
            print("STOP: the source md5 is %s, not v9c5's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r:" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line[:160]))
        return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v9c5 byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c6.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c6 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c6.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
