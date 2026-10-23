#!/usr/bin/env python
"""pt_lib_v1.py (busi lane, analyst builder; inbox #428, the page-curve decisive tests). Shared machinery for Test A (reviewer A's
prescribed-recession chain runs, ReviewA_pc6_v1.md sec. 4) and Test B (reviewer B's state-resolved outgoing-packet transfer audit,
ReviewB_pc6_v1.md Round 4).

Engine operators reused, not rewritten (md5s printed by engine_md5s()):
  code_v6/page_ops_v6.py (31a85c00): chain_sparse (the light chain h_0 and the two-step current Q of the paper's k-channels),
      clocked_csr (h[N, v] = (1/2){N, h_0} + (1/2){v, Q}), FlowAverager, flow_dense, cheb_order (the Chebyshev order by the Bessel tail),
      CF4_C / CF4_ALPHA (the commutator-free fourth-order step's nodes and weights);
  code_v6/page_scatter_v5.py (82568abf): gamma_continuum (the continuum massless greybody Gamma_kappa(omega r_s) of the comparator L_H).
What is new here (and checked against the engine by selftest()):
  * the chain's h[N, v] applied on the GPU by one fused kernel per Chebyshev term (cupy RawKernel on torch memory), its diagonals built
    on the device from the site fields with exactly clocked_csr's arithmetic (checked against clocked_csr's CSR to rounding);
  * exp(-i tau h) X by the Chebyshev series (the order from OPS.cheb_order at tol 1e-15, the tail recorded), CF4 for time-dependent
    fields (OPS.cf4_propagate's node rule, v and N both linear in the fields);
  * occupations <X|f(h)|X> of a smoothed negative-energy step f(E) = erfc(E/(sqrt 2 ws))/2 (obs_raymap_solved_v9's
    cheb_step_occupation2 rule, J3P's smoothed negative spectral occupation) from Chebyshev moments computed once (the doubling
    mu_2k = 2<T_k|T_k> - mu_0, mu_2k+1 = 2<T_k+1|T_k> - mu_1), so several widths ws come from one sweep;
  * outgoing Killing-frequency packets built from WKB stationary modes of the local chain symbol (unit flux, Gaussian in omega), with
    the packet's energy weights rho(omega) kept for the transmission-weighted single-packet Fermi fit n = sum rho Gamma f(omega; T).
Device: --device cuda:1 or cuda:2 only (never cuda:0), or cpu (torch banded fallback, slow)."""
import hashlib
import math
import os
import sys
import time

import numpy as np
from scipy.special import erfc, jv
from scipy.optimize import brentq

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code_v6"))
import page_ops_v6 as OPS  # noqa: E402

CD = torch.complex128
RD = torch.float64
OFFS = (-4, -3, -1, 1, 3, 4)          # the massless chain's nonzero diagonals (h_0: +-1, +-3; Q: +-4)
_T0 = time.time()


def log(msg):
    print("[%9.1f s] %s" % (time.time() - _T0, msg), flush=True)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def engine_md5s():
    d = os.path.join(HERE, "code_v6")
    return {f: md5_file(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".py")}


# ------------------------------------------------------------------------------------------------------------------ the kernel
_KSRC = r'''
#include <cupy/complex.cuh>
extern "C" __global__ void cheb_term(const complex<double>* __restrict__ T0, const complex<double>* __restrict__ T1,
    complex<double>* __restrict__ T2, complex<double>* __restrict__ Y, const complex<double>* __restrict__ D,
    long long rows_total, long long rows, int K, int ldk, double scale, double cre, double cim, int first, int accum)
{
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= rows * (long long)K) return;
    long long r = idx / K; int c = (int)(idx % K);
    const int offs[6] = {-4, -3, -1, 1, 3, 4};
    complex<double> acc(0.0, 0.0);
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        long long rr = r + offs[j];
        if (rr >= 0 && rr < rows) acc += D[j * rows_total + r] * T1[rr * ldk + c];
    }
    long long o = r * ldk + c;
    complex<double> t2 = first ? scale * acc : scale * acc - T0[o];
    T2[o] = t2;
    if (accum) Y[o] += complex<double>(cre, cim) * t2;
}
'''
_DSRC = r'''
#include <cupy/complex.cuh>
extern "C" __global__ void coldots(const complex<double>* __restrict__ A, const complex<double>* __restrict__ B,
    double* __restrict__ outAA, double* __restrict__ outAB, long long rows, int K, int ldk, long long chunk)
{
    long long r0 = (long long)blockIdx.x * chunk;
    long long r1 = r0 + chunk; if (r1 > rows) r1 = rows;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        double s1 = 0.0, s2 = 0.0;
        for (long long r = r0; r < r1; r++) {
            complex<double> a = A[r * ldk + c], b = B[r * ldk + c];
            s1 += a.real() * a.real() + a.imag() * a.imag();
            s2 += a.real() * b.real() + a.imag() * b.imag();
        }
        atomicAdd(&outAA[c], s1);
        atomicAdd(&outAB[c], s2);
    }
}
'''
_CUPY = {}


def coldots(A, B, rows, K):
    """(sum_r |A_rc|^2, sum_r Re conj(A_rc) B_rc) per column c < K on the leading rows (device kernel; torch on a CPU)."""
    if not A.is_cuda:
        a = A[:rows, :K]; b = B[:rows, :K]
        return (a.conj() * a).real.sum(dim=0), (a.conj() * b).real.sum(dim=0)
    cp, _ = _kernel()
    if "d" not in _CUPY:
        _CUPY["d"] = cp.RawKernel(_DSRC, "coldots")
    o1 = torch.zeros(K, dtype=RD, device=A.device); o2 = torch.zeros(K, dtype=RD, device=A.device)
    chunk = 256
    nb = (rows + chunk - 1) // chunk
    with cp.cuda.Device(A.device.index):
        _CUPY["d"]((nb,), (min(1024, ((K + 31) // 32) * 32),), (cp.asarray(A), cp.asarray(B), cp.asarray(o1), cp.asarray(o2), np.int64(rows), np.int32(K),
                                  np.int32(A.shape[1]), np.int64(chunk)))
    return o1, o2


def _kernel():
    if "k" not in _CUPY:
        import cupy as cp
        _CUPY["cp"] = cp
        _CUPY["k"] = cp.RawKernel(_KSRC, "cheb_term")
    return _CUPY["cp"], _CUPY["k"]


STATS = {"terms": 0, "tail_acc": 0.0, "tail_max": 0.0, "K_max": 0, "exp_calls": 0}


class Chain:
    """the light chain of channel kappa on the sites R = 1..n (massless): h_0, Q from OPS.chain_sparse; h[N, v] on the device."""

    def __init__(self, n, kappa, device):
        self.n, self.kappa, self.device = int(n), float(kappa), torch.device(device)
        self.rows = 2 * self.n
        R = np.arange(1, self.n + 1, dtype=float)
        self.R = R
        h0, Q = OPS.chain_sparse(R, 0.0, self.kappa)
        self.h0, self.Q = h0, Q
        hd = np.zeros((len(OFFS), self.rows), complex)
        qd = np.zeros((len(OFFS), self.rows), complex)
        slo = np.zeros((len(OFFS), self.rows), np.int64)
        shi = np.zeros((len(OFFS), self.rows), np.int64)
        r = np.arange(self.rows)
        for j, d in enumerate(OFFS):
            m = self.rows - abs(d)
            rr = r[:m] if d > 0 else r[-d:]
            hd[j, rr] = h0.diagonal(d)
            qd[j, rr] = Q.diagonal(d)
            slo[j] = r // 2
            shi[j] = np.clip(r + d, 0, self.rows - 1) // 2
        # the diagonal entry (r, r + d) of (1/2){N, h0}: h0_{r,r+d} (N_{s(r)} + N_{s(r+d)})/2; of (1/2){v, Q} likewise
        dev = self.device
        self.hd = torch.as_tensor(hd, dtype=CD, device=dev)
        self.qd = torch.as_tensor(qd, dtype=CD, device=dev)
        self.slo = torch.as_tensor(slo, device=dev)
        self.shi = torch.as_tensor(shi, device=dev)
        self.Rt = torch.as_tensor(R, dtype=RD, device=dev)

    def diags(self, N, v):
        """the six diagonals (6, rows) of h[N, v] for device site fields N, v (n,) float64."""
        Nm = 0.5 * (N[self.slo] + N[self.shi])
        vm = 0.5 * (v[self.slo] + v[self.shi])
        return (self.hd * Nm + self.qd * vm).contiguous()

    @staticmethod
    def bound(D):
        return float(D.abs().sum(dim=0).max()) * 1.0001

    def csr(self, N, v):
        return OPS.clocked_csr(self.h0, self.Q, np.asarray(N, float), np.asarray(v, float))


def _launch(T0, T1, T2, Y, D, rows, K, scale, coef, first, accum):
    rows_total = D.shape[1]
    ldk = T1.shape[1]
    if T1.is_cuda:
        cp, k = _kernel()
        tot = rows * K
        bs = 256
        gs = (tot + bs - 1) // bs
        a = lambda t: cp.asarray(t) if t is not None else cp.asarray(T1)
        with cp.cuda.Device(T1.device.index):
            k((gs,), (bs,), (a(T0), a(T1), a(T2), a(Y), cp.asarray(D), np.int64(rows_total), np.int64(rows), np.int32(K), np.int32(ldk),
                             np.float64(scale), np.float64(coef.real), np.float64(coef.imag), np.int32(first), np.int32(accum)))
        return
    # CPU fallback: torch banded, same arithmetic
    X1 = T1[:rows, :K]
    acc = torch.zeros_like(X1)
    for j, d in enumerate(OFFS):
        Dj = D[j, :rows]
        if d > 0:
            acc[: rows - d] += Dj[: rows - d, None] * X1[d:]
        else:
            acc[-d:] += Dj[-d:, None] * X1[: rows + d]
    t2 = scale * acc if first else scale * acc - T0[:rows, :K]
    T2[:rows, :K] = t2
    if accum:
        Y[:rows, :K] += coef * t2


class Prop:
    """work buffers for Chebyshev series on (rows_total, ldk) column blocks; X is propagated in place on its leading rows and columns."""

    def __init__(self, rows_total, ldk, device):
        self.B = [torch.zeros((rows_total, ldk), dtype=CD, device=device) for _ in range(3)]
        self.rows_total, self.ldk = rows_total, ldk

    def apply_h(self, D, X, out, rows, K, w=1.0):
        """out = (h/w) X on the leading rows, columns."""
        _launch(None, X, out, None, D, rows, K, 1.0 / w, 0j, 1, 0)

    def expm(self, D, X, tau, rows, K, tol=1e-15, w=None):
        """X <- exp(-i tau h) X (tau of either sign) on X[:rows, :K]; returns the tail bound."""
        if tau == 0.0 or K == 0:
            return 0.0
        w = Chain.bound(D[:, :rows]) if w is None else w
        z = w * tau
        Kc, tail = OPS.cheb_order(z, tol)
        kk = np.arange(Kc + 1)
        ck = jv(kk, z) * ((-1j) ** kk) * np.where(kk == 0, 1.0, 2.0)
        Y, T1, T2 = self.B
        Xs = X[:rows, :K]
        Y[:rows, :K] = complex(ck[0]) * Xs
        _launch(None, X, T1, Y, D, rows, K, 1.0 / w, complex(ck[1]), 1, 1)
        T0 = X                        # X's own buffer becomes scratch once T_2 exists (Y holds the sum)
        for k_ in range(2, Kc + 1):
            _launch(T0, T1, T2, Y, D, rows, K, 2.0 / w, complex(ck[k_]), 0, 1)
            T0, T1, T2 = T1, T2, T0
        X[:rows, :K] = Y[:rows, :K]
        STATS["terms"] += Kc; STATS["tail_acc"] += tail; STATS["tail_max"] = max(STATS["tail_max"], tail)
        STATS["K_max"] = max(STATS["K_max"], Kc); STATS["exp_calls"] += 1
        return tail

    def series(self, D, X, coefs, rows, K, w):
        """Y = sum_k c_k T_k(h/w) X (real or complex c_k), returned as a new tensor (rows, K)."""
        Y = torch.zeros((self.rows_total, self.ldk), dtype=CD, device=X.device)
        T0 = X.clone()
        T1, T2 = self.B[1], self.B[2]
        Y[:rows, :K] = complex(coefs[0]) * T0[:rows, :K]
        if len(coefs) > 1:
            _launch(None, T0, T1, Y, D, rows, K, 1.0 / w, complex(coefs[1]), 1, 1)
        for k_ in range(2, len(coefs)):
            _launch(T0, T1, T2, Y, D, rows, K, 2.0 / w, complex(coefs[k_]), 0, 1)
            T0, T1, T2 = T1, T2, T0
        STATS["terms"] += len(coefs)
        return Y[:rows, :K]

    def moments(self, D, X, M, rows, K, w):
        """the Chebyshev moments mu_k = <X_c|T_k(h/w)|X_c>, k = 0 .. M - 1, per column (M, K) float64, by the doubling identities."""
        half = (M + 1) // 2 + 1
        mu = torch.zeros((M, K), dtype=RD, device=X.device)
        T0 = X.clone()
        T1, T2 = self.B[1], self.B[2]
        _launch(None, T0, T1, None, D, rows, K, 1.0 / w, 0j, 1, 0)
        mu0, mu1 = coldots(T0, T1, rows, K)
        mu[0] = mu0
        if M > 1:
            mu[1] = mu1
        # k = 1: mu_2 = 2<T1|T1> - mu0, mu_3 = 2<T2|T1> - mu1 ...
        Tk_prev, Tk = T0, T1          # T_{k-1}, T_k
        for k_ in range(1, half):
            dkk, dkp = coldots(Tk, Tk_prev, rows, K)
            if 2 * k_ < M:
                mu[2 * k_] = 2.0 * dkk - mu0
            if 2 * k_ - 1 < M and k_ >= 1:
                mu[2 * k_ - 1] = 2.0 * dkp - mu1
            if 2 * k_ + 1 >= M:
                break
            _launch(Tk_prev, Tk, T2, None, D, rows, K, 2.0 / w, 0j, 0, 0)
            Tk_prev, Tk, T2 = Tk, T2, Tk_prev
        STATS["terms"] += half
        mu[1] = mu1
        return mu


def step_coeffs(w, ws, M):
    """Chebyshev coefficients on [-w, w] of f(E) = erfc(E/(sqrt 2 ws))/2 (obs_raymap_solved_v9.cheb_step_coeffs's DCT), first M."""
    from scipy.fft import dct
    Mn = 1
    while Mn < max(8 * w / ws, 2 * M):
        Mn *= 2
    th = (np.arange(Mn) + 0.5) * math.pi / Mn
    fx = 0.5 * erfc(w * np.cos(th) / (math.sqrt(2.0) * ws))
    c = dct(fx, type=2) / Mn
    c[0] /= 2.0
    return c[:M]


def step_order(w, ws, tol=1e-13):
    c = step_coeffs(w, ws, int(16 * w / ws) + 64)
    keep = np.nonzero(np.abs(c) > tol)[0]
    return int(keep[-1]) + 1 if len(keep) else 1


def occupation_from_moments(mu, w, ws):
    """<X|f_ws(h)|X> per column from the moments (numpy (M, K)); the last coefficient's size returned as the truncation indicator."""
    M = mu.shape[0]
    c = step_coeffs(w, ws, M)
    return c @ mu, float(abs(c[-1]))


# ---------------------------------------------------------------------------------------------------------- the packets
def branch_k_vec(om, Nn, vn, yn, iters=14):
    """vectorized: the upper band's outgoing momentum k in (0, k_max) with v sin(2k)/2 + N sqrt(sin^2 k + y^2) = om (om > 0), by
    safeguarded Newton from om/(N + v); returns (k, ok) with ok where the residual is below 1e-13 om and the group velocity positive."""
    om = np.broadcast_to(np.asarray(om, float), np.broadcast(om, Nn).shape)
    k = np.clip(om / np.maximum(Nn + vn, 1e-3), 1e-12, 1.2)
    for _ in range(iters):
        s, c = np.sin(k), np.cos(k)
        e = np.sqrt(s * s + yn * yn)
        f = vn * 0.5 * np.sin(2 * k) + Nn * e - om
        fp = vn * np.cos(2 * k) + Nn * s * c / e
        step = f / np.where(np.abs(fp) > 1e-12, fp, 1e-12)
        k = np.clip(k - np.clip(step, -0.05, 0.05), 1e-12, 1.5)
    s, c = np.sin(k), np.cos(k)
    e = np.sqrt(s * s + yn * yn)
    f = vn * 0.5 * np.sin(2 * k) + Nn * e - om
    vg = vn * np.cos(2 * k) + Nn * s * c / e
    ok = (np.abs(f) <= 1e-13 * np.maximum(om, 1e-300) + 1e-17) & (vg > 0)
    return k, ok, vg, e


def wkb_packet(Nf, vf, kappa, omega0, sig_om, x0, n_sites, n_om=161, width_sd=7.0):
    """an outgoing packet of Killing frequency omega0 (sign = band: + particle, - the negative-energy branch) centred at the site x0
    at the observation clock: psi_n = sum_j g_j u(n; w_j) exp(i [Phi(n; w_j) - Phi(x0; w_j)]) / sqrt(v_g(n; w_j)), g_j the Gaussian
    amplitude exp(-(w_j - omega0)^2/(4 sig_om^2)) (energy density of standard deviation sig_om), Phi the cumulative WKB phase of the
    local outgoing momentum on the chain symbol h(k) = N(sin k sx + (kappa/R) sy) + v sin(2k)/2 (the negative band: the mirror
    k -> -k of the upper band at |omega|, lower-band spinor). Nf, vf: callables of R for the fields at that clock.
    Returns (psi (2 n_sites,) complex, info) with info["oms"], info["rho"] the packet's energy weights for the fits."""
    band = 1 if omega0 > 0 else -1
    oms = omega0 + sig_om * np.linspace(-6.0, 6.0, n_om)
    oms = oms[np.sign(oms) == band]
    sd_t = 1.0 / (2.0 * sig_om)
    lo = int(max(1, math.floor(x0 - width_sd * sd_t * 1.05)))
    hi = int(min(n_sites, math.ceil(x0 + width_sd * sd_t * 1.05)))
    Rn = np.arange(lo, hi + 1, dtype=float)
    Nn = Nf(Rn); vn = vf(Rn); yn = kappa / Rn
    g = np.exp(-(oms - omega0) ** 2 / (4.0 * sig_om * sig_om))
    ix0 = int(round(x0)) - lo
    psi = np.zeros((len(Rn), 2), complex)
    used = []
    dropped = 0.0
    for j, om in enumerate(oms):
        kup, ok, vg, e = branch_k_vec(abs(om), Nn, vn, yn)
        if not ok[ix0]:
            dropped += g[j] ** 2
            continue
        k = kup * band
        kz = np.where(ok, k, 0.0)
        Phi = np.zeros(len(Rn))
        Phi[1:] = np.cumsum(0.5 * (kz[1:] + kz[:-1]))
        Phi -= Phi[ix0]
        amp = np.where(ok, 1.0 / np.sqrt(np.where(ok, vg, 1.0)), 0.0)
        s = np.sin(k)
        u0 = (s - 1j * yn) / (math.sqrt(2.0) * e)
        u1 = band * e / (math.sqrt(2.0) * e)
        ph = g[j] * amp * np.exp(1j * Phi)
        psi[:, 0] += ph * u0
        psi[:, 1] += ph * u1
        used.append(j)
    if not used:
        raise RuntimeError("no omega of the packet reaches x0 on the outgoing branch")
    full = np.zeros((n_sites, 2), complex)
    full[lo - 1: hi] = psi
    nrm = math.sqrt(float(np.sum(np.abs(full) ** 2)))
    full /= nrm
    rho = g[used] ** 2
    dropped_frac = dropped / (rho.sum() + dropped)
    rho = rho / rho.sum()
    edge = float(np.sum(np.abs(full[lo - 1: lo - 1 + 20]) ** 2) + np.sum(np.abs(full[hi - 20: hi]) ** 2))
    _, ok0, vg0, _ = branch_k_vec(abs(omega0), Nn[ix0:ix0 + 1], vn[ix0:ix0 + 1], yn[ix0:ix0 + 1])
    info = {"omega0": float(omega0), "sig_om": float(sig_om), "x0": float(x0), "band": band, "oms": oms[used], "rho": rho, "lo": lo, "hi": hi,
            "vg0": float(vg0[0]), "sd_t": sd_t, "edge_weight": edge, "n_om_used": len(used), "n_om": len(oms), "dropped_weight": float(dropped_frac)}
    return full.reshape(-1), info


def packet_energy_check(chain, N_sites, v_sites, psi):
    """<h> and the energy standard deviation of a packet on the CSR h[N, v] (host; a check of the packet's Killing frequency)."""
    H = chain.csr(N_sites, v_sites)
    hp = H @ psi
    e1 = float(np.real(np.vdot(psi, hp)))
    e2 = float(np.real(np.vdot(hp, hp)))
    return e1, math.sqrt(max(e2 - e1 * e1, 0.0))


def retarded_travel(vf, R_a, R_b, nsub=4):
    """the continuum outgoing null ray's travel time from R_a to R_b (N = 1): int dr/(1 + v)."""
    rr = np.linspace(R_a, R_b, int(max(2, (R_b - R_a) * nsub)) + 1)
    f = 1.0 / (1.0 + vf(rr))
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(rr)))


# ------------------------------------------------------------------------------------------------------ the greybody and fits
_GT = {}


def gamma_k(kappa, x):
    """the continuum massless greybody Gamma_kappa(x = omega r_s) of page_scatter_v5 (cached on a log grid, interpolated)."""
    import page_scatter_v5 as SC
    key = int(kappa)
    if key not in _GT:
        xs = np.exp(np.linspace(math.log(5e-4), math.log(6.0), 220))
        G = []
        for xi in xs:
            try:
                G.append(SC.gamma_continuum(float(xi), float(kappa))[0])
            except RuntimeError:
                G.append(0.0)
        _GT[key] = (xs, np.array(G))
    xs, G = _GT[key]
    x = np.abs(np.asarray(x, float))
    lg = np.interp(np.log(np.maximum(x, xs[0])), np.log(xs), np.log(np.maximum(G, 1e-300)))
    return np.where(x < xs[0], 0.0, np.exp(lg))


def fermi(om, T):
    return 1.0 / (np.exp(np.clip(np.asarray(om, float) / T, -700, 700)) + 1.0)


def hawking_n(info, kappa, rs, T):
    """sum rho Gamma f(|omega|; T): the outgoing particle occupation (omega > 0) or hole occupation (omega < 0) of the packet."""
    om = np.abs(info["oms"])
    return float(np.sum(info["rho"] * gamma_k(kappa, om * rs) * fermi(om, T)))


def fit_T(info, kappa, rs, n_meas, T_ref):
    """the transmission-weighted single-packet Fermi fit: T with sum rho Gamma f(|omega|; T) = n_meas (n_meas the particle occupation
    for omega > 0, 1 - occupation for omega < 0); None outside the bracket."""
    om = np.abs(info["oms"]); G = gamma_k(kappa, om * rs); rho = info["rho"]
    f = lambda T: float(np.sum(rho * G / (np.exp(np.clip(om / T, -700, 700)) + 1.0))) - n_meas
    try:
        return brentq(f, 1e-4 * T_ref, 1e3 * T_ref, xtol=1e-16)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest(device="cpu", n=600, kappa=1.0, seed=1):
    """the fused kernel's h against OPS.clocked_csr, the Chebyshev exponential against OPS.cheb_expm (the engine's own, N = 1) and
    against expm on a dense h, the moments against direct series; returns a dict of the largest deviations."""
    rng = np.random.default_rng(seed)
    ch = Chain(n, kappa, device)
    R = ch.R
    N = 1.0 - 0.3 * np.exp(-((R - 40) / 15) ** 2)
    v = -np.sqrt(30.0 / np.maximum(R, 5.0)) * (1 - np.exp(-R / 10.0))
    Nt = torch.as_tensor(N, dtype=RD, device=device); vt = torch.as_tensor(v, dtype=RD, device=device)
    D = ch.diags(Nt, vt)
    H = ch.csr(N, v)
    out = {}
    X = rng.standard_normal((ch.rows, 5)) + 1j * rng.standard_normal((ch.rows, 5))
    Xt = torch.as_tensor(X, dtype=CD, device=device)
    P = Prop(ch.rows, 5, device)
    Yt = torch.zeros_like(Xt)
    P.apply_h(D, Xt, Yt, ch.rows, 5, 1.0)
    out["apply_vs_clocked_csr"] = float(np.max(np.abs(Yt.cpu().numpy() - H @ X)))
    # exponential against the dense expm
    from scipy.linalg import expm
    Hd = H.toarray()
    U = expm(-1j * 7.3 * Hd)
    Xc = Xt.clone()
    P.expm(D, Xc, 7.3, ch.rows, 5)
    out["expm_vs_dense"] = float(np.max(np.abs(Xc.cpu().numpy() - U @ X)))
    Xb = Xt.clone()
    P.expm(D, Xb, -7.3, ch.rows, 5)
    out["expm_back_vs_dense"] = float(np.max(np.abs(Xb.cpu().numpy() - U.conj().T @ X)))
    # the engine's own Chebyshev on N = 1
    ones = torch.ones(n, dtype=RD, device=device)
    D1 = ch.diags(ones, vt)
    B = OPS.ChainBanded(ch.h0, ch.Q, R, torch.device("cpu"))
    Xe = OPS.cheb_expm(B, torch.as_tensor(X, dtype=CD), 5.0, torch.as_tensor(v, dtype=RD))
    X1 = Xt.clone(); P.expm(D1, X1, 5.0, ch.rows, 5)
    out["expm_vs_engine_cheb_expm"] = float(np.max(np.abs(X1.cpu().numpy() - Xe.numpy())))
    # moments against the direct series and the dense spectral step
    w = Chain.bound(D)
    M = 400
    mu = P.moments(D, Xt, M, ch.rows, 5, w).cpu().numpy()
    ws = 0.05
    occ, _ = occupation_from_moments(mu, w, ws)
    e, V = np.linalg.eigh(Hd)
    fE = 0.5 * erfc(e / (math.sqrt(2) * ws))
    occ_d = np.real(np.einsum("ij,i,ij->j", (V.conj().T @ X).conj(), fE, V.conj().T @ X))
    out["moments_vs_dense_step"] = float(np.max(np.abs(occ - occ_d) / np.abs(occ_d)))
    c = step_coeffs(w, ws, M)
    Ys = P.series(D, Xt, c, ch.rows, 5, w).cpu().numpy()
    occ_s = np.real(np.sum(X.conj() * Ys, axis=0))
    out["series_vs_moments"] = float(np.max(np.abs(occ_s - occ) / np.abs(occ)))
    return out
