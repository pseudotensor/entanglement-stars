#!/usr/bin/env python3
"""obs_box3d_v2.py (version 2 of obs_box3d_v1.py, whose smoke record smoke_L32.json was written by version 1; the only change: the
option --dtype complex64 | complex128 sets the evolution's precision in the j4 and timing modes, Z~ stored in complex64 either way, and the
timing record's name carries the precision; and the Jackson weight's factor (K + 2 - k) for K + 1 moments, version 1 had K + 1 - k, so its
g_0 was (K + 1)/(K + 2) instead of 1; version 1's smoke record does not use the small-ball routine).  The lead's design with the key routines drafted; SPEC_final4b_observables.md secs. 1.2,
4.2, 4.4; J4 and J5 on GPU.

The 3-dimensional periodic box L^3 of 4-component sites: H_m[N, v] by the sine stencil with the site-symmetrized lapse and the two-step flow
term with the endpoint-symmetric v^i = v(R) x^i/R; the CF4 propagator with Chebyshev exponentials; the flat sea by the exact momentum-space
projector; the Heisenberg route for every region quantity: C_A(t) = Z^dagger Z with Z = Pi_sea U(t)^dagger Pi_A (the region's basis evolved
backward through the history and projected on the initial sea; the dust orbitals of task 2 add Pi_dust to the projector).  Regions: balls and
shells; observables: entropies, mutual information, the cross block and its two-point functions, the partner of a 3-dimensional packet, the
small-ball comparator (J9's method).  BUILDER: the memory plan for Z (complex64 on the GPU for n_A <= 8600 at L = 64; chunks on the host
beyond), the history format from the engine, the lattice Y_lm packets, the smoke-test harness (L = 32: pure-state identity, idempotency,
the dt ladder), the writer.

Builder's completion (analyst; every choice labelled BUILDER here and in the records; the lead's routines are kept, apply_h_ref is the
draft's apply_h unchanged and is the reference the kernel is checked against):
(B1) Boundary: antiperiodic in all 3 directions (the sign -1 on the bonds that wrap, folded into the bond arrays), because at m = 0 a periodic
     grid of any L holds the node k = 0 (an odd L does not cure it: the periodic grid always holds k = 0, and an odd antiperiodic grid holds
     k = pi); with an even L the antiperiodic grid k = 2 pi (n + 1/2)/L holds no node; the flat sea's FFT carries the twist phase.  The twist
     is a diagonal gauge transformation away from a uniform flux, so every region spectrum is that of the twisted torus.
(B2) The box centre at the cell centre c = (L - 1)/2 (even L): the twisted box is then symmetric under the cubic group about the centre
     (with c = L//2 the box breaks inversion by one site); R >= sqrt(3)/2 at every site, so v^i = v(R) x^i/R needs no special site.
(B3) The stencil by one fused CUDA kernel (CuPy RawKernel; one read of each neighbour and one write per application, the Chebyshev
     recursion T_{k+1} = 2 h T_k/r - T_{k-1} fused into the same pass, in place on T_{k-1}); the draft's einsum route needs about 90
     memory passes per application.  The kernel is checked against apply_h_ref (smoke).
(B4) The Chebyshev bound per call r(s) = sqrt(3 + m^2) + 3 max|Nb - 1| + m max|N - 1| + |s| (sqrt(3)/2) max_x |vb(x)| (a rigorous norm bound:
     the uniform Dirac part's exact norm, Gershgorin on the lapse's deviation, Cauchy-Schwarz on the two-step flow term), instead of the
     draft's Gershgorin 3 + m + 1.5 max|v^i| at s = 1 (the CF4 weights reach |s| = 1.077, which the draft's bound did not cover); on the
     exact profile at r_s = 8, R_m = 2 it is 3.46 against 6.0, i.e. 42 percent fewer terms; both bounds give the same result (smoke).
(B5) The holds are chained: exp(+i h_f h_j) X = exp(+i h_f (h_j - h_{j-1})) exp(+i h_f h_{j-1}) X, one Chebyshev segment per hold
     interval, so all holds cost the largest hold's term count with one accumulator (sharing one recursion with an accumulator per hold
     costs the same applications plus a third of an application per accumulator, measured); each hold's ramp backward and compression run
     as soon as its segment ends.  cheb_expm_multi (the shared form) is kept for reference.
(B6) The sea's projection is stored compressed: Z~(k) = V_-(k)^dagger FFT(phase Y)(k) with V_-(k) the 2 occupied spinors per momentum (unitary
     FFT), half the storage of P Y, the same Gram matrix; Z~ is kept in complex64; the Gram C = Z~^dagger Z~ is accumulated in complex128 over
     thin row slabs (slab GEMMs in complex64 without TF32, the slab results summed in complex128; per-entry error about eps_32 slab/sqrt(M));
     the complex128 GEMM runs at 0.52 TFLOPS on the A6000 against 17.5 for complex64 (measured), so the slab route is the default and the
     complex128 Gram is the smoke's comparator.  Eigenvalues in complex128.  The region Gram is PSD by construction, so the rounding of Z~
     moves the eigenvalues near 0 only at second order.
(B7) One region holds every J4 region: the ball R < 1.5 r_s contains the ball R < r_s and the shells [r_s - d, r_s + d) for d <= 4 when
     r_s + 4 <= 1.5 r_s (r_s >= 8); its block gives every other block as a sub-block (one evolution instead of five).
(B8) The small ball (J9's method): Theta(mu - h) by the Chebyshev expansion of the smoothed step (1 - erf((h - mu)/w))/2 with w = gap/4.75
     (erfc(4.75)/2 = 1.2e-11 at the gap edge; Chebyshev coefficients decaying like exp(-(k w/r)^2/4), so the term count from the gap is
     K = 2 sqrt(ln(1/tol)) r/w); the Jackson-damped expansion of the step (the spec's kernel) is implemented beside it and compared on a
     dense small box: its error at the gap edge falls only algebraically (about (K gap/r)^-3), see the smoke record.
(B10, version 2) The particle-hole Gram entropy: with a complex64 evolution the region's basis is an isometry only to about 1e-7, which moves
     the eigenvalues of C near 1 at first order (the smoke's whole-box check: S(A) - S(complement) = -0.024 at L = 12 in complex64 against
     -3e-9 in complex128); storing also the empty-band amplitudes gives C-bar = (Q Y)^dagger (Q Y), and S = sum_{lambda(C) < 1/2} h2 +
     sum_{mu(C-bar) < 1/2} h2 takes every small eigenvalue from a PSD Gram matrix (second order in the defect); the cost is twice the
     storage and the Gram time; option --ph in j4, checked by mode phcheck against complex128.
(B9) The 3-dimensional partial-wave packet (j = 1/2): psi(x) = (1/R)[u_1(R) Omega_{kappa, m_j}(n); -u_2(R) Omega_{-kappa, m_j}(n)] with
     kappa = -kappa_j (kappa_j the chain's sign convention of flow_chart_v4, on-site m SZ + (kappa_j/R) SY), Omega_{-1, m} = chi_m/sqrt(4 pi),
     Omega_{+1, m} = -(sigma.n) chi_m/sqrt(4 pi) (the lattice harmonic by the sites' direction cosines), (u_1, u_2)(R) the radial packet of
     the chain (Gaussian envelope of density width sigma_x, the carrier k_0 of Killing frequency omega on the local outgoing branch of the
     chain's symbol at R_0, the local positive spinor flow_chart_v4.onsite_spinor(k_0, m, kappa_j/R_0)); the map is derived from the radial
     Dirac equations and checked by the packet's energy against the chain's (c3d record).
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import jv, erf  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]], complex); SZ = np.diag([1.0, -1.0]).astype(complex)
GAMMA = np.array([np.kron(SX, s) for s in (SX, SY, SZ)]); BETA = np.kron(SZ, np.eye(2))
SIG = np.array([SX, SY, SZ])
T0 = time.time()
LABEL_PG = ("exact Painleve-Gullstrand profile (control; N = 1, v = -sqrt(r_s/R) outside the ball R_m, the ball's linear profile inside; "
            "v declared); truncated by (1 - tanh((R - R_cut)/4))/2 on the periodic box (a declared truncation of the far exterior)")


def log(msg):
    print("[%7.0f s] %s" % (time.time() - T0, msg), flush=True)


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


# ------------------------------------------------------------------------------------------------ profiles
def pg_v(R, rs, Rm):
    """the exact Painleve-Gullstrand profile (control): v = -sqrt(r_s/R) for R >= R_m, -sqrt(r_s/R_m^3) R inside."""
    R = np.asarray(R, float)
    return np.where(R >= Rm, -np.sqrt(rs / np.maximum(R, 1e-300)), -np.sqrt(rs / Rm ** 3) * R)


def truncation(R, Rcut, width=4.0):
    return 0.5 * (1.0 - np.tanh((np.asarray(R, float) - Rcut) / width))


def gapped_v(R, rs, amp=0.6):
    """J9's prescribed gapped profile: v = -amp sqrt(r_s/R) for R >= R_m = 2 r_s; BUILDER (the spec leaves the inner part open): the C^1 odd
    cubic v = V (7/4 (R/R_m) - 3/4 (R/R_m)^3) inside, V = v(R_m), matching value and slope at R_m (max|v| = 1.029 |V| = 0.437; a smooth
    vector field v^i = (a + b R^2) x^i at the centre)."""
    R = np.asarray(R, float)
    Rm = 2.0 * rs
    V = -amp * math.sqrt(rs / Rm)
    u = R / Rm
    return np.where(R >= Rm, -amp * np.sqrt(rs / np.maximum(R, 1e-300)), V * (1.75 * u - 0.75 * u ** 3))


def box_coords(L):
    """site coordinates relative to the cell centre c = (L - 1)/2 (B2), shape (3, L, L, L), and R."""
    c = (L - 1) / 2.0
    g = np.indices((L, L, L)).astype(float) - c
    return g, np.sqrt((g ** 2).sum(0))


def radial_field(L, vfun):
    """N = 1 and v^i = v(R) x^i/R on the box (B2: R > 0 at every site)."""
    g, R = box_coords(L)
    v = vfun(R)
    return np.ones((L, L, L)), [v * g[i] / R for i in range(3)], R


# ------------------------------------------------------------------------------------------------ the stencil kernel (B3)
KERNEL_SRC = r'''
#include <cupy/complex.cuh>
template<typename R>
__device__ __forceinline__ void hpsi_impl(const complex<R>* __restrict__ psi, complex<R>* out, const complex<R>* prev,
    const R* __restrict__ Nb, const R* __restrict__ vb, const R* __restrict__ mN, const int L, const int nb,
    const R s, const R alpha, const R beta)
{
    const long long L3 = (long long)L * L * L;
    const long long total = L3 * 4LL * nb;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int n = (int)(idx % nb);
    const long long r = idx / nb;
    const int a = (int)(r & 3);
    const long long site = r >> 2;
    int c[3];
    c[2] = (int)(site % L); c[1] = (int)((site / L) % L); c[0] = (int)(site / ((long long)L * L));
    long long st[3];
    st[0] = (long long)L * L; st[1] = L; st[2] = 1;
    complex<R> hop((R)0, (R)0), flow((R)0, (R)0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int ci = c[i];
        const int cp1 = (ci + 1 < L) ? ci + 1 : ci + 1 - L;
        const int cm1 = (ci >= 1) ? ci - 1 : ci - 1 + L;
        const int cp2 = (ci + 2 < L) ? ci + 2 : ci + 2 - L;
        const int cm2 = (ci >= 2) ? ci - 2 : ci - 2 + L;
        const long long sp1 = site + (long long)(cp1 - ci) * st[i];
        const long long sm1 = site + (long long)(cm1 - ci) * st[i];
        const long long sp2 = site + (long long)(cp2 - ci) * st[i];
        const long long sm2 = site + (long long)(cm2 - ci) * st[i];
        const int ap = (i == 2) ? (a ^ 2) : (3 - a);
        const complex<R> D = Nb[i * L3 + site] * psi[(sp1 * 4 + ap) * nb + n] - Nb[i * L3 + sm1] * psi[(sm1 * 4 + ap) * nb + n];
        complex<R> cf;
        if (i == 0) cf = complex<R>((R)0, (R)-0.5);
        else if (i == 1) cf = complex<R>((a & 1) ? (R)0.5 : (R)-0.5, (R)0);
        else cf = complex<R>((R)0, (a & 1) ? (R)0.5 : (R)-0.5);
        hop += cf * D;
        flow += vb[i * L3 + site] * psi[(sp2 * 4 + a) * nb + n] - vb[i * L3 + sm2] * psi[(sm2 * 4 + a) * nb + n];
    }
    complex<R> acc = hop + complex<R>((R)0, (R)-0.25 * s) * flow;
    acc += ((a < 2) ? mN[site] : -mN[site]) * psi[idx];
    complex<R> res = alpha * acc;
    if (beta != (R)0) res += beta * prev[idx];
    out[idx] = res;
}
extern "C" __global__ void hpsi_c64(const complex<float>* psi, complex<float>* out, const complex<float>* prev, const float* Nb,
    const float* vb, const float* mN, const int L, const int nb, const float s, const float alpha, const float beta)
{ hpsi_impl<float>(psi, out, prev, Nb, vb, mN, L, nb, s, alpha, beta); }
extern "C" __global__ void hpsi_c128(const complex<double>* psi, complex<double>* out, const complex<double>* prev, const double* Nb,
    const double* vb, const double* mN, const int L, const int nb, const double s, const double alpha, const double beta)
{ hpsi_impl<double>(psi, out, prev, Nb, vb, mN, L, nb, s, alpha, beta); }
'''
_KERN = {}


def _kernel(dtype):
    import cupy as cp
    key = (torch.cuda.current_device(), dtype)
    if key not in _KERN:
        mod = cp.RawModule(code=KERNEL_SRC)
        _KERN[key] = mod.get_function("hpsi_c64" if dtype == torch.complex64 else "hpsi_c128")
    return _KERN[key]


class Fields:
    """N(x), v^i(x) on the box; the bond averages prepared once: Nb[i] = (N(x) + N(x + e_i))/2, vb[i] = (v^i(x) + v^i(x + 2 e_i))/2,
    with the twist sign on the bonds that wrap (B1); the draft's complex tensors for apply_h_ref and the real arrays for the kernel (B3)."""

    def __init__(self, L, N, vvec, m, dev, dtype=torch.complex64, antiperiodic=True):
        self.L, self.m, self.dev, self.dtype, self.antiperiodic = L, float(m), dev, dtype, antiperiodic
        N = np.asarray(N, float)
        vvec = [np.asarray(v, float) for v in vvec]
        s1 = [np.ones((L, L, L)) for _ in range(3)]
        s2 = [np.ones((L, L, L)) for _ in range(3)]
        if antiperiodic:
            for i in range(3):
                ix = [slice(None)] * 3
                ix[i] = L - 1
                s1[i][tuple(ix)] = -1.0
                ix[i] = slice(L - 2, L)
                s2[i][tuple(ix)] = -1.0
        Nb_avg = [0.5 * (N + np.roll(N, -1, axis=i)) for i in range(3)]
        vb_avg = [0.5 * (vvec[i] + np.roll(vvec[i], -2, axis=i)) for i in range(3)]
        Nb = [Nb_avg[i] * s1[i] for i in range(3)]
        vb = [vb_avg[i] * s2[i] for i in range(3)]
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=dev)
        self.N = t(N)
        self.Nb = [t(b) for b in Nb]
        self.vb = [t(b) for b in vb]
        self.G = [torch.as_tensor(g, dtype=dtype, device=dev) for g in GAMMA]
        self.B = torch.as_tensor(BETA, dtype=dtype, device=dev)
        rdt = torch.float32 if dtype == torch.complex64 else torch.float64
        self.kNb = torch.as_tensor(np.stack(Nb).reshape(3, -1), dtype=rdt, device=dev).contiguous()
        self.kvb = torch.as_tensor(np.stack(vb).reshape(3, -1), dtype=rdt, device=dev).contiguous()
        self.kmN = torch.as_tensor((m * N).reshape(-1), dtype=rdt, device=dev).contiguous()
        self.bound_gershgorin = float(3.0 * np.max(N) + m * np.max(N) + 1.5 * max(np.max(np.abs(v)) for v in vvec)) * 1.001
        self.r_dirac = math.sqrt(3.0 + m * m) + 3.0 * max(float(np.max(np.abs(b - 1.0))) for b in Nb_avg) + m * float(np.max(np.abs(N - 1.0)))
        self.r_flow = (math.sqrt(3.0) / 2.0) * float(np.max(np.sqrt(sum(b ** 2 for b in vb_avg))))
        self.use_gershgorin = False
        self.bound = self.bound_s(1.0)
        self.vmax = float(np.max(np.sqrt(sum(v ** 2 for v in vvec))))

    def bound_s(self, s):
        if self.use_gershgorin:
            return self.bound_gershgorin * max(1.0, abs(s))
        return (self.r_dirac + abs(s) * self.r_flow) * 1.0001 + 1e-6


def apply_h_ref(psi, F, s=1.0, flow_only=False):
    """the draft's apply_h (kept unchanged as the reference): h psi for psi of shape (L, L, L, 4, nb)."""
    out = torch.zeros_like(psi)
    for i in range(3):
        if not flow_only:
            fwd = F.Nb[i][..., None, None] * torch.roll(psi, -1, dims=i)
            bwd = torch.roll(F.Nb[i], 1, dims=i)[..., None, None] * torch.roll(psi, 1, dims=i)
            out += torch.einsum("ab,xyzbn->xyzan", F.G[i], fwd - bwd) / 2j
        if s != 0.0:
            fwd = F.vb[i][..., None, None] * torch.roll(psi, -2, dims=i)
            bwd = torch.roll(F.vb[i], 2, dims=i)[..., None, None] * torch.roll(psi, 2, dims=i)
            out += s * (fwd - bwd) / 4j
    if not flow_only:
        out += F.m * F.N[..., None, None] * torch.einsum("ab,xyzbn->xyzan", F.B, psi)
    return out


def apply_h(psi, F, s=1.0, alpha=1.0, beta=0.0, prev=None, out=None):
    """out = alpha h(s) psi + beta prev by the kernel (B3); psi contiguous of shape (L, L, L, 4, nb); out may alias prev, never psi."""
    import cupy as cp
    if out is None:
        out = torch.empty_like(psi)
    if prev is None:
        prev, beta = psi, 0.0
    rt = np.float32 if F.dtype == torch.complex64 else np.float64
    tot = psi.numel()
    _kernel(F.dtype)(((tot + 255) // 256,), (256,), (cp.from_dlpack(psi), cp.from_dlpack(out), cp.from_dlpack(prev), cp.from_dlpack(F.kNb),
                                                     cp.from_dlpack(F.kvb), cp.from_dlpack(F.kmN), np.int32(F.L), np.int32(psi.shape[-1]),
                                                     rt(s), rt(alpha), rt(beta)))
    return out


# ------------------------------------------------------------------------------------------------ Chebyshev exponentials
def term_count(a, tol=1e-15):
    a = abs(a)
    K = int(a + 12)
    while abs(jv(K, a)) > tol or K < a + 4:
        K += 4
    return K


def cheb_coeffs(a, K):
    """c_k of exp(-i a x) = sum_k c_k T_k(x): (2 - delta_k0) (-i)^k J_k(a), with J_k(-|a|) = (-1)^k J_k(|a|)."""
    k = np.arange(K + 1)
    J = jv(k, abs(a))
    if a < 0:
        J = J * (-1.0) ** k
    return (2.0 - (k == 0)) * ((-1j) ** k) * J


def cheb_expm_multi(F, X, taus, s, tol=1e-15, consume=False):
    """[exp(-i tau h(s)) X for tau in taus] from one Chebyshev recursion (B5); X is overwritten when consume."""
    r = F.bound_s(s)
    Ks = [term_count(r * t, tol) for t in taus]
    K = max(Ks)
    cs = [cheb_coeffs(r * t, K) for t in taus]
    T0_ = X if consume else X.clone()
    T1_ = apply_h(T0_, F, s, alpha=1.0 / r)
    Ys = [complex(c[0]) * T0_ + complex(c[1]) * T1_ for c in cs]
    for k in range(2, K + 1):
        apply_h(T1_, F, s, alpha=2.0 / r, beta=-1.0, prev=T0_, out=T0_)
        T0_, T1_ = T1_, T0_
        for j, c in enumerate(cs):
            if k <= Ks[j]:
                Ys[j].add_(T1_, alpha=complex(c[k]))
    del T0_, T1_
    return Ys


def cheb_expm(F, X, tau, s, tol=1e-15):
    """exp(-i tau h(s)) X by the Chebyshev expansion on [-r, r] (the draft's interface; r by B4)."""
    return cheb_expm_multi(F, X, [tau], s, tol)[0]


SQ3 = math.sqrt(3.0)
CF_C = (0.5 - SQ3 / 6, 0.5 + SQ3 / 6)
CF_A = (0.25 + SQ3 / 6, 0.25 - SQ3 / 6)


def ramp_envelope(t_r):
    return lambda t: 0.0 if t <= 0 else (1.0 if t >= t_r else math.sin(math.pi * t / (2 * t_r)) ** 2)


def history_ramp(t_r, dt, envelope):
    """the exact-profile control's history: the list of (s_a, s_b, h) per CF4 step for v(t) = envelope(t) v; the engine's history replaces
    envelope by the stored (N(t), v(t)) snapshots (BUILDER, J4': a Fields object per snapshot and an interpolation rule between steps)."""
    nst = max(1, int(math.ceil(t_r / dt - 1e-9)))
    h = t_r / nst
    steps = []
    for i in range(nst):
        t = i * h
        s1, s2 = envelope(t + CF_C[0] * h), envelope(t + CF_C[1] * h)
        steps.append((2 * (CF_A[0] * s1 + CF_A[1] * s2), 2 * (CF_A[1] * s1 + CF_A[0] * s2), h))
    return steps


def evolve(F, X, steps, hold=0.0, backward=False):
    """U X (forward) or U^dagger X (backward) through the CF4 steps and a hold of length `hold` on the final background (s = 1)."""
    if backward:
        if hold > 0:
            X = cheb_expm(F, X, -hold, 1.0)
        for sa, sb, h in reversed(steps):
            X = cheb_expm(F, X, -h / 2, sb)
            X = cheb_expm(F, X, -h / 2, sa)
        return X
    for sa, sb, h in steps:
        X = cheb_expm(F, X, h / 2, sa)
        X = cheb_expm(F, X, h / 2, sb)
    return cheb_expm(F, X, hold, 1.0) if hold > 0 else X


def backward_holds(F, X, steps, holds):
    """yields (hold, U(t_r + hold)^dagger X) for the holds in ascending order (B5: the hold evolution chained segment by segment,
    exp(+i h_f h_j) X = exp(+i h_f (h_j - h_{j-1})) exp(+i h_f h_{j-1}) X, one accumulator per segment; the ramp backward per hold)."""
    Y = X.clone()
    t = 0.0
    for h in sorted(set(float(x) for x in holds)):
        if h > t:
            Y = cheb_expm(F, Y, -(h - t), 1.0)
            t = h
        Z = Y
        for sa, sb, dt in reversed(steps):
            Z = cheb_expm(F, Z, -dt / 2, sb)
            Z = cheb_expm(F, Z, -dt / 2, sa)
        yield h, Z
        del Z


def backward_multi(F, X, steps, holds):
    """{hold: U(t_r + hold)^dagger X} (B5)."""
    return dict(backward_holds(F, X, steps, holds))


def forward(F, X, steps, hold):
    return evolve(F, X, steps, hold=hold, backward=False)


# ------------------------------------------------------------------------------------------------ the uniform-coupling sea
class FlatSea:
    """the box's uniform-coupling ground state by the exact momentum-space projector applied by FFT (B1: the twist phase).  Default: the
    flat sea P(k) = (1 - H(k)/E(k))/2 (v = 0, N = 1), the initial state of the formation.  Generalized (J9's comparator): the states of
    N h_0(k) + sum_i v_i sin(2 k_i)/2 below mu (the bands -N E + d and N E + d, d = sum_i v_i sin(2 k_i)/2)."""

    def __init__(self, L, m, dev, dtype=torch.complex64, antiperiodic=True, N=1.0, v=(0.0, 0.0, 0.0), mu=0.0):
        self.L, self.dev, self.dtype = L, dev, dtype
        th = 0.5 if antiperiodic else 0.0
        k = 2 * np.pi * (np.arange(L) + th) / L
        kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
        s = np.stack([np.sin(kx), np.sin(ky), np.sin(kz)], -1)
        H = np.einsum("xyzi,iab->xyzab", s, GAMMA) + m * BETA
        E = np.sqrt(m * m + (s * s).sum(-1))
        if E.min() < 1e-14:
            raise ValueError("a node on the momentum grid: use the antiperiodic twist with an even L (B1)")
        self.Emin = float(E.min())
        d = 0.5 * (v[0] * np.sin(2 * kx) + v[1] * np.sin(2 * ky) + v[2] * np.sin(2 * kz))
        low = (-N * E + d) < mu
        up = (N * E + d) < mu
        Pm = 0.5 * (np.eye(4) - H / E[..., None, None])
        Pp = 0.5 * (np.eye(4) + H / E[..., None, None])
        P = low[..., None, None] * Pm + up[..., None, None] * Pp
        self.flat = bool(low.all() and not up.any())
        self.n_lower_empty, self.n_upper_filled = int((~low).sum()), int(up.sum())
        self.gap_to_mu = float(min(np.min(np.where(low, mu - (-N * E + d), np.inf)), np.min(np.where(~up, (N * E + d) - mu, np.inf))))
        self.P = torch.as_tensor(P, dtype=dtype, device=dev)
        self.antiperiodic = antiperiodic
        if antiperiodic:
            x = np.arange(L)
            ph = np.exp(-1j * np.pi * (x[:, None, None] + x[None, :, None] + x[None, None, :]) / L)
            self.ph = torch.as_tensor(ph[..., None, None], dtype=dtype, device=dev)
        if self.flat:
            w, V = np.linalg.eigh(H.reshape(-1, 4, 4))
            self.Vm = torch.as_tensor(V[:, :, :2].reshape(L, L, L, 4, 2), dtype=dtype, device=dev)
            self.Vp = torch.as_tensor(V[:, :, 2:].reshape(L, L, L, 4, 2), dtype=dtype, device=dev)

    def _fwd(self, psi):
        return torch.fft.fftn(psi * self.ph if self.antiperiodic else psi, dim=(0, 1, 2), norm="ortho")

    def _inv(self, f):
        g = torch.fft.ifftn(f, dim=(0, 1, 2), norm="ortho")
        return g * self.ph.conj() if self.antiperiodic else g

    def apply(self, psi):
        return self._inv(torch.matmul(self.P, self._fwd(psi)))

    def compress(self, psi):
        """B6: the occupied-band amplitudes (L^3 * 2, nb); their Gram matrix is (P psi)^dagger (P psi)."""
        z = torch.matmul(self.Vm.conj().transpose(-1, -2), self._fwd(psi))
        return z.reshape(-1, psi.shape[-1])

    def compress_both(self, psi):
        """B10: the occupied- and the empty-band amplitudes; their Gram matrices are C = (P psi)^dagger (P psi) and C-bar = (Q psi)^dagger (Q psi)."""
        f = self._fwd(psi)
        nb = psi.shape[-1]
        return (torch.matmul(self.Vm.conj().transpose(-1, -2), f).reshape(-1, nb), torch.matmul(self.Vp.conj().transpose(-1, -2), f).reshape(-1, nb))


# ------------------------------------------------------------------------------------------------ regions and blocks
def region_indices(L, pred):
    """the (x, y, z) sites of the region pred(R) with R measured from the box centre (B2: the cell centre); returns the sites and R."""
    g, R = box_coords(L)
    sites = np.indices((L, L, L)).reshape(3, -1).T
    Rf = R.reshape(-1)
    sel = pred(Rf)
    return sites[sel], Rf[sel]


def onehot(L, sites, m0, nb, dtype, dev):
    X = torch.zeros((L, L, L, 4, nb), dtype=dtype, device=dev)
    for jb in range(nb):
        mode = m0 + jb
        x, y, z = sites[mode // 4]
        X[x, y, z, mode % 4, jb] = 1.0
    return X


def gather_rows(Y, sites):
    """the rows of Y (L, L, L, 4, nb) at the region's sites, site-major with 4 components: (4 n_sites, nb)."""
    s = torch.as_tensor(np.asarray(sites), device=Y.device, dtype=torch.long)
    return Y[s[:, 0], s[:, 1], s[:, 2]].reshape(-1, Y.shape[-1])


def gram(Z, dev, slab=256, fp64=False):
    """C = Z^dagger Z in complex128 for Z (M x n) complex64 on the host or the device (B6): slab GEMMs in complex64 summed in complex128,
    or (fp64) every slab upcast and multiplied in complex128."""
    M, n = Z.shape
    C = torch.zeros((n, n), dtype=torch.complex128, device=dev)
    step = 4096 if fp64 else slab
    for r0 in range(0, M, step):
        S = Z[r0:r0 + step].to(dev, non_blocking=True)
        if fp64:
            S = S.to(torch.complex128)
            C.addmm_(S.conj().T, S)
        else:
            C.add_(S.conj().T @ S)
        del S
    return C


def region_Z(F, sea, steps, hold, sites, chunk=512):
    """Z~ (B6) of the region at one hold on the device, in the box's precision."""
    L = F.L
    nA = 4 * len(sites)
    Z = torch.empty((2 * L ** 3, nA), dtype=F.dtype, device=F.dev)
    for c0 in range(0, nA, chunk):
        nb = min(chunk, nA - c0)
        X = onehot(L, sites, c0, nb, F.dtype, F.dev)
        Y = backward_multi(F, X, steps, [hold])[float(hold)]
        Z[:, c0:c0 + nb] = sea.compress(Y)
        del X, Y
    return Z


def region_correlation(F, sea, steps, hold, sites, dust=None, chunk=512, slab=256, fp64=None):
    """C_A(t) = Z^dagger Z, Z = Pi (U^dagger Pi_A) with Pi = Pi_sea (dust: BUILDER, J4' adds the dust orbitals' projector the same way).
    Returns the n_A x n_A block in complex128 (modes ordered site-major, 4 spinor components per site); Z~ (B6) on the device in the box's
    precision, the Gram by the slab route for complex64 and in complex128 for a complex128 box."""
    if dust is not None:
        raise NotImplementedError("J4': the dust orbitals' projector")
    Z = region_Z(F, sea, steps, hold, sites, chunk)
    C = gram(Z, F.dev, slab=slab, fp64=(F.dtype == torch.complex128) if fp64 is None else fp64)
    del Z
    return C


def h2(lam):
    lam = np.clip(np.asarray(lam, float), 0.0, 1.0)
    q = lam[(lam > 1e-16) & (lam < 1 - 1e-16)]
    return float(np.sum(-q * np.log(q) - (1 - q) * np.log1p(-q)))


def spectrum(C):
    """eigenvalues of the Hermitian block in complex128 (C is symmetrized when it is not a Gram matrix)."""
    if C.dtype != torch.complex128:
        C = C.to(torch.complex128)
    return torch.linalg.eigvalsh(C).real.cpu().numpy()


def entropy_ph(C, Cbar):
    """B10: S = sum over the eigenvalues of C below 1/2 of h2 + sum over the eigenvalues of C-bar below 1/2 of h2, both PSD Gram matrices, so
    the small eigenvalues on either side are accurate to second order in the evolution's isometry defect; the plain S(C), the defect
    max|lambda_C + lambda_Cbar - 1| over the paired spectra and the counts are reported beside it."""
    lam = spectrum(C)
    mu = spectrum(Cbar)
    S_ph = h2(lam[lam < 0.5]) + h2(mu[mu < 0.5])
    pair = np.abs(np.sort(lam) + np.sort(mu)[::-1] - 1.0)
    return {"S": S_ph, "S_plain": h2(lam), "S_ph_minus_plain": S_ph - h2(lam), "pair_defect_max": float(pair.max()),
            "n_lam_below_half": int((lam < 0.5).sum()), "n_mu_below_half": int((mu < 0.5).sum()), "n": int(len(lam)),
            "lam_min": float(lam.min()), "lam_max": float(lam.max()), "mu_min": float(mu.min())}


def entropy(C, floors=(1e-10, 1e-8)):
    """S from the eigenvalues (complex128); S restricted to eigenvalues in (f, 1 - f) for the floors as the precision indicator."""
    lam = spectrum(C)
    out = {"S": h2(lam), "lam_min": float(lam.min()), "lam_max": float(lam.max()), "n": int(len(lam))}
    for f in floors:
        q = lam[(lam > f) & (lam < 1 - f)]
        out["S_floor_%g" % f] = h2(q)
    return out


# ------------------------------------------------------------------------------------------------ shells, two-point functions, partner
def site_positions(L, sites):
    c = (L - 1) / 2.0
    p = np.asarray(sites, float) - c
    return p, np.sqrt((p ** 2).sum(1))


def two_point_bins(Gxy, p_in, p_out, rbin=1.0, nang=18):
    """BUILDER: |C_xy|^2 summed over the 4 x 4 spinor block for x in A_in, y in A_out (Gxy of shape (4 n_in, 4 n_out)), binned by |x - y|
    (bins of width rbin) and by the angle between x's and y's radial directions (nang bins over [0, 180] degrees): pair counts, sums, means."""
    n_in, n_out = len(p_in), len(p_out)
    g2 = (Gxy.abs() ** 2).reshape(n_in, 4, n_out, 4).sum(dim=(1, 3)).cpu().numpy()
    d = np.sqrt(((p_in[:, None, :] - p_out[None, :, :]) ** 2).sum(-1))
    ri, ro = np.sqrt((p_in ** 2).sum(1)), np.sqrt((p_out ** 2).sum(1))
    cosang = np.clip((p_in @ p_out.T) / (ri[:, None] * ro[None, :]), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    out = {}
    for name, val, width, nmax in (("by_distance", d, rbin, int(np.ceil(d.max() / rbin)) + 1), ("by_angle_deg", ang, 180.0 / nang, nang)):
        b = np.minimum((val / width).astype(int), nmax - 1).reshape(-1)
        cnt = np.bincount(b, minlength=nmax)
        sm = np.bincount(b, weights=g2.reshape(-1), minlength=nmax)
        keep = cnt > 0
        out[name] = {"bin_lo": (np.arange(nmax)[keep] * width).tolist(), "width": width, "pairs": cnt[keep].tolist(), "sum": sm[keep].tolist(),
                     "mean": (sm[keep] / cnt[keep]).tolist()}
    out["total"] = float(g2.sum())
    return out


def blk(M, i, j, dev):
    """the sub-block M[i][:, j] moved to dev (M on the host or the device)."""
    i = torch.as_tensor(np.asarray(i), device=M.device)
    j = torch.as_tensor(np.asarray(j), device=M.device)
    return M.index_select(0, i).index_select(1, j).to(dev)


def shell_observables(C, Cflat, idx_in, idx_out, p_in, p_out, bins=True, dev=None, Cbar=None):
    """I(A_in : A_out) formed and flat, the cross block's singular values and ||G||_HS^2, the excess block's, the two-point bins (formed,
    flat, excess).  C, Cflat: the union region's blocks (complex128, any superset ordering, on the host or the device); idx_*: the mode
    indices of A_in, A_out in it; every sub-block is computed on dev."""
    dev = dev or C.device
    both = np.concatenate([idx_in, idx_out])
    rec = {}
    for name, M in (("formed", C), ("flat", Cflat)):
        if name == "formed" and Cbar is not None:
            ent = lambda i: entropy_ph(blk(C, i, i, dev), blk(Cbar, i, i, dev))
        else:
            ent = lambda i, M=M: entropy(blk(M, i, i, dev))
        Si = ent(idx_in)
        So = ent(idx_out)
        Sb = ent(both)
        G = blk(M, idx_in, idx_out, dev)
        sv = torch.linalg.svdvals(G).cpu().numpy()
        rec[name] = {"S_in": Si["S"], "S_out": So["S"], "S_both": Sb["S"], "I": Si["S"] + So["S"] - Sb["S"],
                     "I_plain" if "S_plain" in Si else "I_floor_1e-8": (Si["S_plain"] + So["S_plain"] - Sb["S_plain"]) if "S_plain" in Si else
                     (Si["S_floor_1e-08"] + So["S_floor_1e-08"] - Sb["S_floor_1e-08"]),
                     "G_sv_top": sv[:24].tolist(), "G_HS2": float(np.sum(sv ** 2)), "n_in": len(idx_in), "n_out": len(idx_out)}
        if bins:
            rec[name]["two_point"] = two_point_bins(G, p_in, p_out)
        del G
    dG = blk(C, idx_in, idx_out, dev) - blk(Cflat, idx_in, idx_out, dev)
    svd = torch.linalg.svdvals(dG).cpu().numpy()
    rec["excess"] = {"I": rec["formed"]["I"] - rec["flat"]["I"], "dG_sv_top": svd[:24].tolist(), "dG_HS2": float(np.sum(svd ** 2))}
    if bins:
        rec["excess"]["two_point"] = two_point_bins(dG, p_in, p_out)
    return rec


def shells_mi(F, sea, steps, hold, rs, d):
    """I(A_in : A_out), the cross block's singular values and the two-point functions binned by separation and by the radial-direction angle
    (the draft's interface on its own region; J4 uses shell_observables on the big ball's block, B7)."""
    si, Ri = region_indices(F.L, lambda R: (R >= rs - d) & (R < rs))
    so, Ro = region_indices(F.L, lambda R: (R >= rs) & (R < rs + d))
    sites = np.concatenate([si, so])
    C = region_correlation(F, sea, steps, hold, sites)
    Cf = flat_block(sea, F.L, sites, F.dev)
    ni = 4 * len(si)
    p, _ = site_positions(F.L, sites)
    out = shell_observables(C, Cf, np.arange(ni), np.arange(ni, 4 * len(sites)), p[:len(si)], p[len(si):])
    return out, C[:ni, ni:]


def flat_block(sea, L, sites, dev, chunk=512, out_device=None):
    """Pi_A P Pi_A of a uniform-coupling sea exactly by FFT (no evolution), complex128 (P is exact, so the block is Hermitian to rounding and
    eigvalsh reads one triangle; no symmetrization, which would triple the memory of a 13 GB block); out_device 'cpu' builds it on the host."""
    nA = 4 * len(sites)
    C = torch.empty((nA, nA), dtype=torch.complex128, device=out_device or dev)
    for c0 in range(0, nA, chunk):
        nb = min(chunk, nA - c0)
        X = onehot(L, sites, c0, nb, sea.dtype, dev)
        C[:, c0:c0 + nb] = gather_rows(sea.apply(X), sites).to(torch.complex128).to(C.device)
        del X
    return C


def region_weights(vec, L, edges):
    """the density |vec(x)|^2 summed over the shells edges[i] <= R < edges[i + 1]."""
    _, R = box_coords(L)
    rho = (vec.abs() ** 2).sum(dim=tuple(range(3, vec.dim()))).cpu().numpy()
    return [float(rho[(R >= lo) & (R < hi)].sum()) for lo, hi in zip(edges[:-1], edges[1:])]


def partner_3d(F, sea, steps, hold, w, edges=None):
    """w_p = (1 - |w><w|) C w with C w = U Pi U^dagger w: 2 vector evolutions; n = <w|C|w>, ||w_p||^2 against n(1 - n), and (BUILDER) the
    weights of w_p in the shells of `edges` (default: R < R_inner, [R_inner, r_s), [r_s, oo) set by the caller)."""
    y = backward_multi(F, w.clone(), steps, [hold])[float(hold)]
    y = sea.apply(y)
    Cw = forward(F, y, steps, hold)
    n = float(torch.vdot(w.reshape(-1), Cw.reshape(-1)).real)
    wp = Cw - n * w
    nrm2 = float(torch.vdot(wp.reshape(-1), wp.reshape(-1)).real)
    rec = {"n": n, "norm2": nrm2, "n_1_minus_n": n * (1 - n), "identity_error": nrm2 - n * (1 - n)}
    if edges is not None:
        wts = region_weights(wp, F.L, edges)
        rec["weights"] = {"edges": list(edges), "w": wts, "fraction": [x / nrm2 if nrm2 > 0 else None for x in wts]}
    return rec, wp


# ------------------------------------------------------------------------------------------------ the small ball (J9's method, B8)
def step_coeffs(x0, K, kind="erf", ws=None, nq=None):
    """Chebyshev coefficients of the occupation Theta(x0 - x) on [-1, 1] (x = h/r): kind 'jackson' (the exact step's coefficients with the
    Jackson kernel g_k of K + 1 moments) or 'erf' (the smoothed step (1 - erf((x - x0)/ws))/2 by Chebyshev-Gauss quadrature)."""
    k = np.arange(K + 1)
    if kind == "jackson":
        th0 = math.acos(x0)
        a = np.empty(K + 1)
        a[0] = (math.pi - th0) / math.pi
        a[1:] = -2.0 * np.sin(k[1:] * th0) / (math.pi * k[1:])
        q = math.pi / (K + 2)
        g = ((K + 2 - k) * np.cos(q * k) + np.sin(q * k) / math.tan(q)) / (K + 2)
        return a * g
    nq = nq or int(2 ** math.ceil(math.log2(max(8 * K, 64.0 / ws))))
    th = math.pi * (np.arange(nq) + 0.5) / nq
    f = 0.5 * (1.0 - erf((np.cos(th) - x0) / ws))
    a = np.array([(2.0 if kk else 1.0) / nq * np.dot(f, np.cos(kk * th)) for kk in k])
    return a


def erf_terms(ws, tol=1e-12):
    """the term count of the erf step from its coefficients' decay: K with |a_k| < tol for every k beyond it (B8)."""
    K = int(2.0 * math.sqrt(math.log(1.0 / tol)) / ws * 1.3) + 16
    a = step_coeffs(0.0, K, "erf", ws)
    big = np.nonzero(np.abs(a) > tol)[0]
    return int(big[-1]) + 1 if len(big) else 1


def symbol_gap(m, vmax, mu=0.0, N=1.0, nk=96):
    """BUILDER: the uniform symbol's distance from mu to the bands, minimized over the momentum grid and over flow directions (1,0,0),
    (1,1,0), (1,1,1) and 6 random ones at |v| = vmax: the term count's gap."""
    k = 2 * np.pi * (np.arange(nk) + 0.5) / nk
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    E = np.sqrt(m * m + np.sin(kx) ** 2 + np.sin(ky) ** 2 + np.sin(kz) ** 2)
    rng = np.random.default_rng(5)
    dirs = [np.array(d, float) for d in ((1, 0, 0), (1, 1, 0), (1, 1, 1))] + [rng.normal(size=3) for _ in range(6)]
    g = np.inf
    for dv in dirs:
        dv = vmax * dv / np.linalg.norm(dv)
        d = 0.5 * (dv[0] * np.sin(2 * kx) + dv[1] * np.sin(2 * ky) + dv[2] * np.sin(2 * kz))
        g = min(g, float(np.min(N * E + d - mu)), float(np.min(mu - (-N * E + d))))
    return g


def small_ball_block(F, mu, sites, gap, kind="erf", nterms=None, tol=1e-12, chunk=512, ws_factor=4.75):
    """J9's method: C_ball = Pi_ball Theta(mu - h) Pi_ball for the static operator h (gap from the caller), by the Chebyshev expansion of
    the step on the ball's basis (B8: kind 'erf' with w = gap/ws_factor and the term count from the coefficients' decay, or 'jackson' with
    nterms).  Returns C_ball (complex128, symmetrized), the projector defect max_j |<Y_j|Y_j> - C_jj| and the term count."""
    r = F.bound_s(1.0)
    x0 = mu / r
    if kind == "erf":
        ws = gap / ws_factor / r
        K = nterms or erf_terms(ws, tol)
        a = step_coeffs(x0, K, "erf", ws)
    else:
        K = nterms
        a = step_coeffs(x0, K, "jackson")
    nA = 4 * len(sites)
    C = torch.empty((nA, nA), dtype=torch.complex128, device=F.dev)
    defect = 0.0
    for c0 in range(0, nA, chunk):
        nb = min(chunk, nA - c0)
        X = onehot(F.L, sites, c0, nb, F.dtype, F.dev)
        T0_ = X
        T1_ = apply_h(T0_, F, 1.0, alpha=1.0 / r)
        Y = float(a[0]) * T0_ + float(a[1]) * T1_
        for k in range(2, K + 1):
            apply_h(T1_, F, 1.0, alpha=2.0 / r, beta=-1.0, prev=T0_, out=T0_)
            T0_, T1_ = T1_, T0_
            Y.add_(T1_, alpha=float(a[k]))
        del T0_, T1_, X
        C[:, c0:c0 + nb] = gather_rows(Y, sites).to(torch.complex128)
        nrm = (Y.abs() ** 2).sum(dim=(0, 1, 2, 3)).to(torch.float64)
        dg = torch.diagonal(C[c0:c0 + nb, c0:c0 + nb]).real
        defect = max(defect, float((nrm - dg).abs().max()))
        del Y
    return 0.5 * (C + C.conj().T), {"terms": int(K), "projector_defect": defect, "kind": kind, "r": r, "x0": x0}


# ------------------------------------------------------------------------------------------------ the 3-dimensional partial-wave packet (B9)
def onsite_spinor(k, m, a):
    """the positive-energy spinor of m SZ + a SY + sin k SX (as flow_chart_v4.onsite_spinor)."""
    H = np.array([[m, np.sin(k) - 1j * a], [np.sin(k) + 1j * a, -m]], complex)
    return np.linalg.eigh(H)[1][:, -1]


def outgoing_k(omega, m, a, v):
    """the carrier of Killing frequency omega on the local outgoing branch of the chain's symbol sqrt(m^2 + a^2 + sin^2 k) + v sin(2k)/2
    (group velocity > 0), the smallest such k in (0, pi/2)."""
    from scipy.optimize import brentq
    om = lambda k: np.sqrt(m * m + a * a + np.sin(k) ** 2) + v * np.sin(2 * k) / 2 - omega
    vg = lambda k: np.sin(k) * np.cos(k) / np.sqrt(m * m + a * a + np.sin(k) ** 2) + v * np.cos(2 * k)
    ks = np.linspace(1e-6, np.pi / 2 - 1e-6, 20001)
    f = om(ks)
    for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        k0 = brentq(om, ks[i], ks[i + 1], xtol=1e-14)
        if vg(k0) > 0:
            return float(k0), float(vg(k0))
    raise ValueError("no outgoing carrier at omega %g (m %g, a %g, v %g)" % (omega, m, a, v))


def escaping_packet_3d(F, kappa_j, mj, omega, R0, sigma_x, vfun, dev=None, kappa_std_sign=-1):
    """a lattice Y_lm times radial outgoing packet (B9, j = 1/2): the lattice harmonic by the sites' direction cosines, the radial Gaussian
    (density width sigma_x) with the local outgoing momentum of Killing frequency omega and the local positive spinor; normalized.
    kappa_std_sign = -1 is the derived map kappa = -kappa_j (+1 is the wrong map, kept for the check)."""
    if abs(abs(kappa_j) - 1.0) > 1e-12:
        raise NotImplementedError("j = 1/2 only")
    m = F.m
    v0 = float(vfun(np.array([R0]))[0])
    a0 = kappa_j / R0
    k0, vg0 = outgoing_k(omega, m, a0, v0)
    u = onsite_spinor(k0, m, a0)
    g, R = box_coords(F.L)
    nvec = g / R
    env = np.exp(-(R - R0) ** 2 / (4 * sigma_x ** 2) + 1j * k0 * R) / R
    chi = np.array([1.0, 0.0]) if mj > 0 else np.array([0.0, 1.0])
    sdotn = np.einsum("iab,ixyz->xyzab", SIG, nvec)
    Om_m1 = np.broadcast_to(chi / math.sqrt(4 * math.pi), (F.L, F.L, F.L, 2))
    Om_p1 = -np.einsum("xyzab,b->xyza", sdotn, chi) / math.sqrt(4 * math.pi)
    kap = kappa_std_sign * kappa_j
    up_ang, lo_ang = (Om_m1, Om_p1) if kap < 0 else (Om_p1, Om_m1)
    psi = np.concatenate([(env * u[0])[..., None] * up_ang, (-env * u[1])[..., None] * lo_ang], axis=-1)
    psi = psi / np.linalg.norm(psi)
    w = torch.as_tensor(psi[..., None], dtype=F.dtype, device=dev or F.dev).contiguous()
    return w, {"kappa_j": kappa_j, "kappa_std": kap, "m_j": mj, "omega": omega, "R0": R0, "sigma_x": sigma_x, "k0": k0, "vg0": vg0,
               "v_R0": v0, "a_R0": a0, "spinor": [[float(x.real), float(x.imag)] for x in u]}


def packet_energy(F, w, s=1.0):
    hw = apply_h(w, F, s)
    e = float(torch.vdot(w.reshape(-1), hw.reshape(-1)).real)
    e2 = float(torch.vdot(hw.reshape(-1), hw.reshape(-1)).real)
    return e, math.sqrt(max(e2 - e * e, 0.0))


# ------------------------------------------------------------------------------------------------ writer
def jdump(obj, path):
    def conv(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def fmt(x, p=6):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return ("%." + str(p) + "g") % x
    return str(x)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ box builders
def build_pg_box(L, rs, Rm, Rcut, m, dev, dtype, trunc=True, flow=True):
    vfun = (lambda R: pg_v(R, rs, Rm) * truncation(R, Rcut)) if trunc else (lambda R: pg_v(R, rs, Rm))
    if not flow:
        vfun = lambda R: 0.0 * np.asarray(R, float)
    N, vv, R = radial_field(L, vfun)
    F = Fields(L, N, vv, m, dev, dtype, antiperiodic=True)
    face = max(float(np.max(np.abs(v[ix]))) for v in vv for ix in ((0,), (L - 1,), (slice(None), 0), (slice(None), L - 1),
                                                                         (slice(None), slice(None), 0), (slice(None), slice(None), L - 1)))
    return F, vfun, {"L": L, "rs": rs, "Rm": Rm, "Rcut": Rcut if trunc else None, "m": m, "max_abs_v_on_faces": face, "vmax": F.vmax,
                     "bound_B4": F.bound, "bound_gershgorin": F.bound_gershgorin}


def ball_sites(L, x0, rb):
    """the sites with |x - x0| <= rb ordered by distance from x0 (sub-balls are prefixes)."""
    g = np.indices((L, L, L)).reshape(3, -1).T
    d = np.sqrt(((g - np.asarray(x0)) ** 2).sum(1))
    sel = np.nonzero(d <= rb + 1e-9)[0]
    order = np.argsort(d[sel], kind="stable")
    return g[sel][order], d[sel][order]


def dense_h(F, s=1.0, chunk=1024):
    n = 4 * F.L ** 3
    H = torch.empty((n, n), dtype=torch.complex128, device=F.dev)
    for c0 in range(0, n, chunk):
        nb = min(chunk, n - c0)
        X = torch.zeros((n, nb), dtype=F.dtype, device=F.dev)
        X[c0 + torch.arange(nb, device=F.dev), torch.arange(nb, device=F.dev)] = 1.0
        H[:, c0:c0 + nb] = apply_h(X.reshape(F.L, F.L, F.L, 4, nb).contiguous(), F, s).reshape(n, nb).to(torch.complex128)
    return H


def rel(a, b):
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-300))


# ------------------------------------------------------------------------------------------------ mode: smoke (J4's smoke harness)
def mode_smoke(a):
    dev = torch.device("cuda:0")
    rs, Rm, Rcut = a.rs, a.Rm, a.rcut_over_rs * a.rs
    kappa = 1.0 / (2 * rs)
    out = {"label": LABEL_PG, "builder_decisions": ["B1", "B2", "B3", "B4", "B5", "B6"], "params": {"rs": rs, "Rm": Rm, "Rcut": Rcut, "m": a.mass}}
    path = os.path.join(a.outdir, "smoke_%s.json" % a.tag)
    rng = torch.Generator(device="cpu").manual_seed(7)
    rnd = lambda shape, dt: (torch.randn(shape, generator=rng, dtype=torch.float64) + 1j * torch.randn(shape, generator=rng, dtype=torch.float64)).to(dt).to(dev)
    # (a) the kernel against the draft's apply_h; Hermiticity
    ka = {}
    for dtname, dt in (("complex128", torch.complex128), ("complex64", torch.complex64)):
        for L in (8, a.L):
            F, _, info = build_pg_box(L, rs, Rm, Rcut, a.mass, dev, dt)
            X = rnd((L, L, L, 4, 3), dt)
            Y = rnd((L, L, L, 4, 3), dt)
            for s in (0.0, 1.0, 1.0774):
                ka["%s L%d s%g kernel_vs_ref" % (dtname, L, s)] = rel(apply_h(X, F, s), apply_h_ref(X, F, s))
            hy, hx = apply_h(Y, F, 1.0774), apply_h(X, F, 1.0774)
            lhs = torch.vdot(X.reshape(-1), hy.reshape(-1)); rhs = torch.vdot(hx.reshape(-1), Y.reshape(-1))
            ka["%s L%d hermiticity" % (dtname, L)] = float(abs(lhs - rhs) / (X.norm() * hy.norm()))
            if L == a.L and dtname == "complex64":
                out["box_info"] = info
    out["kernel"] = ka
    log("kernel checks: %s" % {k: "%.1e" % v for k, v in ka.items()})
    jdump(out, path)
    # (b) Chebyshev against dense expm at L = 8 (complex128), both bounds; the bound against the spectral radius
    F8, _, _ = build_pg_box(8, rs, Rm, Rcut, a.mass, dev, torch.complex128)
    cb = {}
    for s in (1.0, 1.0774, 0.0):
        Hs = dense_h(F8, s)
        cb["herm_dense_s%g" % s] = float((Hs - Hs.conj().T).abs().max())
        ev = torch.linalg.eigvalsh(Hs)
        cb["spectral_radius_s%g" % s] = float(ev.abs().max())
        cb["bound_B4_s%g" % s] = F8.bound_s(s)
        cb["bound_gershgorin_s%g" % s] = F8.bound_gershgorin * max(1.0, abs(s))
        del Hs
    H = dense_h(F8, 1.0)
    ev, V = torch.linalg.eigh(H)
    X = rnd((8, 8, 8, 4, 4), torch.complex128)
    Xf = X.reshape(-1, 4)
    for gersh in (False, True):
        F8.use_gershgorin = gersh
        tag = "gershgorin" if gersh else "B4"
        for tau in (0.5, 50.0, 400.0):
            ex = (V @ (torch.exp(-1j * tau * ev)[:, None] * (V.conj().T @ Xf))).reshape(X.shape)
            cb["%s tau%g" % (tag, tau)] = rel(cheb_expm(F8, X, tau, 1.0), ex)
        ys = cheb_expm_multi(F8, X, [-50.0, -400.0], 1.0)
        for tau, y in zip((-50.0, -400.0), ys):
            ex = (V @ (torch.exp(-1j * tau * ev)[:, None] * (V.conj().T @ Xf))).reshape(X.shape)
            cb["%s multi tau%g" % (tag, tau)] = rel(y, ex)
    F8.use_gershgorin = False
    Hs = dense_h(F8, 1.0774)
    ev2, V2 = torch.linalg.eigh(Hs)
    ex = (V2 @ (torch.exp(-1j * 0.5 * ev2)[:, None] * (V2.conj().T @ Xf))).reshape(X.shape)
    cb["B4 s1.0774 tau0.5"] = rel(cheb_expm(F8, X, 0.5, 1.0774), ex)
    del H, V, Hs, V2
    # the flat sea against the dense flat operator (the twist's consistency between the bonds and the FFT phase)
    F0, _, _ = build_pg_box(8, rs, Rm, Rcut, a.mass, dev, torch.complex128, flow=False)
    H0 = dense_h(F0, 0.0)
    e0, V0 = torch.linalg.eigh(H0)
    Vo = V0[:, e0 < 0]
    sea8 = FlatSea(8, a.mass, dev, torch.complex128)
    I = torch.eye(4 * 512, dtype=torch.complex128, device=dev).reshape(8, 8, 8, 4, 2048)
    Pfft = sea8.apply(I).reshape(2048, 2048)
    cb["flat_sea_fft_vs_dense"] = rel(Pfft, Vo @ Vo.conj().T)
    cb["flat_sea_idempotency"] = rel(Pfft @ Pfft, Pfft)
    cb["flat_sea_min_E_on_grid"] = sea8.Emin
    cb["dense_flat_min_abs_eigenvalue"] = float(e0.abs().min())
    Zc = sea8.compress(I)
    cb["compressed_gram_vs_P"] = rel(Zc.conj().T @ Zc, Pfft)
    del H0, V0, I, Pfft, Zc
    out["chebyshev_and_sea_L8"] = cb
    log("chebyshev/sea: %s" % {k: "%.2e" % v for k, v in cb.items()})
    jdump(out, path)
    # (c) the whole box at L = 16: C^2 = C, S(A) = S(complement), the region block against the full C's sub-block
    fb = {}
    holds = [0.0, 12.0 / kappa]
    for dtname, dt in (("complex64", torch.complex64), ("complex128", torch.complex128)):
        F16, _, info16 = build_pg_box(a.full_L, rs, Rm, Rcut, a.mass, dev, dt)
        sea16 = FlatSea(a.full_L, a.mass, dev, dt)
        steps = history_ramp(2 * rs, 1.0, ramp_envelope(2 * rs))
        allsites = np.indices((a.full_L,) * 3).reshape(3, -1).T
        n = 4 * len(allsites)
        Z = {h: torch.empty((2 * a.full_L ** 3, n), dtype=dt, device=dev) for h in holds}
        t0 = time.time()
        for c0 in range(0, n, 512):
            nb = min(512, n - c0)
            X = onehot(a.full_L, allsites, c0, nb, dt, dev)
            Ys = backward_multi(F16, X, steps, holds)
            for h in holds:
                Z[h][:, c0:c0 + nb] = sea16.compress(Ys[h]).to(Z[h].dtype)
            del X, Ys
        fb["%s evolution_s" % dtname] = time.time() - t0
        _, Rall = box_coords(a.full_L)
        Rall = Rall.reshape(-1)
        Aidx = np.nonzero(Rall < a.full_RA)[0]
        Amodes = (4 * Aidx[:, None] + np.arange(4)).reshape(-1)
        Bmodes = np.setdiff1d(np.arange(n), Amodes)
        for h in holds:
            C = gram(Z[h], dev, slab=256, fp64=(dt == torch.complex128))
            lam = spectrum(C)
            key = "%s hold%g/kappa" % (dtname, h * kappa)
            fb[key + " idempotency max lam(1-lam)"] = float(np.max(lam * (1 - lam)))
            fb[key + " trace minus 2 L^3"] = float(lam.sum() - 2 * a.full_L ** 3)
            am = torch.as_tensor(Amodes, device=dev); bm = torch.as_tensor(Bmodes, device=dev)
            SA = entropy(C[am][:, am]); SB = entropy(C[bm][:, bm])
            fb[key + " S(A) ball"] = SA["S"]; fb[key + " S(complement)"] = SB["S"]; fb[key + " S(A)-S(complement)"] = SA["S"] - SB["S"]
            if h == holds[-1]:
                CA = region_correlation(F16, sea16, steps, h, allsites[Aidx])
                fb[key + " region block vs full sub-block"] = rel(CA, C[am][:, am])
            del C
        del Z
    out["full_box"] = fb
    out["full_box_info"] = dict(info16, A_radius=a.full_RA, modes=4 * a.full_L ** 3)
    log("full box: %s" % {k: "%.3e" % v for k, v in fb.items()})
    jdump(out, path)
    # (d) L = a.L: the ball R < r_s (production: complex64, dt = 1, slab Gram), holds 0 and 12/kappa, the flat reference; the dt ladder, the
    # precision and the Gram route on the ball R < 5
    rc = {}
    F, _, info = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, torch.complex64)
    sea = FlatSea(a.L, a.mass, dev, torch.complex64)
    sea128 = FlatSea(a.L, a.mass, dev, torch.complex128)
    steps1 = history_ramp(2 * rs, 1.0, ramp_envelope(2 * rs))
    sites, Rs_ = region_indices(a.L, lambda R: R < rs)
    rc["ball_rs_modes"] = 4 * len(sites)
    Cf = flat_block(sea128, a.L, sites, dev)
    rc["S_flat_ball_rs"] = entropy(Cf)
    for h in holds:
        t0 = time.time()
        Z = region_Z(F, sea, steps1, h, sites)
        rc["evolution_wall_s hold%g/kappa" % (h * kappa)] = time.time() - t0
        t0 = time.time()
        C = gram(Z, dev, slab=256)
        rc["gram_slab256_wall_s hold%g/kappa" % (h * kappa)] = time.time() - t0
        rc["S_formed_ball_rs hold%g/kappa" % (h * kappa)] = entropy(C)
        rc["excess hold%g/kappa" % (h * kappa)] = rc["S_formed_ball_rs hold%g/kappa" % (h * kappa)]["S"] - rc["S_flat_ball_rs"]["S"]
        if h == holds[-1]:
            t0 = time.time()
            C_fp64 = gram(Z, dev, fp64=True)
            rc["gram_fp64_wall_s"] = time.time() - t0
            C_s1024 = gram(Z, dev, slab=1024)
            rc["gram slab256 vs fp64 maxabs"] = float((C - C_fp64).abs().max())
            rc["gram slab1024 vs fp64 maxabs"] = float((C_s1024 - C_fp64).abs().max())
            e64 = entropy(C_fp64)
            rc["S fp64 gram"] = e64
            rc["S slab256 minus fp64"] = rc["S_formed_ball_rs hold%g/kappa" % (h * kappa)]["S"] - e64["S"]
            rc["S slab1024 minus fp64"] = entropy(C_s1024)["S"] - e64["S"]
            del C_fp64, C_s1024
        del C, Z
        jdump(dict(out, region=rc), path)
    small, _ = region_indices(a.L, lambda R: R < 5.0)
    rc["small_ball_R<5_modes"] = 4 * len(small)
    h = holds[-1]
    C1 = region_correlation(F, sea, steps1, h, small)
    C05 = region_correlation(F, sea, history_ramp(2 * rs, 0.5, ramp_envelope(2 * rs)), h, small)
    F128, _, _ = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, torch.complex128)
    C128 = region_correlation(F128, sea128, steps1, h, small)
    S1, S05, S128 = entropy(C1)["S"], entropy(C05)["S"], entropy(C128)["S"]
    rc["dt_ladder R<5 hold12/kappa"] = {"S_dt1": S1, "S_dt0.5": S05, "dS": S1 - S05, "max_abs_dC": float((C1 - C05).abs().max())}
    rc["precision R<5 hold12/kappa"] = {"S_complex64": S1, "S_complex128": S128, "dS": S1 - S128, "max_abs_dC": float((C1 - C128).abs().max())}
    out["region"] = rc
    out["region_info"] = info
    log("region L%d: %s" % (a.L, json.dumps(rc, default=float)[:1500]))
    out["wall_s"] = time.time() - T0
    out["finished_utc"] = utc()
    jdump(out, path)
    write_smoke_md(out, path.replace(".json", ".md"))


def write_smoke_md(o, path):
    L = ["# Smoke harness of obs_box3d_v1.py (J4's smoke; %s)" % o["label"], "",
         "Script obs_box3d_v1.py, mode smoke, finished %s, wall %.0f s; r_s %g, R_m %g, R_cut %g, m %g." % (
             o.get("finished_utc"), o.get("wall_s", 0), o["params"]["rs"], o["params"]["Rm"], o["params"]["Rcut"], o["params"]["m"]), ""]
    L.append("Box at L = %d: max|v| on the faces %s (the truncation's residual at the wrap), max|v| %s, Chebyshev bound B4 %s, Gershgorin %s." % (
        o["box_info"]["L"], fmt(o["box_info"]["max_abs_v_on_faces"]), fmt(o["box_info"]["vmax"]), fmt(o["box_info"]["bound_B4"]),
        fmt(o["box_info"]["bound_gershgorin"])))
    L += ["", "## Kernel against the draft's apply_h, Hermiticity (relative)", "", md_table(["check", "value"], sorted(o["kernel"].items()))]
    L += ["", "## Chebyshev against dense expm, the flat sea against the dense flat operator (L = 8, complex128)", "",
          md_table(["check", "value"], sorted(o["chebyshev_and_sea_L8"].items()))]
    L += ["", "## The whole box at L = %d (%d modes; holds 0 and 12/kappa; A = ball R < %g)" % (o["full_box_info"]["L"], o["full_box_info"]["modes"],
          o["full_box_info"]["A_radius"]), "", md_table(["check", "value"], sorted(o["full_box"].items()))]
    rc = o.get("region", {})
    rows = []
    for k, v in sorted(rc.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                rows.append((k + " : " + kk, vv))
        else:
            rows.append((k, v))
    L += ["", "## Region checks at L = %d" % o["box_info"]["L"], "", md_table(["quantity", "value"], rows)]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ mode: smallball (J5 part 1, J9's dry run)
def grad_invariants(vfun, pts, h=1e-4):
    """I1 = sum_ij (d_i v_j)^2 and I2 = (d_i v_i)^2 of v^j = v(R) x^j/R at the points (relative to the centre), by central differences of the
    analytic profile."""
    def vvec(p):
        R = np.sqrt((p ** 2).sum(-1))
        return vfun(R)[..., None] * p / R[..., None]
    J = np.zeros(pts.shape[:-1] + (3, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = h
        J[..., i, :] = (vvec(pts + e) - vvec(pts - e)) / (2 * h)
    return (J ** 2).sum((-1, -2)), np.trace(J, axis1=-2, axis2=-1) ** 2


def mode_smallball(a):
    dev = torch.device("cuda:0")
    out = {"label": "prescribed gapped profile (J9's dry run): v = -0.6 sqrt(r_s/R) for R >= 2 r_s, the C^1 cubic inside (BUILDER), N = 1, "
                    "mu = 0 in the gap; truncated by (1 - tanh((R - R_cut)/4))/2 with R_cut = L/2 - 6 (BUILDER) on the antiperiodic box",
           "builder_decisions": ["B1", "B2", "B3", "B4", "B8", "inner cubic", "R_cut = L/2 - 6", "r_s = %g" % a.rs]}
    path = os.path.join(a.outdir, "smallball_%s.json" % a.tag)
    m, mu = a.mass, 0.0
    # (1) the dense control at L = a.dense_L
    if a.dense_L > 0:
        Ld = a.dense_L
        rsd = a.dense_rs
        vfun_d = lambda R: gapped_v(R, rsd) * truncation(R, Ld / 2 - 2, 2.0)
        N, vv, _ = radial_field(Ld, vfun_d)
        Fd = Fields(Ld, N, vv, m, dev, torch.complex128)
        H = dense_h(Fd, 1.0)
        ev, V = torch.linalg.eigh(H)
        del H
        gap_dense = float(min((ev[ev > mu] - mu).min(), (mu - ev[ev < mu]).min()))
        gap_sym = symbol_gap(m, Fd.vmax, mu)
        x0 = np.full(3, Ld // 2)
        sites, dist = ball_sites(Ld, x0, 3)
        idx = torch.as_tensor(np.ravel_multi_index(tuple(sites.T), (Ld, Ld, Ld)), device=dev)
        modes = (4 * idx[:, None] + torch.arange(4, device=dev)).reshape(-1)
        Vo = V[:, ev < mu][modes]
        Cex = Vo @ Vo.conj().T
        del V, Vo
        Sex = entropy(Cex)["S"]
        dc = {"L": Ld, "rs": rsd, "gap_dense": gap_dense, "gap_symbol_estimate": gap_sym, "ball_modes": int(len(modes)), "S_exact": Sex, "rows": []}
        Ce, info = small_ball_block(Fd, mu, sites, min(gap_sym, gap_dense), "erf")
        dc["rows"].append({"kind": "erf", "terms": info["terms"], "max_abs_dC": float((Ce - Cex).abs().max()), "dS": entropy(Ce)["S"] - Sex,
                           "projector_defect": info["projector_defect"]})
        for fac in (1, 4, 16):
            Cj, ij = small_ball_block(Fd, mu, sites, gap_sym, "jackson", nterms=fac * info["terms"])
            dc["rows"].append({"kind": "jackson", "terms": ij["terms"], "max_abs_dC": float((Cj - Cex).abs().max()), "dS": entropy(Cj)["S"] - Sex,
                               "projector_defect": ij["projector_defect"]})
        out["dense_control"] = dc
        log("dense control: %s" % json.dumps(dc, default=float))
        jdump(out, path)
    # (2) the box
    L, rs = a.L, a.rs
    Rcut = L / 2 - 6
    vfun = lambda R: gapped_v(R, rs) * truncation(R, Rcut)
    N, vv, Rbox = radial_field(L, vfun)
    F = Fields(L, N, vv, m, dev, torch.complex128)
    gap = symbol_gap(m, F.vmax, mu)
    out["box"] = {"L": L, "rs": rs, "Rm": 2 * rs, "Rcut": Rcut, "m": m, "mu": mu, "vmax": F.vmax, "gap_symbol": gap, "bound": F.bound}
    c = (L - 1) / 2.0
    rows = []
    for Rt in (0.0, rs, 2 * rs, 3 * rs):
        x0 = np.array([int(round(c + Rt)) if Rt > 0 else L // 2, L // 2, L // 2])
        p0 = x0 - c
        sites, dist = ball_sites(L, x0, 5)
        t0 = time.time()
        C, info = small_ball_block(F, mu, sites, gap, "erf")
        wall = time.time() - t0
        vloc = vfun(np.array([np.linalg.norm(p0)]))[0] * p0 / np.linalg.norm(p0)
        seau = FlatSea(L, m, dev, torch.complex128, N=1.0, v=tuple(vloc), mu=mu)
        Cu = flat_block(seau, L, sites, dev)
        pts = sites - c
        I1, I2 = grad_invariants(vfun, pts)
        I1c, I2c = grad_invariants(vfun, p0[None, :].astype(float))
        row = {"R_centre": float(np.linalg.norm(p0)), "x0": x0.tolist(), "v_loc": vloc.tolist(), "|v_loc|": float(np.linalg.norm(vloc)),
               "I1_centre": float(I1c[0]), "I2_centre": float(I2c[0]), "terms": info["terms"], "projector_defect": info["projector_defect"],
               "uniform_state_is_flat_sea": seau.flat, "uniform_gap_to_mu": seau.gap_to_mu, "wall_s": wall, "balls": {}}
        for rb in (3, 4, 5):
            k = int(np.sum(dist <= rb + 1e-9))
            md = torch.arange(4 * k, device=dev)
            S = entropy(C[md][:, md]); Su = entropy(Cu[md][:, md])
            I1b = float(I1[:k].mean())
            row["balls"][str(rb)] = {"sites": k, "S_profile": S["S"], "S_uniform": Su["S"], "dS": S["S"] - Su["S"], "I1_ball_mean": I1b,
                                     "I2_ball_mean": float(I2[:k].mean()), "dS_over_I1": (S["S"] - Su["S"]) / I1b if I1b > 0 else None,
                                     "dS_over_I1_rb2": (S["S"] - Su["S"]) / (I1b * rb * rb) if I1b > 0 else None,
                                     "dS_over_I1_rb3": (S["S"] - Su["S"]) / (I1b * rb ** 3) if I1b > 0 else None,
                                     "lam_min": S["lam_min"], "lam_max": S["lam_max"]}
        rows.append(row)
        log("ball at R %.2f: %s" % (row["R_centre"], json.dumps(row["balls"], default=float)))
        out["positions"] = rows
        jdump(out, path)
        del C, Cu
    out["wall_s"] = time.time() - T0
    out["finished_utc"] = utc()
    jdump(out, path)
    write_smallball_md(out, path.replace(".json", ".md"))


def write_smallball_md(o, path):
    L = ["# J5 part 1: the small-ball dry run (J9's method) on the prescribed gapped profile", "", "Label: " + o["label"] + ".", "",
         "Script obs_box3d_v1.py, mode smallball, finished %s, wall %.0f s." % (o.get("finished_utc"), o.get("wall_s", 0)), ""]
    if "dense_control" in o:
        d = o["dense_control"]
        L += ["## Dense control (L = %d, r_s = %g, ball of radius 3 at the centre, %d modes)" % (d["L"], d["rs"], d["ball_modes"]), "",
              "Gap from the dense spectrum %s; the symbol's estimate %s; S_exact %s." % (fmt(d["gap_dense"]), fmt(d["gap_symbol_estimate"]), fmt(d["S_exact"])),
              "", md_table(["kernel", "terms", "max|dC|", "dS", "projector defect"],
                           [(r["kind"], r["terms"], r["max_abs_dC"], r["dS"], r["projector_defect"]) for r in d["rows"]]), ""]
    b = o["box"]
    L += ["## The box (L = %d, r_s = %g, R_m = %g, R_cut = %g, m = %g, mu = %g; max|v| %s; symbol gap %s; bound %s)" % (
        b["L"], b["rs"], b["Rm"], b["Rcut"], b["m"], b["mu"], fmt(b["vmax"]), fmt(b["gap_symbol"]), fmt(b["bound"])), ""]
    rows = []
    for p in o.get("positions", []):
        for rb, q in p["balls"].items():
            rows.append((p["R_centre"], p["|v_loc|"], rb, q["sites"], q["S_profile"], q["S_uniform"], q["dS"], q["I1_ball_mean"], q["dS_over_I1"],
                         q["dS_over_I1_rb2"], p["I1_centre"], p["terms"], p["projector_defect"], p["uniform_state_is_flat_sea"]))
    L += [md_table(["R centre", "|v_loc|", "r_ball", "sites", "S(profile)", "S(uniform)", "dS", "<I1>_ball", "dS/<I1>", "dS/(<I1> r^2)",
                    "I1(centre)", "terms", "proj. defect", "uniform = flat sea"], rows), "",
          "I1 = sum_ij (d_i v_j)^2 in lattice units (a = 1), averaged over the ball's sites; dS = S_ball(profile) - S_ball(uniform state at the "
          "centre's couplings)."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ mode: c3d (J5 part 2, the control C-3D)
C3D_HOLDS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]


def mode_c3d(a):
    dev = torch.device("cuda:0")
    rs, Rm, Rcut = a.rs, a.Rm, a.rcut_over_rs * a.rs
    kappa = 1.0 / (2 * rs)
    dt = torch.complex128
    F, vfun, info = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, dt)
    sea = FlatSea(a.L, a.mass, dev, dt)
    steps = history_ramp(2 * rs, a.dt, ramp_envelope(2 * rs))
    out = {"label": LABEL_PG, "box": info, "kappa": kappa, "t_r": 2 * rs, "dt": a.dt, "holds_over_kappa": C3D_HOLDS,
           "builder_decisions": ["B1", "B2", "B3", "B4", "B5", "B9", "complex128 for single vectors"], "packets": []}
    path = os.path.join(a.outdir, "c3d_L%d_%s.json" % (a.L, a.tag))
    omega = a.omega_over_kappa * kappa
    for kj, mj in ((-1.0, 0.5), (1.0, 0.5), (-1.0, -0.5)):
        w, pinfo = escaping_packet_3d(F, kj, mj, omega, a.R0_over_rs * rs, a.sigma_x, vfun)
        E, dE = packet_energy(F, w)
        n_flat = float((sea.compress(w).abs() ** 2).sum())
        t0 = time.time()
        Ys = backward_multi(F, w, steps, [h / kappa for h in C3D_HOLDS])
        nw = [float((sea.compress(Ys[h / kappa]).abs() ** 2).sum()) for h in C3D_HOLDS]
        rho0 = region_weights(w, a.L, [0, rs, 1.5 * rs, 2 * rs, 2.5 * rs, 1e9])
        row = dict(pinfo, E_killing=E, E_spread=dE, flat_sea_weight=n_flat, n_w=nw, wall_s=time.time() - t0, packet_weights_by_R={
            "edges": [0, rs, 1.5 * rs, 2 * rs, 2.5 * rs, "inf"], "w": rho0})
        out["packets"].append(row)
        log("c3d kappa_j %g m_j %g: E %.5f (omega %.5f) dE %.4f; n_w %s" % (kj, mj, E, omega, dE, ["%.3e" % x for x in nw]))
        jdump(out, path)
        del Ys
    # the map check on the flat box (v = 0): kappa = -kappa_j (derived) and +kappa_j (wrong), small R0 where the centrifugal term is visible
    F0, vf0, _ = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, dt, flow=False)
    mc = []
    for kj in (-1.0, 1.0):
        for sgn in (-1, 1):
            w, pi_ = escaping_packet_3d(F0, kj, 0.5, a.map_omega, a.map_R0, a.map_sigma, vf0, kappa_std_sign=sgn)
            E, dE = packet_energy(F0, w, 0.0)
            mc.append({"kappa_j": kj, "map": "derived kappa = -kappa_j" if sgn < 0 else "wrong kappa = +kappa_j", "E": E, "E_spread": dE,
                       "k0": pi_["k0"], "R0": a.map_R0, "sigma_x": a.map_sigma})
    out["map_check_flat_box"] = mc
    out["wall_s"] = time.time() - T0
    out["finished_utc"] = utc()
    jdump(out, path)
    log("map check: %s" % json.dumps(mc, default=float))


# ------------------------------------------------------------------------------------------------ mode: j4 (the box job) and its timing
DTYPES = {"complex64": torch.complex64, "complex128": torch.complex128}


def mode_phcheck(a):
    """B10's check: complex64 evolution with the particle-hole Gram entropy against the complex128 reference, (i) the pure-state identity
    S(A) = S(complement) on the whole box at L = a.full_L, (ii) the ball R < 5 at L = a.L (hold 12/kappa)."""
    dev = torch.device("cuda:0")
    rs, Rm, Rcut = a.rs, a.Rm, a.rcut_over_rs * a.rs
    kappa = 1.0 / (2 * rs)
    steps = history_ramp(2 * rs, 1.0, ramp_envelope(2 * rs))
    out = {"label": LABEL_PG, "builder_decisions": ["B10"], "started_utc": utc()}
    path = os.path.join(a.outdir, "phcheck_%s.json" % a.tag)
    h = 12.0 / kappa
    Lf = a.full_L
    allsites = np.indices((Lf,) * 3).reshape(3, -1).T
    n = 4 * len(allsites)
    _, Rall = box_coords(Lf)
    Aidx = np.nonzero(Rall.reshape(-1) < a.full_RA)[0]
    Am = (4 * Aidx[:, None] + np.arange(4)).reshape(-1)
    Bm = np.setdiff1d(np.arange(n), Am)
    res = {}
    for dtname, dt in (("complex64", torch.complex64), ("complex128", torch.complex128)):
        F, _, _ = build_pg_box(Lf, rs, Rm, Rcut, a.mass, dev, dt)
        sea = FlatSea(Lf, a.mass, dev, dt)
        Zp = torch.empty((2 * Lf ** 3, n), dtype=dt, device=dev)
        Zq = torch.empty((2 * Lf ** 3, n), dtype=dt, device=dev)
        for c0 in range(0, n, 512):
            nb = min(512, n - c0)
            Y = backward_multi(F, onehot(Lf, allsites, c0, nb, dt, dev), steps, [h])[h]
            Zp[:, c0:c0 + nb], Zq[:, c0:c0 + nb] = sea.compress_both(Y)
        fp = dt == torch.complex128
        C = gram(Zp, dev, slab=256, fp64=fp)
        Cb = gram(Zq, dev, slab=256, fp64=fp)
        eA = entropy_ph(blk(C, Am, Am, dev), blk(Cb, Am, Am, dev))
        eB = entropy_ph(blk(C, Bm, Bm, dev), blk(Cb, Bm, Bm, dev))
        res[dtname] = {"S_A_ph": eA["S"], "S_B_ph": eB["S"], "S_A_minus_S_B_ph": eA["S"] - eB["S"], "S_A_plain": eA["S_plain"], "S_B_plain": eB["S_plain"],
                       "S_A_minus_S_B_plain": eA["S_plain"] - eB["S_plain"], "pair_defect_A": eA["pair_defect_max"], "pair_defect_B": eB["pair_defect_max"]}
        del Zp, Zq, C, Cb
    res["S_A_ph complex64 minus complex128"] = res["complex64"]["S_A_ph"] - res["complex128"]["S_A_ph"]
    res["S_B_ph complex64 minus complex128"] = res["complex64"]["S_B_ph"] - res["complex128"]["S_B_ph"]
    out["full_box"] = dict(res, L=Lf, A_radius=a.full_RA, hold_over_kappa=12.0)
    log("phcheck full box: %s" % json.dumps(out["full_box"], default=float))
    jdump(out, path)
    small, _ = region_indices(a.L, lambda R: R < 5.0)
    ns = 4 * len(small)
    rb = {}
    for dtname, dt in (("complex64", torch.complex64), ("complex128", torch.complex128)):
        F, _, _ = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, dt)
        sea = FlatSea(a.L, a.mass, dev, dt)
        Zp = torch.empty((2 * a.L ** 3, ns), dtype=dt, device=dev)
        Zq = torch.empty((2 * a.L ** 3, ns), dtype=dt, device=dev)
        for c0 in range(0, ns, 512):
            nb = min(512, ns - c0)
            Y = backward_multi(F, onehot(a.L, small, c0, nb, dt, dev), steps, [h])[h]
            Zp[:, c0:c0 + nb], Zq[:, c0:c0 + nb] = sea.compress_both(Y)
        fp = dt == torch.complex128
        e = entropy_ph(gram(Zp, dev, slab=256, fp64=fp), gram(Zq, dev, slab=256, fp64=fp))
        rb[dtname] = e
        del Zp, Zq
    rb["S_ph complex64 minus complex128"] = rb["complex64"]["S"] - rb["complex128"]["S"]
    rb["S_plain complex64 minus complex128"] = rb["complex64"]["S_plain"] - rb["complex128"]["S_plain"]
    out["ball_R5"] = dict(rb, L=a.L, modes=ns, hold_over_kappa=12.0)
    log("phcheck ball R<5: %s" % json.dumps(out["ball_R5"], default=float))
    out["finished_utc"] = utc()
    jdump(out, path)


def hold_groups(holds, bytes_per_hold, budget):
    hs = sorted(holds, reverse=True)
    per = max(1, int(budget // max(bytes_per_hold, 1)))
    return [hs[i:i + per] for i in range(0, len(hs), per)]


def mode_j4(a):
    """the box job of spec 4.2 on the exact profile (control): one region (B7) per box, the holds in groups by the host budget (B5, B6)."""
    dev = torch.device("cuda:0")
    rs, Rm, Rcut = a.rs, a.Rm, a.rcut_over_rs * a.rs
    kappa = 1.0 / (2 * rs)
    F, vfun, info = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, DTYPES[a.dtype])
    sea = FlatSea(a.L, a.mass, dev, DTYPES[a.dtype])
    sea128 = FlatSea(a.L, a.mass, dev, torch.complex128)
    steps = history_ramp(2 * rs, a.dt, ramp_envelope(2 * rs))
    Rbig = max(1.5 * rs, rs + max(a.shells))
    sites, Rsite = region_indices(a.L, lambda R: R < Rbig)
    order = np.argsort(Rsite, kind="stable")
    sites, Rsite = sites[order], Rsite[order]
    nA = 4 * len(sites)
    M = 2 * a.L ** 3
    holds = [h / kappa for h in a.holds]
    groups = hold_groups(holds, M * nA * 8 * (2 if a.ph else 1), a.host_budget_gb * 1e9)
    path = os.path.join(a.outdir, "j4_L%d_%s.json" % (a.L, a.tag))
    out = {"label": LABEL_PG, "box": info, "kappa": kappa, "t_r": 2 * rs, "dt": a.dt, "holds_over_kappa": a.holds, "region_R_max": Rbig, "modes": nA,
           "rows_compressed": M, "host_bytes_per_hold": M * nA * 8, "hold_groups_over_kappa": [[h * kappa for h in g] for g in groups],
           "chunk": a.chunk, "slab": a.slab, "evolution_dtype": a.dtype, "particle_hole_grams": bool(a.ph),
           "builder_decisions": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"] + (["B10"] if a.ph else []), "wall_echo_note":
           "the wall's first echo at about 6 r_s = 3/kappa; the data after it are the periodic box's", "started_utc": utc()}
    jdump(out, path)
    p_all, _ = site_positions(a.L, sites)
    msk = lambda f: np.nonzero(f(Rsite))[0]
    modes_of = lambda si: (4 * si[:, None] + np.arange(4)).reshape(-1)
    sub = {"ball_rs": modes_of(msk(lambda R: R < rs)), "ball_1.5rs": modes_of(msk(lambda R: R < 1.5 * rs))}
    shells = {d: (msk(lambda R, d=d: (R >= rs - d) & (R < rs)), msk(lambda R, d=d: (R >= rs) & (R < rs + d))) for d in a.shells}
    # the flat reference (exact; hold-independent) on the host
    t0 = time.time()
    Cflat = flat_block(sea128, a.L, sites, dev, out_device="cpu")
    out["flat"] = {k: entropy(blk(Cflat, v, v, dev)) for k, v in sub.items()}
    out["timing_s"] = {"flat": time.time() - t0}
    jdump(out, path)
    # the partner packets (2 vector evolutions per hold)
    pk = []
    for kj in (-1.0, 1.0):
        w, pinfo = escaping_packet_3d(F, kj, 0.5, a.omega_over_kappa * kappa, a.R0_over_rs * rs, a.sigma_x, vfun)
        Rin = math.sqrt(Rm ** 3 / rs)
        rows = {}
        for h in holds:
            rec, wp = partner_3d(F, sea, steps, h, w, edges=[0, Rin, rs, 1.5 * rs, 2.5 * rs, 1e9])
            rows["%g" % (h * kappa)] = rec
            del wp
        pk.append(dict(pinfo, holds=rows))
    out["partner"] = pk
    jdump(out, path)
    out["holds"] = {}
    for grp in groups:
        t0 = time.time()
        Zh = {h: torch.empty((M, nA), dtype=torch.complex64) for h in grp}
        Zq = {h: torch.empty((M, nA), dtype=torch.complex64) for h in grp} if a.ph else {}
        for c0 in range(0, nA, a.chunk):
            nb = min(a.chunk, nA - c0)
            X = onehot(a.L, sites, c0, nb, F.dtype, dev)
            for h, Y in backward_holds(F, X, steps, grp):
                if a.ph:
                    zp, zq = sea.compress_both(Y)
                    Zh[h][:, c0:c0 + nb].copy_(zp.to(torch.complex64).cpu())
                    Zq[h][:, c0:c0 + nb].copy_(zq.to(torch.complex64).cpu())
                    del zp, zq
                else:
                    Zh[h][:, c0:c0 + nb].copy_(sea.compress(Y).to(torch.complex64).cpu())
                del Y
            del X
            if c0 == 0:
                torch.cuda.synchronize()
                out["timing_s"]["first_chunk_group_%g" % (grp[0] * kappa)] = time.time() - t0
                jdump(out, path)
        out["timing_s"]["evolution_group_%g" % (grp[0] * kappa)] = time.time() - t0
        for h in grp:
            t1 = time.time()
            C = gram(Zh.pop(h), dev, slab=a.slab).cpu()
            torch.cuda.empty_cache()
            Cb = gram(Zq.pop(h), dev, slab=a.slab).cpu() if a.ph else None
            torch.cuda.empty_cache()
            tg = time.time() - t1
            rec = {"gram_s": tg}
            for k, v in sub.items():
                rec["S_" + k] = entropy_ph(blk(C, v, v, dev), blk(Cb, v, v, dev)) if a.ph else entropy(blk(C, v, v, dev))
                rec["excess_" + k] = rec["S_" + k]["S"] - out["flat"][k]["S"]
            for d, (si, so) in shells.items():
                rec["shell_d%g" % d] = shell_observables(C, Cflat, modes_of(si), modes_of(so), p_all[si], p_all[so], dev=dev, Cbar=Cb)
            del C, Cb
            rec["obs_s"] = time.time() - t1 - tg
            out["holds"]["%g" % (h * kappa)] = rec
            out["finished_hold_utc_%g" % (h * kappa)] = utc()
            jdump(out, path)
            log("hold %g/kappa: S excess ball r_s %.5f, ball 1.5 r_s %.5f" % (h * kappa, rec["excess_ball_rs"], rec["excess_ball_1.5rs"]))
    out["wall_s"] = time.time() - T0
    out["finished_utc"] = utc()
    jdump(out, path)
    write_j4_md(out, path.replace(".json", ".md"))


def write_j4_md(o, path):
    b = o["box"]
    L = ["# J4: the 3-dimensional box on the exact profile (L = %d)" % b["L"], "", "Label: " + o["label"] + ".", "",
         "Script obs_box3d_v1.py, mode j4, started %s, finished %s, wall %.0f s; r_s %g, R_m %g, R_cut %g, m %g, t_r %g, dt %g; region R < %g (%d modes); "
         "max|v| on the faces %s; %s." % (o["started_utc"], o.get("finished_utc"), o.get("wall_s", 0), b["rs"], b["Rm"], b["Rcut"], b["m"], o["t_r"], o["dt"],
                                          o["region_R_max"], o["modes"], fmt(b["max_abs_v_on_faces"]), o["wall_echo_note"]), ""]
    rows = []
    for h, r in sorted(o["holds"].items(), key=lambda x: float(x[0])):
        rows.append((h, r["S_ball_rs"]["S"], o["flat"]["ball_rs"]["S"], r["excess_ball_rs"], r["S_ball_1.5rs"]["S"], o["flat"]["ball_1.5rs"]["S"],
                     r["excess_ball_1.5rs"]))
    L += ["## Ball entropies", "", md_table(["hold kappa", "S(R<r_s)", "flat", "excess", "S(R<1.5 r_s)", "flat", "excess"], rows), ""]
    rows = []
    for h, r in sorted(o["holds"].items(), key=lambda x: float(x[0])):
        for k, s in r.items():
            if k.startswith("shell_d"):
                rows.append((h, k[7:], s["formed"]["I"], s["flat"]["I"], s["excess"]["I"], s["formed"]["G_HS2"], s["flat"]["G_HS2"], s["excess"]["dG_HS2"]))
    L += ["## Shells: I(A_in : A_out) and the cross block", "", md_table(["hold kappa", "d", "I formed", "I flat", "excess I", "||G||^2", "flat", "||dG||^2"], rows), ""]
    rows = []
    for p in o.get("partner", []):
        for h, r in p["holds"].items():
            rows.append((p["kappa_j"], h, r["n"], r["norm2"], r["n_1_minus_n"]) + tuple(r["weights"]["w"]))
    L += ["## Partner of the 3-dimensional packet (omega/kappa as in the args)", "",
          md_table(["kappa_j", "hold kappa", "n", "||w_p||^2", "n(1-n)", "w core", "w [R_in,r_s)", "w [r_s,1.5r_s)", "w [1.5,2.5)r_s", "w beyond"], rows), ""]
    L += ["Two-point bins by distance and by angle are in the JSON (holds -> shell_d -> formed | flat | excess -> two_point)."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def mode_timing(a):
    """J4's timing: one chunk through the shared-hold backward evolution and the compression at a.L; a Gram and the eigenvalues at a smaller n;
    the J4 total extrapolated from the counts."""
    dev = torch.device("cuda:0")
    rs, Rm, Rcut = a.rs, a.Rm, a.rcut_over_rs * a.rs
    kappa = 1.0 / (2 * rs)
    F, vfun, info = build_pg_box(a.L, rs, Rm, Rcut, a.mass, dev, DTYPES[a.dtype])
    sea = FlatSea(a.L, a.mass, dev, DTYPES[a.dtype])
    steps = history_ramp(2 * rs, a.dt, ramp_envelope(2 * rs))
    sites, Rsite = region_indices(a.L, lambda R: R < max(1.5 * rs, rs + max(a.shells)))
    holds = [h / kappa for h in a.holds]
    out = {"box": info, "chunk": a.chunk, "region_modes": 4 * len(sites), "evolution_dtype": a.dtype, "holds_over_kappa": a.holds, "started_utc": utc()}
    X = onehot(a.L, sites, 0, a.chunk, F.dtype, dev)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        apply_h(X, F, 1.0)
    torch.cuda.synchronize(); out["apply_s"] = (time.time() - t0) / 20
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    tc = 0.0
    for h, Y in backward_holds(F, X, steps, holds):
        torch.cuda.synchronize()
        t1 = time.time()
        if a.ph:
            zp, zq = sea.compress_both(Y)
            z = (zp.to(torch.complex64).cpu(), zq.to(torch.complex64).cpu())
            del zp, zq
        else:
            z = sea.compress(Y).to(torch.complex64).cpu()
        tc += time.time() - t1
        del Y
    torch.cuda.synchronize(); out["backward_all_holds_and_compress_s"] = time.time() - t0
    out["compress_to_host_s"] = tc
    out["peak_gpu_GB"] = torch.cuda.max_memory_allocated() / 1e9
    del X, z
    r = F.bound_s(1.0)
    hs = sorted(holds)
    segs = [h1 - h0 for h0, h1 in zip([0.0] + hs[:-1], hs) if h1 > h0]
    out["applies_per_chunk"] = sum(term_count(r * s_) for s_ in segs) + len(holds) * sum(
        term_count(F.bound_s(sa) * h_ / 2) + term_count(F.bound_s(sb) * h_ / 2) for sa, sb, h_ in steps)
    n = a.gram_n
    Z = (torch.randn(2 * a.L ** 3, n, dtype=torch.complex64, device=dev) / math.sqrt(2 * a.L ** 3))
    torch.cuda.synchronize(); t0 = time.time()
    C = gram(Z, dev, slab=a.slab)
    torch.cuda.synchronize(); out["gram_s_at_n"] = [n, time.time() - t0]
    del Z
    torch.cuda.synchronize(); t0 = time.time()
    spectrum(C)
    torch.cuda.synchronize(); out["eigvalsh_s_at_n"] = [n, time.time() - t0]
    del C
    nch = math.ceil(4 * len(sites) / a.chunk)
    nA = 4 * len(sites)
    out["particle_hole_grams"] = bool(a.ph)
    out["estimate"] = {"chunks": nch, "evolution_all_holds_s": nch * out["backward_all_holds_and_compress_s"],
                       "gram_per_hold_s": out["gram_s_at_n"][1] * (nA / n) ** 2 * (2 if a.ph else 1),
                       "eigvalsh_per_hold_s_nA": out["eigvalsh_s_at_n"][1] * (nA / n) ** 3 * (2 if a.ph else 1)}
    log(json.dumps(out, default=float))
    out["finished_utc"] = utc()
    jdump(out, os.path.join(a.outdir, "timing_L%d_%s%s_%s.json" % (a.L, a.dtype, "_ph" if a.ph else "", a.tag)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["smoke", "smallball", "c3d", "j4", "timing", "phcheck"])
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--rs", type=float, default=8.0)
    p.add_argument("--Rm", type=float, default=2.0)
    p.add_argument("--rcut-over-rs", type=float, default=2.5)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--holds", type=float, nargs="+", default=[0.0, 12.0, 24.0, 48.0], help="holds in units of 1/kappa")
    p.add_argument("--shells", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--slab", type=int, default=256)
    p.add_argument("--host-budget-gb", type=float, default=130.0)
    p.add_argument("--omega-over-kappa", type=float, default=4.0)
    p.add_argument("--R0-over-rs", type=float, default=2.2)
    p.add_argument("--sigma-x", type=float, default=4.0)
    p.add_argument("--map-omega", type=float, default=0.5)
    p.add_argument("--map-R0", type=float, default=6.0)
    p.add_argument("--map-sigma", type=float, default=1.5)
    p.add_argument("--dense-L", type=int, default=16)
    p.add_argument("--dense-rs", type=float, default=2.0)
    p.add_argument("--gram-n", type=int, default=8192)
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--ph", action="store_true", help="B10: the particle-hole Gram entropies in j4")
    p.add_argument("--full-L", type=int, default=12)
    p.add_argument("--full-RA", type=float, default=3.0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    log("start %s mode %s args %s" % (utc(), a.mode, vars(a)))
    {"smoke": mode_smoke, "smallball": mode_smallball, "c3d": mode_c3d, "j4": mode_j4, "timing": mode_timing, "phcheck": mode_phcheck}[a.mode](a)
    log("done %s" % utc())


if __name__ == "__main__":
    main()
