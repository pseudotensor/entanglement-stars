#!/usr/bin/env python
"""o5_chain_export_v1.py -- O5 (GAP10_T7_final_v1.1 section 9): the chain side of the same-quantity witness exchange.

One radial-chain column of HORIZON's construction (HORIZON_single_object_v2 definition (vi): the local symbol
N D(q) + (v/2) sin 2q with D(q) = m sigma_z + sin q sigma_x + (kappa/r) sigma_y; the column a radial Gaussian of width sigma
times the carrier exp(i a k_0 (r - R_c)) times the particle spinor u_+(k_0) of D at R_c) on the T7 slice's applied N and v
held fixed (record_slice.npz read in place: the table r = 1 .. n a, N_applied, v_applied <= 0; the chain's radius grid is the
table's), propagated exactly on the chain's held h (the dense eigendecomposition of the 2n x 2n Hermitian matrix; the
propagation error is that decomposition's rounding, printed as a first-order bound), exported per section 9 (a) to (e):
  (a) the chain vector on its radius grid with its spinor convention (SPINOR_CONVENTION; o5_lift_v1.py reads it);
  (b) its Killing split in the 5 bins (-inf, -b), (-b, -m), (-m, m), (m, b), (b, inf) with b = b1 = sqrt(m^2 + 1/a^2) (the
      chain's free top, 1.0198 at ma 0.2) and with b = b3 = sqrt(m^2 + 3/a^2) (the cube's, 1.7436), both printed, every edge
      scaled by N_inf (the lapse at the table's last shell: the asymptotic gap is |omega| < N_inf m, both energy signs kept);
      the weight within --edge_window of any edge is printed beside the bins and assigned to neither (the threshold columns);
  (c) its weight beyond r_h (the table's outer root of N - |v|, or --rh) and beyond 1.25, 1.5, 2 and 3 r_s (r_s 29 a: 43.5,
      58, 87 a) at every sweep end (29 a per sweep, tau = --tau_rs r_s), by the cell rule (the chain cell [j - 1/2, j + 1/2)
      counted by its part beyond the radius: the chain's estimate of the integral the cube's 'r > R' site sum estimates) and by
      the strict site rule (r_j > R), both stored, each with the band 2 sqrt(p) delta + delta^2, delta the propagation error;
  (d) its fixed-cut number current through r_h, J_> = <psi, i[h, P_>] psi> = 2 Im <P_> psi, h P_< psi> (P_> the strict projector
      on r_j > r_h; outward positive; HORIZON's rate (1), GAP10 section 3), at every readout time; its exact time integral at
      the sweep ends (the spectral form on the eigendecomposition) and the composite-Simpson integral on a --fine_sub times
      finer grid at every readout time, both against the strict weight's change (the fixed-cut identity; to rounding for the
      exact form);
  (e) its complex amplitude on the energy-matched detector packets at 2 r_s, both carrier signs: the chain's detector is the
      radial Gaussian of width --sigma_det at r = --det_frac r_s with the outgoing root k_det of the chain's own local
      dispersion at the column's mean Killing energy (group velocity positive; the smallest such root), or the column's own
      carrier when no outgoing root exists there (flagged 'not energy matched'); +-k_det each with its particle spinor, read at
      every readout time (the times the cube's driver reads: dt_read within sweeps of tau).
Everything raw is saved (the states at every readout time, the eigenvalues, the detector vectors, the shell weights, the
current). Resumable: an existing output pair is skipped unless --force. No plots. CPU, 1 to 3 minutes per column at n 1024.
Nothing here is a result of the object: every readout is on 1 held record at 1 averaging width.
Outputs: <out>/o5_chain_Rc<Rc>_k<kappa><tag>.npz and .json.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.optimize as spo

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

SPINOR_CONVENTION = (
    "chain 2-spinor (psi_1, psi_2) on the radius grid r_j = j a, j = 1 .. n, in the representation of HORIZON v2 definition (vi): "
    "h = N_j [m sigma_z + (kappa/r_j) sigma_y] + sigma_x [n^+_j psi_{j+1} - n^-_j psi_{j-1}]/(2i) + [w^+_j psi_{j+2} - w^-_j psi_{j-2}]/(4i), "
    "n^+-_j = (N_j + N_{j+-1})/2, w^+-_j = (v_j + v_{j+-2})/2 (v <= 0 inward), open at j = 1 (the regular core) and at j = n; the local "
    "symbol is N D(q) + (v/2) sin 2q with D(q) = m sigma_z + sin q sigma_x + (kappa/r) sigma_y, so h's local eigenvectors are D's "
    "(u_+(k) proportional to (E + m, sin k + i kappa/r), E = sqrt(m^2 + sin^2 k + (kappa/r)^2)). Against the Dirac radial functions with "
    "Psi = (1/r)(G Omega_{-kappa, m_j}, i F Omega_{+kappa, m_j}) (Omega the spinor spherical harmonics with (sigma.L + 1) Omega_kappa = "
    "-kappa Omega_kappa and sigma.rhat Omega_kappa = -Omega_{-kappa}) this representation is psi_1 = G, psi_2 = -i F, so the 4-component "
    "lift is Psi = (1/r)(psi_1 Omega_{-kappa, m_j}, -psi_2 Omega_{+kappa, m_j}): o5_lift_v1.py's default (--lower_sign +1 --kappa_assign +1), "
    "whose reduction check prints this convention against its 3 alternatives.")


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ------------------------------------------------------------------ the record
def load_record(path):
    """The slice's table: r (a), N (N_applied) and v (v_applied, <= 0 inward; or u = |v|). The chain's grid is the table's."""
    z = np.load(path)
    r = np.asarray(z['r'], dtype=float)
    N = np.asarray(z['N'], dtype=float) if 'N' in z.files else np.ones_like(r)
    if 'v' in z.files:
        v = -np.abs(np.asarray(z['v'], dtype=float))
    elif 'u' in z.files:
        v = -np.abs(np.asarray(z['u'], dtype=float))
    else:
        raise ValueError(f"{path}: neither 'v' nor 'u' in the table")
    o = np.argsort(r)
    r, N, v = r[o], N[o], v[o]
    if r.size < 8 or (not np.allclose(np.diff(r), 1.0, atol=1e-9)) or abs(r[0] - 1.0) > 1e-9:
        raise ValueError("the chain reads the record's own radius grid r = 1, 2, .. n (a = 1): this table is not that grid")
    return r, N, v


def table_roots(r, N, v):
    """Zeros of N - |v| between the table's shells by linear interpolation (the solve lane's applied-root rule): [(r0, slope)]."""
    c = N - np.abs(v)
    out = []
    for j in range(r.size - 1):
        if c[j] == c[j + 1]:
            continue
        if c[j] == 0.0 or np.sign(c[j]) != np.sign(c[j + 1]):
            t = c[j] / (c[j] - c[j + 1])
            if 0.0 <= t <= 1.0:
                out.append((float(r[j] + t * (r[j + 1] - r[j])), float((c[j + 1] - c[j]) / (r[j + 1] - r[j]))))
    return out


# ------------------------------------------------------------------ the chain operator
class RadialChain:
    """h on the half-line grid r_j = j a (open at both ends) in channel kappa: the matrix of SPINOR_CONVENTION, site-major layout
    (index 2 j + s for site j = 0 .. n-1, component s = 0, 1)."""

    def __init__(self, r, N, v, m, kappa):
        r = np.asarray(r, dtype=float)
        N = np.asarray(N, dtype=float)
        v = np.asarray(v, dtype=float)
        n = r.size
        self.r, self.N, self.v, self.m, self.kappa, self.n, self.dim = r, N, v, float(m), float(kappa), n, 2 * n
        H = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        hop = SX / (2.0j)
        flw = I2 / (4.0j)
        for j in range(n):
            a = 2 * j
            H[a:a + 2, a:a + 2] = N[j] * (self.m * SZ + (self.kappa / r[j]) * SY)
            if j + 1 < n:
                w = 0.5 * (N[j] + N[j + 1])
                H[a:a + 2, a + 2:a + 4] += w * hop
                H[a + 2:a + 4, a:a + 2] += -w * hop
            if j + 2 < n:
                w = 0.5 * (v[j] + v[j + 2])
                H[a:a + 2, a + 4:a + 6] += w * flw
                H[a + 4:a + 6, a:a + 2] += -w * flw
        self.H = H
        self.w = self.V = None
        self.eps_res = self.eps_orth = self.wmax = None

    def hermiticity_defect(self):
        return float(np.max(np.abs(self.H - self.H.conj().T)))

    def particle_hole_defect(self):
        """|| D h^* D^dag + h ||_max with D = diag_j((-1)^j) x sigma_y: h anticommutes with the on-site antiunitary (-1)^j sigma_y K."""
        D = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for j in range(self.n):
            D[2 * j:2 * j + 2, 2 * j:2 * j + 2] = ((-1.0) ** j) * SY
        return float(np.max(np.abs(D @ np.conj(self.H) @ D.conj().T + self.H)))

    def diagonalize(self):
        w, V = np.linalg.eigh(self.H)
        self.w, self.V = w, V
        R = self.H @ V - V * w[None, :]
        self.eps_res = float(np.linalg.norm(R))                                   # Frobenius: bounds ||h - V Lambda V^dag||_2 to first order
        self.eps_orth = float(np.linalg.norm(V.conj().T @ V - np.eye(self.dim)))
        self.wmax = float(np.max(np.abs(w)))
        return w, V

    def prop_error_bound(self, t):
        """||exp(-i h t) psi - V exp(-i Lambda t) V^dag psi|| <= eps_res t + 2 eps_orth (1 + |Lambda|_max t): the stated propagation error."""
        t = np.asarray(t, dtype=float)
        return self.eps_res * t + 2.0 * self.eps_orth * (1.0 + self.wmax * t)

    def expectation(self, psi):
        hp = self.H @ psi
        return float(np.real(np.vdot(psi, hp))), float(np.real(np.vdot(hp, hp)))

    def cut_operator(self, mask_sites):
        """Q = i [h, P] for the site projector P = diag(mask): J = <psi, Q psi> is the number current through the cut, outward positive."""
        mk = np.repeat(np.asarray(mask_sites, dtype=bool), 2).astype(float)
        return 1.0j * (self.H * mk[None, :] - mk[:, None] * self.H)

    def propagate(self, psi0, times):
        """States at the given times, (ntimes, n, 2), and the eigen-coefficients c0 = V^dag psi0."""
        c0 = self.V.conj().T @ psi0
        ph = np.exp(-1.0j * np.outer(self.w, np.asarray(times, dtype=float)))
        st = self.V @ (ph * c0[:, None])
        return np.ascontiguousarray(st.T).reshape(len(times), self.n, 2), c0


# ------------------------------------------------------------------ the column, the dispersion, the detector
def particle_spinor(k, R, m, kappa):
    """u_+(k) of D(k) = m sigma_z + sin k sigma_x + (kappa/R) sigma_y (HORIZON's particle spinor) and its energy E(k)."""
    s, c = math.sin(k), kappa / R
    E = math.sqrt(m * m + s * s + c * c)
    u = np.array([E + m, s + 1.0j * c], dtype=np.complex128) / math.sqrt(2.0 * E * (E + m))
    return u, E


def gaussian_column(r, Rc, sigma, k0, cut, spinor):
    """psi_j = Z exp(-(r_j - R_c)^2/(4 sigma^2)) exp(i k_0 (r_j - R_c)) u on |r_j - R_c| <= cut sigma, unit norm on the grid; the amplitude
    envelope's 4 sigma^2 makes sigma the density's standard deviation (GAP10's packet_mode convention)."""
    d = r - Rc
    env = np.exp(-d * d / (4.0 * sigma * sigma)) * (np.abs(d) <= cut * sigma)
    s2 = float(np.sum(env * env))
    if s2 <= 0.0:
        raise ValueError("empty column support on the grid")
    Z = 1.0 / math.sqrt(s2)
    f = Z * env * np.exp(1.0j * k0 * d)
    psi = f[:, None] * np.asarray(spinor, dtype=np.complex128)[None, :]
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    info = dict(Z=Z, n_support=int(np.count_nonzero(env)), gaussian_weight_outside_cut=2.0 * (1.0 - Phi(cut)),
                gaussian_weight_below_grid=Phi((r[0] - 0.5 - Rc) / sigma), gaussian_weight_beyond_grid=1.0 - Phi((r[-1] + 0.5 - Rc) / sigma))
    return psi, info


def chain_omega(k, N, v, m, kappa, R):
    k = np.asarray(k, dtype=float)
    E = np.sqrt(m * m + np.sin(k) ** 2 + (kappa / R) ** 2)
    return N * E + 0.5 * v * np.sin(2.0 * k)


def chain_vg(k, N, v, m, kappa, R):
    k = np.asarray(k, dtype=float)
    E = np.sqrt(m * m + np.sin(k) ** 2 + (kappa / R) ** 2)
    return N * np.sin(k) * np.cos(k) / E + v * np.cos(2.0 * k)


def outgoing_root_chain(omega, N, v, m, kappa, R, kmax, nscan):
    """All roots k in (0, kmax) of omega(k) = N E(k) + (v/2) sin 2k, each with its group velocity; the smallest root with positive group
    velocity is the outgoing root (None if there is none: the energy lies below the outgoing branch's minimum at R)."""
    kg = np.linspace(1e-6, kmax, int(nscan))
    g = chain_omega(kg, N, v, m, kappa, R) - omega
    idx = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
    roots = []
    for i in idx:
        try:
            kr = spo.brentq(lambda k: float(chain_omega(k, N, v, m, kappa, R) - omega), kg[i], kg[i + 1], xtol=1e-14, rtol=1e-14)
        except ValueError:
            continue
        roots.append((float(kr), float(chain_vg(kr, N, v, m, kappa, R))))
    out = [k for k, vg in roots if vg > 0.0]
    return (min(out) if out else None), roots


def detector_packets(r, r_det, sigma_det, cut, k_det, m, kappa, spinor_mode):
    members = []
    for sgn in (+1, -1):
        k = sgn * k_det
        if spinor_mode == 'particle':
            u, E = particle_spinor(k, r_det, m, kappa)
        else:
            u, E = np.array([1.0, 0.0], dtype=np.complex128), m
        vec, info = gaussian_column(r, r_det, sigma_det, k, cut, u)
        members.append(dict(sign=sgn, k=float(k), spinor=[[u[0].real, u[0].imag], [u[1].real, u[1].imag]], E_symbol=E, Z=info['Z'],
                            label=f"o5det|r={r_det:.3f}|s={'+' if sgn > 0 else '-'}|k={k:+.5f}|sig={sigma_det:g}|lifted", vec=vec))
    return members


# ------------------------------------------------------------------ readouts
def beyond_weights(r, dens, radii, rule):
    """dens (nt, n) -> (nt, nrad): 'midpoint' counts the cell [j - 1/2, j + 1/2) by its part beyond R; 'strict' counts r_j > R."""
    radii = np.asarray(radii, dtype=float)
    if rule == 'midpoint':
        frac = np.clip(r[:, None] + 0.5 - radii[None, :], 0.0, 1.0)
    elif rule == 'strict':
        frac = (r[:, None] > radii[None, :]).astype(float)
    else:
        raise ValueError(rule)
    return dens @ frac


def killing_bins(ev, wts, m_e, b_e, eps):
    """The 5 bins with the edges +-m_e, +-b_e; the weight within eps of an edge is listed beside and assigned to no bin."""
    near = np.zeros(ev.size, dtype=bool)
    thr = {}
    for name, e in (('-b', -b_e), ('-m', -m_e), ('+m', m_e), ('+b', b_e)):
        sel = np.abs(ev - e) <= eps
        thr[name] = float(wts[sel].sum())
        near |= sel
    bins = dict(below_minus_b=float(wts[(ev < -b_e) & ~near].sum()), minus_b_to_minus_m=float(wts[(ev > -b_e) & (ev < -m_e) & ~near].sum()),
                gap=float(wts[(ev > -m_e) & (ev < m_e) & ~near].sum()), m_to_b=float(wts[(ev > m_e) & (ev < b_e) & ~near].sum()),
                above_b=float(wts[(ev > b_e) & ~near].sum()))
    return dict(edges=dict(m=float(m_e), b=float(b_e)), bins=bins, threshold_columns=thr, edge_window=float(eps),
                total=float(sum(bins.values()) + sum(thr.values())))


def killing_three(ev, wts, m_e, eps):
    near = np.abs(np.abs(ev) - m_e) <= eps
    return dict(above_m=float(wts[(ev > m_e) & ~near].sum()), gap=float(wts[(np.abs(ev) < m_e) & ~near].sum()),
                below_minus_m=float(wts[(ev < -m_e) & ~near].sum()), threshold_plus_m=float(wts[near & (ev > 0)].sum()),
                threshold_minus_m=float(wts[near & (ev < 0)].sum()), edge_window=float(eps))


def lattice_scale_weight(psi, pad=4):
    """The unsigned windowed share of the column at folded momentum |q| > pi/4 (zero-padded transform of both components)."""
    n = psi.shape[0]
    nf = 1 << int(math.ceil(math.log2(pad * n)))
    F = np.fft.fft(psi, n=nf, axis=0)
    q = 2.0 * math.pi * np.fft.fftfreq(nf)
    folded = np.minimum(np.abs(q), math.pi - np.abs(q))
    P = np.sum(np.abs(F) ** 2, 1)
    tot = float(P.sum())
    return dict(lattice_scale=float(P[folded > math.pi / 4.0].sum() / tot), q=q, power=P / tot)


def integrated_current_exact(w, V, Q, c0, times):
    """int_0^t <psi(s), Q psi(s)> ds in the spectral form: sum_{n n'} conj(c_n) M_{n n'} c_{n'} K(E_n - E_{n'}, t), M = V^dag Q V,
    K(Omega, t) = (exp(i Omega t) - 1)/(i Omega) (t at Omega = 0)."""
    M = V.conj().T @ Q @ V
    A = np.conj(c0)[:, None] * M * c0[None, :]
    Om = w[:, None] - w[None, :]
    small = np.abs(Om) < 1e-12
    Oms = np.where(small, 1.0, Om)
    out = np.zeros(len(times))
    for i, t in enumerate(np.asarray(times, dtype=float)):
        K = np.where(small, t, (np.exp(1.0j * Oms * t) - 1.0) / (1.0j * Oms))
        out[i] = float(np.real(np.sum(A * K)))
    return out


def cumulative_simpson(Jf, dtf, sub):
    """The cumulative integral at the coarse points of a function sampled at the fine step dtf, composite Simpson over each
    coarse interval of sub (even) fine steps."""
    if sub % 2:
        raise ValueError("fine_sub must be even")
    nb = (Jf.size - 1) // sub
    out = np.zeros(nb + 1)
    for b in range(nb):
        seg = Jf[b * sub:b * sub + sub + 1]
        out[b + 1] = out[b] + (dtf / 3.0) * (seg[0] + seg[-1] + 4.0 * seg[1:-1:2].sum() + 2.0 * seg[2:-1:2].sum())
    return out


def half_weight_clock(t, p, final_index=-1):
    """HORIZON's arrival rule on a sampled p(t): the first clock at which p reaches half its value at the last sample, linear
    interpolation between samples; nan when the final value is not positive or the level is never reached."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    pf = p[final_index]
    if not (pf > 0.0):
        return float('nan')
    target = 0.5 * pf
    if p[0] >= target:
        return float(t[0])
    for i in range(1, p.size if final_index == -1 else final_index + 1):
        if p[i] >= target:
            if p[i] == p[i - 1]:
                return float(t[i])
            return float(t[i - 1] + (target - p[i - 1]) / (p[i] - p[i - 1]) * (t[i] - t[i - 1]))
    return float('nan')


# ------------------------------------------------------------------ the run
def build_parser():
    ap = argparse.ArgumentParser(description="O5 chain side: 1 radial-chain column of HORIZON's construction on the T7 slice's held fields, exported per section 9 (a) to (e)")
    ap.add_argument('--record', required=True, help="record_slice.npz (keys r, N, v)")
    ap.add_argument('--record_json', default='', help="record_slice.json (provenance; its roots and md5 are cross-checked when given)")
    ap.add_argument('--Rc', type=float, required=True, help="the column's centre in a (20.0 the T7 d 2 radius; 14.0 the d 8 radius)")
    ap.add_argument('--kappa', type=int, default=1, help="the angular channel (HORIZON's kappa; 1 the lowest, 2 the second member)")
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--ak0', type=float, default=2.0 * math.pi / 64.0, help="the carrier a k_0 (2 pi/64 HORIZON's long-wavelength column; pi/4 its threshold column; pi/2 its lattice-scale column)")
    ap.add_argument('--sigma', type=float, default=8.0, help="the radial profile's width in a (the density's standard deviation; HORIZON's columns 16, 8, 4)")
    ap.add_argument('--cut', type=float, default=5.0, help="the profile is set to 0 beyond cut sigma from R_c (GAP10's ball cut)")
    ap.add_argument('--spinor', choices=['particle', 'rest'], default='particle', help="particle: u_+(k_0) of D at R_c; rest: (1, 0), the fallback shell's spinor")
    ap.add_argument('--rs', type=float, default=29.0, help="r_s in a: the shells and the sweep (T7's 29 a; the chain's 2GE 29.25 a is printed beside)")
    ap.add_argument('--tau_rs', type=float, default=1.0, help="sweep length in units of r_s")
    ap.add_argument('--nsweeps', type=int, default=10, help="10 for the exchange window; 16 for the collar leg")
    ap.add_argument('--dt_read', type=float, default=0.5, help="readout step within a sweep (the cube driver's dt_read)")
    ap.add_argument('--radii_rs', type=float, nargs='*', default=[1.25, 1.5, 2.0, 3.0], help="shells beyond f r_s read beside r_h")
    ap.add_argument('--rh', type=float, default=0.0, help="the cut radius; 0 = the table's outer root of N - |v|")
    ap.add_argument('--det_frac', type=float, default=2.0, help="detector radius in r_s")
    ap.add_argument('--sigma_det', type=float, default=6.0, help="detector width in a (GAP10's receiver packets' 6 a)")
    ap.add_argument('--det_cut', type=float, default=5.0)
    ap.add_argument('--kdet_kmax', type=float, default=math.pi / 2.0, help="the outgoing root is searched on (0, kdet_kmax)")
    ap.add_argument('--kdet_nscan', type=int, default=20001)
    ap.add_argument('--edge_window', type=float, default=0.005, help="half-width of the threshold columns about +-m and +-b (assigned to no bin)")
    ap.add_argument('--fine_sub', type=int, default=8, help="the Simpson integral of the current runs on dt_read/fine_sub (even)")
    ap.add_argument('--exact_integral', choices=['sweeps', 'all'], default='sweeps', help="the exact spectral integral at the sweep ends, or at every readout time (about a minute more)")
    ap.add_argument('--out', default='o5_chain')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    return ap


def run(args):
    rs, m = float(args.rs), float(args.ma)
    stem = f"o5_chain_Rc{args.Rc:g}_k{args.kappa}{args.tag}"
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, stem)
    if (not args.force) and os.path.exists(base + '.json') and os.path.exists(base + '.npz'):
        print(f"[skip] {base}.npz and .json exist (--force to redo)", flush=True)
        return 0
    tw = time.time()
    r, N, v = load_record(args.record)
    n = r.size
    rec_md5 = md5(args.record)
    print(f"[record] {args.record} (md5 {rec_md5}): n={n}, r={r[0]:g}..{r[-1]:g} a; N in [{N.min():.6f}, {N.max():.10f}]; v in [{v.min():.6f}, {v.max():.6f}] (<= 0 inward)", flush=True)
    roots = table_roots(r, N, v)
    if args.rh > 0.0:
        r_h, slope_h = float(args.rh), float('nan')
    elif roots:
        r_h, slope_h = roots[-1]
    else:
        raise ValueError("no root of N - |v| on the table: give --rh")
    kappa_h = abs(slope_h) if np.isfinite(slope_h) else float('nan')
    N_inf = float(N[-1])
    b1, b3 = math.sqrt(m * m + 1.0), math.sqrt(m * m + 3.0)
    print(f"[roots] N - |v| = 0 on the table at {[round(x[0], 4) for x in roots]} (slopes {[round(x[1], 6) for x in roots]}); the cut r_h = {r_h:.6f} a, "
          f"kappa_h = |slope| = {kappa_h:.6f}; N_inf = {N_inf:.10f}; b1 = {b1:.4f} (chain), b3 = {b3:.4f} (cube); edges scaled by N_inf", flush=True)
    rec_json_info = None
    if args.record_json:
        with open(args.record_json) as fh:
            RJ = json.load(fh)
        rj = RJ.get('roots_on_table', {}).get('roots', [])
        diffs = [abs(a_[0] - float(b_['r'])) for a_, b_ in zip(roots, rj)]
        rec_json_info = dict(path=args.record_json, md5=md5(args.record_json), roots_json=rj, root_differences=diffs,
                             two_gE_tail_last=RJ.get('slice', {}).get('two_gE_tail_last'), N_max_json=RJ.get('fields_written', {}).get('N_max'),
                             md5_written_npz=RJ.get('md5_written_npz'), md5_matches=(RJ.get('md5_written_npz') == rec_md5),
                             axes_r_h={k: RJ.get('axes', {}).get(k, {}).get('r_h') for k in ('100', '111')})
        print(f"[record json] roots on table {[round(float(x['r']), 4) for x in rj]} against this script's: differences {['%.1e' % d for d in diffs]}; "
              f"2GE tail {rec_json_info['two_gE_tail_last']}; the json's md5 of the written npz matches the file read: {rec_json_info['md5_matches']}; "
              f"the cube's allocated r_h [100] {rec_json_info['axes_r_h']['100']}, [111] {rec_json_info['axes_r_h']['111']} (the cube reads the chain's cut with --o5_rh)", flush=True)

    # ---- the chain
    t0 = time.time()
    chain = RadialChain(r, N, v, m, args.kappa)
    herm = chain.hermiticity_defect()
    ph = chain.particle_hole_defect()
    chain.diagonalize()
    t_eig = time.time() - t0
    print(f"[chain] kappa={args.kappa} m={m:g} dim={chain.dim}: Hermiticity defect {herm:.1e}; particle-hole anticommutation defect {ph:.1e} (with (-1)^j sigma_y K); "
          f"eigendecomposition {t_eig:.1f}s: spectrum [{chain.w[0]:+.6f}, {chain.w[-1]:+.6f}]; ||hV - V Lambda||_F {chain.eps_res:.2e}; ||V^dag V - I||_F {chain.eps_orth:.2e}", flush=True)

    # ---- the column
    if args.spinor == 'particle':
        u, E_sym = particle_spinor(args.ak0, args.Rc, m, args.kappa)
    else:
        u, E_sym = np.array([1.0, 0.0], dtype=np.complex128), m
    psi0, cinfo = gaussian_column(r, args.Rc, args.sigma, args.ak0, args.cut, u)
    psi0f = psi0.reshape(-1)
    E_mean, E2 = chain.expectation(psi0f)
    E_rms = math.sqrt(max(E2 - E_mean * E_mean, 0.0))
    c0 = chain.V.conj().T @ psi0f
    wts = np.abs(c0) ** 2
    edges = dict(m=N_inf * m, b1=N_inf * b1, b3=N_inf * b3)
    kb1 = killing_bins(chain.w, wts, edges['m'], edges['b1'], args.edge_window)
    kb3 = killing_bins(chain.w, wts, edges['m'], edges['b3'], args.edge_window)
    k3 = killing_three(chain.w, wts, edges['m'], args.edge_window)
    ls = lattice_scale_weight(psi0)
    print(f"[column] R_c={args.Rc:g} a, sigma={args.sigma:g} a, a k_0={args.ak0:+.5f} (wavelength {(2 * math.pi / abs(args.ak0)) if args.ak0 else float('inf'):.1f} a), cut {args.cut:g} sigma, "
          f"spinor {args.spinor} u=({u[0]:.5f}, {u[1]:.5f}) (E of the symbol {E_sym:.5f}); support {cinfo['n_support']} sites, Z={cinfo['Z']:.6e}; the Gaussian's weight "
          f"outside the cut {cinfo['gaussian_weight_outside_cut']:.1e}, below the grid (r < 1/2) {cinfo['gaussian_weight_below_grid']:.3e}, beyond it {cinfo['gaussian_weight_beyond_grid']:.1e}; "
          f"norm {float(np.vdot(psi0f, psi0f).real):.12f}; <h> = {E_mean:.6f}, rms {E_rms:.6f}; unsigned windowed share at folded |q| > pi/4: {ls['lattice_scale']:.4f}", flush=True)
    for nm, kb in (('b1', kb1), ('b3', kb3)):
        bb, th = kb['bins'], kb['threshold_columns']
        print(f"[killing {nm}={kb['edges']['b']:.4f}] below -b {bb['below_minus_b']:.3e}; (-b, -m) {bb['minus_b_to_minus_m']:.4f}; gap (-m, m) {bb['gap']:.4f}; (m, b) {bb['m_to_b']:.4f}; "
              f"above b {bb['above_b']:.3e}; threshold columns (within {args.edge_window:g} of an edge, assigned to neither): -b {th['-b']:.1e}, -m {th['-m']:.1e}, +m {th['+m']:.1e}, +b {th['+b']:.1e}; "
              f"total {kb['total']:.10f} (exact: the dense eigendecomposition, no KPM resolution)", flush=True)

    # ---- the readout times and the states
    tau = args.tau_rs * rs
    nread = int(round(tau / args.dt_read))
    if abs(nread * args.dt_read - tau) > 1e-9:
        raise ValueError("tau must be a multiple of dt_read")
    S = int(args.nsweeps)
    times = args.dt_read * np.arange(0, nread * S + 1)
    sweep_idx = nread * np.arange(0, S + 1)
    sweep_t = times[sweep_idx]
    t0 = time.time()
    states, _ = chain.propagate(psi0f, times)
    flat = states.reshape(len(times), chain.dim)
    norms = np.sum(np.abs(flat) ** 2, 1)
    energies = np.real(np.sum(np.conj(flat) * (flat @ chain.H.T), 1))
    dens = np.sum(np.abs(states) ** 2, 2)
    rad_names = ['r_h'] + [f'{f:g}rs' for f in args.radii_rs]
    radii = np.array([r_h] + [f * rs for f in args.radii_rs], dtype=float)
    p_mid = beyond_weights(r, dens, radii, 'midpoint')
    p_strict = beyond_weights(r, dens, radii, 'strict')
    prop_err = chain.prop_error_bound(times)
    print(f"[propagation] {len(times)} readout times to t={times[-1]:g} a ({times[-1] / rs:g} r_s), {S} sweeps of tau={tau:g} a, dt_read={args.dt_read:g} ({time.time() - t0:.1f}s); "
          f"norm drift max {np.max(np.abs(norms - 1.0)):.1e}; energy drift max {np.max(np.abs(energies - energies[0])):.1e}; "
          f"the stated propagation error at the last sweep end {prop_err[-1]:.2e} (eps_res t + 2 eps_orth (1 + |Lambda|_max t))", flush=True)

    # ---- the fixed-cut current through r_h
    mask = r > r_h
    Q = chain.cut_operator(mask)
    Jc = np.sum(np.conj(flat) * (flat @ Q.T), 1)
    J = np.real(Jc)
    J_imag_max = float(np.max(np.abs(np.imag(Jc))))
    t0 = time.time()
    Jint_sweeps = integrated_current_exact(chain.w, chain.V, Q, c0, sweep_t)
    Jint_all = integrated_current_exact(chain.w, chain.V, Q, c0, times) if args.exact_integral == 'all' else None
    sub = int(args.fine_sub)
    t_fine = (args.dt_read / sub) * np.arange(0, nread * S * sub + 1)
    states_f, _ = chain.propagate(psi0f, t_fine)
    flat_f = states_f.reshape(len(t_fine), chain.dim)
    J_fine = np.real(np.sum(np.conj(flat_f) * (flat_f @ Q.T), 1))
    del states_f, flat_f
    Jint_simpson = cumulative_simpson(J_fine, args.dt_read / sub, sub)
    dp_strict = p_strict[:, 0] - p_strict[0, 0]
    ident_exact = Jint_sweeps - dp_strict[sweep_idx]
    ident_simpson = Jint_simpson - dp_strict
    print(f"[current] J_>(r_h) at t 0 {J[0]:+.4e}; the fixed-cut identity (strict weight change minus the integrated current): exact spectral form at the sweep ends "
          f"max {np.max(np.abs(ident_exact)):.1e} (rounding), Simpson on dt_read/{sub} at every readout time max {np.max(np.abs(ident_simpson)):.1e} (quadrature); "
          f"Im J max {J_imag_max:.1e} ({time.time() - t0:.1f}s)", flush=True)

    # ---- the detectors at 2 r_s, both carrier signs
    r_det = args.det_frac * rs
    N_det, v_det = float(np.interp(r_det, r, N)), float(np.interp(r_det, r, v))
    k_root, roots_k = outgoing_root_chain(E_mean, N_det, v_det, m, args.kappa, r_det, args.kdet_kmax, args.kdet_nscan)
    if k_root is None:
        k_det, matched = abs(args.ak0), False
        note = ("no outgoing root of the chain's local dispersion omega(k) = N E(k) + (v/2) sin 2k at r_det at the column's mean Killing energy (the energy lies below "
                "the outgoing branch's minimum there, or above the band): the detector carries the column's own carrier |a k_0|, not energy matched")
    else:
        k_det, matched = k_root, True
        note = "k_det the smallest root with positive group velocity of omega(k) = N E(k) + (v/2) sin 2k at r_det at the column's mean Killing energy <h>"
    det = detector_packets(r, r_det, args.sigma_det, args.det_cut, k_det, m, args.kappa, args.spinor)
    det_vecs = np.stack([d['vec'] for d in det], 0)
    amp = np.stack([np.einsum('js,tjs->t', np.conj(d['vec']), states) for d in det], 1)
    print(f"[detector] r_det={r_det:g} a (N {N_det:.5f}, v {v_det:+.5f}); roots of the local dispersion at <h>={E_mean:.5f}: {[(round(k, 5), round(vg, 4)) for k, vg in roots_k]}; "
          f"k_det={k_det:+.5f} energy matched: {matched}; width {args.sigma_det:g} a, cut {args.det_cut:g}; "
          + '; '.join(f"{d['label']}: |A| peak {np.max(np.abs(amp[:, i])):.4e} at t={times[int(np.argmax(np.abs(amp[:, i])))]:g} a ({times[int(np.argmax(np.abs(amp[:, i])))] / rs:.2f} r_s)" for i, d in enumerate(det)), flush=True)

    # ---- the sweep-end table
    delta_s = prop_err[sweep_idx]
    band_mid = 2.0 * np.sqrt(np.clip(p_mid[sweep_idx], 0.0, None)) * delta_s[:, None] + delta_s[:, None] ** 2
    band_strict = 2.0 * np.sqrt(np.clip(p_strict[sweep_idx], 0.0, None)) * delta_s[:, None] + delta_s[:, None] ** 2
    sweeps = []
    print(f"[sweeps] t/rs | " + ' | '.join(f"p>{nm} (cell; strict)" for nm in rad_names) + " | J_>(r_h) | int J (exact) | dp_strict(r_h) | prop. error", flush=True)
    for s in range(S + 1):
        i = sweep_idx[s]
        row = dict(s=s, t=float(times[i]), t_rs=float(times[i] / rs), p_mid={nm: float(p_mid[i, k]) for k, nm in enumerate(rad_names)},
                   p_strict={nm: float(p_strict[i, k]) for k, nm in enumerate(rad_names)}, band_mid={nm: float(band_mid[s, k]) for k, nm in enumerate(rad_names)},
                   band_strict={nm: float(band_strict[s, k]) for k, nm in enumerate(rad_names)}, prop_err=float(delta_s[s]), J=float(J[i]), Jint_exact=float(Jint_sweeps[s]),
                   Jint_simpson=float(Jint_simpson[i]), dp_strict_rh=float(dp_strict[i]), identity_defect_exact=float(ident_exact[s]), identity_defect_simpson=float(ident_simpson[i]),
                   norm_minus_1=float(norms[i] - 1.0), energy_drift=float(energies[i] - energies[0]), amp_abs=[float(abs(amp[i, kd])) for kd in range(len(det))])
        sweeps.append(row)
        print(f"[sweeps] {row['t_rs']:4.1f} | " + ' | '.join(f"{row['p_mid'][nm]:.5f}; {row['p_strict'][nm]:.5f}" for nm in rad_names)
              + f" | {row['J']:+.4e} | {row['Jint_exact']:+.6f} | {row['dp_strict_rh']:+.6f} | {row['prop_err']:.1e}", flush=True)
    i15 = rad_names.index('1.5rs') if '1.5rs' in rad_names else 1
    clk = dict(half_weight_1p5rs_sweeps=half_weight_clock(sweep_t, p_mid[sweep_idx, i15]), half_weight_1p5rs_dense=half_weight_clock(times, p_mid[:, i15]))
    clk.update(half_weight_1p5rs_sweeps_rs=clk['half_weight_1p5rs_sweeps'] / rs, half_weight_1p5rs_dense_rs=clk['half_weight_1p5rs_dense'] / rs)
    print(f"[clocks] half-weight crossing of p beyond 1.5 r_s (half its value at the last sweep end): on the sweep ends {clk['half_weight_1p5rs_sweeps_rs']:.2f} r_s, "
          f"on the dense readout {clk['half_weight_1p5rs_dense_rs']:.2f} r_s", flush=True)

    # ---- write
    arrays = dict(r=r, N=N, v=v, psi0=psi0, times=times, sweep_t=sweep_t, sweep_idx=sweep_idx, states=states, eigvals=chain.w, c0=c0, killing_weights=wts,
                  radii=radii, p_mid=p_mid, p_strict=p_strict, J=J, Jint_simpson=Jint_simpson, Jint_exact_sweeps=Jint_sweeps, amp=amp, det_vecs=det_vecs,
                  prop_err=prop_err, norms=norms, energies=energies, fft_q=ls['q'], fft_power=ls['power'], mask_rh=mask, t_fine=t_fine, J_fine=J_fine)
    if Jint_all is not None:
        arrays['Jint_exact_all'] = Jint_all
    np.savez_compressed(base + '.npz', **arrays)
    meta = dict(script='o5_chain_export_v1.py', args=vars(args),
                record=dict(path=args.record, md5=rec_md5, n=int(n), r_first=float(r[0]), r_last=float(r[-1]), N_inf=N_inf, N_min=float(N.min()),
                            roots_on_table=[[a_, b_] for a_, b_ in roots], r_h=float(r_h), kappa_h=float(kappa_h), record_json=rec_json_info),
                spinor_convention=SPINOR_CONVENTION,
                chain=dict(kappa=int(args.kappa), m=m, b1=b1, b3=b3, N_inf=N_inf, edges=edges, dim=int(chain.dim), hermiticity_defect=herm, particle_hole_defect=ph,
                           eigen=dict(eps_res=chain.eps_res, eps_orth=chain.eps_orth, abs_max=chain.wmax, min=float(chain.w[0]), max=float(chain.w[-1]), wall=t_eig)),
                column=dict(Rc=float(args.Rc), sigma=float(args.sigma), ak0=float(args.ak0), cut=float(args.cut), spinor_mode=args.spinor,
                            spinor=[[u[0].real, u[0].imag], [u[1].real, u[1].imag]], E_symbol=float(E_sym), Z=cinfo['Z'], n_support=cinfo['n_support'],
                            gaussian_weight_outside_cut=cinfo['gaussian_weight_outside_cut'], gaussian_weight_below_grid=cinfo['gaussian_weight_below_grid'],
                            gaussian_weight_beyond_grid=cinfo['gaussian_weight_beyond_grid'], E_mean=E_mean, E_rms=E_rms, norm=float(np.vdot(psi0f, psi0f).real)),
                killing=dict(b1=kb1, b3=kb3, three=k3, resolution='exact (dense eigendecomposition)'),
                lattice_scale_weight=ls['lattice_scale'],
                readout=dict(rs=rs, tau=tau, nsweeps=S, dt_read=float(args.dt_read), nread=nread, radii_names=rad_names, radii=[float(x) for x in radii],
                             primary_cut_rule='midpoint', two_gE_tail_last=(rec_json_info or {}).get('two_gE_tail_last')),
                sweeps=sweeps,
                detector=dict(r_det=float(r_det), sigma_det=float(args.sigma_det), cut=float(args.det_cut), omega_target=E_mean, k_det=float(k_det), matched=bool(matched), note=note,
                              roots=[[k, vg] for k, vg in roots_k], N_det=N_det, v_det=v_det,
                              members=[dict(sign=d['sign'], k=d['k'], spinor=d['spinor'], E_symbol=d['E_symbol'], Z=d['Z'], label=d['label'],
                                            peak_abs=float(np.max(np.abs(amp[:, i]))), peak_t=float(times[int(np.argmax(np.abs(amp[:, i])))]),
                                            peak_t_rs=float(times[int(np.argmax(np.abs(amp[:, i])))] / rs)) for i, d in enumerate(det)]),
                clocks=clk,
                propagation_error=dict(eps_res=chain.eps_res, eps_orth=chain.eps_orth, abs_max=chain.wmax, formula='eps_res t + 2 eps_orth (1 + |Lambda|_max t)',
                                       at_sweep_ends=[float(x) for x in delta_s]),
                checks=dict(norm_drift_max=float(np.max(np.abs(norms - 1.0))), energy_drift_max=float(np.max(np.abs(energies - energies[0]))),
                            identity_defect_exact_max=float(np.max(np.abs(ident_exact))), identity_defect_simpson_max=float(np.max(np.abs(ident_simpson))), J_imag_max=J_imag_max),
                files=dict(npz=base + '.npz', json=base + '.json'), wall=time.time() - tw)
    with open(base + '.json', 'w') as fh:
        json.dump(meta, fh, indent=1, default=_json_default)
    print(f"[done] {base}.npz and .json ({time.time() - tw:.0f}s)", flush=True)
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
