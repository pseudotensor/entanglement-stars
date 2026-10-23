#!/usr/bin/env python3
"""obs_raymap_solved_v7.py (builder analyst; engine spec 31.3 ruling (1) and (2), 31.6 BUILDER item 2: "the frozen certified late
exterior means a certified record ..., frozen from its clock on with the stored history before it as the formation, taken at a stationary
phase of the late oscillation: the records in the window where the stored kappa_T's difference changes sign, the last maximum and the last
minimum, both run; the spread of the readings between them is a fifth budget term, the exterior's non-stationarity, attributed to
nothing"). Version 7 = version 6 with the following changes and nothing else.
(1) --freeze-record T/rs: the frozen record chosen by its clock among the stored records (default: the last one, version 6's choice); the
stored history before it is the formation, the records after it are not propagated; the holds count from its clock. The record's place
against S1's certified late window (--certified-window, 28.6 to 39.9 on the production, 30.5; the same window applied to the tau 4 record,
labelled so by --window-note) is printed; a frozen record at the last stored one carries 31.3 (1)'s label "frozen at the uncertified
one-sided last record, near a maximum of kappa_m".
(2) The stationary-phase records of the history read from obs_late_phase_v1.md (--late-phase-md, --late-phase-label: its line
"- stationary phases of <label>: last maximum T/rs x (kappa_T y), last minimum T/rs z (kappa_T w)") and printed first, with the frozen
record's phase (the last maximum, the last minimum, or neither, labelled).
(3) The oscillation term, the fifth budget term (the exterior's non-stationarity, attributed to nothing): with --oscillation-partner-json
(the same history's run frozen at its other extremum) the spread of the readings |T/T_H - T/T_H(other)| at the matched packet and hold
(both above their floors); otherwise the frozen-record level |kappa_T(max) - kappa_T(min)|/kappa_T of the two extrema's stored records.
(4) The quasi-frozen field table on both sides of the frozen record (T - T_freeze = -4 ... +4 r_s where stored records exist); the
observable check as version 6's (the background frozen 1 r_s before the frozen record).
(5) The step's T level is labelled with the partner run's frozen clock and phase (a partner frozen at its own extremum is the matched
phase, not the matched time); the frozen-record level at the matched time from --step-partner-record beside, as version 6.
Outputs J3P_raymap_v7_<history>_<phase>; --from-json rebuilds with --step-partner-json and --oscillation-partner-json. Version 6's
docstring follows.
obs_raymap_solved_v6.py (builder analyst; engine spec 29.4 (J3' version 6, binding), 28.3 (the readouts (a) to (e)), the goal's task 3:
"these readouts are taken on the frozen certified form of the production's late exterior ... with fixed packets (fixed omega r_s and m/omega
across histories and step sizes)"). Version 6 = version 5 with the following changes and nothing else.

(1) The background (29.4): the history's last stored record frozen, its applied site fields a static exterior for every t >= T_end, its
exterior the solved vacuum N = 1, v = -sqrt(2 g E/R) at the record's E beyond the engine's chain (version 5's continuation); the stored history
before T_end is the formation (the probe's sea P_0 at t = 0 carried through it, as in version 5: a frozen stationary state radiates nothing,
GAP03 D5). The surface times are T_surf = T_end + hold, hold in --holds-over-rs (16, 32, 48, 64 nominal r_s), so that every packet reaches
the surface and converts on the frozen record; per row the margin (T_surf - T_end)/s_t, s_t the packet's temporal width, is printed and a
row within 3 s_t of T_end is labelled (its surface passage overlaps the stored history).
(2) Fixed packets (29.4): omega r_s in --omega-rs (0.25, 0.5, 1), m/omega in --m-over-omega (0, 0.25), r_s = --rs-packets (64, the nominal
r_s of all three histories), sigma_omega = omega/6 (version 5's), x_0 from the nominal exterior v = -sqrt(r_s/x) by version 5's rule
x_0 = max(8 r_s, r_s + 4 sigma_x), the erf step's width w_s = --ws-rs/r_s (1/80: kappa_0/40 at kappa_0 = 1/(2 r_s)), the scattering's
outer lead at --rlead-over-rs r_s: the same numbers on every history and step size, never re-tuned to a history's kappa or E (version 5's
packets re-tuned per history are the labelled diagnostic). BUILDER, labelled: the carrier (the outgoing-branch momenta and the local
spinor at x_0) is taken from the frozen record's own fields at x_0, so that w is the outgoing packet of Killing frequency omega on that
background (the frequency content, width, position and mass are the fixed numbers).
(3) The readouts of 28.3. (a) r_H,m and kappa_m of the back-averaged N_bar + v_bar (the frozen record's applied site fields; the outermost
negative-to-positive root by obs_j3p_scope_v2's five-site quadratic fit, the spline's root and slope beside) against the prediction
2 r_s kappa_m = r_H,m/r_s = 1 - (9/4) sigma^2/r_s^2 at the record's own sigma^2 = ell^2 + a^2/12 (a = 1; this lane's Gaussian of standard
deviation ell, 29.0 (1); never the compact kernel's ell^2/15 + a^2/12), r_s = 2 g E of the record; the mass outside the root, 1 - r_H,m/r_s
(2 g M = r on a trapping surface), printed as the non-vacuum share of the exterior at the root (29.4: test (a) is moot where the exterior
at the root is not vacuum). (b) kappa_p, the ray-map exponent of the back-trace psi(tau) = e^{+i h_f tau} w on the frozen operator by
observables sec. 8.1's estimator (the k-route with its window set in x: the symbol's backward ray of the packet's central frequency on the
frozen (N, v) selects the taus with the local rate d_R(N + v) within 10 percent of its surface value, k_c < 0.3 and k_c > 10 |kappa_j|/R_H;
the wave's near-surface part (R < R_b = 1.25 R_H) must there hold a norm stationary within 1 percent, the weight inside R_H below 1 percent
of it and the share at |k| > k_s/2 below 5 percent; the window must span 1.5 e-folds of k_in with 12 samples; kappa_p the window mean of
d ln k_in/d tau, the rate's trend across the window and the wave-to-symbol ratio printed; withheld with the reason otherwise, the symbol's
exponent the control); on the frozen background the chain's time is the Killing time, so kappa_p against the external retarded time equals
the frozen estimator; the two tests kappa_p - d_R(N + v)|_H (the record's dense root R_t and slope kappa_T, the full N + v; the
back-averaged root and slope beside) and 4 g E d_R(N + v)|_H - 1. BUILDER, labelled: in the probe channel (kappa_j = 1) the angular
criterion k_c > 10 |kappa_j|/R_H is 0.29 at R_H about 35 against k_c < 0.3, so its window is empty on these records; the symbol's exponent
of the axial channel (kappa_j = 0, the same packet's frequency, mass and x_0, the criterion void) is printed as the control beside; where the
wave is unresolved the budget's exponent-window term is the larger of the two channels' symbol local-rate spreads over their selectable
windows, labelled. (c) T, sec. 8.1's transmission-weighted fit n_w = sum rho_w Gamma f(omega; T) per hold, against T_H = 1/(8 pi g E) of the frozen record, with the total bias budget (the shift of (a) at the record's sigma, the
exponent window, the step by the tau 4 control at matched times, label A's envelope; linear sum, the quadrature sum beside), no term
attributed to (kappa a)^2, no coefficient fitted (D8); T_conv = omega/ln(Gamma_w/n_w - 1) the narrow-band comparator; T/(kappa_p/2 pi) only
where kappa_p is resolved, T/(kappa_m/2 pi) beside. (d) The stationary conversion on the identical back-averaged profile (the frozen
record's N_bar + v_bar as obs_plateau's radial profile: the site splines imported beyond R_cut and continued below it by the log blend, the
outer saturation at the lead) with the lattice stencil at a and the same scattering at a/4 (the profile, the blend, the lead, the mass,
the frequency and the centrifugal term rescaled by 4: the labelled continuum proxy), at each packet's central frequency for the axial
channel (kappa_j = 0, the endpoint operator of the settled coefficient) and the probe's channel: Gamma, beta^2, theta = T_conv/(kappa/2 pi)
with the blended chain's own kappa, T_conv/T_H; the difference a minus a/4 the only solved-exterior estimate of the lattice term.
(e) Every probe here is light: m/kappa_m printed per packet, a row above 0.4 flagged; all are labelled a ray-map exponent diagnostic (the
primary's m/T_H = 160.85 at E0 = 128 has no propagating light channel; the frozen record's own m/T_H printed beside).
(4) The transmission (BUILDER, labelled): Gamma for the T fit and n_stat by the stationary scattering on the same blended back-averaged
profile with R_cut inside the frozen record's trapped annulus [r_in,m, r_H,m]: cut A at r_in,m + 0.5 (r_H,m - r_in,m), cut B at 0.35 (the
cut check), w = min(4, (r_H,m - R_cut)/6) (version 5's R_t/4 and R_t/2 lie in the untrapped core on the late records, where the inner
lead has no conversion channel: Gamma 3e-6 at its R_t/4 cut); the blend's analyticity estimate e^(-(pi/2)(pi w)) printed.
(5) Printed first in every output: the bias budget and the quasi-frozen check (the frozen record against the history's neighbouring records
at T_end - (0.125, 0.5, 1, 2, 4) r_s: E, the back-averaged and dense roots and slopes, max |Delta(N + v)| near the surface and over the
chain; and n_w at the hold --qf-hold-over-rs with the background frozen at the record T_end - --qf-neighbour-over-rs r_s instead, its
relative change and d theta; version 5's Duhamel drift bound beside). The step term: the partner of the tau pair (--step-partner-record and
--step-partner-fields) at the matched time T_end gives the frozen-record level step (kappa_m and E); with --step-partner-json (the partner's
version 6 output) the T level step |T - T_partner|/T_H at the matched packet and hold. Label A's envelope from clz_ladder_v2.md's summary
line (--labelA-md) or the record's values (E/E0 3.70e-4, R_t 0.341 over T/rs 11 to 21.5; 28.7), read as an indicator, never a bound.
(6) Version 5's end-norm (the back-traced packet's norm inside R_t(T_end) + 5 ell at T_end) is printed as information, no longer a
pass/fail check: on the frozen reading the packet reaches the object inside the frozen stage by design, and the end-norm is the share
still there when the frozen stage begins (the rest was reflected by the barrier or has returned outward through the core, G2). The no-formation
floor n_0 = <w|f(h_0)|w> printed per packet, and observables 8.1's rule applied: T (and its comparators) is withheld where n_0 exceeds
--floor-max (0.03) of n_w, the plateau is read over the frozen-stage holds above the floor (spread below 10 percent of their mean), and a
partner's rows at their floor give no T level step. G2 (version 5's first-passage share) once per packet on the frozen record.
(7) BUILDER, the box: --workers N runs the packets in N forked single-threaded processes (CPU only; export OMP/MKL/OPENBLAS_NUM_THREADS=1;
the GPU path is version 5's sequential loop), so that a full history fits the box when no box GPU is free; the rows are identical.
--from-json J rebuilds the floor rule, the budget, the json and the md from a stored version 6 json J without propagating (the stored
arguments kept; --step-partner-json, --floor-max, --out-json and --md from the command line), e.g. the tau 4 control's budget against the
production's json once both exist.
Every number in the outputs is printed by the script. Args beyond version 5's: --history-label --rs-packets --omega-rs --m-over-omega
--holds-over-rs --ws-rs --rlead-over-rs --cut-fractions --blend-w-max --blend-room --fine --qf-neighbours-over-rs --qf-neighbour-over-rs
--qf-hold-over-rs --step-partner-record/-fields/-label/-json --labelA-md/-E-spread/-Rt-spread --trace-dt-fine --trace-dt-coarse --no-ws-half
--floor-max --from-json --workers. Version 5's docstring follows.

obs_raymap_solved_v5.py (builder analyst; engine spec 28.3, S2 of REVIEWA_REVIEW_web40e: J3' re-scoped to an order-one test of T against
T_H, no coefficient fitted, the light probe a ray-map exponent diagnostic only). Version 5 = version 4 with: (1) the greybody step per
T_surf guarded on a slice with no surface N + v = 0 on the blended chain; (2) the json written after each T_surf; (3) per T_surf T/T_H with
T_H = 1/(8 pi g E) of the record; (4) the header labels. Versions 1 to 4 (the history, the probe, P_0, the Chebyshev propagation, Gamma by
the stationary scattering on the blended slice, theta (fit) and theta (T_conv), G2, the Duhamel estimate, the quasi-frozen hold check) are
described in obs_raymap_solved_v5.py's docstring."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import re
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc, jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as V4  # noqa: E402
import obs_greybody_radial_v1 as GR  # noqa: E402
import obs_plateau_v4 as P4  # noqa: E402
import obs_plateau_v5 as P5  # noqa: E402
import obs_raymap_pg_v2 as RPG  # noqa: E402

BW = 5
SCRIPT = "obs_raymap_solved_v7.py"


# ---------------------------------------------------------------- version 5's propagation (verbatim)
class Diags2:
    """the probe's operator as its component diagonals; apply() takes a vector or a matrix of columns (numpy, or torch on a GPU)."""

    def __init__(self, N, v, m, y, torch=None, dev=None):
        ab = GR.band_matrix_y(np.asarray(v, float), np.asarray(N, float), m, np.asarray(y, float))
        dim = ab.shape[1]
        self.n, self.torch, self.dev = dim, torch, dev
        self.d = []
        g = np.zeros(dim)
        for off in range(-BW, BW + 1):
            dg = ab[BW - off, off:] if off >= 0 else ab[BW - off, :dim + off]
            if np.max(np.abs(dg)) == 0:
                continue
            if off >= 0:
                g[:dim - off] += np.abs(dg)
            else:
                g[-off:] += np.abs(dg)
            self.d.append((off, torch.as_tensor(dg, device=dev) if torch else dg))
        self.bound = float(g.max()) * 1.0001

    def apply(self, X):
        Y = self.torch.zeros_like(X) if self.torch else np.zeros_like(X)
        n = self.n
        two = X.ndim == 2
        for off, D in self.d:
            Dc = D[:, None] if two else D
            if off >= 0:
                Y[:n - off] += Dc * X[off:]
            else:
                Y[-off:] += Dc * X[:n + off]
        return Y


def cheb_expm2(B, X, tau, tol=1e-15):
    """exp(-i tau h) X for a vector or a matrix of columns (tau < 0: the backward step)."""
    if tau == 0:
        return X
    r = B.bound
    a = r * tau
    K = int(abs(a) + 12)
    while abs(jv(K, abs(a))) > tol or K < abs(a) + 4:
        K += 4
    T0, T1 = X, B.apply(X) / r
    Y = jv(0, a) * T0 + 2 * (-1j) * jv(1, a) * T1
    for k in range(2, K + 1):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2
        T0, T1 = T1, T2
    return Y


def cheb_step_coeffs(r, ws, nmax=400000):
    M = 1
    while M < 8 * r / ws:
        M *= 2
    M = min(M, nmax)
    th = (np.arange(M) + 0.5) * math.pi / M
    fx = 0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws))
    from scipy.fft import dct
    c = dct(fx, type=2) / M
    c[0] /= 2.0
    keep = np.nonzero(np.abs(c) > 1e-13)[0]
    K = int(keep[-1]) + 1 if len(keep) else 1
    return c[:K]


def cheb_step_occupation2(B, X, ws):
    """per column: <X| f(h) |X> and ||f(h) X||^2, f(E) = erfc(E/(sqrt 2 ws))/2 (Chebyshev on [-r, r])."""
    c = cheb_step_coeffs(B.bound, ws)
    r = B.bound
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + (c[1] * T1 if len(c) > 1 else 0 * T0)
    for k in range(2, len(c)):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    if B.torch:
        occ = (X.conj() * Y).sum(dim=0).real.cpu().numpy(); occ2 = (Y.conj() * Y).sum(dim=0).real.cpu().numpy()
    else:
        occ = np.real(np.sum(X.conj() * Y, axis=0)); occ2 = np.real(np.sum(Y.conj() * Y, axis=0))
    return np.atleast_1d(occ), np.atleast_1d(occ2), len(c)


def history(record, fields, g, nsites, synthetic=False, fields_key="applied"):
    D = json.load(open(record))
    recs = [r for r in D["records"] if "E" in r]
    Z = np.load(fields, allow_pickle=True)
    Re = np.asarray(Z["R"], float)
    T = np.asarray(Z["T"], float)
    R = np.arange(1, nsites + 1, dtype=float)
    ne = min(len(Re), nsites)
    slices = []
    for k in range(len(T)):
        rec = min(recs, key=lambda q: abs(q["T"] - T[k]))
        E = float(rec["E"])
        N = np.ones(nsites); v = -np.sqrt(2 * g * E / R)
        if fields_key == "applied":
            Nk = np.asarray(Z["N_applied"][k], float); vk = np.asarray(Z["v_applied"][k], float)
        else:
            Nk = np.asarray(Z["N"][k], float); vk = np.asarray(Z["v"][k], float)
        N[:ne] = Nk[:ne]; v[:ne] = vk[:ne]
        jN = abs(Nk[ne - 1] - 1.0); jv_ = abs(vk[ne - 1] + math.sqrt(2 * g * E / Re[ne - 1]))
        slices.append({"T": float(T[k]), "N": N, "v": v, "E": E, "rec": rec, "junction_N": jN, "junction_v": jv_})
    args = D.get("args", {})
    info = {"source": ("synthetic history %s" if synthetic else "collapse record %s") % os.path.basename(record), "fields_key": fields_key, "tau": float(Z["tau"]), "ell": float(Z["ell"]),
            "label": str(Z["label"]), "engine_nsites": int(len(Re)), "rs_nominal": float(args.get("rs", 2 * g * slices[-1]["E"])),
            "junction_max_N": max(s["junction_N"] for s in slices), "junction_max_v": max(s["junction_v"] for s in slices)}
    return R, slices, info


def axial_travel(N, v, R, m, omega, x0, kstar=0.3):
    """the axial symbol's ray at omega from k = kstar near the surface out to x0: (travel time, R_*), or (None, None)."""
    kk = np.linspace(1e-7, math.pi / 2 - 1e-7, 4001)
    Ni = lambda x: float(np.interp(x, R, N)); vi = lambda x: float(np.interp(x, R, v))
    om_k = lambda k, n_, v_: v_ * np.sin(k) * np.cos(k) + n_ * np.sqrt(m * m + np.sin(k) ** 2)
    vg_k = lambda k, n_, v_: v_ * np.cos(2 * k) + n_ * np.sin(k) * np.cos(k) / np.sqrt(m * m + np.sin(k) ** 2)
    xs = np.arange(x0, 1.0, -0.25)
    t, Rstar, kprev = 0.0, None, None
    for i, x in enumerate(xs):
        n_, v_ = Ni(x), vi(x)
        if abs(v_) >= n_:
            break
        om = om_k(kk, n_, v_)
        imax = int(np.argmax(om))
        if om[imax] < omega or om[0] > omega:
            break
        k = brentq(lambda q: om_k(q, n_, v_) - omega, kk[0], kk[imax], xtol=1e-14)
        vg = vg_k(k, n_, v_)
        if vg <= 0:
            break
        if i > 0:
            t += 0.25 / vg
        if k >= kstar:
            Rstar = x
            break
        kprev = k
    return (t, Rstar) if Rstar is not None else (None, kprev)


# ---------------------------------------------------------------- version 6: the frozen record's profile and its readouts
def outer_root(R, f):
    """obs_j3p_scope_v2's: the outermost negative-to-positive sign change of f on R, refined by a quadratic fit through the nearest five
    sites; (root, slope)."""
    s = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(s) == 0:
        return None, None
    i = int(s[-1])
    lo = max(i - 2, 0); hi = min(i + 3, len(R))
    x = R[lo:hi] - R[i]; c = np.polyfit(x, f[lo:hi], 2)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= R[i] - R[i] - 1e-9) & (roots <= R[i + 1] - R[i] + 1e-9)]
    if len(roots) == 0:
        r0 = -f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])
    else:
        r0 = float(roots[0])
    return float(R[i] + r0), float(2 * c[0] * r0 + c[1])


def inner_root(R, f, below):
    """the annulus' inner edge: the outermost positive-to-negative sign change of f below `below` (linear interpolation); None when the
    trapped region reaches the first site."""
    s = np.nonzero((f[:-1] >= 0) & (f[1:] < 0) & (R[1:] <= below))[0]
    if len(s) == 0:
        return None
    i = int(s[-1])
    return float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i]))


class Profile:
    """the frozen record's (N, v) as smooth functions of radius: obs_plateau_v5's dense route on the propagated site fields (the degree-7
    parity splines through the site values) inside the engine's last site - 8, the analytic exterior N = 1, v = -sqrt(2 g E/R) beyond; a
    grid of 1/16 site for the symbol's ray."""

    def __init__(self, Rs, Ns, vs, g, E):
        self.Rs = np.asarray(Rs, float); self.Ns = np.asarray(Ns, float); self.vs = np.asarray(vs, float)
        self.Nf = P5.site_spline(self.Rs, self.Ns, +1, 7); self.vf = P5.site_spline(self.Rs, self.vs, -1, 7)
        self.dNf = self.Nf.derivative(); self.dvf = self.vf.derivative()
        self.rmax = float(self.Rs[-1]) - 8.0
        self.c = 2.0 * g * E
        self.grid = None

    def fn(self, rho):
        rho = np.asarray(rho, float)
        ins = rho < self.rmax
        r_ = np.minimum(rho, self.rmax)
        return (np.where(ins, self.Nf(r_), 1.0), np.where(ins, self.vf(r_), -np.sqrt(self.c / np.maximum(rho, 1e-12))))

    def dfn(self, rho):
        rho = np.asarray(rho, float)
        ins = rho < self.rmax
        r_ = np.minimum(rho, self.rmax)
        return (np.where(ins, self.dNf(r_), 0.0), np.where(ins, self.dvf(r_), 0.5 * np.sqrt(self.c) * np.maximum(rho, 1e-12) ** -1.5))

    def make_grid(self, rmax_grid):
        x = np.arange(0.5, rmax_grid + 1.0, 1.0 / 16)
        N, v = self.fn(x); dN, dv = self.dfn(x)
        self.grid = (x, N, v, dN, dv)

    def at(self, R):
        x, N, v, dN, dv = self.grid
        return (float(np.interp(R, x, N)), float(np.interp(R, x, v)), float(np.interp(R, x, dN)), float(np.interp(R, x, dv)))

    def root(self, near):
        """the spline's outer root of N + v near `near` and the slope d_R(N + v) there."""
        f = lambda r: float(np.sum(self.fn(np.array([r]))))
        lo, hi = near - 2.0, near + 2.0
        if f(lo) * f(hi) > 0:
            return None, None
        rr = brentq(f, lo, hi, xtol=1e-13)
        dN, dv = self.dfn(np.array([rr]))
        return float(rr), float(dN[0] + dv[0])


def record_readout(Rs, Ns, vs, rec, g, ell, T, rs_nominal):
    """(a)'s numbers and the record's dense facts for one stored record."""
    E = float(rec["E"]); rs = 2.0 * g * E
    f = Ns + vs
    rHm, km = outer_root(Rs, f)
    rin = inner_root(Rs, f, rHm) if rHm is not None else None
    sig2 = ell * ell + 1.0 / 12.0
    pred = 1.0 - 2.25 * sig2 / rs ** 2
    out = {"T": float(T), "T_over_rs": float(T) / rs_nominal, "E": E, "r_s": rs, "E_over_E0": rec.get("E_over_E0"), "sigma2": sig2,
           "sigma2_compact_not_used": ell * ell / 15.0 + 1.0 / 12.0, "prediction": pred, "shift_predicted": 1.0 - pred,
           "r_Hm": rHm, "kappa_m": km, "r_in_m": rin, "min_N_plus_v": float(np.min(f)), "R_at_min_N_plus_v": float(Rs[int(np.argmin(f))]),
           "R_t": rec.get("R_t"), "kappa_T": rec.get("kappa_T"), "r_in": rec.get("r_in"), "four_gE_kappa_T": rec.get("four_gE_kappa_T"),
           "R_sh": rec.get("R_sh"), "R_c": rec.get("R_c"), "late_surface": rec.get("late_surface"), "pairs_inside_Rt": rec.get("pairs_inside_Rt"),
           "n_pocket": (sum(rec["n_pocket"].values()) if isinstance(rec.get("n_pocket"), dict) else None), "trapping_zeros": rec.get("trapping_zeros")}
    if rHm is not None:
        out.update({"2rs_kappa_m": 2 * rs * km, "rHm_over_rs": rHm / rs, "mass_outside_root_over_E": 1.0 - rHm / rs,
                    "4gE_kappa_m_minus_1": 4 * g * E * km - 1.0})
    if rec.get("kappa_T") is not None:
        out["4gE_kappa_T_minus_1"] = 4 * g * E * float(rec["kappa_T"]) - 1.0
        if km is not None:
            out["kappa_m_over_kappa_T_minus_1"] = km / float(rec["kappa_T"]) - 1.0
        if rHm is not None and rec.get("R_t"):
            out["rHm_over_R_t"] = rHm / float(rec["R_t"])
    return out


def load_record_fields(fields, record, idx, fields_key):
    Z = np.load(fields, allow_pickle=True)
    D = json.load(open(record))
    recs = [r for r in D["records"] if "E" in r]
    T = np.asarray(Z["T"], float)
    k = idx if idx >= 0 else len(T) + idx
    rec = min(recs, key=lambda q: abs(q["T"] - T[k]))
    kN, kv = ("N_applied", "v_applied") if fields_key == "applied" else ("N", "v")
    return (np.asarray(Z["R"], float), np.asarray(Z[kN][k], float), np.asarray(Z[kv][k], float), rec, float(T[k]), k, T,
            float(Z["ell"]), float(Z["tau"]), D.get("args", {}))


def symbol_ray(prof, m, kj, x0, k0, tmax, h=1.0):
    """the axial symbol H = v sin k cos k + N sqrt(m^2 + (kj/R)^2 + sin^2 k) on the frozen profile, integrated backward (RK4, step h) from
    (x0, k0): dR/dtau = -dH/dk, dk/dtau = +dH/dR; returns the arrays tau, R_c, k_c."""
    def rhs(R, k):
        N, v, dN, dv = prof.at(R)
        y = kj / R
        E = math.sqrt(m * m + y * y + math.sin(k) ** 2)
        Hk = v * math.cos(2 * k) + N * math.sin(k) * math.cos(k) / E
        HR = dv * math.sin(k) * math.cos(k) + dN * E + N * (-(kj * kj) / R ** 3) / E
        return -Hk, HR
    n = int(math.ceil(tmax / h))
    taus = np.zeros(n + 1); Rc = np.zeros(n + 1); kc = np.zeros(n + 1)
    R, k = float(x0), float(k0)
    Rc[0], kc[0] = R, k
    for i in range(n):
        a1 = rhs(R, k); a2 = rhs(R + 0.5 * h * a1[0], k + 0.5 * h * a1[1])
        a3 = rhs(R + 0.5 * h * a2[0], k + 0.5 * h * a2[1]); a4 = rhs(R + h * a3[0], k + h * a3[1])
        R += h * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]) / 6.0
        k += h * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]) / 6.0
        taus[i + 1] = (i + 1) * h; Rc[i + 1] = R; kc[i + 1] = k
        if R < 1.0 or abs(k) > math.pi / 2:
            taus, Rc, kc = taus[:i + 2], Rc[:i + 2], kc[:i + 2]
            break
    return taus, Rc, kc


def surface_ks(omega, NH, m, yH):
    """observables 3.2's k_s with the lapse: the smallest outgoing-branch root of sqrt(m^2 + y_H^2 + sin^2 k) - sin k cos k = omega/N_H at the
    surface (N = -v = N_H), with an outward group velocity; and every root."""
    f = lambda k: math.sqrt(m * m + yH * yH + math.sin(k) ** 2) - math.sin(k) * math.cos(k) - omega / NH
    kg = np.linspace(1e-9, math.pi / 2 - 1e-9, 4001)
    fg = np.array([f(x) for x in kg])
    roots = [brentq(f, kg[i], kg[i + 1], xtol=1e-14) for i in range(len(kg) - 1) if fg[i] * fg[i + 1] < 0]
    vg = lambda k: math.sin(k) * math.cos(k) / math.sqrt(m * m + yH * yH + math.sin(k) ** 2) - math.cos(2 * k)
    outward = [r for r in roots if vg(r) > 0]
    return (min(outward) if outward else None), [{"k": float(r), "vg_surface": float(vg(r))} for r in roots]


def symbol_window(st, Rc, kc, kappa_s, prof, RH, kj):
    """observables 8.1 (i) on the symbol's dense backward ray: the taus with the local rate d_R(N + v)(R_c) within 10 percent of its surface
    value, k_c < 0.3, k_c > 10 |kappa_j|/R_H (and R_c > R_H); the selected run, its e-folds of k_c, the symbol's exponent (the window mean of
    d ln k_c/d tau) and the local rate's largest relative departure over it."""
    rate = np.array([sum(prof.at(r)[2:]) for r in Rc])
    sel = (np.abs(rate / kappa_s - 1.0) < 0.1) & (kc < 0.3) & (kc > 10.0 * abs(kj) / RH) & (kc > 0) & (Rc > RH)
    out = {"kappa_s_surface": kappa_s, "n_symbol_selected": int(sel.sum())}
    if sel.sum() >= 2:
        idx = np.nonzero(sel)[0]
        # the longest contiguous run of the dense ray
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        r_ = max(runs, key=len)
        out["symbol_window_tau"] = [float(st[r_[0]]), float(st[r_[-1]])]
        out["symbol_window_efolds"] = float(math.log(kc[r_[-1]] / kc[r_[0]]))
        out["kappa_symbol"] = float((math.log(kc[r_[-1]]) - math.log(kc[r_[0]])) / (st[r_[-1]] - st[r_[0]])) if st[r_[-1]] > st[r_[0]] else None
        out["symbol_window_rate_spread"] = float(np.max(np.abs(rate[r_] / kappa_s - 1.0)))
        out["symbol_window_R"] = [float(Rc[r_[0]]), float(Rc[r_[-1]])]
    return out


def kroute_estimator(S, kappa_s, prof, RH, kj, ks, symwin):
    """observables 8.1's k-route on the wave samples S (tau; norm_near, norm_inside, k_in, share_high from the wave; R_c, k_c from the symbol)
    inside the symbol's window symwin (symbol_window's)."""
    tau = np.array([s["tau"] for s in S]); kc = np.array([s["k_c"] for s in S])
    out = dict(symwin); out["k_s"] = ks
    if "symbol_window_tau" in symwin:
        t0_, t1_ = symwin["symbol_window_tau"]
        sel_i = (tau >= t0_ - 1e-9) & (tau <= t1_ + 1e-9)
    else:
        sel_i = np.zeros(len(tau), bool)
    nn = np.array([s.get("norm_near", 0.0) for s in S]); ni = np.array([s.get("norm_inside", 0.0) for s in S])
    kin = np.array([s.get("k_in") if s.get("k_in") is not None else np.nan for s in S])
    sh = np.array([s.get("share_high") if s.get("share_high") is not None else 1.0 for s in S])
    ok = sel_i & (ni < 0.01 * np.maximum(nn, 1e-300)) & (sh < 0.05) & np.isfinite(kin) & (kin > 0)
    # the longest contiguous run
    best, cur = (0, -1), None
    for i in range(len(ok)):
        if ok[i]:
            cur = (cur[0], i) if cur else (i, i)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    reasons = []
    if best[1] < best[0]:
        out.update({"resolved": False, "n_window": 0, "n_wave_in_symbol_window": int(sel_i.sum()),
                    "reasons": [("no wave sample inside the symbol's window" if not sel_i.any() else "no wave sample in the symbol's window satisfies the pointwise conditions (inside < 1 percent, |k| > k_s/2 share < 5 percent)")
                                if "symbol_window_tau" in symwin else "the symbol's window is empty (%d dense ray points selected)" % symwin.get("n_symbol_selected", 0)]})
        return out
    i0, i1 = best
    w_t = tau[i0:i1 + 1]; w_k = kin[i0:i1 + 1]; w_n = nn[i0:i1 + 1]
    out["window_tau"] = [float(w_t[0]), float(w_t[-1])]
    out["n_window"] = int(i1 - i0 + 1)
    out["window_efolds_k_in"] = float(math.log(w_k[-1] / w_k[0])) if len(w_k) > 1 else 0.0
    out["window_norm_variation"] = float(np.max(w_n) / np.min(w_n) - 1.0) if np.min(w_n) > 0 else None
    out["wave_to_symbol_max_dev"] = float(np.max(np.abs(w_k / kc[i0:i1 + 1] - 1.0)))
    if len(w_t) >= 2:
        out["kappa_p_window_mean"] = float((math.log(w_k[-1]) - math.log(w_k[0])) / (w_t[-1] - w_t[0]))
    if len(w_t) >= 3:
        out["kappa_p_fit"] = float(np.polyfit(w_t, np.log(w_k), 1)[0])
        dl = np.gradient(np.log(w_k), w_t)
        tr = np.polyfit(w_t, dl, 1)[0] * (w_t[-1] - w_t[0])
        out["rate_trend_over_window"] = float(tr / out["kappa_p_window_mean"]) if out.get("kappa_p_window_mean") else None
    if out["n_window"] < 12:
        reasons.append("%d samples in the window (12 required)" % out["n_window"])
    if out["window_efolds_k_in"] < 1.5:
        reasons.append("%.2f e-folds of k_in (1.5 required)" % out["window_efolds_k_in"])
    if out["window_norm_variation"] is None or out["window_norm_variation"] > 0.01:
        reasons.append("the near-surface norm varies by %s over the window (1 percent required)" % (("%.3f" % out["window_norm_variation"]) if out["window_norm_variation"] is not None else "-"))
    if out["wave_to_symbol_max_dev"] > 0.05:
        reasons.append("the wave-to-symbol ratio departs by %.3f (5 percent accepted)" % out["wave_to_symbol_max_dev"])
    out["resolved"] = not reasons
    out["reasons"] = reasons
    if out["resolved"]:
        out["kappa_p"] = out["kappa_p_window_mean"]
    return out


def blended_profile_chain(prof, s, rs, Rcut, w, RL, outer_w, kappa_ref):
    """obs_plateau_v4's blended chain on the frozen record's back-averaged profile at spacing a/s (s = 1 the lattice stencil, s = 4 the
    continuum proxy): every length in units of a/s; returns (v, N, y-map rho, info) with info's kappa in units of s/a."""
    fn_s = (lambda rho: prof.fn(np.asarray(rho, float) / s))
    Rg, vg_, Ng_, gi = P4.blended_chain(fn_s, s * rs, s * Rcut, s * w, s * RL, outer_w, kappa_ref / s)
    rho = s * Rcut + s * w * np.logaddexp(0.0, (Rg - s * Rcut) / (s * w))
    wL = outer_w * s * RL
    rho = s * RL - wL * np.logaddexp(0.0, (s * RL - rho) / wL)
    return vg_, Ng_, rho, gi


def scatter_on(vg_, Ng_, rho, m, kj, omega, kappa_chain):
    y = kj / rho
    y[:4] = y[4]; y[-4:] = y[-5]
    sc = GR.scatter_y(vg_, Ng_, m, y, float(omega))
    rd = P4.hawking_reading(sc, float(omega), kappa_chain / (2 * math.pi))
    rd["unitarity"] = sc["unitarity"]
    rd.pop("channels", None)
    return rd


def parse_labelA(path, E_spread, Rt_spread):
    out = {"source": "the record's values (engine spec 28.7, clz_ladder_v2's summary)", "E_over_E0_spread": E_spread, "R_t_spread": Rt_spread}
    if path:
        try:
            txt = open(path).read()
            mm = re.search(r"largest spreads over T/rs ([0-9.]+) to ([0-9.]+): E/E0 ([0-9.eE+-]+), R_t ([0-9.eE+-]+), the pairs' energy ([0-9.eE+-]+), N ([0-9.eE+-]+), v ([0-9.eE+-]+)", txt)
            if mm:
                out = {"source": os.path.basename(path), "T_over_rs_range": [float(mm.group(1)), float(mm.group(2))], "E_over_E0_spread": float(mm.group(3)),
                       "R_t_spread": float(mm.group(4)), "pairs_energy_spread": float(mm.group(5)), "N_spread": float(mm.group(6)), "v_spread": float(mm.group(7))}
            else:
                out["source"] += " (%s has no summary line)" % os.path.basename(path)
        except OSError as exc:
            out["source"] += " (%s unreadable: %s)" % (path, exc)
    return out


def phase_of(a, T_over_rs, k_end, k_last, T_last_over_rs):
    """version 7: the frozen record's place: the history's stationary phases read from obs_late_phase_v1.md, its phase among them, its place
    against the certified late window, version 6's label at the last stored record."""
    out = {"source": os.path.basename(a.late_phase_md) if a.late_phase_md else None, "late_phase_label": a.late_phase_label, "T_over_rs": T_over_rs,
           "last_record_T_over_rs": T_last_over_rs, "window": list(a.certified_window), "window_note": a.window_note}
    if a.late_phase_md and a.late_phase_label:
        try:
            txt = open(a.late_phase_md).read()
            mm = re.search(r"^- stationary phases of " + re.escape(a.late_phase_label) +
                           r": last maximum T/rs ([0-9.]+|-) \(kappa_T ([0-9.eE+-]+|-)\), last minimum T/rs ([0-9.]+|-) \(kappa_T ([0-9.eE+-]+|-)\)", txt, re.M)
            if mm:
                fl = lambda x_: None if x_ == "-" else float(x_)
                out.update({"line": mm.group(0)[2:], "last_maximum": fl(mm.group(1)), "kappa_T_max": fl(mm.group(2)), "last_minimum": fl(mm.group(3)), "kappa_T_min": fl(mm.group(4))})
            else:
                out["error"] = "no stationary-phase line for %s in %s" % (a.late_phase_label, out["source"])
        except OSError as exc:
            out["error"] = "%s unreadable: %s" % (a.late_phase_md, exc)
    name = None
    if out.get("last_maximum") is not None and abs(T_over_rs - out["last_maximum"]) < 1e-6:
        name = "the last maximum of the stored kappa_T in the window"
    elif out.get("last_minimum") is not None and abs(T_over_rs - out["last_minimum"]) < 1e-6:
        name = "the last minimum of the stored kappa_T in the window"
    if k_end == k_last:
        name = (name + "; " if name else "") + "frozen at the uncertified one-sided last record, near a maximum of kappa_m (31.3 (1))"
    out["name"] = name or "not a stationary phase of the late-phase table (labelled)"
    lo, hi = a.certified_window
    out["window_status"] = ("inside %s [%g, %g]" % (a.window_note, lo, hi)) if lo - 1e-9 <= T_over_rs <= hi + 1e-9 else ("outside %s [%g, %g]" % (a.window_note, lo, hi))
    return out


def report(a, label, hinfo, FR, neigh, partner, LA, cuts, rsP, nsites, rows, t_start, rebuilt_from=None):
    """the floor rule, the bias budget, the final json and the md, from the computed rows or a stored version 6 json's (--from-json)."""
    T_end = FR["T"]; E_end = FR["E"]; rs = FR["r_s"]; rsn = FR["T"] / FR["T_over_rs"]; RH = FR.get("r_Hm")
    ell = hinfo["ell"]; tau = hinfo["tau"]; kj = a.sign * (a.j + 0.5); g = a.g
    EE0 = FR.get("E_over_E0") or 1.0
    # observables 8.1's no-formation floor: T is withheld where n_0 = <w|f(h_0)|w> exceeds --floor-max of n_w; the plateau over the rest
    for pr in rows:
        if "series" not in pr:
            continue
        n0 = pr.get("floor_n0")
        for s_ in pr["series"]:
            fr_ = (n0 / s_["n_w"]) if (n0 is not None and s_.get("n_w", 0) > 0) else None
            s_["floor_ratio"] = fr_
            s_["T_withheld"] = ("the no-formation floor n_0 is %.3g of n_w (%g allowed, observables 8.1)" % (fr_, a.floor_max)) if (fr_ is not None and fr_ > a.floor_max) else None
        fz = [s_["n_w"] for s_ in pr["series"] if s_["frozen_stage"] and not s_["T_withheld"]]
        sp = (float(np.ptp(fz) / np.mean(fz)) if len(fz) > 1 and np.mean(fz) > 0 else None)
        pr["plateau_reading"] = {"n_holds": len(fz), "spread_over_mean": sp, "plateau": bool(sp is not None and sp < 0.1)}
    partner_rows = None
    if a.step_partner_json:
        try:
            PJ = json.load(open(a.step_partner_json))
            partner_rows = {}
            for pr in PJ.get("packets", []):
                n0p = pr.get("floor_n0")
                for s_ in pr.get("series", []):
                    if s_.get("T") is None or (n0p is not None and s_.get("n_w", 0) > 0 and n0p / s_["n_w"] > a.floor_max):
                        continue
                    partner_rows[(round(pr["omega_rs"], 9), round(pr["m_over_omega"], 9), round(s_["hold_over_rs"], 9))] = (s_["T"], PJ.get("frozen", {}).get("E"))
        except (OSError, ValueError, KeyError) as exc:
            partner_rows = {"error": str(exc)}
    osc_rows, osc_info = None, None
    if getattr(a, "oscillation_partner_json", None):
        try:
            OJ = json.load(open(a.oscillation_partner_json))
            osc_rows = {}
            for pr in OJ.get("packets", []):
                n0p = pr.get("floor_n0")
                for s_ in pr.get("series", []):
                    if s_.get("T_over_T_H") is None or (n0p is not None and s_.get("n_w", 0) > 0 and n0p / s_["n_w"] > a.floor_max):
                        continue
                    osc_rows[(round(pr["omega_rs"], 9), round(pr["m_over_omega"], 9), round(s_["hold_over_rs"], 9))] = s_["T_over_T_H"]
            osc_info = {"file": os.path.basename(a.oscillation_partner_json), "T_over_rs": OJ.get("frozen", {}).get("T_over_rs"),
                        "phase": (OJ.get("frozen", {}).get("phase") or {}).get("name"), "rows": len(osc_rows)}
        except (OSError, ValueError, KeyError) as exc:
            osc_rows, osc_info = {"error": str(exc)}, {"error": str(exc)}
    step_info = None
    if a.step_partner_json and partner_rows is not None and "error" not in partner_rows:
        try:
            PJf = json.load(open(a.step_partner_json)).get("frozen", {})
            step_info = {"T_over_rs": PJf.get("T_over_rs"), "phase": (PJf.get("phase") or {}).get("name")}
        except (OSError, ValueError):
            step_info = None
    ph = FR.get("phase") or {}
    osc_record = None
    if ph.get("kappa_T_max") is not None and ph.get("kappa_T_min") is not None and FR.get("kappa_T"):
        osc_record = abs(ph["kappa_T_max"] - ph["kappa_T_min"]) / FR["kappa_T"]
    budget_rows = []
    # ------------------------------------------------ the bias budget per packet and hold
    shift_a = FR["shift_predicted"]
    for pr in rows:
        if "series" not in pr:
            continue
        kr = ((pr.get("trace") or {}).get("kroute") or {})
        ax = pr.get("axial_symbol") or {}
        sym = [x_ for x_ in (kr.get("symbol_window_rate_spread"), ax.get("symbol_window_rate_spread")) if x_ is not None]
        if kr.get("resolved"):
            win_term, win_lab = abs(kr.get("rate_trend_over_window") or 0.0), "the wave's rate trend across the resolved window"
        elif sym:
            win_term, win_lab = max(sym), "unresolved: the larger of the probe and axial (kappa_j = 0) channels' symbol local-rate spreads over their selectable windows"
        else:
            win_term, win_lab = None, "no window in either channel"
        for s_ in pr["series"]:
            b = {"omega_rs": pr["omega_rs"], "m_over_omega": pr["m_over_omega"], "hold_over_rs": s_["hold_over_rs"], "frozen_stage": s_["frozen_stage"],
                 "T_over_T_H": (None if s_.get("T_withheld") else s_.get("T_over_T_H")), "T_withheld": s_.get("T_withheld"), "floor_ratio": s_.get("floor_ratio"),
                 "shift_a": shift_a, "window": win_term, "window_label": win_lab, "labelA": LA["term"]}
            step, step_lab = None, "none (no tau partner)"
            if partner_rows and isinstance(partner_rows, dict) and "error" not in partner_rows:
                key = (round(pr["omega_rs"], 9), round(pr["m_over_omega"], 9), round(s_["hold_over_rs"], 9))
                if key in partner_rows and s_.get("T") and not s_.get("T_withheld") and partner_rows[key][0]:
                    step = abs(s_["T"] - partner_rows[key][0]) * 8 * math.pi * g * E_end
                    step_lab = "T level: |T - T_partner|/T_H at the matched packet and hold" + (
                        (" (the partner frozen at T/rs %.4f, %s)" % (step_info["T_over_rs"], step_info["phase"] or "its frozen record")) if step_info and step_info.get("T_over_rs") is not None else "")
            if step is None and partner and partner.get("d_kappa_m_rel") is not None:
                step = max(abs(partner["d_kappa_m_rel"]), abs(partner["dE_rel"]))
                step_lab = "frozen-record level: max(|d kappa_m|/kappa_m, |dE|/E) against the partner at the matched time"
            b["step"], b["step_label"] = step, step_lab
            osc, osc_lab = None, "none (no stationary phases read)"
            key = (round(pr["omega_rs"], 9), round(pr["m_over_omega"], 9), round(s_["hold_over_rs"], 9))
            if osc_rows and "error" not in osc_rows and b["T_over_T_H"] is not None and key in osc_rows:
                osc = abs(b["T_over_T_H"] - osc_rows[key])
                osc_lab = "T level: |T/T_H - T/T_H(other extremum)| at the matched packet and hold (the other run frozen at T/rs %s, %s)" % (
                    ("%.4f" % osc_info["T_over_rs"]) if osc_info.get("T_over_rs") is not None else "-", osc_info.get("phase") or "-")
            elif osc_record is not None:
                osc = osc_record
                osc_lab = "frozen-record level: |kappa_T(max) - kappa_T(min)|/kappa_T of the two extrema's stored records"
            b["oscillation"], b["oscillation_label"] = osc, osc_lab
            terms = [b["shift_a"], b["window"], b["step"], b["labelA"], b["oscillation"]]
            b["complete"] = all(t_ is not None for t_ in terms)
            b["total_linear"] = float(sum(t_ for t_ in terms if t_ is not None))
            b["total_quadrature"] = float(math.sqrt(sum(t_ * t_ for t_ in terms if t_ is not None)))
            if b["T_over_T_H"] is not None:
                dev_ = b["T_over_T_H"] - 1.0
                b["T_over_T_H_minus_1"] = dev_
                b["within_budget"] = abs(dev_) <= b["total_linear"]
            budget_rows.append(b)
    J = {"script": SCRIPT, "args": vars(a), "history_label": label, "history": hinfo, "frozen": FR, "neighbours": neigh, "partner": partner, "labelA": LA,
         "cuts": {l_: {k_: v_ for k_, v_ in c_.items() if not k_.startswith("chain_")} for l_, c_ in cuts.items()}, "rs_packets": rsP, "nsites": nsites,
         "packets": rows, "final": True, "budget": budget_rows, "rebuilt_from": rebuilt_from}
    json.dump(J, open(a.out_json, "w"), indent=1, default=float)

    # ------------------------------------------------ the md
    f_ = lambda x, fmt: (fmt % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and np.isfinite(x) else "-"
    L = ["## J3' version 7 (%s): %s frozen at T/rs %s (%s), fixed packets (engine 31.3, 29.4, 28.3)" % (SCRIPT, label, f_(T_end / rsn, "%.4f"), ph.get("name") or "-"), "",
         "- %s; the frozen record T/rs %s (%s), its applied site fields a static exterior from T_end on, its exterior the solved vacuum N = 1, v = -sqrt(2 g E/R) at E %s beyond the engine's %d sites (junction max |N - 1| %s, |v - exterior| %s); the stored history before T_end is the formation; tau %g, ell %g; chain %d sites (%.1f r_s)." % (
             hinfo["source"], f_(T_end / rsn, "%.4f"), hinfo["label"].split(";")[0], f_(E_end, "%.6f"), hinfo["engine_nsites"], f_(hinfo["junction_max_N"], "%.2e"),
             f_(hinfo["junction_max_v"], "%.2e"), tau, ell, nsites, nsites / rsP),
         "- fixed packets: omega r_s in %s, m/omega in %s, r_s = %g (the nominal r_s of the record %g), sigma_omega = omega/%g, the erf step w_s = %s/r_s; the same on every history and step size (version 5's re-tuned packets are the labelled diagnostic); every probe is light and a ray-map exponent diagnostic (28.3 (e)); no (kappa a)^2 coefficient is fitted on a solved exterior (D8)." % (
             a.omega_rs, a.m_over_omega, rsP, rsn, 1.0 / a.sigma_omega_over_omega, f_(a.ws_rs, "%.4g")), ""]
    # version 7: the frozen record and the stationary phases, printed first
    L += ["### The frozen record and the stationary phases (31.3 (1); printed first)", "",
          "- the stationary-phase records of %s (%s): %s" % (ph.get("late_phase_label") or label, ph.get("source") or "no late-phase table given", ph.get("line") or ph.get("error") or "-"),
          "- this run freezes the stored record T/rs %s (index %s of %s stored records; the last stored record T/rs %s): %s; %s." % (
              f_(T_end / rsn, "%.4f"), hinfo.get("frozen_index", "-"), hinfo.get("stored_records", "-"), f_(ph.get("last_record_T_over_rs"), "%.4f"), ph.get("name") or "-", ph.get("window_status") or "-"),
          "- the oscillation term (the fifth budget term, the exterior's non-stationarity, attributed to nothing): %s." % (
              ("the other extremum's run %s (frozen at T/rs %s, %s; %d packet and hold rows above its floor)" % (osc_info["file"], f_(osc_info.get("T_over_rs"), "%.4f"), osc_info.get("phase") or "-", osc_info["rows"]))
              if (osc_info and "error" not in osc_info) else (("the other extremum's run unreadable: %s; " % osc_info["error"]) if osc_info else "") +
              ("the frozen-record level %s (|kappa_T(max) - kappa_T(min)|/kappa_T, the two extrema's stored records %s and %s against this record's %s)" % (
                  f_(osc_record, "%.4e"), f_(ph.get("kappa_T_max"), "%.6e"), f_(ph.get("kappa_T_min"), "%.6e"), f_(FR.get("kappa_T"), "%.6e")) if osc_record is not None else "not available")), ""]
    # the bias budget
    L += ["### The bias budget (28.3 (c), 31.3 (1): five terms; printed first)", "",
          "- the shift of (a): (9/4) sigma^2/r_s^2 = %s at sigma^2 = ell^2 + a^2/12 = %s (this lane's Gaussian; the compact kernel's %s is not used), r_s = 2 g E = %s; %s" % (
              f_(shift_a, "%.4e"), f_(FR["sigma2"], "%.6f"), f_(FR["sigma2_compact_not_used"], "%.6f"), f_(rs, "%.4f"),
              ("the measured 2 r_s kappa_m - 1 = %s, 4 g E d_R(N + v)|_H - 1 = %s (the record's dense root), the mass outside the back-averaged root %s of E (%s)." % (
                  f_(FR["2rs_kappa_m"] - 1.0, "%+.4e"), f_(FR.get("4gE_kappa_T_minus_1"), "%+.4e"), f_(FR["mass_outside_root_over_E"], "%.4f"),
                  "larger than the shift: the exterior at the root is not vacuum and (a) is moot, 29.4" if FR["mass_outside_root_over_E"] > shift_a else "below the shift"))
              if RH is not None else ("no back-averaged root on the propagated fields: (a) has no reading on this record; 4 g E d_R(N + v)|_H - 1 = %s at the record's dense root." % f_(FR.get("4gE_kappa_T_minus_1"), "%+.4e"))),
          "- label A's envelope (%s): E/E0 spread %s (dE/E %s at E/E0 %s), R_t spread %s (relative to the frozen R_t %s: %s); the term %s (an indicator of the convention's step sensitivity, never a bound; 28.7)." % (
              LA["source"], f_(LA["E_over_E0_spread"], "%.3e"), f_(LA["dE_over_E"], "%.3e"), f_(EE0, "%.6f"), f_(LA["R_t_spread"], "%.3e"), f_(FR.get("R_t"), "%.4f"),
              f_(LA["dR_t_over_R_t"], "%.3e"), f_(LA["term"], "%.3e"))]
    if partner:
        L.append("- the step (%s, %s, tau %g, ell %g) at T/rs %s (matched to %s): E %s (relative %s), back-averaged root %s, kappa_m %s (relative %s), record R_t %s, kappa_T %s (relative %s)." % (
            partner["label"], partner["record"], partner["tau"], partner["ell"], f_(partner["T_over_rs"], "%.4f"), f_(partner["matched_dT"], "%+.3f"), f_(partner["E"], "%.6f"),
            f_(partner["dE_rel"], "%+.4e"), f_(partner.get("r_Hm"), "%.4f"), f_(partner.get("kappa_m"), "%.6e"), f_(partner.get("d_kappa_m_rel"), "%+.4e"), f_(partner.get("R_t"), "%.4f"),
            f_(partner.get("kappa_T"), "%.6e"), f_(partner.get("d_kappa_T_rel"), "%+.4e")))
    else:
        L.append("- the step: no tau partner given for this history (the term is absent and the budget incomplete).")
    if partner_rows is not None:
        L.append("- the partner's version 6 or 7 output: %s" % (("%s (%d packet and hold rows with a T reading above its floor%s)" % (os.path.basename(a.step_partner_json), len(partner_rows),
                 ("; frozen at T/rs %.4f, %s" % (step_info["T_over_rs"], step_info["phase"] or "its frozen record")) if step_info and step_info.get("T_over_rs") is not None else "")) if "error" not in partner_rows else ("unreadable: %s" % partner_rows["error"])))
    L += ["", "| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H | T/T_H - 1 | shift (a) | exponent window | step | label A | oscillation | total (linear; quadrature) | complete | within |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in budget_rows:
        L.append("| %g | %g | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s | %s |" % (b["omega_rs"], b["m_over_omega"], b["hold_over_rs"], "yes" if b["frozen_stage"] else "no (within 3 s_t)",
                 ("withheld (floor %.2f of n_w)" % b["floor_ratio"]) if b.get("T_withheld") else f_(b["T_over_T_H"], "%.5f"), f_(b.get("T_over_T_H_minus_1"), "%+.4e"), f_(b["shift_a"], "%.3e"), f_(b["window"], "%.3e"), f_(b["step"], "%.3e"), f_(b["labelA"], "%.3e"),
                 f_(b["oscillation"], "%.3e"), f_(b["total_linear"], "%.3e"), f_(b["total_quadrature"], "%.3e"), "yes" if b["complete"] else "no", ("yes" if b.get("within_budget") else "no") if "within_budget" in b else "-"))
    lab_w = sorted({b["window_label"] for b in budget_rows}); lab_s = sorted({b["step_label"] for b in budget_rows}); lab_o = sorted({b["oscillation_label"] for b in budget_rows})
    L += ["", "- the exponent window term: %s; the step term: %s; the oscillation term: %s (attributed to nothing, 31.3 (1)); T is withheld where the no-formation floor n_0 exceeds %g of n_w (observables 8.1); a residual outside the budget is not attributed (no term is assigned to (kappa a)^2)." % (
        "; ".join(lab_w), "; ".join(lab_s), "; ".join(lab_o), a.floor_max), ""]
    # the quasi-frozen check
    L += ["### The quasi-frozen check (the frozen record against the history's neighbouring records; printed first)", "",
          "| neighbour T/rs | T_freeze - T (r_s) | E | dE/E | r_H,m | d r_H,m | kappa_m | d kappa_m/kappa_m | r_in,m | R_t (record) | d R_t | kappa_T (record) | d kappa_T/kappa_T | max abs d(N + v) near the surface | over the chain |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L.append("| %s (frozen) | 0 | %s | 0 | %s | 0 | %s | 0 | %s | %s | 0 | %s | 0 | 0 | 0 |" % (f_(T_end / rsn, "%.4f"), f_(E_end, "%.6f"), f_(RH, "%.4f"), f_(FR.get("kappa_m"), "%.6e"),
             f_(FR.get("r_in_m"), "%.3f"), f_(FR.get("R_t"), "%.4f"), f_(FR.get("kappa_T"), "%.6e")))
    for rn in neigh:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (f_(rn["T_over_rs"], "%.4f"), f_(rn["dT_over_rs"], "%.3f"), f_(rn["E"], "%.6f"), f_(rn["dE_over_E"], "%+.3e"),
                 f_(rn.get("r_Hm"), "%.4f"), f_(rn.get("d_rHm"), "%+.4f"), f_(rn.get("kappa_m"), "%.6e"), f_(rn.get("d_kappa_m_rel"), "%+.3e"), f_(rn.get("r_in_m"), "%.3f"),
                 f_(rn.get("R_t"), "%.4f"), f_(rn.get("d_R_t"), "%+.4f"), f_(rn.get("kappa_T"), "%.6e"), f_(rn.get("d_kappa_T_rel"), "%+.3e"), f_(rn.get("max_abs_d_Nv_surface"), "%.3e"),
                 f_(rn.get("max_abs_d_Nv_chain"), "%.3e")))
    L += ["", "| omega r_s | m/omega | hold/r_s | background frozen at T/rs | n_w (frozen record) | n_w (neighbour frozen) | relative change | d theta | Duhamel B at that hold (bound on abs dn_w) |", "|---|---|---|---|---|---|---|---|---|"]
    for pr in rows:
        q = pr.get("qf_check")
        if not q:
            continue
        sh = pr["series"][a.holds_over_rs.index(a.qf_hold_over_rs)]
        L.append("| %g | %g | %g | %s | %s | %s | %s | %s | %s (%s) |" % (pr["omega_rs"], pr["m_over_omega"], q["hold_over_rs"], f_(q["neighbour_T_over_rs"], "%.4f"), f_(q["n_w_frozen_record"], "%.6e"),
                 f_(q["n_w_neighbour"], "%.6e"), f_(q["rel_diff"], "%.3e"), f_(q.get("dtheta"), "%.2e"), f_(sh.get("duhamel_B"), "%.2e"), f_(sh.get("duhamel_dn_bound"), "%.2e")))
    L += ["", "- the neighbouring records' fields are the same key (%s) as the propagation's; d theta = theta^2 (dn/n)/(2 pi (omega/kappa_m)(1 - n/Gamma)); the Duhamel B is version 5's continued-drift bound (the last %g r_s' field change extrapolated linearly over the frozen stage)." % (
        a.fields_key, a.drift_window_over_rs), ""]
    # (a)
    L += ["### (a) The back-averaged root and slope against the prediction at the record's own sigma", "",
          "| T/rs | E | r_s = 2gE | sigma^2 | r_H,m | kappa_m | 2 r_s kappa_m | r_H,m/r_s | prediction | r_H,m (spline), slope | r_in,m | mass outside the root / E | R_t (dense) | r_H,m/R_t | kappa_m/kappa_T - 1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rr in [FR] + neigh:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s | %s | %s | %s | %s | %s |" % (f_(rr["T_over_rs"], "%.4f"), f_(rr["E"], "%.6f"), f_(rr["r_s"], "%.4f"), f_(rr["sigma2"], "%.4f"),
                 f_(rr.get("r_Hm"), "%.4f"), f_(rr.get("kappa_m"), "%.6e"), f_(rr.get("2rs_kappa_m"), "%.6f"), f_(rr.get("rHm_over_rs"), "%.6f"), f_(rr["prediction"], "%.6f"),
                 f_(rr.get("r_Hm_spline"), "%.4f"), f_(rr.get("kappa_m_spline"), "%.6e"), f_(rr.get("r_in_m"), "%.3f"), f_(rr.get("mass_outside_root_over_E"), "%.4f"), f_(rr.get("R_t"), "%.4f"),
                 f_(rr.get("rHm_over_R_t"), "%.6f"), f_(rr.get("kappa_m_over_kappa_T_minus_1"), "%+.4e")))
    if RH is None:
        L.append("")
        L.append("- the frozen record's propagated fields (%s) have no trapping root: (a), (b)'s exponent, the stationary transmission and (d) are not defined on it; the record's dense trapping zeros %s." % (a.fields_key, FR.get("trapping_zeros")))
    L.append("")
    # (b)
    L += ["### (b) The ray-map exponent on the frozen operator and the two tests", "",
          "| omega r_s | m/omega | k_s | symbol window (taus) | symbol e-folds | kappa (symbol) | wave window samples | wave e-folds | norm variation | wave/symbol max dev | kappa_p | resolved | kappa_p - kappa_T (rel) | kappa_p - kappa_m (rel) | 4gE kappa_p - 1 | axial symbol: window (taus), e-folds, rate spread | kappa (axial symbol) | its - kappa_T (rel) | its - kappa_m (rel) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pr in rows:
        if "series" not in pr:
            continue
        kr = ((pr.get("trace") or {}).get("kroute") or {})
        kpp = kr.get("kappa_p")
        ax = pr.get("axial_symbol") or {}
        kax = ax.get("kappa_symbol")
        L.append("| %g | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (pr["omega_rs"], pr["m_over_omega"], f_((pr.get("trace") or {}).get("k_s"), "%.4f"),
                 ("%.0f to %.0f" % tuple(kr["symbol_window_tau"])) if kr.get("symbol_window_tau") else "-", f_(kr.get("symbol_window_efolds"), "%.3f"), f_(kr.get("kappa_symbol"), "%.6e"),
                 kr.get("n_window", "-"), f_(kr.get("window_efolds_k_in"), "%.3f"), f_(kr.get("window_norm_variation"), "%.3e"), f_(kr.get("wave_to_symbol_max_dev"), "%.3e"),
                 f_(kr.get("kappa_p_window_mean"), "%.6e"), ("yes" if kr.get("resolved") else ("no: " + "; ".join(kr.get("reasons", [])))) if kr else "-",
                 f_((kpp / FR["kappa_T"] - 1.0) if (kpp and FR.get("kappa_T")) else None, "%+.4e"), f_((kpp / FR["kappa_m"] - 1.0) if (kpp and FR.get("kappa_m")) else None, "%+.4e"),
                 f_((4 * g * E_end * kpp - 1.0) if kpp else None, "%+.4e"),
                 (("%.0f to %.0f, %s, %s" % (ax["symbol_window_tau"][0], ax["symbol_window_tau"][1], f_(ax.get("symbol_window_efolds"), "%.3f"), f_(ax.get("symbol_window_rate_spread"), "%.3e")))
                  if ax.get("symbol_window_tau") else ("none (%d points)" % ax.get("n_symbol_selected", 0) if ax else "-")),
                 f_(kax, "%.6e"), f_((kax / FR["kappa_T"] - 1.0) if (kax and FR.get("kappa_T")) else None, "%+.4e"),
                 f_((kax / FR["kappa_m"] - 1.0) if (kax and FR.get("kappa_m")) else None, "%+.4e")))
    L += ["", "- the record's dense R_H = R_t %s, d_R(N + v)|_H = kappa_T %s, 4 g E d_R(N + v)|_H - 1 = %s (the full N + v); the back-averaged r_H,m %s, kappa_m %s, 4 g E kappa_m - 1 = %s; N at r_H,m %s; the wave column \"kappa_p\" is the window mean of d ln k_in/d tau, a reading only where resolved (observables 8.1's rules); the symbol's exponent is the control, in the probe channel and in the axial channel (kappa_j = 0, where 8.1's angular criterion k_c > 10 |kappa_j|/R_H is void; in the probe channel it is %s against k_c < 0.3)." % (
        f_(FR.get("R_t"), "%.4f"), f_(FR.get("kappa_T"), "%.6e"), f_(FR.get("4gE_kappa_T_minus_1"), "%+.4e"), f_(RH, "%.4f"), f_(FR.get("kappa_m"), "%.6e"), f_(FR.get("4gE_kappa_m_minus_1"), "%+.4e"),
        f_(FR.get("N_at_r_Hm"), "%.6f"), f_((10.0 * abs(kj) / RH) if RH else None, "%.3f")), ""]
    # (c)
    L += ["### (c) T against 1/(8 pi g E) on the frozen record (T_H %s; cut A's Gamma the reading, cut B's the check)" % f_(1.0 / (8 * math.pi * g * E_end), "%.6e"), "",
          "| omega r_s | m/omega | hold/r_s | T_obs/r_s | margin (s_t) | n_w | n_w (w_s/2) | norm at T_0 | end-norm inside R_t + 5 ell | Gamma_w (A) | n_stat (A) | n at T_H | T/T_H (A; B) | T_conv/T_H | T/(kappa_m/2 pi) | T/(kappa_p/2 pi) | T/(kappa_T/2 pi) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pr in rows:
        if "series" not in pr:
            L.append("| %g | %g | %s | | | | | | | | | | | | | | |" % (pr["omega_rs"], pr["m_over_omega"], pr.get("skipped") or ("failed: " + pr.get("error", "")[-200:].replace("\n", " "))))
            continue
        for s_ in pr["series"]:
            L.append("| %g | %g | %g | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s | %s | %s | %s |" % (pr["omega_rs"], pr["m_over_omega"], s_["hold_over_rs"], s_["T_obs_over_rs"],
                     f_(s_["frozen_margin_s_t"], "%.2f"), f_(s_["n_w"], "%.6e"), f_(s_.get("n_w_ws_half"), "%.6e"), f_(s_["norm_at_T0"], "%.10f"), f_(s_.get("endnorm_inside_Rt_plus_5ell"), "%.3e"),
                     f_(s_.get("Gamma_w"), "%.6f"), f_(s_.get("n_stat"), "%.6e"), f_(s_.get("n_at_T_H"), "%.6e"),
                     ("withheld (floor %.2f of n_w)" % s_["floor_ratio"]) if s_.get("T_withheld") else f_(s_.get("T_over_T_H"), "%.5f"),
                     "-" if s_.get("T_withheld") else f_(s_.get("T_over_T_H_cutB"), "%.5f"), *(["-"] * 4 if s_.get("T_withheld") else
                     [f_(s_.get("T_conv_over_T_H"), "%.5f"), f_(s_.get("theta_m"), "%.5f"), f_(s_.get("theta_p"), "%.5f"), f_(s_.get("theta_T"), "%.5f")])))
        pl = pr["plateau_reading"]
        L.append("| | | log slope of n_w per r_s %s | floor n_0 %s (%s of the largest n_w) | frozen-stage holds above the floor %d, spread/mean %s, plateau %s | | | | | | | | | | | | |" % (
            f_(pr.get("log_slope_per_rs"), "%.3e"), f_(pr["floor_n0"], "%.3e"), f_(pr.get("floor_over_n_w_max"), "%.2e"), pl["n_holds"], f_(pl["spread_over_mean"], "%.3e"), "yes" if pl["plateau"] else "no"))
    L += ["", "- T is sec. 8.1's transmission-weighted fit n_w = sum rho_w Gamma f(omega; T) over the packet's spectral measure with cut A's Gamma (cut B's T/T_H beside), withheld where the no-formation floor n_0 exceeds %g of n_w; the plateau is read over the frozen-stage holds above the floor (spread below 10 percent of the mean, 8.1);" % a.floor_max + " T_conv = omega/ln(Gamma_w/n_w - 1) the narrow-band comparator; n_stat the stationary in-state's occupation on the same scattering; n at T_H the occupation the measured Gamma would give at T = T_H; the end-norm is information: the share of the back-traced packet still at the object (inside R_t + 5 ell) when the frozen stage begins, the rest reflected by the barrier or returned outward through the core (G2).", ""]
    # the cuts and (d)
    L += ["### (d) The lattice endpoint operator against its continuum proxy (a/%d) on the identical back-averaged profile" % a.fine, ""]
    for lab, cu in cuts.items():
        L.append("- cut %s: R_cut %s (r_in,m + %g (r_H,m - r_in,m)), w %s, |v|/N there %s (%s), analyticity estimate %s; the blended chain's root and kappa at a: %s, %s; at a/%d: %s, %s; %s." % (
            lab, f_(cu["R_cut"], "%.3f"), cu["fraction"], f_(cu["w"], "%.3f"), f_(cu["absv_over_N_at_cut"], "%.4f"), "trapped" if cu["trapped"] else "NOT trapped", f_(cu["analyticity_estimate"], "%.1e"),
            f_(cu.get("R_h_chain_1"), "%.4f"), f_(cu.get("kappa_chain_1"), "%.6e"), a.fine, f_(cu.get("R_h_chain_%d" % a.fine), "%.4f"), f_(cu.get("kappa_chain_%d" % a.fine), "%.6e"),
            ("error: " + cu["error"]) if "error" in cu else "the chain sites %s and %s" % (cu.get("nsites_chain_1"), cu.get("nsites_chain_%d" % a.fine))))
    L += ["", "| omega r_s | m/omega | cut | kappa_j | Gamma (a; a/%d) | beta^2 (a; a/%d) | theta (a) | theta (a/%d) | theta difference | T_conv/T_H (a; a/%d) | T_conv/T_H difference | unitarity (a; a/%d) |" % ((a.fine,) * 5),
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pr in rows:
        for er in (pr.get("endpoint") or []):
            e1, e4 = er["a/1"], er["a/%d" % a.fine]
            L.append("| %g | %g | %s | %g | %s; %s | %s; %s | %s | %s | %s | %s; %s | %s | %s; %s |" % (pr["omega_rs"], pr["m_over_omega"], er["cut"], er["kappa_j"], f_(e1.get("Gamma"), "%.6f"), f_(e4.get("Gamma"), "%.6f"),
                     f_(e1.get("beta2"), "%.4e"), f_(e4.get("beta2"), "%.4e"), f_(e1.get("theta"), "%.6f"), f_(e4.get("theta"), "%.6f"), f_(er.get("theta_a_minus_fine"), "%+.3e"),
                     f_(e1.get("T_conv_over_T_H"), "%.6f"), f_(e4.get("T_conv_over_T_H"), "%.6f"), f_(er.get("Tconv_over_TH_a_minus_fine"), "%+.3e"), f_(e1.get("unitarity"), "%.1e"), f_(e4.get("unitarity"), "%.1e")))
    L += ["", "- theta = T_conv/(kappa/2 pi) with each chain's own kappa (the blended chain's d_R(N + v) at its root); the a/%d chain is the same profile, blend, lead, mass, frequency and centrifugal term on %d times the sites (the labelled continuum proxy, which keeps 1/%d of a (kappa a)^2 term); the difference is the only solved-exterior estimate of the lattice term; nothing is fitted." % (
        a.fine, a.fine, a.fine * a.fine), ""]
    # (e) and the packets
    L += ["### (e) The light probes and the fixed packets", "",
          "| omega r_s | m/omega | omega | m | omega/T_H (record; nominal) | m/kappa_m | m/kappa_0 | light-probe label | x_0/r_s | s_x/r_s | s_t/r_s | k_0 | N, v at x_0 | Gaussian tail beyond the chain | below-threshold measure | G2 (min, max; plateau) | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pr in rows:
        if "series" not in pr:
            continue
        g2 = (pr.get("trace") or {}).get("G2") or {}
        L.append("| %g | %g | %.6e | %.6e | %s; %s | %s | %s | %s | %.2f | %.2f | %.2f | %.5f | %.6f, %.6f | %.1e | %.1e | %s | %.0f |" % (pr["omega_rs"], pr["m_over_omega"], pr["omega"], pr["m"],
                 f_(pr["omega_over_T_H"], "%.3f"), f_(pr["omega_over_T_H_nominal"], "%.3f"), f_(pr.get("m_over_kappa_m"), "%.3f"), f_(pr["m_over_kappa_0"], "%.3f"),
                 "a ray-map exponent diagnostic" + (" (m/kappa above 0.4)" if (pr.get("m_over_kappa_m") or 0) > 0.4 else ""), pr["x0_over_rs"], pr["s_x_density"] / rsP, pr["s_t_over_rs"], pr["k0"],
                 pr["N_x0"], pr["v_x0"], pr["gaussian_tail_beyond_chain"], pr["below_threshold_measure"],
                 ("%.4f, %.4f; %s" % (g2["share_min"], g2["share_max"], f_(g2.get("Gamma"), "%.4f"))) if g2 else "-", pr["seconds"]))
    L += ["", "- the primary's m/T_H = 160.85 (m a 0.2 at E0 = 128) has no propagating light channel; the frozen record's m/T_H = %s (m a %g); every row above is a light probe and a ray-map exponent diagnostic only; x_0 from the nominal exterior at r_s %g (version 5's rule), the carrier from the frozen record's fields at x_0." % (
        f_(FR["m_engine_over_T_H"], "%.2f"), FR["m_engine"], rsP),
          "- the frozen record: R_sh %s, R_c %s, the late-surface flag %s, the pairs' energy inside R_t %s, pockets %s, trapping zeros %s; every number in this file is printed by the script (%s)." % (
              f_(FR.get("R_sh"), "%.1f"), f_(FR.get("R_c"), "%.3f"), FR.get("late_surface"), f_(FR.get("pairs_inside_Rt"), "%.4e"), FR.get("n_pocket"), FR.get("trapping_zeros"),
              ("rebuilt by --from-json from %s; the packets' own seconds %s" % (os.path.basename(rebuilt_from), ", ".join("%.0f" % pr["seconds"] for pr in rows if "seconds" in pr)))
              if rebuilt_from else ("%.0f s" % (time.time() - t_start)))]
    open(a.md, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--synthetic", action="store_true", help="the record/fields are a synthetic history in the engine's layout (the control)")
    ap.add_argument("--history-label", default=None)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--j", type=float, default=0.5)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--rs-packets", type=float, default=64.0, help="the fixed packets' r_s (lattice units; the nominal r_s of the three histories)")
    ap.add_argument("--omega-rs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--m-over-omega", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--sigma-omega-over-omega", type=float, default=1.0 / 6.0)
    ap.add_argument("--holds-over-rs", type=float, nargs="+", default=[16.0, 32.0, 48.0, 64.0], help="T_surf = T_end + hold (nominal r_s)")
    ap.add_argument("--rend-over-rs", type=float, default=128.0)
    ap.add_argument("--ws-rs", type=float, default=1.0 / 80.0, help="the erf step's width times r_s (fixed)")
    ap.add_argument("--no-ws-half", action="store_true", help="skip the w_s/2 control (tests only)")
    ap.add_argument("--kstar", type=float, default=0.3)
    ap.add_argument("--gamma-ngrid", type=int, default=24)
    ap.add_argument("--rlead-over-rs", type=float, default=64.0)
    ap.add_argument("--cut-fractions", type=float, nargs=2, default=[0.5, 0.35], help="R_cut = r_in,m + f (r_H,m - r_in,m): cut A, cut B")
    ap.add_argument("--blend-w-max", type=float, default=4.0)
    ap.add_argument("--blend-room", type=float, default=6.0, help="w = min(w_max, (r_H,m - R_cut)/room)")
    ap.add_argument("--fine", type=int, default=4, help="the continuum proxy's refinement a/fine")
    ap.add_argument("--qf-neighbours-over-rs", type=float, nargs="+", default=[0.125, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--qf-neighbour-over-rs", type=float, default=1.0)
    ap.add_argument("--qf-hold-over-rs", type=float, default=32.0)
    ap.add_argument("--drift-window-over-rs", type=float, default=4.0, help="the Duhamel estimate's rate window before T_end (nominal r_s)")
    ap.add_argument("--duhamel-dt-over-rs", type=float, default=1.0)
    ap.add_argument("--no-fp", action="store_true", help="skip the frozen back-trace (kappa_p and G2)")
    ap.add_argument("--trace-dt-fine", type=float, default=8.0, help="the back-trace's sampling near the surface (lattice time)")
    ap.add_argument("--trace-dt-coarse", type=float, default=32.0)
    ap.add_argument("--step-partner-record", default=None)
    ap.add_argument("--step-partner-fields", default=None)
    ap.add_argument("--step-partner-label", default="the tau pair's partner")
    ap.add_argument("--step-partner-json", default=None)
    ap.add_argument("--labelA-md", default=None)
    ap.add_argument("--labelA-E-spread", type=float, default=3.70e-4)
    ap.add_argument("--labelA-Rt-spread", type=float, default=0.341)
    ap.add_argument("--freeze-record", type=float, default=None, help="version 7: the frozen record's T/rs (default the last stored record)")
    ap.add_argument("--late-phase-md", default=None, help="version 7: obs_late_phase_v1.md")
    ap.add_argument("--late-phase-label", default=None, help="version 7: the history's label in obs_late_phase_v1.md")
    ap.add_argument("--certified-window", type=float, nargs=2, default=[28.6, 39.9], help="version 7: S1's certified late window (T/rs)")
    ap.add_argument("--window-note", default="S1's certified late window (30.5)")
    ap.add_argument("--oscillation-partner-json", default=None, help="version 7: the same history's run frozen at the other extremum")
    ap.add_argument("--floor-max", type=float, default=0.03, help="observables 8.1: T withheld where n_0 exceeds this share of n_w")
    ap.add_argument("--from-json", default=None, help="rebuild the budget, the json and the md from a stored version 6 json (no propagation)")
    ap.add_argument("--workers", type=int, default=1, help="CPU only: packets in parallel forked processes (export OMP/MKL/OPENBLAS_NUM_THREADS=1)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fields-key", default="applied", choices=["applied", "stored"], help="engine spec 19.2: the applied (averaged) site fields")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    if a.from_json:
        J = json.load(open(a.from_json))
        A = vars(ap.parse_args(["--out-json", a.out_json, "--md", a.md]))
        A.update(J.get("args", {}))
        A.update({"out_json": a.out_json, "md": a.md, "floor_max": a.floor_max, "from_json": a.from_json,
                  "step_partner_json": a.step_partner_json or A.get("step_partner_json"),
                  "oscillation_partner_json": a.oscillation_partner_json or A.get("oscillation_partner_json")})
        report(argparse.Namespace(**A), J["history_label"], J["history"], J["frozen"], J["neighbours"], J.get("partner"), J["labelA"], J.get("cuts", {}),
               J["rs_packets"], J["nsites"], J["packets"], time.time(), rebuilt_from=a.from_json)
        print("rebuilt %s and %s from %s" % (a.md, a.out_json, a.from_json), flush=True)
        return
    torch = dev = None
    if a.gpu is not None:
        import torch as _t
        torch, dev = _t, _t.device("cuda:%d" % a.gpu)
    t_start = time.time()
    g = a.g
    kj = a.sign * (a.j + 0.5)
    rsP = a.rs_packets
    label = a.history_label or os.path.basename(a.record).replace("collapse_", "").replace(".json", "")

    # ------------------------------------------------ the frozen record, its neighbours, the partner, label A
    _, _, _, _, _, k_last, Tall, _, _, rargs = load_record_fields(a.fields, a.record, -1, a.fields_key)
    rsn = float(rargs.get("rs", rsP))
    k_freeze = k_last
    if a.freeze_record is not None:
        k_freeze = int(np.argmin(np.abs(Tall / rsn - a.freeze_record)))
        if abs(Tall[k_freeze] / rsn - a.freeze_record) > 1e-6:
            raise SystemExit("no stored record at T/rs %g (the nearest is %.6f)" % (a.freeze_record, Tall[k_freeze] / rsn))
    Rs, Ns, vs, rec_end, T_end, k_end, Tall, ell, tau_h, rargs = load_record_fields(a.fields, a.record, k_freeze, a.fields_key)
    E_end = float(rec_end["E"]); rs = 2 * g * E_end
    FR = record_readout(Rs, Ns, vs, rec_end, g, ell, T_end, rsn)
    prof = Profile(Rs, Ns, vs, g, E_end)
    RH = FR["r_Hm"]
    if RH is not None:
        rr_, ks_ = prof.root(RH)
        FR["r_Hm_spline"], FR["kappa_m_spline"] = rr_, ks_
        FR["N_at_r_Hm"] = float(prof.fn(np.array([RH]))[0][0])
    kappa_m = FR.get("kappa_m_spline") or FR.get("kappa_m")
    kappa_ref = kappa_m or 1.0 / (2 * rsP)
    FR["phase"] = phase_of(a, T_end / rsn, k_end, k_last, Tall[k_last] / rsn)
    FR["m_engine"] = float(rargs.get("m", 0.2))
    FR["m_engine_over_T_H"] = FR["m_engine"] * 8 * math.pi * g * E_end
    print("frozen record of %s: T/rs %.4f (index %d of %d), E %.6f, 2gE %.4f; back-averaged root %s slope %s; record R_t %s kappa_T %s; prediction %.6f (sigma^2 %.6f)" % (
        label, T_end / rsn, k_end, len(Tall), E_end, rs, RH, FR.get("kappa_m"), FR.get("R_t"), FR.get("kappa_T"), FR["prediction"], FR["sigma2"]), flush=True)
    neigh = []
    dts = sorted({d_ for d_ in a.qf_neighbours_over_rs} | {-d_ for d_ in a.qf_neighbours_over_rs}, reverse=True)
    for d_ in dts:
        kn = int(np.argmin(np.abs(Tall - (T_end - d_ * rsn))))
        if kn == k_end or abs(Tall[kn] - (T_end - d_ * rsn)) > 0.51 * abs(d_) * rsn:
            continue
        Rn, Nn, vn, recn, Tn, kn, _, _, _, _ = load_record_fields(a.fields, a.record, kn, a.fields_key)
        rn = record_readout(Rn, Nn, vn, recn, g, ell, Tn, rsn)
        dfull = np.abs((Nn + vn) - (Ns + vs))
        near = (Rs >= (FR.get("r_in_m") or 1.0) - 2 * ell) & (Rs <= (RH or rs) + 5 * ell)
        rn.update({"index": kn, "dT_over_rs": (T_end - Tn) / rsn, "max_abs_d_Nv_chain": float(dfull.max()), "max_abs_d_Nv_surface": float(dfull[near].max()) if near.any() else None,
                   "dE_over_E": rn["E"] / E_end - 1.0,
                   "d_rHm": (rn["r_Hm"] - RH) if (rn.get("r_Hm") is not None and RH is not None) else None,
                   "d_kappa_m_rel": (rn["kappa_m"] / FR["kappa_m"] - 1.0) if (rn.get("kappa_m") and FR.get("kappa_m")) else None,
                   "d_kappa_T_rel": (rn["kappa_T"] / FR["kappa_T"] - 1.0) if (rn.get("kappa_T") and FR.get("kappa_T")) else None,
                   "d_R_t": (rn["R_t"] - FR["R_t"]) if (rn.get("R_t") is not None and FR.get("R_t") is not None) else None})
        neigh.append(rn)
    partner = None
    if a.step_partner_record and a.step_partner_fields:
        Zp_T = np.asarray(np.load(a.step_partner_fields, allow_pickle=True)["T"], float)
        kp = int(np.argmin(np.abs(Zp_T - T_end)))
        Rp, Np_, vp_, recp, Tp, kp, _, ellp, taup, _ = load_record_fields(a.step_partner_fields, a.step_partner_record, kp, a.fields_key)
        partner = record_readout(Rp, Np_, vp_, recp, g, ellp, Tp, rsn)
        partner.update({"label": a.step_partner_label, "record": os.path.basename(a.step_partner_record), "tau": taup, "ell": ellp, "matched_dT": Tp - T_end,
                        "d_kappa_m_rel": (partner["kappa_m"] / FR["kappa_m"] - 1.0) if (partner.get("kappa_m") and FR.get("kappa_m")) else None,
                        "d_kappa_T_rel": (partner["kappa_T"] / FR["kappa_T"] - 1.0) if (partner.get("kappa_T") and FR.get("kappa_T")) else None,
                        "dE_rel": partner["E"] / E_end - 1.0})
    LA = parse_labelA(a.labelA_md, a.labelA_E_spread, a.labelA_Rt_spread)
    EE0 = FR.get("E_over_E0") or 1.0
    LA["dE_over_E"] = LA["E_over_E0_spread"] / EE0
    LA["dR_t_over_R_t"] = (LA["R_t_spread"] / FR["R_t"]) if FR.get("R_t") else None
    LA["term"] = max(LA["dE_over_E"], LA["dR_t_over_R_t"] or 0.0)

    # ------------------------------------------------ the fixed packets
    packs = []
    for mo in a.m_over_omega:
        for wr in a.omega_rs:
            omega = wr / rsP; m = mo * omega; sg = a.sigma_omega_over_omega * omega
            below = 0.5 * erfc((omega - m) / (math.sqrt(2) * sg)) if m > 0 else 0.0
            if m > 0 and omega - 3 * sg < m * (1 - 1e-9):
                packs.append({"omega_rs": wr, "m_over_omega": mo, "skipped": "the packet's 3 sigma window reaches below the threshold m"})
                continue
            x0 = 8 * rsP
            for _ in range(8):
                vext = -math.sqrt(rsP / x0); me = math.sqrt(m * m + (kj / x0) ** 2)
                k_n, om_n, a2_n = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
                vg_n = abs(float(np.sum(a2_n * V4.group_velocity(k_n, me, vext) * np.gradient(k_n))))
                x1 = max(8 * rsP, rsP + 4 * vg_n / sg)
                if abs(x1 - x0) < 1e-9:
                    break
                x0 = x1
            N0, v0 = (float(q[0]) for q in prof.fn(np.array([x0])))
            me = math.sqrt(m * m + (kj / x0) ** 2)
            k, om_s, a2 = V4.branch_momenta(omega / N0, sg / N0, me, v0 / N0, outgoing=True)
            om = N0 * om_s
            vg0 = abs(N0 * float(np.sum(a2 * V4.group_velocity(k, me, v0 / N0) * np.gradient(k))))
            packs.append({"omega_rs": wr, "m_over_omega": mo, "m": m, "omega": omega, "sigma_omega": sg, "x0": x0, "x0_over_rs": x0 / rsP,
                          "vg_nominal": vg_n, "sigma_x_spec": vg0 / sg, "s_x_density": vg0 / (2 * sg), "vg_far": vg0, "N_x0": N0, "v_x0": v0,
                          "k": k, "om": om, "a2": a2, "me": me, "k0": float(np.sum(a2 * k * np.gradient(k))), "below_threshold_measure": below,
                          "m_over_kappa_m": (m / kappa_m) if kappa_m else None, "m_over_kappa_0": m * 2 * rsP, "omega_over_kappa_m": (omega / kappa_m) if kappa_m else None,
                          "omega_over_T_H": omega * 8 * math.pi * g * E_end, "omega_over_T_H_nominal": omega * 4 * math.pi * rsP})
    Rend = max([a.rend_over_rs * rsP] + [max(p["x0"] + 4 * p["sigma_x_spec"], p["x0"] + 5 * p["s_x_density"]) + 8 * rsP for p in packs if "x0" in p])
    nsites = int(math.ceil(Rend / 512.0)) * 512
    for p in packs:
        if "x0" in p:
            p["gaussian_tail_beyond_chain"] = 0.5 * erfc((nsites - p["x0"]) / (math.sqrt(2.0) * p["s_x_density"]))
            p["s_t"] = p["s_x_density"] / p["vg_far"]
            p["s_t_over_rs"] = p["s_t"] / rsn
    R, slices, hinfo = history(a.record, a.fields, g, nsites, a.synthetic, a.fields_key)
    hinfo.update({"frozen_index": k_end, "stored_records": len(slices), "frozen_T_over_rs": T_end / rsn})
    slices = slices[:k_end + 1]
    y = kj / R
    times = np.array([s_["T"] for s_ in slices])
    assert abs(times[-1] - T_end) < 1e-9, (times[-1], T_end)
    tau = hinfo["tau"]
    prof.make_grid(nsites + 8)
    print("history %s: %d slices to T_end/r_s %.2f (nominal r_s %g; frozen 2gE %.4f; kappa_m %s); chain %d sites (%.1f r_s); junction max |N-1| %.2e |v - ext| %.2e; packets on r_s %g" % (
        hinfo["source"], len(slices), T_end / rsn, rsn, rs, kappa_m, nsites, nsites / rsP, hinfo["junction_max_N"], hinfo["junction_max_v"], rsP), flush=True)
    ops = {}

    def op(k, m, held=None):
        kk = held if (held is not None and k >= held) else k
        key = (kk, m)
        if key not in ops:
            if len(ops) > 48:
                ops.clear()
            s_ = slices[kk]
            ops[key] = Diags2(s_["N"], s_["v"], m, y, torch, dev)
        return ops[key]

    def slice_at(t):
        """the slice governing time t (held over [T_k, T_k + tau))."""
        return max(i for i, Ti in enumerate(times) if Ti <= t + 1e-9) if t >= times[0] else 0

    def slice_below(t):
        """the slice governing the interval just below t."""
        c = [i for i, Ti in enumerate(times) if Ti < t - 1e-9]
        return max(c) if c else 0

    # version 3's drift operator for the Duhamel estimate (the last stored interval's rate, extrapolated linearly)
    k_rate = slice_at(T_end - a.drift_window_over_rs * rsn)
    dt_rate = T_end - float(times[k_rate])
    dN_rate = slices[-1]["N"] - slices[k_rate]["N"]; dv_rate = slices[-1]["v"] - slices[k_rate]["v"]
    k_qf = slice_at(T_end - a.qf_neighbour_over_rs * rsn)

    def colnorms(Y):
        return ((Y.conj() * Y).real.sum(dim=0).sqrt().cpu().numpy()) if torch else np.sqrt(np.sum(np.abs(Y) ** 2, axis=0))

    def back_propagate(w, Tobs_list, m, held=None, check_end=True):
        """version 5's: columns for each T_obs (descending insertion): phi_j = U(T_obs_j)^dagger w; the per-column norm inside
        R_t(T_end) + 5 ell at T_end and the per-column Duhamel estimate."""
        order = np.argsort(-np.asarray(Tobs_list))
        cols = []
        t = float(max(Tobs_list))
        X = None
        endnorm = [None] * len(Tobs_list)
        duh = [None] * len(Tobs_list)
        last = {}
        Rt_end = slices[-1]["rec"].get("R_t") or rs
        inner = np.repeat(R < Rt_end + 5 * hinfo["ell"], 2)
        inner_t = torch.as_tensor(inner, device=dev) if torch else inner
        pending = [int(j) for j in order]
        wt = torch.as_tensor(w, device=dev) if torch else w
        Dop = Diags2(dN_rate, dv_rate, m, y, torch, dev) if (check_end and held is None and dt_rate > 0) else None
        dsub = a.duhamel_dt_over_rs * rsn

        def duhamel_sample(tt):
            if Dop is None or X is None or tt < T_end - 1e-9:
                return
            f = colnorms(Dop.apply(X)) * (tt - T_end) / dt_rate
            for c, j in enumerate(cols):
                if Tobs_list[j] <= T_end + 1e-9:
                    continue
                if j in last:
                    t0_, f0_ = last[j]
                    duh[j] = (duh[j] or 0.0) + 0.5 * (f0_ + float(f[c])) * (t0_ - tt)
                else:
                    duh[j] = 0.0
                last[j] = (tt, float(f[c]))
        while True:
            added = False
            while pending and Tobs_list[pending[0]] >= t - 1e-9:
                j = pending.pop(0)
                col = wt.clone() if torch else wt.copy()
                X = col[:, None] if X is None else (torch.cat([X, col[:, None]], dim=1) if torch else np.concatenate([X, col[:, None]], axis=1))
                cols.append(j)
                added = True
            if added:
                duhamel_sample(t)
            if t <= 1e-9:
                break
            k = slice_below(t)
            t_lo = float(times[k]) if t > times[0] + 1e-9 else 0.0
            nxt = [t_lo]
            if pending:
                nxt.append(Tobs_list[pending[0]])
            if check_end and t > T_end + 1e-9:
                nxt.append(T_end)
                if Dop is not None:
                    nxt.append(max(T_end, t - dsub))
            t_next = max(x for x in nxt if x < t - 1e-9) if any(x < t - 1e-9 for x in nxt) else 0.0
            if X is not None:
                X = cheb_expm2(op(k, m, held), X, -(t - t_next))
            t = t_next
            if t >= T_end - 1e-9:
                duhamel_sample(t)
            if check_end and abs(t - T_end) < 1e-9 and X is not None:
                nin = ((X.conj() * X).real * inner_t[:, None]).sum(dim=0).cpu().numpy() if torch else np.sum(np.abs(X) ** 2 * inner[:, None], axis=0)
                for c, j in enumerate(cols):
                    if Tobs_list[j] > T_end + 1e-9 and endnorm[j] is None:
                        endnorm[j] = float(nin[c])
            if t <= 1e-9 and not pending:
                break
        return X, cols, endnorm, duh

    # ------------------------------------------------ the cuts inside the frozen record's trapped annulus
    cuts = {}
    if RH is not None:
        rin = FR.get("r_in_m") or 1.0
        for lab, fr in (("A", a.cut_fractions[0]), ("B", a.cut_fractions[1])):
            Rc_ = rin + fr * (RH - rin)
            w_ = min(a.blend_w_max, (RH - Rc_) / a.blend_room)
            Nc_, vc_ = (float(q[0]) for q in prof.fn(np.array([Rc_])))
            cuts[lab] = {"R_cut": Rc_, "w": w_, "fraction": fr, "N_plus_v_at_cut": Nc_ + vc_, "absv_over_N_at_cut": abs(vc_) / Nc_, "trapped": (Nc_ + vc_) < 0,
                         "analyticity_estimate": math.exp(-0.5 * math.pi * math.pi * w_)}
            try:
                for s in (1, a.fine):
                    vg_, Ng_, rho, gi = blended_profile_chain(prof, s, rs, Rc_, w_, a.rlead_over_rs * rsP, 0.05, kappa_ref)
                    cuts[lab]["chain_%d" % s] = (vg_, Ng_, rho, gi)
                    cuts[lab]["kappa_chain_%d" % s] = gi["kappa"] * s
                    cuts[lab]["R_h_chain_%d" % s] = gi["R_h"] / s
                    cuts[lab]["nsites_chain_%d" % s] = gi["nsites"]
            except RuntimeError as exc:
                cuts[lab]["error"] = str(exc)
        print("cuts: %s" % "; ".join("%s R_cut %.3f w %.3f |v|/N %.4f kappa(a) %s kappa(a/%d) %s" % (l_, c_["R_cut"], c_["w"], c_["absv_over_N_at_cut"], c_.get("kappa_chain_1"), a.fine,
                                                                                   c_.get("kappa_chain_%d" % a.fine)) for l_, c_ in cuts.items()), flush=True)

    def dump(rows, final=False):
        J = {"script": SCRIPT, "args": vars(a), "history_label": label, "history": hinfo, "frozen": FR, "neighbours": neigh, "partner": partner,
             "labelA": LA, "cuts": {l_: {k_: v_ for k_, v_ in c_.items() if not k_.startswith("chain_")} for l_, c_ in cuts.items()},
             "rs_packets": rsP, "nsites": nsites, "packets": rows, "final": final}
        if final:
            J["budget"] = budget_rows
        json.dump(J, open(a.out_json, "w"), indent=1, default=float)

    budget_rows = []
    rows = []
    hold_ts = [T_end + h_ * rsn for h_ in a.holds_over_rs]

    def run_packet(p):
        """one fixed packet: its occupations, the quasi-frozen check, the stationary scattering and (d), T, the frozen back-trace."""
        if "skipped" in p:
            return p
        m, omega = p["m"], p["omega"]
        t_p = time.time()
        w = V4.build_packet(nsites, int(round(p["x0"] - 1)), p["k"], p["a2"], p["me"], m=m, a=kj / p["x0"])
        rho_w = p["a2"] * np.gradient(p["k"]); rho_w = rho_w / rho_w.sum()
        series, Tobs = [], []
        trav, Rstar = axial_travel(slices[-1]["N"], slices[-1]["v"], R, m, omega, p["x0"], a.kstar)
        if trav is None:
            trav = (p["x0"] - (RH or rs)) / p["vg_far"]
            Rstar = None
        for hh, T_surf in zip(a.holds_over_rs, hold_ts):
            Tobs.append(T_surf + trav)
            series.append({"hold_over_rs": hh, "T_surf": T_surf, "T_surf_over_rs": T_surf / rsn, "travel": trav, "R_star": Rstar, "T_obs": T_surf + trav,
                           "T_obs_over_rs": (T_surf + trav) / rsn, "frozen_margin_s_t": (T_surf - T_end) / p["s_t"],
                           "frozen_stage": (T_surf - T_end) >= 3 * p["s_t"]})
        X, cols, endnorm, duh = back_propagate(w, Tobs, m)
        B0 = Diags2(slices[0]["N"], slices[0]["v"], m, y, torch, dev)
        wt0 = torch.as_tensor(w, device=dev)[:, None] if torch else w[:, None]
        Xa = torch.cat([X, wt0], dim=1) if torch else np.concatenate([X, wt0], axis=1)
        ws = a.ws_rs / rsP
        occ, occ2, Kd = cheb_step_occupation2(B0, Xa, ws)
        n0 = float(occ[-1])
        occh = cheb_step_occupation2(B0, X, 0.5 * ws)[0] if not a.no_ws_half else [None] * len(cols)
        nrm = ((X.conj() * X).real.sum(dim=0).cpu().numpy()) if torch else np.sum(np.abs(X) ** 2, axis=0)
        for c, j in enumerate(cols):
            s_ = series[j]
            s_.update({"n_w": float(occ[c]), "n_w_fsq": float(occ2[c]), "n_w_ws_half": (float(occh[c]) if occh[c] is not None else None), "norm_at_T0": float(nrm[c]),
                       "cheb_degree": Kd, "endnorm_inside_Rt_plus_5ell": endnorm[j], "duhamel_B": duh[j],
                       "duhamel_dn_bound": (2 * duh[j] + duh[j] ** 2) if duh[j] is not None else None})
        print("omega r_s %g m/omega %g: n_w %s (floor n_0 %.3e; %.0f s)" % (p["omega_rs"], p["m_over_omega"], ", ".join("%.4e" % s_["n_w"] for s_ in series), n0, time.time() - t_p), flush=True)
        # the quasi-frozen check at the observable level: the background frozen at the neighbouring record
        qf = None
        if a.qf_hold_over_rs in a.holds_over_rs:
            jq = a.holds_over_rs.index(a.qf_hold_over_rs)
            Xh, _, _, _ = back_propagate(w, [series[jq]["T_obs"]], m, held=k_qf, check_end=False)
            oh = cheb_step_occupation2(B0, Xh, ws)[0]
            qf = {"hold_over_rs": a.qf_hold_over_rs, "neighbour_T_over_rs": float(times[k_qf]) / rsn, "n_w_frozen_record": series[jq]["n_w"], "n_w_neighbour": float(oh[0]),
                  "rel_diff": float(abs(oh[0] - series[jq]["n_w"]) / max(abs(series[jq]["n_w"]), 1e-300))}
            print("  quasi-frozen check: hold %g r_s with the background frozen at T/rs %.3f: n_w %.6e against %.6e (relative %.2e)" % (
                a.qf_hold_over_rs, qf["neighbour_T_over_rs"], qf["n_w_neighbour"], qf["n_w_frozen_record"], qf["rel_diff"]), flush=True)
        # the stationary scattering on the frozen back-averaged profile: Gamma and n_stat (cut A the reading, cut B the check), and (d)
        stat, endpoint = {}, []
        om_grid = p["om"][:: max(1, len(p["om"]) // a.gamma_ngrid)]
        for lab, cu in cuts.items():
            if "error" in cu or "chain_1" not in cu:
                stat[lab] = {"error": cu.get("error", "no chain")}
                continue
            vg_, Ng_, rho, gi = cu["chain_1"]
            G, NH = [], []
            for o in om_grid:
                rd = scatter_on(vg_, Ng_, rho, m, kj, o, gi["kappa"])
                G.append(rd.get("Gamma", np.nan)); NH.append(rd.get("n_H", np.nan))
            Gi = np.interp(p["om"], om_grid, np.array(G)); Ni_ = np.interp(p["om"], om_grid, np.array(NH))
            stat[lab] = {"Gamma_w": float(np.sum(rho_w * Gi)), "n_stat": float(np.sum(rho_w * Ni_)), "Gi": Gi}
            for kjj in sorted({0.0, kj}):
                er = {"cut": lab, "kappa_j": kjj}
                for s in (1, a.fine):
                    vgs, Ngs, rhos, gis = cu["chain_%d" % s]
                    rd = scatter_on(vgs, Ngs, rhos, m / s, kjj, omega / s, gis["kappa"])
                    er["a/%d" % s] = {"Gamma": rd.get("Gamma"), "beta2": rd.get("beta2"), "n_H": rd.get("n_H"), "theta": rd.get("theta"),
                                      "T_conv_over_T_H": (rd["T_conv"] * s * 8 * math.pi * g * E_end) if rd.get("T_conv") else None, "unitarity": rd.get("unitarity"),
                                      "kappa_chain": gis["kappa"] * s, "error": rd.get("error")}
                t1, t4 = er["a/1"].get("theta"), er["a/%d" % a.fine].get("theta")
                er["theta_a_minus_fine"] = (t1 - t4) if (t1 is not None and t4 is not None) else None
                c1, c4 = er["a/1"].get("T_conv_over_T_H"), er["a/%d" % a.fine].get("T_conv_over_T_H")
                er["Tconv_over_TH_a_minus_fine"] = (c1 - c4) if (c1 is not None and c4 is not None) else None
                endpoint.append(er)
        # T per hold (cut A's Gamma the reading, cut B's the check)
        for s_ in series:
            nw = s_["n_w"]
            for lab in ("A", "B"):
                st = stat.get(lab)
                if not st or "Gi" not in st:
                    continue
                Gi = st["Gi"]
                fT = lambda T: float(np.sum(rho_w * Gi / (np.exp(np.clip(p["om"] / T, -700, 700)) + 1.0))) - nw
                try:
                    Tf = brentq(fT, 1e-6 * kappa_ref, 100 * kappa_ref, xtol=1e-16)
                except ValueError:
                    Tf = None
                key = "" if lab == "A" else "_cutB"
                s_["T" + key] = Tf
                s_["T_over_T_H" + key] = (Tf * 8 * math.pi * g * E_end) if Tf else None
                if lab == "A":
                    Gw = st["Gamma_w"]
                    s_["Gamma_w"] = Gw; s_["n_stat"] = st["n_stat"]
                    s_["T_conv_over_T_H"] = ((omega / math.log(Gw / nw - 1.0)) * 8 * math.pi * g * E_end) if (nw > 0 and Gw / nw > 2.0) else None
                    s_["theta_m"] = (Tf / (kappa_m / (2 * math.pi))) if (Tf and kappa_m) else None
                    s_["theta_T"] = (Tf / (FR["kappa_T"] / (2 * math.pi))) if (Tf and FR.get("kappa_T")) else None
                    s_["n_at_T_H"] = float(np.sum(rho_w * Gi / (np.exp(np.clip(p["om"] * 8 * math.pi * g * E_end, -700, 700)) + 1.0)))
        # the axial channel's symbol exponent (kappa_j = 0: 8.1's angular criterion void; the probe channel's k_c > 10 |kappa_j|/R_H meets
        # k_c < 0.3 at R_H about 35), a labelled control computed from the symbol's backward ray alone
        axial = None
        if RH is not None and kappa_m:
            k_ax, _, a2_ax = V4.branch_momenta(omega / p["N_x0"], p["sigma_omega"] / p["N_x0"], m, p["v_x0"] / p["N_x0"], outgoing=True)
            k0_ax = float(np.sum(a2_ax * k_ax * np.gradient(k_ax)))
            tmax_ax = 2.0 * (p["x0"] + 5 * p["s_x_density"]) / max(p["vg_far"], 0.2)
            sa_, Ra_, ka_ = symbol_ray(prof, m, 0.0, p["x0"], k0_ax, tmax_ax)
            axial = symbol_window(sa_, Ra_, ka_, kappa_m, prof, RH, 0.0)
            axial["k0"] = k0_ax
        # the frozen back-trace: kappa_p (observables 8.1) and G2 (version 5's first-passage share)
        trace = None
        if not a.no_fp:
            Lfp = p["x0"] + 5 * p["s_x_density"]
            tfp = 2.0 * Lfp / max(p["vg_far"], 0.2)
            nfp = int(math.ceil((Lfp + tfp + 8 * rsP) / 512.0)) * 512
            Rf = np.arange(1, nfp + 1, dtype=float)
            Nf = np.ones(nfp); vf = -np.sqrt(2 * g * E_end / Rf)
            nl = min(nsites, nfp)
            Nf[:nl] = slices[-1]["N"][:nl]; vf[:nl] = slices[-1]["v"][:nl]
            if nfp > nsites:
                prof.make_grid(nfp + 8)
            Bf = Diags2(Nf, vf, m, kj / Rf, torch, dev)
            wf = V4.build_packet(nfp, int(round(p["x0"] - 1)), p["k"], p["a2"], p["me"], m=m, a=kj / p["x0"])
            tr = {"t_fp": tfp, "chain": nfp}
            ks_s, roots_s = (None, [])
            if RH is not None:
                NH_ = FR.get("N_at_r_Hm") or 1.0
                ks_s, roots_s = surface_ks(omega, NH_, m, kj / RH)
            tr["k_s"], tr["k_s_roots"] = ks_s, roots_s
            # the symbol's backward ray (the window selection and the control)
            st_, Rc_, kc_ = symbol_ray(prof, m, kj, p["x0"], p["k0"], tfp)
            # the fine sampling around the near-surface passage
            if RH is not None:
                near_i = np.nonzero(Rc_ < 1.5 * RH)[0]
                conv_i = np.nonzero(np.abs(kc_) > 0.6)[0]
                ta = float(st_[near_i[0]]) - 2 * rsP if len(near_i) else tfp
                tb = (float(st_[conv_i[0]]) + 2 * rsP) if len(conv_i) else (ta + 24 * rsP)
            else:
                ta, tb = tfp, tfp
            samp = list(np.arange(a.trace_dt_coarse, max(ta, a.trace_dt_coarse), a.trace_dt_coarse)) + list(np.arange(max(ta, 0.0), min(tb, tfp), a.trace_dt_fine)) + \
                list(np.arange(min(tb, tfp), tfp + 1e-9, a.trace_dt_coarse))
            samp = sorted(set(float(round(t_, 6)) for t_ in samp if t_ > 0))
            Rb = 1.25 * RH if RH is not None else None
            Rt_rec = slices[-1]["rec"].get("R_t") or rs
            outside = Rf > Rt_rec + 2
            nf_out = 1
            while nf_out < 2 * nfp:
                nf_out *= 2
            kq_out = 2 * np.pi * np.fft.fftfreq(nf_out)
            nf_in = 16384
            kq_in = 2 * np.pi * np.fft.fftfreq(nf_in)
            X1 = torch.as_tensor(wf, device=dev) if torch else wf
            t_now = 0.0
            S, share = [], []
            for t_s in samp:
                X1 = cheb_expm2(Bf, X1, -(t_s - t_now))
                t_now = t_s
                ph = X1.cpu().numpy() if torch else X1
                psi = ph.reshape(-1, 2)
                row = {"tau": t_s}
                po = psi.copy(); po[~outside] = 0.0
                pk = (np.abs(np.fft.fft(po, n=nf_out, axis=0)) ** 2).sum(axis=1)
                tot = float(pk.sum()); out_norm = float((np.abs(po) ** 2).sum())
                row["share_out_kneg"] = out_norm * float(pk[kq_out < 0].sum()) / tot if tot > 0 else 0.0
                share.append(row["share_out_kneg"])
                if Rb is not None:
                    nb = int(min(math.ceil(Rb), nfp))
                    pn = psi[:nb]
                    rho_n = (np.abs(pn) ** 2).sum(axis=1)
                    row["norm_near"] = float(rho_n.sum())
                    row["norm_inside"] = float(rho_n[Rf[:nb] < RH].sum())
                    if row["norm_near"] > 1e-30:
                        row["x_in"] = float((rho_n * Rf[:nb]).sum() / rho_n.sum())
                        pkn = (np.abs(np.fft.fft(pn, n=nf_in, axis=0)) ** 2).sum(axis=1)
                        pos = np.abs(kq_in) > 0
                        row["k_in"] = float(np.exp((pkn[pos] * np.log(np.abs(kq_in[pos]))).sum() / pkn[pos].sum()))
                        row["share_high"] = float(pkn[np.abs(kq_in) > 0.5 * ks_s].sum() / pkn.sum()) if ks_s else None
                ii = int(np.argmin(np.abs(st_ - t_s)))
                row["R_c"], row["k_c"] = float(Rc_[ii]), float(kc_[ii])
                S.append(row)
            win = RPG.plateau_window(np.array([s_["tau"] for s_ in S]), np.array(share), rsP)
            tr["G2"] = {"series": [[S[i]["tau"] / rsn, S[i]["share_out_kneg"]] for i in range(0, len(S), 4)], "share_min": float(min(share)), "share_max": float(max(share)),
                        "Gamma": (1.0 - win["median"]) if win else None, "spread": win["spread"] if win else None, "window": ([win["start"], win["end"]] if win else None),
                        "final_reflected_share": share[-1]}
            if RH is not None and kappa_m:
                tr["kroute"] = kroute_estimator(S, kappa_m, prof, RH, kj, ks_s, symbol_window(st_, Rc_, kc_, kappa_m, prof, RH, kj))
            tr["samples"] = [{k_: v_ for k_, v_ in s_.items()} for s_ in S if ta <= s_["tau"] <= tb]
            tr["fine_window"] = [ta, tb]
            trace = tr
            kr = tr.get("kroute") or {}
            print("  back-trace: kappa_p %s (resolved %s: %s); symbol %s; G2 %s (%.0f s)" % (kr.get("kappa_p"), kr.get("resolved"), "; ".join(kr.get("reasons", [])),
                  kr.get("kappa_symbol"), tr["G2"]["Gamma"], time.time() - t_p), flush=True)
        kp = (trace or {}).get("kroute", {}).get("kappa_p")
        for s_ in series:
            s_["theta_p"] = (s_["T"] / (kp / (2 * math.pi))) if (s_.get("T") and kp) else None
        Ts = np.array([s_["hold_over_rs"] for s_ in series]); ns = np.array([s_["n_w"] for s_ in series])
        slope = float(np.polyfit(Ts, np.log(np.maximum(ns, 1e-300)), 1)[0]) if len(ns) > 1 and np.all(ns > 0) else None
        fz = [s_["n_w"] for s_ in series if s_["frozen_stage"]]
        plateau = {"n_holds": len(fz), "spread_over_mean": (float(np.ptp(fz) / np.mean(fz)) if len(fz) > 1 and np.mean(fz) > 0 else None)}
        plateau["plateau"] = (plateau["spread_over_mean"] is not None and plateau["spread_over_mean"] < 0.1)
        pr = {k_: v_ for k_, v_ in p.items() if k_ not in ("k", "om", "a2")}
        pr.update({"series": series, "log_slope_per_rs": slope, "qf_check": qf, "floor_n0": n0, "floor_over_n_w_max": (n0 / max(ns)) if max(ns) > 0 else None,
                   "stationary": {l_: {k_: v_ for k_, v_ in s2.items() if k_ != "Gi"} for l_, s2 in stat.items()}, "endpoint": endpoint, "trace": trace, "axial_symbol": axial,
                   "plateau": plateau, "seconds": time.time() - t_p})
        if qf is not None:
            sh = series[a.holds_over_rs.index(a.qf_hold_over_rs)]
            th_ = sh.get("theta_m") or 1.0; q_ = (sh["n_w"] / sh["Gamma_w"]) if sh.get("Gamma_w") else 0.0
            q_ = q_ if q_ < 1 else 0.0
            ok_ = p["omega_over_kappa_m"] or omega * 2 * rsP
            qf["dtheta"] = th_ ** 2 * qf["rel_diff"] / (2 * math.pi * ok_ * max(1 - q_, 1e-12))
        for s_ in series:
            th_ = s_.get("theta_m") or 1.0
            q_ = s_["n_w"] / s_["Gamma_w"] if s_.get("Gamma_w") else 0.0
            q_ = q_ if q_ < 1 else 0.0
            ok_ = p["omega_over_kappa_m"] or omega * 2 * rsP
            s_["dtheta_duhamel"] = (th_ ** 2 * (s_["duhamel_dn_bound"] / s_["n_w"]) / (2 * math.pi * ok_ * max(1 - q_, 1e-12))) if (s_.get("duhamel_dn_bound") is not None and s_["n_w"] > 0) else None
        return pr

    if a.workers > 1 and torch is None:
        # BUILDER (CPU only): one forked process per packet, at most --workers at a time; each writes its row to a temporary json
        import traceback
        if any(os.environ.get(v_) != "1" for v_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
            print("warning: --workers with a threaded BLAS (set OMP/MKL/OPENBLAS_NUM_THREADS=1 before the launch for a fork-safe run)", flush=True)
        todo = list(enumerate(packs)); running = {}; results = {}
        tmpl = a.out_json + ".packet%d.tmp.json"
        while todo or running:
            while todo and len(running) < a.workers:
                i, p = todo.pop(0)
                if "skipped" in p:
                    results[i] = p
                    continue
                sys.stdout.flush()
                pid = os.fork()
                if pid == 0:
                    code = 0
                    try:
                        json.dump(run_packet(p), open(tmpl % i, "w"), default=float)
                    except BaseException:
                        json.dump({"omega_rs": p["omega_rs"], "m_over_omega": p["m_over_omega"], "error": traceback.format_exc()}, open(tmpl % i, "w"))
                        code = 1
                    sys.stdout.flush()
                    os._exit(code)
                running[pid] = i
            if not running:
                continue
            pid, st = os.wait()
            i = running.pop(pid)
            try:
                results[i] = json.load(open(tmpl % i))
                os.remove(tmpl % i)
            except (OSError, ValueError) as exc:
                results[i] = {"omega_rs": packs[i]["omega_rs"], "m_over_omega": packs[i]["m_over_omega"], "error": "worker %d status %s: %s" % (pid, st, exc)}
            print("packet %d (omega r_s %g, m/omega %g) collected%s" % (i, packs[i]["omega_rs"], packs[i]["m_over_omega"], (": " + results[i]["error"][-300:]) if "error" in results[i] else ""), flush=True)
            dump([results[j] for j in sorted(results)])
        rows.extend(results[j] for j in range(len(packs)))
    else:
        for p in packs:
            rows.append(run_packet(p))
            dump(rows)

    report(a, label, hinfo, FR, neigh, partner, LA, cuts, rsP, nsites, rows, t_start)
    print("done (%.0f s)" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
