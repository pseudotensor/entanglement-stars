#!/usr/bin/env python
"""Task 4, sec. 1 (SPEC_task4_injection.md; builder analyst): packet injection on the static chart, the lattice at rest (v = 0), on this lane's
locked background L frozen at the external time where the shell lapse is N_s (0.1 and 0.01).

Operator (the page code's polar law): per channel (j, kappa = sign (j + 1/2)) the radial polar operator A' = m psi^-1 + psi^-3 D_j on the int and
half nodes of a uniform grid h (strand5_w0fast_v1.channel_matrix's construction with the kappa sign as a parameter, as in strandS_dos_v3.py), its
polar factors U, E_0 from the SVD, B_0 = E_0^(1/2) diag(chi) E_0^(1/2) (chi = N psi), the chiral h_* = [[0, B_0 U^T], [U B_0, 0]] on (domain, range),
its modes from eigh(B_0) = (eps_r, z_r): a_r^+- = (z_r; +-U z_r)/sqrt2 at +-eps_r (one eigendecomposition per frozen configuration and channel).
Packets (BUILDER, labelled): psi_0 = g(r) e^(-i k_0 r) on the domain's int nodes, g a Gaussian of width sigma_x = 1/(2 sigma_omega) centred at
r_0 = 6 r_h, k_0 = omega_c (the flat chain's dispersion is eps = k + O(k^3) at these frequencies); the packet is its projection on the positive
modes weighted by the frequency window exp(-(eps - omega_c)^2/(4 sigma_omega^2)) (the projection keeps the ingoing branch: the positive modes at
momentum -k_0 move inward), normalized; evolved exactly by the modes.  The radial density sums the four components at each node's radius
(the int and half sets of the domain and the range).
Measured: (i) the reflected fraction, the norm beyond r_ref = 3 r_h at the end (after the outgoing centre has passed r_ref) and the largest norm
inside the shell (r < s); (ii) the delay: the outgoing centre's crossing of r_ref minus the ingoing centre's crossing (the centre of the density
beyond r_ref... computed as the time the density's first moment over r > r_ref/2 crosses r_ref, inward then outward), against the WKB integral
2 int_{r_t}^{r_ref} dr psi^2/N (massless) and the slope 16 A_0 in log(1/(a omega)) across the frequencies; (iii) the innermost radius reached (the
inner edge of the density at 1 percent of its peak over the run) against r_t(omega) solving N(r_t) = h omega psi(r_t)^2; (iv) the release test:
an outgoing packet at the band top (the positive modes' window at the top of the spectrum, sigma_x = 8) placed at the shell: its outward
progress over 40 r_h against N psi^-2.  The flat chart (A_0 = 0) is the zero of the scale.  Args: see --help.  One JSON per configuration.
Version 2 (after the smoke run): the crossings are timed by the norm beyond r_ref (t_in when it falls through half its initial value, t_out
when it rises back through half its final value; version 1's centroid of the part beyond r_ref/2 jumped to a small far component while the
bulk was inside); the delay is compared with two WKB integrals, the lead's light-speed 2 int psi^2/N dr and the one with the lattice group
velocity N psi^-2 cos p (sin p = h omega psi^2/N; the reviewer B red team's point 7, labelled), 2 int psi^2/(N sqrt(1 - (h omega psi^2/N)^2)) dr;
r_0 = max(6 r_h, r_ref + 4 sigma_x + 2h) per packet and one box R_max = max over the packets of r_0 + 5 sigma_x (BUILDER, labelled: the spec's
r_0 = 6 r_h cannot hold the lower frequencies' width); the innermost radius both as the 1 percent edge and as the density peak's innermost
position.
Version 3: the run length adds the interior round trip at the interior's light speed, 4 s psi_s^2/N_s (version 2's 1.3 x WKB + 8 sigma_x ended
before the part that entered the shell interior had returned at N_s = 0.1); nt scales with the run length (at least --nt)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("STRAND7_CODE", os.path.join(HERE, "..", "strand7")))
sys.path.insert(0, os.environ.get("STRANDS_CODE", os.path.join(HERE, "..", "strandS")))
from strand7_sea_page_v5 import LockedBackground  # noqa: E402
from strandS_dos_v3 import channel_matrix  # noqa: E402


class Flat:
    def at(self, T, rr):
        rr = np.asarray(rr, float)
        return np.ones_like(rr), np.ones_like(rr), 0.0, 0.0


def build(bg, T, j, sign, m, h, n):
    r_int = h * np.arange(1, n + 1, dtype=float)
    r_half = h * (np.arange(n, dtype=float) + 0.5)
    psi_i, chi_i, A, B = bg.at(T, r_int)
    psi_h, chi_h, _, _ = bg.at(T, r_half)
    Ap, k0 = channel_matrix(r_int, r_half, psi_i, psi_h, m, j, h, sign)
    mm = n - k0
    rdom = np.concatenate([r_int[k0:], r_half[k0:]])
    chi = np.concatenate([chi_i[k0:], chi_h[k0:]])
    W, S, Vh = np.linalg.svd(Ap)
    U = W @ Vh
    Gp = (Vh.T * np.sqrt(S)) @ Vh
    B0 = Gp @ (chi[:, None] * Gp)
    B0 = 0.5 * (B0 + B0.T)
    eps, Z = np.linalg.eigh(B0)
    return eps, Z, U, rdom, mm


def evolve_density(eps, Z, U, c, ts, rdom):
    """the radial density of sum_r c_r e^(-i eps_r t) a_r^+ at the times ts: |dom|^2 + |range|^2 per node radius (both on rdom)."""
    out = []
    for t in ts:
        ph = c * np.exp(-1j * eps * t)
        dom = Z @ ph / math.sqrt(2)
        rng = U @ (Z @ ph) / math.sqrt(2)
        out.append(np.abs(dom) ** 2 + np.abs(rng) ** 2)
    return np.array(out)


def radial_profile(rho, rdom, edges):
    idx = np.clip(np.searchsorted(edges, rdom) - 1, 0, len(edges) - 2)
    return np.array([np.bincount(idx, weights=row, minlength=len(edges) - 1) for row in rho])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", type=float, default=16.0)
    ap.add_argument("--A0-over-rs0", type=float, default=0.234465)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--Ns", type=float, default=0.1, help="the shell lapse at the frozen time; 1 = the flat chart (A_0 = 0)")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Rmax-over-rh", type=float, default=30.0)
    ap.add_argument("--j", type=float, default=0.5)
    ap.add_argument("--sign", type=int, default=-1)
    ap.add_argument("--mass", type=float, default=0.0)
    ap.add_argument("--omegas-rh", default="0.5,1,2,4")
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rs0 = 2 * a.E0
    A0 = a.A0_over_rs0 * rs0
    rh = 4 * A0
    h = a.h
    flat = a.Ns >= 1
    if flat:
        bg, Tf, s = Flat(), 0.0, 0.0
        Nf = lambda r: 1.0
        psif = lambda r: 1.0
    else:
        bg = LockedBackground(A0, rs0, a.R0_over_rs * rs0, 80 * rs0)
        Tf = brentq(lambda T: bg.N_s(T) - a.Ns, 0.0, 80 * rs0 * 0.999, xtol=1e-10)
        s = bg.s(Tf)
        Nf = lambda r: (1 - A0 / max(r, s)) / (1 + A0 / max(r, s))
        psif = lambda r: 1 + A0 / max(r, s)
    rref = 3 * rh
    plan = []
    for omr in [float(v) for v in a.omegas_rh.split(",")]:
        om = omr / rh
        sw = om / 4
        sx = 1 / (2 * sw)
        r0 = max(6 * rh, rref + 4 * sx + 2 * h)
        plan.append((omr, om, sw, sx, r0))
    Rmax = max(a.Rmax_over_rh * rh, max(p_[4] + 5 * p_[3] for p_ in plan))
    n = int(round(Rmax / h))
    eps, Z, U, rdom, mm = build(bg, Tf, a.j, a.sign, a.mass, h, n)
    pos = eps > 0
    edges = np.arange(0.0, rdom.max() + h, h)
    rc = 0.5 * (edges[1:] + edges[:-1])
    out = {"meta": dict(vars(a), rs0=rs0, A0=A0, r_h=rh, T_frozen=Tf, s=s, n=n, R_max=Rmax, eps_max=float(eps.max()), flat=flat), "packets": []}
    for omr, om, sw, sx, r0 in plan:
        psi0 = np.zeros(2 * mm, complex)
        g = np.exp(-(rdom - r0) ** 2 / (4 * sx ** 2) - 1j * om * rdom)
        psi0[:mm] = g[:mm]
        # the projection on the positive modes a_r^+ = (z; U z)/sqrt2, psi0 placed in the domain part only
        c = (Z.T @ psi0) / math.sqrt(2)
        c = np.where(pos, c * np.exp(-(eps - om) ** 2 / (4 * sw ** 2)), 0.0)
        c = c / np.linalg.norm(c)
        # the run: until the outgoing centre has passed r_ref, from the WKB round trip with margin
        if flat:
            twkb = 2 * (r0 - 0.0)
        else:
            rt = brentq(lambda r: Nf(r) - h * om * psif(r) ** 2, s + 1e-9, r0) if Nf(s + 1e-9) < h * om * psif(s + 1e-9) ** 2 else s
            twkb = 2 * quad(lambda r: psif(r) ** 2 / Nf(r), rt, r0, limit=400)[0]
        T = 1.3 * twkb + 8 * sx + (0.0 if flat else 4 * s * psif(s) ** 2 / Nf(s))
        nt = max(a.nt, int(T / (0.5 * sx)) + 1)
        ts = np.linspace(0, T, nt)
        rho = evolve_density(eps, Z, U, c, ts, rdom)
        prof = radial_profile(rho, rdom, edges)
        tot = prof.sum(1)
        beyond = prof[:, rc > rref].sum(1)
        inside = prof[:, rc < max(s, h)].sum(1)
        far = rc > rref / 2
        cen = (prof[:, far] @ rc[far]) / np.maximum(prof[:, far].sum(1), 1e-300)
        # crossings by the norm beyond r_ref: inward through half its initial value, outward through half its final value
        fb = beyond / np.maximum(tot, 1e-300)
        tin = tout = None
        h0, h1 = 0.5 * fb[0], 0.5 * fb[-1]
        kmin = int(np.argmin(fb))
        for k in range(1, kmin + 1):
            if tin is None and fb[k - 1] > h0 >= fb[k]:
                tin = ts[k - 1] + (fb[k - 1] - h0) / (fb[k - 1] - fb[k]) * (ts[k] - ts[k - 1])
        for k in range(kmin + 1, len(ts)):
            if tout is None and fb[k - 1] < h1 <= fb[k]:
                tout = ts[k - 1] + (h1 - fb[k - 1]) / (fb[k] - fb[k - 1]) * (ts[k] - ts[k - 1])
        peak = prof.max()
        inner = float(min(rc[np.nonzero(row > 0.01 * peak)[0][0]] for row in prof if np.any(row > 0.01 * peak)))
        inner_peak = float(min(rc[int(np.argmax(row))] for row in prof))
        if flat:
            r_t = 0.0
            wkb_delay = 2 * rref
            wkb_lattice = 2 * quad(lambda r: 1 / math.sqrt(max(1 - (h * om) ** 2, 1e-300)), 0.0, rref)[0]
        else:
            r_t = brentq(lambda r: Nf(r) - h * om * psif(r) ** 2, s + 1e-9, rref) if Nf(s + 1e-9) < h * om * psif(s + 1e-9) ** 2 else None
            wkb_delay = 2 * quad(lambda r: psif(r) ** 2 / Nf(r), r_t if r_t else s, rref, limit=400)[0]
            lat = lambda r: psif(r) ** 2 / Nf(r) / math.sqrt(max(1 - (h * om * psif(r) ** 2 / Nf(r)) ** 2, 1e-300))
            wkb_lattice = 2 * quad(lat, r_t if r_t else s, rref, limit=800)[0]
        out["packets"].append({"omega_rh": omr, "omega": om, "sigma_omega": sw, "sigma_x": sx, "r0": r0, "T_run": T, "norm_end": float(tot[-1]),
                               "reflected": float(beyond[-1] / tot[-1]), "inside_shell_max": float(inside.max() / tot[0]),
                               "t_in": tin, "t_out": tout, "delay": (tout - tin) if (tin is not None and tout is not None) else None,
                               "wkb_delay": wkb_delay, "wkb_lattice": wkb_lattice, "r_t_pred": r_t, "innermost_1pct": inner, "innermost_peak": inner_peak,
                               "beyond_series": [float(x) for x in fb[::max(1, nt // 50)]], "nt": nt,
                               "centre_series": [float(x) for x in cen[::max(1, nt // 50)]], "t_series": [float(x) for x in ts[::max(1, nt // 50)]]})
        print("Ns %g j %g sign %d m %g omega r_h %g: reflected %.6f, inside max %.3e, delay %s (WKB light %.1f, lattice %.1f), innermost 1pct %.2f peak %.2f (r_t %s), s %.2f (%.0f s)" % (
            a.Ns, a.j, a.sign, a.mass, omr, beyond[-1] / tot[-1], inside.max() / tot[0], "%.1f" % (tout - tin) if (tin and tout) else "-", wkb_delay,
            wkb_lattice, inner, inner_peak, "%.2f" % r_t if r_t else "-", s, time.time() - t0), flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
