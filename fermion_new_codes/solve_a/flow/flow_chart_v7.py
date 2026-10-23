#!/usr/bin/env python3
"""flow_chart_v7.py: the lead's flow_chart version 5 of SPEC_task3_prescribed_flow.md sec. 11.4 (file v7 because v5 and v6 are task 5's
files).  Builder analyst.  flow_chart_v4.py is imported unchanged (its operator, propagation and packet constructions stand); this file
carries the sec. 11.4 fixes of the reviewer B red team of the derivations
(reviewb/review_output/reviewb_azure_redteam_flow_derivations_20261014_205632/result.md, sec. 9) and the sec. 11.3 jobs.  Every record carries
the label "prescribed flow (simpler version); ... v declared" and the script name, never a hash.

The fixes (each function names its red-team item):
 9.2   group_velocity: one-sided limits at massless nodes (lim sin k cos k/|sin k| = side; v4 divided 0 by 0).  turning_entry: the
       branch-qualified zero of the positive-band v_g with the unsquared w = sin k cos k/(E cos 2k); invalid cases marked: cos 2k = 0
       (|k| = pi/4) "no zero"; w < 0 (+k > pi/4, or -pi/4 < k < 0) "no zero on the inward flow at this momentum" (the stall is at -k);
       cos k = 0 "no zero at finite flow"; sin k = 0 with m_c > 0 "zero only at w = 0".
 9.3   vg_max_branch: the maximum labelled by its interval, (0, pi/2) = the outgoing representative on the inward flow, beside the full
       zone's; the turning tables carry both signs of k.
 9.6   build_packet: w_n ~ sum_i sqrt(rho_i) q_i e^{i k_i (n - x0)} u(k_i), the quadrature weights q_i separate from the density rho
       (v4: sqrt(rho_i q_i), which equals this only on a uniform grid after normalization); killing_content: the constructed packet's
       actual Killing-frequency content under the frozen generator.
 9.7   threshold: the minimum of the exterior positive band over the closure of the chosen branch (v_g > 0 outgoing, < 0 ingoing), no
       clamping (v4: m_c sqrt(1 - v^2), clamped to 0 at |v| >= 1).
 9.9   invert_theta: the bracket [0.2, 3] widened by factors of 2 up to [1e-4, 1e4]; outside it the record says "no positive-temperature
       fit" (v4 raised and ended the scan).
 9.10  (v3's fix kept) sigma_omega = min(omega_c/4, (omega_c - omega_th)/4); the test omega_c > omega_th separate; the window truncated at
       omega_th.
 9.11  packet_geometry: for every packet the norms on the n_tail outermost sites at both ends, the untruncated construction's norm lost
       beyond each end, the norm on the far side of the surface and inside the ball; rejection (recorded; not evolved unless
       --keep-rejected) when an end tail or a lost norm exceeds --tail-tol, and in convert when the surface's far side or the non-uniform
       region holds more; band_weights: the positive/negative normal-energy weights from the flow-free generator's eigenvectors (the exact
       chain projector), for every packet.
 9.12  c_w: int rho c(omega/kappa) d_theta f / int rho d_theta f at theta = 1 (c(r) = 1/3 + 5 r^2/18), beside c(omega_c/kappa).
 9.15  zero_velocity_radii: the radial zero-velocity radius from the unsquared sin k cos k/E + v(R) cos 2k = 0 with
       E = sqrt(m^2 + (kappa_j/R)^2 + sin^2 k) on v4's pg_profile, solved for R (every root, exterior or inside the ball);
       exact_velocity: <psi| i[h, X] |psi> = -2 Im <psi| h X |psi> at t = 0 beside v4's fitted centre slope; symbol_vg_spatial (BUILDER):
       the positive-band symbol averaged over the packet's site density and its Gaussian momentum weights, beside v4's launch-radius
       average.
 9.16, 9.17  not in this version (D3 version 5 is later, spec 11.3).
 9.18  momentum_readout: the circular mean arg sum p(k) e^{ik} and its length, per normal-energy band (the flow-free generator's
       projector) and per sector (|k| < pi/4 long wave, [pi/4, 3pi/4] short wave, > 3pi/4 the doubler valley) with the k > 0 weight; v4's
       arithmetic k @ p kept as carrier_arith_v4 for the comparison only.
 9.20  numpy arrays serialized with tolist (RecordEncoder); the record is written whole; stdout carries the whole JSON only without
       --output (no truncation).

Modes: tests (the fixes against v4), symbol (A1 re-emitted), radial (A2, A3, D1, D2, D4 with the fixes; D2 with the exact rest condition;
D1 with the momentum-resolved norm inside R_m, the symbol's ODE and the symbol's ray ensemble; D3 deferred), planar_dust (D1 on the planar
two-component chain), a2compare (spec 11.3's A2 comparator on stored v4 records, no evolution), convert (part (b) with the fixes; the planar
part (b) is the record's package, so this mode is smoke-tested only), compare_v4 (v7 records against v4 records, the validation).
BUILDER choices (labelled in the records): the A2 launch scan keeps v4's positions (f x the squared law's radius) so that the records
compare; the ray ensemble (Gauss-Hermite 24 x 24 over the initial Wigner Gaussian, RK4 at dt = 0.01 and 0.02); the planar D1 chain
x in [-L, L] (L = 384 by default) with the profile even in x as spec 11.3 writes it.
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from scipy.special import expit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FLOW_CODE_DIR", HERE))
import flow_chart_v4 as V4  # noqa: E402

SCRIPT = os.path.basename(__file__)
LABEL_PG = "prescribed flow (simpler version); PG radial; v declared"
LABEL_TANH = "prescribed flow (simpler version); planar tanh; v declared"
LABEL_PLANAR_DUST = ("prescribed flow (simpler version); planar two-component chain (no centrifugal term), v(x) = -sqrt(r_s/|x|) for "
                     "|x| >= R_m and -sqrt(r_s/R_m^3)|x| inside (even in x, as spec 11.3 writes it); v declared")
PI = np.pi


# ---------------------------------------------------------------- output (9.20)
class RecordEncoder(json.JSONEncoder):
    """9.20: numpy arrays by tolist, numpy scalars by item, complex numbers as [re, im]."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return super().default(o)


# ---------------------------------------------------------------- the symbol (9.2, 9.3)
def group_velocity(k, mc, v, band=1, side=1.0):
    """9.2: band * sin k cos k/E + v cos 2k on the positive (band = +1) or negative (band = -1) normal-energy band of the uniform symbol,
    E = sqrt(mc^2 + sin^2 k).  At a massless node (mc = 0, sin k = 0) the one-sided limit from the side `side` is used:
    lim sin k cos k/|sin k| = side at k0 = 0 and at k0 = pi (v4 divided 0 by 0 there)."""
    k = np.asarray(k, float)
    E = np.sqrt(mc * mc + np.sin(k) ** 2)
    s = np.sin(k) * np.cos(k)
    first = np.where(E > 0, s / np.where(E > 0, E, 1.0), float(side))
    return band * first + v * np.cos(2 * k)


def turning_entry(k, mc, side=1.0, tol=1e-12):
    """9.2, 9.3 (red team 2.1 to 2.4): the zero of the positive-band v_g at signed momentum k on an inward flow v = -w, w > 0, from the
    unsquared w = sin k cos k/(E cos 2k), branch-qualified; R/r_s = 1/w^2 on the PG exterior (= v4's squared law where valid)."""
    k = float(k)
    E = math.sqrt(mc * mc + math.sin(k) ** 2)
    c2, ck = math.cos(2 * k), math.cos(k)
    out = {"k": k, "mc": float(mc)}
    if abs(c2) < tol:
        out.update(status="no zero", reading="cos 2k = 0 (|k| = pi/4): v_g = sin k cos k/E =/= 0 at every finite flow; the squared "
                   "law's R = 0 needs unbounded flow (red team 2.3)")
        return out
    if abs(ck) < tol:
        out.update(status="no zero at finite flow", reading="cos k = 0 (|k| = pi/2, the band maximum): v_g = +w (red team 3.4)")
        return out
    if E == 0:
        sE = float(side)
        out["one_sided"] = "massless node: the limit k -> 0 from the side %+d" % int(side)
    else:
        sE = math.sin(k) * ck / E
    w = sE / c2
    out["w_unsquared"] = w
    if w == 0.0:
        out.update(status="zero only at w = 0", reading="sin k = 0 with m_c > 0: v_g = v; the zero needs v = 0 (R = infinity on PG)")
        return out
    if w < 0:
        out.update(status="no zero on the inward flow at this momentum",
                   reading="the unsquared w < 0; the inward-flow stall is at momentum -k (red team 2.2)")
        return out
    if (0 < k < PI / 4) or (E == 0 and side > 0):
        branch = "outgoing representative, 0 < k < pi/4"
    elif -PI / 2 < k < -PI / 4:
        branch = "ingoing at momentum k, pi/4 < |k| < pi/2 (the stencil's short-wave sector)"
    else:
        branch = "other (outside (-pi/2, pi/2))"
    ratio = 1.0 / (w * w)
    out.update(status="zero", branch=branch, R_over_rs=ratio, inside_rs=bool(ratio < 1),
               omega_at_locus=E - w * math.sin(2 * k) / 2)
    return out


def vg_max_branch(mc, w=1.0, ngrid=200001):
    """9.3: the maximum of the positive-band v_g at v = -w by interval: (0, pi/2) is the outgoing representative on the inward flow
    (w >= 0; no bound for w > 1 or w < 0, red team 3.1, 3.4); the full zone beside it (the symbol has period pi)."""
    f = lambda x: -float(group_velocity(x, mc, -w))  # noqa: E731
    out = {"w": w, "mc": mc, "interval_reading": "(0, pi/2): the outgoing representative on the inward flow w >= 0"}
    for name, lo, hi in (("interval_0_pi2", 1e-6, PI / 2 - 1e-6), ("full_zone", -PI, PI)):
        k = np.linspace(lo, hi, ngrid)
        vg = group_velocity(k, mc, -w)
        i = int(np.argmax(vg))
        r = minimize_scalar(f, bounds=(k[max(i - 1, 0)], k[min(i + 1, ngrid - 1)]), method="bounded", options={"xatol": 1e-13})
        best, kb = (-float(r.fun), float(r.x)) if -float(r.fun) >= float(vg[i]) else (float(vg[i]), float(k[i]))
        out[name] = {"max": best, "k": kb}
    return out


def zero_velocity_radii(k, m, kj, rs, Rm, R_lo=0.5, R_hi_over_rs=1e4, n=400001):
    """9.15 (red team 2.5): every R with sin k cos k/E(R) + v(R) cos 2k = 0, E = sqrt(m^2 + (kappa_j/R)^2 + sin^2 k), v = v4's pg_profile
    (exterior and ball), solved for R (sign changes on a geometric grid, refined by brentq)."""
    k = float(k)
    s, c2, s2 = math.sin(k) * math.cos(k), math.cos(2 * k), math.sin(k) ** 2
    out = {"k": k, "equation": "sin k cos k/E + v(R) cos 2k = 0, E = sqrt(m^2 + (kappa_j/R)^2 + sin^2 k), v = pg_profile"}
    if abs(c2) < 1e-12:
        out.update(status="no zero", roots=[], reading="cos 2k = 0: v_g = sin k cos k/E > 0 at every finite flow (red team 2.3)")
        return out

    def g(R):
        R = np.asarray(R, float)
        return s / np.sqrt(m * m + (kj / R) ** 2 + s2) + V4.pg_profile(R, rs, Rm) * c2

    Rg = np.geomspace(R_lo, R_hi_over_rs * rs, n)
    gv = g(Rg)
    roots = []
    for i in np.nonzero(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0)[0]:
        r = brentq(lambda x: float(g(x)), Rg[i], Rg[i + 1], xtol=1e-13, rtol=1e-15)
        roots.append({"R": r, "R_over_rs": r / rs, "region": "exterior (R >= R_m)" if r >= Rm else "inside the ball (R < R_m)",
                      "inside_rs": bool(r < rs)})
    out.update(status="zero" if roots else "no zero on pg_profile", roots=roots)
    return out


def carrier_label(k0, m, kj, rs, Rm, Rmax):
    """spec 11.3's labels for an A2 carrier: "no zero" (cos 2k = 0), the sign condition (turning_entry with m_c = m), "inside the ball"
    (the squared exterior law's radius below R_m), else "exterior zero"; with the unsquared zero-velocity radii on pg_profile."""
    te = turning_entry(k0, m)
    zv = zero_velocity_radii(k0, m, kj, rs, Rm)
    sq = rs * float(V4.turning_ratio(abs(k0), m))
    ext = [r["R"] for r in zv["roots"] if r["R"] >= Rm]
    if te["status"] == "no zero":
        label = "no zero"
    elif te["status"] != "zero":
        label = te["status"]
    elif sq < Rm:
        label = "inside the ball"
    else:
        label = "exterior zero"
    out = {"k0": float(k0), "label": label, "squared_law_R_v4": sq, "branch": te.get("branch"), "zero_velocity": zv,
           "exterior_root": ext[0] if ext else None, "exterior_root_within_chain": bool(ext and ext[0] <= Rmax)}
    if label == "inside the ball":
        out["reading"] = ("the squared exterior law's radius lies inside R_m, where the declared profile is the ball's linear one: not a "
                          "test of the exterior law")
    if label == "no zero":
        out["reading"] = te["reading"]
    return out


# ---------------------------------------------------------------- the planar threshold, packets, temperature (9.6, 9.7, 9.9, 9.10, 9.12)
def threshold(mc, vext, branch="out", ngrid=200001):
    """9.7 (red team 5.2): the minimum of the exterior positive band omega_+(k) = E + v sin 2k/2 over the closure of the chosen branch
    (v_g > 0 outgoing, v_g < 0 ingoing): the grid minimum on the branch refined at the zeros of v_g that bound it; no clamping."""
    k = np.linspace(-PI, PI, ngrid)
    vg = group_velocity(k, mc, vext)
    om = V4.bands(k, mc, vext)[0]
    sel = vg > 0 if branch == "out" else vg < 0
    res = {"branch": branch, "mc": float(mc), "vext": float(vext), "continuum_v4_clamped": float(V4.threshold(mc, vext)),
           "method": "min of omega_+ over the branch's closure (grid, refined at the bounding zeros of v_g)"}
    if not np.any(sel):
        res.update(omega_th=None, status="no such branch")
        return res
    i = int(np.argmin(np.where(sel, om, np.inf)))
    best, kb = float(om[i]), float(k[i])
    f = lambda x: float(group_velocity(x, mc, vext))  # noqa: E731
    for j in np.nonzero(np.sign(vg[:-1]) != np.sign(vg[1:]))[0]:
        if not (sel[j] or sel[j + 1]):
            continue
        kz = brentq(f, k[j], k[j + 1], xtol=1e-15)
        oz = float(V4.bands(kz, mc, vext)[0])
        if oz < best:
            best, kb = oz, float(kz)
    w = -vext
    if 0 <= w < 1:
        F = 1 - w * w
        res["expansion_red_team_5.2"] = mc * math.sqrt(F) + mc ** 3 * w ** 4 / (2 * F ** 1.5)
    res.update(omega_th=best, k_at_min=float((kb + PI / 2) % PI - PI / 2), status="ok",
               k_note="k_at_min in (-pi/2, pi/2]: the band has period pi (the doubler valley repeats it)")
    return res


def branch_momenta(omega_c, sigma, mc, vext, omega_th, outgoing=True, nk=4001, kmax=PI / 2 - 1e-6):
    """the branch's momenta (ascending grid; v_g > 0 outgoing, < 0 ingoing) with Killing frequency in omega_c +- 5 sigma and above
    omega_th (9.10: the window truncated at the threshold); returns k, omega(k), the k-density rho (|A|^2 dk = G(omega) d omega,
    normalized with q) and the momentum-cell weights q (the grid spacing), separate (9.6).  None when the window is empty."""
    sgn = 1.0 if outgoing else -1.0
    k = np.sort(sgn * np.linspace(1e-6, kmax, nk))
    dk = float(abs(k[1] - k[0]))
    om = V4.bands(k, mc, vext)[0]
    vg = group_velocity(k, mc, vext)
    ok = (vg > 0) if outgoing else (vg < 0)
    ok &= (np.abs(om - omega_c) < 5 * sigma) & (om > omega_th)
    if not np.any(ok):
        return None
    k, om, vg = k[ok], om[ok], vg[ok]
    q = np.full(len(k), dk)
    rho = np.exp(-(om - omega_c) ** 2 / (2 * sigma * sigma)) * np.abs(vg)
    return k, om, rho / float(np.sum(rho * q)), q


def trap_weights(k):
    """trapezoid weights on a (possibly non-uniform) ascending grid: sum(f q) = trapz(f, k)."""
    k = np.asarray(k, float)
    q = np.zeros_like(k)
    if len(k) > 1:
        d = np.diff(k)
        q[:-1] += d / 2
        q[1:] += d / 2
    return q


def build_packet(nsites, x_index, k, rho, q, spinor):
    """9.6: w_n ~ sum_i sqrt(rho_i) q_i e^{i k_i (n - x0)} u(k_i), normalized; rho the momentum density, q the quadrature weights,
    spinor(k) the 2-spinor (v4's positive_spinor or onsite_spinor)."""
    n = np.arange(nsites)
    order = np.argsort(k)
    k, rho, q = np.asarray(k)[order], np.asarray(rho)[order], np.asarray(q)[order]
    A = np.sqrt(np.abs(rho)) * q
    u = np.array([spinor(kk) for kk in k])
    w = ((np.exp(1j * np.outer(n - x_index, k)) * A) @ u).reshape(-1)
    return w / np.linalg.norm(w)


def fermi_average(k, om, rho, q, theta, kappa):
    """F_w(theta) = sum rho q f(omega), f = 1/(exp(2 pi omega/(kappa theta)) + 1) (red team 9.8's measure), overflow-safe."""
    return float(np.sum(rho * q * expit(-2 * PI * np.asarray(om) / (kappa * theta))))


def invert_theta(nw, k, om, rho, q, kappa, lo=0.2, hi=3.0, lo_min=1e-4, hi_max=1e4):
    """9.9: theta_r with F_w(theta_r) = n_w; the bracket [lo, hi] widened by factors of 2 to [lo_min, hi_max]; no sign change there:
    "no positive-temperature fit", recorded (v4 raised)."""
    if not nw > 0:
        return {"theta": None, "status": "no positive-temperature fit", "reason": "n_w <= 0"}
    F = lambda th: fermi_average(k, om, rho, q, th, kappa) - nw  # noqa: E731
    a, b = lo, hi
    fa, fb = F(a), F(b)
    while fa * fb > 0 and (a > lo_min or b < hi_max):
        a, b = max(a / 2, lo_min), min(b * 2, hi_max)
        fa, fb = F(a), F(b)
    if fa == 0:
        return {"theta": a, "status": "fit", "bracket": [a, b]}
    if fb == 0:
        return {"theta": b, "status": "fit", "bracket": [a, b]}
    if fa * fb > 0:
        return {"theta": None, "status": "no positive-temperature fit", "bracket": [a, b], "F_minus_n_at_bracket": [fa, fb]}
    return {"theta": float(brentq(F, a, b, xtol=1e-12)), "status": "fit", "bracket": [a, b]}


def c_mono(r):
    """the record's monochromatic coefficient c(r) = 1/3 + 5 r^2/18."""
    return 1.0 / 3.0 + 5.0 * r * r / 18.0


def c_w(k, om, rho, q, kappa):
    """9.12: c_w = int rho c(omega/kappa) d_theta f / int rho d_theta f at theta = 1, d_theta f|_1 = f (1 - f) y, y = 2 pi omega/kappa."""
    y = 2 * PI * np.asarray(om) / kappa
    f = expit(-y)
    d = rho * q * f * (1 - f) * y
    return float(np.sum(d * c_mono(np.asarray(om) / kappa)) / np.sum(d))


# ---------------------------------------------------------------- packet diagnostics (9.11, 9.15, 9.18)
def exact_velocity(h, psi, X):
    """9.15: d<X>/dt at t = 0 = <psi| i[h, X] |psi> = -2 Im <psi| h X |psi> / <psi|psi> (X per site, both components)."""
    Xpsi = np.repeat(np.asarray(X, float), 2) * psi
    return float(-2.0 * np.imag(np.vdot(psi, h @ Xpsi)) / np.vdot(psi, psi).real)


def exact_velocity_pos_band(h, psi, X, bands):
    """9.15 with 9.4's budget: the exact initial velocity of the packet's normalized positive normal-energy projection (the flow-free
    generator's projector), which removes the interband (zitterbewegung) term of the single-spinor construction."""
    pos, _ = band_parts(psi, bands)
    return exact_velocity(h, pos / np.linalg.norm(pos), X)


def symbol_vg_spatial(psi, X, k0, sigma, m, kj, vfun, nk=801):
    """9.15 (BUILDER, labelled): the positive-band symbol velocity averaged over the packet's site density and its Gaussian momentum
    weights (v4's packet_avg_vg grid), with v(R) and kappa_j/R at every site (the spatial distribution v4's launch-radius average omits)."""
    rho = V4.density(psi)
    rho = rho / rho.sum()
    sel = rho > 1e-16 * rho.max()
    Xs, rs_ = np.asarray(X, float)[sel], rho[sel]
    kk = np.linspace(k0 - 4 / (2 * sigma), k0 + 4 / (2 * sigma), nk)
    wk = np.exp(-2 * sigma ** 2 * (kk - k0) ** 2)
    wk /= wk.sum()
    a = (kj / Xs) if kj else np.zeros_like(Xs)
    vg = V4.symbol_vg(kk[None, :], m, a[:, None], np.asarray(vfun(Xs), float)[:, None])
    return float(rs_ @ (vg @ wk) / rs_.sum())


def killing_content(Q, e, psi, thr=None):
    """9.6: the packet's Killing-frequency content under the frozen generator (mean, spread, the weight below 0 and below thr)."""
    c = Q.conj().T @ psi
    p = np.abs(c) ** 2
    p = p / p.sum()
    mu = float(p @ e)
    out = {"mean": mu, "std": float(np.sqrt(p @ (e - mu) ** 2)), "weight_negative": float(p[e < 0].sum())}
    if thr is not None:
        out["weight_below_threshold"] = float(p[e < thr].sum())
        out["threshold"] = float(thr)
    return out


def make_bands(mz, yy):
    """the flow-free generator's eigendecomposition (the normal-energy bands; 9.11, 9.18), the positive-band columns kept."""
    e0, Q0 = V4.spectrum(V4.chain_hamiltonian(mz, yy, np.zeros(len(mz))))
    return {"e0": e0, "Q0": Q0, "Qpos": np.ascontiguousarray(Q0[:, e0 > 0])}


def band_weights(psi, bands, tol=1e-12):
    """9.11: the positive/negative normal-energy weights from the flow-free generator's eigenvectors (exact chain projector)."""
    e0 = bands["e0"]
    p = np.abs(bands["Q0"].conj().T @ psi) ** 2
    nrm = float(p.sum())
    return {"positive": float(p[e0 > tol].sum()) / nrm, "negative": float(p[e0 < -tol].sum()) / nrm,
            "zero_modes": float(p[np.abs(e0) <= tol].sum()) / nrm}


def band_parts(psi, bands):
    Qp = bands["Qpos"]
    pos = Qp @ (Qp.conj().T @ psi)
    return pos, psi - pos


def _readout(a):
    """one momentum readout of the amplitudes a (nsites x 2): norm, circular mean and length, sector norms, the long-wave mean."""
    norm = float(np.sum(np.abs(a) ** 2))
    if norm <= 0:
        return {"norm": 0.0}
    n = 4 * len(a)
    k = 2 * PI * np.fft.fftfreq(n)
    pk = (np.abs(np.fft.fft(a, n=n, axis=0)) ** 2).sum(axis=1)
    pk = pk / pk.sum()
    z = complex(np.sum(pk * np.exp(1j * k)))
    ak = np.abs(k)
    lo, sh, db = ak < PI / 4, (ak >= PI / 4) & (ak <= 3 * PI / 4), ak > 3 * PI / 4
    out = {"norm": norm, "circ_mean": float(np.angle(z)), "circ_R": float(abs(z)), "long": norm * float(pk[lo].sum()),
           "short": norm * float(pk[sh].sum()), "doubler": norm * float(pk[db].sum()), "k_positive": norm * float(pk[k > 0].sum())}
    if pk[lo].sum() > 0:
        out["long_mean"] = float(np.sum(pk[lo] * k[lo]) / pk[lo].sum())
    return out


def momentum_readout(psi, bands=None, mask=None):
    """9.18: the circular, band- and sector-resolved momentum readout of psi (restricted to the sites in mask when given; the band
    projection first)."""
    def cut(x):
        a = x.reshape(-1, 2).copy()
        if mask is not None:
            a[~mask] = 0.0
        return a
    res = {"all": _readout(cut(psi))}
    if bands is not None:
        pos, neg = band_parts(psi, bands)
        res["pos"], res["neg"] = _readout(cut(pos)), _readout(cut(neg))
    return res


def kresolved_mask(psi, mask):
    """v4's kresolved with a general site mask (the planar |x| < R_m): the short-wave ([pi/4, 3pi/4]) and long-wave norms in the mask."""
    a = psi.reshape(-1, 2).copy()
    a[~mask] = 0.0
    tot = float(np.sum(np.abs(a) ** 2))
    if tot <= 0:
        return 0.0, 0.0
    n = 4 * len(a)
    k = 2 * PI * np.fft.fftfreq(n)
    pk = (np.abs(np.fft.fft(a, n=n, axis=0)) ** 2).sum(axis=1)
    pk = pk / pk.sum() * tot
    hi = (np.abs(k) >= PI / 4) & (np.abs(k) <= 3 * PI / 4)
    return float(pk[hi].sum()), float(pk[~hi].sum())


def gaussian_lost(X, R0, sigma):
    """9.11: the fraction of the untruncated Gaussian density exp(-(R - R0)^2/(2 sigma^2)) (v4's gaussian_carrier, continued on the
    integer lattice past the chain) that lies beyond the inner and the outer end."""
    first, last = float(X[0]), float(X[-1])
    pad = int(math.ceil(12 * sigma)) + 2
    Rext = np.arange(first - pad, last + pad + 1, 1.0)
    d = np.exp(-(Rext - R0) ** 2 / (2 * sigma * sigma))
    tot = float(d.sum())
    return float(d[Rext < first - 0.5].sum() / tot), float(d[Rext > last + 0.5].sum() / tot)


def packet_geometry(psi, X, n_tail, tol, surface=None, ball=None, lost=None, uniform_region=None, reject_surface=False):
    """9.11: tail norms at both ends (the n_tail outermost sites), the lost norm of the untruncated construction, the norm on the far side
    of the surface (from the packet's centre) and inside the ball (|X| < ball); the verdict with its reasons."""
    X = np.asarray(X, float)
    rho = V4.density(psi)
    rho = rho / rho.sum()
    c = float(rho @ X)
    out = {"n_tail": int(n_tail), "tol": float(tol), "centre": c, "width": float(np.sqrt(rho @ (X - c) ** 2)),
           "tail_inner": float(rho[:n_tail].sum()), "tail_outer": float(rho[-n_tail:].sum())}
    reasons = []
    for end in ("inner", "outer"):
        if out["tail_" + end] > tol:
            reasons.append("%s end tail %.3g > %g" % (end, out["tail_" + end], tol))
    if lost is not None:
        out["lost_inner"], out["lost_outer"] = float(lost[0]), float(lost[1])
        for end in ("inner", "outer"):
            if out["lost_" + end] > tol:
                reasons.append("the construction lost %.3g beyond the %s end" % (out["lost_" + end], end))
    if surface is not None:
        far = (X < surface) if c > surface else (X > surface)
        out["surface"] = float(surface)
        out["tail_surface"] = float(rho[far].sum())
        if reject_surface and out["tail_surface"] > tol:
            reasons.append("far side of the surface %.3g > %g" % (out["tail_surface"], tol))
    if ball is not None:
        out["inside_ball"] = float(rho[np.abs(X) < ball].sum())
    if uniform_region is not None:
        lo_, hi_ = uniform_region
        out["outside_uniform_region"] = float(rho[(X < lo_) | (X > hi_)].sum())
        if out["outside_uniform_region"] > tol:
            reasons.append("outside the uniform region [%g, %g]: %.3g > %g" % (lo_, hi_, out["outside_uniform_region"], tol))
    out["accepted"] = not reasons
    out["reasons"] = reasons
    return out


def packet_diag(psi, X, h, bands, args, Qe=None, lost=None, surface=None, ball=None, thr=None):
    """every packet's record (9.11, 9.15, 9.18): geometry, band weights, exact initial velocity, momentum readout, Killing content."""
    d = {"geometry": packet_geometry(psi, X, args.n_tail, args.tail_tol, surface=surface, ball=ball, lost=lost),
         "bands": band_weights(psi, bands), "v_exact": exact_velocity(h, psi, X), "v_exact_pos_band": exact_velocity_pos_band(h, psi, X, bands),
         "momentum": momentum_readout(psi, bands)}
    if Qe is not None:
        d["killing"] = killing_content(Qe[1], Qe[0], psi, thr)
    return d


def track_v7(Q, e, psi0, times, X, regions, bands, n_tail, kres=None, v4_kres=None):
    """v4's track with the fixes: centre, width, region norms, the end tails at every output (9.11), the circular band-resolved momentum
    readout (9.18; v4's arithmetic carrier kept as carrier_arith_v4), the momentum-resolved norms in the kres masks; v4_kres = R gives
    v4's inside_short_wave/inside_long_wave (X < R) under v4's names."""
    rec = []
    for t in times:
        psi = V4.propagate(Q, e, t, psi0)
        c, wdt = V4.centre_width(psi, X)
        kv, pv = V4.momentum_distribution(psi)
        rho = V4.density(psi)
        row = {"t": float(t), "centre": c, "width": wdt, "carrier_arith_v4": float(kv @ pv),
               **{name: float(rho[sel].sum()) for name, sel in regions.items()},
               "tail_inner": float(rho[:n_tail].sum()), "tail_outer": float(rho[-n_tail:].sum()),
               "momentum": momentum_readout(psi, bands)}
        if v4_kres is not None:
            row["inside_short_wave"], row["inside_long_wave"] = V4.kresolved(psi, X, v4_kres)
        for name, (Rc, mask) in (kres or {}).items():
            sw, lw = kresolved_mask(psi, mask)
            row["kres_" + name] = {"R": float(Rc), "short_wave": sw, "long_wave": lw, "momentum": momentum_readout(psi, bands, mask=mask)}
        rec.append(row)
    return rec


# ---------------------------------------------------------------- the symbol's rays (D1, D2: the ODE and the ensemble)
def planar_profile(x, rs, Rm):
    """the planar D1 flow of spec 11.3: -sqrt(r_s/|x|) for |x| >= R_m, -sqrt(r_s/R_m^3)|x| inside (even in x)."""
    ax = np.abs(np.asarray(x, float))
    return np.where(ax >= Rm, -np.sqrt(rs / np.maximum(ax, 1e-300)), -np.sqrt(rs / Rm ** 3) * ax)


def planar_dprofile(x, rs, Rm):
    x = np.asarray(x, float)
    ax, sg = np.abs(x), np.sign(x)
    return np.where(ax >= Rm, 0.5 * np.sqrt(rs) * np.maximum(ax, 1e-300) ** -1.5 * sg, -np.sqrt(rs / Rm ** 3) * sg)


def planar_rhs(m, rs, Rm):
    """Hamilton's equations of omega = sqrt(m^2 + sin^2 p) + v(x) sin(2p)/2 (no centrifugal term): p = 0 is invariant, dx/dt = v."""
    def f(y):
        x, p = y
        E = np.sqrt(m * m + np.sin(p) ** 2)
        return np.vstack([np.sin(p) * np.cos(p) / E + planar_profile(x, rs, Rm) * np.cos(2 * p),
                          -planar_dprofile(x, rs, Rm) * np.sin(2 * p) / 2])
    return f


def radial_rhs(m, kj, rs, Rm):
    """v4's symbol_ode right-hand side (with the centrifugal force kappa_j^2/(E R^3)) for an array of rays; R floored at 0.5."""
    C = math.sqrt(rs / Rm ** 3)

    def f(y):
        R, p = np.maximum(y[0], 0.5), y[1]
        E = np.sqrt(m * m + (kj / R) ** 2 + np.sin(p) ** 2)
        v = np.where(R >= Rm, -np.sqrt(rs / R), -C * R)
        dv = np.where(R >= Rm, 0.5 * np.sqrt(rs) * R ** -1.5, -C)
        return np.vstack([np.sin(p) * np.cos(p) / E + v * np.cos(2 * p), (kj * kj / R ** 3) / E - dv * np.sin(2 * p) / 2])
    return f


def planar_symbol_ode(x0, p0, m, rs, Rm, ts):
    rhs = planar_rhs(m, rs, Rm)
    sol = solve_ivp(lambda t, y: rhs(np.asarray(y)[:, None])[:, 0], (ts[0], ts[-1]), [x0, p0], t_eval=ts, rtol=1e-10, atol=1e-12)
    return sol.y[0].tolist(), sol.y[1].tolist()


def ray_ensemble(rhs, x0, p0, sigma_x, ts, n=24, dt=0.01, floor=None):
    """BUILDER (labelled): the symbol's Hamilton flow averaged over the Gaussian packet's initial Wigner distribution (x ~ N(x0, sigma_x^2),
    p ~ N(p0, (1/(2 sigma_x))^2)), Gauss-Hermite n x n, fixed-step RK4 at dt and 2 dt (the integrator check): the finite packet's
    symbol prediction for the centre and the classical width."""
    xh, wh = np.polynomial.hermite_e.hermegauss(n)
    wts = np.outer(wh, wh).ravel() / (2 * PI)
    X0 = np.repeat(x0 + sigma_x * xh, n)
    P0 = np.tile(p0 + xh / (2 * sigma_x), n)
    res = {}
    for name, step in (("dt", dt), ("2dt", 2 * dt)):
        y = np.vstack([X0, P0])
        t = float(ts[0])
        means, stds, pmeans, low = [], [], [], 0
        for tn in ts:
            nst = int(round((tn - t) / step))
            if nst > 0:
                hh = (tn - t) / nst
                for _ in range(nst):
                    k1 = rhs(y)
                    k2 = rhs(y + 0.5 * hh * k1)
                    k3 = rhs(y + 0.5 * hh * k2)
                    k4 = rhs(y + hh * k3)
                    y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = float(tn)
            mu = float(wts @ y[0])
            means.append(mu)
            stds.append(float(np.sqrt(wts @ (y[0] - mu) ** 2)))
            pmeans.append(float(wts @ y[1]))
            if floor is not None:
                low = max(low, int(np.sum(y[0] < floor)))
        res[name] = (means, stds, pmeans, low)
    m1, s1, p1, low1 = res["dt"]
    out = {"centre": m1, "width": s1, "p_mean": p1, "n_rays": n * n, "dt": dt,
           "max_centre_change_dt_vs_2dt": float(np.max(np.abs(np.array(m1) - np.array(res["2dt"][0])))),
           "label": "BUILDER: the symbol's rays over the packet's initial Wigner Gaussian (Gauss-Hermite, RK4)"}
    if floor is not None:
        out["max_rays_below_%g" % floor] = low1
    return out


def centre_summary(times, track, refs, t_cut=None):
    """the largest relative deviations of the measured centre from each reference curve (to t_cut and over the run)."""
    t = np.asarray(times, float)
    c = np.array([r["centre"] for r in track])
    out = {}
    for name, ref in refs.items():
        ref = np.asarray(ref, float)
        n = min(len(ref), len(c))
        rel = np.abs(c[:n] - ref[:n]) / np.abs(ref[:n])
        out["max_rel_centre_minus_" + name] = float(rel.max())
        if t_cut is not None:
            sel = t[:n] <= t_cut + 1e-9
            out["max_rel_centre_minus_%s_to_rs" % name] = float(rel[sel].max())
    return out


# ---------------------------------------------------------------- modes
def mode_symbol(args):
    """A1 re-emitted: C1 (v4's ring check, unchanged), v4's vg_max_at_surface beside vg_max_branch (9.3), the branch-qualified turning
    table at both signs of k and the one-sided massless k = 0 entries (9.2) beside v4's squared law."""
    ks = PI * np.array([1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8])
    out = {"label": "the symbol (A1), flow_chart version 5", "C1_ring_max_error": {}, "vg_max_surface_v4": {}, "vg_max_branch": {},
           "turning_table": {}, "turning_table_v4_squared": {}}
    for mc in (0.0, 0.1, 0.3):
        for w in (0.0, 0.5, 1.0, 1.5, 2.0):
            out["C1_ring_max_error"][f"mc={mc},w={w}"] = V4.ring_dispersion_check(64, mc, w)
        out["vg_max_surface_v4"][f"mc={mc}"] = V4.vg_max_at_surface(mc)
        out["vg_max_branch"][f"mc={mc}"] = vg_max_branch(mc, 1.0)
        tab = [turning_entry(s * kk, mc) for kk in ks for s in (1.0, -1.0)]
        tab += [turning_entry(0.0, mc, side=1.0), turning_entry(0.0, mc, side=-1.0)]
        out["turning_table"][f"mc={mc}"] = tab
        with np.errstate(all="ignore"):
            out["turning_table_v4_squared"][f"mc={mc}"] = {"k": ks.tolist(), "R_turn_over_rs": V4.turning_ratio(ks, mc).tolist()}
    out["bound_check_9_over_8"] = out["vg_max_branch"]["mc=0.0"]["interval_0_pi2"]["max"] - 9 / 8
    return out


def mode_radial(args):
    """A2, A3, D1, D2, D4 on the PG radial channel with the version-5 fixes (D3 deferred)."""
    N = args.nsites
    rs, Rm, m, kj = args.rs, args.Rm, args.mass, args.kappa_j
    mz, yy, R = V4.radial_fields(N, m, kj)
    v = V4.pg_profile(R, rs, Rm)
    lab = {"label": LABEL_PG, "launch": args.launch, "rs": rs, "Rm": Rm, "mass": m, "kappa_j": kj, "N": N, "R_max": float(R[-1]),
           "n_tail": args.n_tail, "tail_tol": args.tail_tol, "keep_rejected": bool(args.keep_rejected),
           "tail_note": "radial chain: tail_inner is the norm on the n_tail sites at the regular centre (R = 1 .. n_tail, a physical "
                        "boundary; the packet's initial construction is rejected there, its later arrival is recorded), tail_outer the box end"}
    if args.launch == "D3":
        return dict(lab, deferred="D3 version 5 (spec 11.3) is not in this version; v4's D3 records stand until it lands")
    if args.launch in ("D1", "D2") and m == 0:
        return dict(lab, rejected="massless rest dust rejected (spec 11.3; red team 9.4)")
    h = V4.chain_hamiltonian(mz, yy, v)
    e, Q = V4.spectrum(h)
    bands = make_bands(mz, yy)
    loc = lambda R0: (float(V4.pg_profile(R0, rs, Rm)), kj / R0)  # noqa: E731
    vfun = lambda x: V4.pg_profile(x, rs, Rm)  # noqa: E731

    def diag(psi, R0, sg, k0):
        d = packet_diag(psi, R, h, bands, args, Qe=(e, Q), lost=gaussian_lost(R, R0, sg), surface=rs, ball=Rm, thr=m)
        d["v_symbol_spatial"] = symbol_vg_spatial(psi, R, k0, sg, m, kj, vfun)
        return d

    if args.launch == "A2":
        rows = []
        carriers = [(PI / 16, 32.0), (PI / 8, 16.0), (3 * PI / 16, 16.0), (PI / 4, 16.0), (-3 * PI / 8, 8.0)]
        clab = {}
        for k0, sg in carriers:
            kabs = abs(k0)
            Rt = rs * float(V4.turning_ratio(kabs, m))
            cl = carrier_label(k0, m, kj, rs, Rm, float(R[-1]))
            clab["%.6f" % k0] = dict(cl, sigma_x=sg)
            if Rt < 1e-9:
                launches = [0.25 * rs, 0.5 * rs, 0.75 * rs]
                note = "no zero: launched at 0.25, 0.5, 0.75 r_s (v4's scan kept: the measurement that no zero exists)"
            else:
                launches = [f * Rt for f in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)]
                note = None if cl["label"] == "exterior zero" else cl["label"] + " (v4's scan positions kept)"
            for R0 in launches:
                if R0 - 4 * sg < 2 or R0 + 4 * sg > R[-1] - 2:
                    rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "skipped": "outside the chain's 4 sigma margin"})
                    continue
                vloc, aloc = loc(R0)
                psi0 = V4.gaussian_carrier(R, R0, sg, k0, m, aloc)
                d0 = diag(psi0, R0, sg, k0)
                if not d0["geometry"]["accepted"] and not args.keep_rejected:
                    rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "rejected": "inadequate geometry", "initial": d0})
                    continue
                vm, cs = V4.initial_velocity(Q, e, psi0, R)
                dk = 1e-5
                dRt = rs * abs(float(V4.turning_ratio(kabs + dk, m)) - float(V4.turning_ratio(kabs - dk, m))) / (2 * dk)
                dv = (float(V4.pg_profile(R0 + 1e-4, rs, Rm)) - float(V4.pg_profile(R0 - 1e-4, rs, Rm))) / 2e-4
                dvg = (V4.symbol_vg(k0, m, kj / (R0 + 1e-4), float(V4.pg_profile(R0 + 1e-4, rs, Rm))) -
                       V4.symbol_vg(k0, m, kj / (R0 - 1e-4), float(V4.pg_profile(R0 - 1e-4, rs, Rm)))) / 2e-4
                budget = dRt / (2 * sg) + sg * abs(dv) / max(abs(dvg), 1e-300)
                rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "R_turn_law": Rt, "note": note, "v_measured": vm,
                             "v_symbol_packet": V4.packet_avg_vg(k0, sg, m, aloc, vloc), "turn_budget": budget, "centres": cs,
                             "v_exact": d0["v_exact"], "v_exact_pos_band": d0["v_exact_pos_band"],
                             "v_symbol_spatial": d0["v_symbol_spatial"], "initial": d0})
        turn = {}
        for k0, sg in carriers:
            key = "%.6f" % k0
            rr = [r for r in rows if r.get("k0") == k0 and "v_measured" in r]
            ent = {}
            for vkey, name in (("v_measured", "measured"), ("v_exact", "measured_exact"), ("v_exact_pos_band", "measured_exact_pos_band"),
                               ("v_symbol_spatial", "symbol_spatial_zero")):
                for r1, r2 in zip(rr[:-1], rr[1:]):
                    if r1[vkey] * r2[vkey] < 0:
                        x1, x2, y1, y2 = r1["R_launch"], r2["R_launch"], r1[vkey], r2[vkey]
                        ent[name] = x1 - y1 * (x2 - x1) / (y2 - y1)
                        if name == "measured":
                            ent["budget"] = 0.5 * (r1["turn_budget"] + r2["turn_budget"])
            if ent:
                ent.update(law=clab[key]["exterior_root"], law_squared_v4=rr[0]["R_turn_law"], label=clab[key]["label"])
                turn[key] = ent
        return dict(lab, rows=rows, turning=turn, carriers=clab,
                    builder_note="the launch scan keeps v4's positions (f x the squared law's radius); the law is the unsquared root")
    if args.launch == "A3":
        k0, sg, R0 = float(np.arccos(0.25)), 16.0, rs
        vloc, aloc = loc(R0)
        psi0 = V4.gaussian_carrier(R, R0, sg, k0, m, aloc)
        d0 = diag(psi0, R0, sg, k0)
        vm, cs = V4.initial_velocity(Q, e, psi0, R)
        kk = np.linspace(1e-6, PI / 2 - 1e-6, 200001)
        return dict(lab, k0=k0, R_launch=R0, v_measured=vm, v_symbol_packet=V4.packet_avg_vg(k0, sg, m, aloc, vloc),
                    max_k_vg=float(np.max(V4.symbol_vg(kk, m, aloc, vloc))), bound_9_8=9 / 8, centres=cs, initial=d0)
    if args.launch in ("D1", "D2"):
        R0, sg = 4 * rs, 8.0
        vloc, aloc = loc(R0)
        En = lambda pp: math.sqrt(m * m + aloc * aloc + math.sin(pp) ** 2)  # noqa: E731
        extra = {}
        if args.launch == "D1":
            p0 = 0.0
        else:
            f = lambda pp: math.sin(pp) * math.cos(pp) / En(pp) + vloc * math.cos(2 * pp)  # noqa: E731
            f4 = lambda pp: math.sin(pp) * math.cos(pp) / En(pp) + vloc  # noqa: E731
            p0 = float(brentq(f, 1e-9, PI / 4, xtol=1e-15))
            extra = {"rest_condition": "sin p0 cos p0/E = -v(R0) cos 2p0 (exact; red team 4.4, spec 11.3)",
                     "p0_v4_condition": float(brentq(f4, 1e-9, PI / 4, xtol=1e-15)),
                     "v4_condition": "sin p0 cos p0/E = -v(R0) (v4: the small-p0 approximation)"}
        extra["v_symbol_at_R0_p0"] = float(V4.symbol_vg(p0, m, aloc, vloc))
        psi0 = V4.gaussian_carrier(R, R0, sg, p0, m, aloc)
        d0 = diag(psi0, R0, sg, p0)
        if not d0["geometry"]["accepted"] and not args.keep_rejected:
            return dict(lab, R0=R0, p0=p0, sigma_x=sg, rejected="inadequate geometry", initial=d0, **extra)
        tc = (2.0 / 3.0) * (R0 ** 1.5 - Rm ** 1.5) / np.sqrt(rs)
        tcross = (2.0 / 3.0) * (R0 ** 1.5 - rs ** 1.5) / np.sqrt(rs)
        times = np.arange(0, tc + 1e-9, args.tsample)
        rec = track_v7(Q, e, psi0, times, R, {"inside_rs": R < rs, "inside_Rm": R < Rm}, bands, args.n_tail,
                       kres={"Rm": (Rm, R < Rm)})
        out = dict(lab, R0=R0, p0=p0, sigma_x=sg, sigma_k=1 / (2 * sg), initial=d0, times=times.tolist(), track=rec, **extra)
        Rs, Ps = V4.symbol_ode(R0, p0, m, kj, rs, Rm, times)
        out["symbol_ode_R"], out["symbol_ode_p"] = Rs, Ps
        out["symbol_ensemble"] = ray_ensemble(radial_rhs(m, kj, rs, Rm), R0, p0, sg, times, floor=1.0)
        refs = {"ode": Rs, "ensemble": out["symbol_ensemble"]["centre"]}
        if args.launch == "D1":
            out["dust_law"] = V4.dust_law(times, R0, rs, Rm).tolist()
            out["crossing_time_law"] = tcross
            refs["law"] = out["dust_law"]
        summ = centre_summary(times, rec, refs, t_cut=tcross)
        sel = [r for r in rec if r["t"] <= tcross + 1e-9]
        summ["max_abs_pos_band_circ_mean_to_rs"] = float(max(abs(r["momentum"]["pos"]["circ_mean"]) for r in sel))
        summ["sigma_k_over_4"] = 1 / (8 * sg)
        summ["max_tail_inner"] = float(max(r["tail_inner"] for r in rec))
        summ["max_tail_outer"] = float(max(r["tail_outer"] for r in rec))
        out["summary"] = summ
        return out
    if args.launch == "D4":
        k0, sg, R0 = PI / 2, 8.0, rs / 2
        vloc, aloc = loc(R0)
        psi0 = V4.gaussian_carrier(R, R0, sg, k0, m, aloc)
        d0 = diag(psi0, R0, sg, k0)
        if not d0["geometry"]["accepted"] and not args.keep_rejected:
            return dict(lab, k0=k0, R0=R0, sigma_x=sg, rejected="inadequate geometry", initial=d0)
        times = np.arange(0, 3 * rs + 1e-9, args.tsample)
        rec = track_v7(Q, e, psi0, times, R, {"inside_rs": R < rs, "beyond_2rs": R > 2 * rs}, bands, args.n_tail, v4_kres=rs)
        return dict(lab, k0=k0, R0=R0, sigma_x=sg, initial=d0, times=times.tolist(), track=rec,
                    w_along=[float(-V4.pg_profile(r["centre"], rs, Rm)) for r in rec],
                    geometry_verdict=("accepted" if d0["geometry"]["accepted"] else
                                      "rejected (evolved under --keep-rejected): " + "; ".join(d0["geometry"]["reasons"])))
    raise ValueError(args.launch)


def mode_planar_dust(args):
    """spec 11.3's D1 planar: the two-component planar chain (no centrifugal term), the even PG-like flow with the ball, the packet at rest
    (p = 0) released at x0 = 4 r_s; the centre against the dust law, the planar symbol's ODE and the symbol's ray ensemble."""
    m, rs, Rm, sg, L = args.mass, args.rs, args.Rm, args.sigma_x, int(args.half_length)
    x0 = args.x0 if args.x0 is not None else 4 * rs
    lab = {"label": LABEL_PLANAR_DUST, "launch": "D1 planar", "rs": rs, "Rm": Rm, "mass": m, "kappa_j": None, "sigma_x": sg,
           "sigma_k": 1 / (2 * sg), "x0": x0, "half_length": L, "N": 2 * L + 1, "n_tail": args.n_tail, "tail_tol": args.tail_tol,
           "keep_rejected": bool(args.keep_rejected)}
    if m == 0:
        return dict(lab, rejected="massless rest dust rejected (spec 11.3; red team 9.4)")
    lab["sigma_k_over_m"] = 1 / (2 * sg * m)
    X = np.arange(-L, L + 1, dtype=float)
    N = len(X)
    mz, yy = np.full(N, float(m)), np.zeros(N)
    h = V4.chain_hamiltonian(mz, yy, planar_profile(X, rs, Rm))
    e, Q = V4.spectrum(h)
    bands = make_bands(mz, yy)
    psi0 = V4.gaussian_carrier(X, x0, sg, 0.0, m, 0.0)
    d0 = packet_diag(psi0, X, h, bands, args, Qe=(e, Q), lost=gaussian_lost(X, x0, sg), surface=rs, ball=Rm, thr=m)
    d0["v_symbol_spatial"] = symbol_vg_spatial(psi0, X, 0.0, sg, m, 0.0, lambda x: planar_profile(x, rs, Rm))
    if not d0["geometry"]["accepted"] and not args.keep_rejected:
        return dict(lab, rejected="inadequate geometry", initial=d0)
    tm = (2.0 / 3.0) * (x0 ** 1.5 - Rm ** 1.5) / np.sqrt(rs)
    tcross = (2.0 / 3.0) * (x0 ** 1.5 - rs ** 1.5) / np.sqrt(rs)
    times = np.arange(0, (args.tmax if args.tmax_planar else tm) + 1e-9, args.tsample)
    rec = track_v7(Q, e, psi0, times, X, {"inside_rs": np.abs(X) < rs, "inside_Rm": np.abs(X) < Rm, "x_negative": X < 0}, bands,
                   args.n_tail, kres={"Rm": (Rm, np.abs(X) < Rm)})
    law = V4.dust_law(times, x0, rs, Rm).tolist()
    xs, ps = planar_symbol_ode(x0, 0.0, m, rs, Rm, times)
    ens = ray_ensemble(planar_rhs(m, rs, Rm), x0, 0.0, sg, times)
    summ = centre_summary(times, rec, {"law": law, "ode": xs, "ensemble": ens["centre"]}, t_cut=tcross)
    sel = [r for r in rec if r["t"] <= tcross + 1e-9]
    summ["max_abs_pos_band_circ_mean_to_rs"] = float(max(abs(r["momentum"]["pos"]["circ_mean"]) for r in sel))
    summ["sigma_k_over_4"] = 1 / (8 * sg)
    summ["max_tail_inner"] = float(max(r["tail_inner"] for r in rec))
    summ["max_tail_outer"] = float(max(r["tail_outer"] for r in rec))
    summ["ode_minus_law_max_abs"] = float(np.max(np.abs(np.array(xs) - np.array(law[:len(xs)]))))
    return dict(lab, initial=d0, times=times.tolist(), track=rec, dust_law=law, crossing_time_law=tcross, time_to_Rm_law=tm,
                symbol_ode_x=xs, symbol_ode_p=ps, symbol_ensemble=ens, summary=summ)


def mode_a2compare(args):
    """spec 11.3's A2 comparator on stored flow_chart_v4.py A2 records (no evolution): per carrier the unsquared zero-velocity radii on
    pg_profile with spec 11.3's labels, the measured locus from the record's fitted slopes and from the exact initial velocities; per
    launch the packet rebuilt exactly as v4 built it (gaussian_carrier from the record's parameters), its exact <i[h, R]>, the spatially
    averaged symbol, geometry, band weights and momentum readout."""
    out = {"label": LABEL_PG + "; spec 11.3's A2 comparator on stored flow_chart_v4.py records, no new evolution", "records": {}}
    for path in args.records:
        d = json.load(open(path))
        N, rs, Rm, m, kj = int(d["N"]), float(d["rs"]), float(d["Rm"]), float(d["mass"]), float(d["kappa_j"])
        mz, yy, R = V4.radial_fields(N, m, kj)
        h = V4.chain_hamiltonian(mz, yy, V4.pg_profile(R, rs, Rm))
        bands = make_bands(mz, yy)
        vfun = lambda x, rs=rs, Rm=Rm: V4.pg_profile(x, rs, Rm)  # noqa: E731
        rows, groups = [], {}
        for r in d["rows"]:
            k0, sg, R0 = float(r["k0"]), float(r["sigma_x"]), float(r["R_launch"])
            g = groups.setdefault("%.6f" % k0, {"k0": k0, "sigma_x": sg, "rows": []})
            if "skipped" in r:
                rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "skipped": r["skipped"]})
                continue
            psi0 = V4.gaussian_carrier(R, R0, sg, k0, m, kj / R0)
            row = {"k0": k0, "sigma_x": sg, "R_launch": R0, "v_fitted_v4": r["v_measured"], "v_exact": exact_velocity(h, psi0, R),
                   "v_exact_pos_band": exact_velocity_pos_band(h, psi0, R, bands),
                   "v_symbol_packet_v4": r["v_symbol_packet"], "v_symbol_spatial": symbol_vg_spatial(psi0, R, k0, sg, m, kj, vfun),
                   "centre0_rebuilt_minus_record": V4.centre_width(psi0, R)[0] - r["centres"][0],
                   "geometry": packet_geometry(psi0, R, args.n_tail, args.tail_tol, surface=rs, ball=Rm, lost=gaussian_lost(R, R0, sg)),
                   "bands": band_weights(psi0, bands), "momentum0": momentum_readout(psi0, bands)}
            rows.append(row)
            g["rows"].append(row)
        carriers = {}
        for key, g in groups.items():
            ent = dict(carrier_label(g["k0"], m, kj, rs, Rm, float(R[-1])), sigma_x=g["sigma_x"], n_launches_run=len(g["rows"]))
            rr = g["rows"]
            for vkey, name in (("v_fitted_v4", "measured_fitted"), ("v_exact", "measured_exact"),
                               ("v_exact_pos_band", "measured_exact_pos_band"), ("v_symbol_spatial", "symbol_spatial_zero")):
                for r1, r2 in zip(rr[:-1], rr[1:]):
                    if r1[vkey] * r2[vkey] < 0:
                        x1, x2, y1, y2 = r1["R_launch"], r2["R_launch"], r1[vkey], r2[vkey]
                        ent[name] = x1 - y1 * (x2 - x1) / (y2 - y1)
            law = ent["exterior_root"]
            for name in ("measured_fitted", "measured_exact", "measured_exact_pos_band", "symbol_spatial_zero"):
                if name in ent and law is not None and ent["label"] == "exterior zero":
                    ent[name + "_minus_law"] = ent[name] - law
                    ent[name + "_rel"] = (ent[name] - law) / law
            if key in d.get("turning", {}):
                ent["record_turning_v4"] = d["turning"][key]
            if not rr:
                ent["note"] = "no launch ran in the record (all outside the chain's 4 sigma margin)"
            carriers[key] = ent
        out["records"][os.path.basename(path)] = {"source_script": d.get("script"), "rs": rs, "Rm": Rm, "mass": m, "kappa_j": kj,
                                                  "N": N, "carriers": carriers, "rows": rows}
    return out


def mode_convert(args):
    """part (b) on the planar chain with the version-5 fixes (9.6, 9.7, 9.9, 9.10, 9.11, 9.12); retired for the planar part (the record's
    package is the instrument), smoke-tested only."""
    kappa = 1.0 / args.inv_kappa
    delta, width = args.delta, args.delta * args.inv_kappa
    N = args.nsites
    mz, yy, x = V4.planar_fields(N, args.mass)
    vfinal = V4.tanh_profile(x, delta, width)
    vext = -1.0 + delta
    bands = make_bands(mz, yy)
    V0 = bands["Q0"][:, bands["e0"] < 0]
    Uramp = V4.ramp_unitary(mz, yy, vfinal, args.ramp_time, args.dt)
    ef, Qf = V4.spectrum(V4.chain_hamiltonian(mz, yy, vfinal))
    unit = float(np.linalg.norm(Uramp.conj().T @ Uramp - np.eye(2 * N)))
    mc = float(mz[0])
    th = threshold(mc, vext, "out")
    om_th = th["omega_th"]
    x0 = int(round((N - 1) / 2 + args.x0_over_kappa / kappa))
    results = {}
    for r in args.ratios:
        omega_c, key = r * kappa, f"r={r}"
        if not omega_c > om_th:
            results[key] = {"skipped": "omega_c <= omega_th (the lattice threshold, 9.7)", "omega_th": om_th}
            continue
        sigma = min(omega_c / 4, (omega_c - om_th) / 4)
        bm = branch_momenta(omega_c, sigma, mc, vext, om_th, outgoing=True)
        if bm is None:
            results[key] = {"skipped": "no outgoing momenta in the window above omega_th", "omega_th": om_th}
            continue
        k, om, rho, q = bm
        w = build_packet(N, x0, k, rho, q, lambda kk: V4.positive_spinor(kk, mc))
        geo = packet_geometry(w, x, args.n_tail, args.tail_tol, surface=0.0, uniform_region=(3 * width, float(x[-1])),
                              reject_surface=True)
        mu_t = float(np.sum(rho * q * om))
        entry = {"omega_c": omega_c, "sigma_omega": sigma, "omega_th": om_th, "x0": float(x[x0]), "geometry": geo,
                 "bands": band_weights(w, bands), "killing": killing_content(Qf, ef, w, om_th),
                 "killing_target": {"mean": mu_t, "std": float(np.sqrt(np.sum(rho * q * (om - mu_t) ** 2)))},
                 "c_pred": c_mono(r), "c_w": c_w(k, om, rho, q, kappa), "F_w(1)": fermi_average(k, om, rho, q, 1.0, kappa)}
        if not geo["accepted"] and not args.keep_rejected:
            entry["rejected"] = "inadequate geometry: " + "; ".join(geo["reasons"])
            results[key] = entry
            continue
        plateau = {}
        for kt in np.arange(140, 201, 10):
            nw = V4.occupation_backward(w, Uramp, Qf, ef, kt / kappa - args.ramp_time, V0)
            plateau[str(int(kt))] = dict(invert_theta(nw, k, om, rho, q, kappa), n_w=nw)
        ths = np.array([p["theta"] for p in plateau.values() if p["theta"] is not None])
        entry.update(plateau=plateau, n_fits=int(len(ths)))
        if len(ths):
            entry.update(theta_mean=float(ths.mean()), theta_spread=float(np.ptp(ths)), c_r=float((1 - ths.mean()) * args.inv_kappa ** 2))
        if not geo["accepted"]:
            entry["geometry_verdict"] = "rejected (evolved under --keep-rejected): " + "; ".join(geo["reasons"])
        results[key] = entry
    return {"label": LABEL_TANH, "kappa_a": kappa, "mass": args.mass, "m_over_kappa": args.mass / kappa, "delta": delta, "width": width,
            "N": N, "x0_index": x0, "ramp_time": args.ramp_time, "dt": args.dt, "threshold": th, "C2_unitarity": unit,
            "n_tail": args.n_tail, "tail_tol": args.tail_tol, "keep_rejected": bool(args.keep_rejected), "results": results}


# ---------------------------------------------------------------- the validation against v4
def _md(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = min(x.size, y.size)
    return float(np.max(np.abs(x.ravel()[:n] - y.ravel()[:n]))) if n else None


def _cmp_A2(a, b):
    key = lambda r: (round(float(r["k0"]), 9), round(float(r["R_launch"]), 6))  # noqa: E731
    A, B = {key(r): r for r in a["rows"]}, {key(r): r for r in b["rows"]}
    common = sorted(set(A) & set(B))
    res = {"rows_v4": len(A), "rows_v7": len(B), "rows_common": len(common), "only_v4": [list(c) for c in sorted(set(A) - set(B))],
           "only_v7": [list(c) for c in sorted(set(B) - set(A))],
           "skipped_identical": [c for c in common if "skipped" in A[c]] == [c for c in common if "skipped" in B[c]],
           "reproduced": {}, "changed": [], "new_fields": sorted(set().union(*[set(B[c]) - set(A[c]) for c in common]))}
    for q in ("R_launch", "R_turn_law", "v_measured", "v_symbol_packet", "turn_budget"):
        dd = [abs(A[c][q] - B[c][q]) for c in common if q in A[c] and q in B[c]]
        res["reproduced"][q] = {"max_abs_diff": max(dd) if dd else None, "n": len(dd)}
    dd = [_md(A[c]["centres"], B[c]["centres"]) for c in common if "centres" in A[c] and "centres" in B[c]]
    res["reproduced"]["centres"] = {"max_abs_diff": max(dd) if dd else None, "n": len(dd)}
    rel = [abs(A[c]["v_measured"] - B[c]["v_measured"]) / abs(A[c]["v_measured"]) for c in common if "v_measured" in A[c] and "v_measured" in B[c]]
    res["reproduced"]["v_measured (relative)"] = {"max_abs_diff": max(rel) if rel else None, "n": len(rel), "relative": True}
    seen = set()
    for c in common:
        n4, n7 = A[c].get("note"), B[c].get("note")
        if ("skipped" not in A[c]) and n4 != n7 and (c[0], n4, n7) not in seen:
            seen.add((c[0], n4, n7))
            res["changed"].append({"quantity": "rows[k0=%.6f].note" % c[0], "item": "9.2, 9.3, spec 11.3 (the labels)", "old": n4, "new": n7})
    for kk in sorted(set(a.get("turning", {})) | set(b.get("turning", {}))):
        t4, t7 = a.get("turning", {}).get(kk), b.get("turning", {}).get(kk)
        if t4 is None or t7 is None:
            res["changed"].append({"quantity": "turning[%s]" % kk, "item": "-", "old": t4, "new": t7})
            continue
        for q in ("measured", "budget"):
            res["reproduced"]["turning[%s].%s" % (kk, q)] = {"max_abs_diff": abs(t4[q] - t7[q]), "n": 1}
        res["reproduced"]["turning[%s].law_squared_v4 vs v4 law" % kk] = {"max_abs_diff": abs(t4["law"] - t7["law_squared_v4"]), "n": 1}
        res["changed"].append({"quantity": "turning[%s].law" % kk, "item": "9.15 (red team 2.5): the unsquared zero-velocity equation with "
                               "kappa_j^2/R^2 on pg_profile, solved for R", "old": t4["law"], "new": t7["law"], "label": t7.get("label"),
                               "measured_minus_law_old": t4["measured"] - t4["law"],
                               "measured_minus_law_new": (t7["measured"] - t7["law"]) if t7.get("law") is not None else None})
    return res


def _cmp_track(a, b, qs, extra_lists=()):
    ta, tb = a["track"], b["track"]
    n = min(len(ta), len(tb))
    res = {"n_times_v4": len(ta), "n_times_v7": len(tb), "reproduced": {}, "changed": [],
           "new_fields": sorted(set(tb[0]) - set(ta[0])) if n else []}
    res["reproduced"]["times"] = {"max_abs_diff": _md(a["times"], b["times"]), "n": n}
    for q in qs:
        if n and q in ta[0] and q in tb[0]:
            res["reproduced"][q] = {"max_abs_diff": _md([r[q] for r in ta[:n]], [r[q] for r in tb[:n]]), "n": n}
    if n and "carrier" in ta[0] and "carrier_arith_v4" in tb[0]:
        res["reproduced"]["carrier (v4) vs carrier_arith_v4 (v7)"] = {
            "max_abs_diff": _md([r["carrier"] for r in ta[:n]], [r["carrier_arith_v4"] for r in tb[:n]]), "n": n}
        step = max(1, n // 8)
        res["changed"].append({"quantity": "the carrier readout", "item": "9.18: the circular, band-resolved readout replaces the "
                               "arithmetic k @ p", "t": [ta[i]["t"] for i in range(0, n, step)],
                               "old_arith": [ta[i]["carrier"] for i in range(0, n, step)],
                               "new_circ_mean_all": [tb[i]["momentum"]["all"]["circ_mean"] for i in range(0, n, step)],
                               "new_circ_mean_pos_band": [tb[i]["momentum"]["pos"].get("circ_mean") for i in range(0, n, step)]})
    for name in extra_lists:
        if name in a and name in b:
            res["reproduced"][name] = {"max_abs_diff": _md(a[name], b[name]), "n": min(len(a[name]), len(b[name]))}
    return res


def _cmp_A1(a, b):
    res = {"reproduced": {}, "changed": []}
    ca, cb = a["C1_ring_max_error"], b["C1_ring_max_error"]
    res["reproduced"]["C1_ring_max_error"] = {"max_abs_diff": max(abs(ca[k] - cb[k]) for k in ca), "n": len(ca)}
    for mc, v in a["vg_max_surface"].items():
        res["reproduced"]["vg_max_surface[%s]" % mc] = {"max_abs_diff": _md(v, b["vg_max_surface_v4"][mc]), "n": 2}
        res["changed"].append({"quantity": "vg_max at w = 1 [%s]" % mc, "item": "9.3: labelled by interval, refined",
                               "old": v, "new": b["vg_max_branch"][mc]})
    for mc, tab in a["turning_table"].items():
        for k, r in zip(tab["k"], tab["R_turn_over_rs"]):
            new = [e for e in b["turning_table"][mc] if abs(abs(e["k"]) - k) < 1e-12]
            res["changed"].append({"quantity": "turning_table[%s] k = %.6f" % (mc, k), "item": "9.2, 9.3 (branch-qualified, both signs)",
                                   "old_R_over_rs": r, "new": [{"k": e["k"], "status": e["status"], "R_over_rs": e.get("R_over_rs"),
                                                                "branch": e.get("branch")} for e in new]})
    return res


def mode_compare_v4(args):
    """the validation of spec 11.4: v7 records against v4 records (reproduced quantities with their largest differences; changed
    quantities with old and new values and the red-team item that changed them)."""
    out = {"label": "flow_chart_v7.py (the lead's version 5) against flow_chart_v4.py on v4's records (spec 11.4)", "pairs": {}}
    for pair in args.pairs:
        p4, p7 = pair.split(":")
        a, b = json.load(open(p4)), json.load(open(p7))
        name = os.path.basename(p4) + " vs " + os.path.basename(p7)
        if a.get("launch") == "A2":
            res = _cmp_A2(a, b)
        elif a.get("launch") == "D4":
            res = _cmp_track(a, b, ("centre", "width", "inside_rs", "beyond_2rs", "inside_short_wave", "inside_long_wave"), ("w_along",))
        elif a.get("launch") == "D1":
            res = _cmp_track(a, b, ("centre", "width", "inside_rs", "inside_Rm"), ("dust_law", "symbol_ode_R", "symbol_ode_p"))
        elif a.get("launch") == "D2":
            res = _cmp_track(a, b, ("centre", "width", "inside_rs", "inside_Rm"))
            res["changed"].append({"quantity": "p0", "item": "red team 4.4, spec 11.3 (the exact rest condition)", "old": a["p0"],
                                   "new": b["p0"]})
        elif "turning_table" in a:
            res = _cmp_A1(a, b)
        else:
            res = {"error": "unknown record kind"}
        out["pairs"][name] = res
    return out


# ---------------------------------------------------------------- tests of the fixes
def mode_tests(args):
    """each version-5 fix exercised against v4 on small cases (no physics claims)."""
    out = {"label": "unit tests of the flow_chart version-5 fixes (flow_chart_v7.py against flow_chart_v4.py)"}
    # 9.2: one-sided limits at massless nodes
    with np.errstate(all="ignore"):
        v4n = float(V4.group_velocity(0.0, 0.0, -0.5))
        kg = np.linspace(-PI, PI, 4001)
        kg = kg[np.abs(np.sin(kg)) > 1e-9]
        off = max(float(np.max(np.abs(group_velocity(kg, mc, vv) - V4.group_velocity(kg, mc, vv))))
                  for mc in (0.0, 0.1, 0.3) for vv in (-2.0, -1.0, -0.5, 0.0, 0.5))
    t = {"v4_at_node_k0_mc0_v-0.5": v4n if math.isfinite(v4n) else str(v4n),
         "v7_side_plus": float(group_velocity(0.0, 0.0, -0.5, side=1)), "v7_side_minus": float(group_velocity(0.0, 0.0, -0.5, side=-1)),
         "v4_at_k_plus_1e-9": float(V4.group_velocity(1e-9, 0.0, -0.5)), "v4_at_k_minus_1e-9": float(V4.group_velocity(-1e-9, 0.0, -0.5)),
         "v7_negative_band_side_plus": float(group_velocity(0.0, 0.0, -0.5, band=-1, side=1)),
         "max_abs_v7_minus_v4_off_nodes": off}
    t["pass"] = bool(abs(t["v7_side_plus"] - t["v4_at_k_plus_1e-9"]) < 1e-8 and abs(t["v7_side_minus"] - t["v4_at_k_minus_1e-9"]) < 1e-8
                     and off < 1e-14 and not math.isfinite(v4n))
    out["9.2_group_velocity_nodes"] = t
    # 9.2, 9.3: the branch-qualified table against v4's squared law
    ks = PI * np.array([1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8])
    tab, worst = {}, 0.0
    for mc in (0.0, 0.1, 0.3):
        ent = []
        for kk in ks:
            sq = float(V4.turning_ratio(kk, mc))
            for s in (1.0, -1.0):
                en = turning_entry(s * kk, mc)
                en["v4_squared_law"] = sq
                if en["status"] == "zero":
                    worst = max(worst, abs(en["R_over_rs"] - sq) / sq)
                ent.append(en)
        ent += [turning_entry(0.0, mc, side=1.0), turning_entry(0.0, mc, side=-1.0)]
        tab[f"mc={mc}"] = ent
    st = lambda kv: next(e["status"] for e in tab["mc=0.0"] if abs(e["k"] - kv) < 1e-12)  # noqa: E731
    t = {"table": tab, "max_rel_diff_valid_entries_vs_v4": worst,
         "vg_max": {f"mc={mc}": vg_max_branch(mc) for mc in (0.0, 0.1, 0.3)},
         "red_team_crossing_pi3": turning_entry(-PI / 3, 0.0)}
    t["pass"] = bool(worst < 1e-12 and st(PI / 4) == "no zero" and st(5 * PI / 16).startswith("no zero on the inward")
                     and st(-5 * PI / 16) == "zero" and abs(t["red_team_crossing_pi3"]["R_over_rs"] - 1) < 1e-12
                     and abs(t["vg_max"]["mc=0.0"]["interval_0_pi2"]["max"] - 9 / 8) < 1e-12)
    out["9.2_9.3_turning"] = t
    # 9.6: the packet quadrature on uniform and non-uniform grids
    mc, vext, kappa = 0.02, -0.5, 1 / 8
    th = threshold(mc, vext)["omega_th"]
    omega_c = 0.5 * kappa
    sigma = min(omega_c / 4, (omega_c - th) / 4)
    k, om, rho, q = branch_momenta(omega_c, sigma, mc, vext, th)
    spin = lambda kk: V4.positive_spinor(kk, mc)  # noqa: E731
    Np, x0 = 1024, 700
    w7u = build_packet(Np, x0, k, rho, q, spin)
    w4u = V4.build_packet(Np, x0, k, rho / float(np.sum(rho * q)), mc)
    ku = np.linspace(0.0, 1.0, 1601)
    knu = k[0] + (k[-1] - k[0]) * ku ** 2
    vgn = group_velocity(knu, mc, vext)
    omn = V4.bands(knu, mc, vext)[0]
    rhon = np.exp(-(omn - omega_c) ** 2 / (2 * sigma * sigma)) * np.abs(vgn)
    qn = trap_weights(knu)
    rhon = rhon / float(np.sum(rhon * qn))
    w7n = build_packet(Np, x0, knu, rhon, qn, spin)
    A4 = np.sqrt(np.abs(rhon * qn))
    w4n = ((np.exp(1j * np.outer(np.arange(Np) - x0, knu)) * A4) @ np.array([spin(kk) for kk in knu])).reshape(-1)
    w4n = w4n / np.linalg.norm(w4n)
    xu = np.arange(Np, dtype=float) - (Np - 1) / 2
    hu = V4.chain_hamiltonian(np.full(Np, mc), np.zeros(Np), np.full(Np, vext))
    eu, Qu = V4.spectrum(hu)
    mu_t = float(np.sum(rho * q * om))
    t = {"uniform_grid_max_abs_v7_minus_v4": float(np.max(np.abs(w7u - w4u))),
         "nonuniform_overlap_v7_with_uniform": float(abs(np.vdot(w7u, w7n))),
         "nonuniform_overlap_v4_formula_with_uniform": float(abs(np.vdot(w7u, w4n))),
         "killing_content_uniform_exterior_chain": killing_content(Qu, eu, w7u, th),
         "killing_target": {"mean": mu_t, "std": float(np.sqrt(np.sum(rho * q * (om - mu_t) ** 2)))},
         "geometry_uniform_chain": packet_geometry(w7u, xu, args.n_tail, args.tail_tol), "omega_th": th, "omega_c": omega_c,
         "sigma_omega": sigma}
    t["pass"] = bool(t["uniform_grid_max_abs_v7_minus_v4"] < 1e-12 and t["nonuniform_overlap_v7_with_uniform"] > 1 - 1e-6
                     and t["nonuniform_overlap_v4_formula_with_uniform"] < t["nonuniform_overlap_v7_with_uniform"])
    out["9.6_packet_quadrature"] = t
    # 9.7: the threshold by minimization
    rows = []
    for mc_, w_ in ((0.0, 0.5), (0.0078125, 0.5), (0.015625, 0.5), (0.03125, 0.5), (0.0625, 0.5), (0.125, 0.5), (0.1, 0.9), (0.1, 1.0),
                    (0.1, 1.2), (0.1, 1.5)):
        r_ = threshold(mc_, -w_)
        r_["over_mc_sqrtF"] = (r_["omega_th"] / (mc_ * math.sqrt(1 - w_ * w_))) if (mc_ > 0 and w_ < 1) else None
        r_["ingoing_branch"] = threshold(mc_, -w_, branch="in")["omega_th"]
        r_["overtilted_red_team_F2.21"] = bool(w_ > mc_ + math.sqrt(1 + mc_ * mc_))
        rows.append(r_)
    exp_err = [abs(r_["omega_th"] - r_["expansion_red_team_5.2"]) / max(r_["omega_th"], 1e-300) for r_ in rows
               if "expansion_red_team_5.2" in r_ and r_["mc"] > 0 and r_["mc"] <= 0.0625]
    t = {"rows": rows, "max_rel_diff_vs_expansion_mc_le_0.0625": max(exp_err)}
    t["pass"] = bool(t["max_rel_diff_vs_expansion_mc_le_0.0625"] < 1e-4 and rows[0]["omega_th"] < 1e-12
                     and rows[-1]["omega_th"] < 0 and rows[-1]["continuum_v4_clamped"] == 0.0)
    out["9.7_threshold"] = t
    # 9.9: the inversion
    mc, vext, kappa = 0.0, -0.5, 1 / 8
    k, om, rho, q = branch_momenta(0.5 * kappa, 0.5 * kappa / 4, mc, vext, 0.0)
    qt = trap_weights(k)
    a2 = rho / float(np.sum(rho * qt))
    cases = []
    for th_true in (1.0, 0.1, 5.0):
        nw = V4.fermi_average(k, om, a2, th_true, kappa)
        try:
            t4 = V4.invert_theta(nw, k, om, a2, kappa)
        except ValueError as ex:
            t4 = "raised: " + str(ex)
        cases.append({"theta_true": th_true, "n_w": nw, "v4": t4, "v7": invert_theta(nw, k, om, a2, qt, kappa)})
    for nw in (0.6, 0.0, -1e-3):
        try:
            t4 = V4.invert_theta(nw, k, om, a2, kappa)
        except ValueError as ex:
            t4 = "raised: " + str(ex)
        cases.append({"n_w": nw, "v4": t4, "v7": invert_theta(nw, k, om, a2, qt, kappa)})
    t = {"cases": cases}
    t["pass"] = bool(abs(cases[0]["v7"]["theta"] - cases[0]["v4"]) < 1e-9 and abs(cases[1]["v7"]["theta"] - 0.1) < 1e-8
                     and isinstance(cases[1]["v4"], str) and abs(cases[2]["v7"]["theta"] - 5.0) < 1e-7
                     and all(c["v7"]["status"] == "no positive-temperature fit" for c in cases[3:]))
    out["9.9_inversion"] = t
    # 9.11: geometry on D4's packet, an A2 packet, the spec's r = 1/4 convert packet
    R = np.arange(1, 257, dtype=float)
    psiD4 = V4.gaussian_carrier(R, 16.0, 8.0, PI / 2, 0.0, -1.0 / 16.0)
    gD4 = packet_geometry(psiD4, R, args.n_tail, args.tail_tol, surface=32.0, ball=8.0, lost=gaussian_lost(R, 16.0, 8.0))
    R2 = np.arange(1, 1025, dtype=float)
    psiA2 = V4.gaussian_carrier(R2, 149.961328, 16.0, PI / 8, 0.0, -1.0 / 149.961328)
    gA2 = packet_geometry(psiA2, R2, args.n_tail, args.tail_tol, surface=256.0, ball=64.0, lost=gaussian_lost(R2, 149.961328, 16.0))
    Nc, ik, dl = 1024, 8.0, 0.5
    mzc, yyc, xc = V4.planar_fields(Nc, 0.0)
    kap = 1 / ik
    thc = threshold(0.0, -1 + dl)["omega_th"]
    kc, omc, rhoc, qc = branch_momenta(0.25 * kap, 0.25 * kap / 4, 0.0, -1 + dl, thc)
    wc = build_packet(Nc, int(round((Nc - 1) / 2 + 4.0 / kap)), kc, rhoc, qc, lambda kk: V4.positive_spinor(kk, 0.0))
    gC = packet_geometry(wc, xc, args.n_tail, args.tail_tol, surface=0.0, uniform_region=(3 * dl * ik, float(xc[-1])), reject_surface=True)
    t = {"D4_packet": gD4, "A2_packet_rs256_k_pi8_launch_at_law": gA2, "convert_r_quarter_kappa_a_eighth_N1024": gC}
    t["pass"] = bool((not gD4["accepted"]) and gA2["accepted"] and (not gC["accepted"]))
    out["9.11_geometry"] = t
    # 9.12: c_w for the spec's packet measure (sigma_omega = omega_c/4, massless, delta = 0.5)
    rows = []
    for ik_ in (8.0, 32.0):
        kap = 1 / ik_
        for r_ in (0.25, 0.5, 1.0, 2.0):
            k_, om_, rho_, q_ = branch_momenta(r_ * kap, r_ * kap / 4, 0.0, -0.5, 0.0)
            rows.append({"inv_kappa": ik_, "r": r_, "c(omega_c/kappa)": c_mono(r_), "c_w": c_w(k_, om_, rho_, q_, kap),
                         "narrow_window_c_w": c_w(*branch_momenta(r_ * kap, r_ * kap / 400, 0.0, -0.5, 0.0, nk=40001), kap),
                         "c_w_spread_0.04_kappa (indicative: a Gaussian of the package's measured spread, its packets not rebuilt)":
                             c_w(*branch_momenta(r_ * kap, 0.04 * kap, 0.0, -0.5, 0.0, nk=40001), kap)})
    t = {"rows": rows}
    t["pass"] = bool(all(abs(r_["narrow_window_c_w"] - r_["c(omega_c/kappa)"]) / r_["c(omega_c/kappa)"] < 1e-3 for r_ in rows))
    out["9.12_c_w"] = t
    # 9.15: the exact initial velocity against the centred finite difference of the evolution
    Nr, rs_, Rm_ = 256, 32.0, 8.0
    rows = []
    for (mm, kk0, RR0, sg_) in ((0.0, PI / 8, 100.0, 8.0), (0.0, PI / 2, 16.0, 8.0), (0.5, 0.0, 128.0, 8.0), (0.1, -3 * PI / 8, 150.0, 8.0)):
        mz, yy, Rr = V4.radial_fields(Nr, mm, -1.0)
        hr = V4.chain_hamiltonian(mz, yy, V4.pg_profile(Rr, rs_, Rm_))
        er, Qr = V4.spectrum(hr)
        psi = V4.gaussian_carrier(Rr, RR0, sg_, kk0, mm, -1.0 / RR0)
        ve = exact_velocity(hr, psi, Rr)
        dtt = 1e-3
        cp = V4.centre_width(V4.propagate(Qr, er, dtt, psi), Rr)[0]
        cm = V4.centre_width(V4.propagate(Qr, er, -dtt, psi), Rr)[0]
        rows.append({"m": mm, "k0": kk0, "R0": RR0, "sigma_x": sg_, "v_exact": ve, "v_fd_centred_dt_1e-3": (cp - cm) / (2 * dtt),
                     "v_exact_pos_band": exact_velocity_pos_band(hr, psi, Rr, make_bands(mz, yy)),
                     "v_fitted_v4_0_to_4": V4.initial_velocity(Qr, er, psi, Rr)[0],
                     "v_symbol_packet_v4": V4.packet_avg_vg(kk0, sg_, mm, -1.0 / RR0, float(V4.pg_profile(RR0, rs_, Rm_))),
                     "v_symbol_spatial": symbol_vg_spatial(psi, Rr, kk0, sg_, mm, -1.0, lambda x: V4.pg_profile(x, rs_, Rm_))})
    t = {"rows": rows}
    t["pass"] = bool(all(abs(r_["v_exact"] - r_["v_fd_centred_dt_1e-3"]) < 1e-6 for r_ in rows))
    out["9.15_exact_velocity"] = t
    # 9.18: the circular readout (a packet across the zone edge, the band-top packet, a split packet)
    Nn = 512
    n = np.arange(Nn, dtype=float)
    mz, yy = np.full(Nn, 0.3), np.zeros(Nn)
    bnd = make_bands(mz, yy)
    env = lambda c0, kk0: (np.exp(-(n - c0) ** 2 / (4 * 64.0) + 1j * kk0 * n)[:, None] * np.array([1.0, 0.0])[None, :]).reshape(-1)  # noqa: E731
    cases = {}
    for name, psi in (("k0=pi (the zone edge)", env(256, PI)), ("k0=3.05", env(256, 3.05)), ("k0=pi/2", env(256, PI / 2)),
                      ("split k0=0.3 and -2.9", env(200, 0.3) + env(320, -2.9))):
        psi = psi / np.linalg.norm(psi)
        kv, pv = V4.momentum_distribution(psi)
        cases[name] = {"v4_arith": float(kv @ pv), "v7": momentum_readout(psi, bnd)}
    psiK = env(256, PI / 2) / np.linalg.norm(env(256, PI / 2))
    sw4 = V4.kresolved(psiK, n + 1, 300.0)
    sw7 = kresolved_mask(psiK, (n + 1) < 300.0)
    t = {"cases": cases, "kresolved_mask_vs_v4": float(max(abs(sw4[0] - sw7[0]), abs(sw4[1] - sw7[1])))}
    ce, c5 = cases["k0=pi (the zone edge)"], cases["k0=3.05"]
    t["pass"] = bool(abs(abs(ce["v7"]["all"]["circ_mean"]) - PI) < 0.01 and abs(ce["v4_arith"]) < 0.5
                     and abs(c5["v7"]["all"]["circ_mean"] - 3.05) < 0.01 and abs(c5["v4_arith"] - 3.05) > 0.2
                     and abs(cases["k0=pi/2"]["v7"]["all"]["circ_mean"] - PI / 2) < 0.01 and t["kresolved_mask_vs_v4"] < 1e-14)
    out["9.18_momentum_readout"] = t
    # 9.20: serialization
    probe = {"array": np.arange(3.0), "f": np.float64(1.5), "i": np.int64(2), "b": np.bool_(True), "c": np.complex128(1 + 2j),
             "nested": [np.array([[1, 2], [3, 4]])]}
    try:
        json.dumps(probe, default=float)
        v4s = "ok"
    except (TypeError, ValueError) as ex:
        v4s = "v4's default=float fails: " + str(ex)
    back = json.loads(json.dumps(probe, cls=RecordEncoder))
    t = {"v4_serialization": v4s, "v7_round_trip": back}
    t["pass"] = bool(back["array"] == [0.0, 1.0, 2.0] and back["nested"] == [[[1, 2], [3, 4]]] and back["c"] == [1.0, 2.0])
    out["9.20_serialization"] = t
    out["all_pass"] = bool(all(v.get("pass") for v in out.values() if isinstance(v, dict) and "pass" in v))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["tests", "symbol", "radial", "planar_dust", "a2compare", "convert", "compare_v4"])
    p.add_argument("--inv-kappa", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--nsites", type=int, default=1024)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--ramp-time", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--x0-over-kappa", type=float, default=4.0)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--rs", type=float, default=32.0)
    p.add_argument("--Rm", type=float, default=8.0)
    p.add_argument("--kappa-j", type=float, default=-1.0)
    p.add_argument("--launch", default="D1")
    p.add_argument("--sigma-x", type=float, default=16.0, help="planar_dust: the packet's width (spec 11.3: 16)")
    p.add_argument("--x0", type=float, default=None, help="planar_dust: the release point (default 4 r_s)")
    p.add_argument("--half-length", type=int, default=384, help="planar_dust: the chain x in [-L, L]")
    p.add_argument("--tmax", type=float, default=200.0)
    p.add_argument("--tmax-planar", action="store_true", help="planar_dust: run to --tmax instead of the dust law's time to R_m")
    p.add_argument("--tsample", type=float, default=2.0)
    p.add_argument("--n-tail", type=int, default=8, help="9.11: the end buffer (sites) for the tail norms")
    p.add_argument("--tail-tol", type=float, default=1e-4, help="9.11: the rejection tolerance for tail and lost norms")
    p.add_argument("--keep-rejected", action="store_true", help="9.11: evolve rejected packets anyway (labelled), for the v4 comparison")
    p.add_argument("--records", nargs="+", default=[], help="a2compare: the v4 A2 records")
    p.add_argument("--pairs", nargs="+", default=[], help="compare_v4: v4_record:v7_record pairs")
    p.add_argument("--note", default=None, help="a label written into the record (e.g. SMOKE TEST)")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    t0 = time.time()
    utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    V4.GPU["dev"] = a.gpu
    out = {"tests": mode_tests, "symbol": mode_symbol, "radial": mode_radial, "planar_dust": mode_planar_dust, "a2compare": mode_a2compare,
           "convert": mode_convert, "compare_v4": mode_compare_v4}[a.mode](a)
    if a.note:
        out["note"] = a.note
    out["wall_s"] = time.time() - t0
    out["utc_start"] = utc
    out["script"] = SCRIPT
    out["args"] = vars(a)
    text = json.dumps(out, indent=1, cls=RecordEncoder)
    if a.output:
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
        with open(a.output, "w") as f:
            f.write(text + "\n")
        print("%s %s: wrote %s (%d characters, %.1f s)" % (SCRIPT, a.mode, a.output, len(text), out["wall_s"]), flush=True)
    else:
        print(text, flush=True)


if __name__ == "__main__":
    main()
