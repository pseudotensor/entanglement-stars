#!/usr/bin/env python
"""raymap_w9_selftest_v1.py (reviewer A for the busi lane; W9): the self-test of raymap_solved_flow_v3's segment readout, by import of
raymap_solved_flow_v3 (and through it raymap_solved_flow_v2b's packet, propagation, readouts and ray), on manufactured inputs and on
manufactured static flows. Every tolerance is a named argument; PASS/FAIL per criterion, the whole on every criterion; launch W9 on PASS
only.

Tests.
  T0 (manufactured rows, no propagation): an exactly exponential map on a uniform clock grid, k_T0(u) = k_L e^(kappa_H u), U(u) = U_H +
     C e^(-kappa_H u), <h_0> = c_in k with constant c_in, flat (kappa = 0) before a manufactured onset at u_on, with a manufactured sine
     cutoff alpha_d = (2 omega)^(1/3) reached inside the run: the k, E, Ede and U readouts' ratios are 1 to 1e-9 on the exponential
     clocks and 0 on the flat ones; the onset(0.05) and (0.20) fall 2 clocks after u_on (the first clock at which both the 1-step and
     the 2-step difference see the exponential; the 50-percent onset falls at u_on); the dispersive end is the first clock with k_T0 >=
     alpha_d, the segment is truncated there, and its e-fold count equals kappa_H x extent exactly (the telescoped log blueshift equal to
     it). A second manufactured run has the growth saturate below alpha_d (a manufactured end of the map without a sine cutoff): the
     segment ends at the band's exit, the dispersive end reads not reached, the largest k_T0 prints against alpha_d.
  LIN (the exactly exponential static flow, propagated through v2b's code path): N = 1, v = c - 1 with the outgoing speed c(r) =
     kappa_H (r - r_s) up to r_s + 2 r_s (kappa_H = 1/(2 r_s), r_s = 2gE), blended C^1 over +-w into c = 1 beyond (flat): a static flow in
     the Painleve-Gullstrand form (flat slices, N = 1, a flow v(r)) whose ray map is U - r_s proportional to e^(-kappa_H u) exactly and whose
     peeling rate is kappa_H at every clock while the packet is on the linear part. On the lattice the operator (1/2){v, Q} with v linear
     in r is a first-order transport operator in momentum space, so every spectral component is transported exactly along dk/dT =
     -v' sin(2k)/2 and the traced-back wavenumber obeys tan k_T0 = tan k_L e^(kappa_H (u - u_1)) exactly, the k readout's ratio being
     sin(2k)/(2k) = 1 - 2k^2/3 + ... Checks: (S1a) the k readout's ratio is 1 within --ratio-tol on the exact segment (the packet on the
     linear part by 3 sigma_U, k_T0 <= --k-exact-max); (S1b) the onset(0.05) within 3 sigma_u + 3 steps + 2w/r_s of the analytic junction
     clock u_1; (S1c) the e-fold count within 5 percent of kappa_H equals the tan law's count over the same clocks within --efold-tol
     (kappa_H x extent beside); (S1d) the dispersive end (k_T0 reaching alpha_d = (2 a omega)^(1/3)) within --u-tol of the tan law's
     clock; (S1e) the 20-percent segment ends at the dispersive end (the band's own exit, sin(2k)/(2k) = 0.8, lies beyond alpha_d);
     (S1f) the 5-percent segment is longer than 2 sigma_u (not resolution-limited); (S4) the forward re-propagation returns the launch
     and the norms hold.
  PG (the static Painleve-Gullstrand exterior at the same charge, propagated): N = 1, v = -sqrt(2GE/r) with the ball flow inside
     (page_ops_v6.flow_dense), the known law: on any static flow the traced-back point moves backward along the ray flow as u grows, so
     kappa_k(u) = -(dk/dT)(U, k_0)/k_0 and kappa_U(u) = -d ln|dr/dT|/du from the lattice ray equations (raymap_solved_flow_v2b.RayTracer.rhs),
     kappa_H (r_s/U)^(3/2) sin(2k)/(2k) in the centrifugal-free continuum of PG, which reaches the band only asymptotically. Checks: (S2a)
     the k readout follows the exact ray law at the readout's own U and k_0 within --law-tol x kappa_H on admitted clocks (|k_0| <=
     --k-law-max, the packet outside the horizon by 3 sigma_U); (S2b) the band readout applied to the measured ratio and to the law's ratio
     agree in onset and e-fold count for every fraction that has a run in either; (S4) as above. The fixture also shows the review's size
     statement at this r_s/a: whether a 5- or 20-percent band is reached before the dispersive end prints as computed, not as a test.
  COMP: the comparator formula reproduces the review's quoted 0.63 at f_d for omega r_s = 1 and r_s/a 64 and the constants -6.0 and -3.7
     r_s at 80 and 50 percent within --comp-tol; the 95-percent root prints beside the quoted 0.021 with the difference (a note for
     icloud X1, not a failure).
Writes <out>/raymap_w9_selftest_<tag>_<fixture>.json/.npz (v2b's stores), <out>/w9_selftest_<tag>_summary.json and <out>/w9_selftest_<tag>.md."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.optimize as sopt

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import raymap_solved_flow_v3 as W9  # noqa: E402
RM = W9.RM
OPS = W9.OPS
log = RM.log
_g = RM._g


class StaticFields(RM.FieldsBase):
    """time-independent fields on a given chain as a record history (every record the same fields), the charge 2gE = 2 g E."""
    kind = "static"

    def __init__(self, R, T, N, v, E, g, m=0.0, label=""):
        self._chain = {}
        self.m = float(m)
        self.R = np.asarray(R, float)
        self.T = np.asarray(T, float)
        self.N0 = np.asarray(N, float)
        self.v0 = np.asarray(v, float)
        self.E = np.full(len(self.T), float(E))
        self.g = float(g)
        self.tau = float(np.median(np.diff(self.T))) if len(self.T) > 1 else 64.0
        self.rs_nom = 2.0 * self.g * float(E)
        self.label = label
        self.nsites = len(self.R)
        self.notes = []

    def fields(self, k):
        return self.N0.copy(), self.v0.copy()

    def join_check(self, k, lo, hi):
        return (0.0, 0.0)


def linear_flow(R, rs, x1_over_rs=2.0, w=32.0):
    """the exactly exponential static flow: N = 1, c = kappa_H (r - r_s) with kappa_H = 1/(2 r_s) up to x_1 = x1_over_rs r_s where c
    reaches 1, blended C^1 (cubic Hermite over +-w) into the flat c = 1 beyond; v = c - 1 (below -1 inside r_s). (N, v, c, kappa_H, x_1)."""
    R = np.asarray(R, float)
    kH = 0.5 / rs
    x = R - rs
    x1 = x1_over_rs * rs
    line = kH * x
    s = np.clip((x - (x1 - w)) / (2.0 * w), 0.0, 1.0)
    Hs = s * s * (3.0 - 2.0 * s)
    blend = (1.0 - Hs) * line + Hs * 1.0
    c = np.where(x < x1 - w, line, np.where(x > x1 + w, 1.0, blend))
    return np.ones(len(R)), c - 1.0, c, kH, x1


def g_sinc2(k):
    k = np.asarray(k, float)
    return np.where(np.abs(k) > 1e-12, np.sin(2.0 * k) / (2.0 * np.where(np.abs(k) > 1e-12, k, 1.0)), 1.0)


def k_exit(frac):
    """the wavenumber at which sin(2k)/(2k) = 1 - frac: the k readout's band exit on the linear flow."""
    return float(sopt.brentq(lambda k: float(g_sinc2(k)) - (1.0 - frac), 1e-6, math.pi / 2.0 - 1e-6))


def ray_law(F, kappa, U, k0, dt=0.5, dr=1e-2, dk=1e-6):
    """the exact ray law on a static flow at the readout's own (U, k_0): the traced-back point moves backward along the ray flow as the
    launch clock grows (dU/du = -dr/dT, dk_0/du = -dk/dT), so kappa_k = -(dk/dT)/k_0 and kappa_U = -d ln|dr/dT|/du; the derivatives
    of v2b's RayTracer.rhs by central differences; (kappa_k, kappa_U) in per lattice time."""
    tr = RM.RayTracer(F, kappa, dt)
    R = np.asarray(F.R, float)
    N, v = F.fields(0)
    dN = np.gradient(N, R)
    dv = np.gradient(v, R)
    r = np.asarray(U, float)
    k = np.asarray(k0, float)
    f0, gk = tr.rhs(r, k, N, v, dN, dv, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa_k = -gk / k
        frp = tr.rhs(r + dr, k, N, v, dN, dv, R)[0]
        frm = tr.rhs(r - dr, k, N, v, dN, dv, R)[0]
        fkp = tr.rhs(r, k + dk, N, v, dN, dv, R)[0]
        fkm = tr.rhs(r, k - dk, N, v, dN, dv, R)[0]
        d_r = (frp - frm) / (2.0 * dr)
        d_k = (fkp - fkm) / (2.0 * dk)
        dfdu = -d_r * f0 - d_k * gk
        kappa_U = -dfdu / f0
    return kappa_k, kappa_U


def criterion(name, ok, detail):
    log("%s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))
    return {"name": name, "pass": bool(ok), "detail": detail}


def base_cfg(a):
    cfg = RM.cfg_from_args(argparse.Namespace())
    cfg.update({"width_rule": "sigma-u", "sigma_u_rs": a.sigma_u_rs_test, "cheb_tol": a.cheb_tol, "ray_dt": a.ray_dt, "ray_wigner_nodes": a.ray_wigner_nodes,
                "rs_nominal": float(a.rs), "norm_tol": a.norm_tol, "fb_tol": a.fb_tol, "in_vacuum": False, "save_states": True, "force": False, "m": a.m,
                "ray_compare": True, "ball_rs": None, "branch_edge_rule": "refuse", "branch_edge_margin": 0.1, "u_step_rs": a.u_step_rs,
                "band_fracs": [0.05, 0.20, 0.50], "band_err_frac": "same", "band_min_clocks": a.band_min_clocks, "cutoff_inner": 2.0})
    return cfg


# ------------------------------------------------------------------------------------------------------------ T0: synthetic
def synthetic_tests(a, cfg):
    crits = []
    rs = 64.0
    kH = 0.5                                                              # per r_s: rs/(2 x 2gE) with 2gE = rs
    step = 0.5
    u = np.arange(0.0, 20.0 + 1e-9, step)
    n = len(u)
    u_on = 5.0
    j_on = int(np.argmin(np.abs(u - u_on)))
    k_L = 0.02
    C = 10.0 * rs
    U_H = 1.0 * rs
    c_in = 1.2

    def arrays(k):
        U = np.where(u < u_on, U_H + C + C * kH * (u_on - u), U_H + C * np.exp(-kH * (u - u_on)))
        A = {"u": u.copy(), "two_gE_launch": np.full(n, rs), "omega": np.full(n, k_L), "U": U, "sigma_U": np.full(n, 10.0), "k_abs": k, "k_signed": -k, "E0": c_in * k,
             "c_in": np.full(n, c_in), "Uin": np.full(n, np.nan), "U_ray": np.full(n, np.nan), "k_ray": np.full(n, np.nan), "sigma_u_rs": np.full(n, 0.1),
             "frac_k_super": np.zeros(n), "k_launch_abs": np.full(n, k_L)}
        A["kH_rs"] = rs / (2.0 * A["two_gE_launch"])
        return A
    # run 1: the exponential map with the sine cutoff inside the run
    k1 = np.where(u < u_on, k_L, k_L * np.exp(kH * (u - u_on)))
    A1 = arrays(k1)
    res1 = W9.analyse_jet_arrays(A1, cfg, k_L)
    alpha_d = res1["dispersive_end"]["alpha_d"]
    j_d = next(j for j in range(n) if k1[j] >= alpha_d)
    expo = np.arange(j_on + 2, n - 2)
    flat = np.arange(2, j_on - 1)
    dev_e = 0.0
    dev_f = 0.0
    for name in ("k", "E", "Ede", "U"):
        r_ = res1["readouts"][name]["ratio1"]
        dev_e = max(dev_e, float(np.max(np.abs(r_[expo] - 1.0))))
        dev_f = max(dev_f, float(np.max(np.abs(r_[flat]))))
    crits.append(criterion("T0a manufactured exponential: the k, E, Ede and U ratios are 1 on the exponential clocks and 0 on the flat ones", dev_e <= 1e-9 and dev_f <= 1e-9,
                           "%d exponential clocks: max |ratio - 1| %.2e; %d flat clocks: max |ratio| %.2e (tol 1e-9); alpha_d %.4f reached at clock %d (u %.1f)" % (len(expo), dev_e, len(flat), dev_f, alpha_d, j_d, u[j_d])))
    bk = res1["readouts"]["k"]["bands"]
    ok_on = (bk["0.05"]["onset_u"] is not None and abs(bk["0.05"]["onset_u"] - u[j_on + 2]) < 1e-12 and bk["0.20"]["onset_u"] is not None and abs(bk["0.20"]["onset_u"] - u[j_on + 2]) < 1e-12
             and bk["0.50"]["onset_u"] is not None and abs(bk["0.50"]["onset_u"] - u[j_on]) < 1e-12)
    for name in ("E", "U"):
        b_ = res1["readouts"][name]["bands"]["0.05"]
        ok_on = ok_on and (b_["onset_u"] is not None and abs(b_["onset_u"] - u[j_on + 2]) < 1e-12)
    crits.append(criterion("T0b the onsets: 5 and 20 percent 2 clocks after the kink (the 2-step difference sees the kink), 50 percent at the kink", ok_on,
                           "k: onset(0.05) %s, (0.20) %s, (0.50) %s against %.1f, %.1f, %.1f; E and U onset(0.05) %s, %s" % (
                               _g(bk["0.05"]["onset_u"], "%.1f"), _g(bk["0.20"]["onset_u"], "%.1f"), _g(bk["0.50"]["onset_u"], "%.1f"), u[j_on + 2], u[j_on + 2], u[j_on],
                               _g(res1["readouts"]["E"]["bands"]["0.05"]["onset_u"], "%.1f"), _g(res1["readouts"]["U"]["bands"]["0.05"]["onset_u"], "%.1f"))))
    de = res1["dispersive_end"]
    seg = bk["0.05"]["segment"]
    ok_d = bool(de["reached"] and abs(de["u"] - u[j_d]) < 1e-12 and seg is not None and abs(seg["u_end"] - u[j_d - 1]) < 1e-12 and bk["0.05"]["segment_truncated_at_dispersive_end"])
    if seg is not None:
        expected = kH * seg["extent_rs"]
        ok_d = ok_d and abs(seg["efolds"] - expected) <= 1e-9 and abs(seg["efolds_at_kappa_H"] - expected) <= 1e-9 and seg["log_blueshift_k_telescoped"] is not None and abs(seg["log_blueshift_k_telescoped"] - seg["efolds"]) <= 1e-9
    crits.append(criterion("T0c the dispersive end truncates the segment and the e-fold count is kappa_H x extent exactly (the telescoped log blueshift equal to it)", ok_d,
                           "dispersive end at u %s (clock %d, alpha_d %.4f); segment %s to %s, %d clocks, extent %s r_s, e-folds %s against kappa_H x extent %s, kappa_H count %s, telescoped ln k %s, plain ln k %s; truncated %s" % (
                               _g(de["u"], "%.1f"), j_d, alpha_d, _g(seg["u_start"] if seg else None, "%.1f"), _g(seg["u_end"] if seg else None, "%.1f"), seg["n_clocks"] if seg else 0,
                               _g(seg["extent_rs"] if seg else None, "%.2f"), _g(seg["efolds"] if seg else None, "%.6f"), _g(kH * seg["extent_rs"] if seg else None, "%.6f"),
                               _g(seg["efolds_at_kappa_H"] if seg else None, "%.6f"), _g(seg["log_blueshift_k_telescoped"] if seg else None, "%.6f"), _g(seg["log_blueshift_k"] if seg else None, "%.6f"), bk["0.05"]["segment_truncated_at_dispersive_end"])))
    # run 2: the growth saturates below alpha_d (a manufactured end of the map without a sine cutoff)
    u_sat = 10.0
    j_sat = int(np.argmin(np.abs(u - u_sat)))
    k2 = np.where(u < u_on, k_L, np.where(u <= u_sat, k_L * np.exp(kH * (u - u_on)), k_L * math.exp(kH * (u_sat - u_on))))
    A2 = arrays(k2)
    res2 = W9.analyse_jet_arrays(A2, cfg, k_L)
    de2 = res2["dispersive_end"]
    b2 = res2["readouts"]["k"]["bands"]["0.05"]
    s2 = b2["segment"]
    ok_s = bool((not de2["reached"]) and s2 is not None and abs(s2["u_end"] - u[j_sat - 2]) < 1e-12 and abs(s2["u_start"] - u[j_on + 2]) < 1e-12 and (not b2["segment_truncated_at_dispersive_end"])
                and abs(s2["efolds"] - kH * s2["extent_rs"]) <= 1e-9 and de2["k_T0_max_over_alpha_d"] is not None and 0.0 < de2["k_T0_max_over_alpha_d"] < 1.0)
    crits.append(criterion("T0d a manufactured saturation below alpha_d: the segment ends at the band's exit, the dispersive end reads not reached", ok_s,
                           "segment %s to %s (expected %.1f to %.1f), e-folds %s = kappa_H x extent %s; dispersive end reached %s, max k_T0 %s = %s of alpha_d %.4f" % (
                               _g(s2["u_start"] if s2 else None, "%.1f"), _g(s2["u_end"] if s2 else None, "%.1f"), u[j_on + 2], u[j_sat - 2], _g(s2["efolds"] if s2 else None, "%.6f"),
                               _g(kH * s2["extent_rs"] if s2 else None, "%.6f"), de2["reached"], _g(de2["k_T0_max"], "%.4f"), _g(de2["k_T0_max_over_alpha_d"], "%.3f"), de2["alpha_d"])))
    return crits


# ------------------------------------------------------------------------------------------------ the propagated fixtures
def fixture_run(name, F, kappa, q, R_L, u_list, cfg, rs, dev, out, tag):
    n_L = int(np.argmin(np.abs(np.asarray(F.R, float) - R_L)))
    R_L = float(F.R[n_L])
    clocks = RM.clock_records(F, R_L, rs, u_list)
    store = RM.RayStore(os.path.join(out, "raymap_w9_selftest_%s_%s" % (tag, name)), settings={"fixture": name, "q": q, "cfg": {k_: v_ for k_, v_ in cfg.items()}}, label="raymap_w9_selftest_v1 (%s): %s" % (name, F.label))
    specs = [{"k": k_, "u_rs": u_, "q": q, "omega": q / rs} for k_, u_ in clocks]
    t0 = time.time()
    rows = RM.run_channel(F, kappa, specs, cfg, rs, n_L, dev, store)
    good = [r for r in rows if r.get("T0") is not None]
    W9.add_w9_readouts(good, F, n_L)
    store.flush()
    A, good = W9.jet_arrays(good, rs)
    res = W9.analyse_jet_arrays(A, cfg, q / rs)
    log("%s: %d rows (%d analysable) in %.0f s; launch sigma_u %.3f r_s, k_L %.5f, energy spread/omega %.3f, edge distance %.1f sigma_k" % (
        name, len(rows), len(good), time.time() - t0, good[0]["launch"]["sigma_u_rs"], good[0]["launch"]["k_mean_abs"], good[0]["launch"]["energy_spread_over_omega"], good[0]["launch"]["k0_minus_kturn_over_sigma_k"]))
    return A, res, good, n_L, R_L, time.time() - t0


def s4_check(name, A, a):
    fb = A["fb_defect"]
    nrm = A["norm_T0"]
    ok = bool(np.all(np.isfinite(fb)) and np.nanmax(fb) <= a.fb_tol and np.all(np.isfinite(nrm)) and np.max(np.abs(nrm - 1.0)) <= a.norm_tol)
    return criterion("S4 forward-backward returns the launch and the norms hold (%s)" % name, ok,
                     "max ||psi_fwd - psi_L|| %s (tol %g), max |norm - 1| %s (tol %g), %d rows" % (_g(float(np.nanmax(fb)) if np.isfinite(fb).any() else None, "%.2e"), a.fb_tol, _g(float(np.max(np.abs(nrm - 1.0))) if np.isfinite(nrm).any() else None, "%.2e"), a.norm_tol, len(fb)))


def lin_tests(a, cfg, dev, summary):
    crits = []
    rs = float(a.rs)
    g = float(a.g)
    E = rs / (2.0 * g)
    R = np.arange(a.nsites, dtype=float) + 1.0
    N, v, c, kH_lat, x1 = linear_flow(R, rs, a.x1_over_rs, a.blend_w)
    n_L = int(np.argmin(np.abs(R - a.launch_rs * rs)))
    R_L = float(R[n_L])
    omega = a.q_lin / rs
    k_L0 = math.asin(min(omega, 0.999))
    T_1L = (R_L - (rs + x1 + a.blend_w)) / math.cos(k_L0)                   # the flat flight at the lattice group speed to the junction
    u_1 = OPS.chart_retarded_time(T_1L, rs, R_L) / rs                        # the junction clock: the packet's centre at the junction on T = 0
    rstar = T_1L - u_1 * rs
    u_list = np.arange(u_1 - a.u_before, u_1 + a.u_after + 1e-9, a.u_step_rs)
    T_max = (u_1 + a.u_after) * rs + rstar + 3.0 * a.tau
    T = np.arange(0.0, T_max + a.tau, a.tau)
    F = StaticFields(R, T, N, v, E, g, a.m, label="manufactured static flow in PG form with exactly exponential peeling: N = 1, c = 1 + v = kappa_H (r - r_s) to r_s + %g r_s (C^1 blend half-width %g), flat beyond; 2gE %g" % (a.x1_over_rs, a.blend_w, rs))
    log("LIN: 2gE %g, kappa_H %.4e per spacing (%.3f per r_s), junction at %.1f, launch site %d (R %.1f = %g r_s), %d records of tau %g to T %.0f, analytic junction clock u_1 %.4f r_s, clocks u/r_s %.3f to %.3f step %g, q %g (k_L %.5f, alpha_d %.4f)" % (
        rs, kH_lat, kH_lat * rs, rs + x1, n_L, R_L, R_L / rs, len(T), a.tau, T[-1], u_1, u_list[0], u_list[-1], a.u_step_rs, a.q_lin, k_L0, (2.0 * omega) ** (1.0 / 3.0)))
    A, res, rows, n_L, R_L, secs = fixture_run("LIN", F, a.kappa, a.q_lin, R_L, u_list, cfg, rs, dev, a.out, a.tag)
    u = A["u"]
    n = len(u)
    kH_rs = float(np.nanmedian(A["kH_rs"]))
    k_L = float(np.nanmedian(A["k_launch_abs"]))
    step = res["step_rs"]
    sig_u = res["sigma_u_rs_median"]

    def k_law(uu):
        d = np.maximum(np.asarray(uu, float) - u_1, 0.0)
        return np.arctan(math.tan(k_L) * np.exp(kH_rs * d))
    rk = res["readouts"]["k"]["ratio1"]
    rU = res["readouts"]["U"]["ratio1"]
    rE = res["readouts"]["Ede"]["ratio1"]
    mask = np.isfinite(rk) & (A["U"] - 3.0 * A["sigma_U"] > rs) & (A["U"] + 3.0 * A["sigma_U"] < rs + x1 - a.blend_w) & (A["k_abs"] <= a.k_exact_max)
    dev_k = float(np.max(np.abs(rk[mask] - 1.0))) if mask.any() else float("nan")
    dev_U = float(np.nanmax(np.abs(rU[mask] - 1.0))) if mask.any() else float("nan")
    dev_E = float(np.nanmax(np.abs(rE[mask] - 1.0))) if mask.any() else float("nan")
    crits.append(criterion("S1a LIN: the k readout's ratio to kappa_H is 1 on the exact segment", bool(mask.any() and dev_k <= a.ratio_tol),
                           "%d clocks with the packet on the linear part (3 sigma_U) and k_T0 <= %g: max |ratio_k - 1| %.4f (tol %g; the tan law's 1 - sin(2k)/(2k) at most %.4f there); the position map's %.4f and the de-lapsed in-energy's %.4f beside (lattice corrections of order k^2/(kappa_H x))" % (
                               int(mask.sum()), a.k_exact_max, dev_k, a.ratio_tol, 1.0 - float(g_sinc2(a.k_exact_max)), dev_U, dev_E)))
    b05 = res["readouts"]["k"]["bands"]["0.05"]
    tol_on = 3.0 * sig_u + 3.0 * step + 2.0 * a.blend_w / rs
    ok_b = b05["onset_u"] is not None and abs(b05["onset_u"] - u_1) <= tol_on
    crits.append(criterion("S1b LIN: the onset(0.05) at the analytic junction clock", bool(ok_b), "onset %s against u_1 %.4f (tol %.3f = 3 sigma_u %.3f + 3 steps %.3f + 2w/r_s %.3f); the ratio before the junction %s to %s" % (
        _g(b05["onset_u"], "%.4f"), u_1, tol_on, 3.0 * sig_u, 3.0 * step, 2.0 * a.blend_w / rs, _g(float(np.nanmin(rk[u < u_1 - 1.0])) if (u < u_1 - 1.0).any() else None, "%.4f"), _g(float(np.nanmax(rk[u < u_1 - 1.0])) if (u < u_1 - 1.0).any() else None, "%.4f"))))
    seg = b05["segment"]
    if seg is not None and seg["i0"] >= 1 and seg["i1"] + 1 < n:
        lnk = np.log(k_law(u))
        analytic = 0.5 * (lnk[seg["i1"] + 1] + lnk[seg["i1"]] - lnk[seg["i0"]] - lnk[seg["i0"] - 1])
        ok_c = abs(seg["efolds"] - analytic) <= a.efold_tol * max(abs(analytic), 1e-9) + 1e-3
        det_c = "segment u/r_s %.4f to %.4f (%d clocks, %.3f r_s): e-folds %.4f against the tan law's %.4f over the same clocks (tol %g relative), kappa_H x extent %.4f, the telescoped log blueshift of <|k|>_T0 %s, the plain %s; ratio mean %.4f" % (
            seg["u_start"], seg["u_end"], seg["n_clocks"], seg["extent_rs"], seg["efolds"], analytic, a.efold_tol, seg["efolds_at_kappa_H"], _g(seg["log_blueshift_k_telescoped"], "%.4f"), _g(seg["log_blueshift_k"], "%.4f"), seg["ratio_mean"])
    else:
        ok_c = False
        analytic = None
        det_c = "no 5-percent segment with interior ends"
    crits.append(criterion("S1c LIN: the e-fold count within 5 percent of kappa_H equals the analytic (tan law) count", bool(ok_c), det_c))
    de = res["dispersive_end"]
    alpha_d = de["alpha_d"]
    u_d_pred = u_1 + math.log(math.tan(alpha_d) / math.tan(k_L)) / kH_rs
    later = u[u >= u_d_pred - 1e-9]
    u_d_clock = float(later[0]) if len(later) else None
    ok_d = bool(de["reached"] and u_d_clock is not None and abs(de["u"] - u_d_clock) <= a.u_tol)
    crits.append(criterion("S1d LIN: the dispersive end at the tan law's clock", ok_d, "measured %s against the first clock %s at or after the predicted %.4f (tol %g r_s); alpha_d %.4f per spacing = %.1f times k_L %.5f; max <|k|>_T0 %s" % (
        _g(de["u"], "%.4f"), _g(u_d_clock, "%.4f"), u_d_pred, a.u_tol, alpha_d, alpha_d / k_L, k_L, _g(de["k_T0_max"], "%.4f"))))
    kx20 = k_exit(0.20)
    kx05 = k_exit(0.05)
    b20 = res["readouts"]["k"]["bands"]["0.20"]
    s20 = b20["segment"]
    if kx20 > alpha_d:
        ok_e = bool(s20 is not None and b20["segment_truncated_at_dispersive_end"] and de["reached"] and abs(s20["u_end"] - (de["u"] - step)) <= a.u_tol)
        det_e = "the 20-percent band's own exit at k %.4f lies beyond alpha_d %.4f, so the segment must end at the dispersive end: segment end %s against %s, truncated %s; the 5-percent exit at k %.4f (predicted u %.4f) %s alpha_d" % (
            kx20, alpha_d, _g(s20["u_end"] if s20 else None, "%.4f"), _g((de["u"] - step) if de["reached"] else None, "%.4f"), b20["segment_truncated_at_dispersive_end"], kx05, u_1 + math.log(math.tan(kx05) / math.tan(k_L)) / kH_rs, "before" if kx05 < alpha_d else "after")
    else:
        ok_e = bool(s20 is not None and not b20["segment_truncated_at_dispersive_end"])
        det_e = "the 20-percent band's own exit at k %.4f lies before alpha_d %.4f: the segment ends by the band's exit (not truncated %s)" % (kx20, alpha_d, s20 is not None and not b20["segment_truncated_at_dispersive_end"])
    crits.append(criterion("S1e LIN: the 20-percent segment ends where the law says (the dispersive end or the band's exit)", ok_e, det_e))
    crits.append(criterion("S1f LIN: the 5-percent segment is resolved (longer than 2 sigma_u)", bool(seg is not None and seg["resolution_limited"] is False),
                           "extent %s r_s against 2 sigma_u %s r_s" % (_g(seg["extent_rs"] if seg else None, "%.3f"), _g(seg["two_sigma_u_rs"] if seg else None, "%.3f"))))
    crits.append(s4_check("LIN", A, a))
    summary["LIN"] = {"rs": rs, "kappa_H_per_rs": kH_rs, "u_1": u_1, "k_L": k_L, "alpha_d": alpha_d, "u_d_pred": u_d_pred, "k_exit_0.05": kx05, "k_exit_0.20": kx20, "seconds": secs,
                      "u": u.tolist(), "U_over_rs": (A["U"] / rs).tolist(), "sigma_U_over_rs": (A["sigma_U"] / rs).tolist(), "k_T0": A["k_abs"].tolist(), "k_tan_law": k_law(u).tolist(),
                      "ratio_k": rk.tolist(), "ratio_U": rU.tolist(), "ratio_Ede": rE.tolist(), "ratio_E": res["readouts"]["E"]["ratio1"].tolist(), "exact_mask": mask.tolist(),
                      "inband_0.05": b05["inband"].tolist(), "inband_0.20": b20["inband"].tolist(), "analysis": W9.strip_arrays(res)}
    return crits


def pg_tests(a, cfg, dev, summary):
    crits = []
    rs = float(a.rs)
    g = float(a.g)
    E = rs / (2.0 * g)
    R = np.arange(a.nsites, dtype=float) + 1.0
    vfun = OPS.flow_dense(rs, a.Rb_over_rs * rs)
    v = np.asarray(vfun(R), float)
    N = np.ones(len(R))
    n_L = int(np.argmin(np.abs(R - a.launch_rs * rs)))
    R_L = float(R[n_L])
    rstar = 0.0 - OPS.chart_retarded_time(0.0, rs, R_L)
    u_list = np.arange(a.pg_u_from, a.pg_u_to + 1e-9, a.u_step_rs)
    T_max = a.pg_u_to * rs + rstar + 3.0 * a.tau
    T = np.arange(0.0, T_max + a.tau, a.tau)
    F = StaticFields(R, T, N, v, E, g, a.m, label="static PG exterior 2GE %g with the ball flow inside %g 2GE (page_ops_v6.flow_dense)" % (rs, a.Rb_over_rs))
    log("PG: 2gE %g, launch site %d (R %.1f = %g r_s, v_L %.4f), %d records of tau %g, clocks u/r_s %.3f to %.3f step %g, q %g" % (rs, n_L, R_L, R_L / rs, float(v[n_L]), len(T), a.tau, u_list[0], u_list[-1], a.u_step_rs, a.q_pg))
    A, res, rows, n_L, R_L, secs = fixture_run("PG", F, a.kappa, a.q_pg, R_L, u_list, cfg, rs, dev, a.out, a.tag)
    u = A["u"]
    kH_rs = A["kH_rs"]
    kk, kU = ray_law(F, a.kappa, A["U"], A["k_signed"], a.ray_dt)
    law_k = kk * rs / kH_rs
    law_U = kU * rs / kH_rs
    meas_k = res["readouts"]["k"]["ratio1"]
    meas_U = res["readouts"]["U"]["ratio1"]
    adm = np.isfinite(meas_k) & np.isfinite(law_k) & (np.abs(A["k_signed"]) <= a.k_law_max) & (A["U"] - 3.0 * A["sigma_U"] > rs)
    dev_k = float(np.max(np.abs(meas_k[adm] - law_k[adm]))) if adm.any() else float("nan")
    admU = adm & np.isfinite(meas_U) & np.isfinite(law_U)
    dev_U = float(np.max(np.abs(meas_U[admU] - law_U[admU]))) if admU.any() else float("nan")
    crits.append(criterion("S2a PG: the k readout follows the exact ray law at its own (U, k_0)", bool(adm.any() and dev_k <= a.law_tol),
                           "%d admitted clocks (|k_0| <= %g, the packet outside the horizon by 3 sigma_U): max |ratio_k - law| %.4f x kappa_H (tol %g); the law's ratio runs %.3f to %.3f there, the measured %.3f to %.3f; the position map against its law: max %.4f on %d clocks (a lattice-corrected centroid statistic, informational)" % (
                               int(adm.sum()), a.k_law_max, dev_k, a.law_tol, float(np.min(law_k[adm])) if adm.any() else float("nan"), float(np.max(law_k[adm])) if adm.any() else float("nan"),
                               float(np.min(meas_k[adm])) if adm.any() else float("nan"), float(np.max(meas_k[adm])) if adm.any() else float("nan"), dev_U, int(admU.sum()))))
    h = res["arrays"]["h"]
    step = res["step_rs"]
    j_d = res["dispersive_end"]["j"]
    sig_med = res["sigma_u_rs_median"]
    ok_b = True
    dets = []
    for key in sorted(res["readouts"]["k"]["bands"].keys(), key=float):
        frac = float(key)
        bm = res["readouts"]["k"]["bands"][key]
        bl = W9.band_readout(u, h, step, law_k * kH_rs, kH_rs, law_k, np.zeros_like(law_k), A["k_abs"], frac, frac, cfg["band_min_clocks"], j_d, sig_med)
        sm, sl = bm["segment"], bl["segment"]
        if sm is None and sl is None:
            dets.append("%s: no run in either" % key)
            continue
        if (sm is None) != (sl is None):
            s_ = sm or sl
            near = bool(np.nanmin(np.abs(law_k[s_["i0"]: s_["i1"] + 1] - (1.0 - frac))) <= a.law_tol or np.nanmin(np.abs(meas_k[s_["i0"]: s_["i1"] + 1] - (1.0 - frac))) <= a.law_tol)
            ok_b = ok_b and near
            dets.append("%s: a run in the %s only (%.3f to %.3f), the other side within %g of the band edge: %s" % (key, "measured" if sm else "law", s_["u_start"], s_["u_end"], a.law_tol, near))
            continue
        d_on = abs(sm["u_start"] - sl["u_start"])
        d_ef = abs(sm["efolds"] - sl["efolds"])
        okk = d_on <= 3.0 * step and d_ef <= 0.1 + 0.15 * abs(sl["efolds"])
        ok_b = ok_b and okk
        dets.append("%s: onset %.3f against the law's %.3f, e-folds %.3f against %.3f (segment %.3f to %.3f against %.3f to %.3f): %s" % (key, sm["u_start"], sl["u_start"], sm["efolds"], sl["efolds"], sm["u_start"], sm["u_end"], sl["u_start"], sl["u_end"], "ok" if okk else "off"))
    crits.append(criterion("S2b PG: the band readout on the measured ratio agrees with the band readout on the law's ratio", bool(ok_b), "; ".join(dets)))
    crits.append(s4_check("PG", A, a))
    de = res["dispersive_end"]
    log("PG as computed at r_s/a %g, omega r_s %g: the dispersive end %s; the k readout's ratio max %s at u/r_s %s; 5-percent segment %s, 20-percent %s, 50-percent %s" % (
        rs, a.q_pg, ("at u/r_s %.3f" % de["u"]) if de["reached"] else "not reached", _g(res["readouts"]["k"]["ratio_max"], "%.3f"), _g(res["readouts"]["k"]["u_ratio_max"], "%.3f"),
        "found" if res["readouts"]["k"]["bands"]["0.05"]["segment"] else "none", "found" if res["readouts"]["k"]["bands"]["0.20"]["segment"] else "none", "found" if res["readouts"]["k"]["bands"]["0.50"]["segment"] else "none"))
    summary["PG"] = {"rs": rs, "q": a.q_pg, "seconds": secs, "u": u.tolist(), "U_over_rs": (A["U"] / rs).tolist(), "sigma_U_over_rs": (A["sigma_U"] / rs).tolist(), "k_T0": A["k_abs"].tolist(),
                     "ratio_k": meas_k.tolist(), "law_k": law_k.tolist(), "ratio_U": meas_U.tolist(), "law_U": law_U.tolist(), "admitted": adm.tolist(), "analysis": W9.strip_arrays(res)}
    return crits


def comparator_tests(a):
    c = W9.comparator_table(1.0, 64.0, [0.05, 0.20, 0.50], 2.0)
    d = c["formula_minus_quoted"]
    ok = abs(d["ratio_at_f_d_q1_rs64"]) <= a.comp_tol and abs(d["constant_rs_at_0.80"]) <= 10.0 * a.comp_tol and abs(d["constant_rs_at_0.50"]) <= 10.0 * a.comp_tol and abs(d["f_d_q1_rs64"]) <= a.comp_tol
    b = c["bands"]
    det = ("ratio at f_d %.5f for omega r_s = 1 at r_s/a 64: %.4f (quoted 0.63); constants 2 ln f_s: %.3f at 80 percent (quoted -6.0), %.3f at 50 (quoted -3.7), %.3f at 95 (quoted -7.6; f_s %.5f against the quoted 0.021: a difference of %.4f, noted for X1, not a failure); "
           "at omega = T_H and r_s/a 64 the 5-percent segment: %s" % (b["0.05"]["f_d"], b["0.05"]["ratio_at_f_d"], b["0.20"]["constant_rs"], b["0.50"]["constant_rs"], b["0.05"]["constant_rs"], b["0.05"]["f_s"], d["f_s_at_0.95"],
                                                                      "empty by the formula, %.2f r_s with the quoted f_s" % (d["segment_rs_at_TH_0.95_rs64_with_quoted_f_s"] + 0.27)))
    return [criterion("COMP the comparator formula reproduces the review's 0.63, -6.0 and -3.7", bool(ok), det)], c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="w9")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rs", type=float, default=1024.0, help="the fixtures' charge 2GE (lattice units)")
    ap.add_argument("--nsites", type=int, default=8192)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=64.0, help="the record spacing of the static histories (exact at any spacing); the clock step must be a multiple of tau/r_s for a uniform grid")
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--launch-rs", dest="launch_rs", type=float, default=6.0, help="the fixtures' receiver radius (2GE)")
    ap.add_argument("--u-step-rs", dest="u_step_rs", type=float, default=0.125, help="the clock step (r_s); 0.125 = 2 tau at the defaults")
    ap.add_argument("--sigma-u-rs-test", dest="sigma_u_rs_test", type=float, default=0.1, help="the probes' width in u (r_s), the sigma-u rule")
    ap.add_argument("--q-lin", dest="q_lin", type=float, default=30.0, help="LIN: omega r_s (k_L = 0.029, alpha_d = 0.388, reached after 5.3 r_s of exponential peeling)")
    ap.add_argument("--x1-over-rs", dest="x1_over_rs", type=float, default=2.0, help="LIN: the linear part's extent (c reaches 1 there)")
    ap.add_argument("--blend-w", dest="blend_w", type=float, default=32.0, help="LIN: the C^1 junction's half-width (spacings)")
    ap.add_argument("--u-before", dest="u_before", type=float, default=2.5, help="LIN: clocks before the analytic junction clock (r_s)")
    ap.add_argument("--u-after", dest="u_after", type=float, default=6.5, help="LIN: clocks after it (r_s)")
    ap.add_argument("--k-exact-max", dest="k_exact_max", type=float, default=0.05, help="S1a: the k_T0 up to which the ratio counts as exact (1 - 2k^2/3 >= 0.998)")
    ap.add_argument("--ratio-tol", dest="ratio_tol", type=float, default=0.02, help="S1a: |ratio_k - 1| on the exact segment")
    ap.add_argument("--efold-tol", dest="efold_tol", type=float, default=0.05, help="S1c: relative tolerance of the e-fold count against the tan law")
    ap.add_argument("--u-tol", dest="u_tol", type=float, default=0.26, help="S1d, S1e: clock tolerance (r_s; 2 steps)")
    ap.add_argument("--q-pg", dest="q_pg", type=float, default=30.0, help="PG: omega r_s")
    ap.add_argument("--Rb-over-rs", dest="Rb_over_rs", type=float, default=0.5, help="PG: the ball flow inside this radius (2GE)")
    ap.add_argument("--pg-u-from", dest="pg_u_from", type=float, default=-4.0)
    ap.add_argument("--pg-u-to", dest="pg_u_to", type=float, default=4.0)
    ap.add_argument("--k-law-max", dest="k_law_max", type=float, default=0.5, help="S2a: the admitted |k_0|")
    ap.add_argument("--law-tol", dest="law_tol", type=float, default=0.05, help="S2a: |ratio_k - law| x kappa_H")
    ap.add_argument("--band-min-clocks", dest="band_min_clocks", type=int, default=2)
    ap.add_argument("--comp-tol", dest="comp_tol", type=float, default=0.02, help="COMP: the ratio and f_d tolerance (the constants 10x)")
    ap.add_argument("--fb-tol", dest="fb_tol", type=float, default=1e-10)
    ap.add_argument("--norm-tol", dest="norm_tol", type=float, default=1e-10)
    ap.add_argument("--cheb-tol", dest="cheb_tol", type=float, default=1e-14)
    ap.add_argument("--ray-dt", dest="ray_dt", type=float, default=0.5)
    ap.add_argument("--ray-wigner-nodes", dest="ray_wigner_nodes", type=int, default=5)
    ap.add_argument("--skip-pg", dest="skip_pg", action="store_true", help="skip the PG fixture")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cfg = base_cfg(a)
    log("raymap_w9_selftest_v1: v3 %s over v2b %s (%s); device %s" % (W9.CODE, RM.CODE, os.path.abspath(RM.__file__), dev))
    summary = {"settings": vars(a), "device": str(dev), "code": W9.CODE, "v2b": RM.CODE}
    crits = []
    crits += synthetic_tests(a, cfg)
    crits += lin_tests(a, cfg, dev, summary)
    if not a.skip_pg:
        crits += pg_tests(a, cfg, dev, summary)
    cc, comp = comparator_tests(a)
    crits += cc
    summary["comparator"] = comp
    allpass = all(c["pass"] for c in crits)
    summary["criteria"] = crits
    summary["pass"] = allpass
    summary["seconds"] = time.time() - t0
    pj = os.path.join(a.out, "w9_selftest_%s_summary.json" % a.tag)
    with open(pj, "w") as f:
        json.dump(summary, f, default=RM._jdefault, indent=1)
    md = ["# raymap_w9_selftest_v1: %s (%d of %d criteria; %.0f s)" % ("PASS" if allpass else "FAIL", sum(c["pass"] for c in crits), len(crits), summary["seconds"]), "",
          "| criterion | result | detail |", "|---|---|---|"]
    md += ["| %s | %s | %s |" % (c["name"], "PASS" if c["pass"] else "FAIL", c["detail"].replace("|", "/")) for c in crits]
    if "LIN" in summary:
        S = summary["LIN"]
        md += ["", "LIN (2gE %g, kappa_H %.3f per r_s, u_1 %.4f, k_L %.5f, alpha_d %.4f, the dispersive end predicted at u %.4f): u/r_s, U/r_s, sigma_U/r_s, k_T0, the tan law's k, ratio_k, ratio_E, ratio_Ede, ratio_U, exact-segment mask, in band 5, 20 percent" % (
            S["rs"], S["kappa_H_per_rs"], S["u_1"], S["k_L"], S["alpha_d"], S["u_d_pred"]), "",
            "| u/r_s | U/r_s | sigma_U/r_s | k_T0 | k law | r_k | r_E | r_Ede | r_U | exact | band 5 | band 20 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in range(len(S["u"])):
            md.append("| %.4f | %.4f | %.4f | %s | %.5f | %s | %s | %s | %s | %d | %d | %d |" % (S["u"][j], S["U_over_rs"][j], S["sigma_U_over_rs"][j], _g(S["k_T0"][j], "%.5f"), S["k_tan_law"][j],
                                                                                           _g(S["ratio_k"][j], "%.4f"), _g(S["ratio_E"][j], "%.4f"), _g(S["ratio_Ede"][j], "%.4f"), _g(S["ratio_U"][j], "%.4f"), int(S["exact_mask"][j]), int(S["inband_0.05"][j]), int(S["inband_0.20"][j])))
    if "PG" in summary:
        S = summary["PG"]
        md += ["", "PG (2gE %g, omega r_s %g): u/r_s, U/r_s, sigma_U/r_s, k_T0, ratio_k measured and the law's, ratio_U measured and the law's, admitted" % (S["rs"], S["q"]), "",
               "| u/r_s | U/r_s | sigma_U/r_s | k_T0 | r_k | law_k | r_U | law_U | adm |", "|---|---|---|---|---|---|---|---|---|"]
        for j in range(len(S["u"])):
            md.append("| %.4f | %.4f | %.4f | %s | %s | %s | %s | %s | %d |" % (S["u"][j], S["U_over_rs"][j], S["sigma_U_over_rs"][j], _g(S["k_T0"][j], "%.5f"), _g(S["ratio_k"][j], "%.4f"), _g(S["law_k"][j], "%.4f"),
                                                                       _g(S["ratio_U"][j], "%.4f"), _g(S["law_U"][j], "%.4f"), int(S["admitted"][j])))
    md += ["", "The comparator: " + comp["note"], ""]
    with open(os.path.join(a.out, "w9_selftest_%s.md" % a.tag), "w") as f:
        f.write("\n".join(md) + "\n")
    log("raymap_w9_selftest_v1: %s (%d of %d criteria) in %.0f s; %s" % ("PASS" if allpass else "FAIL", sum(c["pass"] for c in crits), len(crits), summary["seconds"], pj))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
