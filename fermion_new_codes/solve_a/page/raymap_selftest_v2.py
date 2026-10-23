#!/usr/bin/env python
"""raymap_selftest_v2.py (reviewer A for the solve builder): the self-test of raymap_solved_flow_v2 on manufactured exteriors, by import of its
routines (the packet, the two propagations, the readouts, the difference rule, the plateau finder, the ray tracer, the branch-edge rule;
page_d13_readout_v1 and page_ops_v6 beneath them). Changes against raymap_selftest_v1.py, from the builder's run on the box CPU (S1 FAIL
with the packet's position within 0.093 spacings of the ray's and its kappa off by 0.107 x 1/(4GE) against 0.05; S2 PASS; a ValueError in
S3 at the branch edge):
(S1) the test's construction. v1 set the kappa tolerance 0.05 x 1/(4GE) beside a position tolerance of 2 spacings without propagating the
     one through the estimator to the other: kappa is the log-difference of first differences at the clock step h = 0.125 r_s, so a
     position difference eps moves it by up to 2 eps (1/|D_j| + 1/|D_{j-1}|)/h^2 (D the first difference, 170 to 250 spacings per r_s on
     these clocks): 0.28 x 1/(4GE) for the achieved 0.093 spacings (the observed 0.107 is inside it) and 4 x 1/(4GE) for the stated 2
     spacings. The two tolerances were inconsistent and the kappa criterion unreachable at that step for any realistic packet-ray
     agreement; the script's packet reproduced the Wigner-averaged ray to a tenth of a spacing over 59 clocks. v2: S1a is the position
     criterion with --U-tol-spacings 0.5 (half a spacing; v1's 2); S1b compares the two kappas at the production's clock step
     (--kappa-compare-step 4 clocks = 0.5 r_s, the step-n estimator log_diff_kappa) with the per-clock tolerance --kappa-tol-rel x 1/(4GE)
     + the amplification of --U-tol-spacings at that step (stated and printed per clock; the floor is the irreducible part); the fine-step
     difference and its amplification bound are printed beside as v1's criterion would have read.
(S3) the test's construction and the script's handling. v1's continuum frequency q_cont 0.02 lies below the channel's branch minimum at the
     launch site: the centrifugal term kappa/R is a mass in the flow, so the outgoing branch starts at omega_turn = N (kappa/R_L)
     sqrt(1 - (v_L/N)^2) (2.28e-4 here, q 0.117 for kappa 1 at 8 r_s; the crash's 2.29e-4 was the 3.9e-5 grid's value). On this lattice
     (2GE = 512) the kappa = 1 channel has no window where the lattice correction, the centrifugal correction and 1 - the reach are all
     below 2 percent, so the continuum check runs the ray on the centrifugal-free dispersion (--kappa-cont 0: the same chain without the
     kappa/R term, a manufactured dispersion for this check, not a channel of the model), admits the clocks with eps <= --eps-cont-max
     (0.003, so that the first-order lattice correction 3 eps/(1 + eps) stays below half of --cont-tol; v1 admitted eps up to 0.05, a
     correction of 0.14 against a tolerance of 0.02) and runs the ray clocks to --u-ray-to 8 r_s. The script's exception is replaced by
     the branch-edge rule (raymap_solved_flow_v2.outgoing_branch), tested in S5.
(S5) new: the branch-edge rule. At --q-edge-frac of the channel's gap at the launch site: outgoing_branch refuses without an exception,
     its omega_turn agrees with the small-k gap to --gap-tol-rel (S5a); the production path writes a refusal row (run_channel, no
     propagation; S5b); the nearest rule returns an admitted carrier at omega_turn (1 + --branch-edge-margin) with v_+ > 0 and
     build_packet returns a unit packet (S5c; its energy matching and cut fraction are printed: near the gap the packet's spectral width
     exceeds its distance to the edge, the reason the default rule refuses). The gaps of kappa 1 to 4 at the test's and the production's
     launch radii are printed.
The width rule's sigma is the density's standard deviation in v2 (the self-test's sigma_u 0.1 r_s now measures as 0.1; v1 measured 0.07).

Tests (every tolerance a named argument; PASS/FAIL per criterion, the whole on every criterion):
  STATIC: the static Schwarzschild-like exterior N = 1, v = -sqrt(2GE/r) in the Painleve-Gullstrand form at fixed 2GE = --rs (the lane's
    exterior-plus-ball flow page_ops_v6.flow_dense with the ball at --Rb-over-rs 2GE, so the horizon at 2GE is on the chain), the clock
    T - r_*(R) of page_ops_v6.chart_retarded_time. Probes at --q-geo (the geometric-optics frequency: wavelength well below 2GE, width
    --sigma-u-rs-test in u by the sigma-u rule) launched at --launch-rs 2GE at clocks --u-geo-from to --u-geo-to (r_s) in steps of
    --u-step-rs, back-propagated to T = 0 and read out exactly as the production code does.
    (S1a) the position map reproduces the Wigner-averaged lattice ray map: |U_packet - U_ray| <= --U-tol-spacings on every clock whose ray
          reaches T = 0 with |k| <= --k-ray-max;
    (S1b) the kappas agree at the production's clock step: |kappa_packet - kappa_ray| <= --kappa-tol-rel x 1/(4GE) + 2 U_tol (1/|D_j| +
          1/|D_{j-n}|)/(n h)^2 at n = --kappa-compare-step on the same clocks;
    (S2) the ray map reproduces U - 2GE proportional to e^{-u/(4GE)} to the stated order: with x_0 = r_0 - 2GE and eps = 2GE k_0^2/x_0
         (the lattice dispersion's correction, from the near-horizon ray equations dx/dT = kappa x + (3/2) k^2, dk/dT = -kappa k),
         kappa_ray(u) = (1/(4GE)) [(2GE/r_0)^(3/2) - 3 eps/(1 + eps)] within --order-tol x 1/(4GE) at the clocks with eps <= --eps-max and
         x_0 <= --x-max-over-rs 2GE, at the frequency --q-mid on the channel --kappa;
    (S3) the continuum limit on the dispersion --kappa-cont: at --q-cont the ray map's kappa(u) equals the continuum (1/(4GE))(2GE/r_0(u))^
         (3/2) within --cont-tol x 1/(4GE) wherever eps <= --eps-cont-max, and reaches at least --kappa-reach-min x 1/(4GE); r_0(u) from
         r_*(r_0) = -u exactly beside it;
    (S4) the forward re-propagation returns the launch to --fb-tol in ||psi_fwd - psi_L|| and the norms stay within --norm-tol of 1;
    (S5) the branch-edge rule, above.
  FLAT: N = 1, v = 0 on the same chain (the clock u = T - R): kappa(u) = 0 within --flat-kappa-tol x 1/(4GE) of the static test's charge
    at every clock, and the forward check as in (S4).
The static tests' record spacing --tau (exact at any spacing for a time-independent h) mirrors the production path's per-record Chebyshev
step. Writes <out>/raymap_selftest_<tag>.json/.npz/.md through the production store and <out>/selftest_<tag>_summary.json."""
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
import raymap_solved_flow_v2 as RM  # noqa: E402
import page_ops_v6 as OPS  # noqa: E402

log = RM.log
_g = RM._g


class ArrayFields(RM.FieldsBase):
    """time-independent fields on a given chain as a record history (every record the same fields), the charge 2gE = E 2g."""
    kind = "array"

    def __init__(self, R, T, N, v, E, g, m=0.0, label=""):
        self._chain = {}
        self.m = float(m); self.R = np.asarray(R, float); self.T = np.asarray(T, float)
        self.N0 = np.asarray(N, float); self.v0 = np.asarray(v, float)
        self.E = np.full(len(self.T), float(E)); self.g = float(g)
        self.tau = float(np.median(np.diff(self.T))) if len(self.T) > 1 else 8.0
        self.rs_nom = 2.0 * self.g * float(E); self.label = label; self.nsites = len(self.R); self.notes = []

    def fields(self, k):
        return self.N0.copy(), self.v0.copy()


def continuum_r0(u_lat, rs, R_L):
    """r_0 with r_*(r_0) = -u in the PG chart (chart_retarded_time(0, rs, r_0) = u), by brentq on (rs(1 + 1e-13), R_L]."""
    f = lambda r: OPS.chart_retarded_time(0.0, rs, r) - u_lat   # noqa: E731
    a, b = rs * (1.0 + 1e-13), R_L
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    return float(sopt.brentq(f, a, b, xtol=1e-10, maxiter=500))


def criterion(name, ok, detail):
    log("%s: %s (%s)" % (name, "PASS" if ok else "FAIL", detail))
    return {"name": name, "pass": bool(ok), "detail": detail}


def amplification(D, n, h):
    """the step-n estimator's amplification of a position error, per spacing: 2 (1/|D_j| + 1/|D_{j-n}|)/(n h)^2 in 1/r_s (D the forward
    step-n differences in spacings per r_s, h the clock step in r_s); nan where undefined."""
    N = len(D); amp = np.full(N, np.nan)
    for j in range(n, N - n):
        if np.isfinite(D[j]) and np.isfinite(D[j - n]) and D[j] != 0 and D[j - n] != 0:
            amp[j] = 2.0 * (1.0 / abs(D[j]) + 1.0 / abs(D[j - n])) / (n * h) ** 2
    return amp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="selftest")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rs", type=float, default=512.0, help="the static exterior's charge 2GE (lattice units)")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--Rb-over-rs", dest="Rb_over_rs", type=float, default=0.5, help="the ball flow inside this radius (page_ops_v6.flow_dense)")
    ap.add_argument("--launch-rs", dest="launch_rs", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=64.0, help="the record spacing of the static history (exact at any spacing)")
    ap.add_argument("--kappa", type=int, default=1); ap.add_argument("--m", type=float, default=0.0)
    ap.add_argument("--q-geo", dest="q_geo", type=float, default=32.0, help="omega r_s of the geometric-optics probes")
    ap.add_argument("--q-mid", dest="q_mid", type=float, default=1.0, help="omega r_s of the ray map's first-order check (S2, on the channel --kappa)")
    ap.add_argument("--q-cont", dest="q_cont", type=float, default=0.02, help="omega r_s of the ray map's continuum check (S3, on --kappa-cont)")
    ap.add_argument("--kappa-cont", dest="kappa_cont", type=int, default=0, help="S3's dispersion: 0 the centrifugal-free chain (not a channel of the model), else that channel")
    ap.add_argument("--u-geo-from", dest="u_geo_from", type=float, default=-7.5); ap.add_argument("--u-geo-to", dest="u_geo_to", type=float, default=0.5)
    ap.add_argument("--u-ray-to", dest="u_ray_to", type=float, default=8.0, help="the ray checks' last clock (r_s)")
    ap.add_argument("--u-step-rs", dest="u_step_rs", type=float, default=0.125)
    ap.add_argument("--sigma-u-rs-test", dest="sigma_u_rs_test", type=float, default=0.1, help="the geometric probes' width in u (r_s; the density's standard deviation)")
    ap.add_argument("--flat-T-from-rs", dest="flat_T_from_rs", type=float, default=0.5); ap.add_argument("--flat-T-to-rs", dest="flat_T_to_rs", type=float, default=3.0)
    ap.add_argument("--fb-tol", dest="fb_tol", type=float, default=1e-10); ap.add_argument("--norm-tol", dest="norm_tol", type=float, default=1e-10)
    ap.add_argument("--flat-kappa-tol", dest="flat_kappa_tol", type=float, default=0.02, help="|kappa| <= tol x 1/(4GE) in flat space")
    ap.add_argument("--U-tol-spacings", dest="U_tol_spacings", type=float, default=0.5, help="S1a: |U_packet - U_ray| in spacings (v1: 2)")
    ap.add_argument("--kappa-compare-step", dest="kappa_compare_step", type=int, default=4, help="S1b: the kappa comparison's step in clocks (4 x 0.125 r_s = the production's 0.5 r_s)")
    ap.add_argument("--kappa-tol-rel", dest="kappa_tol_rel", type=float, default=0.05, help="S1b: the tolerance's floor (x 1/(4GE)); the amplification of --U-tol-spacings at the comparison step is added per clock")
    ap.add_argument("--k-ray-max", dest="k_ray_max", type=float, default=0.5, help="clocks whose ray reaches T = 0 above this |k| are excluded from S1")
    ap.add_argument("--eps-max", dest="eps_max", type=float, default=0.05, help="S2: the admitted lattice parameter 2GE k_0^2/x_0")
    ap.add_argument("--eps-cont-max", dest="eps_cont_max", type=float, default=0.003, help="S3: the admitted eps (3 eps/(1 + eps) below half of --cont-tol)")
    ap.add_argument("--x-max-over-rs", dest="x_max_over_rs", type=float, default=0.15)
    ap.add_argument("--order-tol", dest="order_tol", type=float, default=0.05); ap.add_argument("--cont-tol", dest="cont_tol", type=float, default=0.02)
    ap.add_argument("--kappa-reach-min", dest="kappa_reach_min", type=float, default=0.9)
    ap.add_argument("--q-edge-frac", dest="q_edge_frac", type=float, default=0.5, help="S5: the frequency below the gap, as a fraction of the gap")
    ap.add_argument("--gap-tol-rel", dest="gap_tol_rel", type=float, default=1e-4, help="S5a: |omega_turn - the small-k gap| relative to the gap")
    ap.add_argument("--branch-edge-margin", dest="branch_edge_margin", type=float, default=0.1, help="S5c: the nearest rule's margin")
    ap.add_argument("--cheb-tol", dest="cheb_tol", type=float, default=1e-14); ap.add_argument("--ray-dt", dest="ray_dt", type=float, default=0.5)
    ap.add_argument("--ray-wigner-nodes", dest="ray_wigner_nodes", type=int, default=7)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    os.makedirs(a.out, exist_ok=True)
    rs = float(a.rs); g = float(a.g); E = rs / (2.0 * g)
    R = np.arange(a.nsites, dtype=float) + 1.0
    kH_rs = 0.5                                                             # 1/(4GE) in units of 1/r_s (r_s = 2GE here)
    cfg = RM.cfg_from_args(argparse.Namespace())
    cfg.update({"launch_rs": a.launch_rs, "width_rule": "sigma-u", "sigma_u_rs": a.sigma_u_rs_test, "cheb_tol": a.cheb_tol, "ray_dt": a.ray_dt, "ray_wigner_nodes": a.ray_wigner_nodes,
                "rs_nominal": rs, "norm_tol": a.norm_tol, "fb_tol": a.fb_tol, "in_vacuum": False, "save_states": True, "force": False, "m": a.m, "ray_compare": True, "ball_rs": None,
                "branch_edge_rule": "refuse", "branch_edge_margin": a.branch_edge_margin})
    n_L = int(np.argmin(np.abs(R - a.launch_rs * rs))); R_L = float(R[n_L])
    crits = []
    summary = {"settings": vars(a), "device": str(dev), "static": {}, "flat": {}, "edge": {}}
    # ------------------------------------------------------------------------------------------------------- STATIC
    vfun = OPS.flow_dense(rs, a.Rb_over_rs * rs)
    v_pg = vfun(R); N_pg = np.ones(len(R))
    u_lat_max = a.u_ray_to * rs
    T_max = u_lat_max + (0.0 - OPS.chart_retarded_time(0.0, rs, R_L)) + 2.0 * a.tau
    T = np.arange(0.0, T_max + a.tau, a.tau)
    Fs = ArrayFields(R, T, N_pg, v_pg, E, g, a.m, label="static PG exterior 2GE %g with the ball flow inside %g 2GE (page_ops_v6.flow_dense)" % (rs, a.Rb_over_rs))
    store = RM.RayStore(os.path.join(a.out, "raymap_selftest_%s" % a.tag), settings=dict(vars(a), cfg=cfg), label="raymap_selftest_v2: " + Fs.label)
    u_geo = np.arange(a.u_geo_from, a.u_geo_to + 1e-9, a.u_step_rs)
    clocks = RM.clock_records(Fs, R_L, rs, u_geo)
    v_L = float(v_pg[n_L])
    log("static: 2GE %g, %d records of tau %g to T/r_s %.2f, launch site %d (R %.1f = %g r_s, v_L %.4f), %d geometric clocks u/r_s %.3f to %.3f (records %d to %d), q_geo %g" % (
        rs, len(T), a.tau, T[-1] / rs, n_L, R_L, R_L / rs, v_L, len(clocks), clocks[0][1], clocks[-1][1], clocks[0][0], clocks[-1][0], a.q_geo))
    # the gaps at the launch sites (N = 1, v = -sqrt(2GE/R)): omega_turn r_s = kappa (2GE/R_L) sqrt(1 - 2GE/R_L)
    gaps = []
    for kap_ in (1, 2, 3, 4):
        for RL_rs in sorted(set([a.launch_rs, 16.0])):
            gaps.append({"kappa": kap_, "launch_rs": RL_rs, "q_gap": kap_ * (1.0 / RL_rs) * math.sqrt(max(1.0 - 1.0 / RL_rs, 0.0))})
    log("the centrifugal gap at the launch site (omega_turn r_s, small-k form kappa (2GE/R_L) sqrt(1 - 2GE/R_L)): " + "; ".join(
        "kappa %d at %g r_s: %.4f" % (d["kappa"], d["launch_rs"], d["q_gap"]) for d in gaps) + "; the production's lowest omega r_s is 0.25")
    summary["edge"]["gaps"] = gaps
    specs = [{"k": k_, "u_rs": u_, "q": a.q_geo, "omega": a.q_geo / rs} for (k_, u_) in clocks]
    t0 = time.time()
    rows = RM.run_channel(Fs, a.kappa, specs, cfg, rs, n_L, dev, store)
    rows = [r for r in rows if r.get("T0") is not None]
    m = RM.analyse_map(rows, rs, cfg)
    A = m["arrays"]
    log("static geometric probes: %d rows in %.0f s; packet minus ray max %.3f spacings; launch sigma_u %.3f r_s (rule %g), energy spread/omega %.3f, edge distance %.1f sigma_k" % (
        len(rows), time.time() - t0, m.get("packet_minus_ray_max", float("nan")), rows[0]["launch"]["sigma_u_rs"], a.sigma_u_rs_test, rows[0]["launch"]["energy_spread_over_omega"],
        rows[0]["launch"]["k0_minus_kturn_over_sigma_k"]))
    # S4 on the static rows
    fb_def = np.array([r["fb"]["defect_norm"] if r.get("fb") else np.nan for r in rows]); nrm = np.array([r["T0"]["norm_T0"] for r in rows])
    nmin = np.array([r["T0"]["norm_min_along"] for r in rows]); nmax = np.array([r["T0"]["norm_max_along"] for r in rows])
    s4 = bool(np.all(np.isfinite(fb_def)) and np.nanmax(fb_def) <= a.fb_tol and np.max(np.abs(nrm - 1.0)) <= a.norm_tol and np.max(np.abs(nmin - 1.0)) <= a.norm_tol and np.max(np.abs(nmax - 1.0)) <= a.norm_tol)
    crits.append(criterion("S4 forward-backward returns the launch (static)", s4, "max ||psi_fwd - psi_L|| %.2e (tol %g), max |norm - 1| %.2e (tol %g), %d rows" % (
        float(np.nanmax(fb_def)), a.fb_tol, float(max(np.max(np.abs(nrm - 1.0)), np.max(np.abs(nmin - 1.0)), np.max(np.abs(nmax - 1.0)))), a.norm_tol, len(rows))))
    # S1a: the position against the Wigner-averaged lattice ray; S1b: kappa at the production's clock step
    u = A["u_over_rs"]; U = A["U"]; kH = A["kappa_H_per_rs"]
    kray = A.get("k_ray")
    n_cmp = max(1, int(a.kappa_compare_step)); h_rs = float(np.median(np.diff(u))) if len(u) > 1 else a.u_step_rs
    s1_extra = {}
    if "U_ray" in A and kray is not None:
        Ur = A["U_ray"]
        selU = np.isfinite(Ur) & (np.abs(kray) <= a.k_ray_max)
        dU = np.abs(U - Ur)
        s1a = bool(selU.any() and np.max(dU[selU]) <= a.U_tol_spacings)
        det_a = "%d clocks with |k_ray| <= %g: max |U_packet - U_ray| %.3f spacings (tol %g; v1's 2); the Wigner average over %d^2 nodes" % (
            int(selU.sum()), a.k_ray_max, float(np.max(dU[selU])) if selU.any() else float("nan"), a.U_tol_spacings, a.ray_wigner_nodes)
        _, kp_n, _, _ = RM.log_diff_kappa(u, U, n_cmp); _, kr_n, _, Dr_n = RM.log_diff_kappa(u, Ur, n_cmp)
        amp_n = amplification(Dr_n, n_cmp, h_rs)
        tol_n = a.kappa_tol_rel * kH + a.U_tol_spacings * amp_n
        dk_n = np.abs(kp_n - kr_n)
        sel = np.isfinite(kp_n) & np.isfinite(kr_n) & np.isfinite(amp_n) & (np.abs(kray) <= a.k_ray_max)
        # the fine step (v1's criterion) as a diagnostic
        _, kp_1, _, _ = RM.log_diff_kappa(u, U, 1); _, kr_1, _, Dr_1 = RM.log_diff_kappa(u, Ur, 1)
        amp_1 = amplification(Dr_1, 1, h_rs); dk_1 = np.abs(kp_1 - kr_1)
        sel1 = np.isfinite(kp_1) & np.isfinite(kr_1) & np.isfinite(amp_1) & (np.abs(kray) <= a.k_ray_max)
        eps_ach = float(np.max(dU[selU])) if selU.any() else float("nan")
        if sel.any():
            ratio = dk_n[sel] / tol_n[sel]
            s1b = bool(np.max(ratio) <= 1.0)
            det_b = ("%d clocks at the step %d x %g r_s: max |kappa_packet - kappa_ray| %.4f x 1/(4GE); the tolerance per clock = floor %g + the amplification of %g spacings "
                     "(%.4f to %.4f x 1/(4GE)); the largest ratio to the tolerance %.3f; kappa x 4GE from %.3f to %.3f; at the fine step (v1's criterion): max |d kappa| %.4f "
                     "x 1/(4GE) against the amplification bound of the achieved %.3f spacings %.4f (v1's fixed tolerance %g)") % (
                int(sel.sum()), n_cmp, h_rs, float(np.max(dk_n[sel]) / kH_rs), a.kappa_tol_rel, a.U_tol_spacings, float(np.min(tol_n[sel]) / kH_rs), float(np.max(tol_n[sel]) / kH_rs),
                float(np.max(ratio)), float(np.min(kp_n[sel] / kH_rs)), float(np.max(kp_n[sel] / kH_rs)), float(np.max(dk_1[sel1]) / kH_rs) if sel1.any() else float("nan"),
                eps_ach, float(np.max(eps_ach * amp_1[sel1]) / kH_rs) if sel1.any() else float("nan"), 0.05)
        else:
            s1b = False; det_b = "no clock admitted at the step %d (the run is too short or every ray beyond k_ray_max)" % n_cmp
        s1_extra = {"kappa_n_packet_x4GE": (kp_n / kH_rs).tolist(), "kappa_n_ray_x4GE": (kr_n / kH_rs).tolist(), "amplification_n_per_spacing_x4GE": (amp_n / kH_rs).tolist(),
                    "tolerance_n_x4GE": (tol_n / kH_rs).tolist(), "n_compare": n_cmp, "dU_spacings": dU.tolist(),
                    "amplification_1_per_spacing_x4GE": (amp_1 / kH_rs).tolist(), "dkappa_1_x4GE": (dk_1 / kH_rs).tolist()}
    else:
        s1a = s1b = False; det_a = det_b = "no ray data"
    crits.append(criterion("S1a packet centroid against the Wigner-averaged lattice ray: position (static, q_geo)", s1a, det_a))
    crits.append(criterion("S1b packet centroid against the Wigner-averaged lattice ray: kappa at the production's clock step (static, q_geo)", s1b, det_b))
    summary["static"]["geometric"] = dict({"n_rows": len(rows), "map": {k_: v_ for k_, v_ in m.items() if k_ != "arrays"}, "u": u.tolist(), "U": U.tolist(),
                                           "U_ray": A["U_ray"].tolist() if "U_ray" in A else None, "kappa1_x4GE": (A["kappa1"] / kH).tolist(),
                                           "kappa_ray1_x4GE": (A["kappa_ray1"] / kH).tolist() if "kappa_ray1" in A else None, "k_ray": kray.tolist() if kray is not None else None,
                                           "k_mean_abs_T0": A["k_mean_abs_T0"].tolist(), "fb_defect": fb_def.tolist()}, **s1_extra)
    # S2 and S3: the ray map alone on a dense clock set
    u_ray = np.arange(a.u_geo_from, a.u_ray_to + 1e-9, a.u_step_rs)
    rclocks = RM.clock_records(Fs, R_L, rs, u_ray)
    for name, q, tol, mode, kap_, eps_cap in (("S2", a.q_mid, a.order_tol, "order", a.kappa, a.eps_max), ("S3", a.q_cont, a.cont_tol, "cont", a.kappa_cont, a.eps_cont_max)):
        omega = q / rs
        disp = ("the channel kappa %d" % kap_) if kap_ != 0 else "the centrifugal-free dispersion (kappa 0: the chain without the kappa/R term; not a channel of the model)"
        lab = "first-order lattice formula (2GE/r_0)^(3/2) - 3 eps/(1 + eps)" if mode == "order" else "continuum (2GE/r_0)^(3/2)"
        br = RM.outgoing_branch(1.0, v_L, R_L, kap_, a.m, omega, "refuse")
        if not br["admitted"]:
            crits.append(criterion("%s ray map against the %s" % (name, lab), False, "q %g on %s: %s" % (q, disp, br.get("reason"))))
            summary["static"]["ray_" + name] = {"q": q, "kappa": kap_, "admitted_at_launch": False, "branch": br}
            continue
        tr = RM.RayTracer(Fs, kap_, a.ray_dt)
        r0, k0, lost = tr.trace_back([k_ for k_, _ in rclocks], [R_L] * len(rclocks), [br["k_0"]] * len(rclocks))
        uu = np.array([u_ for _, u_ in rclocks])
        dr = RM.map_derivatives(uu, r0)
        x0 = r0 - rs
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = 2.0 * rs * k0 ** 2 / x0
            kappa_cont = kH_rs * (rs / r0) ** 1.5
            kappa_1st = kH_rs * ((rs / r0) ** 1.5 - 3.0 * eps / (1.0 + eps))
            k_ray = dr["kappa1"]
        valid = np.isfinite(k_ray) & ~lost & (x0 > 0) & (eps <= eps_cap)
        if mode == "order":
            valid &= (x0 <= a.x_max_over_rs * rs)
            ref = kappa_1st
        else:
            ref = kappa_cont
        if valid.any():
            dev_ = np.abs(k_ray - ref)[valid] / kH_rs
            ok = bool(np.max(dev_) <= tol)
            det = "q %g on %s (k_0 at the launch %.3e): %d clocks admitted (eps <= %g%s): max |kappa_ray - %s| %.4f x 1/(4GE) (tol %g); kappa_ray x 4GE from %.3f to %.3f; x_0 from %.1f to %.1f spacings; eps from %.2e to %.2e" % (
                q, disp, br["k_0"], int(valid.sum()), eps_cap, (", x_0 <= %g 2GE" % a.x_max_over_rs) if mode == "order" else "", lab, float(np.max(dev_)), tol, float(np.min(k_ray[valid] / kH_rs)),
                float(np.max(k_ray[valid] / kH_rs)), float(np.min(x0[valid])), float(np.max(x0[valid])), float(np.min(eps[valid])), float(np.max(eps[valid])))
            if mode == "cont":
                reach = float(np.max(k_ray[valid] / kH_rs))
                ok = ok and reach >= a.kappa_reach_min
                det += "; reaches %.4f of 1/(4GE) (min %g); the admitted lattice correction 3 eps/(1 + eps) at most %.4f" % (reach, a.kappa_reach_min, 3.0 * eps_cap / (1.0 + eps_cap))
        else:
            ok = False; det = "q %g on %s: no clock admitted on this lattice (eps cap %g, x_max %g 2GE)" % (q, disp, eps_cap, a.x_max_over_rs)
        crits.append(criterion("%s ray map against the %s" % (name, lab), ok, det))
        rc0 = [continuum_r0(u_ * rs, rs, R_L) for u_ in uu]
        summary["static"]["ray_" + name] = {"q": q, "kappa": kap_, "admitted_at_launch": True, "k_0_launch": br["k_0"], "u": uu.tolist(), "r0_ray": r0.tolist(), "k0_ray": k0.tolist(),
                                            "lost": lost.tolist(), "kappa_ray_x4GE": (k_ray / kH_rs).tolist(), "kappa_cont_x4GE": (kappa_cont / kH_rs).tolist(),
                                            "kappa_1st_x4GE": (kappa_1st / kH_rs).tolist(), "eps": eps.tolist(), "x0": x0.tolist(), "r0_continuum_exact": rc0, "admitted": valid.tolist()}
        for key, arr in (("r0", r0), ("k0", k0), ("kappa_ray", k_ray), ("kappa_cont", kappa_cont), ("kappa_1st", kappa_1st), ("eps", eps), ("u", uu)):
            store.Z["selftest_ray_%s_%s" % (name, key)] = np.asarray(arr, float)
    store.flush()
    # ---------------------------------------------------------------------------------------- S5: the branch-edge rule
    gap = (a.kappa / R_L) * 1.0 * math.sqrt(max(1.0 - v_L ** 2, 0.0))
    omega_edge = a.q_edge_frac * gap; q_edge = omega_edge * rs
    b_ref = RM.outgoing_branch(1.0, v_L, R_L, a.kappa, a.m, omega_edge, "refuse", a.branch_edge_margin)
    s5a = bool((not b_ref["admitted"]) and b_ref["edge"] == "below" and b_ref.get("omega_turn") is not None and abs(b_ref["omega_turn"] - gap) <= a.gap_tol_rel * gap and b_ref["k_0"] is None)
    crits.append(criterion("S5a the branch edge: a frequency below the gap is refused without an exception, omega_turn equals the small-k gap", s5a,
                           "kappa %d at R_L %.0f (v_L %.4f): omega_turn %.6e against N (kappa/R_L) sqrt(1 - (v_L/N)^2) %.6e (relative difference %.2e, tol %g); k_turn %s, omega_peak %s (q %s); omega %.4e (q %.4f, %g of the gap): admitted %s, edge %s; reason: %s" % (
                               a.kappa, R_L, v_L, b_ref.get("omega_turn") or float("nan"), gap, abs((b_ref.get("omega_turn") or float("nan")) - gap) / gap, a.gap_tol_rel, _g(b_ref.get("k_turn"), "%.4e"),
                               _g(b_ref.get("omega_peak"), "%.4e"), _g((b_ref.get("omega_peak") if b_ref.get("omega_peak") is not None else float("nan")) * rs, "%.4f"), omega_edge, q_edge, a.q_edge_frac,
                               b_ref["admitted"], b_ref["edge"], b_ref.get("reason"))))
    spec_e = [{"k": clocks[0][0], "u_rs": clocks[0][1], "q": q_edge, "omega": omega_edge}]
    try:
        rows_e = RM.run_channel(Fs, a.kappa, spec_e, cfg, rs, n_L, dev, store)
        ok_e = len(rows_e) == 1 and bool(rows_e[0].get("refused")) and rows_e[0].get("T0") is None and rows_e[0].get("fb") is None
        det_e = "run_channel on one spec at q %.4f: %d row(s), refused %s, T0 %s, reason: %s; the row is complete for the resume (row_done %s)" % (
            q_edge, len(rows_e), rows_e[0].get("refused") if rows_e else None, "absent" if (rows_e and rows_e[0].get("T0") is None) else "present",
            rows_e[0].get("reason") if rows_e else None, store.row_done(store.row_key(a.kappa, clocks[0][1], q_edge)))
    except Exception as ex:                                                 # noqa: BLE001
        ok_e = False; det_e = "run_channel raised %s: %s" % (type(ex).__name__, ex)
    crits.append(criterion("S5b the production path writes a refusal row (no propagation, no exception)", ok_e, det_e))
    b_near = RM.outgoing_branch(1.0, v_L, R_L, a.kappa, a.m, omega_edge, "nearest", a.branch_edge_margin)
    ok_n = bool(b_near["admitted"] and b_near["edge"] == "below" and b_near["k_0"] is not None and b_near["v_g_k0"] > 0
                and abs(b_near["omega_used"] - b_ref["omega_turn"] * (1.0 + a.branch_edge_margin)) <= 1e-12 * max(b_ref["omega_turn"], 1e-300))
    det_n = "nearest rule (margin %g): admitted %s, edge %s, omega_used %.6e = omega_turn x %.4f (q %.4f), k_0 %s, v_+(k_0) %s" % (
        a.branch_edge_margin, b_near["admitted"], b_near["edge"], b_near["omega_used"], b_near["omega_used"] / b_ref["omega_turn"] if b_ref.get("omega_turn") else float("nan"),
        b_near["omega_used"] * rs, _g(b_near.get("k_0"), "%.4e"), _g(b_near.get("v_g_k0"), "%.4f"))
    try:
        cfg_n = dict(cfg, branch_edge_rule="nearest")
        psi_n, info_n, (N_n, v_n) = RM.build_packet(Fs, clocks[0][0], a.kappa, omega_edge, n_L, cfg_n, rs, dev)
        if psi_n is None:
            ok_n = False; det_n += "; build_packet refused: %s" % info_n.get("reason")
        else:
            h_n = Fs.h_csr(a.kappa, N_n, v_n); Bh_n = Fs.banded(a.kappa, h_n, dev); Vb_n = OPS.Banded(OPS.velocity_csr(h_n, R), dev)
            R_t = torch.as_tensor(R, dtype=RM.RD, device=dev)
            dn, _, _, _ = RM.packet_diagnostics(psi_n, R_t, Bh_n, Vb_n, len(R), cfg["k_super"], cfg["k_dispersive"])
            ok_n = ok_n and abs(dn["norm"] - 1.0) <= 1e-12
            det_n += "; build_packet: a unit packet (norm - 1 %.1e), energy matched %s, <h>/omega_used %.3f (spread %.3f), group velocity %.4f, cut fraction %.3f, edge distance %.2f sigma_k, sigma_x %.1f spacings: %s" % (
                dn["norm"] - 1.0, info_n["energy_matched"], dn["energy_h"] / info_n["omega_used"], dn["energy_spread_h"] / info_n["omega_used"], dn["group_velocity"],
                info_n["weight_cut_fraction"], info_n["k0_minus_kturn_over_sigma_k"], dn["sigma_x"],
                "a probe at the requested frequency" if (info_n["energy_matched"] and info_n["weight_cut_fraction"] < 0.1) else "not a probe at omega_used (the spectral width exceeds the distance to the edge), which is why the default rule refuses")
            summary["edge"]["nearest_packet"] = {k_: v_ for k_, v_ in info_n.items()}
            summary["edge"]["nearest_packet"].update({"energy_h_over_omega_used": dn["energy_h"] / info_n["omega_used"], "group_velocity": dn["group_velocity"], "sigma_x_measured": dn["sigma_x"]})
    except Exception as ex:                                                 # noqa: BLE001
        ok_n = False; det_n += "; build_packet raised %s: %s" % (type(ex).__name__, ex)
    crits.append(criterion("S5c the nearest rule returns an admitted carrier above the gap and a unit packet", ok_n, det_n))
    summary["edge"].update({"gap_small_k": gap, "q_gap": gap * rs, "omega_edge": omega_edge, "q_edge": q_edge, "refuse": b_ref, "nearest": b_near})
    store.flush()
    # --------------------------------------------------------------------------------------------------------- FLAT
    Ff = ArrayFields(R, np.arange(0.0, a.flat_T_to_rs * rs + 2.0 * a.tau, a.tau), np.ones(len(R)), np.zeros(len(R)), E, g, a.m, label="flat space N = 1, v = 0")
    Tf = Ff.T
    T_launch = np.arange(a.flat_T_from_rs * rs, a.flat_T_to_rs * rs + 1e-9, a.u_step_rs * rs)
    specs_f = []; seen = set()
    for TL in T_launch:
        k = int(np.argmin(np.abs(Tf - TL)))
        if k in seen:
            continue
        seen.add(k); specs_f.append({"k": k, "u_rs": (Tf[k] - R_L) / rs, "q": a.q_geo, "omega": a.q_geo / rs})
    store_f = RM.RayStore(os.path.join(a.out, "raymap_selftest_%s_flat" % a.tag), settings=dict(vars(a), cfg=cfg), label="raymap_selftest_v2: " + Ff.label)
    t0 = time.time()
    rows_f = [r for r in RM.run_channel(Ff, a.kappa, specs_f, cfg, rs, n_L, dev, store_f) if r.get("T0") is not None]
    mf = RM.analyse_map(rows_f, rs, cfg)
    Af = mf["arrays"]
    kap = np.abs(Af["kappa1"] / Af["kappa_H_per_rs"]); kap = kap[np.isfinite(kap)]
    fb_f = np.array([r["fb"]["defect_norm"] if r.get("fb") else np.nan for r in rows_f]); nrm_f = np.array([r["T0"]["norm_T0"] for r in rows_f])
    s_flat = bool(len(kap) > 0 and np.max(kap) <= a.flat_kappa_tol)
    crits.append(criterion("FLAT kappa(u) = 0", s_flat, "%d clocks: max |kappa| %.2e x 1/(4GE) (tol %g); the map's slope dU/du %.5f to %.5f spacings per r_s; <|k|> at T = 0 %.5f to %.5f against the launch's %.5f" % (
        len(kap), float(np.max(kap)) if len(kap) else float("nan"), a.flat_kappa_tol, float(np.nanmin(Af["dUdu1"])), float(np.nanmax(Af["dUdu1"])), float(np.min(Af["k_mean_abs_T0"])),
        float(np.max(Af["k_mean_abs_T0"])), rows_f[0]["launch"]["k_mean_abs"])))
    s4f = bool(np.all(np.isfinite(fb_f)) and np.nanmax(fb_f) <= a.fb_tol and np.max(np.abs(nrm_f - 1.0)) <= a.norm_tol)
    crits.append(criterion("S4 forward-backward returns the launch (flat)", s4f, "max ||psi_fwd - psi_L|| %.2e, max |norm - 1| %.2e, %d rows, %.0f s" % (
        float(np.nanmax(fb_f)), float(np.max(np.abs(nrm_f - 1.0))), len(rows_f), time.time() - t0)))
    summary["flat"] = {"n_rows": len(rows_f), "u": Af["u_over_rs"].tolist(), "U": Af["U"].tolist(), "kappa1_x4GE": (Af["kappa1"] / Af["kappa_H_per_rs"]).tolist(), "fb_defect": fb_f.tolist()}
    store_f.flush()
    # ------------------------------------------------------------------------------------------------------- the verdict
    allpass = all(c["pass"] for c in crits)
    summary["criteria"] = crits; summary["pass"] = allpass
    pj = os.path.join(a.out, "selftest_%s_summary.json" % a.tag)
    with open(pj, "w") as f:
        json.dump(summary, f, default=RM._jdefault, indent=1)
    md = ["# raymap_selftest_v2: %s" % ("PASS" if allpass else "FAIL"), "", Fs.label, "", "| criterion | result | detail |", "|---|---|---|"]
    md += ["| %s | %s | %s |" % (c["name"], "PASS" if c["pass"] else "FAIL", c["detail"]) for c in crits]
    md += ["", "The centrifugal gap at the launch site, omega_turn r_s = kappa (2GE/R_L) sqrt(1 - 2GE/R_L) (N = 1, the vacuum flow):", "", "| kappa | launch radius (r_s) | q_gap |", "|---|---|---|"]
    md += ["| %d | %g | %.4f |" % (d["kappa"], d["launch_rs"], d["q_gap"]) for d in gaps]
    md += ["", "Static geometric probes (q %g): u/r_s, U/r_s, U_ray/r_s, |dU| (spacings), kappa x 4GE (packet, ray) at the fine step, at the step %d (packet, ray, the tolerance), <|k|> at T = 0, fb defect" % (a.q_geo, n_cmp), "",
           "| u/r_s | U/r_s | U_ray/r_s | dU | kappa x4GE | kappa_ray x4GE | kappa(n) x4GE | kappa_ray(n) x4GE | tol(n) x4GE | <abs k> T0 | fb |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    kpn = np.asarray(s1_extra.get("kappa_n_packet_x4GE", [np.nan] * len(rows))); krn = np.asarray(s1_extra.get("kappa_n_ray_x4GE", [np.nan] * len(rows)))
    tln = np.asarray(s1_extra.get("tolerance_n_x4GE", [np.nan] * len(rows))); dUs = np.asarray(s1_extra.get("dU_spacings", [np.nan] * len(rows)))
    for j in range(len(rows)):
        md.append("| %.3f | %.4f | %s | %s | %s | %s | %s | %s | %s | %.4f | %s |" % (
            u[j], U[j] / rs, _g(A["U_ray"][j] / rs, "%.4f") if "U_ray" in A else "-", _g(float(dUs[j]), "%.3f"), _g(A["kappa1"][j] / kH[j], "%.4f"),
            _g(A["kappa_ray1"][j] / kH[j], "%.4f") if "kappa_ray1" in A else "-", _g(float(kpn[j]), "%.4f"), _g(float(krn[j]), "%.4f"), _g(float(tln[j]), "%.4f"),
            A["k_mean_abs_T0"][j], _g(fb_def[j], "%.1e")))
    for name in ("S2", "S3"):
        d = summary["static"].get("ray_" + name) or {}
        if not d.get("admitted_at_launch"):
            md += ["", "Ray map %s: not admitted at the launch site (%s)" % (name, (d.get("branch") or {}).get("reason"))]
            continue
        md += ["", "Ray map at q %g on %s (%s): u/r_s, r_0 - 2GE, k_0, eps, kappa_ray x4GE, continuum x4GE, first-order x4GE, admitted" % (
            d["q"], ("kappa %d" % d["kappa"]) if d["kappa"] != 0 else "the centrifugal-free dispersion (kappa 0)", name), "", "| u/r_s | x_0 | k_0 | eps | ray | cont | 1st | adm |", "|---|---|---|---|---|---|---|---|"]
        for j in range(len(d["u"])):
            md.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (d["u"][j], _g(d["x0"][j], "%.2f"), _g(d["k0_ray"][j], "%.4e"), _g(d["eps"][j], "%.2e"), _g(d["kappa_ray_x4GE"][j], "%.4f"),
                                                                 _g(d["kappa_cont_x4GE"][j], "%.4f"), _g(d["kappa_1st_x4GE"][j], "%.4f"), int(d["admitted"][j])))
    with open(os.path.join(a.out, "selftest_%s.md" % a.tag), "w") as f:
        f.write("\n".join(md) + "\n")
    log("raymap_selftest_v2: %s (%d of %d criteria); %s" % ("PASS" if allpass else "FAIL", sum(c["pass"] for c in crits), len(crits), pj))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
