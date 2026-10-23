#!/usr/bin/env python
"""raymap_selftest_v1.py (reviewer A for the solve builder): the self-test of raymap_solved_flow_v1 on manufactured exteriors, by import of its
routines (the packet, the two propagations, the readouts, the difference rule, the plateau finder, the ray tracer; page_d13_readout_v1 and
page_ops_v6 beneath them).

Tests (every tolerance a named argument; PASS/FAIL per criterion, the whole on every criterion):
  STATIC: the static Schwarzschild-like exterior N = 1, v = -sqrt(2GE/r) in the Painleve-Gullstrand form at fixed 2GE = --rs (the lane's
    exterior-plus-ball flow page_ops_v6.flow_dense with the ball at --Rb-over-rs 2GE, so the horizon at 2GE is on the chain), the clock
    T - r_*(R) of page_ops_v6.chart_retarded_time. Probes at --q-geo (the geometric-optics frequency: wavelength well below 2GE, width
    --sigma-u-rs-test in u by the sigma-u rule) launched at --launch-rs 2GE at clocks --u-geo-from to --u-geo-to (r_s) in steps of
    --u-step-rs, back-propagated to T = 0 and read out exactly as the production code does.
    (S1) the position map reproduces the lattice geometric-optics ray map: |U_packet - U_ray| <= --U-tol-spacings and |kappa_packet -
         kappa_ray| <= --kappa-tol-rel x 1/(4GE) on every clock whose ray reaches T = 0 with |k| <= --k-ray-max;
    (S2) the ray map reproduces U - 2GE proportional to e^{-u/(4GE)} to the stated order: with x_0 = r_0 - 2GE and eps = 2GE k_0^2/x_0
         (the lattice dispersion's correction, from the near-horizon ray equations dx/dT = kappa x + (3/2) k^2, dk/dT = -kappa k),
         kappa_ray(u) = (1/(4GE)) [(2GE/r_0)^(3/2) - 3 eps/(1 + eps)] within --order-tol x 1/(4GE) at the clocks with eps <= --eps-max and
         x_0 <= --x-max-over-rs 2GE, at the frequency --q-mid (where such clocks exist on this lattice; the test reports when none do);
    (S3) the continuum limit: at the frequency --q-cont the ray map's kappa(u) equals the continuum (1/(4GE))(2GE/r_0(u))^(3/2) within
         --cont-tol x 1/(4GE) wherever eps <= --eps-max, and reaches at least --kappa-reach-min x 1/(4GE) (U - 2GE is exponential with
         the exponent 1/(4GE) up to the stated finite-r_0 factor); r_0(u) from r_*(r_0) = -u exactly;
    (S4) the forward re-propagation returns the launch to --fb-tol in ||psi_fwd - psi_L|| and the norms stay within --norm-tol of 1.
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
import raymap_solved_flow_v1 as RM  # noqa: E402
import page_d13_readout_v1 as D13  # noqa: E402
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
    ap.add_argument("--q-mid", dest="q_mid", type=float, default=1.0, help="omega r_s of the ray map's first-order check")
    ap.add_argument("--q-cont", dest="q_cont", type=float, default=0.02, help="omega r_s of the ray map's continuum check")
    ap.add_argument("--u-geo-from", dest="u_geo_from", type=float, default=-7.5); ap.add_argument("--u-geo-to", dest="u_geo_to", type=float, default=0.5)
    ap.add_argument("--u-ray-to", dest="u_ray_to", type=float, default=6.0, help="the ray checks' last clock (r_s)")
    ap.add_argument("--u-step-rs", dest="u_step_rs", type=float, default=0.125)
    ap.add_argument("--sigma-u-rs-test", dest="sigma_u_rs_test", type=float, default=0.1, help="the geometric probes' width in u (r_s)")
    ap.add_argument("--flat-T-from-rs", dest="flat_T_from_rs", type=float, default=0.5); ap.add_argument("--flat-T-to-rs", dest="flat_T_to_rs", type=float, default=3.0)
    ap.add_argument("--fb-tol", dest="fb_tol", type=float, default=1e-10); ap.add_argument("--norm-tol", dest="norm_tol", type=float, default=1e-10)
    ap.add_argument("--flat-kappa-tol", dest="flat_kappa_tol", type=float, default=0.02, help="|kappa| <= tol x 1/(4GE) in flat space")
    ap.add_argument("--kappa-tol-rel", dest="kappa_tol_rel", type=float, default=0.05, help="packet against ray: |d kappa| <= tol x 1/(4GE)")
    ap.add_argument("--U-tol-spacings", dest="U_tol_spacings", type=float, default=2.0, help="packet against ray: |dU| in spacings")
    ap.add_argument("--k-ray-max", dest="k_ray_max", type=float, default=0.5, help="clocks whose ray reaches T = 0 above this |k| are excluded from S1")
    ap.add_argument("--eps-max", dest="eps_max", type=float, default=0.05); ap.add_argument("--x-max-over-rs", dest="x_max_over_rs", type=float, default=0.15)
    ap.add_argument("--order-tol", dest="order_tol", type=float, default=0.05); ap.add_argument("--cont-tol", dest="cont_tol", type=float, default=0.02)
    ap.add_argument("--kappa-reach-min", dest="kappa_reach_min", type=float, default=0.9)
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
                "rs_nominal": rs, "norm_tol": a.norm_tol, "fb_tol": a.fb_tol, "in_vacuum": False, "save_states": True, "force": False, "m": a.m, "ray_compare": True, "ball_rs": None})
    n_L = int(np.argmin(np.abs(R - a.launch_rs * rs))); R_L = float(R[n_L])
    crits = []
    summary = {"settings": vars(a), "device": str(dev), "static": {}, "flat": {}}
    # ------------------------------------------------------------------------------------------------------- STATIC
    vfun = OPS.flow_dense(rs, a.Rb_over_rs * rs)
    v_pg = vfun(R); N_pg = np.ones(len(R))
    u_lat_max = a.u_ray_to * rs
    T_max = u_lat_max + (0.0 - OPS.chart_retarded_time(0.0, rs, R_L)) + 2.0 * a.tau
    T = np.arange(0.0, T_max + a.tau, a.tau)
    Fs = ArrayFields(R, T, N_pg, v_pg, E, g, a.m, label="static PG exterior 2GE %g with the ball flow inside %g 2GE (page_ops_v6.flow_dense)" % (rs, a.Rb_over_rs))
    store = RM.RayStore(os.path.join(a.out, "raymap_selftest_%s" % a.tag), settings=dict(vars(a), cfg=cfg), label="raymap_selftest_v1: " + Fs.label)
    u_geo = np.arange(a.u_geo_from, a.u_geo_to + 1e-9, a.u_step_rs)
    clocks = RM.clock_records(Fs, R_L, rs, u_geo)
    log("static: 2GE %g, %d records of tau %g to T/r_s %.2f, launch site %d (R %.1f = %g r_s), %d geometric clocks u/r_s %.3f to %.3f (records %d to %d), q_geo %g" % (
        rs, len(T), a.tau, T[-1] / rs, n_L, R_L, R_L / rs, len(clocks), clocks[0][1], clocks[-1][1], clocks[0][0], clocks[-1][0], a.q_geo))
    specs = [{"k": k_, "u_rs": u_, "q": a.q_geo, "omega": a.q_geo / rs} for (k_, u_) in clocks]
    t0 = time.time()
    rows = RM.run_channel(Fs, a.kappa, specs, cfg, rs, n_L, dev, store)
    rows = [r for r in rows if r.get("T0") is not None]
    m = RM.analyse_map(rows, rs, cfg)
    A = m["arrays"]
    log("static geometric probes: %d rows in %.0f s; packet minus ray max %.3f spacings" % (len(rows), time.time() - t0, m.get("packet_minus_ray_max", float("nan"))))
    # S4 on the static rows
    fb_def = np.array([r["fb"]["defect_norm"] if r.get("fb") else np.nan for r in rows]); nrm = np.array([r["T0"]["norm_T0"] for r in rows])
    nmin = np.array([r["T0"]["norm_min_along"] for r in rows]); nmax = np.array([r["T0"]["norm_max_along"] for r in rows])
    s4 = bool(np.all(np.isfinite(fb_def)) and np.nanmax(fb_def) <= a.fb_tol and np.max(np.abs(nrm - 1.0)) <= a.norm_tol and np.max(np.abs(nmin - 1.0)) <= a.norm_tol and np.max(np.abs(nmax - 1.0)) <= a.norm_tol)
    crits.append(criterion("S4 forward-backward returns the launch (static)", s4, "max ||psi_fwd - psi_L|| %.2e (tol %g), max |norm - 1| %.2e (tol %g), %d rows" % (
        float(np.nanmax(fb_def)), a.fb_tol, float(max(np.max(np.abs(nrm - 1.0)), np.max(np.abs(nmin - 1.0)), np.max(np.abs(nmax - 1.0)))), a.norm_tol, len(rows))))
    # S1 packet against the Wigner-averaged lattice ray
    kray = A.get("k_ray"); sel = np.isfinite(A["kappa1"]) & np.isfinite(A.get("kappa_ray1", np.full(len(rows), np.nan))) & (np.abs(kray) <= a.k_ray_max if kray is not None else False)
    if "U_ray" in A and sel.any():
        dU = np.abs(A["U"] - A["U_ray"])[sel]; dk = (np.abs(A["kappa1"] - A["kappa_ray1"]) / A["kappa_H_per_rs"])[sel]
        s1 = bool(np.max(dU) <= a.U_tol_spacings and np.max(dk) <= a.kappa_tol_rel)
        det = "%d clocks with |k_ray| <= %g: max |U_packet - U_ray| %.3f spacings (tol %g), max |kappa_packet - kappa_ray| %.4f x 1/(4GE) (tol %g); kappa x 4GE from %.3f to %.3f" % (
            int(sel.sum()), a.k_ray_max, float(np.max(dU)), a.U_tol_spacings, float(np.max(dk)), a.kappa_tol_rel, float(np.nanmin((A["kappa1"] / A["kappa_H_per_rs"])[sel])), float(np.nanmax((A["kappa1"] / A["kappa_H_per_rs"])[sel])))
    else:
        s1 = False; det = "no clock admitted (no ray data or every ray beyond k_ray_max)"
    crits.append(criterion("S1 packet centroid against the lattice ray map (static, q_geo)", s1, det))
    summary["static"]["geometric"] = {"n_rows": len(rows), "map": {k_: v_ for k_, v_ in m.items() if k_ != "arrays"}, "u": A["u_over_rs"].tolist(), "U": A["U"].tolist(),
                                      "U_ray": A["U_ray"].tolist() if "U_ray" in A else None, "kappa1_x4GE": (A["kappa1"] / A["kappa_H_per_rs"]).tolist(),
                                      "kappa_ray1_x4GE": (A["kappa_ray1"] / A["kappa_H_per_rs"]).tolist() if "kappa_ray1" in A else None, "k_ray": kray.tolist() if kray is not None else None,
                                      "k_mean_abs_T0": A["k_mean_abs_T0"].tolist(), "fb_defect": fb_def.tolist()}
    # S2 and S3: the ray map alone at q_mid and q_cont on a dense clock set
    u_ray = np.arange(a.u_geo_from, a.u_ray_to + 1e-9, a.u_step_rs)
    rclocks = RM.clock_records(Fs, R_L, rs, u_ray)
    for name, q, tol, mode in (("S2", a.q_mid, a.order_tol, "order"), ("S3", a.q_cont, a.cont_tol, "cont")):
        omega = q / rs
        br = RM.outgoing_branch(1.0, float(v_pg[n_L]), R_L, a.kappa, a.m, omega)
        tr = RM.RayTracer(Fs, a.kappa, a.ray_dt)
        r0, k0, lost = tr.trace_back([k_ for k_, _ in rclocks], [R_L] * len(rclocks), [br["k_0"]] * len(rclocks))
        uu = np.array([u_ for _, u_ in rclocks])
        dr = RM.map_derivatives(uu, r0)
        x0 = r0 - rs
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = 2.0 * rs * k0 ** 2 / x0
            kappa_cont = kH_rs * (rs / r0) ** 1.5
            kappa_1st = kH_rs * ((rs / r0) ** 1.5 - 3.0 * eps / (1.0 + eps))
            k_ray = dr["kappa1"]
        valid = np.isfinite(k_ray) & ~lost & (x0 > 0) & (eps <= a.eps_max)
        if mode == "order":
            valid &= (x0 <= a.x_max_over_rs * rs)
            ref = kappa_1st; lab = "first-order lattice formula (2GE/r_0)^(3/2) - 3 eps/(1 + eps)"
        else:
            ref = kappa_cont; lab = "continuum (2GE/r_0)^(3/2)"
        if valid.any():
            dev_ = np.abs(k_ray - ref)[valid] / kH_rs
            ok = bool(np.max(dev_) <= tol)
            det = "q %g: %d clocks admitted (eps <= %g%s): max |kappa_ray - %s| %.4f x 1/(4GE) (tol %g); kappa_ray x 4GE from %.3f to %.3f; x_0 from %.1f to %.1f spacings; eps from %.2e to %.2e" % (
                q, int(valid.sum()), a.eps_max, (", x_0 <= %g 2GE" % a.x_max_over_rs) if mode == "order" else "", lab, float(np.max(dev_)), tol, float(np.min(k_ray[valid] / kH_rs)),
                float(np.max(k_ray[valid] / kH_rs)), float(np.min(x0[valid])), float(np.max(x0[valid])), float(np.min(eps[valid])), float(np.max(eps[valid])))
            if mode == "cont":
                reach = float(np.max(k_ray[valid] / kH_rs))
                ok = ok and reach >= a.kappa_reach_min
                det += "; reaches %.4f of 1/(4GE) (min %g)" % (reach, a.kappa_reach_min)
        else:
            ok = False; det = "q %g: no clock admitted on this lattice (eps_max %g, x_max %g 2GE)" % (q, a.eps_max, a.x_max_over_rs)
        crits.append(criterion("%s ray map against the %s" % (name, lab), ok, det))
        rc0 = [continuum_r0(u_ * rs, rs, R_L) for u_ in uu]
        summary["static"]["ray_" + name] = {"q": q, "u": uu.tolist(), "r0_ray": r0.tolist(), "k0_ray": k0.tolist(), "lost": lost.tolist(), "kappa_ray_x4GE": (k_ray / kH_rs).tolist(),
                                            "kappa_cont_x4GE": (kappa_cont / kH_rs).tolist(), "kappa_1st_x4GE": (kappa_1st / kH_rs).tolist(), "eps": eps.tolist(), "x0": x0.tolist(),
                                            "r0_continuum_exact": rc0, "admitted": valid.tolist()}
        for key, arr in (("r0", r0), ("k0", k0), ("kappa_ray", k_ray), ("kappa_cont", kappa_cont), ("kappa_1st", kappa_1st), ("eps", eps), ("u", uu)):
            store.Z["selftest_ray_%s_%s" % (name, key)] = np.asarray(arr, float)
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
    store_f = RM.RayStore(os.path.join(a.out, "raymap_selftest_%s_flat" % a.tag), settings=dict(vars(a), cfg=cfg), label="raymap_selftest_v1: " + Ff.label)
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
    md = ["# raymap_selftest_v1: %s" % ("PASS" if allpass else "FAIL"), "", Fs.label, "", "| criterion | result | detail |", "|---|---|---|"]
    md += ["| %s | %s | %s |" % (c["name"], "PASS" if c["pass"] else "FAIL", c["detail"]) for c in crits]
    md += ["", "Static geometric probes (q %g): u/r_s, U/r_s, U_ray/r_s, kappa x 4GE (packet, ray), <|k|> at T = 0, fb defect" % a.q_geo, "", "| u/r_s | U/r_s | U_ray/r_s | kappa x4GE | kappa_ray x4GE | <abs k> T0 | fb |", "|---|---|---|---|---|---|---|"]
    for j in range(len(rows)):
        md.append("| %.3f | %.4f | %s | %s | %s | %.4f | %s |" % (A["u_over_rs"][j], A["U"][j] / rs, _g(A["U_ray"][j] / rs, "%.4f") if "U_ray" in A else "-", _g(A["kappa1"][j] / A["kappa_H_per_rs"][j], "%.4f"),
                                                             _g(A["kappa_ray1"][j] / A["kappa_H_per_rs"][j], "%.4f") if "kappa_ray1" in A else "-", A["k_mean_abs_T0"][j], _g(fb_def[j], "%.1e")))
    for name in ("S2", "S3"):
        d = summary["static"]["ray_" + name]
        md += ["", "Ray map at q %g (%s): u/r_s, r_0 - 2GE, k_0, eps, kappa_ray x4GE, continuum x4GE, first-order x4GE, admitted" % (d["q"], name), "", "| u/r_s | x_0 | k_0 | eps | ray | cont | 1st | adm |", "|---|---|---|---|---|---|---|---|"]
        for j in range(len(d["u"])):
            md.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (d["u"][j], _g(d["x0"][j], "%.2f"), _g(d["k0_ray"][j], "%.4f"), _g(d["eps"][j], "%.2e"), _g(d["kappa_ray_x4GE"][j], "%.4f"),
                                                                 _g(d["kappa_cont_x4GE"][j], "%.4f"), _g(d["kappa_1st_x4GE"][j], "%.4f"), int(d["admitted"][j])))
    with open(os.path.join(a.out, "selftest_%s.md" % a.tag), "w") as f:
        f.write("\n".join(md) + "\n")
    log("raymap_selftest_v1: %s (%d of %d criteria); %s" % ("PASS" if allpass else "FAIL", sum(c["pass"] for c in crits), len(crits), pj))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
