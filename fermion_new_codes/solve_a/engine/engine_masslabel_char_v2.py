#!/usr/bin/env python
"""engine_masslabel_char_v2.py (builder analyst). Version 2 = version 1 with two readouts added: the first trapping among the labels
outside the central averaging cell (r > --r-exclude, default the run's ell; version 1's first trapping on the S11 test profile was a label
at r 1.16a, inside the cell, where the single stream with no central flux piles labels onto the centre), and the label starting at R0
(--R0-rs, in r_s) with its trapping clock beside the closed-form crossing from R0 in the exterior of the whole charge. Nothing else changes.
Version 1's docstring follows.
engine_masslabel_char_v1.py (builder analyst; the quasi-frozen ruling's S11, owner-approved (inbox #36): "the mass-label characteristic
solve of P28 sec. 6 run beside it as the continuum predictor (the crossing test (t_H - t_0)/r_s = 1 + 2 ln(3/2) = 1.811 from R_0 = 4 r_s)";
P28's report is an idea file: the construction below is re-derived here, not cited as a source).

The construction (continuum, a single ingoing stream of positive null matter with no central flux; flat slices, N normalized at infinity):
with w = v/N = -s, M = r w^2/(2G) the declared constraints give M' = 4 pi r^2 (rho + w J) and (ln N)' = -4 pi G r J/w; for J = -rho
(null, ingoing) M' = 4 pi r^2 rho (1 + s) and the ingoing characteristic speed is A = N - v = N (1 + s). Labelling the ingoing
characteristics by the enclosed mass, M = E u^2 (0 <= u <= 1), b(u, t) = (2 G E / r(u, t))^(1/2) and s = u b:
    dr/dt (u, t) = -N(u, t) [1 + u b(u, t)],        ln N(u, t) = int_u^1 b(z, t) / (1 + z b(z, t)) dz,
the second from (ln N)' = -4 pi G r J/w = 4 pi G r rho / s with M_u = 2 E u and M' = 4 pi r^2 rho (1 + s) (so (ln N)_u = -(ln N)_r r_u =
-b/(1 + u b)), normalized N(1) = 1 at the outermost label (the vacuum exterior). Checks printed:
(i) the energy identity E = int_0^E N^2/(1 + s) dM (d(N^2 M)/dM = N^2/(1 + s) from the two equations; N(E) = 1), a quadrature check of N;
(ii) the outermost label's crossing of r_s = 2GE: on u = 1, dr/dt = -(1 + (r_s/r)^(1/2)), so t(R0 -> r) = F_in(R0) - F_in(r) with
F_in(R) = R - 2 (r_s R)^(1/2) + 2 r_s ln(1 + (R/r_s)^(1/2)); from R0 = 4 r_s, (t_H - t_0)/r_s = 1 + 2 ln(3/2);
(iii) the label spacings r_u > 0 and U_phys = 2 G N^2 M >= 0;
(iv) the local derivative norm of the velocity map V[r](u) = -N(u)[1 + u b(u)] (its Frechet derivative has the nonnegative kernel
N(1 + ub)[ub/(2r(1 + ub)) delta + b(z)/(2 r(z) (1 + z b(z))^2) on z > u]; its infinity norm is the largest row sum), and the implicit
midpoint map's tau ||DV|| / 2 at the engine's tau.
Labels that reach r <= --r-min (the regular centre) are held there and counted: the single-stream construction (no central flux) is no
longer the history past that clock for them; their mass fraction is printed at every output clock. The trapping: s(u, t) >= 1, i.e.
r(u, t) <= 2 G M(u); t_H(u) per label by interpolation between steps, the first trapping clock t_* = min t_H and its label and radius,
and the outer root R_h(t) = 2 G M(q_+(t)) on the outer branch at the output clocks.

Input: --run dir/tag (a collapse run's record json and fields npz; the T = 0 shell totals re-solved by --fields-module, M(r) = W(r)/(2g) on
the solve's fine grid, E its charge) or --profile thinlayer:R0,width (the unit-test layer r(u, 0) = R0 + width (u^2 - 1/2)); --rs (the
r_s of the clocks' unit), --g, --nu (labels, uniform in u on (0, 1]), --dt, --t-end, --out-every; --unit-tests runs the ladder below.
Every number is printed by the script."""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def F_in(R, rs):
    return R - 2.0 * math.sqrt(rs * R) + 2.0 * rs * math.log(1.0 + math.sqrt(R / rs))


def interval_integrals(f, h):
    """the integrals of the cubic through four neighbouring points over each interval of a uniform grid (fourth order): h/24 (-f_{i-1} +
    13 f_i + 13 f_{i+1} - f_{i+2}) inside, h/24 (9 f_0 + 19 f_1 - 5 f_2 + f_3) and its mirror at the ends; the trapezoid below 4 points."""
    n = len(f)
    if n < 4:
        return 0.5 * h * (f[1:] + f[:-1])
    I = np.empty(n - 1)
    I[1:-1] = h / 24.0 * (-f[:-3] + 13.0 * f[1:-2] + 13.0 * f[2:-1] - f[3:])
    I[0] = h / 24.0 * (9.0 * f[0] + 19.0 * f[1] - 5.0 * f[2] + f[3])
    I[-1] = h / 24.0 * (f[-4] - 5.0 * f[-3] + 19.0 * f[-2] + 9.0 * f[-1])
    return I


def cum_from_top(f, u):
    """int_u^1 f(z) dz on the label grid (uniform in u; fourth order), from the top label u = 1 down."""
    out = np.zeros_like(f)
    seg = interval_integrals(f, float(u[1] - u[0])) if len(u) > 1 else np.zeros(0)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def lapse(r, u, E, G, r_min):
    b = np.sqrt(2.0 * G * E / np.maximum(r, r_min))
    lnN = cum_from_top(b / (1.0 + u * b), u)
    return np.exp(lnN), b


def velocity(r, u, E, G, r_min, held):
    N, b = lapse(r, u, E, G, r_min)
    V = -N * (1.0 + u * b)
    V[held] = 0.0
    return V


def dv_norm(r, u, E, G, r_min):
    """the infinity norm of DV (the largest row sum of the nonnegative kernel), on the labels above the centre."""
    N, b = lapse(r, u, E, G, r_min)
    ub = u * b
    diag = ub / (2.0 * np.maximum(r, r_min) * (1.0 + ub))
    k = b / (2.0 * np.maximum(r, r_min) * (1.0 + ub) ** 2)
    tail = cum_from_top(k, u)
    rows = N * (1.0 + ub) * (diag + tail)
    return float(np.max(rows))


def energy_identity(r, u, E, G, r_min):
    """int_0^E N^2/(1 + s) dM with M = E u^2 (dM = 2 E u du), against E."""
    N, b = lapse(r, u, E, G, r_min)
    s = u * b
    f = N ** 2 / (1.0 + s) * 2.0 * E * u
    # the integrand vanishes at u = 0 (dM = 2 E u du): the grid extended by u = 0, f = 0 (uniform spacing u_0 = u_1 - u_0), fourth order
    return float(np.sum(interval_integrals(np.concatenate([[0.0], f]), float(u[0]))))


def solve(u, r0, E, G, rs_unit, dt, t_end, r_min, out_every, tau_engine, log=print, r_exclude=0.0, R0_label=None):
    r = np.array(r0, float)
    held = r <= r_min
    t = 0.0
    nsteps = int(round(t_end / dt))
    tH = np.full(len(u), np.nan)
    s_prev = u * np.sqrt(2.0 * G * E / np.maximum(r, r_min))
    tH[(s_prev >= 1.0) & ~held] = 0.0
    rows = []
    first_bad = None

    def snapshot(t_, r_, held_):
        N, b = lapse(r_, u, E, G, r_min)
        s = u * b
        Mlab = E * u ** 2
        trapped = (s >= 1.0) & ~held_
        ru = np.diff(r_)
        Uphys = 2.0 * G * N ** 2 * Mlab
        Rh = float(np.max(np.where(trapped, 2.0 * G * Mlab, -np.inf))) if trapped.any() else None
        return {"t": t_, "t_over_rs": t_ / rs_unit, "r_outer": float(r_[-1]), "r_inner_unheld": float(r_[~held_][0]) if (~held_).any() else None,
                "held_mass_fraction": float(np.max(np.where(held_, u ** 2, 0.0))) if held_.any() else 0.0, "N_centre_label": float(N[0]),
                "s_max": float(np.max(np.where(held_, 0.0, s))), "u_at_s_max": float(u[int(np.argmax(np.where(held_, 0.0, s)))]),
                "trapped_labels": int(trapped.sum()), "R_h_outer": Rh, "min_label_spacing": float(ru.min()) if len(ru) else None,
                "U_phys_min": float(Uphys.min()), "energy_identity_rel": energy_identity(r_, u, E, G, r_min) / E - 1.0,
                "dV_norm": dv_norm(r_, u, E, G, r_min), "midpoint_contraction_at_tau": 0.5 * tau_engine * dv_norm(r_, u, E, G, r_min)}
    rows.append(snapshot(t, r, held))
    k1 = velocity(r, u, E, G, r_min, held)
    for k in range(1, nsteps + 1):
        k2 = velocity(r + 0.5 * dt * k1, u, E, G, r_min, held)
        k3 = velocity(r + 0.5 * dt * k2, u, E, G, r_min, held)
        k4 = velocity(r + dt * k3, u, E, G, r_min, held)
        r_new = r + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        newly = (r_new <= r_min) & ~held
        r_new[newly] = r_min
        held = held | newly
        k1_new = velocity(r_new, u, E, G, r_min, held)
        s_new = u * np.sqrt(2.0 * G * E / np.maximum(r_new, r_min))
        cross = np.isnan(tH) & (s_new >= 1.0) & ~held
        for i in np.nonzero(cross)[0]:
            # the crossing r = 2 G M(u) on the cubic Hermite interpolant of r between the steps (the step-end velocities; fourth order)
            rt = 2.0 * G * E * u[i] ** 2
            h00 = lambda q: 2 * q ** 3 - 3 * q ** 2 + 1; h10 = lambda q: q ** 3 - 2 * q ** 2 + q; h01 = lambda q: -2 * q ** 3 + 3 * q ** 2; h11 = lambda q: q ** 3 - q ** 2
            rq = lambda q: h00(q) * r[i] + h10(q) * dt * k1[i] + h01(q) * r_new[i] + h11(q) * dt * k1_new[i] - rt
            lo_, hi_ = 0.0, 1.0
            if rq(lo_) <= 0.0:
                tH[i] = t
                continue
            for _ in range(80):
                mid = 0.5 * (lo_ + hi_)
                if rq(mid) > 0.0:
                    lo_ = mid
                else:
                    hi_ = mid
            tH[i] = t + dt * 0.5 * (lo_ + hi_)
        r = r_new; s_prev = s_new; t = k * dt; k1 = k1_new
        if np.any(np.diff(r[~held]) <= 0.0) and first_bad is None:
            first_bad = t
        if k % out_every == 0 or k == nsteps:
            rows.append(snapshot(t, r, held))
    ok = ~np.isnan(tH)
    first = None
    # version 2: the first trapping outside the central cell (the trapping radius 2 G M(u) > r_exclude) and the label starting at R0_label
    rstar = 2.0 * G * E * u ** 2
    okx = ok & (rstar > r_exclude)
    first_out = None
    if okx.any():
        ix = int(np.argmin(np.where(okx, tH, np.inf)))
        first_out = {"t_star": float(tH[ix]), "t_star_over_rs": float(tH[ix]) / rs_unit, "u": float(u[ix]), "M_over_E": float(u[ix] ** 2), "r_star": float(rstar[ix]),
                     "r_star_over_rs": float(rstar[ix]) / rs_unit, "r_exclude": r_exclude}
    lab0 = None
    if R0_label is not None:
        i0 = int(np.argmin(np.abs(np.asarray(r0, float) - R0_label)))
        lab0 = {"u": float(u[i0]), "r0": float(r0[i0]), "M_over_E": float(u[i0] ** 2), "t_H": None if np.isnan(tH[i0]) else float(tH[i0]),
                "t_H_over_rs": None if np.isnan(tH[i0]) else float(tH[i0]) / rs_unit, "r_final": float(r[i0])}
    if ok.any():
        i = int(np.nanargmin(tH))
        first = {"t_star": float(tH[i]), "t_star_over_rs": float(tH[i]) / rs_unit, "u": float(u[i]), "M_over_E": float(u[i] ** 2), "r_star": float(2.0 * G * E * u[i] ** 2),
                 "r_star_over_rs": float(2.0 * G * E * u[i] ** 2) / rs_unit}
    return {"rows": rows, "t_H": [None if np.isnan(x) else float(x) for x in tH], "first_trapping": first, "first_trapping_outside_cell": first_out, "label_at_R0": lab0,
            "first_nonmonotone_labels_t": first_bad,
            "outer_label_t_H": None if np.isnan(tH[-1]) else float(tH[-1]), "r_final": [float(x) for x in r]}


def labels_from_run(a):
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]; g = float(ra["g"]); ell = float(ra["ell"])
    FM = importlib.import_module(a.fields_module or ("engine_fields_v17" if ra.get("fields_v17") else "engine_fields_v16"))
    k = 0
    _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, np.asarray(Z["R"], float), ell,
                                             collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
    # the labels are the Misner-Sharp mass M = r w^2/(2g) = W/(2 g N^2) (the field module's W = r v^2); at the collar N = 1 and M = E
    M = np.asarray(info["W_field"](rf), float) / (2.0 * g * np.asarray(info["N_dense"](rf), float) ** 2)
    E = float(FM.charge(info, g))
    mono = bool(np.all(np.diff(M) >= -1e-14 * abs(E)))
    Mc = np.maximum.accumulate(M)                                             # the monotone envelope (equal to M where M is monotone)
    u = np.linspace(0.0, 1.0, a.nu + 1)[1:]
    target = E * u ** 2
    target[-1] = min(target[-1], Mc[-1])
    r0 = np.interp(target, Mc, rf)
    src = {"run": a.run, "T": float(Z["T"][k]), "E": E, "g": g, "ell": ell, "field_module": FM.__name__, "M_monotone": mono,
           "M_max_decrease": float(np.max(np.maximum(-np.diff(M), 0.0))) if len(M) > 1 else 0.0, "r_collar": rc,
           "r_at_M_half": float(np.interp(0.5 * E, Mc, rf)), "r_at_M_099": float(np.interp(0.99 * E, Mc, rf))}
    return u, r0, E, g, src, float(ra["rs"]), float(ra["tau"]), ell


def unit_tests(a):
    """the ladder of the docstring's checks on the unit-test layer r(u, 0) = 64 + 4 (u^2 - 1/2) (G 1/4, E 32: r_s = 16) to t = 24, and the
    outermost label's crossing against F_in from R0 = 4 r_s (a thin layer: every label at R0, the outer label's ray exact)."""
    out = {"layer": [], "crossing": []}
    G = 0.25; E = 32.0; rs = 2.0 * G * E
    for nu, dt in ((64, 0.25), (128, 0.125), (256, 0.0625)):
        u = np.linspace(0.0, 1.0, nu + 1)[1:]
        r0 = 64.0 + 4.0 * (u ** 2 - 0.5)
        res = solve(u, r0, E, G, rs, dt, 24.0, 1e-6, max(1, int(round(4.0 / dt))), 2.0, log=lambda s: None)
        last = res["rows"][-1]
        out["layer"].append({"nu": nu, "dt": dt, "energy_identity_rel_max": max(abs(q["energy_identity_rel"]) for q in res["rows"]),
                             "min_label_spacing": min(q["min_label_spacing"] for q in res["rows"]), "U_phys_min": min(q["U_phys_min"] for q in res["rows"]),
                             "r_outer_at_24": last["r_outer"], "s_max_at_24": last["s_max"]})
        print("unit test, the layer 64 + 4 (u^2 - 1/2) to t 24 (n_u %d, dt %g): the energy identity's largest relative residual %.3e; the smallest label spacing %.6f; U_phys min %.6e; r_outer(24) %.10f; s_max(24) %.8f" % (
            nu, dt, out["layer"][-1]["energy_identity_rel_max"], out["layer"][-1]["min_label_spacing"], out["layer"][-1]["U_phys_min"], last["r_outer"], last["s_max"]), flush=True)
    exact = (F_in(4.0 * rs, rs) - F_in(rs, rs)) / rs
    for dt in (1.0, 0.5, 0.25, 0.125):
        u = np.linspace(0.0, 1.0, 65)[1:]
        r0 = np.full(len(u), 4.0 * rs)
        r0 = r0 + 1e-9 * u                                                   # a thin layer at R0 (the labels' order kept)
        res = solve(u, r0, E, G, rs, dt, 3.0 * rs, 1e-6, 10 ** 9, 2.0, log=lambda s: None)
        tH1 = res["outer_label_t_H"]
        out["crossing"].append({"dt": dt, "t_H_outer_over_rs": tH1 / rs if tH1 is not None else None, "closed_form": exact,
                                "difference": (tH1 / rs - exact) if tH1 is not None else None})
        print("unit test, the outermost label from R0 = 4 r_s (dt %g): (t_H - t_0)/r_s %.10f against 1 + 2 ln(3/2) = %.10f (difference %.3e; the crossing on the cubic Hermite interpolant between steps)" % (
            dt, tH1 / rs if tH1 is not None else float("nan"), exact, (tH1 / rs - exact) if tH1 is not None else float("nan")), flush=True)
    out["closed_form_crossing"] = exact
    out["closed_form_check"] = 1.0 + 2.0 * math.log(1.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--fields-module", default=None)
    ap.add_argument("--profile", default=None, help="thinlayer:R0,width")
    ap.add_argument("--rs", type=float, default=None)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--E", type=float, default=None)
    ap.add_argument("--nu", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.0625)
    ap.add_argument("--t-end-rs", type=float, default=5.0)
    ap.add_argument("--out-every-rs", type=float, default=0.125)
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--tau-engine", type=float, default=2.0)
    ap.add_argument("--r-exclude", type=float, default=None, help="version 2: the central cell excluded from the first trapping (default the run's ell)")
    ap.add_argument("--R0-rs", type=float, default=4.0, help="version 2: the label starting nearest this radius (in r_s)")
    ap.add_argument("--unit-tests", action="store_true")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    out = {"script": "engine_masslabel_char_v2.py", "args": vars(a), "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    if a.unit_tests:
        out["unit_tests"] = unit_tests(a)
    if a.run or a.profile:
        if a.run:
            u, r0, E, g, src, rs, tau, ell_run = labels_from_run(a)
            out["source"] = src
            print("labels from %s at T %.3f: E %.10f (field module %s); M monotone %s (largest decrease %.3e); r at M = E/2 %.4f, at 0.99 E %.4f" % (
                a.run, src["T"], E, src["field_module"], src["M_monotone"], src["M_max_decrease"], src["r_at_M_half"], src["r_at_M_099"]), flush=True)
        else:
            R0, w = [float(x) for x in a.profile.split(":")[1].split(",")]
            g = a.g; E = a.E; rs = a.rs or 2.0 * g * E; tau = a.tau_engine; ell_run = 0.0
            u = np.linspace(0.0, 1.0, a.nu + 1)[1:]
            r0 = R0 + w * (u ** 2 - 0.5)
        G = g
        rs_unit = a.rs or 2.0 * G * E
        rx = a.r_exclude if a.r_exclude is not None else ell_run
        R0l = a.R0_rs * rs_unit if a.R0_rs else None
        res = solve(u, r0, E, G, rs_unit, a.dt, a.t_end_rs * rs_unit, a.r_min, max(1, int(round(a.out_every_rs * rs_unit / a.dt))), a.tau_engine, r_exclude=rx, R0_label=R0l)
        out["result"] = res
        out["crossing_closed_form_from_R0_4rs"] = 1.0 + 2.0 * math.log(1.5)
        R0o = float(r0[-1])
        out["outer_label_closed_form"] = {"R0": R0o, "R0_over_rs": R0o / (2 * G * E), "t_H_over_rs": (F_in(R0o, 2 * G * E) - F_in(2 * G * E, 2 * G * E)) / (2 * G * E) if R0o > 2 * G * E else None}
        if res.get("label_at_R0"):
            l0 = res["label_at_R0"]
            out["label_at_R0_closed_form_t_H_over_rs"] = (F_in(l0["r0"], 2 * G * E) - F_in(2 * G * E, 2 * G * E)) / (2 * G * E) if l0["r0"] > 2 * G * E else None
            print("the label starting at R0 (r0 %.4f, u %.6f, M/E %.6f): t_H/r_s %s, against the closed-form crossing from that radius in the exterior of the whole charge %s; its r at the end %.4f" % (
                l0["r0"], l0["u"], l0["M_over_E"], ("%.8f" % l0["t_H_over_rs"]) if l0["t_H_over_rs"] is not None else None,
                ("%.8f" % out["label_at_R0_closed_form_t_H_over_rs"]) if out["label_at_R0_closed_form_t_H_over_rs"] is not None else None, l0["r_final"]), flush=True)
        print("the first trapping outside the central cell (r > %.3f): %s" % (rx, json.dumps(res.get("first_trapping_outside_cell"))), flush=True)
        print("the outermost label from R0 %.4f (%.4f r_s): t_H/r_s integrated %s, closed form %s; the first trapping %s; labels non-monotone from t %s" % (
            R0o, R0o / (2 * G * E), ("%.8f" % (res["outer_label_t_H"] / rs_unit)) if res["outer_label_t_H"] is not None else None,
            ("%.8f" % out["outer_label_closed_form"]["t_H_over_rs"]) if out["outer_label_closed_form"]["t_H_over_rs"] else None, json.dumps(res["first_trapping"]),
            res["first_nonmonotone_labels_t"]), flush=True)
        for q in res["rows"]:
            print("t/r_s %.4f: r_outer %.6f, innermost free label r %s, held mass fraction %.6f, s_max %.6f at u %.4f, trapped labels %d, R_h %s, min spacing %.3e, U_phys min %.3e, energy identity %.2e, ||DV|| %.4e (midpoint tau ||DV||/2 %.4e)" % (
                q["t_over_rs"], q["r_outer"], ("%.4f" % q["r_inner_unheld"]) if q["r_inner_unheld"] is not None else None, q["held_mass_fraction"], q["s_max"], q["u_at_s_max"],
                q["trapped_labels"], ("%.4f" % q["R_h_outer"]) if q["R_h_outer"] is not None else None, q["min_label_spacing"] if q["min_label_spacing"] is not None else float("nan"),
                q["U_phys_min"], q["energy_identity_rel"], q["dV_norm"], q["midpoint_contraction_at_tau"]), flush=True)
    out["seconds"] = time.time() - t0
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
