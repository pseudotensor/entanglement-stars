#!/usr/bin/env python
"""static_rows_selftest_v1.py (lead reviewer A; the self-test of static_rows_v1.py; run before the records: PASS on every criterion required).

(A) The exact static pair: the continuum Oppenheimer-Volkoff member in the flat-slice chart (engine_ovpg_v1.shoot) with the fluid seen by the
normal observers (w = sqrt(2 g m(r)/r), F = 1 - w^2): rho = (e + P)/F - P, J = (e + P) w/F, S_r = (e + P) w^2/F + P, S_t = P, N = e^Phi/sqrt(F),
v = -N w. Both constraints and both rows hold on it (engine_rows_v2's validation (c), (c')), so BI4 demands W_0 = W_r = 0; analytically
v rho + N J = N w P, v J + N S_r = N P, W_0 = -v x TOV and W_r = N x TOV on these forms (the lead's derivation). Two equations of state:
the analytic polytrope P = K (mu - m)^2, n = 2 K (mu - m), e = K (mu - m)(mu + m) (smooth: the PASS criteria), and the table's consistent one
(P the table's cubic spline, e = mu dP/dmu - P: the engine's own, printed with looser thresholds because the spline's third derivative jumps
at the knots).
(B) The quadrature against the direct rows with the stresses perturbed (B1: both stresses times 1.3; B2: S_t times 1 + 0.5 (r/R_*)^2; both keep
S_r - S_t = O(r^2) at the centre): the constraints are unchanged, so BI5 must reproduce engine_rows_v2.brackets' rows; this tests the
transcription of BI2 and the sign R = -rhs/N.
(C) The regular-centre limit on B1: R_r(0) = R_t(0) = (4 pi g/(3 v_1)) W_0(0) = 1.2 pi g P_c, and W_r = O(r) with the pair's slope,
W_r/W_0 -> (28 pi g/9) N_0 (e_c + P_c) e_c r/(P_c v_1) (the lead's derivation, STATIC_rows_v1_selftest_ruling.md: the size of W_r/W_0 at a
fixed fraction of R_* scales with R_* and is no regularity constant).
(D) The units: the constraints H = B^2 + 2 A B - 8 pi g rho and C_r = B N'/N + 4 pi g J of BI1 on the pair by the same finite differences (the
normalization of rho and J against N, v with the engine's g), and the dimensional statement printed.
(E) Optional, --record and --NB: the lattice side on a member: the rebuilt shell totals against the stored ones, the smoothed density's charge
against the shell sum, the stress trace identity, the constraint residuals of the re-solve, the virial residual (printed; its known value
about 0.2 is not a criterion).
Writes <out>/static_rows_selftest_v1.md; exit code 1 on any FAIL.
Usage: python static_rows_selftest_v1.py --out <dir> [--code <engine code dir>] [--eos <eos_v4_tables.npz>] [--m 0.2 --g 0.25] [--zc 0.3,0.8]
       [--K 0.01] [--record <json> --NB 138.844] [--threads 8]"""
import os
import sys


def _early_threads():
    for i, x in enumerate(sys.argv):
        if x == "--threads" and i + 1 < len(sys.argv):
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(k, sys.argv[i + 1])


_early_threads()

import argparse
import json
import math
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def eos_table(eos, m):
    from scipy.interpolate import CubicSpline
    t = np.load(eos)
    sel = np.abs(t["m_hat"] - m) < 1e-12
    mu_t, P_t = t["mu_hat"][sel], t["P_hat"][sel]
    o = np.argsort(mu_t); mu_t, P_t = mu_t[o], P_t[o]
    Ps = CubicSpline(mu_t, P_t); dPs = Ps.derivative()
    P2 = lambda mu: float(Ps(mu)) if mu > m else 0.0
    n2 = lambda mu: float(dPs(mu)) if mu > m else 0.0
    e2 = lambda mu: (mu * float(dPs(mu)) - float(Ps(mu))) if mu > m else 0.0
    return e2, P2, n2


def eos_polytrope(K, m):
    P2 = lambda mu: K * (mu - m) ** 2 if mu > m else 0.0
    n2 = lambda mu: 2.0 * K * (mu - m) if mu > m else 0.0
    e2 = lambda mu: K * (mu - m) * (mu + m) if mu > m else 0.0
    return e2, P2, n2


def ovpg_pair(OV, m, g, zc, e_of, P_of, n_of):
    """the OV member and its normal-observer fields as vectorized callables (the recipe of engine_rows_v2.validate (c))."""
    res = OV.shoot(m, m, g, e_of, P_of, n_of, m * math.sqrt(1.0 + zc * zc))
    Rst, Mst, c, EF, sol = res["R_star"], res["M"], res["c"], res["EF_consistent"], res["sol"]

    def mm_phi(x):
        x = np.atleast_1d(np.asarray(x, float)); ax = np.abs(x)
        inside = ax <= Rst
        y = sol.sol(np.clip(ax, 1e-6, Rst))
        mm = np.where(inside, y[0], Mst)
        Phi = np.where(inside, y[1] + c, 0.5 * np.log(np.maximum(1.0 - 2.0 * g * Mst / np.maximum(ax, 1e-300), 1e-300)))
        return x, ax, mm, Phi

    def Nf(x):
        _, ax, mm, Phi = mm_phi(x)
        return np.exp(Phi) / np.sqrt(1.0 - 2.0 * g * mm / np.maximum(ax, 1e-300))

    def vf(x):
        x, ax, mm, Phi = mm_phi(x)
        return -np.sign(x) * Nf(ax) * np.sqrt(2.0 * g * mm / np.maximum(ax, 1e-300))

    def fluid(x):
        x, ax, mm, Phi = mm_phi(x)
        F = 1.0 - 2.0 * g * mm / np.maximum(ax, 1e-300)
        mu = EF / np.exp(Phi)
        e = np.array([e_of(float(u)) for u in mu]); P = np.array([P_of(float(u)) for u in mu])
        w2 = 2.0 * g * mm / np.maximum(ax, 1e-300); w = np.sqrt(w2)
        return (e + P) / F - P, np.sign(x) * (e + P) * w / F, (e + P) * w2 / F + P, P
    fields = {"N": Nf, "v": vf, "rho": lambda x: fluid(x)[0], "J": lambda x: fluid(x)[1], "S": lambda x: fluid(x)[2:4]}
    return fields, {"R_star": float(Rst), "M": float(Mst), "compactness": float(2.0 * g * Mst / Rst), "EF": float(EF), "zc": zc}


def perturbed(fields, kind, Rst):
    f = dict(fields)
    if kind == "B1":
        f["S"] = lambda x, S=fields["S"]: tuple(1.3 * np.asarray(s, float) for s in S(x))
    elif kind == "B2":
        def S2(x, S=fields["S"]):
            Sr, St = S(x)
            x = np.atleast_1d(np.asarray(x, float))
            return np.asarray(Sr, float), np.asarray(St, float) * (1.0 + 0.5 * (x / Rst) ** 2)
        f["S"] = S2
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="<internal>/code")
    ap.add_argument("--eos", default="<internal>/eos_v4_tables.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--zc", default="0.3,0.8")
    ap.add_argument("--K", type=float, default=0.01, help="the polytrope's constant P = K (mu - m)^2 (lattice units)")
    ap.add_argument("--record", default=None)
    ap.add_argument("--NB", type=float, default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.isdir(a.code) and a.code not in sys.path:
        sys.path.insert(0, a.code)
    import static_rows_v1 as SR
    SR.engine(a.code)
    import engine_ovpg_v1 as OV
    m, g = a.m, a.g
    results = []
    L = ["## static_rows_selftest_v1 (the self-test of static_rows_v1.py; m a %g, g %g)" % (m, g), ""]

    def crit(name, value, thr, lower_is_pass=True, extra=""):
        ok = (value <= thr) if lower_is_pass else (value >= thr)
        results.append((name, value, thr, ok))
        L.append("- %s: %.3e against %.1e: %s%s" % (name, value, thr, "PASS" if ok else "FAIL", extra))

    def note(txt):
        L.append("- " + txt)

    eoss = []
    try:
        eoss.append(("polytrope", eos_polytrope(a.K, m), {"W": 1e-5, "rows": 1e-5, "quad": 1e-5, "cons": 1e-6}))
    except Exception as ex:
        note("the polytrope equation of state could not be built: %s" % ex)
    if os.path.exists(a.eos):
        eoss.append(("table (consistent: P the cubic spline, e = mu dP/dmu - P)", eos_table(a.eos, m), {"W": 1e-3, "rows": 1e-3, "quad": 1e-3, "cons": 1e-6}))
    else:
        note("the table %s is absent: the table equation of state is skipped (pass --eos)" % a.eos)
    zcs = [float(x) for x in a.zc.split(",")]
    for eos_name, (e_of, P_of, n_of), thr in eoss:
        for zc in zcs:
            t0 = time.time()
            try:
                fields, meta = ovpg_pair(OV, m, g, zc, e_of, P_of, n_of)
            except Exception as ex:
                note("%s, z_c %g: the OV shoot failed: %s" % (eos_name, zc, ex)); continue
            Rst = meta["R_star"]; h = Rst / 512.0
            grid = np.arange(0.01 * Rst, 0.95 * Rst + 1e-12, Rst / 512.0)
            band = (grid >= 0.05 * Rst) & (grid <= 0.95 * Rst)
            L.append("")
            L.append("### %s, z_c %g: R_* %.3f, M %.6f, 2gM/R_* %.4f, E_F %.6f (h = R_*/512; the band 0.05 to 0.95 R_*)" % (eos_name, zc, Rst, meta["M"], meta["compactness"], meta["EF"]))
            tag = "%s z_c %g" % (eos_name.split(" ")[0], zc)
            # (A) the exact pair
            d = SR.defects_static(fields, g, grid, h)
            dr = SR.direct_rows(fields, g, grid, h)
            crit("(A) %s: max abs(W_0)/scale on the exact pair" % tag, float(np.max(np.abs(d["W0"][band]) / np.maximum(d["scale0"][band], 1e-300))), thr["W"])
            crit("(A) %s: max abs(W_r)/scale on the exact pair" % tag, float(np.max(np.abs(d["Wr"][band]) / np.maximum(d["scaler"][band], 1e-300))), thr["W"])
            crit("(A) %s: max abs(R_t)/term direct on the exact pair" % tag, float(np.max(dr["rel_t"][band])), thr["rows"])
            crit("(A) %s: max abs(R_r)/term direct on the exact pair" % tag, float(np.max(dr["rel_r"][band])), thr["rows"])
            q = SR.quadrature_rows(d, g)
            note("(A) %s: the quadrature's R_t and R_r on the exact pair, largest over the band relative to the rows' largest terms: %.2e, %.2e (both rows vanish there)." % (
                tag, float(np.max(np.abs(q["Rt_quad"][band]) / np.maximum(dr["terms_tt"][band], 1e-300))), float(np.max(np.abs(q["Rr_quad"][band]) / np.maximum(dr["terms_rr"][band], 1e-300)))))
            # (D) the units: the constraints by the same differences
            A = d["v1"] / d["N"]; B = d["v"] / (d["r"] * d["N"])
            H = B ** 2 + 2.0 * A * B - 8.0 * math.pi * g * d["rho"]
            Cr = B * d["N1"] / d["N"] + 4.0 * math.pi * g * d["J"]
            crit("(D) %s: the Hamiltonian constraint H/(8 pi g rho)" % tag, float(np.max(np.abs(H[band]) / (8.0 * math.pi * g * np.abs(d["rho"][band])))), thr["cons"])
            crit("(D) %s: the momentum constraint C_r/(4 pi g abs J)" % tag, float(np.max(np.abs(Cr[band]) / (4.0 * math.pi * g * np.abs(d["J"][band])))), thr["cons"])
            # (B) the perturbed stresses
            for kind in ("B1", "B2"):
                fp = perturbed(fields, kind, Rst)
                dp = SR.defects_static(fp, g, grid, h); qp = SR.quadrature_rows(dp, g); drp = SR.direct_rows(fp, g, grid, h)
                tt = np.maximum(drp["terms_tt"][band], 1e-300); tr = np.maximum(drp["terms_rr"][band], 1e-300)
                crit("(%s) %s: max abs(R_t quadrature - R_t direct)/term" % (kind, tag), float(np.max(np.abs(qp["Rt_quad"][band] - drp["Rt_dir"][band]) / tt)), thr["quad"],
                     extra=" (the direct row itself up to %.2e of its largest term; the defects up to %.2e, %.2e of their scales)" % (
                         float(np.max(drp["rel_t"][band])), float(np.max(np.abs(dp["W0"][band]) / np.maximum(dp["scale0"][band], 1e-300))),
                         float(np.max(np.abs(dp["Wr"][band]) / np.maximum(dp["scaler"][band], 1e-300)))))
                crit("(%s) %s: max abs(R_r quadrature - R_r direct)/term" % (kind, tag), float(np.max(np.abs(qp["Rr_quad"][band] - drp["Rr_dir"][band]) / tr)), thr["quad"],
                     extra=" (the direct row itself up to %.2e of its largest term)" % float(np.max(drp["rel_r"][band])))
                note("(%s) %s: the spline quadrature against the trapezoid, relative on the band: %.2e." % (kind, tag, float(np.max(np.abs(qp["I_cum"][band] - qp["I_trap"][band])) / max(float(np.max(np.abs(qp["I_cum"][band]))), 1e-300))))
                if kind == "B1":
                    # (C) the regular centre on B1
                    v1 = float(dp["v1"][0])
                    tgt = 4.0 * math.pi * g * dp["W0"] / (3.0 * v1)
                    rows_c = []
                    for frac in (0.04, 0.02):
                        rc_ = frac * Rst
                        Rt_c = float(np.interp(rc_, grid, qp["Rt_quad"])); Rr_c = float(np.interp(rc_, grid, qp["Rr_quad"]))
                        tg_c = float(np.interp(rc_, grid, tgt)); W0c = float(np.interp(rc_, grid, dp["W0"])); Wrc = float(np.interp(rc_, grid, dp["Wr"]))
                        rows_c.append((frac, Rt_c, Rr_c, tg_c, W0c, Wrc))
                    for frac, Rt_c, Rr_c, tg_c, W0c, Wrc in rows_c:
                        note("(C) %s: r = %.2f R_*: R_t quadrature %.6e, R_r quadrature %.6e, 4 pi g W_0/(3 v_1) %.6e, W_r/W_0 %.3e (v_1 %.4e)." % (tag, frac, Rt_c, Rr_c, tg_c, Wrc / W0c if W0c else float("nan"), v1))
                    f4, f2 = rows_c[0], rows_c[1]
                    crit("(C) %s: abs(R_t(0.02 R_*) - 4 pi g W_0/(3 v_1))/abs(target)" % tag, abs(f2[1] - f2[3]) / max(abs(f2[3]), 1e-300), 2e-2)
                    crit("(C) %s: abs(R_r(0.02 R_*) - 4 pi g W_0/(3 v_1))/abs(target)" % tag, abs(f2[2] - f2[3]) / max(abs(f2[3]), 1e-300), 2e-2)
                    # W_r = O(r) with the pair's slope, not a regularity constant (STATIC_rows_v1_selftest_ruling.md): on B1 (both stresses times 1.3;
                    # W_0 = W_r = 0 on the exact pair by (A)) the defects are the stress terms alone, W_0 = 0.3 (S_r v' + 2 S_t v/r) and
                    # W_r = 0.3 (N S_r' + N' S_r + 2 N (S_r - S_t)/r); at the chart's regular centre (w_1^2 = 8 pi g e_c/3, v_1 = -N_0 w_1,
                    # N'/N = 4 pi g (e_c + P_c) r, S_r - S_t = (e_c + P_c) w_1^2 r^2) with the TOV centre P = P_c - (2 pi g/3)(e_c + P_c)(e_c + 3 P_c) r^2:
                    # W_0(0) = 0.9 P_c v_1 and W_r = 2.8 pi g N_0 (e_c + P_c) e_c r + O(r^3), so W_r/W_0 = (28 pi g/9) N_0 (e_c + P_c) e_c r/(P_c v_1) + O(r^3)
                    # and 4 pi g W_0(0)/(3 v_1) = 1.2 pi g P_c (the lead's derivation). e_c = rho(0), P_c = S_t(0), N_0 and v_1 from the exact pair's
                    # fields at the first grid point, extrapolated to the centre by parity (f(0) = f(r_0) - r_0 f'(r_0)/2 for the even f and for v').
                    r0 = float(grid[0])
                    ec = float(d["rho"][0] - 0.5 * r0 * d["rho1"][0]); Pc = float(d["St"][0] - 0.5 * r0 * d["St1"][0])
                    N0 = float(d["N"][0] - 0.5 * r0 * d["N1"][0]); v1c = float(d["v1"][0] - 0.5 * r0 * d["v2"][0])
                    slope = (28.0 * math.pi * g * N0 * (ec + Pc) * ec / (9.0 * Pc * v1c)) if (Pc and v1c) else float("nan")
                    pred2 = slope * f2[0] * Rst; pred4 = slope * f4[0] * Rst
                    note("(C) %s: the leading-order slope of W_r/W_0 on B1, (28 pi g/9) N_0 (e_c + P_c) e_c/(P_c v_1) = %.4e per a (e_c %.4e, P_c %.4e, N_0 %.5f, v_1 %.4e; "
                         "w_1 = sqrt(8 pi g e_c/3) %.4e against -v_1/N_0 %.4e): predicted W_r/W_0 %.4e at 0.02 R_* and %.4e at 0.04 R_* against the computed %.4e and %.4e; "
                         "the centre target 4 pi g W_0(0)/(3 v_1) = 1.2 pi g P_c = %.6e (the rows' shift on B1 at the centre)." % (
                             tag, slope, ec, Pc, N0, v1c, math.sqrt(max(8.0 * math.pi * g * ec / 3.0, 0.0)), (-v1c / N0) if N0 else float("nan"), pred2, pred4,
                             (f2[5] / f2[4]) if f2[4] else float("nan"), (f4[5] / f4[4]) if f4[4] else float("nan"), 1.2 * math.pi * g * Pc))
                    crit("(C) %s: W_r/W_0 at 0.02 R_* against the leading-order slope times r, relative (W_r = O(r) with the pair's coefficient)" % tag,
                         (abs(f2[5] / f2[4] - pred2) / abs(pred2)) if (f2[4] and pred2 and np.isfinite(pred2)) else float("inf"), 2e-2)
                    ratio = (abs(f4[5] / f4[4]) / abs(f2[5] / f2[4])) if (f4[4] and f2[4] and f2[5]) else float("inf")
                    crit("(C) %s: the ratio of abs(W_r/W_0) at 0.04 against 0.02 R_* (O(r): about 2) minus 2, absolute" % tag, abs(ratio - 2.0), 1e-1)
            note("%s: %.0f s." % (tag, time.time() - t0))
    L.append("")
    L.append("- (D) the units: lattice units a = 1; g = G/a^2 dimensionless; m a %g; rho, J, S_r, S_t per site volume a^3 (energies in 1/a); W_0, W_r per a^4 (a density per lattice time); "
             "N, v dimensionless; A, B, k_r, k_t in 1/a; R_t, R_r and the rows' terms in 1/a^2; 4 pi g r^2 (N W_0 + v W_r) in 1/a and its integral dimensionless, divided by N r^2 v (a^2) "
             "gives 1/a^2 as R_t must be; the same g enters the constraints H and C_r checked above and the rows' sources 4 pi g N (...): the normalization of rho against (N, v) "
             "is fixed by H = 0 and of J by C_r = 0 on the exact pair." % m)
    # (E) the lattice side on a member
    if a.record and a.NB is not None:
        L.append("")
        L.append("### (E) the lattice side: %s, N_B %.3f" % (os.path.basename(a.record), a.NB))
        try:
            Jf = json.load(open(a.record))
            mem = [q for q in Jf["members"] if not q.get("failed") and abs(float(q["N_B"]) - a.NB) < 2e-3 * max(1.0, a.NB)][0]
            half = bool((Jf.get("args") or {}).get("half_offset", 0))
            t0 = time.time()
            R, eps, J, SV, checks = SR.rebuild_member(mem, half, a.threads)
            crit("(E) the rebuilt eps against the stored shell totals, relative", checks["eps_rebuilt_vs_stored_rel"], 1e-8)
            crit("(E) the rebuilt J against the stored shell totals, relative", checks["J_rebuilt_vs_stored_rel"], 1e-8)
            crit("(E) the stress route's eps against the sources' eps, relative", checks["eps_stress_route_vs_sources_rel"], 1e-8)
            crit("(E) the momentum-balance identity's largest difference (engine_stress_v1)", checks["stress_identity_max_abs"], 1e-12)
            note("(E) the virial residual %.3e relative (the known value about 0.2 on class-1 members, inputs 06 and 07: printed, not a criterion); %d channels, %d columns." % (
                checks["virial_rel"], checks["channels_occupied"], checks["columns"]))
            ell = float(mem["ell"]); gg = float(mem["g"])
            fields, info, Ns, vs = SR.resolve_fields(eps, J, gg, R, ell, SV)
            if fields is None:
                L.append("- (E) the re-solve failed: %s: FAIL" % info.get("failed")); results.append(("(E) re-solve", 1.0, 0.0, False))
            else:
                crit("(E) the smoothed density's charge against the shell sum, abs(int 4 pi r^2 rho_ell / sum eps - 1)", abs(float(info["charge_conservation_quadrature"])), 1e-6)
                crit("(E) the re-solve's Hamiltonian residual (engine_fields_v14, relative)", float(info["residual_W"]), 1e-7)
                crit("(E) the re-solve's momentum residual (engine_fields_v14, relative)", float(info["residual_invN"]), 1e-7)
                rc = float(info["r_collar"]); grid = np.arange(ell / 16.0, rc - 2.0 * ell + 1e-9, ell / 32.0)
                Sr_l, St_l, trerr = SR.engine().STR.smooth_stress(np.asarray(SV["S_r"], float), np.asarray(SV["S_t"], float), np.asarray(R, float), ell, grid)
                crit("(E) the stress smoothing's trace identity error (engine_stress_v1.smooth_stress)", float(trerr), 1e-10)
                note("(E) E re-solve %.6f against the record's E %.6f (relative %.2e); the dictionary line: %s." % (info["E_collar"], float(mem["E"]), info["E_collar"] / float(mem["E"]) - 1.0, info.get("dictionary")))
                d = SR.defects_static(fields, gg, grid, ell / 32.0); q = SR.quadrature_rows(d, gg); dr = SR.direct_rows(fields, gg, grid, ell / 32.0)
                Rs = float(mem["R_surf"]) if mem.get("R_surf") else float(mem["R_999"])
                band = (grid >= ell) & (grid <= Rs)
                crit("(E) the quadrature against the direct rows on the member, R_t, over the largest term on [ell, R_surf]",
                     float(np.max(np.abs(q["Rt_quad"][band] - dr["Rt_dir"][band]) / np.maximum(dr["terms_tt"][band], 1e-300))), 1e-2)
                crit("(E) the same for R_r", float(np.max(np.abs(q["Rr_quad"][band] - dr["Rr_dir"][band]) / np.maximum(dr["terms_rr"][band], 1e-300))), 1e-2)
                note("(E) on [ell, R_surf] of this member: max abs(R_t)/term %.3e, max abs(R_r)/term %.3e, max abs(W_0)/scale %.3e, max abs(W_r)/scale %.3e (inputs 06 and 07 gave 4.7e-2 to 8.0e-2 for the rows on N_B 112 at ell 8); %.0f s." % (
                    float(np.max(dr["rel_t"][band])), float(np.max(dr["rel_r"][band])), float(np.max(np.abs(d["W0"][band]) / np.maximum(d["scale0"][band], 1e-300))),
                    float(np.max(np.abs(d["Wr"][band]) / np.maximum(d["scaler"][band], 1e-300))), time.time() - t0))
        except Exception as ex:
            import traceback
            L.append("- (E) FAILED with an exception: %s" % traceback.format_exc()); results.append(("(E) exception", 1.0, 0.0, False))
    nfail = sum(1 for r in results if not r[3])
    L += ["", "- %d criteria, %d FAIL; overall %s." % (len(results), nfail, "FAIL" if nfail else "PASS"),
          "- every number above is printed by the script; the thresholds are the lead's (the table's lines carry the looser thresholds of the engine's own equation of state; every FAIL, polytrope or table, sets the exit code)."]
    open(os.path.join(a.out, "static_rows_selftest_v1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
