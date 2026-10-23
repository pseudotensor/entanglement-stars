#!/usr/bin/env python
"""engine_rows_v5.py (builder analyst; engine spec 31.5, the records composer chain on the ladder's members): version 4 with the
runs of engine_collapse_v27, v29, v30 and v31 given engine_fields_v16 (version 4 chose it for v24 to v26 only, so a v27 run fell to
engine_fields_v12); nothing else changes. Version 4: engine_rows_v4.py (builder analyst; version 4 = version 3 with the time differences taken over the nodes the neighbouring records share
(the coarse nodes are i ell from ell, so the common prefix is the same radii; a run whose collar moves outward has more nodes later: the r_s 64
ell 8 control has 48 at T = 0 and 254 at T/rs 1.25, on which version 3 stopped). Nothing else changes. Version 3's docstring: version 3 = version 2 with the run's field module chosen by its engine: engine_fields_v12 for
engine_collapse_v17, v14 for v19, v16 for v24, v25 and v26 (version 2 took v12 for every run that is not v19, which is wrong for the v24 ell 8
control; engine spec 28.6 delivers that control's records). Nothing else changes. Version 2's docstring: version 2 = version 1 with the validation's check (c') added: version 1's (c) left the rr bracket at
5e-4 inside the OV-PG member because the equation-of-state table's cubic interpolants are not thermodynamically consistent (mu dP/dmu
against e + P off by 1e-3 to 8e-2 near mu = m, printed); (c') repeats (c) with P(mu) the table's cubic spline and e = mu dP/dmu - P from
it. Nothing else changes. Version 1's docstring: engine spec 20.1, the rows [R, lead's arithmetic; a red-team target]): the residuals of the two
geometric evolution rows of the flat-slice dictionary (3-metric flat, shift -v, the ADM extrinsic curvature -K for the goal's K, the
matter term -8 pi G N [S_ij - (1/2) gamma_ij (S - rho)]), orthonormal components, k_r = v'/N, k_t = v/(rN), k = k_r + 2 k_t, g = G/a^2:
  res_rr := d_t k_r - [N'' - N (k k_r - 2 k_r^2) - v k_r' - 2 k_r v' + 4 pi g N (rho + S_r - 2 S_t)_ell],
  res_tt := d_t k_t - [N'/r - N (k k_t - 2 k_t^2) - v (k_t' + 2 k_t/r) + 4 pi g N (rho - S_r)_ell].
The builder rederived both from the ADM evolution equation (dt K_ij = -D_i D_j N + N (R_ij - 2 K_ik K^k_j + K K_ij) - 8 pi G N (S_ij -
(1/2) gamma_ij (S - rho)) + L_beta K_ij with K = -k, R_ij = 0, beta = -v): the same two rows. BUILDER (labelled): the spatial derivatives
of the dense lapse and flow by Richardson-combined central differences (h and h/2, h = ell/32 by default) of the solve's dense functions,
not the constraint-differentiated route of 18.1; validated below.
--validate: (a) the Painleve-Gullstrand vacuum (N = 1, v = -sqrt(2M/r), no sources): both brackets vanish; (b) the flat
Friedmann-Lemaitre dust ball in flat slicing (N = 1, v = -h r, rho = 3 h^2/(8 pi g), h(t) = 2/(3 (t_c - t)), no stress): the
brackets equal d_t k_r = d_t k_t = -4 pi g rho, the residuals with d_t k from two times at +-dt (the lead's "N' = 4 pi g rho r" dust ball
is not this solution: there the brackets vanish only for a static ball; the builder's arithmetic, reported to the lead); (c) the continuum
Oppenheimer-Volkoff-PG member (engine_ovpg_v1 with the run's equation of state): the static fluid seen by the normal observers (w = sqrt(2 g
m/r), gamma^2 = 1/F): rho_n = (e + P)/F - P, J_n = (e + P) w/F, S_r = (e + P) w^2/F + P, S_t = P; both brackets vanish (d_t k = 0).
--run dir/tag --handoff npz: per stored record the dense fields re-solved from the stored shell totals by the run's own field module
(engine_fields_v12 for engine_collapse_v17, v14 for v19), the smoothed sources (rho_ell of the solve; S_r,ell and S_t,ell from the handoff's
per-shell stresses of the dust and the pairs, smoothed by engine_stress_v1.smooth_stress at the evaluation radii), the brackets at the
coarse nodes r_i = i ell inside the collar and at R_t, R_sh and r_in; d_t k by the centred difference of successive records (one-sided at
the ends, labelled), the second difference over tau beside it (the stepping floor); per record the largest |res|/max(|terms|) over the
matter's support (rho_ell above 1e-6 of its largest value), the exterior's largest |res| (scaled by the support's largest term), the
three radii's values. Args: --validate --md | --run --handoff --md --out-json [--h-frac 1/32] [--records K]."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def deriv(F, r, h):
    """F, F', F'' at r (arrays) by central differences at h and h/2, Richardson-combined."""
    r = np.asarray(r, float)
    f0 = F(r)
    d1 = lambda hh: (F(r + hh) - F(r - hh)) / (2.0 * hh)
    d2 = lambda hh: (F(r + hh) - 2.0 * f0 + F(r - hh)) / (hh * hh)
    return f0, (4.0 * d1(0.5 * h) - d1(h)) / 3.0, (4.0 * d2(0.5 * h) - d2(h)) / 3.0


def brackets(Nf, vf, rho, Sr, St, g, r, h):
    """the two rows' right sides at r and their terms (docstring); rho, Sr, St arrays at r."""
    r = np.asarray(r, float)
    N, N1, N2 = deriv(Nf, r, h)
    v, v1, v2 = deriv(vf, r, h)
    kr = v1 / N; kt = v / (r * N); k = kr + 2.0 * kt
    kr1 = (v2 * N - v1 * N1) / N ** 2
    kt1 = v1 / (r * N) - v / (r * r * N) - v * N1 / (r * N * N)
    src_rr = 4.0 * math.pi * g * N * (rho + Sr - 2.0 * St)
    src_tt = 4.0 * math.pi * g * N * (rho - Sr)
    t_rr = [N2, -N * (k * kr - 2.0 * kr * kr), -v * kr1, -2.0 * kr * v1, src_rr]
    t_tt = [N1 / r, -N * (k * kt - 2.0 * kt * kt), -v * (kt1 + 2.0 * kt / r), src_tt]
    return {"rhs_rr": sum(t_rr), "rhs_tt": sum(t_tt), "k_r": kr, "k_t": kt, "terms_rr": np.max(np.abs(np.array(t_rr)), axis=0),
            "terms_tt": np.max(np.abs(np.array(t_tt)), axis=0), "N": N, "v": v}


def validate(a):
    L = ["## engine_rows_v1 validation (the brackets of the two evolution rows on known solutions)", ""]
    g = 0.25
    # (a) PG vacuum
    M = 10.0
    r = np.linspace(2.0 * M + 1.0, 200.0, 400)
    one = lambda x: np.ones_like(np.asarray(x, float)); vpg = lambda x: -np.sqrt(2.0 * M / np.asarray(x, float))
    z = np.zeros_like(r)
    b = brackets(one, vpg, z, z, z, g, r, a.h)
    L.append("- (a) Painleve-Gullstrand vacuum (2 g E = %g, r from %g to 200): max |rr bracket|/max term %.2e, max |tt bracket|/max term %.2e." % (
        2 * M, r[0], float(np.max(np.abs(b["rhs_rr"]) / b["terms_rr"])), float(np.max(np.abs(b["rhs_tt"]) / b["terms_tt"]))))
    # (b) flat FLRW dust ball, N = 1, v = -h r
    tc, t1, dt = 100.0, 40.0, 1e-3
    hh = lambda t: 2.0 / (3.0 * (tc - t)); rho_t = lambda t: 3.0 * hh(t) ** 2 / (8.0 * math.pi * g)
    r = np.linspace(1.0, 50.0, 200)
    res = {}
    for t in (t1 - dt, t1, t1 + dt):
        vf = (lambda x, t=t: -hh(t) * np.asarray(x, float))
        res[t] = brackets(one, vf, np.full_like(r, rho_t(t)), np.zeros_like(r), np.zeros_like(r), g, r, a.h)
    dkr = (res[t1 + dt]["k_r"] - res[t1 - dt]["k_r"]) / (2 * dt); dkt = (res[t1 + dt]["k_t"] - res[t1 - dt]["k_t"]) / (2 * dt)
    rr_ = (dkr - res[t1]["rhs_rr"]) / np.maximum(res[t1]["terms_rr"], np.abs(dkr)); rt_ = (dkt - res[t1]["rhs_tt"]) / np.maximum(res[t1]["terms_tt"], np.abs(dkt))
    L.append("- (b) flat dust ball (N = 1, v = -h r, h = %.6f, rho = 3 h^2/(8 pi g) = %.6e, d_t k by +-%g): residual/max term rr %.2e, tt %.2e; d_t k_t %.6e against -4 pi g rho %.6e." % (
        hh(t1), rho_t(t1), dt, float(np.max(np.abs(rr_))), float(np.max(np.abs(rt_))), float(dkt[0]), -4.0 * math.pi * g * rho_t(t1)))
    nprime = lambda x: 1.0 + 2.0 * math.pi * g * rho_t(t1) * np.asarray(x, float) ** 2
    bn = brackets(nprime, lambda x: -hh(t1) * np.asarray(x, float), np.full_like(r, rho_t(t1)), np.zeros_like(r), np.zeros_like(r), g, r, a.h)
    L.append("  the lead's variant (N = 1 + 2 pi g rho r^2, so N' = 4 pi g rho r, the same v and rho): tt bracket at r = %g: %.4e (the static value 0 needs 8 pi g rho N = 3 h^2 N: %.4e against %.4e); "
             "the brackets vanish only when the ball does not evolve, so this check is (b)'s geodesic slicing above." % (
                 r[10], float(bn["rhs_tt"][10]), 8 * math.pi * g * rho_t(t1), 3 * hh(t1) ** 2))
    # (c) OV-PG member
    import engine_ovpg_v1 as OV
    e_of, P_of, n_of = OV.load_eos(a.eos, a.m)
    gg = a.g
    resv = OV.shoot(a.m, a.m, gg, e_of, P_of, n_of, a.m * math.sqrt(1 + a.zc * a.zc))
    Rst = resv["R_star"]; Mst = resv["M"]; c = resv["c"]; EF = resv["EF_consistent"]; sol = resv["sol"]

    def mm_phi(x):
        x = np.atleast_1d(np.asarray(x, float))
        inside = x <= Rst
        y = sol.sol(np.clip(x, 1e-6, Rst))
        mm = np.where(inside, y[0], Mst)
        Phi = np.where(inside, y[1] + c, 0.5 * np.log(1 - 2 * gg * Mst / x))
        return mm, Phi

    def Nf(x):
        mm, Phi = mm_phi(x)
        return np.exp(Phi) / np.sqrt(1 - 2 * gg * mm / np.asarray(x, float))

    def vf(x):
        mm, Phi = mm_phi(x)
        return -Nf(x) * np.sqrt(2 * gg * mm / np.asarray(x, float))

    def fluid(x):
        mm, Phi = mm_phi(x)
        F = 1 - 2 * gg * mm / x
        mu = EF / np.exp(Phi)
        e = np.array([float(e_of(u)) if u > a.m else 0.0 for u in mu]); P = np.array([float(P_of(u)) if u > a.m else 0.0 for u in mu])
        w2 = 2 * gg * mm / x
        return (e + P) / F - P, (e + P) * w2 / F + P, P
    hstar = a.h * Rst / 64.0
    mus = np.linspace(a.m * 1.006, EF / math.exp(float(sol.sol(1e-6)[1]) + c), 6)
    cons = []
    for mu in mus:
        dmu = 1e-6 * mu
        dP = (float(P_of(mu + dmu)) - float(P_of(mu - dmu))) / (2 * dmu)
        cons.append("%.4f: %.2e" % (mu / a.m, mu * dP / (float(e_of(mu)) + float(P_of(mu))) - 1.0))
    L.append("- the table's interpolants' consistency mu dP/dmu/(e + P) - 1 at mu/m = " + ", ".join(cons) + ".")
    for lab, r in (("interior r from 0.05 R_* to 0.95 R_*", np.linspace(0.05 * Rst, 0.95 * Rst, 200)), ("exterior r from 1.05 R_* to 3 R_*", np.linspace(1.05 * Rst, 3 * Rst, 200))):
        rho, Sr, St = fluid(r)
        b = brackets(Nf, vf, rho, Sr, St, gg, r, hstar)
        L.append("- (c) OV-PG member (m a %g, g %g, z_c %g: R_* %.3f, M %.6f, 2gM/R_* %.4f), %s: max |rr bracket|/max term %.2e, max |tt bracket|/max term %.2e (h = %.4f)." % (
            a.m, gg, a.zc, Rst, Mst, 2 * gg * Mst / Rst, lab, float(np.max(np.abs(b["rhs_rr"]) / b["terms_rr"])), float(np.max(np.abs(b["rhs_tt"]) / b["terms_tt"])), hstar))
    # (c') the thermodynamically consistent equation of state from the table's P(mu): e = mu dP/dmu - P
    from scipy.interpolate import CubicSpline
    t = np.load(a.eos)
    sel = np.abs(t["m_hat"] - a.m) < 1e-12
    mu_t, P_t = t["mu_hat"][sel], t["P_hat"][sel]
    o = np.argsort(mu_t); mu_t, P_t = mu_t[o], P_t[o]
    Ps = CubicSpline(mu_t, P_t); dPs = Ps.derivative()
    P2 = lambda mu: float(Ps(mu)) if mu > a.m else 0.0
    n2 = lambda mu: float(dPs(mu)) if mu > a.m else 0.0
    e2 = lambda mu: (mu * float(dPs(mu)) - float(Ps(mu))) if mu > a.m else 0.0
    res2 = OV.shoot(a.m, a.m, gg, e2, P2, n2, a.m * math.sqrt(1 + a.zc * a.zc))
    Rst, Mst, c, EF, sol = res2["R_star"], res2["M"], res2["c"], res2["EF_consistent"], res2["sol"]
    e_of, P_of = e2, P2
    for lab, r in (("interior r from 0.05 R_* to 0.95 R_*", np.linspace(0.05 * Rst, 0.95 * Rst, 200)), ("exterior r from 1.05 R_* to 3 R_*", np.linspace(1.05 * Rst, 3 * Rst, 200))):
        rho, Sr, St = fluid(r)
        b = brackets(Nf, vf, rho, Sr, St, gg, r, hstar)
        L.append("- (c') the same member with the consistent equation of state (P the table's cubic spline, e = mu dP/dmu - P; R_* %.3f, M %.6f), %s: max |rr bracket|/max term %.2e, max |tt bracket|/max term %.2e." % (
            Rst, Mst, lab, float(np.max(np.abs(b["rhs_rr"]) / b["terms_rr"])), float(np.max(np.abs(b["rhs_tt"]) / b["terms_tt"]))))
    L += ["", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def shell_stresses(H, k, part, R):
    F = np.asarray(H["F_face_%s" % part][k], float); fc = np.asarray(H["fc_%s" % part][k], float)
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    Srf = F / (4.0 * math.pi * Rf ** 2)
    Sr = 0.5 * (np.concatenate([[Srf[0]], Srf[:-1]]) + Srf)
    St = fc / (8.0 * math.pi * R)
    return Sr, St


def records(a):
    import engine_stress_v1 as STR
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(a.handoff, allow_pickle=True)
    script = J.get("script", "")
    if "v19" in script:
        import engine_fields_v14 as FM
    elif any(x in script for x in ("v24", "v25", "v26", "v27", "v29", "v30", "v31")):
        import engine_fields_v16 as FM
    else:
        import engine_fields_v12 as FM
    ra = J["args"]
    rs = float(ra["rs"]); g = float(ra["g"]); ell = float(ra["ell"]); tau = float(ra["tau"])
    R = np.asarray(Z["R"], float)
    Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    K = len(TH) if a.records is None else min(len(TH), a.records)
    h = ell * a.h_frac
    per = []
    for k in range(K):
        T = float(TH[k]); kz = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[kz] - T) > 1e-9:
            continue
        rec = recs.get(round(T, 6), {})
        e = np.asarray(Z["eps_tot"][kz], float); Jt = np.asarray(Z["J_tot"][kz], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
        if info.get("failed"):
            per.append({"T": T, "failed": info["failed"]}); continue
        rc = float(info["r_collar"])
        nodes = np.arange(ell, rc - 2 * ell + 1e-9, ell)
        extra = [x for x in (rec.get("R_t"), rec.get("R_sh"), rec.get("r_in")) if x]
        rr = np.concatenate([nodes, np.asarray(extra, float)])
        Srd, Std = shell_stresses(H, k, "dust", R); Srp, Stp = shell_stresses(H, k, "pairs", R)
        Sr_l, St_l, tr_err = STR.smooth_stress(Srd + Srp, Std + Stp, R, ell, rr)
        rho = np.asarray(info["rho_ell"](rr), float)
        b = brackets(info["N_dense"], info["v_dense"], rho, Sr_l, St_l, g, rr, h)
        per.append({"T": T, "r": rr, "n_nodes": len(nodes), "extra": extra, "b": b, "rho": rho, "trace_err": tr_err})
        if k % 10 == 0:
            print("record T/rs %.3f: brackets at %d radii" % (T / rs, len(rr)), flush=True)
    ok = [p for p in per if "b" in p]
    L = ["## The evolution rows' residuals of %s (engine_rows_v5.py; the stresses from %s; fields %s)" % (tag, os.path.basename(a.handoff), FM.__name__), "",
         "d_t k by the centred difference of successive records (one-sided at the ends); res/max term over the matter's support (rho_ell above 1e-6 of its largest value) and in the exterior; the stepping floor: the second difference of k over tau, largest over the support.", "",
         "| T/rs | d_t | res_rr support | res_tt support | res_rr exterior | res_tt exterior | at R_t (rr, tt) | at R_sh (rr, tt) | at r_in (rr, tt) | stepping floor (rr, tt) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for i, p in enumerate(ok):
        n_full = p["n_nodes"]
        if 0 < i < len(ok) - 1 and abs((ok[i + 1]["T"] - ok[i - 1]["T"]) - 2 * tau) < 1e-6:
            p = dict(p, n_nodes=min(p["n_nodes"], ok[i - 1]["n_nodes"], ok[i + 1]["n_nodes"]))
            dkr = (ok[i + 1]["b"]["k_r"][:p["n_nodes"]] - ok[i - 1]["b"]["k_r"][:p["n_nodes"]]) / (2 * tau)
            dkt = (ok[i + 1]["b"]["k_t"][:p["n_nodes"]] - ok[i - 1]["b"]["k_t"][:p["n_nodes"]]) / (2 * tau)
            sfr = (ok[i + 1]["b"]["k_r"][:p["n_nodes"]] - 2 * p["b"]["k_r"][:p["n_nodes"]] + ok[i - 1]["b"]["k_r"][:p["n_nodes"]]) / tau
            sft = (ok[i + 1]["b"]["k_t"][:p["n_nodes"]] - 2 * p["b"]["k_t"][:p["n_nodes"]] + ok[i - 1]["b"]["k_t"][:p["n_nodes"]]) / tau
            how = "centred"
        else:
            j = i + 1 if i + 1 < len(ok) else i - 1
            if j < 0 or j >= len(ok):
                continue
            p = dict(p, n_nodes=min(p["n_nodes"], ok[j]["n_nodes"]))
            sgn = 1.0 if j > i else -1.0
            dkr = sgn * (ok[j]["b"]["k_r"][:p["n_nodes"]] - p["b"]["k_r"][:p["n_nodes"]]) / tau
            dkt = sgn * (ok[j]["b"]["k_t"][:p["n_nodes"]] - p["b"]["k_t"][:p["n_nodes"]]) / tau
            sfr = sft = None
            how = "one-sided"
        n = p["n_nodes"]
        b = p["b"]
        resr = dkr - b["rhs_rr"][:n]; rest = dkt - b["rhs_tt"][:n]
        tr = np.maximum(b["terms_rr"][:n], np.abs(dkr)); tt = np.maximum(b["terms_tt"][:n], np.abs(dkt))
        rho = p["rho"][:n]
        sup = np.abs(rho) > 1e-6 * max(float(np.max(np.abs(rho))), 1e-300)
        ext = ~sup & (p["r"][:n] > (p["r"][:n][sup].max() if sup.any() else 0.0))
        scr = float(np.max(tr[sup])) if sup.any() else 1.0; sct = float(np.max(tt[sup])) if sup.any() else 1.0
        f_ = lambda x: ("%.2e" % x) if x is not None else "-"
        row = {"T_over_rs": p["T"] / rs, "dt": how, "res_rr_support": float(np.max(np.abs(resr[sup]) / tr[sup])) if sup.any() else None,
               "res_tt_support": float(np.max(np.abs(rest[sup]) / tt[sup])) if sup.any() else None,
               "res_rr_exterior": float(np.max(np.abs(resr[ext]))) / scr if ext.any() else None, "res_tt_exterior": float(np.max(np.abs(rest[ext]))) / sct if ext.any() else None,
               "stepping_floor": [float(np.max(np.abs(sfr[sup]))) / scr if (sfr is not None and sup.any()) else None,
                                  float(np.max(np.abs(sft[sup]))) / sct if (sft is not None and sup.any()) else None],
               "trace_identity_rel": p["trace_err"]}
        # the three radii: d_t k there interpolated linearly from the nodes' values
        radii = {}
        for jx, x in enumerate(p["extra"]):
            idx = n_full + jx
            dr_ = float(np.interp(x, p["r"][:n], dkr)); dt_ = float(np.interp(x, p["r"][:n], dkt))
            radii[jx] = ((dr_ - float(b["rhs_rr"][idx])) / max(float(b["terms_rr"][idx]), abs(dr_), 1e-300), (dt_ - float(b["rhs_tt"][idx])) / max(float(b["terms_tt"][idx]), abs(dt_), 1e-300))
        rec = recs.get(round(p["T"], 6), {})
        names = [lab for lab in ("R_t", "R_sh", "r_in") if rec.get(lab)]
        rv = {names[j]: radii[j] for j in range(len(names)) if j in radii}
        row["radii"] = rv
        out.append(row)
        g2 = lambda lab: ("%.2e, %.2e" % rv[lab]) if lab in rv else "-"
        L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row["T_over_rs"], how, f_(row["res_rr_support"]), f_(row["res_tt_support"]), f_(row["res_rr_exterior"]),
                                                                        f_(row["res_tt_exterior"]), g2("R_t"), g2("R_sh"), g2("r_in"),
                                                                        ("%s, %s" % (f_(row["stepping_floor"][0]), f_(row["stepping_floor"][1])))))
    L += ["", "- every number above is printed by the script; the derivatives by Richardson-combined central differences of the dense solve (h = %g ell)." % a.h_frac]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_rows_v5.py", "fields": FM.__name__, "run": a.run, "handoff": a.handoff, "rows": out}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[-8:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--md", required=True)
    ap.add_argument("--h", type=float, default=0.125)
    ap.add_argument("--eos", default="EOS/eos_v4_tables.npz")
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--zc", type=float, default=0.3)
    ap.add_argument("--run", default=None)
    ap.add_argument("--handoff", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--h-frac", type=float, default=1.0 / 32.0)
    ap.add_argument("--records", type=int, default=None)
    a = ap.parse_args()
    if a.validate:
        validate(a)
    else:
        records(a)


if __name__ == "__main__":
    main()
