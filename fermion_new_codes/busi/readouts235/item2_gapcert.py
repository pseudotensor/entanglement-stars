#!/usr/bin/env python
"""item2_gapcert.py (busi readouts #235, item 2): the frozen-symbol gap certificate of Lemma B16 / Eq. (B-gapcert) of
REVIEWA_deconditional_v1.md on the stored back-averaged fields of the sigma/a 8 member of the (ma 0.2, r_s/a 64, tau 8) fixed-preparation
averaging ladder (collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg, engine_collapse_v27), at every stored record with 12.75 <= T/r_s <= 16.0.

Symbol (the note's Lemma B16): A(x,k) = n(x) h0(k) + w(x).q(k), eigenvalues w.q +- n E(k), E(k) = sqrt(mu^2 + sum_i sin^2 k_i),
q_i(k) = sin(k_i) cos(k_i) = sin(2 k_i)/2, mu = m a = 0.2; n = N, w = v(r) xhat on the 3D lattice.
g(x) = n(x) min_k [E(k) - (|v|/N) |xhat.q(k)|] on the 64^3 grid k_j = 2 pi j / 64.
Primary reading: the infimum over all site directions xhat (the 3D lattice's sites at radius r >> 1 cover the sphere), which equals
min_k [E - u |q|]; the axis directions [100], [110], [111] are printed beside. The engine's own radial chain symbol per channel kappa
(sqrt(m^2 + kappa^2/R^2 + sin^2 k) against |v| |sin 2k|/2, kappa = 1 the smallest mass) is printed beside as a control.
Lipschitz constant C1eps = max over lattice bonds of |f(x + e_i) - f(x)| for f in {N, w_1, w_2, w_3}: the radial chain's bond differences of
N and v, and the transverse change of w = v xhat, |v|/r (a unit step perpendicular to xhat turns xhat by about 1/r).
delta = [g*(g* - c3 eps^1/2)]_+^1/2 - 3 C1eps with c1 = (sqrt3 + 1)(Omega + 3/2) C1, c2 = (3 sqrt3 pi/2)(nbar + wbar),
c3 = 4 c1 + 2 sqrt27 c2, Omega = sqrt(mu^2 + 3); hypothesis eps <= min{1/16, (g*/2c1)^2}.
Reading A: C1 = 1, eps = C1eps. Reading B: the split (C1, eps) with C1 eps fixed chosen to maximize delta inside the hypothesis' range.
Usage: python item2_gapcert.py <fields.npz> <run.json> <out_prefix>
"""
import json
import math
import sys

import numpy as np

MU = 0.2
T0, T1 = 12.75, 16.0
NK = 64


def symbol_tables():
    k = 2 * np.pi * np.arange(NK) / NK
    s, c = np.sin(k), np.cos(k)
    S1, S2, S3 = np.meshgrid(s * s, s * s, s * s, indexing="ij")
    E = np.sqrt(MU ** 2 + S1 + S2 + S3)
    q1d = s * c
    Q1, Q2, Q3 = np.meshgrid(q1d, q1d, q1d, indexing="ij")
    tabs = {}
    qabs = np.sqrt(Q1 ** 2 + Q2 ** 2 + Q3 ** 2)
    tabs["all"] = (E, qabs)
    for name, d in (("100", (1, 0, 0)), ("110", (1, 1, 0)), ("111", (1, 1, 1))):
        d = np.array(d, float) / np.linalg.norm(d)
        tabs[name] = (E, np.abs(d[0] * Q1 + d[1] * Q2 + d[2] * Q3))
    out = {}
    for name, (e, q) in tabs.items():
        P = np.unique(np.round(np.stack([e.ravel(), q.ravel()], 1), 14), axis=0)
        out[name] = P
    return out


def G_of_u(P, u):
    """min over the unique (E, |q|) pairs of E - u |q|, vectorized over u."""
    u = np.atleast_1d(u)
    res = np.empty_like(u)
    for i0 in range(0, len(u), 4096):
        uu = u[i0:i0 + 4096]
        res[i0:i0 + 4096] = (P[None, :, 0] - uu[:, None] * P[None, :, 1]).min(axis=1)
    return res


def G_continuum(u):
    """the continuum infimum over k and all directions: min over S in [0, 3] of sqrt(mu^2 + S) - u sqrt(S - S^2/3) (Corollary B5's reduction)."""
    S = np.linspace(0.0, 3.0, 30001)
    a = np.sqrt(MU ** 2 + S); b = np.sqrt(np.maximum(S - S * S / 3, 0.0))
    u = np.atleast_1d(u); res = np.empty_like(u)
    for i0 in range(0, len(u), 64):
        res[i0:i0 + 64] = (a[None, :] - u[i0:i0 + 64, None] * b[None, :]).min(axis=1)
    return res


def chain_gap(N, v, R, kappa=1.0):
    k = 2 * np.pi * np.arange(NK) / NK
    mc2 = MU ** 2 + (kappa / R) ** 2
    e = np.sqrt(mc2[:, None] + np.sin(k)[None, :] ** 2)
    q = np.abs(np.sin(2 * k))[None, :] / 2
    u = np.abs(v) / N
    return N * (e - u[:, None] * q).min(axis=1)


def certificate(gs, L, nbar, wbar):
    Om = math.sqrt(MU ** 2 + 3)
    K = (math.sqrt(3) + 1) * (Om + 1.5)
    c2 = 1.5 * math.sqrt(3) * math.pi * (nbar + wbar)
    out = {"Omega": Om, "K": K, "c2": c2}
    # reading A: C1 = 1, eps = L
    eps = L; c1 = K; c3 = 4 * c1 + 2 * math.sqrt(27) * c2
    hyp = eps <= min(1 / 16, (gs / (2 * c1)) ** 2) if gs > 0 else False
    term = c3 * math.sqrt(eps)
    dA = math.sqrt(max(0.0, gs * (gs - term))) - 3 * L if gs > 0 else float("nan")
    out.update(c1_A=c1, c3_A=c3, c3sqrteps_A=term, ratio_A=term / gs if gs > 0 else float("inf"), hyp_A=bool(hyp), delta_A=dA,
               eps_needed_A=(gs / c3) ** 2 if gs > 0 else float("nan"))
    # reading B: eps free, C1 = L/eps; sqrt(eps) = s in [2KL/gs, 1/4]; c3 sqrt(eps) = 4KL/s + 2 sqrt27 c2 s
    if gs > 0:
        lo, hi = 2 * K * L / gs, 0.25
        a, b = 4 * K * L, 2 * math.sqrt(27) * c2
        sopt = math.sqrt(a / b)
        if lo <= hi:
            s = min(max(sopt, lo), hi)
            termB = a / s + b * s
            dB = math.sqrt(max(0.0, gs * (gs - termB))) - 3 * L
            out.update(hyp_B=True, sqrt_eps_B=s, c3sqrteps_B=termB, delta_B=dB)
        else:
            termB = a / sopt + b * sopt
            out.update(hyp_B=False, sqrt_eps_B=float("nan"), c3sqrteps_B=termB, delta_B=float("nan"))
        out["c3sqrteps_min_unconstrained"] = 2 * math.sqrt(a * b)
    return out


def main():
    fnpz, fjson, outp = sys.argv[1:4]
    z = np.load(fnpz, allow_pickle=True)
    R = z["R"]; T = z["T"]; rs = 64.0
    Trs = T / rs
    j = json.load(open(fjson))
    recs = {round(r["T_over_rs"], 6): r for r in j["records"]}
    tabs = symbol_tables()
    # known-answer controls on the symbol grid
    g1 = G_of_u(tabs["all"], np.array([1.0]))[0]
    print("CONTROL 64^3 grid: min_k[E - |q|] at u = 1: %.5f (Corollary B5's gamma(0.2) = 0.0577 in the continuum)" % g1)
    print("CONTROL [100] at u = 1: %.5f (the 1D frozen gap of CHECK 11, 0.07416 on a fine grid)" % G_of_u(tabs["100"], np.array([1.0]))[0])
    ucs = {}
    for name in tabs:
        uu = np.linspace(0.5, 3.0, 25001); gg = G_of_u(tabs[name], uu)
        ucs[name] = float(uu[np.argmax(gg < 0)]) if (gg < 0).any() else float("inf")
    print("CONTROL pocket thresholds u_c on the grid (first u with G < 0):", {k: round(v, 4) for k, v in ucs.items()},
          "; w_c^(3)(0.2) = %.5f" % (math.sqrt(1 + MU ** 2 / 3) + MU / math.sqrt(3)))
    UT = np.linspace(0.0, 1.3, 13001); GT = G_continuum(UT)
    print("CONTROL continuum G(1) = %.5f (gamma(0.2) = 0.05770), continuum u_c = %.5f" % (np.interp(1.0, UT, GT), UT[np.argmax(GT < 0)] if (GT < 0).any() else float("inf")))
    sel = np.nonzero((Trs >= T0 - 1e-9) & (Trs <= T1 + 1e-9))[0]
    rows = []
    for i in sel:
        rowr = {"T_over_rs": float(Trs[i])}
        for fset in ("applied", "solved"):
            N = z["N_applied"][i] if fset == "applied" else z["N"][i]
            v = z["v_applied"][i] if fset == "applied" else z["v"][i]
            u = np.abs(v) / N
            gx = {name: N * G_of_u(tabs[name], u) for name in tabs}
            g_all = gx["all"]; im = int(np.argmin(g_all))
            gco = N * np.interp(u, UT, GT)
            gch = chain_gap(N, v, R)
            LN = float(np.abs(np.diff(N)).max()); Lv = float(np.abs(np.diff(v)).max()); Lt = float((np.abs(v) / R).max())
            L = max(LN, Lv, Lt)
            nbar, nmin, wbar = float(N.max()), float(N.min()), float(np.abs(v).max())
            gs = float(g_all.min())
            d = {"g_star": gs, "r_at_g_star": float(R[im]), "u_at_g_star": float(u[im]), "N_at_g_star": float(N[im]),
                 "g_star_100": float(gx["100"].min()), "g_star_110": float(gx["110"].min()), "g_star_111": float(gx["111"].min()),
                 "g_star_cont": float(gco.min()), "r_g_star_cont": float(R[int(np.argmin(gco))]), "g_star_chain_k1": float(gch.min()), "n_sites_g_neg": int((g_all < 0).sum()), "n_sites_chain_neg": int((gch < 0).sum()),
                 "n_sites_cont_neg": int((gco < 0).sum()), "max_u": float(u.max()), "r_max_u": float(R[int(np.argmax(u))]),
                 "L_N": LN, "L_v_radial": Lv, "L_v_transverse": Lt, "C1eps": L, "r_L": None,
                 "nbar": nbar, "n_star": nmin, "wbar": wbar, "pocket": bool(gs <= 0)}
            for nm, arr in (("L_N", np.abs(np.diff(N))), ("L_v_radial", np.abs(np.diff(v))), ("L_v_transverse", np.abs(v) / R)):
                if abs(d[nm] - L) == 0: d["r_L"] = float(R[int(np.argmax(arr))]); d["L_from"] = nm
            d.update(certificate(min(gs, MU), L, nbar, wbar) if gs > 0 else {})
            rowr[fset] = d
        rec = recs.get(round(float(Trs[i]), 6), {})
        rowr["stored_n_pocket"] = rec.get("n_pocket"); rowr["stored_threshold_radius"] = rec.get("threshold_radius")
        rowr["stored_min_N_minus_absv"] = rec.get("min_N_minus_absv"); rowr["stored_R_t"] = rec.get("R_t")
        rows.append(rowr)
    json.dump({"controls": {"G_all_u1": g1, "u_c": ucs}, "rows": rows}, open(outp + ".json", "w"), indent=1, default=str)
    hdr = ("T/rs   g*        r(g*)  u(g*)  N(g*)  g*[100]  g*[111]  g*cont   g*chain  C1eps     from          nbar   n*     wbar   "
           "c3*sqrt(eps)/g*(A)  delta_A     c3sqrt(eps)min(B)/g*  delta_B   hypB  stored_npocket  R_t")
    lines = [hdr]
    for r in rows:
        a = r["applied"]
        if a["pocket"]:
            lines.append("%6.3f  pocket (g* = %.4g at r %.0f, u %.3f; %d sites with g < 0)" % (r["T_over_rs"], a["g_star"], a["r_at_g_star"], a["u_at_g_star"], a["n_sites_g_neg"]))
            continue
        lines.append("%6.3f  %.3e  %5.0f  %.3f  %.3f  %.3e  %.3e  %.3e  %.3e  %.3e  %-13s %.3f  %.3f  %.3f  %.3e           %+.3e  %.3e             %s  %s  %s  %s" % (
            r["T_over_rs"], a["g_star"], a["r_at_g_star"], a["u_at_g_star"], a["N_at_g_star"], a["g_star_100"], a["g_star_111"],
            a["g_star_cont"], a["g_star_chain_k1"], a["C1eps"], "%s@%s" % (a.get("L_from"), a.get("r_L")), a["nbar"], a["n_star"], a["wbar"], a["ratio_A"],
            a["delta_A"], a["c3sqrteps_min_unconstrained"] / a["g_star"], ("%+.3e" % a["delta_B"]) if a["hyp_B"] else "n/a", a["hyp_B"],
            (sum(r["stored_n_pocket"].values()) if isinstance(r["stored_n_pocket"], dict) else r["stored_n_pocket"]), r["stored_R_t"]))
    lines.append("")
    lines.append("solved-field (N, v) against applied-field g*: max |difference| %.3e; max relative C1eps difference %.3e" % (
        max(abs(r["applied"]["g_star"] - r["solved"]["g_star"]) for r in rows),
        max(abs(r["applied"]["C1eps"] - r["solved"]["C1eps"]) / r["applied"]["C1eps"] for r in rows)))
    open(outp + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
