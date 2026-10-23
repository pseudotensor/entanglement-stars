#!/usr/bin/env python
"""cv_fields.py (busi lane, correctness reading of CURVATURE_readout_v1): own recomputation, per member, of
  (a) the spheres' curvature K_t^2 = 2GM/r^3 = (v/(rN))^2 (engine_fields_v12.misner_sharp: M = r v^2/(2 g N^2)) on a 0.125a grid to 256a,
      at the nodes (exact), on the barred site fields, the exact centre limit (v'(0)/N(0))^2 of the spline, and reviewer A's centre identity
      (8/3) pi g rho_ell(0) from the engine's kernel formula applied to the stored eps_tot; the cap (8/3) pi g rho_bar and the gate scale;
  (b) the gates (barred and unbarred one-term test sigma max|d_r v| > 0.1; strain form), formation record, trapping-zero episodes, and the
      trapping zeros of my spline against the stored ones;
  (c) the slice-frame Riemann norm from the unbarred spline fields with time differences at cadence 1, 2, 4; the Kretschmann scalar; the
      components at the maximum; the stencil variants (spline at nodes, FD2/FD4 at nodes, barred sites FD2);
  (d) the source route at the nodes sigma..160a (Riemann from rho_ell, J_ell, S_r_ell, S_t_ell by the Einstein identities) against the
      (N, v) route: constraint-row residuals and the evolution residual;
  (e) housekeeping claims: interp_floor, N vs N_applied, converged, gauge labels, the r^-3 tail.
Unbarred fields: the engine's quintic parity spline through the stored nodes (its own definition). Read-only on every input.
Usage: python cv_fields.py --member P64_s4_c19 --out DIR"""
import argparse
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_common as CC  # noqa: E402

H_GRID = 0.125
R_MAX = 256.0
MARKS = [0, 5, 10, 11, 11.5, 11.75, 12, 12.125, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 14, 15, 17, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100]


def tderiv(F, T, c):
    n = F.shape[0]
    D = np.full(F.shape, np.nan)
    for k in range(n):
        lo, hi = k - c, k + c
        if lo >= 0 and hi < n:
            D[k] = (F[hi] - F[lo]) / (T[hi] - T[lo])
        elif hi < n:
            D[k] = (F[hi] - F[k]) / (T[hi] - T[k])
        elif lo >= 0:
            D[k] = (F[k] - F[lo]) / (T[k] - T[lo])
    return D


def riemann(r, N, Nr, Nrr, v, vr, vrr, Nt, vt, vtr):
    """the five independent orthonormal components (own derivation, cv_riemann_sympy.py), the frame norm and the Kretschmann scalar."""
    R0101 = (N * Nrr - v * vrr - vr ** 2 - vtr) / N ** 2 + (Nr * v * vr + Nt * vr) / N ** 3
    R0202 = (N ** 2 * Nr - N * v * vr - N * vt + Nr * v ** 2 + Nt * v) / (N ** 3 * r)
    R0212 = Nr * v / (N ** 2 * r)
    R1212 = v * vr / (N ** 2 * r)
    R2323 = v ** 2 / (N ** 2 * r ** 2)
    base = 4.0 * (R0101 ** 2 + 2.0 * R0202 ** 2 + 2.0 * R1212 ** 2 + R2323 ** 2)
    return dict(R0101=R0101, R0202=R0202, R0212=R0212, R1212=R1212, R2323=R2323, norm=np.sqrt(base + 16.0 * R0212 ** 2), kret=base - 16.0 * R0212 ** 2)


def fd_parity(x, y, parity, order):
    """central differences (order 2 or 4) on a uniform grid x = h, 2h, ... (x[0] = 0 allowed) with the parity extension through the centre;
    NaN at the outer end where the stencil runs out."""
    h = x[1] - x[0]
    if x[0] == 0.0:
        ye = np.concatenate([parity * y[2:0:-1], y, [np.nan, np.nan]])          # ghosts y(-h), y(-2h) from y(h), y(2h)
    else:
        # sites h, 2h, ...: y(0) = 0 for odd, y(0) ~ y(h) + O(h^2) for even (only the first point uses it); y(-h) by parity
        ye = np.concatenate([[parity * y[0], 0.0 if parity < 0 else y[0]], y, [np.nan, np.nan]])
    i = np.arange(len(y)) + 2
    if order == 2:
        return (ye[i + 1] - ye[i - 1]) / (2 * h), (ye[i + 1] - 2 * ye[i] + ye[i - 1]) / h ** 2
    return ((-ye[i + 2] + 8 * ye[i + 1] - 8 * ye[i - 1] + ye[i - 2]) / (12 * h),
            (-ye[i + 2] + 16 * ye[i + 1] - 30 * ye[i] + 16 * ye[i - 1] - ye[i - 2]) / (12 * h ** 2))


def my_zeros(sN, sv, ell, rmax):
    rr = np.arange(0.0, rmax + 1e-9, ell / 64.0)
    f = sN(rr) - np.abs(sv(rr))
    fn = lambda x: float(sN(x) - abs(sv(x)))
    z = []
    for i in range(len(rr) - 1):
        if f[i] == 0.0:
            z.append(float(rr[i]))
        elif f[i] * f[i + 1] < 0.0:
            z.append(float(brentq(fn, rr[i], rr[i + 1], xtol=1e-12)))
    return sorted(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    M = CC.load(a.member)
    name, T, Trs, rs, g, ell, mA = M["name"], M["T"], M["Trs"], M["rs"], M["g"], M["ell"], M["m"]
    nrec = len(T); rn = M["rn"]; Nn = M["Nn"]; vn = M["vn"]; R = M["R"]
    rho_bar = 4.0 * np.sqrt(mA ** 2 + 1.5)
    cap = 8.0 / 3.0 * np.pi * g * rho_bar
    gate_scale = (CC.GATE / ell) ** 2
    rg = np.arange(H_GRID, R_MAX + 1e-9, H_GRID)
    seln = (rn > 0) & (rn <= R_MAX); rnc = rn[seln]
    F = {k: np.zeros((nrec, len(rg))) for k in ("N", "Nr", "Nrr", "v", "vr", "vrr")}
    Fn = {k: np.zeros((nrec, len(rnc))) for k in ("N", "Nr", "Nrr", "v", "vr", "vrr")}
    N0 = np.zeros(nrec); v10 = np.zeros(nrec); v30 = np.zeros(nrec); N20 = np.zeros(nrec)
    zero_diff = np.full(nrec, np.nan); zero_count_mismatch = np.zeros(nrec, int); my_zero_lists = []
    rz = min(float(rn[-1]), 512.0)
    for k in range(nrec):
        sN = CC.parity_spline(rn, Nn[k], +1); sv = CC.parity_spline(rn, vn[k], -1)
        for key, sp_, d in (("N", sN, 0), ("Nr", sN, 1), ("Nrr", sN, 2), ("v", sv, 0), ("vr", sv, 1), ("vrr", sv, 2)):
            F[key][k] = sp_(rg, d); Fn[key][k] = sp_(rnc, d)
        N0[k] = sN(0.0); N20[k] = sN(0.0, 2); v10[k] = sv(0.0, 1); v30[k] = sv(0.0, 3)
        zk = my_zeros(sN, sv, ell, rz); my_zero_lists.append(zk)
        st = M["zeros"][k]
        if len(st) or len(zk):
            if len(st) == len(zk):
                zero_diff[k] = max(abs(x - y) for x, y in zip(sorted(st), zk)) if len(st) else 0.0
            else:
                zero_count_mismatch[k] = 1
    # (a) spheres' curvature
    Kt2 = (F["v"] / (rg * F["N"])) ** 2
    j = np.argmax(Kt2, axis=1); Kt2_max = Kt2[np.arange(nrec), j]; Kt2_rmax = rg[j]
    Kt2_centre = (v10 / N0) ** 2                       # exact centre limit of the spline
    Kt2_025 = Kt2[:, 1]                                # r = 0.25a (the delivery's first grid point)
    kt2n = (vn[:, 1:] / (rn[1:] * Nn[:, 1:])) ** 2; jn = np.argmax(kt2n, axis=1)
    Kt2_nodes_max = kt2n[np.arange(nrec), jn]; Kt2_nodes_rmax = rn[1:][jn]
    selR = R <= R_MAX
    Kt2_bar = (M["vbar"][:, selR] / (R[selR] * M["Nbar"][:, selR])) ** 2; Kt2_bar_max = Kt2_bar.max(axis=1); Kt2_bar_rmax = R[selR][np.argmax(Kt2_bar, axis=1)]
    Kt2_app_max = ((M["vapp"][:, selR] / (R[selR] * M["Napp"][:, selR])) ** 2).max(axis=1)
    # centre identity from the kernel formula at r = 0: K_0(0, R) = (2 pi ell^2)^(-3/2) exp(-R^2/(2 ell^2))
    K00 = (2.0 * np.pi) ** (-1.5) / ell ** 3 * np.exp(-R ** 2 / (2.0 * ell ** 2))
    rho0 = M["eps"] @ K00
    centre_identity = 8.0 / 3.0 * np.pi * g * rho0
    # gates
    dvbar = np.array([fd_parity(R, M["vbar"][k], -1, 2)[0] for k in range(nrec)])
    onet_bar = ell * np.nanmax(np.abs(dvbar), axis=1)
    onet_unbar = ell * np.abs(F["vr"]).max(axis=1)
    Kr = F["vr"] / F["N"]; Kt = F["v"] / (rg * F["N"])
    strain = ell * np.maximum(np.abs(Kr), np.abs(Kt)).max(axis=1)

    def first_above(x):
        i = np.where(x > CC.GATE)[0]
        return int(i[0]) if len(i) else None
    gate = {"onet_barred": first_above(onet_bar), "onet_unbarred": first_above(onet_unbar), "strain_unbarred": first_above(strain)}
    kgate = gate["onet_barred"]
    pre = np.arange(nrec) < (kgate if kgate is not None else nrec); post = ~pre
    kform = CC.first_zero_record(M)
    # (c) frame norm on the grid, cadences
    norm_res = {}
    for c in (1, 2, 4):
        Nt = tderiv(F["N"], T, c); vt = tderiv(F["v"], T, c); vtr = tderiv(F["vr"], T, c)
        Rm = riemann(rg, F["N"], F["Nr"], F["Nrr"], F["v"], F["vr"], F["vrr"], Nt, vt, vtr)
        jj = np.nanargmax(Rm["norm"], axis=1)
        norm_res[c] = dict(norm_max=Rm["norm"][np.arange(nrec), jj], norm_rmax=rg[jj], kret_absmax=np.nanmax(np.abs(Rm["kret"]), axis=1),
                           kret_at_max=Rm["kret"][np.arange(nrec), jj],
                           comp_at_max=np.array([Rm[k][np.arange(nrec), jj] for k in ("R0101", "R0202", "R0212", "R1212", "R2323")]),
                           comp_absmax=np.array([np.nanmax(np.abs(Rm[k]), axis=1) for k in ("R0101", "R0202", "R0212", "R1212", "R2323")]))
        if c == 1:
            norm_prof = Rm["norm"].astype(np.float32)
    # stencils at the nodes (cadence 1) and barred sites
    Ntn = tderiv(Fn["N"], T, 1); vtn = tderiv(Fn["v"], T, 1); vtrn = tderiv(Fn["vr"], T, 1)
    Rn_spl = riemann(rnc, Fn["N"], Fn["Nr"], Fn["Nrr"], Fn["v"], Fn["vr"], Fn["vrr"], Ntn, vtn, vtrn)
    sten = {"spline_at_nodes": np.nanmax(Rn_spl["norm"], axis=1)}
    for order in (2, 4):
        dN = [fd_parity(rn, Nn[k], +1, order) for k in range(nrec)]; dv = [fd_parity(rn, vn[k], -1, order) for k in range(nrec)]
        Nr_ = np.array([x[0] for x in dN])[:, seln]; Nrr_ = np.array([x[1] for x in dN])[:, seln]
        vr_ = np.array([x[0] for x in dv])[:, seln]; vrr_ = np.array([x[1] for x in dv])[:, seln]
        N_ = Nn[:, seln]; v_ = vn[:, seln]
        Rf = riemann(rnc, N_, Nr_, Nrr_, v_, vr_, vrr_, tderiv(N_, T, 1), tderiv(v_, T, 1), tderiv(vr_, T, 1))
        sten["FD%d_at_nodes" % order] = np.nanmax(Rf["norm"], axis=1)
    Nb = M["Nbar"][:, selR]; vb = M["vbar"][:, selR]
    dNb = [fd_parity(R, M["Nbar"][k], +1, 2) for k in range(nrec)]; dvb = [fd_parity(R, M["vbar"][k], -1, 2) for k in range(nrec)]
    Nbr = np.array([x[0] for x in dNb])[:, selR]; Nbrr = np.array([x[1] for x in dNb])[:, selR]
    vbr = np.array([x[0] for x in dvb])[:, selR]; vbrr = np.array([x[1] for x in dvb])[:, selR]
    Rb = riemann(R[selR], Nb, Nbr, Nbrr, vb, vbr, vbrr, tderiv(Nb, T, 1), tderiv(vb, T, 1), tderiv(vbr, T, 1))
    sten["barred_sites_FD2_from_site2"] = np.nanmax(Rb["norm"][:, 1:], axis=1)
    sten["barred_sites_FD2_all_sites"] = np.nanmax(Rb["norm"], axis=1)
    # (d) source route at the nodes sigma..160a
    src = None
    if "Sr" in M:
        assert np.array_equal(M["hrn"], rn)
        idx_h = {round(float(t_), 6): i for i, t_ in enumerate(M["hT"])}
        self_f = (rnc >= ell) & (rnc <= 160.0); rf = rnc[self_f]
        resid = np.full((nrec, 5), np.nan); scale = np.full(nrec, np.nan); con = np.full((nrec, 4), np.nan)
        norm_src = np.full(nrec, np.nan); medrel = np.full((nrec, 5), np.nan)
        pg = 8.0 * np.pi * g
        for k in range(nrec):
            ih = idx_h.get(round(float(T[k]), 6))
            if ih is None:
                continue
            sel = (np.abs(M["eps"][k]) > 0) | (np.abs(M["Jt"][k]) > 0)
            K0, K1 = CC.kernels(rf, R[sel], ell)
            rho = K0 @ M["eps"][k][sel]; Jl = K1 @ M["Jt"][k][sel]
            Sr = M["Sr"][ih][seln][self_f]; St = M["St"][ih][seln][self_f]
            Cn = {key: Rn_spl[key][k][self_f] for key in ("R0101", "R0202", "R0212", "R1212", "R2323", "norm")}
            Gnn = Cn["R2323"] + 2 * Cn["R1212"]; Gnr = 2 * Cn["R0212"]
            con[k, 0] = np.abs(Gnn - pg * rho).max() / max(np.abs(pg * rho).max(), 1e-300)
            con[k, 1] = np.abs(Gnr + pg * Jl).max() / max(np.abs(pg * Jl).max(), 1e-300)     # T_nr = -J_ell
            con[k, 2] = np.abs(Gnr - pg * Jl).max() / max(np.abs(pg * Jl).max(), 1e-300)     # T_nr = +J_ell
            con[k, 3] = np.abs(Gnn - pg * rho).max()
            Kt2n = Cn["R2323"]
            S = {"R2323": Kt2n, "R1212": (pg * rho - Kt2n) / 2, "R0202": (pg * Sr + Kt2n) / 2, "R0101": pg * (rho - Sr + 2 * St) / 2 - Kt2n,
                 "R0212": -pg * Jl / 2}
            base = 4.0 * (S["R0101"] ** 2 + 2 * S["R0202"] ** 2 + 2 * S["R1212"] ** 2 + S["R2323"] ** 2)
            nsrc = np.sqrt(base + 16.0 * S["R0212"] ** 2)
            norm_src[k] = np.nanmax(nsrc); scale[k] = np.nanmax(Cn["norm"])
            for i, key in enumerate(("R0101", "R0202", "R0212", "R1212", "R2323")):
                resid[k, i] = np.nanmax(np.abs(S[key] - Cn[key]))
                medrel[k, i] = np.nanmedian(np.abs(S[key] - Cn[key]) / np.maximum(Cn["norm"], 1e-300))
        src = dict(resid=resid, scale=scale, con=con, norm_src=norm_src, medrel=medrel, r_range=[float(rf[0]), float(rf[-1])])
    # episodes
    has = np.array([len(z) > 0 for z in M["zeros"]])
    episodes = []
    k = 0
    while k < nrec:
        if has[k]:
            j2 = k
            while j2 + 1 < nrec and has[j2 + 1]:
                j2 += 1
            episodes.append([float(Trs[k]), float(Trs[j2])]); k = j2 + 1
        else:
            k += 1
    # housekeeping
    dNapp = float(np.max(np.abs(M["Nbar"] - M["Napp"]))); dvapp = float(np.max(np.abs(M["vbar"] - M["vapp"])))
    certified = np.array([(M["cert_end"] is not None and t_ <= M["cert_end"] + 1e-9) for t_ in Trs])
    kpk = int(np.argmax(Kt2_max))
    tail = {("r=%g" % x): float(np.interp(x, rg, Kt2[kpk])) for x in (0.25, 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256)}
    tail_r3 = {("r=%g" % x): float(np.interp(x, rg, Kt2[kpk]) * x ** 3 / (2 * g)) for x in (64, 96, 128, 192, 256)}

    def mx(x, m):
        if not m.any():
            return None
        xx = np.where(m, x, -np.inf); jj = int(np.nanargmax(xx))
        return {"value": float(x[jj]), "T_over_rs": float(Trs[jj]), "k": jj}
    series = {}
    for mk in MARKS:
        jj = int(np.argmin(np.abs(Trs - mk)))
        if abs(Trs[jj] - mk) < 1e-6:
            series[str(mk)] = {"Kt2_max": float(Kt2_max[jj]), "r": float(Kt2_rmax[jj]), "Kt2_centre": float(Kt2_centre[jj]), "over_cap": float(Kt2_max[jj] / cap),
                               "norm_c1": float(norm_res[1]["norm_max"][jj]), "norm_r": float(norm_res[1]["norm_rmax"][jj])}
    late = (Trs >= 15.0)
    S = {"member": name, "files": {k_: {"path": p, "md5": CC.md5(p)} for k_, p in M["files"].items()}, "records": nrec, "T_over_rs_range": [float(Trs[0]), float(Trs[-1])],
         "sigma": ell, "tau": M["tau"], "g": g, "m": mA, "rs": rs, "rho_bar": float(rho_bar), "cap": float(cap), "gate_scale": float(gate_scale),
         "n_converged": int(M["converged"].sum()), "unconverged_T_over_rs": [float(Trs[i]) for i in np.where(~M["converged"])[0]],
         "n_gauge_not_intact": int((~M["gauge_intact"]).sum()),
         "gauge_not_intact_T_over_rs": [float(Trs[i]) for i in np.where(~M["gauge_intact"])[0]],
         "gauge_labels": sorted(set(M["gauge_label"])),
         "interp_floor": {"N_max": float(np.nanmax(M["interp_floor_N"])), "v_max": float(np.nanmax(M["interp_floor_v"])),
                          "N_median": float(np.nanmedian(M["interp_floor_N"])), "v_median": float(np.nanmedian(M["interp_floor_v"])),
                          "N_max_pre": (float(np.nanmax(M["interp_floor_N"][pre])) if pre.any() else None), "v_max_pre": (float(np.nanmax(M["interp_floor_v"][pre])) if pre.any() else None),
                          "N_max_post": (float(np.nanmax(M["interp_floor_N"][post])) if post.any() else None), "v_max_post": (float(np.nanmax(M["interp_floor_v"][post])) if post.any() else None)},
         "N_minus_N_applied_max": dNapp, "v_minus_v_applied_max": dvapp,
         "gate": {k_: (None if v_ is None else {"k": v_, "T_over_rs": float(Trs[v_])}) for k_, v_ in gate.items()},
         "formation": (None if kform is None else {"k": kform, "T_over_rs": float(Trs[kform]), "zeros": M["zeros"][kform]}),
         "episodes": episodes, "zeros_end": M["zeros"][-1],
         "my_zeros_vs_stored": {"max_abs_diff": float(np.nanmax(zero_diff)), "records_with_count_mismatch": int(zero_count_mismatch.sum()),
                                "count_mismatch_T_over_rs": [float(Trs[i]) for i in np.where(zero_count_mismatch)[0]][:20]},
         "Kt2_max_all": mx(Kt2_max, np.ones(nrec, bool)), "Kt2_max_pre": mx(Kt2_max, pre), "Kt2_max_post": mx(Kt2_max, post), "Kt2_max_certified": mx(Kt2_max, certified),
         "Kt2_max_r_at_max": float(Kt2_rmax[kpk]), "Kt2_centre_limit_at_peak": float(Kt2_centre[kpk]), "Kt2_025_at_peak": float(Kt2_025[kpk]),
         "Kt2_centre_limit_max": mx(Kt2_centre, np.ones(nrec, bool)), "Kt2_025_max": mx(Kt2_025, np.ones(nrec, bool)),
         "Kt2_nodes_max": mx(Kt2_nodes_max, np.ones(nrec, bool)), "Kt2_nodes_r_at_max": float(Kt2_nodes_rmax[int(np.argmax(Kt2_nodes_max))]),
         "Kt2_bar_max": mx(Kt2_bar_max, np.ones(nrec, bool)), "Kt2_bar_r_at_max": float(Kt2_bar_rmax[int(np.argmax(Kt2_bar_max))]), "Kt2_app_max": mx(Kt2_app_max, np.ones(nrec, bool)),
         "centre_identity_max": mx(centre_identity, np.ones(nrec, bool)),
         "centre_limit_vs_identity": {"rel_max_all": float(np.max(np.abs(Kt2_centre - centre_identity) / np.maximum(np.abs(centre_identity), 1e-300))),
                                      "rel_max_intact": float(np.max((np.abs(Kt2_centre - centre_identity) / np.maximum(np.abs(centre_identity), 1e-300))[M["gauge_intact"]])),
                                      "rel_at_peak": float(abs(Kt2_centre[kpk] - centre_identity[kpk]) / centre_identity[kpk]),
                                      "rel_025_vs_identity_at_peak": float(abs(Kt2_025[kpk] - centre_identity[kpk]) / centre_identity[kpk])},
         "at_formation": (None if kform is None else {"Kt2_max": float(Kt2_max[kform]), "r": float(Kt2_rmax[kform]), "Kt2_centre": float(Kt2_centre[kform])}),
         "at_gate": (None if kgate is None else {"Kt2_max": float(Kt2_max[kgate]), "r": float(Kt2_rmax[kgate])}),
         "frac_cap_max": float(Kt2_max.max() / cap), "over_gate_scale_max": float(Kt2_max.max() / gate_scale),
         "records_above_cap": int((Kt2_max > cap).sum()), "records_above_gate_scale": int((Kt2_max > gate_scale).sum()),
         "first_record_above_gate_scale_T_over_rs": (float(Trs[np.where(Kt2_max > gate_scale)[0][0]]) if (Kt2_max > gate_scale).any() else None),
         "late_range_T15_on": ({"min": float(Kt2_max[late].min()), "max": float(Kt2_max[late].max()), "T_of_max": float(Trs[late][int(np.argmax(Kt2_max[late]))])} if late.any() else None),
         "rise_gate_to_peak": (float(Kt2_max[kpk] / Kt2_max[kgate]) if kgate is not None else None),
         "profile_at_peak": tail, "M_from_Kt2_r3_at_peak": tail_r3, "series": series,
         "norm": {}, "stencil": {k_: {"pre": mx(v_, pre), "post": mx(v_, post)} for k_, v_ in sten.items()}}
    for c, Rr in norm_res.items():
        p = mx(Rr["norm_max"], post); q = mx(Rr["norm_max"], pre)
        S["norm"][str(c)] = {"pre": q, "post": p, "certified": mx(Rr["norm_max"], certified),
                             "pre_intact": mx(Rr["norm_max"], pre & M["gauge_intact"]),
                             "kret_abs_pre": mx(Rr["kret_absmax"], pre), "kret_abs_post": mx(Rr["kret_absmax"], post),
                             "r_at_post_max": (None if p is None else float(Rr["norm_rmax"][p["k"]])), "r_at_pre_max": (None if q is None else float(Rr["norm_rmax"][q["k"]])),
                             "comp_at_post_max_point": (None if p is None else dict(zip(["R0101", "R0202", "R0212", "R1212", "R2323"], [float(x) for x in Rr["comp_at_max"][:, p["k"]]]))),
                             "comp_absmax_at_post_max_record": (None if p is None else dict(zip(["R0101", "R0202", "R0212", "R1212", "R2323"], [float(x) for x in Rr["comp_absmax"][:, p["k"]]]))),
                             "kret_at_post_max_point": (None if p is None else float(Rr["kret_at_max"][p["k"]])),
                             "norm_over_Kt2_at_post_max": (None if p is None else float(p["value"] / Kt2_max[p["k"]]))}
    if src is not None:
        ok = ~np.isnan(src["scale"]); rel = src["resid"] / src["scale"][:, None]
        pre_i = pre & M["gauge_intact"] & ok; post_o = post & ok
        names = ["R0101", "R0202", "R0212", "R1212", "R2323"]
        S["source_route"] = {"r_range": src["r_range"], "records": int(ok.sum()),
                             "constraint_rel_max_pre": {"energy": float(np.nanmax(src["con"][pre & ok, 0])) if (pre & ok).any() else None,
                                                        "momentum_Tnr=-J": float(np.nanmax(src["con"][pre & ok, 1])) if (pre & ok).any() else None,
                                                        "momentum_Tnr=+J": float(np.nanmax(src["con"][pre & ok, 2])) if (pre & ok).any() else None},
                             "constraint_rel_max_pre_intact": {"energy": float(np.nanmax(src["con"][pre_i, 0])) if pre_i.any() else None,
                                                               "momentum_Tnr=-J": float(np.nanmax(src["con"][pre_i, 1])) if pre_i.any() else None},
                             "constraint_rel_max_post": {"energy": float(np.nanmax(src["con"][post_o, 0])) if post_o.any() else None,
                                                         "momentum_Tnr=-J": float(np.nanmax(src["con"][post_o, 1])) if post_o.any() else None,
                                                         "momentum_Tnr=+J": float(np.nanmax(src["con"][post_o, 2])) if post_o.any() else None},
                             "median_resid_over_norm_pre_intact": {nm: (float(np.median(rel[pre_i, i])) if pre_i.any() else None) for i, nm in enumerate(names)},
                             "median_resid_over_norm_post": {nm: (float(np.median(rel[post_o, i])) if post_o.any() else None) for i, nm in enumerate(names)},
                             "max_resid_over_norm_pre_intact": {nm: (float(np.max(rel[pre_i, i])) if pre_i.any() else None) for i, nm in enumerate(names)},
                             "max_resid_over_norm_post": {nm: (float(np.max(rel[post_o, i])) if post_o.any() else None) for i, nm in enumerate(names)},
                             "norm_src_max_pre": float(np.nanmax(src["norm_src"][pre & ok])) if (pre & ok).any() else None,
                             "norm_src_max_post": float(np.nanmax(src["norm_src"][post_o])) if post_o.any() else None,
                             "norm_Nv_nodes_max_pre": float(np.nanmax(src["scale"][pre & ok])) if (pre & ok).any() else None,
                             "norm_Nv_nodes_max_post": float(np.nanmax(src["scale"][post_o])) if post_o.any() else None,
                             "norm_ratio_src_over_Nv_per_record": {"min_pre_intact": float(np.min((src["norm_src"] / src["scale"])[pre_i])) if pre_i.any() else None,
                                                                   "max_pre_intact": float(np.max((src["norm_src"] / src["scale"])[pre_i])) if pre_i.any() else None,
                                                                   "min_post": float(np.min((src["norm_src"] / src["scale"])[post_o])) if post_o.any() else None,
                                                                   "max_post": float(np.max((src["norm_src"] / src["scale"])[post_o])) if post_o.any() else None}}
    S["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    arrs = dict(T=T, Trs=Trs, cap=cap, gate_scale=gate_scale, Kt2_max=Kt2_max, Kt2_rmax=Kt2_rmax, Kt2_centre=Kt2_centre, Kt2_025=Kt2_025,
                Kt2_nodes_max=Kt2_nodes_max, Kt2_bar_max=Kt2_bar_max, Kt2_app_max=Kt2_app_max, centre_identity=centre_identity,
                onet_bar=onet_bar, onet_unbar=onet_unbar, strain=strain, rg=rg, Kt2_prof=Kt2.astype(np.float32), norm_prof=norm_prof,
                certified=certified, gauge_intact=M["gauge_intact"], converged=M["converged"], has_zero=has,
                r_in_rec=np.array([min(z) if z else np.nan for z in M["zeros"]]), r_out_rec=np.array([max(z) if z else np.nan for z in M["zeros"]]),
                kform=-1 if kform is None else kform, kgate=-1 if kgate is None else kgate, N0=N0, v10=v10, zero_diff=zero_diff,
                Nn=Nn, vn=vn, rn=rn, interp_floor_N=M["interp_floor_N"], interp_floor_v=M["interp_floor_v"])
    for c, Rr in norm_res.items():
        for key in ("norm_max", "norm_rmax", "kret_absmax", "comp_absmax", "comp_at_max"):
            arrs["%s_c%d" % (key, c)] = Rr[key]
    for key, arr in sten.items():
        arrs["sten_" + key] = arr
    if src is not None:
        for key in ("resid", "scale", "con", "norm_src", "medrel"):
            arrs["src_" + key] = src[key]
    np.savez_compressed(os.path.join(a.out, "cv_fields_%s.npz" % name), **arrs)
    json.dump(S, open(os.path.join(a.out, "cv_fields_%s.json" % name), "w"), indent=1, default=float)
    print("%s: %d records, %.0f s; max Kt2 %.5g at T/rs %.4g (r %.3f; centre limit %.5g; identity %.5g); cap %.4g; gate %s; formation %s; zeros diff %.2e" % (
        name, nrec, S["seconds"], Kt2_max[kpk], Trs[kpk], Kt2_rmax[kpk], Kt2_centre[kpk], centre_identity[kpk], cap, S["gate"], S["formation"],
        S["my_zeros_vs_stored"]["max_abs_diff"]), flush=True)


if __name__ == "__main__":
    main()
