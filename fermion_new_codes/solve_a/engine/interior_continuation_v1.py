#!/usr/bin/env python
"""interior_continuation_v1.py (reviewer A for the solve builder, the busi lane). Inbox busi #144 item (1), word for word:
"INTERIOR_continuation_v1: on the σ/a 4 member's continuation to T/r_s 100 (B2's stored fields and full states as
<internal>/RUN_RECORD.md records them, every stored slice) and on the σ/a 8 history to 40 (lad_s8_now.json's
slices): every trapping root of N − |v| with its signed slope, min(N − |v|), the dust-energy centroid, the sampled radius of 90 percent of
the dust, the Fermi pocket count, the collar charge, and the dust number and energy beyond 16 and 24 r_s with the Killing split at m, per
slice; labelled with the engine version that produced the records and "past the record's own limit". Deliver to
<internal>/solve_a with 1 line in <internal>/DELIVERIES.md. CPU, no run."
Run on B's records and slices (40 to 100), on B2's (every stored slice, 12 to 100, labelled with B2's engine) and on the sigma/a 8 member's
(12 to 40); the production's 10 slices are admitted by --history prod.

Per slice (the reader slices144_io_v1; the record at the slice's clock from the history's records json; the fields npz for the dense re-solve):
(a) the gate: sigma |d_r v| on the slice's applied fields (sigma = ell, the records' averaging scale; the sites' central difference), its
    largest value over the chain and inside the outer root, against --gate (0.1, requirements 5 I1's working value): the label "past the
    record's own limit" is computed from it on every row (never asserted), beside the engine version that produced the records (the
    inventory's segment label and the record's own key).
(b) every trapping root of N - |v| with its signed slope: on the dense fields re-solved by the history's field module from the record's stored
    shell totals (eps_tot, J_tot of the fields npz; solve_constraints_ivp with the run's collar policy; trapping_radius gives the zeros of
    N + v, identical to those of N - |v| for v <= 0), the slope d_r(N - |v|) = N' - sign(v) v' at each zero by engine_rows_v2.deriv
    (Richardson at ell/--hfrac) on the dense callables, or the local 2-level Richardson central difference where engine_rows_v2 is absent
    (named in the row). A negative slope is an inner root (N - |v| falls below 0 outward), a positive slope an outer root. Beside them the
    record's stored trapping_zeros, kappa_T (its slope of N + v at R_t), R_t, r_in and tangencies; the zeros of the slice's applied
    N_bar - |v_bar| between sites (linear) with the sites' finite-difference slope; the zeros of the fields npz's site fields N, v the same
    way. min(N - |v|): the record's, the dense re-solve's and the applied fields' with their radii.
(c) the tagged dust D = sum_kappa 16 kappa sum_n f_n phi_n phi_n^dagger from the stored dust columns (not Loewdin-mixed: the dust's identity
    is the stored column's; the dust block's own orthonormality defect printed): its number per site; the dust-energy centroid
    sum_x R_x eps(x)/sum_x eps(x) on 2 densities, eps_0(x) = Re Tr[Pi_x h_0 D] (the h_0 density conjugate to N, the records' eps convention,
    summed over the 2 components) and eps_m(x) = Re Tr[Pi_x h_m D] (the Killing density at the applied fields); the number centroid; the
    centroid of the records' own dust energy eps_tot - eps_pairs at the record (the fields npz) with the relative difference of its total to
    eps_0's total (on a complex64 slice this is the floor's measurement); R_50 and R_90, the sampled radii holding 50 and 90 percent of the
    dust number (engine_core_slices_v3.frac_radius).
(d) the Fermi pockets: the record's n_pocket per channel, inversions_total, n_refill_R0 and mask_equals_negative_set; from the slice per
    channel the stored mask against 1(e_k < 0) in the eigenvalue order: the occupied count against the sea's and the dust's column counts,
    the occupied levels at e > 0, the empty levels at e < 0, the number of maximal runs of inverted levels (the slice's own count; the
    engine's definition is the record's).
(e) the collar charge: the record's E and E_over_E0 (W/(2g) at the collar), beside H_in, M1_balance, indicator.collar_screen, R_c.
(f) the dust number and energy beyond 16 and 24 r_s with the Killing split at m: per channel with dust, the exterior block h_YY of h_m
    beyond the cut in its eigenbasis (omega_j, phi_j) (engine_massive_channel_D_v2's convention, the same quantities as its Dv2 outputs),
    n_j = sum_n f_n |phi_j^dagger Pi_Y phi_n|^2; N_out = sum_j n_j (= the dust weight beyond the cut, the completeness check printed),
    E_block = sum_j omega_j n_j (= Re Tr[h_YY D_YY], checked) and E_direct = Re Tr[Pi_Y h_m D] (the cut's bonds included); the split at
    --m-split-at m (1.0): omega >= m (unbound content above m), |omega| < m, omega <= -m; the omega/m bins of
    engine_massive_channel_D_v2.bin_spectrum where importable; the dust's number and Killing-energy currents through the cut
    (page_ops_v6.number_current_csr i[h, Pi_out] and current_out_csr (i/2)[h^2, Pi_out]; outward positive); the sums with 16 kappa.
Outputs per slice: interior_<history>_T<clock>.json and .npz (the per-site densities, the dense fields at the sites, the exterior Killing
spectra with the dust's occupation per level); compose writes INTERIOR_continuation_v1_<history>.md, a B against B2 table at the matched
clocks, and INTERIOR_continuation_v1_all.json. CPU only; no plotting; no mutual information; no dates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slices144_io_v1 as IO  # noqa: E402  (sets the thread environment before numpy is imported)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

READOUT = "interior"
TITLE = "INTERIOR_continuation_v1"
CODE = "interior_continuation_v1.py"
LIMIT_LABEL = "past the record's own limit"


def add_arguments(ap):
    g = ap.add_argument_group("interior_continuation_v1")
    g.add_argument("--gate", type=float, default=0.1, help="the 1-term gate sigma |d_r v| (requirements 5 I1's working value)")
    g.add_argument("--dense-sites", dest="dense_sites", default="on", choices=["on", "off"], help="store the dense re-solved fields sampled at the sites")


def is_done(entry, a):
    return os.path.exists(IO.out_paths(a, READOUT, entry)[0])


# ---------------------------------------------------------------------------------------------------------- the arithmetic
def site_sum(d2):
    d2 = np.asarray(d2)
    return d2[0::2] + d2[1::2]


def dust_number_site(dust, f):
    """the tagged dust's number per site (the 2 components summed)."""
    return site_sum((np.abs(dust) ** 2) @ np.asarray(f, float))


def site_density(X, f, Hop):
    """Re Tr[Pi_x H sum_n f_n x_n x_n^dagger] per site: the symmetrized site density of the operator H on the columns X."""
    Y = Hop @ X
    return site_sum(np.real(np.conj(X) * Y) @ np.asarray(f, float))


def centroid(R, w):
    tot = float(np.sum(w))
    return float(np.sum(np.asarray(R, float) * np.asarray(w, float)) / tot) if tot != 0.0 else None


def rows_beyond(R, x):
    """the rows (2 per site) of the sites with R > x and the first such site (engine_massive_channel_D_v2.region_rows where importable)."""
    if IO.DM is not None:
        ry, i0 = IO.DM.region_rows(np.asarray(R, float), float(x))
        return np.asarray(ry, int), int(i0)
    i0 = int(np.searchsorted(np.asarray(R, float), float(x), side="right"))
    return np.arange(2 * i0, 2 * len(R)), i0


def pocket_stats(e, mask, n_sea, n_dust):
    """the stored mask against 1(e < 0) in the eigenvalue order."""
    e = np.asarray(e, float)
    mask = np.asarray(mask, bool)[: len(e)]
    order = np.argsort(e, kind="stable")
    es = e[order]
    ms = mask[order]
    neg = es < 0.0
    inv = ms != neg
    runs = int(np.sum(inv[1:] & ~inv[:-1]) + (1 if (inv.size and inv[0]) else 0))
    return {"occupied": int(ms.sum()), "negative_levels": int(neg.sum()), "n_sea_columns": int(n_sea), "n_dust_columns": int(n_dust),
            "occupied_at_positive_e": int((ms & ~neg).sum()), "empty_at_negative_e": int((~ms & neg).sum()), "inverted_levels": int(inv.sum()), "inverted_runs": runs,
            "mask_equals_negative_set": bool(not inv.any()), "e_min": float(es[0]), "e_max": float(es[-1]),
            "gap_at_zero": float(es[~neg].min() - es[neg].max()) if (neg.any() and (~neg).any()) else None}


def dust_killing_split(H, R, dust, f, x, m, split_at, T_H=None):
    """the dust beyond the cut x in the Killing eigenbasis of the exterior block h_YY (omega_j, phi_j): N_out, E_block (two forms), E_direct,
    the split at split_at m, the omega/m bins; returns (scalars, arrays)."""
    R = np.asarray(R, float)
    ry, i0 = rows_beyond(R, x)
    hY = H[ry][:, ry].toarray()
    om, ph = IO.eigh(hY)
    Xd = np.asarray(dust)[ry]
    Ad = ph.conj().T @ Xd
    nj = (np.abs(Ad) ** 2) @ np.asarray(f, float)
    weight = float(np.sum(np.asarray(f, float) * np.sum(np.abs(Xd) ** 2, axis=0)))
    hYX = hY @ Xd
    E_block_trace = float(np.real(np.sum(np.asarray(f, float) * np.sum(np.conj(Xd) * hYX, axis=0))))
    HX = (H @ dust)[ry]
    E_direct = float(np.real(np.sum(np.asarray(f, float) * np.sum(np.conj(Xd) * HX, axis=0))))
    th = float(split_at) * float(m)
    above = om >= th
    neg = om <= -th
    mid = ~(above | neg)
    out = {"R_cut": float(x), "first_site": int(i0), "sites": int(len(R) - i0), "levels": int(len(om)), "N_out": float(np.sum(nj)), "dust_weight": weight,
           "completeness_check": float(np.sum(nj) - weight), "E_block": float(np.sum(om * nj)), "E_block_trace": E_block_trace, "E_direct": E_direct,
           "threshold": th, "threshold_over_m": float(split_at),
           "N_above_m": float(np.sum(nj[above])), "E_above_m": float(np.sum(om[above] * nj[above])),
           "N_below_m": float(np.sum(nj[mid])), "E_below_m": float(np.sum(om[mid] * nj[mid])),
           "N_neg_m": float(np.sum(nj[neg])), "E_neg_m": float(np.sum(om[neg] * nj[neg])),
           "omega_range": [float(om.min()), float(om.max())] if len(om) else None,
           "mean_omega_over_m": float(np.sum(om * nj) / np.sum(nj) / m) if np.sum(nj) > 0 else None,
           "basis": "the eigenbasis of the exterior block h_YY of h_m beyond the cut (engine_massive_channel_D_v2's convention)"}
    if IO.DM is not None and T_H:
        try:
            _, mb = IO.DM.bin_spectrum(om, [nj], float(T_H), float(m))
            out["bins_m"] = [{"lo": b["lo"], "hi": b["hi"], "count": b["count"], "n": b["n"][0], "E": b["E"][0]} for b in mb]
            out["bins_source"] = "engine_massive_channel_D_v2.bin_spectrum (omega/m edges M_BINS)"
        except Exception as ex:  # noqa: BLE001
            out["bins_source"] = "bin_spectrum failed: %r" % ex
    return out, {"omega": om, "n_dust": nj}


def cut_current(H, R, x, X, f, kind="number"):
    """the current of sum_n f_n x_n x_n^dagger through the cut at x: number i[h, Pi_out] (page_ops_v6.number_current_csr) or Killing energy
    (i/2)[h^2, Pi_out] (current_out_csr); outward positive. The operators live on the bonds crossing the cut, so only those rows are read."""
    J = IO.OPS.number_current_csr(H, np.asarray(R, float), float(x)) if kind == "number" else IO.OPS.current_out_csr(H, np.asarray(R, float), float(x))
    co = J.tocoo()
    if co.nnz == 0:
        return 0.0
    rows = np.unique(np.concatenate([co.row, co.col]))
    Jb = J[rows][:, rows].toarray()
    Xr = np.asarray(X)[rows]
    return float(np.real(np.sum(np.asarray(f, float) * np.sum(np.conj(Xr) * (Jb @ Xr), axis=0))))


def dense_fields(ctx, entry, k, R, a):
    """the dense fields re-solved from the record's stored shell totals by the history's field module."""
    FM = ctx.field_module()
    eps = ctx.fields_key(entry, "eps_tot")[k]
    J = ctx.fields_key(entry, "J_tot")[k]
    t0 = time.time()
    try:
        res = FM.solve_constraints_ivp(np.asarray(eps, float), np.asarray(J, float), ctx.g, np.asarray(R, float), ctx.ell, collar_policy=a.collar_policy, support_rel=a.support_rel)
        info = res[3]
    except Exception as ex:  # noqa: BLE001
        return {"failed": True, "error": repr(ex), "module": ctx.H["fields_module"]}
    if info.get("failed"):
        return {"failed": True, "info_failed": str(info.get("failed")), "module": ctx.H["fields_module"]}
    tz = FM.trapping_radius(None, None, np.asarray(R, float), info)
    return {"failed": False, "info": info, "zeros": [float(z) for z in tz.get("zeros", [])], "min": tz.get("min"), "r_min": tz.get("r_min"),
            "module": ctx.H["fields_module"], "seconds": time.time() - t0}


def dense_slope(info, roots, h):
    """d_r(N - |v|) at the roots on the dense callables."""
    Nf, vf = info["N_dense"], info["v_dense"]
    roots = np.asarray(roots, float)
    if roots.size == 0:
        return [], "no roots"
    if IO.RW is not None and hasattr(IO.RW, "deriv"):
        try:
            _, N1, _ = IO.RW.deriv(Nf, roots, h)
            v0, v1, _ = IO.RW.deriv(vf, roots, h)
            s = np.asarray(N1, float).reshape(-1) - np.sign(np.asarray(v0, float).reshape(-1)) * np.asarray(v1, float).reshape(-1)
            if s.shape[0] == roots.shape[0]:
                return [float(z) for z in s], "engine_rows_v2.deriv (Richardson at ell/hfrac) on N and v: N' - sign(v) v'"
        except Exception:  # noqa: BLE001
            pass
    f = lambda r: IO.call_dense(Nf, r) - abs(IO.call_dense(vf, r))  # noqa: E731
    return [IO.richardson_d1(f, float(z), h) for z in roots], "local 2-level Richardson central difference on N - |v| (engine_rows_v2 absent or its signature unmatched)"


# ------------------------------------------------------------------------------------------------------------ the slice
def process_slice(rd, ctx, entry, a):
    t0 = time.time()
    log = ctx.log
    pj, pz = IO.out_paths(a, READOUT, entry)
    rec = ctx.record_at(entry, rd.T)
    m, g, ell, rs = ctx.m, ctx.g, ctx.ell, rd.rs
    R, N_app, v_app = rd.R, rd.N_applied, rd.v_applied
    cuts = [(lab, x * rs) for lab, x in IO.cuts_from_arg(a.cuts_rs)]
    row = IO.slice_header(rd, ctx, entry, rec, READOUT, TITLE, CODE)
    arrays = {"R": R, "N_applied": N_app, "v_applied": v_app}
    E_rec = rec.get("E")
    T_H = 1.0 / (8.0 * math.pi * g * float(E_rec)) if E_rec else None
    row["record"] = {"E": E_rec, "T_H": T_H, "m_over_TH": (m / T_H) if T_H else None, "R_c": rec.get("R_c"), "R_t": rec.get("R_t"), "r_in": rec.get("r_in"),
                     "trapping_zeros": rec.get("trapping_zeros"), "tangencies": rec.get("tangencies"), "kappa_T": rec.get("kappa_T"), "min_N_minus_absv": rec.get("min_N_minus_absv"),
                     "r_min_N_minus_absv": rec.get("r_min_N_minus_absv"), "converged": rec.get("converged"), "slice_saved": rec.get("slice_saved")}
    # (a) the gate on the applied fields
    dv = np.gradient(v_app, R)
    sdv = ell * np.abs(dv)
    root_rec = float(max(rec["trapping_zeros"])) if rec.get("trapping_zeros") else None
    inside = R <= (root_rec if root_rec is not None else float(rec.get("R_c") or R[-1]))
    i_all = int(np.argmax(sdv))
    i_in = int(np.argmax(np.where(inside, sdv, -1.0)))
    row["gate"] = {"sigma": ell, "threshold": a.gate, "sigma_abs_dv_max": float(sdv[i_all]), "at_R": float(R[i_all]), "sigma_abs_dv_max_inside_root": float(sdv[i_in]),
                   "at_R_inside": float(R[i_in]), "inside_region": "R <= the record's outer root %s" % root_rec if root_rec is not None else "R <= the record's R_c (no root at this clock)",
                   "exceeded": bool(sdv[i_all] > a.gate), "exceeded_inside_root": bool(sdv[i_in] > a.gate),
                   "definition": "sigma |d_r v| on the applied site fields (numpy.gradient), sigma = ell; requirements 5 I1's working value as the threshold"}
    row["limit_label"] = LIMIT_LABEL if row["gate"]["exceeded"] else "within the record's own limit (the gate is not exceeded on the applied fields)"
    # (b) the roots
    k = ctx.fields_index(entry, rd.T)
    dense = dense_fields(ctx, entry, k, R, a)
    h = ell / a.hfrac
    roots = []
    route = None
    if not dense["failed"]:
        slopes, route = dense_slope(dense["info"], dense["zeros"], h)
        roots = [{"R": float(z), "slope": float(s), "kind": "inner (slope < 0)" if s < 0 else "outer (slope > 0)"} for z, s in zip(dense["zeros"], slopes)]
        if a.dense_sites == "on":
            try:
                Nd = IO.call_dense(dense["info"]["N_dense"], R)
                vd = IO.call_dense(dense["info"]["v_dense"], R)
                arrays["N_dense_sites"] = Nd
                arrays["v_dense_sites"] = vd
                row["dense_vs_applied"] = {"max_abs_dN": float(np.max(np.abs(Nd - N_app))), "max_abs_dv": float(np.max(np.abs(vd - v_app))),
                                           "note": "the dense re-solve sampled at the sites against the applied (averaged) site fields: the averaging's own difference"}
            except Exception as ex:  # noqa: BLE001
                row["dense_vs_applied"] = {"error": repr(ex)}
    f_app = N_app - np.abs(v_app)
    Ns, vs = ctx.fields_key(entry, "N"), ctx.fields_key(entry, "v")
    site_roots = IO.interp_zeros(R, Ns[k] - np.abs(vs[k])) if (Ns is not None and vs is not None) else None
    ia = int(np.argmin(f_app))
    row["roots"] = {"dense": roots, "dense_route": route, "dense_module": dense["module"], "dense_failed": dense["failed"], "dense_error": dense.get("error") or dense.get("info_failed"),
                    "dense_seconds": dense.get("seconds"), "h": h,
                    "record": {"trapping_zeros": rec.get("trapping_zeros"), "kappa_T_at_Rt": rec.get("kappa_T"), "R_t": rec.get("R_t"), "r_in": rec.get("r_in"), "tangencies": rec.get("tangencies")},
                    "applied": IO.interp_zeros(R, f_app), "site_fields_npz": site_roots,
                    "definitions": "dense: the zeros of N + v on the fields re-solved from the record's stored shell totals (identical to N - |v|'s for v <= 0), slope d_r(N - |v|); "
                                   "applied: sign changes of the slice's N_bar - |v_bar| between sites, linear, with the sites' finite-difference slope; site fields: the fields npz's N, v the same way"}
    row["min_N_minus_absv"] = {"record": rec.get("min_N_minus_absv"), "record_r": rec.get("r_min_N_minus_absv"), "dense": dense.get("min"), "dense_r": dense.get("r_min"),
                               "applied": float(f_app[ia]), "applied_r": float(R[ia]),
                               "site_fields_npz": float(np.min(Ns[k] - np.abs(vs[k]))) if (Ns is not None and vs is not None) else None}
    # (c) to (f) per channel
    nd_site = np.zeros(rd.nsites)
    eps0_site = np.zeros(rd.nsites)
    epsm_site = np.zeros(rd.nsites)
    chans = []
    for kap in rd.kappas:
        w = rd.weights[kap]
        ch = {"kappa": int(kap), "weight": w, "n_sea": rd.n_sea[kap], "n_dust": rd.n_dust[kap]}
        e = rd.e(kap)
        mask = rd.mask(kap)
        ch["pockets_slice"] = pocket_stats(e, mask, rd.n_sea[kap], rd.n_dust[kap]) if (e is not None and mask is not None) else None
        npk = rec.get("n_pocket")
        ch["pockets_record"] = (npk.get(str(kap)) if isinstance(npk, dict) else None)
        if rd.n_dust[kap] == 0:
            ch["cuts"] = None
            chans.append(ch)
            continue
        dust, f = rd.dust(kap)
        h0, Q, H = IO.operators(R, m, kap, N_app, v_app)
        if a.check_eigs_kappa == kap and e is not None:
            t1 = time.time()
            ch["eig_vs_stored"] = float(np.max(np.abs(np.sort(IO.eigvalsh(H.toarray())) - np.sort(e))))
            ch["eig_seconds"] = time.time() - t1
        Gd = dust.conj().T @ dust
        ch["dust_orthonormality"] = float(np.max(np.abs(Gd - np.eye(Gd.shape[0]))))
        nd = dust_number_site(dust, f)
        e0 = site_density(dust, f, h0)
        em = site_density(dust, f, H)
        nd_site += w * nd
        eps0_site += w * e0
        epsm_site += w * em
        ch.update({"dust_total": float(nd.sum()), "E_h0_total": float(e0.sum()), "E_killing_total": float(em.sum()), "centroid_number": centroid(R, nd), "centroid_energy_h0": centroid(R, e0)})
        ch["cuts"] = {}
        for lab, x in cuts:
            sp, arr = dust_killing_split(H, R, dust, f, x, m, a.m_split_at, T_H)
            sp["number_current"] = cut_current(H, R, x, dust, f, "number")
            sp["energy_current"] = cut_current(H, R, x, dust, f, "energy")
            ch["cuts"][lab] = sp
            arrays["k%d_cut%s_omega" % (kap, lab)] = arr["omega"]
            arrays["k%d_cut%s_n_dust" % (kap, lab)] = arr["n_dust"]
        arrays["k%d_dust_number_site" % kap] = nd
        arrays["k%d_dust_eps_h0_site" % kap] = e0
        arrays["k%d_dust_eps_killing_site" % kap] = em
        chans.append(ch)
        del dust, Gd
    row["channels"] = chans
    keys = ("N_out", "dust_weight", "E_block", "E_direct", "N_above_m", "E_above_m", "N_below_m", "E_below_m", "N_neg_m", "E_neg_m", "number_current", "energy_current", "completeness_check")
    row["weighted"] = {}
    for lab, x in cuts:
        ws = {"R_cut": x, "R_cut_over_rs": x / rs}
        for key in keys:
            ws[key] = float(sum(c["weight"] * c["cuts"][lab][key] for c in chans if c.get("cuts")))
        ws["max_abs_completeness_check"] = float(max([abs(c["cuts"][lab]["completeness_check"]) for c in chans if c.get("cuts")] or [0.0]))
        ws["max_abs_block_identity"] = float(max([abs(c["cuts"][lab]["E_block"] - c["cuts"][lab]["E_block_trace"]) for c in chans if c.get("cuts")] or [0.0]))
        row["weighted"][lab] = ws
    # (c) the centroids and radii of the weighted dust
    tot = float(nd_site.sum())
    cum = np.cumsum(nd_site) / tot if tot > 0 else None
    eps_tot, eps_pairs = ctx.fields_key(entry, "eps_tot"), ctx.fields_key(entry, "eps_pairs")
    eps_rec = (eps_tot[k] - eps_pairs[k]) if (eps_tot is not None and eps_pairs is not None) else None
    row["dust"] = {"total": tot, "R_50": IO.frac_radius(R, cum, 0.5) if cum is not None else None, "R_90": IO.frac_radius(R, cum, 0.9) if cum is not None else None,
                   "centroid_number": centroid(R, nd_site), "centroid_energy_h0": centroid(R, eps0_site), "centroid_energy_killing": centroid(R, epsm_site),
                   "E_h0_total": float(eps0_site.sum()), "E_killing_total": float(epsm_site.sum()),
                   "centroid_energy_records": centroid(R, eps_rec) if eps_rec is not None else None, "E_records_total": float(np.sum(eps_rec)) if eps_rec is not None else None,
                   "records_vs_columns_rel": (abs(float(np.sum(eps_rec)) - float(eps0_site.sum())) / abs(float(np.sum(eps_rec)))) if (eps_rec is not None and np.sum(eps_rec) != 0) else None,
                   "records_vs_columns_site_max": float(np.max(np.abs(eps_rec - eps0_site))) if eps_rec is not None else None,
                   "definitions": "number per site from the stored dust columns weighted 16 kappa; centroids sum_x R_x q(x)/sum_x q(x) with q the number, the h_0 density (the records' eps convention), "
                                  "the Killing density at the applied fields, and the records' eps_tot - eps_pairs; R_50, R_90 the sampled radii holding 50 and 90 percent of the dust number (linear between sites)"}
    arrays["dust_number_site"] = nd_site
    arrays["dust_eps_h0_site"] = eps0_site
    arrays["dust_eps_killing_site"] = epsm_site
    if eps_rec is not None:
        arrays["dust_eps_records_site"] = eps_rec
    # (d) the record's pocket keys
    row["pockets_record"] = {"n_pocket": rec.get("n_pocket"), "inversions_total": rec.get("inversions_total"), "n_refill_R0": rec.get("n_refill_R0"), "mask_equals_negative_set": rec.get("mask_equals_negative_set")}
    sl = [c["pockets_slice"] for c in chans if c.get("pockets_slice")]
    row["pockets_slice_total"] = {"inverted_runs": int(sum(p["inverted_runs"] for p in sl)), "occupied_at_positive_e": int(sum(p["occupied_at_positive_e"] for p in sl)),
                                  "empty_at_negative_e": int(sum(p["empty_at_negative_e"] for p in sl)), "channels_mask_equals_negative_set": int(sum(1 for p in sl if p["mask_equals_negative_set"])),
                                  "channels": len(sl)}
    # (e) the collar charge
    row["collar"] = {"E": rec.get("E"), "E_over_E0": rec.get("E_over_E0"), "H_in": rec.get("H_in"), "M1_balance": rec.get("M1_balance"),
                     "collar_screen": (rec.get("indicator") or {}).get("collar_screen") if isinstance(rec.get("indicator"), dict) else None, "R_c": rec.get("R_c"), "r_collar": rec.get("r_collar"),
                     "W_over_2g_at_Rt": rec.get("W_over_2g_at_Rt"), "misner_sharp_at_Rt": rec.get("misner_sharp_at_Rt"), "dE_dT": rec.get("dE_dT"), "dHin_dT": rec.get("dHin_dT")}
    row["seconds"] = time.time() - t0
    IO.save_outputs(pj, pz, row, arrays)
    w16 = row["weighted"].get(cuts[0][0], {})
    log("%s T/r_s %.3f (%s; %s): %s; gate sigma|dv| %.3f; dense roots %s (%s); min(N-|v|) rec %s dense %s; dust total %.4f, R_50 %s, R_90 %s, centroid h0 %s (records %s, rel %s); beyond %s r_s N_out %.4e (>= m: %.4e), E_direct %+.4e, current %+.3e; %.0f s" % (
        TITLE, rd.T_over_rs, rd.dtype_str, entry["history"], row["limit_label"], row["gate"]["sigma_abs_dv_max"],
        ", ".join("%.3f (%+.4f)" % (q["R"], q["slope"]) for q in roots) or "none", route or "no re-solve", IO.fx(rec.get("min_N_minus_absv"), "%+.5f"), IO.fx(dense.get("min"), "%+.5f"),
        tot, IO.fx(row["dust"]["R_50"], "%.3f"), IO.fx(row["dust"]["R_90"], "%.3f"), IO.fx(row["dust"]["centroid_energy_h0"], "%.3f"), IO.fx(row["dust"]["centroid_energy_records"], "%.3f"),
        IO.fx(row["dust"]["records_vs_columns_rel"], "%.2e"), cuts[0][0], w16.get("N_out", 0.0), w16.get("N_above_m", 0.0), w16.get("E_direct", 0.0), w16.get("number_current", 0.0), row["seconds"]))
    return row


# ------------------------------------------------------------------------------------------------------------ the tables
def compose(a, hists):
    rows = IO.load_rows(a, READOUT, hists)
    if not rows:
        return
    cl = sorted({lab for r in rows for lab in (r.get("weighted") or {})}, key=float)
    L = ["# %s: the interior on the stored slices (%s)" % (TITLE, CODE), "",
         "Inbox busi #144 item (1). Every row carries the engine version that produced the records (the segment's label and the record's own key) and the label computed from the gate sigma |d_r v| against %g on the slice's applied fields: '%s' where it is exceeded." % (a.gate, LIMIT_LABEL), "",
         "- roots: the zeros of the dense re-solved N + v (the history's field module on the record's stored shell totals) with the slope d_r(N - |v|) by engine_rows_v2.deriv (Richardson at ell/%g); negative slope = inner root, positive = outer root; the record's stored zeros, kappa_T, R_t, r_in beside; the applied N_bar - |v_bar|'s zeros (linear between sites) with the sites' slope." % a.hfrac,
         "- the dust: the stored dust columns weighted 16 kappa; centroids sum R q/sum q on the number, the h_0 density (the records' eps convention), the Killing density at the applied fields and the records' eps_tot - eps_pairs; R_50, R_90 the sampled radii holding 50 and 90 percent of the number.",
         "- pockets: the record's n_pocket (summed), inversions_total, mask_equals_negative_set; from the slice the stored mask against 1(e_k < 0): the maximal runs of inverted levels, the occupied levels at e > 0, the empty at e < 0.",
         "- beyond the cuts (weighted 16 kappa): the dust in the Killing eigenbasis of the exterior block h_YY of h_m beyond the cut: N_out, E_direct = Re Tr[Pi_Y h_m D] (the block form E_block beside, its 2 forms' largest difference in the checks), the split at %g m (unbound content above m), the number and Killing-energy currents through the cut (outward positive); completeness: sum_j n_j - the dust weight beyond the cut." % a.m_split_at,
         "- complex64 slices carry the floor %s to %s (sources) and %s (orthonormality); the records-against-columns relative difference of the dust energy is that floor's measurement on each slice." % tuple(["%.1e" % x for x in IO.C64_FLOOR["source_rel_range"]] + ["%.1e" % IO.C64_FLOOR["orthonormality_defect"]]), ""]
    for hk in hists:
        rs_ = [r for r in rows if r["history"] == hk]
        if not rs_:
            continue
        L += ["## %s: %s" % (hk, rs_[0]["history_label"]), "", "Engine: %s" % rs_[0]["engine"], "", "### The roots and the minimum", "",
              "| T/r_s | dtype | label | sigma max abs(dv) (inside root) | dense roots R (slope) | record zeros; kappa_T; R_t; r_in | applied roots R (slope) | min(N - abs v): record (r); dense (r); applied (r) | re-solve |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rs_:
            ro = r["roots"]
            mn = r["min_N_minus_absv"]
            L.append("| %.3f | %s | %s | %.4f (%.4f) | %s | %s; %s; %s; %s | %s | %s (%s); %s (%s); %s (%s) | %s |" % (
                r["T_over_rs"], r["dtype"], r["limit_label"].split(" (")[0], r["gate"]["sigma_abs_dv_max"], r["gate"]["sigma_abs_dv_max_inside_root"],
                ", ".join("%.3f (%+.4f)" % (q["R"], q["slope"]) for q in ro["dense"]) or ("none" if not ro["dense_failed"] else "re-solve failed"),
                ", ".join("%.3f" % z for z in (ro["record"]["trapping_zeros"] or [])) or "none", IO.fx(ro["record"]["kappa_T_at_Rt"], "%.5f"), IO.fx(ro["record"]["R_t"], "%.3f"), IO.fx(ro["record"]["r_in"], "%.3f"),
                ", ".join("%.3f (%+.4f)" % (q["R"], q["slope"]) for q in ro["applied"]) or "none",
                IO.fx(mn["record"], "%+.5f"), IO.fx(mn["record_r"], "%.2f"), IO.fx(mn["dense"], "%+.5f"), IO.fx(mn["dense_r"], "%.2f"), IO.fx(mn["applied"], "%+.5f"), IO.fx(mn["applied_r"], "%.2f"),
                ro["dense_module"] if not ro["dense_failed"] else "FAILED: %s" % ro["dense_error"]))
        L += ["", "### The dust, the pockets, the collar charge", "",
              "| T/r_s | dust total | R_50 | R_90 | centroid: number; energy h_0; Killing; records (records vs columns, rel) | pockets slice: inverted runs; occ at e > 0; empty at e < 0; channels with mask = neg set | record: n_pocket; inversions_total; mask = neg | E/E0; E; H_in; collar_screen; R_c |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs_:
            d = r["dust"]
            ps = r["pockets_slice_total"]
            pr = r["pockets_record"]
            co = r["collar"]
            npk = pr.get("n_pocket")
            npk_s = sum(float(x) for x in npk.values()) if isinstance(npk, dict) else npk
            L.append("| %.3f | %.4f | %s | %s | %s; %s; %s; %s (%s) | %d; %d; %d; %d/%d | %s; %s; %s | %s; %s; %s; %s; %s |" % (
                r["T_over_rs"], d["total"], IO.fx(d["R_50"], "%.3f"), IO.fx(d["R_90"], "%.3f"), IO.fx(d["centroid_number"], "%.3f"), IO.fx(d["centroid_energy_h0"], "%.3f"),
                IO.fx(d["centroid_energy_killing"], "%.3f"), IO.fx(d["centroid_energy_records"], "%.3f"), IO.fx(d["records_vs_columns_rel"], "%.2e"),
                ps["inverted_runs"], ps["occupied_at_positive_e"], ps["empty_at_negative_e"], ps["channels_mask_equals_negative_set"], ps["channels"],
                IO.fx(npk_s, "%g"), IO.fx(pr.get("inversions_total"), "%s"), IO.fx(pr.get("mask_equals_negative_set"), "%s"),
                IO.fx(co["E_over_E0"], "%.6f"), IO.fx(co["E"], "%.6f"), IO.fx(co["H_in"], "%.6f"), IO.fx(co["collar_screen"], "%s"), IO.fx(co["R_c"], "%.2f")))
        for lab in cl:
            L += ["", "### The dust beyond %s r_s (weighted 16 kappa)" % lab, "",
                  "| T/r_s | N_out | E_direct | E_block | N (omega >= %g m) | E (omega >= %g m) | N (abs omega < m) | N (omega <= -m) | number current | energy current | checks: completeness; block identity |" % (a.m_split_at, a.m_split_at),
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                w = (r.get("weighted") or {}).get(lab)
                if not w:
                    continue
                L.append("| %.3f | %.6e | %+.6e | %+.6e | %.6e | %+.6e | %.6e | %.3e | %+.4e | %+.4e | %.1e; %.1e |" % (
                    r["T_over_rs"], w["N_out"], w["E_direct"], w["E_block"], w["N_above_m"], w["E_above_m"], w["N_below_m"], w["N_neg_m"], w["number_current"], w["energy_current"],
                    w["max_abs_completeness_check"], w["max_abs_block_identity"]))
        if cl:
            lab = cl[0]
            L += ["", "### Per channel beyond %s r_s (unweighted)" % lab, "",
                  "| T/r_s | kappa | n_dust | N_out | E_direct | N (omega >= m) | E (omega >= m) | mean omega/m | number current | dust orthonormality (stored) | eig vs stored e_k | pockets slice: inverted runs (occ at e > 0, empty at e < 0) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs_:
                for c in r["channels"]:
                    if not c.get("cuts"):
                        continue
                    q = c["cuts"][lab]
                    p = c.get("pockets_slice") or {}
                    L.append("| %.3f | %d | %d | %.6e | %+.6e | %.6e | %+.6e | %s | %+.4e | %.1e | %s | %s (%s, %s) |" % (
                        r["T_over_rs"], c["kappa"], c["n_dust"], q["N_out"], q["E_direct"], q["N_above_m"], q["E_above_m"], IO.fx(q.get("mean_omega_over_m"), "%.4f"), q["number_current"],
                        c["dust_orthonormality"], IO.fx(c.get("eig_vs_stored"), "%.1e"), IO.fx(p.get("inverted_runs"), "%d"), IO.fx(p.get("occupied_at_positive_e"), "%d"), IO.fx(p.get("empty_at_negative_e"), "%d")))
        L.append("")
    if "B" in hists and "B2" in hists:
        rb = {round(r["T_over_rs"], 4): r for r in rows if r["history"] == "B"}
        rb2 = {round(r["T_over_rs"], 4): r for r in rows if r["history"] == "B2"}
        common = sorted(set(rb) & set(rb2))
        if common:
            lab = cl[0] if cl else None
            L += ["## B against B2 at the matched clocks (B: 2048 sites; B2: 4096 sites; each with its own engine label)", "",
                  "| T/r_s | dense roots B | dense roots B2 | min(N - abs v) dense B; B2 | centroid h_0 B; B2 | R_90 B; B2 | E/E0 B; B2 | N_out beyond %s B; B2 | N (omega >= m) B; B2 | E_direct B; B2 |" % lab, "|---|---|---|---|---|---|---|---|---|---|"]
            for t in common:
                x, y = rb[t], rb2[t]
                wx = (x.get("weighted") or {}).get(lab, {}) if lab else {}
                wy = (y.get("weighted") or {}).get(lab, {}) if lab else {}
                L.append("| %.3f | %s | %s | %s; %s | %s; %s | %s; %s | %s; %s | %s; %s | %s; %s | %s; %s |" % (
                    t, ", ".join("%.3f (%+.4f)" % (q["R"], q["slope"]) for q in x["roots"]["dense"]) or "none", ", ".join("%.3f (%+.4f)" % (q["R"], q["slope"]) for q in y["roots"]["dense"]) or "none",
                    IO.fx(x["min_N_minus_absv"]["dense"], "%+.5f"), IO.fx(y["min_N_minus_absv"]["dense"], "%+.5f"), IO.fx(x["dust"]["centroid_energy_h0"], "%.3f"), IO.fx(y["dust"]["centroid_energy_h0"], "%.3f"),
                    IO.fx(x["dust"]["R_90"], "%.3f"), IO.fx(y["dust"]["R_90"], "%.3f"), IO.fx(x["collar"]["E_over_E0"], "%.6f"), IO.fx(y["collar"]["E_over_E0"], "%.6f"),
                    IO.fx(wx.get("N_out"), "%.5e"), IO.fx(wy.get("N_out"), "%.5e"), IO.fx(wx.get("N_above_m"), "%.5e"), IO.fx(wy.get("N_above_m"), "%.5e"), IO.fx(wx.get("E_direct"), "%+.5e"), IO.fx(wy.get("E_direct"), "%+.5e")))
            L.append("")
    L.append("- every number above is printed by the script from the stored slices and the records; the reading is the lead's.")
    d = os.path.join(a.out, READOUT)
    for hk in hists:
        sub = [r for r in rows if r["history"] == hk]
        if sub:
            with open(os.path.join(d, "%s_%s.md" % (TITLE, hk)), "w") as f:
                f.write("\n".join(L) + "\n")
    with open(os.path.join(d, "%s_all.json" % TITLE), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "histories": hists, "rows": rows}, f, default=IO.jdefault, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    IO.add_common_arguments(ap)
    add_arguments(ap)
    a = ap.parse_args(argv)
    a.readouts = READOUT
    IO.run(a)


if __name__ == "__main__":
    main()
