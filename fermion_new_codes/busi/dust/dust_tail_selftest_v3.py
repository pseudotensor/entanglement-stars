#!/usr/bin/env python
"""dust_tail_selftest_v3.py (busi lane, reviewer A): the selftest of dust_tail_v3.py's additions on small fixtures with known answers.
Items 1a to 1d: the per-half construction and the sector readouts (dust_tail_v3.outside_weight_halves, escaped_sector, sector_readouts, floors,
verdict) on the 96-site fixture of dust_tail_selftest_v2 (h_m of channel kappa 1 at its fields; cuts at sites 32 and 48): 1a the dimensions per
half equal the count of eigenvalues of V_h^dag Pi_Y V_h at or above 1 - eps_loc computed independently here, F is orthonormal, every column lies
inside one spectral half (F^dag P_- F = diag(half < 0)) and has outside weight >= 1 - eps_loc, both halves are populated; 1b on a random
Gaussian state the first route's formulas re-implemented inline (M = F^dag Phi f Phi^dag F, S_F = sum s(eig M), particles = tr(M - Pi)_++,
holes = -tr(M - Pi)_--, E_F = Re tr[(F^dag h F)(M - Pi)]) equal sector_readouts to 1e-12, the pinching bound holds, the hull's dimension is
the sum over halves of min(d_h, kept) with L orthonormal and reducing Pi; 1c closed forms with the sea the exact Killing vacuum and the dust 2
columns of F_+ with f = (1/2, 1): S_F = ln 2, n_+ = 3/2, n_- = 0, E_F = (1/2) hF_11 + hF_22, kept = 2, S_hull = ln 2, the vacuum identities at
roundoff, the verdict respected; 1d the floors and the verdict's 3 outcomes. Items 2a to 2c: the composer's nested-key read on a synthetic
first-route json in the engine's layout with known numbers. Items 3a to 3f: the whole program (run_main with --deviation) on the synthetic
slice of dust_tail_selftest_v2 with its synthetic run json and a synthetic first-route json in the engine's layout that carries this route's
own v3 numbers under the engine's nested keys. Item 4: the composer recomposes the same table rows from the outputs alone. Writes only under
TMPDIR."""
import json
import math
import os
import tempfile

import numpy as np
import scipy.linalg as sla

import dust_tail_ops_v1 as OPS
import dust_tail_selftest_v2 as ST2
import dust_tail_table_v3 as TBL3


def s_fn(x):
    x = min(max(float(x), 0.0), 1.0)
    out = 0.0
    if x > 0.0:
        out -= x * math.log(x)
    if x < 1.0:
        out -= (1.0 - x) * math.log(1.0 - x)
    return out


def synthetic_engine_json(path):
    """a first-route json in engine_massive_channel_D_v2.py's layout with known numbers: 3 channels, regions '16' and '24' (channel 2 writes '24.0'),
    the nested hull and sector dicts, the flat region keys the engine writes, and the slice's weighted sums formed from the per-channel values."""
    chans = []
    wsum = {"16": {}, "24": {}}
    for kappa in (1, 2, 3):
        w = 16.0 * kappa
        regions = {}
        for cut, dim in (("16", 2046), ("24", 1022)):
            S_hull = 1e-3 * kappa
            S_F = S_hull * (1.0 + 1e-7)
            part = 2.5e-4 * kappa
            holes = 1.2e-4 * kappa
            pPm = 2.533e-4 * kappa
            hPm = 1.161e-4 * kappa
            pd = 1.0e-5 * kappa
            hd = -2.0e-6 * kappa
            dw = pd - hd
            reg = {"hull": {"S_hull": S_hull, "hull_dim": dim, "kept": dim - 72, "dropped": 72, "dropped_bound": 7e-4 * kappa, "vacuum_check": 8e-15, "svd_fallback": False},
                   "sector": {"0.01": {"dim": dim, "dim_plus": dim // 2, "dim_minus": dim // 2, "S_F": S_F, "E_F": 1.6e-4 * kappa, "particles": part, "holes": holes, "F_orthonormality": 1.6e-14,
                                       "outside_weight_min_plus": 0.9928, "outside_weight_min_minus": 0.9927, "hulls": {"1e-09": {"S_hull": S_hull, "hull_dim": dim}}},
                              "0.001": {"dim": dim - 2, "dim_plus": dim // 2 - 1, "dim_minus": dim // 2 - 1, "S_F": S_F * 0.999}},
                   "S_F": S_F, "E_F": 1.6e-4 * kappa, "sector_particles": part, "sector_holes": holes, "S_F_loc_control": S_F * 0.999, "S_hull_loc_control": S_F * 0.999,
                   "particles_Pm": pPm, "holes_Pm": hPm, "particles_dust": pd, "holes_dust": hd, "dust_weight": dw, "region_S_excess": S_hull * 0.99, "E_out_dust": 1e-5 * kappa}
            regions["24.0" if (cut == "24" and kappa == 2) else cut] = reg
            acc = wsum[cut]
            for q, val in (("S_hull", S_hull), ("S_F", S_F), ("sector_particles", part), ("sector_holes", holes), ("particles_Pm", pPm), ("holes_Pm", hPm), ("dust_weight", dw),
                           ("E_F", 1.6e-4 * kappa), ("region_S_excess", S_hull * 0.99), ("dropped_bound", 7e-4 * kappa), ("S_F_loc_control", S_F * 0.999), ("S_hull_loc_control", S_F * 0.999)):
                acc[q] = acc.get(q, 0.0) + w * val
        chans.append({"kappa": kappa, "w": w, "stored_orthonormality": 5e-13 * kappa, "loewdin_orthonormality": 3e-15, "regions": regions})
    J = {"script": "synthetic", "args": {"eps_loc": 0.01, "eps_loc_control": 0.001, "eps_rank": 1e-9, "eps_control": 1e-6},
         "slices": [{"T_over_rs": 40.0, "E_G": 106.2, "dtype": "complex64", "weighted": wsum, "channels": chans}]}
    json.dump(J, open(path, "w"))
    return J


def synthetic_first_route_v3(d, row, eps_loc):
    """a first-route D v2 json in the engine's layout filled with this route's own v3 numbers (regions[cut].hull.S_hull := this S_F, sector dims and
    counts := this route's, the block-basis keys := version 2's dev entries), so the composer's nested read must print them beside with zero
    difference; the tail json comes from dust_tail_selftest_v2.synthetic_first_route_v2."""
    chans = []
    wsum = {}
    v3 = row["v3"]
    for c, c2, c3 in zip(row["channels"], row["v2"]["channels"], v3["channels"]):
        regions = {}
        for key in ("16.0", "24.0"):
            cutk = key.rstrip("0").rstrip(".")
            s = c3["sector"][key]
            sc = (c3.get("sector_control") or {}).get(key) or {}
            dv = c2["dev"][key]
            hull = s["hulls"]["%g" % v3["eps_rank"]]
            nd = c["n_dust"]
            reg = {"hull": {"S_hull": s["S_hull"], "hull_dim": s["hull_dim"], "kept": hull["kept"], "dropped": hull["dropped"], "dropped_bound": hull["dropped_bound"], "vacuum_check": hull["vacuum_check"]},
                   "sector": {"%g" % eps_loc: {"dim": s["dim"], "dim_plus": s["dim_plus"], "dim_minus": s["dim_minus"], "S_F": s["S_F"], "E_F": s["E_F"], "particles": s["n_plus"], "holes": s["n_minus"],
                                               "F_orthonormality": s["F_orthonormality"], "outside_weight_min_plus": s["outside_weight_min_plus"], "outside_weight_min_minus": s["outside_weight_min_minus"]}},
                   "S_F": s["S_F"], "E_F": s["E_F"], "sector_particles": s["n_plus"], "sector_holes": s["n_minus"], "S_F_loc_control": sc.get("S_F"), "S_hull_loc_control": sc.get("S_F"),
                   "particles_Pm": dv["particles_dev"], "holes_Pm": dv["holes_dev"], "particles_dust": dv["particles_dust_block"], "holes_dust": -dv["holes_dust_block"],
                   "dust_weight": (c["beyond"][key]["N"] if nd else 0.0), "region_S_excess": dv["region_excess"], "E_out_dust": (c["beyond"][key]["E_sym"] if nd else 0.0),
                   "E_particles_dust": (c["beyond"][key]["E_particles"] if nd else 0.0), "E_holes_dust": (c["beyond"][key]["E_holes"] if nd else 0.0)}   # busi fix 1: the engine's regions always carry these (Dv2_ext100_b9: all 10 channels, both cuts); version 1's composer reads E_particles_dust
            regions[cutk] = reg
            acc = wsum.setdefault(cutk, {})
            for q in ("S_hull", "S_F", "sector_particles", "sector_holes", "particles_Pm", "holes_Pm", "dust_weight", "E_F", "region_S_excess"):
                acc[q] = acc.get(q, 0.0) + c["w"] * float(reg["hull"]["S_hull"] if q == "S_hull" else reg[q])
        chans.append({"kappa": c["kappa"], "w": c["w"], "stored_orthonormality": c3["stored_orthonormality"], "loewdin_orthonormality": None, "regions": regions})
    J = {"script": "synthetic v3", "args": {"eps_loc": eps_loc, "eps_loc_control": v3["eps_loc_control"], "eps_rank": v3["eps_rank"], "eps_control": v3["eps_control"]},
         "slices": [{"T_over_rs": row["T_over_rs"], "weighted": wsum, "channels": chans}]}
    pth = os.path.join(d, "dv2_v3.json")
    json.dump(J, open(pth, "w"), default=float)
    return pth


def selftest(MAIN):
    rng = np.random.default_rng(23)
    ok = True

    def chk(name, cond, val):
        nonlocal ok
        ok = ok and bool(cond)
        print("%s %s (%s)" % ("PASS" if cond else "FAIL", name, val))

    m = 0.2
    n = 96
    R, N, v = ST2.fixture_fields(n)
    hm = OPS.killing_h(R, m, 1, N, v)
    Hd = hm.toarray()
    e, V = sla.eigh(Hd)
    eps_loc = 0.1
    # 1a the per-half construction
    good = True
    both = True
    detail = []
    for p in (32, 48):
        r0 = 2 * p
        halves = MAIN.outside_weight_halves(V, e, r0)
        F, half, info = MAIN.escaped_sector(halves, eps_loc)
        for hsign, sel in ((1, e >= 0), (-1, e < 0)):
            Vh = V[:, sel]
            w_ind = np.linalg.eigvalsh(Vh[r0:].conj().T @ Vh[r0:])
            good = good and int(np.sum(w_ind >= 1.0 - eps_loc)) == info["dim_plus" if hsign > 0 else "dim_minus"]
        wout = np.sum(np.abs(F[r0:]) ** 2, axis=0)
        Pm = V[:, e < 0]
        FPF = (F.conj().T @ Pm) @ (Pm.conj().T @ F)
        good = good and MAIN.gram_defect(F) < 1e-11 and bool(np.all(wout >= 1.0 - eps_loc - 1e-12)) and np.max(np.abs(FPF - np.diag((half < 0).astype(float)))) < 1e-11 and info["dim"] == F.shape[1]
        both = both and info["dim_plus"] > 0 and info["dim_minus"] > 0 and info["dim"] < 2 * n
        detail.append((p, info["dim_plus"], info["dim_minus"], round(float(wout.min()), 6)))
    chk("1a per-half construction (cuts at sites 32 and 48): the dims per half equal the independent count of eigenvalues of V_h^dag Pi_Y V_h at or above 1 - eps_loc; F orthonormal; every column inside one half (F^dag P_- F = diag(half < 0)) with outside weight >= 1 - eps_loc; both halves populated",
        good and both, detail)
    # 1b the first route's formulas inline on a random Gaussian state
    r0 = 64
    halves = MAIN.outside_weight_halves(V, e, r0)
    F, half, info = MAIN.escaped_sector(halves, eps_loc)
    Vn = halves[-1][0]
    Phi_r = ST2.random_slater(rng, 2 * n, 60)
    f_r = rng.uniform(0.2, 1.0, 60)
    sec = MAIN.sector_readouts(F, half, Phi_r, f_r, 0, hm, Vn, 1e-9, 1e-6, with_hull=True)
    Mref = F.conj().T @ ((Phi_r * f_r[None, :]) @ (Phi_r.conj().T @ F))
    Mref = 0.5 * (Mref + Mref.conj().T)
    Pi = np.diag((half < 0).astype(float))
    MDref = Mref - Pi
    S_ref = sum(s_fn(x) for x in np.linalg.eigvalsh(Mref))
    hFref = F.conj().T @ (Hd @ F)
    hFref = 0.5 * (hFref + hFref.conj().T)
    E_ref = float(np.real(np.sum(hFref * MDref.T)))
    pos = half > 0
    negm = half < 0
    part_ref = float(np.real(np.trace(MDref[np.ix_(pos, pos)])))
    holes_ref = float(-np.real(np.trace(MDref[np.ix_(negm, negm)])))
    hull = sec["hulls"]["1e-09"]
    chk("1b random Gaussian state: the first route's formulas inline (S_F = sum s(eig F^dag C F); particles = tr(M - Pi)_++; holes = -tr(M - Pi)_--; E_F = Re tr[(F^dag h F)(M - Pi)]) equal sector_readouts to 1e-12; "
        "S_F <= block bound <= d_F h2(n_F/d_F); hull dim = sum over halves of min(d_h, kept), L orthonormal and reducing Pi; S_hull = S_F when every eigenvector is kept",
        abs(sec["S_F"] - S_ref) < 1e-12 and abs(sec["E_F"] - E_ref) < 1e-10 and abs(sec["n_plus"] - part_ref) < 1e-12 and abs(sec["n_minus"] - holes_ref) < 1e-12 and sec["S_F"] <= sec["bound_F_blocks"] + 1e-12
        and sec["bound_F_blocks"] <= sec["bound_F"] + 1e-12 and sec["dim"] == F.shape[1] and hull["hull_dim"] == min(sec["dim_plus"], hull["kept"]) + min(sec["dim_minus"], hull["kept"])
        and hull["vacuum_check"] < 1e-10 and hull["hull_orthonormality"] < 1e-10 and (hull["kept"] < sec["dim"] or abs(hull["S_hull"] - sec["S_F"]) < 1e-9),
        (sec["S_F"], S_ref, sec["E_F"], E_ref, sec["n_plus"], part_ref, sec["n_minus"], holes_ref, hull["kept"], hull["hull_dim"]))
    # 1c closed forms: the exact vacuum as the sea, 2 columns of F_+ as the dust
    sea = V[:, e < 0]
    ip = np.flatnonzero(half > 0)
    i1, i2 = int(ip[0]), int(ip[len(ip) // 2])
    dustc = F[:, [i1, i2]].copy()
    fc = np.array([0.5, 1.0])
    Phi_c = np.concatenate([sea, dustc], axis=1)
    Pf_c = np.concatenate([np.ones(sea.shape[1]), fc])
    sec_c = MAIN.sector_readouts(F, half, Phi_c, Pf_c, sea.shape[1], hm, Vn, 1e-9, 1e-6, with_hull=True)
    hF11 = float(np.real(dustc[:, 0].conj() @ (Hd @ dustc[:, 0])))
    hF22 = float(np.real(dustc[:, 1].conj() @ (Hd @ dustc[:, 1])))
    E_exp = 0.5 * hF11 + 1.0 * hF22
    hull_c = sec_c["hulls"]["1e-09"]
    fl_c = MAIN.floors(sec_c["dim"], MAIN.gram_defect(Phi_c), max(sec_c["F_orthonormality"], sec_c["vacuum_plus_overlap_max"], sec_c["vacuum_minus_identity_dev"]))
    vd_c = MAIN.verdict(sec_c["S_F"], sec_c["bound_F"], sec_c["n_F"], fl_c)
    chk("1c closed forms (sea = the exact Killing vacuum; dust = 2 columns of F_+ with f 1/2 and 1): S_F = ln 2; n_+ = 3/2 (all the dust's, the pairs' 0); n_- = 0; E_F = hF_11/2 + hF_22; kept 2; hull dim = min(d_+, 2) + min(d_-, 2); S_hull = ln 2; S_F,dust = ln 2; "
        "F_+^dag P_- F_+ and F_-^dag P_- F_- - 1 at roundoff; the verdict respected",
        abs(sec_c["S_F"] - math.log(2)) < 1e-9 and abs(sec_c["n_plus"] - 1.5) < 1e-10 and abs(sec_c["n_minus"]) < 1e-9 and abs(sec_c["n_plus_dust"] - 1.5) < 1e-10 and abs(sec_c["n_plus_pairs"]) < 1e-10
        and abs(sec_c["n_minus_pairs"]) < 1e-9 and abs(sec_c["n_minus_dust_occ"]) < 1e-12 and abs(sec_c["E_F"] - E_exp) < 1e-9 and hull_c["kept"] == 2
        and hull_c["hull_dim"] == min(sec_c["dim_plus"], 2) + min(sec_c["dim_minus"], 2) and abs(hull_c["S_hull"] - math.log(2)) < 1e-9 and hull_c["vacuum_check"] < 1e-10
        and sec_c["vacuum_plus_overlap_max"] < 1e-10 and sec_c["vacuum_minus_identity_dev"] < 1e-10 and abs(sec_c["S_F_dust"] - math.log(2)) < 1e-9 and vd_c == "respected" and sec_c["S_F"] <= sec_c["bound_F"],
        (sec_c["S_F"], sec_c["n_plus"], sec_c["n_minus"], sec_c["E_F"], E_exp, hull_c["kept"], hull_c["hull_dim"], hull_c["S_hull"], vd_c))
    # 1d the floors and the verdict
    fl = MAIN.floors(100, 1e-13, 2e-14)
    chk("1d floors and verdict: eta the larger defect, n_floor = d eta, S_floor = d h2(eta); 'below floor' when S or n is at its floor, 'EXCEEDED' above the floors when S > bound, 'respected' otherwise",
        abs(fl["eta"] - 1e-13) < 1e-30 and abs(fl["n_floor"] - 1e-11) < 1e-25 and abs(fl["S_floor"] - 100 * s_fn(1e-13)) < 1e-20 and MAIN.verdict(1e-10, 1e-11, 1e-12, fl) == "below floor"
        and MAIN.verdict(0.5, 0.4, 0.3, fl) == "EXCEEDED" and MAIN.verdict(0.3, 0.4, 0.3, fl) == "respected" and MAIN.verdict(1e-9, 1e-12, 1e-13, fl) == "below floor", fl)
    # 2 the nested-key read
    d = tempfile.mkdtemp(prefix="dust_selftest_v3_")
    jpath = os.path.join(d, "synthetic_engine.json")
    synthetic_engine_json(jpath)
    fr = TBL3.read_first_route_raw(jpath)
    frs = fr.get(40.0)
    c1 = TBL3.first_channel_sector(frs["channels"][1], "16", 0.01, 1e-9) if frs else None
    chk("2a per-channel nested read: S_hull from regions.16.hull.S_hull, dim and dim_plus from regions.16.sector.0.01, the sector counts, n_F, the bound d_F h2(n_F/d_F), the pairs' derived counts, the dust's net, the control S_F and the channel's stored orthonormality",
        c1 is not None and abs(c1["S_hull"] - 1e-3) < 1e-15 and c1["d_F"] == 2046 and c1["d_plus"] == 1023 and abs(c1["sector_particles"] - 2.5e-4) < 1e-18 and abs(c1["n_F"] - 3.7e-4) < 1e-15
        and abs(c1["bound_F"] - 2046 * s_fn(3.7e-4 / 2046)) < 1e-15 and abs(c1["pairs_particles"] - 2.433e-4) < 1e-15 and abs(c1["pairs_holes"] - 1.181e-4) < 1e-15 and abs(c1["dust_net"] - 8e-6) < 1e-18
        and abs(c1["S_F_loc_control"] - 1e-3 * (1 + 1e-7) * 0.999) < 1e-15 and c1["d_F_loc_control"] == 2044 and abs(c1["stored_orthonormality"] - 5e-13) < 1e-25 and c1["kept"] == 1974 and abs(c1["ratio"] - 1e-3 / c1["bound_F"]) < 1e-12,
        c1 and (c1["S_hull"], c1["d_F"], c1["n_F"], c1["bound_F"], c1["pairs_particles"], c1["pairs_holes"]))
    fs = TBL3.first_slice_sector(frs, "16") if frs else None
    bound_inline = sum(16.0 * k * 2046 * s_fn(3.7e-4 * k / 2046) for k in (1, 2, 3))
    dw_sum = sum(16.0 * k * 1.2e-5 * k for k in (1, 2, 3))
    chk("2b weighted sums (w = the json's w) equal the json's own weighted keys (difference 0 for S_hull, S_F, the sector counts, the block counts and the dust weight); the bound is the per-channel sum of 16k d_F h2(n_F,k/d_F,k); d_F unweighted and weighted; the dust's count permits d_w h2(N/d_w)",
        fs is not None and all(abs(fs["sum_minus_json"][q]) < 1e-12 for q in ("S_hull", "S_F", "sector_particles", "sector_holes", "particles_Pm", "holes_Pm", "dust_weight")) and abs(fs["bound_F"] - bound_inline) < 1e-12
        and fs["d_F_unw"] == 3 * 2046 and fs["d_F_w"] == 96 * 2046 and abs(fs["dust_weight"] - dw_sum) < 1e-15 and abs(fs["dust_permits_w_form"] - 96 * 2046 * s_fn(dw_sum / (96 * 2046))) < 1e-12
        and abs(fs["ratio"] - fs["S_hull"] / bound_inline) < 1e-12 and fs["ratio_min"] <= fs["ratio_max"] and fs["n_channels"] == 3,
        fs and (fs["S_hull"], fs["json_weighted"]["S_hull"], fs["bound_F"], bound_inline, fs["d_F_unw"], fs["d_F_w"]))
    c2 = TBL3.first_channel_sector(frs["channels"][2], "24", 0.01, 1e-9) if frs else None
    fs24 = TBL3.first_slice_sector(frs, "24") if frs else None
    chk("2c the region key written as '24.0' and the cut asked as '24' (and the eps_loc key '0.01' asked as %g) are read; the slice's 3 channels all enter beyond 24",
        c2 is not None and c2["d_F"] == 1022 and abs(c2["S_hull"] - 2e-3) < 1e-15 and fs24 is not None and fs24["n_channels"] == 3 and fs24["d_F_unw"] == 3 * 1022, (c2 and c2["d_F"], fs24 and fs24["d_F_unw"]))
    # 3 the whole program on the synthetic slice
    path, arrs, known = ST2.synthetic_slice_v2(d, m, rng)
    row, arrays = MAIN.process_slice_v3(path, m, [16.0, 24.0], [8.0, 16.0, 24.0], "dust", True, eps_loc, 1e-3, 1e-9, 1e-6, True, log=lambda s: None)
    v3 = row["v3"]
    w16 = v3["weighted"]["16.0"]
    w24 = v3["weighted"]["24.0"]
    sw = sum(c3["w"] * c3["sector"]["16.0"]["S_F"] for c3 in v3["channels"])
    chk("3a process_slice_v3 on the fixture (2 channels, cuts at sites 32 and 48, --deviation, --loewdin): both cuts filled for every channel; the weighted S_F is the sum of w times the channels' S_F; d_F unw is the sum of dims; the verdict 'respected' in every channel beyond 16 and never 'EXCEEDED' beyond 24; "
        "the v3 eigenvalues equal version 2's to 1e-11; the Loewdin control's S_F differs from the as-stored one by at most the weighted S floor sum w d h2(eta) (the fixture's columns orthonormal to about 2e-13); the localized set's floor-aware verdict present",
        all(c3["sector"].get(k) is not None for c3 in v3["channels"] for k in ("16.0", "24.0")) and abs(w16["S_F"] - sw) < 1e-12 and w16["d_F_unw"] == sum(c3["sector"]["16.0"]["dim"] for c3 in v3["channels"])
        and w16["all_respected"] and w24["verdicts"]["EXCEEDED"] == 0 and all((c3.get("e_vs_v2_max_dev") or 0.0) < 1e-11 for c3 in v3["channels"]) and "loewdin" in w16 and abs(w16["loewdin"]["S_F_minus_stored"]) <= sum(c3["w"] * c3["sector"]["16.0"]["dim"] * s_fn(c3["stored_orthonormality"]) for c3 in v3["channels"])   # busi fix 2: the fixture's columns are orthonormal to 2e-13, not exactly; the bound is the code's own S floor, sum of w d h2(eta)
        and all(c3["loc_verdict"].get(k) is not None and c3["loc_verdict"][k]["verdict"] in MAIN.VERDICTS for c3 in v3["channels"] for k in ("16.0", "24.0")),
        (w16["d_F_unw"], w16["S_F"], w16["verdicts"], w24["verdicts"], v3.get("max_e_vs_v2"), w16["loewdin"]["S_F_minus_stored"] if "loewdin" in w16 else None))
    run = ST2.synthetic_run_v2(d, arrs, m)
    tail, _flat = ST2.synthetic_first_route_v2(d, row)
    dv2 = synthetic_first_route_v3(d, row, eps_loc)
    out = os.path.join(d, "out")
    a = MAIN.default_args(slices=path, m=m, out_dir=out, tag="selftest_v3", run=run, first_route_tail_json=tail, first_route_dv2_jsons=dv2, deviation=True, eps_loc=eps_loc)
    rows = MAIN.run_main(a, log=lambda s: None)
    files = [os.path.join(out, "selftest_v3_%s" % s) for s in ("T0064.00.json", "T0064.00.npz", "all.json", "table.md", "table_v2.md", "table_v3.md")]
    chk("3b run_main writes the per-slice json and npz, the all json, version 1's table, version 2's table and the v3 table", all(os.path.exists(x) for x in files), files)
    md3 = open(files[-1]).read()
    r0w = rows[0]["v3"]["weighted"]["16.0"]
    sF = "%.6e" % r0w["S_F"]
    chk("3c the v3 table: V3-A prints this route's weighted S_F and the synthetic first route's S_hull (the same number) beside it; the verdict column; V3-B carries the first route's clock; the V2-F header is relabelled to version 2's localized set",
        md3.count(sF) >= 2 and "respected" in md3 and "## V3-B" in md3 and ("| %.3f | 16 | dv2_v3.json" % rows[0]["T_over_rs"]) in md3 and "Version 2's localized set F_loc" in md3 and "## V3-D" in md3 and "## V3-E" in md3,
        (sF, md3.count(sF)))
    Zo = np.load(files[1])
    need = ("w_half_plus_16_1", "w_half_minus_24_2", "sec_half_16_1", "sec_M_diag_16_1", "sec_M_diag_dust_24_1", "sec_mu_F_16_2", "sec_hF_diag_16_1", "sec_lam_MD_24_2", "sec_hull_sv_16_1", "e_1", "nu_dust_16_1", "mu_F_16_1")
    chk("3d the npz holds the v3 per-mode arrays (both halves' outside-weight spectra, the half labels, M's diagonal and eigenvalues, the dust's diagonal, diag(F^dag h F), the eigenvalues of M - Pi, the hull's singular values) beside version 1's and 2's",
        all(x in Zo.files for x in need) and Zo["sec_half_16_1"].shape[0] == rows[0]["v3"]["channels"][0]["sector"]["16.0"]["dim"], [x for x in need if x not in Zo.files])
    ok_floor = True
    for c3 in rows[0]["v3"]["channels"]:
        for k in ("16.0", "24.0"):
            lv = c3["loc_verdict"][k]
            dvk = [c for c in rows[0]["v2"]["channels"] if c["kappa"] == c3["kappa"]][0]["dev"][k]
            eta = max(c3["stored_orthonormality"], dvk["vacuum_plus_overlap_max"], dvk["vacuum_minus_identity_dev"])
            ok_floor = ok_floor and abs(lv["n_floor"] - dvk["d_F"] * eta) < 1e-25 and abs(lv["S_floor"] - dvk["d_F"] * s_fn(eta)) < 1e-20 and lv["d_F"] == dvk["d_F"] and lv["S_F"] == dvk["S_F"]
    chk("3e version 2's localized set in the json: its floor-aware verdict uses its own dev entries (d_F, S_F, bound, the vacuum identities) and the channel's stored Gram defect, n_floor = d_F eta and S_floor = d_F h2(eta)", ok_floor, "recomputed from the json")
    lifted_line = [ln for ln in md3.splitlines() if ln.startswith("- regions lifted for version 2's composer")]
    chk("3f the first route's nested keys are lifted for version 2's composer (its V2-F first-route S_hull column prints): lifted regions > 0",
        bool(lifted_line) and not lifted_line[0].rstrip().endswith(": 0"), lifted_line[0] if lifted_line else "no provenance line")
    # 4 the composer alone
    out2, md3b = TBL3.compose_from_outputs(out, "selftest_v3", tail, dv2, "", out_path=os.path.join(d, "recomposed_v3.md"))
    same = [ln for ln in md3.splitlines() if ln.startswith("|")] == [ln for ln in md3b.splitlines() if ln.startswith("|")]
    chk("4 dust_tail_table_v3 recomposes the same table rows from the outputs alone", same and os.path.exists(out2), out2)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok
