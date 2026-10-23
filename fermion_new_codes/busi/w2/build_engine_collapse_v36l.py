#!/usr/bin/env python
"""build_engine_collapse_v36l.py (reviewer A for the busi lane; W2 dg ruling v3): writes engine_collapse_v36l.py from engine_collapse_v36k.py
(the file build_engine_collapse_v36k.py writes) by 36 exact string replacements in that builder's form. Each OLD is copied exactly from
engine_collapse_v36k.py and must occur exactly once (else the build aborts); each NEW contains inserted or changed text so that replacing every
NEW by its OLD restores the source byte for byte (checked here). Every inserted or changed code line carries the marker "v36l attr" (a trailing
comment; the docstring's lines "[v36l attr]"; the review's B1 lines "v36l attr (review B1)"). The new options --dg-chain-trigger {on, off} (default
on) and --dg-attr {off, on} (default off) are read into DGK and used only under --dg-chart frozen, so the defaults are version 36k bitwise and the
json's "args" dict alone gains the keys (engine_collapse_v36l_selftest.py --build-check, --unit, --fixture). Under --dg-chart frozen version 36l
differs from version 36k by: B1 (gate 1's baseline E_start by the endpoint's rule at s = 0, the anchor's identity check); the per-iterate count of
node-channels whose tracked set differs from the anchor's; the chord's switches (the tracked set's index masks compared between consecutive chord
points, each difference with its levels, gaps, anchor weights, two-level gap_min estimate, s* and first-order jump -w gap_min, and under --dg-attr
on the chart's charge with the other point's mask at both points); r_chain split into the switching bin and the remainder in the sub-step entry,
the step entry, the sub-step line, the step line and the record line; and --dg-chain-trigger off, under which gate 1 is printed but does not halve
the sub-step (W2_dg_ruling_v3.md: r_chain is the chart's own charge change along the chord, not a quadrature defect that sub-steps remove).
Usage: python build_engine_collapse_v36l.py [engine_collapse_v36k.py path] [engine_collapse_v36l.py path]   (defaults: beside this script)"""
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36k.py (reviewer A for the busi lane; W2 dg ruling v2): version 36k = version 36f with one option, off by default, so that the defaults are [v36k chart]''',
r'''"""engine_collapse_v36l.py (reviewer A for the busi lane; W2 dg ruling v3): version 36l = version 36k with the review's edit B1 and two options, off by default, so that the [v36l attr]
defaults are version 36k bitwise in every record number, printed line and stored array (the json's "args" dict gains dg_chain_trigger and dg_attr): [v36l attr]
B1 (the independent review of v36k): gate 1's baseline is E_start, the chart's charge at C_0 by the endpoint's own rule (--dg-end-passes quasi-static evaluations from the [v36l attr]
    accepted fields with the anchor), stored and printed as E_start/E_rec - 1 (the solve's floor at a converged record whose set the anchor reproduces: the anchor's identity check). [v36l attr]
--dg-chain-trigger off: gate 1 (r_chain = E_end - E_start on the chart) is measured and printed but no longer halves the sub-step. W2_dg_ruling_v3.md: on the live log r_chain is [v36l attr]
    real, the chart's own charge change along the chord (the maximum-overlap set switching eigenvectors at avoided crossings inside the chord, a jump of the chart's charge at each [v36l attr]
    switch, and the reference's response near the crossings that three Gauss nodes do not integrate); it does not fall with the sub-step length, the endpoint estimate is confirmed by [v36l attr]
    the record solve to 5e-8, and sub-stepping multiplies the cost by ten without removing it. The default on is version 36k's rule. [v36l attr]
The chord's switches (always under --dg-chart frozen; the extra solves under --dg-attr on): between consecutive chord points (the anchor at s = 0, the three Gauss nodes at [v36l attr]
    convergence, the endpoint at s = 1) the tracked set's index masks are compared per channel (dgk_attr_v36l); where they differ, the levels that left and entered the set are paired [v36l attr]
    in index order (the count is held) with the eigenvalue gaps e_new - e_old and the anchor weights of both levels at both points, the two-level estimate of the avoided crossing's [v36l attr]
    minimum gap at each point, gap_min = gap x 2 sqrt(w (1 - w)) with w the set member's anchor weight (exact for one anchor level mixing with one other level; zero for levels that [v36l attr]
    cross in energy order without mixing, a relabeling of indices), s* (the weight-crossing of the level that left, interpolated between the points) and the first-order jump [v36l attr]
    -w_channel gap_min of E_rec (dgk_attr_pairs_v36l); under --dg-attr on the chart's charge is also computed at both points with the other point's mask in the differing channels [v36l attr]
    (dgk_alt_v36l: one eigendecomposition set and one field solve each; exact to second order in the fields' response since the charge's first variation in the fields vanishes [v36l attr]
    at the self-consistent point), D_P = E_P(new mask) - E_P and D_Q = E_Q - E_Q(old mask), and the response beyond first order D + w gap at each point. The sub-step's entry gains [v36l attr]
    attribution (the list of differences), switch_sum_over_E_rec (the first-order jumps summed), n_switch, rest_over_E_rec (r_chain less the switching bin), attribution_solves and [v36l attr]
    vs_anchor_hist (per iterate the node-channels whose set differs from the anchor's, also printed on every iterate: version 36k printed only the changes between iterates, zero [v36l attr]
    at the first iterate by construction); the step's entry gains E_start, E_start_minus_E_rec_over_E_rec, start_passes, track_start (B1), switch_sum_over_E_rec, n_switch and [v36l attr]
    rest_over_E_rec; the sub-step, step and record lines print the switching bin and the remainder. dgk_track_v36k(..., return_w=True) returns the levels' anchor weights as well, [v36l attr]
    dgk_node_v36k returns an eighth value (per channel the eigenvalues and the anchor weights as numpy) and its kept refs carry the eigenvalues as a fifth element, as does the anchor [v36l attr]
    of dgk_anchor_v36k. The engine's own conventions (the record solve, its mixer, its continuation) are unchanged. Every changed code line of this file is marked "v36l attr". [v36l attr]
Version 36k's docstring follows. [v36l attr]
engine_collapse_v36k.py (reviewer A for the busi lane; W2 dg ruling v2): version 36k = version 36f with one option, off by default, so that the defaults are [v36k chart]'''),
("docstring keys",
r'''    keys (E_pred = E_rec + the sub-steps' defects) and gains chart, level, nsub, attempts, substeps, E_end, r_chain_over_E_rec, chart_disagreement_total (the levels the [v36k chart]''',
r'''    keys (E_pred = E_rec + the sub-steps' defects) and gains chart, level, nsub, attempts, substeps, E_start (the chart's charge at C_0 by the endpoint's rule, gate 1's [v36k chart; v36l attr]
    baseline; E_rec to the solve's floor at a converged record, the anchor's identity check), E_end, r_chain_over_E_rec (E_end - E_start), switch_sum_over_E_rec, n_switch and [v36k chart; v36l attr]
    rest_over_E_rec (the switching bin of r_chain and its remainder; each sub-step's attribution lists the switches), chart_disagreement_total (the levels the [v36k chart; v36l attr]'''),
("module state",
r'''DGK = {"on": False, "mix": "plain", "m": 4, "beta": 1.0, "max_change": 2e-2, "sub_max": 3, "chain_tol": 1e-8, "end_passes": 2, "level": 0, "level_min": 0, "sub_len": None}   # v36k chart: the chart-frozen step's state (--dg-chart frozen)''',
r'''DGK = {"on": False, "mix": "plain", "m": 4, "beta": 1.0, "max_change": 2e-2, "sub_max": 3, "chain_tol": 1e-8, "end_passes": 2, "level": 0, "level_min": 0, "sub_len": None}   # v36k chart: the chart-frozen step's state (--dg-chart frozen)
DGK.update({"chain_trigger": True, "attr": False})   # v36l attr: gate 1 as a trigger (version 36k) or a measurement; the chord-switch attribution's extra solves'''),
("track signature",
r'''def dgk_track_v36k(e, U, anchor):   # v36k chart''',
r'''def dgk_track_v36k(e, U, anchor, return_w=False):   # v36k chart; v36l attr: return_w appends the levels' anchor weights (numpy) to the return'''),
("track return",
r'''    return ref, mask, info   # v36k chart''',
r'''    return (ref, mask, info, w) if return_w else (ref, mask, info)   # v36k chart; v36l attr'''),
("anchor spectrum",
r'''        out[ch.kappa] = (ref, int(ref.shape[1]), np.asarray(mask, bool).copy(), {"n_pocket": cinfo.get("n_pocket"), "nsub": cinfo.get("nsub"), "min_occupied_weight": cinfo.get("min_occupied_weight"), "verified": cinfo.get("verified")})   # v36k chart''',
r'''        out[ch.kappa] = (ref, int(ref.shape[1]), np.asarray(mask, bool).copy(), {"n_pocket": cinfo.get("n_pocket"), "nsub": cinfo.get("nsub"), "min_occupied_weight": cinfo.get("min_occupied_weight"), "verified": cinfo.get("verified")}, e.detach().cpu().numpy().copy())   # v36k chart; v36l attr: the eigenvalues at the anchor's fields as the fifth element (the chord's first point)'''),
("node docstring",
r'''Returns (N_out, v_out, E_node, info, per-channel track info, masks, refs)."""   # v36k chart''',
r'''Returns (N_out, v_out, E_node, info, per-channel track info, masks, refs, spec); spec (version 36l) holds per channel the eigenvalues and the anchor weights as numpy, and the kept refs' tuples gain the eigenvalues as their fifth element."""   # v36k chart; v36l attr'''),
("node init",
r'''    eps = np.zeros(n); J = np.zeros(n); tinfo = {}; masks = {}; refs = {}   # v36k chart''',
r'''    eps = np.zeros(n); J = np.zeros(n); tinfo = {}; masks = {}; refs = {}; spec = {}   # v36k chart; v36l attr: spec = per channel (eigenvalues, anchor weights) as numpy'''),
("node track",
r'''            ref, mask, ti = dgk_track_v36k(e, U, anchor[ch.kappa])   # v36k chart''',
r'''            ref, mask, ti, w_ = dgk_track_v36k(e, U, anchor[ch.kappa], return_w=True)   # v36k chart; v36l attr
            spec[ch.kappa] = (e.detach().cpu().numpy().copy(), np.asarray(w_, float).copy())   # v36l attr'''),
("node refs",
r'''                refs[ch.kappa] = (ref, int(ref.shape[1]), mask, None)   # v36k chart''',
r'''                refs[ch.kappa] = (ref, int(ref.shape[1]), mask, None, spec[ch.kappa][0])   # v36k chart; v36l attr: the eigenvalues as the fifth element (the next sub-step's anchor is its chord's first point)'''),
("node return failed",
r'''        return None, None, None, info, tinfo, masks, refs   # v36k chart''',
r'''        return None, None, None, info, tinfo, masks, refs, spec   # v36k chart; v36l attr'''),
("node return",
r'''    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, tinfo, masks, refs   # v36k chart''',
r'''    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, tinfo, masks, refs, spec   # v36k chart; v36l attr'''),
("functions before dg_step",
r'''def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart''',
r'''def dgk_desc_v36l(a):   # v36l attr
    """version 36l: the options in words (the run label)."""   # v36l attr
    return ("gate 1 %s; the chord's switches (the tracked set's index masks compared between consecutive chord points: the anchor, the three nodes, the endpoint) %s, each with the "   # v36l attr
            "first-order jump -w gap_min from the two-level estimate of the anchor weights; gate 1's baseline E_start by the endpoint's rule (review B1)") % (   # v36l attr
        "halves the sub-step above the tolerance (version 36k)" if DGK["chain_trigger"] else "measured and printed only (no sub-step trigger)",   # v36l attr
        "measured as well by the chart's charge with the other point's mask at both points (two eigendecomposition sets and two solves per switch)" if DGK["attr"] else "estimated from the spectra alone")   # v36l attr


def dgk_attr_pairs_v36l(diff, specP, specQ, sP, sQ, wch, E_rec):   # v36l attr
    """version 36l (pure): the levels that left and entered the tracked set between two chord points P and Q, per channel in diff ({channel: (index mask at P, index mask at Q)}, masks that differ; the count is held, so the levels pair in index order), with the eigenvalue gaps e_new - e_old at both points, the anchor weights of both levels at both points, the two-level estimate of the avoided crossing's minimum gap at each point, gap_min = gap x 2 sqrt(w (1 - w)) with w the set member's anchor weight there (exact for one anchor level mixing with one other level; zero for levels that cross in energy order without mixing), the first-order jump -w_channel x gap_min of E_rec (the estimate from the more mixed point: the anchor's own weights are 1 and 0 by construction), and s*, the weight-crossing of the level that left the set interpolated linearly between the points (the midpoint when the weights do not bracket 1/2). Returns (pairs, s_star, jump_sum, first_order_at_P, first_order_at_Q)."""   # v36l attr
    pairs = []; s_num = 0.0; s_cnt = 0; jsum = 0.0; foP = 0.0; foQ = 0.0   # v36l attr
    for kap in sorted(diff, key=str):   # v36l attr
        mP = np.asarray(diff[kap][0], bool); mQ = np.asarray(diff[kap][1], bool)   # v36l attr
        old = np.flatnonzero(mP & ~mQ).tolist(); new = np.flatnonzero(mQ & ~mP).tolist()   # v36l attr
        eP = np.asarray(specP[kap][0], float); wP = np.asarray(specP[kap][1], float); eQ = np.asarray(specQ[kap][0], float); wQ = np.asarray(specQ[kap][1], float)   # v36l attr
        w_ch = float(wch.get(kap, 0.0))   # v36l attr
        for i_o, i_n in zip(old, new):   # v36l attr
            gP = float(eP[i_n] - eP[i_o]); gQ = float(eQ[i_n] - eQ[i_o])   # v36l attr
            woP, wnP, woQ, wnQ = float(wP[i_o]), float(wP[i_n]), float(wQ[i_o]), float(wQ[i_n])   # v36l attr
            mixP = max(0.0, min(1.0, woP)); mixQ = max(0.0, min(1.0, wnQ))   # v36l attr: the set members' anchor weights at the two points
            gmP = gP * 2.0 * float(np.sqrt(mixP * (1.0 - mixP))); gmQ = gQ * 2.0 * float(np.sqrt(mixQ * (1.0 - mixQ)))   # v36l attr
            gm = gmP if abs(gmP) >= abs(gmQ) else gmQ   # v36l attr
            if woP > 0.5 > woQ:   # v36l attr
                s_num += sP + (woP - 0.5) / (woP - woQ) * (sQ - sP); s_cnt += 1   # v36l attr
            jsum += -w_ch * gm / E_rec; foP += -w_ch * gP / E_rec; foQ += -w_ch * gQ / E_rec   # v36l attr
            pairs.append({"channel": str(kap), "old": int(i_o), "new": int(i_n), "gap_at_P": gP, "gap_at_Q": gQ, "weights_at_P": [woP, wnP], "weights_at_Q": [woQ, wnQ], "gap_min_estimates": [gmP, gmQ], "jump_first_order_over_E_rec": -w_ch * gm / E_rec})   # v36l attr
        if len(old) != len(new):   # v36l attr
            pairs.append({"channel": str(kap), "unpaired": len(old) - len(new)})   # v36l attr
    s_star = (s_num / s_cnt) if s_cnt else 0.5 * (sP + sQ)   # v36l attr
    return pairs, s_star, jsum, foP, foQ   # v36l attr


def dgk_alt_v36l(a, chs, R, base_s, Nf, vf, anchor, alt_masks):   # v36l attr
    """version 36l: the chart's charge at a chord point (densities base_s) at the fields (Nf, vf) with the tracked set replaced, in the channels of alt_masks, by the eigenvectors of that index mask (the other chord point's set): one eigendecomposition set and one field solve; the charge's first variation in the fields vanishes at the self-consistent point, so against the point's own charge the difference is exact to second order in the fields' response. Returns (E, solve info); (None, info) when the solve fails."""   # v36l attr
    torch = BK["torch"]   # v36l attr
    n = len(R); eig = eigh_all(chs, Nf, vf, bool(a.streams)); eps = np.zeros(n); J = np.zeros(n)   # v36l attr
    for ch in chs:   # v36l attr
        e, U = eig.pop(ch.kappa); hs_, Qs_, dd, Jd = base_s[ch.kappa]   # v36l attr
        if ch.kappa in alt_masks:   # v36l attr
            ref = U[:, torch.from_numpy(np.asarray(alt_masks[ch.kappa], bool)).to(U.device)]   # v36l attr
        else:   # v36l attr
            ref, _m, _i = dgk_track_v36k(e, U, anchor[ch.kappa])   # v36l attr
        eps += ch.w * ((hs_ - dens(ch.h, ref)) + dd); J += ch.w * ((Qs_ - dens(ch.Q, ref)) + Jd)   # v36l attr
        del e, U, ref   # v36l attr
    del eig   # v36l attr
    Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, None)   # v36l attr
    if info.get("failed"):   # v36l attr
        return None, info   # v36l attr
    return float(FLD.charge(info, a.g)), info   # v36l attr


def dgk_attr_v36l(a, chs, R, anchor, pts, E_rec, T_over_rs, j1, nsub):   # v36l attr
    """version 36l: the chord's switches. pts = the chord points in order (dicts with s, N, v, base, E, masks, spec: the anchor at s = 0, the three Gauss nodes at convergence, the endpoint); for consecutive points whose index masks differ in any channel the pairs, s* and the first-order jump (dgk_attr_pairs_v36l) and, under --dg-attr on, the chart's charge at each point with the other point's mask (dgk_alt_v36l): D_P = E_P(new mask) - E_P, D_Q = E_Q - E_Q(old mask), the response beyond first order D + w gap at each point. Prints one line per difference. Returns (records, the first-order jumps summed over E_rec, solves)."""   # v36l attr
    wch = {ch.kappa: float(ch.w) for ch in chs}; recs = []; jsum_all = 0.0; nsolve = 0   # v36l attr
    for q in range(len(pts) - 1):   # v36l attr
        P, Q = pts[q], pts[q + 1]   # v36l attr
        diff = {kap: (P["masks"][kap], Q["masks"][kap]) for kap in P["masks"] if (kap in Q["masks"] and not np.array_equal(np.asarray(P["masks"][kap], bool), np.asarray(Q["masks"][kap], bool)))}   # v36l attr
        if not diff:   # v36l attr
            continue   # v36l attr
        pairs, s_star, jsum, foP, foQ = dgk_attr_pairs_v36l(diff, P["spec"], Q["spec"], P["s"], Q["s"], wch, E_rec)   # v36l attr
        DP = None; DQ = None   # v36l attr
        if DGK["attr"]:   # v36l attr
            E_altP, _iP = dgk_alt_v36l(a, chs, R, P["base"], P["N"], P["v"], anchor, {kap: m[1] for kap, m in diff.items()}); nsolve += 1   # v36l attr
            E_altQ, _iQ = dgk_alt_v36l(a, chs, R, Q["base"], Q["N"], Q["v"], anchor, {kap: m[0] for kap, m in diff.items()}); nsolve += 1   # v36l attr
            DP = ((E_altP - P["E"]) / E_rec) if (E_altP is not None and P["E"] is not None) else None   # v36l attr
            DQ = ((Q["E"] - E_altQ) / E_rec) if (E_altQ is not None and Q["E"] is not None) else None   # v36l attr
        rec = {"from_s": float(P["s"]), "to_s": float(Q["s"]), "channels": sorted(str(k_) for k_ in diff), "pairs": pairs, "s_star": float(s_star), "jump_first_order_over_E_rec": jsum, "first_order_at_P_over_E_rec": foP, "first_order_at_Q_over_E_rec": foQ,   # v36l attr
               "D_at_P_over_E_rec": DP, "D_at_Q_over_E_rec": DQ, "response_at_P_over_E_rec": ((DP - foP) if DP is not None else None), "response_at_Q_over_E_rec": ((DQ - foQ) if DQ is not None else None)}   # v36l attr
        recs.append(rec); jsum_all += jsum   # v36l attr
        pp = [p_ for p_ in pairs if "old" in p_]   # v36l attr
        print("  v36l attr T/rs %.4f sub-step %d/%d: the tracked set differs between s %.4f and %.4f in channel(s) %s: levels %s; gaps e_new - e_old [%s | %s], the set members' anchor weights [%s | %s], gap_min estimates [%s | %s]; s* %.3f; first-order jump -w gap_min %+.3e of E (-w gap at the points %+.3e | %+.3e)%s" % (   # v36l attr
            T_over_rs, j1, nsub, P["s"], Q["s"], ",".join(rec["channels"]), " ".join("%s:%d->%d" % (p_["channel"], p_["old"], p_["new"]) for p_ in pp) or "-",   # v36l attr
            " ".join("%+.3e" % p_["gap_at_P"] for p_ in pp), " ".join("%+.3e" % p_["gap_at_Q"] for p_ in pp), " ".join("%.4f" % p_["weights_at_P"][0] for p_ in pp), " ".join("%.4f" % p_["weights_at_Q"][1] for p_ in pp),   # v36l attr
            " ".join("%+.3e" % p_["gap_min_estimates"][0] for p_ in pp), " ".join("%+.3e" % p_["gap_min_estimates"][1] for p_ in pp), s_star, jsum, foP, foQ,   # v36l attr
            ("; measured charge difference new - old mask at the points %s | %s of E (response beyond first order %s | %s)" % (fmt(DP, "%+.3e"), fmt(DQ, "%+.3e"), fmt(rec["response_at_P_over_E_rec"], "%+.3e"), fmt(rec["response_at_Q_over_E_rec"], "%+.3e"))) if DGK["attr"] else ""), flush=True)   # v36l attr
    return recs, jsum_all, nsolve   # v36l attr


def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart'''),
("substep init",
r'''    masks_prev = None; changed_hist = []; clipped = 0   # v36k chart''',
r'''    masks_prev = None; changed_hist = []; clipped = 0; vs_anchor_hist = []   # v36k chart; v36l attr'''),
("substep node init",
r'''        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []   # v36k chart''',
r'''        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []; specs = []; bases = []   # v36k chart; v36l attr'''),
("substep node call",
r'''            Nk, vk, Ek, info_k, ti_k, mk, _r = dgk_node_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], anchor)   # v36k chart''',
r'''            Nk, vk, Ek, info_k, ti_k, mk, _r, sp_k = dgk_node_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], anchor)   # v36k chart; v36l attr'''),
("substep node append",
r'''            x_new.append((Nk, vk)); nodeE.append(Ek); tinfos.append(ti_k); masks.append(mk)   # v36k chart''',
r'''            x_new.append((Nk, vk)); nodeE.append(Ek); tinfos.append(ti_k); masks.append(mk); specs.append(sp_k); bases.append(base_s)   # v36k chart; v36l attr'''),
("substep anchor count",
r'''        masks_prev = masks; changed_hist.append(changed)   # v36k chart''',
r'''        masks_prev = masks; changed_hist.append(changed)   # v36k chart
        vs_anchor = (sum(int(not np.array_equal(masks[k][kap], anchor[kap][2])) for k in range(3) for kap in masks[k]) if anchor is not None else 0); vs_anchor_hist.append(vs_anchor)   # v36l attr: the node-channels whose set differs from the anchor's (the chord's own switching, visible from the first iterate)'''),
("substep iterate print",
r'''        print("  v36k chart T/rs %.4f sub-step %d/%d iteration %d: energy defect %+.3e of E (change between iterates %s), node fields' change %.2e [%s], node E/E_rec - 1 [%s]; tracked set changed at %d node-channels; weights min occupied %s, max empty %s" % (   # v36k chart
            T0 / rs, j + 1, nsub, it, dE / E_rec, fmt(chg, "%.3e"), dfields, " ".join("%.1e" % q for q in nres), " ".join("%+.3e" % (q / E_rec - 1.0) for q in nodeE), changed, fmt(wmin, "%.5f"), fmt(wmax, "%.2e")), flush=True)   # v36k chart''',
r'''        print("  v36k chart T/rs %.4f sub-step %d/%d iteration %d: energy defect %+.3e of E (change between iterates %s), node fields' change %.2e [%s], node E/E_rec - 1 [%s]; tracked set changed at %d node-channels, differs from the anchor's at %d; weights min occupied %s, max empty %s" % (   # v36k chart; v36l attr
            T0 / rs, j + 1, nsub, it, dE / E_rec, fmt(chg, "%.3e"), dfields, " ".join("%.1e" % q for q in nres), " ".join("%+.3e" % (q / E_rec - 1.0) for q in nodeE), changed, vs_anchor, fmt(wmin, "%.5f"), fmt(wmax, "%.2e")), flush=True)   # v36k chart; v36l attr'''),
("substep endpoint init",
r'''        passes = []; Ee = None; ti_e = {}; refs_e = {}   # v36k chart''',
r'''        passes = []; Ee = None; ti_e = {}; refs_e = {}; m_e = {}; sp_e = {}; f_in_e = (N1, v1)   # v36k chart; v36l attr'''),
("substep endpoint call",
r'''            Ne, ve, Ee, info_e, ti_e, _m, refs_e = dgk_node_v36k(a, chs, st, R, base1, N1, v1, anchor, keep_ref=last, check=last)   # v36k chart''',
r'''            f_in_e = (N1, v1)   # v36l attr: the pass's input fields, at which its spectra and masks are taken
            Ne, ve, Ee, info_e, ti_e, m_e, refs_e, sp_e = dgk_node_v36k(a, chs, st, R, base1, N1, v1, anchor, keep_ref=last, check=last)   # v36k chart; v36l attr'''),
("substep entry keys",
r'''           "track_nodes_final": tinfos, "history": hist, "r_chain_over_E_rec": None, "gate1": None, "E_end": None, "chart_disagreement": None, "endpoint_passes": None, "track_end": None}   # v36k chart''',
r'''           "track_nodes_final": tinfos, "history": hist, "r_chain_over_E_rec": None, "gate1": None, "E_end": None, "chart_disagreement": None, "endpoint_passes": None, "track_end": None,   # v36k chart
           "vs_anchor_hist": vs_anchor_hist, "attribution": None, "switch_sum_over_E_rec": None, "n_switch": None, "rest_over_E_rec": None, "attribution_solves": 0}   # v36l attr: the chord's switches and the two bins of r_chain'''),
("substep attribution",
r'''            ent.update({"E_end": E_end, "r_chain_over_E_rec": r_rel, "gate1": ("pass" if abs(r_rel) <= DGK["chain_tol"] else "exceeded"), "endpoint_passes": passes,   # v36k chart
                        "chart_disagreement": sum(int(ti.get("convention_differs") or 0) for ti in ti_e.values()), "track_end": ti_e})   # v36k chart''',
r'''            ent.update({"E_end": E_end, "r_chain_over_E_rec": r_rel, "gate1": ("pass" if abs(r_rel) <= DGK["chain_tol"] else "exceeded"), "endpoint_passes": passes,   # v36k chart
                        "chart_disagreement": sum(int(ti.get("convention_differs") or 0) for ti in ti_e.values()), "track_end": ti_e})   # v36k chart
            if anchor is not None:   # v36l attr: the chord's switches between the anchor, the nodes and the endpoint (the switching bin of r_chain; two solves per switch under --dg-attr on)
                pts_ = [{"s": 0.0, "N": np.asarray(f0[0], float), "v": np.asarray(f0[1], float), "base": base0, "E": E0, "masks": {kap: anchor[kap][2] for kap in anchor}, "spec": {kap: (anchor[kap][4], np.asarray(anchor[kap][2], bool).astype(float)) for kap in anchor}}]   # v36l attr
                pts_ += [{"s": s_nodes[k], "N": x_new[k][0], "v": x_new[k][1], "base": bases[k], "E": nodeE[k], "masks": masks[k], "spec": specs[k]} for k in range(3)]   # v36l attr
                pts_.append({"s": 1.0, "N": np.asarray(f_in_e[0], float), "v": np.asarray(f_in_e[1], float), "base": base1, "E": E_end, "masks": m_e, "spec": sp_e})   # v36l attr
                recs_, jsum_, ns_ = dgk_attr_v36l(a, chs, R, anchor, pts_, E_rec, T0 / rs, j + 1, nsub)   # v36l attr
                n_solves += ns_   # v36l attr
                ent.update({"attribution": recs_, "switch_sum_over_E_rec": jsum_, "n_switch": len(recs_), "rest_over_E_rec": r_rel - jsum_, "attribution_solves": ns_})   # v36l attr'''),
("substep print",
r'''    print("T/rs %.4f v36k chart sub-step %d/%d (tau %.4g) to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; gate 1 r_chain %s of E (%s); the convention would reassign %s levels at the end; tracked-set changes %d, clipped updates %d; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36k chart
        T0 / rs, j + 1, nsub, tau_s, (T0 + tau_s) / rs, "converged" if conv else "NOT CONVERGED", len(hist), fmt(ent["energy_defect_final"], "%+.3e"), fmt(ent["energy_change_between_iterates_final"], "%.3e"), fmt(dfields, "%.2e"),   # v36k chart
        fmt(ent["r_chain_over_E_rec"], "%+.3e"), ent["gate1"] or "-", ent["chart_disagreement"] if ent["chart_disagreement"] is not None else "-", sum(changed_hist), clipped, ent["eigendecompositions"], n_solves, ent["seconds"], start, ("; " + why) if why else ""), flush=True)   # v36k chart''',
r'''    print("T/rs %.4f v36k chart sub-step %d/%d (tau %.4g) to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; gate 1 r_chain %s of E (%s; in-chord switching %s over %s differences, remainder %s); the convention would reassign %s levels at the end; tracked-set changes %d, clipped updates %d; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36k chart; v36l attr
        T0 / rs, j + 1, nsub, tau_s, (T0 + tau_s) / rs, "converged" if conv else "NOT CONVERGED", len(hist), fmt(ent["energy_defect_final"], "%+.3e"), fmt(ent["energy_change_between_iterates_final"], "%.3e"), fmt(dfields, "%.2e"),   # v36k chart
        fmt(ent["r_chain_over_E_rec"], "%+.3e"), ent["gate1"] or "-", fmt(ent["switch_sum_over_E_rec"], "%+.3e"), ("%d" % ent["n_switch"]) if ent["n_switch"] is not None else "-", fmt(ent["rest_over_E_rec"], "%+.3e"), ent["chart_disagreement"] if ent["chart_disagreement"] is not None else "-", sum(changed_hist), clipped, ent["eigendecompositions"], n_solves, ent["seconds"], start, ("; " + why) if why else ""), flush=True)   # v36k chart; v36l attr'''),
("step start",
r'''    anchor_info = ({str(k_): q[3] for k_, q in anchor0.items()} if anchor0 is not None else None)   # v36k chart
    cols0 = {ch.kappa: (st[ch.kappa]["sea"], st[ch.kappa]["dust"]) for ch in chs}   # v36k chart
    level = int(DGK["level"]); attempts = []; conv = False; subs = []; why = None   # v36k chart''',
r'''    anchor_info = ({str(k_): q[3] for k_, q in anchor0.items()} if anchor0 is not None else None)   # v36k chart
    cols0 = {ch.kappa: (st[ch.kappa]["sea"], st[ch.kappa]["dust"]) for ch in chs}   # v36k chart
    E_start = E_rec; start_passes = []; track_start = None   # v36l attr (review B1): the chart's charge at C_0 by the endpoint's own rule (gate 1's baseline): E_rec to the solve's floor at a converged record whose set the anchor reproduces, else the record solve's non-convergence or the anchor's mismatch, its own bin
    Ns_, vs_ = np.array(f_acc[0], float), np.array(f_acc[1], float)   # v36l attr (review B1)
    for p_ in range(DGK["end_passes"]):   # v36l attr (review B1)
        Nn_, vn_, Ee_, _i, ti_s, _m, _r, _sp = dgk_node_v36k(a, chs, st, R, base0, Ns_, vs_, anchor0)   # v36l attr (review B1)
        if Nn_ is None:   # v36l attr (review B1)
            break   # v36l attr (review B1)
        start_passes.append(max(float(np.max(np.abs(Nn_ - Ns_))), float(np.max(np.abs(vn_ - vs_))))); Ns_, vs_ = Nn_, vn_; E_start = Ee_; track_start = ti_s   # v36l attr (review B1)
    level = int(DGK["level"]); attempts = []; conv = False; subs = []; why = None   # v36k chart'''),
("step Ecur",
r'''        cols = cols0; base = base0; f = (np.array(f_acc[0], float), np.array(f_acc[1], float)); Ecur = E_rec; anchor = anchor0   # v36k chart''',
r'''        cols = cols0; base = base0; f = (np.array(f_acc[0], float), np.array(f_acc[1], float)); Ecur = E_start; anchor = anchor0   # v36k chart; v36l attr (review B1)'''),
("step trigger",
r'''            if ent["gate1"] == "exceeded" and level < DGK["sub_max"]:   # v36k chart''',
r'''            if DGK["chain_trigger"] and ent["gate1"] == "exceeded" and level < DGK["sub_max"]:   # v36k chart; v36l attr: --dg-chain-trigger off makes gate 1 a measurement'''),
("step quiet",
r'''        quiet = all((q["iterations"] <= max(3, DG["cap"] // 3)) and (max(q["changed_hist"] or [0]) == 0) and (abs(q["r_chain_over_E_rec"] or 0.0) <= DGK["chain_tol"] / 16.0) for q in subs)   # v36k chart''',
r'''        quiet = all((q["iterations"] <= max(3, DG["cap"] // 3)) and (max(q["changed_hist"] or [0]) == 0) and ((not DGK["chain_trigger"]) or abs(q["r_chain_over_E_rec"] or 0.0) <= DGK["chain_tol"] / 16.0) for q in subs)   # v36k chart; v36l attr: gate 1 enters only when it triggers'''),
("step sums",
r'''    rsum = (sum(q["r_chain_over_E_rec"] for q in subs if q.get("r_chain_over_E_rec") is not None) if (conv and subs) else None)   # v36k chart''',
r'''    rsum = (sum(q["r_chain_over_E_rec"] for q in subs if q.get("r_chain_over_E_rec") is not None) if (conv and subs) else None)   # v36k chart
    ssum = (sum(q["switch_sum_over_E_rec"] for q in subs if q.get("switch_sum_over_E_rec") is not None) if (conv and subs) else None)   # v36l attr: the switching bin over the sub-steps
    nsw = (sum(int(q.get("n_switch") or 0) for q in subs) if (conv and subs) else None)   # v36l attr'''),
("step entry",
r'''           "E_pred": ((E_rec * (1.0 + dsum)) if dsum is not None else None), "E_end": E_end, "r_chain_over_E_rec": rsum, "chart_disagreement_total": last.get("chart_disagreement"),   # v36k chart''',
r'''           "E_pred": ((E_rec * (1.0 + dsum)) if dsum is not None else None), "E_end": E_end, "r_chain_over_E_rec": rsum, "chart_disagreement_total": last.get("chart_disagreement"),   # v36k chart
           "E_start": E_start, "E_start_minus_E_rec_over_E_rec": (((E_start - E_rec) / E_rec) if E_rec else None), "start_passes": start_passes, "track_start": track_start,   # v36l attr (review B1)
           "switch_sum_over_E_rec": ssum, "n_switch": nsw, "rest_over_E_rec": ((rsum - ssum) if (rsum is not None and ssum is not None) else None),   # v36l attr: the switching bin of r_chain and its remainder'''),
("step solves",
r'''           "anchor": anchor_info, "eigendecompositions": TM["eigh_n"] - n_eigh0, "solves": sum(q.get("solves", 0) for q in subs)}   # v36k chart''',
r'''           "anchor": anchor_info, "eigendecompositions": TM["eigh_n"] - n_eigh0, "solves": sum(q.get("solves", 0) for q in subs) + len(start_passes)}   # v36k chart; v36l attr (review B1)'''),
("step print",
r'''    print("T/rs %.4f v36k chart step to T/rs %.4f: %s; level %d (%d sub-steps of tau %.4g; %d attempts; next level %d); %d iterations; energy defect %s of E; gate 1 r_chain %s of E; E_end/E_rec - 1 = %s; E_pred/E_rec - 1 = %s; the convention would reassign %s levels; anchor: %s; eigh %d, solves %d, %.1f s%s" % (   # v36k chart
        T / rs, (T + a.tau) / rs, "converged" if conv else "NOT CONVERGED", level, nsub, tau_s, len(attempts), DGK["level"], ent["iterations"], fmt(dsum, "%+.3e"), fmt(rsum, "%+.3e"),   # v36k chart''',
r'''    print("T/rs %.4f v36k chart step to T/rs %.4f: %s; level %d (%d sub-steps of tau %.4g; %d attempts; next level %d); %d iterations; energy defect %s of E; E_start/E_rec - 1 = %s (%d passes); gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s); E_end/E_rec - 1 = %s; E_pred/E_rec - 1 = %s; the convention would reassign %s levels; anchor: %s; eigh %d, solves %d, %.1f s%s" % (   # v36k chart; v36l attr (review B1 and the switching bin)
        T / rs, (T + a.tau) / rs, "converged" if conv else "NOT CONVERGED", level, nsub, tau_s, len(attempts), DGK["level"], ent["iterations"], fmt(dsum, "%+.3e"), fmt(ent["E_start_minus_E_rec_over_E_rec"], "%+.3e"), len(start_passes), fmt(rsum, "%+.3e"), fmt(ssum, "%+.3e"), ("%d" % nsw) if nsw is not None else "-", fmt(ent["rest_over_E_rec"], "%+.3e"),   # v36k chart; v36l attr'''),
("record line",
r'''                print("T/rs %.4f v36k chart: reassignment move (E - E_end)/E0 = %s; the step: level %d (%d sub-steps of tau %.4g, %d attempts), gate 1 r_chain %s of E, gate 2 defect %s of E, tracked-set changes %s, the convention would reassign %s levels at the step's end, anchor %s; the record solve converged: %s" % (   # v36k chart
                    T / rs, fmt(d_.get("E_minus_E_end_over_E0"), "%+.3e"), d_.get("level", 0), d_.get("nsub", 1), d_.get("tau_sub", a.tau), d_.get("attempts", 1), fmt(d_.get("r_chain_over_E_rec"), "%+.3e"), fmt(d_.get("energy_defect_final"), "%+.3e"),   # v36k chart''',
r'''                print("T/rs %.4f v36k chart: reassignment move (E - E_end)/E0 = %s; the step: level %d (%d sub-steps of tau %.4g, %d attempts), start E_start/E_rec - 1 %s, gate 1 r_chain %s of E (in-chord switching %s over %s differences, remainder %s), gate 2 defect %s of E, tracked-set changes %s, the convention would reassign %s levels at the step's end, anchor %s; the record solve converged: %s" % (   # v36k chart; v36l attr (review B1 and the switching bin)
                    T / rs, fmt(d_.get("E_minus_E_end_over_E0"), "%+.3e"), d_.get("level", 0), d_.get("nsub", 1), d_.get("tau_sub", a.tau), d_.get("attempts", 1), fmt(d_.get("E_start_minus_E_rec_over_E_rec"), "%+.3e"), fmt(d_.get("r_chain_over_E_rec"), "%+.3e"), fmt(d_.get("switch_sum_over_E_rec"), "%+.3e"), ("%d" % d_["n_switch"]) if d_.get("n_switch") is not None else "-", fmt(d_.get("rest_over_E_rec"), "%+.3e"), fmt(d_.get("energy_defect_final"), "%+.3e"),   # v36k chart; v36l attr'''),
("run label",
r'''    if DGK["on"]:   # v36k chart
        label += "; v36k: " + dgk_desc_v36k(a)   # v36k chart''',
r'''    if DGK["on"]:   # v36k chart
        label += "; v36k: " + dgk_desc_v36k(a)   # v36k chart
        label += "; v36l: " + dgk_desc_v36l(a)   # v36l attr'''),
("arguments",
r'''    ap.add_argument("--dg-end-passes", dest="dg_end_passes", type=int, default=2, help="(v36k) the quasi-static evaluations at the step's end (gate 1 and the next anchor)")   # v36k chart''',
r'''    ap.add_argument("--dg-end-passes", dest="dg_end_passes", type=int, default=2, help="(v36k) the quasi-static evaluations at the step's end (gate 1 and the next anchor)")   # v36k chart
    ap.add_argument("--dg-chain-trigger", dest="dg_chain_trigger", default="on", choices=["on", "off"], help="(v36l) on: gate 1 above --dg-chain-tol halves the sub-step (v36k); off: gate 1 is measured and printed only (the chart's charge change over the chord is the convention's in-chord reassignment plus the three-node rule's defect, not a conservation gate)")   # v36l attr
    ap.add_argument("--dg-attr", dest="dg_attr", default="off", choices=["off", "on"], help="(v36l) on: at each difference of the tracked set between consecutive chord points the chart's charge is also evaluated at both points with the other point's mask (two eigendecomposition sets and two solves per difference) as the measured jump beside the first-order -w gap_min; needs --step dg --dg-chart frozen")   # v36l attr'''),
("validation",
r'''    if DGK["on"] and DGK["mix"] not in ("plain", "anderson"):   # v36k chart
        ap.error("version 36k: --dg-mix must resolve to plain or anderson")   # v36k chart''',
r'''    DGK["chain_trigger"] = (a.dg_chain_trigger == "on"); DGK["attr"] = (a.dg_attr == "on")   # v36l attr
    if DGK["on"] and DGK["mix"] not in ("plain", "anderson"):   # v36k chart
        ap.error("version 36k: --dg-mix must resolve to plain or anderson")   # v36k chart
    if a.dg_attr == "on" and not DGK["on"]:   # v36l attr
        ap.error("version 36l: --dg-attr on needs --step dg --dg-chart frozen")   # v36l attr'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36l: the OLD string of '%s' occurs %d times in the source (1 required): the source is not build_engine_collapse_v36k.py's engine_collapse_v36k.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36l: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36k.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36l.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36l: the reverse replacements do not restore the source (a NEW text overlaps another)")
    marked = sum(1 for l_ in out.splitlines() if ("v36l attr" in l_))
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
