#!/usr/bin/env python
"""build_engine_collapse_v36k.py (reviewer A for the busi lane; W2 dg ruling v2): writes engine_collapse_v36k.py from engine_collapse_v36f.py
(the file build_engine_collapse_v36f.py writes, input 01) by 11 exact string replacements in that builder's form. Each OLD is copied exactly from
engine_collapse_v36f.py and must occur exactly once (else the build aborts); each NEW contains inserted or changed text so that replacing every
NEW by its OLD restores the source byte for byte (checked here). Every inserted or changed code line carries the marker "v36k chart" (a trailing
comment; the docstring's lines "[v36k chart]"). The option --dg-chart {continued, frozen} is "continued" by default: with it nothing of the
inserted code runs (DGK stays off, dg_step_v36f is called as before) and the json's "args" dict alone gains the new keys, so v36k --step dg is
v36f --step dg bitwise and v36k --step frozen is v36d bitwise (engine_collapse_v36k_selftest.py --fixture, checks (i), (ii)).
Usage: python build_engine_collapse_v36k.py [engine_collapse_v36f.py path] [engine_collapse_v36k.py path]   (defaults: beside this script)"""
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36f.py (reviewer A for the busi lane; WEB40I review, busi W2 and code defect 1): version 36f = version 36d with one option, off by default, [v36f dg]''',
r'''"""engine_collapse_v36k.py (reviewer A for the busi lane; W2 dg ruling v2): version 36k = version 36f with one option, off by default, so that the defaults are [v36k chart]
version 36f bitwise in every record number, printed line and stored array (the json's "args" dict gains the keys below): [v36k chart]
--dg-chart continued (the default): version 36f's discrete-gradient step, the reference re-continued by continue_ref_v34 from the record's accepted entry to each node's [v36k chart]
    trial fields at every iterate. [v36k chart]
--dg-chart frozen [--dg-mix inherit|plain|anderson] [--dg-anderson-m 4] [--dg-beta 1.0] [--dg-max-change 2e-2] [--dg-sub-max 3] [--dg-sub-start 0] [--dg-chain-tol 1e-8] [v36k chart]
    [--dg-end-passes 2]: the chart-frozen step (W2_dg_ruling_v2.md). v36f's bursts near horizon formation are reassignments of the continued reference between iterates (a [v36k chart]
    pocket or inversion resolved the other way when a node's trial fields cross a switching surface): the node's sources jump by 16 kappa times a level's density, the map has [v36k chart]
    no fixed point on either branch and the iteration cycles. The chart is therefore decided once per step, the record's accepted set (the window's last entry continued to the [v36k chart]
    accepted fields themselves, dgk_anchor_v36k), and held along the chord: at each Gauss node the reference is the spectral projector of h_m[N_k, v_k] onto the levels of [v36k chart]
    largest overlap with that set, its count held (dgk_track_v36k), a function of the node fields alone and spectral along the path (Tr(h P-dot) = 0), so that E[C_1] - E[C_0] = [v36k chart]
    Tr(H-bar (C_1 - C_0)) holds on the chart and the step conserves the chart's charge; the convention's reassignment is left to the record solve at T + tau and measured there [v36k chart]
    as E_rec - E_end (the record's dg entry and record line). The fixed point: Anderson mixing (type II, history --dg-anderson-m, beta --dg-beta, the history purged when the [v36k chart]
    tracked set changes at any node between iterates) or plain damping, a trust region of --dg-max-change on the node fields per iterate, convergence as in v36f plus no tracked- [v36k chart]
    set change at the last iterate. Gate 2 (icloud X9 section 4): Tr(G R), G = H-bar at the converged nodes, R = C_1 - exp(-i tau G) C_0 exp(i tau G), equals the printed energy [v36k chart]
    defect identically (U_k commutes with H-bar_k), so it is met at convergence. Gate 1: r_chain = E(C_1) - E(C_0) on the chart, E(C_1) from --dg-end-passes quasi-static [v36k chart]
    evaluations at the propagated state with the tracked reference (the node fields extrapolated to s = 1; second order in that extrapolation), reported per step; above [v36k chart]
    --dg-chain-tol of the record's charge the sub-step is halved. Sub-steps: when a sub-step's fixed point does not converge within --dg-cap, when its tracked set changed at [v36k chart]
    its last iterate or when gate 1 fails, the step is redone as 2^L sub-steps of tau/2^L (L raised by one up to --dg-sub-max, else the run ends at the record as in v36f), each [v36k chart]
    its own fixed point from the previous sub-step's state with the previous sub-step's tracked endpoint set as its anchor, the record at T + tau the composition (no record at [v36k chart]
    the intermediate clocks); the level is sticky across records, lowered by one after a quiet record and never below --dg-sub-start. The record's dg_v36f entry keeps v36f's [v36k chart]
    keys (E_pred = E_rec + the sub-steps' defects) and gains chart, level, nsub, attempts, substeps, E_end, r_chain_over_E_rec, chart_disagreement_total (the levels the [v36k chart]
    convention would reassign at the step's end), E_minus_E_end_over_E0 (the reassignment move); the slices gain dg_sub_len_v36k and dg_level_v36k (a restart at the same [v36k chart]
    sub-step length extrapolates its start as the uninterrupted run does). The engine's own conventions (the record solve, its mixer, its continuation) are unchanged. Every [v36k chart]
    changed code line of this file is marked "v36k chart". [v36k chart]
Version 36f's docstring follows. [v36k chart]
engine_collapse_v36f.py (reviewer A for the busi lane; WEB40I review, busi W2 and code defect 1): version 36f = version 36d with one option, off by default, [v36f dg]'''),
("module state",
r'''DG = {"on": False, "tol": 1e-12, "cap": 30, "rec": None, "nodes": None, "steps": 0, "collars": (16.0, 24.0)}   # v36f dg: the discrete-gradient step's state (--step dg)''',
r'''DG = {"on": False, "tol": 1e-12, "cap": 30, "rec": None, "nodes": None, "steps": 0, "collars": (16.0, 24.0)}   # v36f dg: the discrete-gradient step's state (--step dg)
DGK = {"on": False, "mix": "plain", "m": 4, "beta": 1.0, "max_change": 2e-2, "sub_max": 3, "chain_tol": 1e-8, "end_passes": 2, "level": 0, "level_min": 0, "sub_len": None}   # v36k chart: the chart-frozen step's state (--dg-chart frozen)'''),
("functions before run",
r'''def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)''',
r'''def dgk_desc_v36k(a):   # v36k chart
    """version 36k: --dg-chart frozen in words (the run label, the record's engine_v36k entry)."""   # v36k chart
    return ("the chart-frozen discrete-gradient step: the reference's occupied set decided once per step as the record's accepted set (the window's last entry continued to the "   # v36k chart
            "accepted fields themselves) and held along the chord, at each Gauss node the spectral projector of h_m[N_k, v_k] onto the levels of largest overlap with that set "   # v36k chart
            "(count held); the fixed point in the node fields by %s mixing (history %d, beta %g, at most %g change per iterate), at most %d iterations per sub-step; gate 2 the "   # v36k chart
            "energy defect Tr(G R) below %g of E (the stopping rule); gate 1 the chart's energy change E(C_1) - E(C_0) from %d quasi-static evaluations at the step's end, the "   # v36k chart
            "sub-step halved above %g of E; sub-steps of tau/2^L up to L = %d when the fixed point does not converge, the tracked set changes at the last iterate or gate 1 fails, "   # v36k chart
            "the record at T + tau the composition; the reference's reassignment left to the record solve and measured there (E - E_end)") % (   # v36k chart
        DGK["mix"], DGK["m"], DGK["beta"], DGK["max_change"], DG["cap"], DG["tol"], DGK["end_passes"], DGK["chain_tol"], DGK["sub_max"])   # v36k chart


def dgk_anchor_summary_v36k(info):   # v36k chart
    """version 36k: the anchor's continuation info in a few words (pockets and nsub summed over the channels, the smallest occupied weight)."""   # v36k chart
    if not info:   # v36k chart
        return "-"   # v36k chart
    def tot(key):   # v36k chart
        vals = [q.get(key) for q in info.values() if isinstance(q, dict) and q.get(key) is not None]   # v36k chart
        try:   # v36k chart
            return sum((sum(int(z) for z in v_.values()) if isinstance(v_, dict) else int(v_)) for v_ in vals) if vals else "-"   # v36k chart
        except (TypeError, ValueError):   # v36k chart
            return "-"   # v36k chart
    w_ = [q.get("min_occupied_weight") for q in info.values() if isinstance(q, dict) and q.get("min_occupied_weight") is not None]   # v36k chart
    return "pockets %s, nsub %s, min occupied weight %s" % (tot("n_pocket"), tot("nsub"), ("%.4f" % min(w_)) if w_ else "-")   # v36k chart


class DgkAnderson_v36k:   # v36k chart
    """version 36k: Anderson mixing (type II, Walker-Ni) on the node fields: x_(k+1) = x_k + beta f_k - (dX + beta dF) gamma, gamma the regularized least-squares fit of f_k on the last m residual differences; reset() purges the history (a tracked-set change invalidates the secants)."""   # v36k chart
    def __init__(self, m, beta, reg=1e-10):   # v36k chart
        self.m = int(m); self.beta = float(beta); self.reg = float(reg); self.X = []; self.F = []   # v36k chart
    def reset(self):   # v36k chart
        self.X = []; self.F = []   # v36k chart
    def step(self, x, gx):   # v36k chart
        f = gx - x; self.X.append(np.array(x, float)); self.F.append(np.array(f, float))   # v36k chart
        while len(self.X) > self.m + 1:   # v36k chart
            self.X.pop(0); self.F.pop(0)   # v36k chart
        if len(self.X) < 2:   # v36k chart
            return x + self.beta * f, 0   # v36k chart
        dF = np.stack([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)], axis=1); dX = np.stack([self.X[i + 1] - self.X[i] for i in range(len(self.X) - 1)], axis=1)   # v36k chart
        A = dF.T @ dF; A = A + self.reg * max(float(np.trace(A)) / A.shape[0], 1e-300) * np.eye(A.shape[0])   # v36k chart
        gamma = np.linalg.solve(A, dF.T @ f)   # v36k chart
        return x + self.beta * f - (dX + self.beta * dF) @ gamma, int(dF.shape[1])   # v36k chart


def dgk_track_v36k(e, U, anchor):   # v36k chart
    """version 36k: the frozen chart's reference at the eigendecomposition (e, U) of h_m at a node's fields: each level weighed by its overlap sum_i |<u_j|a_i>|^2 with the anchor's columns A (the record's accepted reference, or the previous sub-step's tracked endpoint); the anchor's count of levels of largest weight is the set. Returns the reference columns U[:, set], the set (bool over the levels) and its sharpness (min occupied weight, max empty weight, inversions against the energy order, equality with the negative set)."""   # v36k chart
    torch = BK["torch"]   # v36k chart
    A = anchor[0]; n_occ = int(anchor[1])   # v36k chart
    w = ((U.conj().T @ A).abs() ** 2).sum(dim=1).cpu().numpy() if n_occ else np.zeros(U.shape[1])   # v36k chart
    order = np.argsort(-w, kind="stable"); mask = np.zeros(w.shape[0], bool); mask[order[:n_occ]] = True   # v36k chart
    ref = U[:, torch.from_numpy(mask).to(U.device)]   # v36k chart
    ev = e.detach().cpu().numpy()   # v36k chart
    info = {"min_occupied_weight": (float(w[mask].min()) if n_occ else None), "max_empty_weight": (float(w[~mask].max()) if (~mask).any() else None),   # v36k chart
            "inversions": (int(np.sum(ev[mask] > ev[~mask].min())) if (n_occ and (~mask).any()) else 0), "mask_equals_negative_set": bool(np.array_equal(mask, ev < 0.0))}   # v36k chart
    return ref, mask, info   # v36k chart


def dgk_anchor_v36k(a, chs, st, f):   # v36k chart
    """version 36k: the frozen chart at a step's start: per channel the record's accepted reference by the engine's own convention continued from the window's last entry to the accepted fields themselves (the identity continuation; one eigendecomposition set), as (columns, count, set, continuation info); None under R1 (no reference along the chord)."""   # v36k chart
    if a.reference != "R2":   # v36k chart
        return None   # v36k chart
    eig = eigh_all(chs, f[0], f[1], bool(a.streams)); out = {}   # v36k chart
    for ch in chs:   # v36k chart
        e, U = eig.pop(ch.kappa); s_ = st[ch.kappa]   # v36k chart
        ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (f[0], f[1]), e, U, s_["n_sea"])   # v36k chart
        out[ch.kappa] = (ref, int(ref.shape[1]), np.asarray(mask, bool).copy(), {"n_pocket": cinfo.get("n_pocket"), "nsub": cinfo.get("nsub"), "min_occupied_weight": cinfo.get("min_occupied_weight"), "verified": cinfo.get("verified")})   # v36k chart
        del e, U   # v36k chart
    del eig   # v36k chart
    return out   # v36k chart


def dgk_node_v36k(a, chs, st, R, base_s, Nk, vk, anchor, keep_ref=False, check=False):   # v36k chart
    """version 36k: one evaluation of a Gauss node's map on the frozen chart at the trial fields (Nk, vk): the eigendecompositions of h_m[Nk, vk], the reference tracked from the anchor (dgk_track_v36k), the node's sources (the node's base densities less the tracked reference's, the dust's added, each channel weighted by 16 kappa) and one field solve. keep_ref returns the tracked columns (the next sub-step's anchor at the endpoint); check continues the reference by the engine's own convention as well and counts the levels it would assign differently (the record solve's likely reassignment). Returns (N_out, v_out, E_node, info, per-channel track info, masks, refs)."""   # v36k chart
    n = len(R)   # v36k chart
    eig = eigh_all(chs, Nk, vk, bool(a.streams))   # v36k chart
    eps = np.zeros(n); J = np.zeros(n); tinfo = {}; masks = {}; refs = {}   # v36k chart
    for ch in chs:   # v36k chart
        e, U = eig.pop(ch.kappa)   # v36k chart
        hs_, Qs_, dd, Jd = base_s[ch.kappa]   # v36k chart
        if anchor is not None:   # v36k chart
            ref, mask, ti = dgk_track_v36k(e, U, anchor[ch.kappa])   # v36k chart
            ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)   # v36k chart
            if check:   # v36k chart
                s_ = st[ch.kappa]   # v36k chart
                ref_c, cinfo, mask_c = continue_ref_v34(a, ch, s_["window"], (Nk, vk), e, U, s_["n_sea"])   # v36k chart
                ti["convention_differs"] = int(np.sum(np.asarray(mask_c, bool) != mask)); ti["convention_n_pocket"] = cinfo.get("n_pocket"); ti["convention_min_occupied_weight"] = cinfo.get("min_occupied_weight")   # v36k chart
                del ref_c   # v36k chart
            masks[ch.kappa] = mask   # v36k chart
            if keep_ref:   # v36k chart
                refs[ch.kappa] = (ref, int(ref.shape[1]), mask, None)   # v36k chart
            else:   # v36k chart
                del ref   # v36k chart
            tinfo[str(ch.kappa)] = ti   # v36k chart
        else:   # v36k chart
            ds = 0.0; Js = 0.0; tinfo[str(ch.kappa)] = {"mode": "R1"}   # v36k chart
        eps += ch.w * (ds + dd); J += ch.w * (Js + Jd)   # v36k chart
        del e, U   # v36k chart
    del eig   # v36k chart
    Nn, vn, Wn, info = solve(eps, J, a.g, R, a.ell, None)   # v36k chart
    if info.get("failed"):   # v36k chart
        return None, None, None, info, tinfo, masks, refs   # v36k chart
    return np.asarray(Nn, float), np.asarray(vn, float), float(FLD.charge(info, a.g)), info, tinfo, masks, refs   # v36k chart


def dgk_start_v36k(prev, f):   # v36k chart
    """version 36k: the node fields' start of a sub-step: the previous sub-step's node fields extrapolated by one sub-step (quadratic Lagrange through its 3 nodes) when it had the same length and the extrapolation stays within 0.5 of the fields at the start, else those fields at every node."""   # v36k chart
    x = [(np.array(f[0], float), np.array(f[1], float)) for _ in range(3)]; start = "the fields at the sub-step's start at every node"   # v36k chart
    if prev is not None:   # v36k chart
        pN, pv = prev; xe = []; ok = True   # v36k chart
        for k in range(3):   # v36k chart
            L = dg_lagrange_v36f(1.0 + GL3_V36F[k][0])   # v36k chart
            Ne = L[0] * pN[0] + L[1] * pN[1] + L[2] * pN[2]; ve = L[0] * pv[0] + L[1] * pv[1] + L[2] * pv[2]   # v36k chart
            ok = ok and bool(np.all(np.isfinite(Ne)) and np.all(np.isfinite(ve)) and np.max(np.abs(Ne - f[0])) < 0.5 and np.max(np.abs(ve - f[1])) < 0.5)   # v36k chart
            xe.append((np.asarray(Ne, float), np.asarray(ve, float)))   # v36k chart
        if ok:   # v36k chart
            x = xe; start = "the previous sub-step's node fields extrapolated by one sub-step (quadratic Lagrange through its 3 nodes)"   # v36k chart
    return x, start   # v36k chart


def dgk_substep_v36k(a, chs, st, R, cols0, base0, f0, E0, E_rec, anchor, tau_s, x, start, T0, rs, level, j, nsub):   # v36k chart
    """version 36k: one sub-step of length tau_s from the state cols0 (site densities base0, chart energy E0, fields f0) on the frozen chart (anchor): the fixed point in the node fields as in dg_step_v36f but with the tracked reference, Anderson or damped mixing, a trust region and the tracked set's change between iterates watched; on convergence the endpoint evaluation at s = 1 (gate 1: r_chain = E(C_1) - E0 on the chart; the engine's convention checked against the frozen set; the tracked endpoint columns as the next sub-step's anchor). Returns (entry, cols1, base1, nodes, anchor_end, f_end, E_end)."""   # v36k chart
    torch = BK["torch"]   # v36k chart
    t0 = time.time(); n_eigh0 = TM["eigh_n"]; n = len(R)   # v36k chart
    s_nodes = [q[0] for q in GL3_V36F]; w_nodes = [q[1] for q in GL3_V36F]   # v36k chart
    dust_only = (anchor is None)   # v36k chart
    eps0, J0 = dg_pairing_v36f(chs, base0, dust_only)   # v36k chart
    mixer = DgkAnderson_v36k(DGK["m"], DGK["beta"]) if DGK["mix"] == "anderson" else None   # v36k chart
    tol = DG["tol"] * max(abs(E_rec), 1e-300)   # v36k chart
    flat = lambda xs: np.concatenate([np.concatenate([q[0], q[1]]) for q in xs])   # v36k chart
    unflat = lambda xm: [(xm[2 * n * k: 2 * n * k + n].copy(), xm[2 * n * k + n: 2 * n * (k + 1)].copy()) for k in range(3)]   # v36k chart
    hist = []; conv = False; why = None; n_solves = 0; dE = None; dE_prev = None; dfields = None; cols1 = None; base1 = None; x_new = None; nodeE = None; nres = None; tinfos = None   # v36k chart
    masks_prev = None; changed_hist = []; clipped = 0   # v36k chart
    for it in range(1, DG["cap"] + 1):   # v36k chart
        Nbar = w_nodes[0] * x[0][0] + w_nodes[1] * x[1][0] + w_nodes[2] * x[2][0]; vbar = w_nodes[0] * x[0][1] + w_nodes[1] * x[1][1] + w_nodes[2] * x[2][1]   # v36k chart
        eig = eigh_all(chs, Nbar, vbar, bool(a.streams)); cols1 = {}   # v36k chart
        for ch in chs:   # v36k chart
            e_b, U_b = eig.pop(ch.kappa)   # v36k chart
            ph = torch.exp(-1j * e_b.to(torch.complex128) * tau_s)   # v36k chart
            prop = lambda X, U_b=U_b, ph=ph: U_b @ (ph[:, None] * (U_b.conj().T @ X))   # v36k chart
            sea0, dust0 = cols0[ch.kappa]   # v36k chart
            cols1[ch.kappa] = (prop(sea0), prop(dust0) if dust0.shape[1] else dust0)   # v36k chart
            del e_b, U_b   # v36k chart
        del eig   # v36k chart
        base1 = dg_densities_v36f(chs, st, cols1); eps1, J1 = dg_pairing_v36f(chs, base1, dust_only)   # v36k chart
        x_new = []; nodeE = []; tinfos = []; nres = []; masks = []   # v36k chart
        for k in range(3):   # v36k chart
            s_k = s_nodes[k]   # v36k chart
            base_s = {ch.kappa: tuple((1.0 - s_k) * b0 + s_k * b1 for b0, b1 in zip(base0[ch.kappa], base1[ch.kappa])) for ch in chs}   # v36k chart
            Nk, vk, Ek, info_k, ti_k, mk, _r = dgk_node_v36k(a, chs, st, R, base_s, x[k][0], x[k][1], anchor)   # v36k chart
            n_solves += 1   # v36k chart
            if Nk is None:   # v36k chart
                why = "the field solve failed at node %d (s = %.6f) of iteration %d: %s" % (k + 1, s_k, it, info_k.get("failed"))   # v36k chart
                break   # v36k chart
            x_new.append((Nk, vk)); nodeE.append(Ek); tinfos.append(ti_k); masks.append(mk)   # v36k chart
            nres.append(max(float(np.max(np.abs(Nk - x[k][0]))), float(np.max(np.abs(vk - x[k][1])))))   # v36k chart
        if why is not None:   # v36k chart
            break   # v36k chart
        changed = 0 if masks_prev is None else sum(int(not np.array_equal(masks[k][kap], masks_prev[k][kap])) for k in range(3) for kap in masks[k])   # v36k chart
        masks_prev = masks; changed_hist.append(changed)   # v36k chart
        Nbar_n = w_nodes[0] * x_new[0][0] + w_nodes[1] * x_new[1][0] + w_nodes[2] * x_new[2][0]; vbar_n = w_nodes[0] * x_new[0][1] + w_nodes[1] * x_new[1][1] + w_nodes[2] * x_new[2][1]   # v36k chart
        dE_prev = dE   # v36k chart
        dE = float(np.sum((Nbar_n - Nbar) * (eps1 - eps0) + (vbar_n - vbar) * (J1 - J0)))   # v36k chart
        dfields = max(nres)   # v36k chart
        chg = (abs(dE - dE_prev) / abs(E_rec)) if (dE_prev is not None and E_rec) else None   # v36k chart
        wocc = [ti.get("min_occupied_weight") for ti_k in tinfos for ti in ti_k.values() if ti.get("min_occupied_weight") is not None]   # v36k chart
        wemp = [ti.get("max_empty_weight") for ti_k in tinfos for ti in ti_k.values() if ti.get("max_empty_weight") is not None]   # v36k chart
        wmin = min(wocc) if wocc else None; wmax = max(wemp) if wemp else None   # v36k chart
        hist.append([it, dE / E_rec, chg, dfields, changed] + [q / E_rec for q in nodeE])   # v36k chart
        print("  v36k chart T/rs %.4f sub-step %d/%d iteration %d: energy defect %+.3e of E (change between iterates %s), node fields' change %.2e [%s], node E/E_rec - 1 [%s]; tracked set changed at %d node-channels; weights min occupied %s, max empty %s" % (   # v36k chart
            T0 / rs, j + 1, nsub, it, dE / E_rec, fmt(chg, "%.3e"), dfields, " ".join("%.1e" % q for q in nres), " ".join("%+.3e" % (q / E_rec - 1.0) for q in nodeE), changed, fmt(wmin, "%.5f"), fmt(wmax, "%.2e")), flush=True)   # v36k chart
        if abs(dE) < tol and dfields < a.stop_fields and changed == 0:   # v36k chart
            conv = True   # v36k chart
            break   # v36k chart
        xf = flat(x); gf = flat(x_new)   # v36k chart
        if mixer is not None:   # v36k chart
            if changed:   # v36k chart
                mixer.reset()   # v36k chart
            xm, _used = mixer.step(xf, gf)   # v36k chart
        else:   # v36k chart
            xm = xf + DGK["beta"] * (gf - xf)   # v36k chart
        if DGK["max_change"] > 0.0:   # v36k chart: the trust region: the whole update scaled so that no node's fields move by more than --dg-max-change
            big = float(np.max(np.abs(xm - xf)))   # v36k chart
            if big > DGK["max_change"]:   # v36k chart
                xm = xf + (xm - xf) * (DGK["max_change"] / big); clipped += 1   # v36k chart
        x = unflat(xm)   # v36k chart
    ent = {"sub": j + 1, "nsub": nsub, "level": level, "tau_sub": tau_s, "from_T_over_rs": T0 / rs, "to_T_over_rs": (T0 + tau_s) / rs, "converged": conv, "iterations": len(hist), "why": why, "start": start,   # v36k chart
           "energy_defect_final": ((dE / E_rec) if dE is not None else None), "energy_change_between_iterates_final": (hist[-1][2] if hist else None), "fields_change_final": dfields,   # v36k chart
           "node_residuals_final": nres, "node_E": nodeE, "node_E_over_E_rec": ([q / E_rec for q in nodeE] if nodeE else None), "E0": E0, "changed_hist": changed_hist, "clipped": clipped,   # v36k chart
           "track_nodes_final": tinfos, "history": hist, "r_chain_over_E_rec": None, "gate1": None, "E_end": None, "chart_disagreement": None, "endpoint_passes": None, "track_end": None}   # v36k chart
    cols_out = None; base_out = None; nodes = None; anchor_end = None; f_end = None; E_end = None   # v36k chart
    if conv:   # v36k chart: the endpoint evaluation at s = 1 on the chart: gate 1, the convention's check, the next anchor
        L1 = dg_lagrange_v36f(1.0)   # v36k chart
        N1 = L1[0] * x_new[0][0] + L1[1] * x_new[1][0] + L1[2] * x_new[2][0]; v1 = L1[0] * x_new[0][1] + L1[1] * x_new[1][1] + L1[2] * x_new[2][1]   # v36k chart
        passes = []; Ee = None; ti_e = {}; refs_e = {}   # v36k chart
        for p_ in range(DGK["end_passes"]):   # v36k chart
            last = (p_ == DGK["end_passes"] - 1)   # v36k chart
            Ne, ve, Ee, info_e, ti_e, _m, refs_e = dgk_node_v36k(a, chs, st, R, base1, N1, v1, anchor, keep_ref=last, check=last)   # v36k chart
            n_solves += 1   # v36k chart
            if Ne is None:   # v36k chart
                conv = False; why = "the field solve failed at the endpoint evaluation (pass %d): %s" % (p_ + 1, info_e.get("failed"))   # v36k chart
                break   # v36k chart
            passes.append(max(float(np.max(np.abs(Ne - N1))), float(np.max(np.abs(ve - v1)))))   # v36k chart
            N1, v1 = Ne, ve   # v36k chart
        if conv:   # v36k chart
            E_end = Ee; f_end = (N1, v1); anchor_end = (refs_e if anchor is not None else None)   # v36k chart
            cols_out = cols1; base_out = base1; nodes = (np.stack([q[0] for q in x_new]), np.stack([q[1] for q in x_new]))   # v36k chart
            r_rel = (E_end - E0) / E_rec   # v36k chart
            ent.update({"E_end": E_end, "r_chain_over_E_rec": r_rel, "gate1": ("pass" if abs(r_rel) <= DGK["chain_tol"] else "exceeded"), "endpoint_passes": passes,   # v36k chart
                        "chart_disagreement": sum(int(ti.get("convention_differs") or 0) for ti in ti_e.values()), "track_end": ti_e})   # v36k chart
    ent["converged"] = conv; ent["why"] = why; ent["eigendecompositions"] = TM["eigh_n"] - n_eigh0; ent["solves"] = n_solves   # v36k chart
    sync_dev(); ent["seconds"] = time.time() - t0   # v36k chart
    print("T/rs %.4f v36k chart sub-step %d/%d (tau %.4g) to T/rs %.4f: %s in %d iterations; energy defect %s of E (change between iterates %s), node fields %s; gate 1 r_chain %s of E (%s); the convention would reassign %s levels at the end; tracked-set changes %d, clipped updates %d; eigh %d, solves %d, %.1f s; start: %s%s" % (   # v36k chart
        T0 / rs, j + 1, nsub, tau_s, (T0 + tau_s) / rs, "converged" if conv else "NOT CONVERGED", len(hist), fmt(ent["energy_defect_final"], "%+.3e"), fmt(ent["energy_change_between_iterates_final"], "%.3e"), fmt(dfields, "%.2e"),   # v36k chart
        fmt(ent["r_chain_over_E_rec"], "%+.3e"), ent["gate1"] or "-", ent["chart_disagreement"] if ent["chart_disagreement"] is not None else "-", sum(changed_hist), clipped, ent["eigendecompositions"], n_solves, ent["seconds"], start, ("; " + why) if why else ""), flush=True)   # v36k chart
    return ent, cols_out, base_out, nodes, anchor_end, f_end, E_end   # v36k chart


def dg_step_v36k(a, chs, st, R, base0, f_acc, E_rec, T, rs):   # v36k chart
    """version 36k: the chart-frozen discrete-gradient step from the record at T to T + tau (--dg-chart frozen): the record's accepted reference as the frozen chart (dgk_anchor_v36k); the step as 2^L sub-steps of tau/2^L, L the sticky level DGK["level"], raised by one when a sub-step's fixed point does not converge within the cap, when its tracked set changed at its last iterate or when gate 1 exceeds --dg-chain-tol, up to --dg-sub-max (else the run ends at the record); the record at T + tau the composition. On success st's columns are replaced by the final state, DG["nodes"] by the last sub-step's node fields, DGK["sub_len"] and DGK["level"] (lowered after a quiet record, never below --dg-sub-start) updated. Returns the step's entry (the next record's dg_v36f; v36f's keys kept)."""   # v36k chart
    t0 = time.time()   # v36k chart
    if "eigh_n" not in TM:   # v36k chart
        tm_reset()   # v36k chart
    n_eigh0 = TM["eigh_n"]   # v36k chart
    anchor0 = dgk_anchor_v36k(a, chs, st, f_acc)   # v36k chart
    anchor_info = ({str(k_): q[3] for k_, q in anchor0.items()} if anchor0 is not None else None)   # v36k chart
    cols0 = {ch.kappa: (st[ch.kappa]["sea"], st[ch.kappa]["dust"]) for ch in chs}   # v36k chart
    level = int(DGK["level"]); attempts = []; conv = False; subs = []; why = None   # v36k chart
    cols = None; nodes_last = None; E_end = None; nsub = 1; tau_s = float(a.tau)   # v36k chart
    while True:   # v36k chart
        nsub = 2 ** level; tau_s = float(a.tau) / nsub   # v36k chart
        cols = cols0; base = base0; f = (np.array(f_acc[0], float), np.array(f_acc[1], float)); Ecur = E_rec; anchor = anchor0   # v36k chart
        prev = DG["nodes"] if (DG["nodes"] is not None and DGK["sub_len"] is not None and abs(float(DGK["sub_len"]) - tau_s) <= 1e-9 * float(a.tau)) else None   # v36k chart
        subs = []; ok = True; why = None   # v36k chart
        for j in range(nsub):   # v36k chart
            x0, start = dgk_start_v36k(prev, f)   # v36k chart
            ent, cols_n, base_n, nodes_n, anchor_n, f_n, E_n = dgk_substep_v36k(a, chs, st, R, cols, base, f, Ecur, E_rec, anchor, tau_s, x0, start, T + j * tau_s, rs, level, j, nsub)   # v36k chart
            subs.append(ent)   # v36k chart
            if not ent["converged"]:   # v36k chart
                ok = False; why = ent.get("why") or "the fixed point did not converge within %d iterations at sub-step %d/%d (level %d)" % (DG["cap"], j + 1, nsub, level)   # v36k chart
                break   # v36k chart
            if ent["gate1"] == "exceeded" and level < DGK["sub_max"]:   # v36k chart
                ok = False; why = "gate 1: r_chain %+.3e of E above %g at sub-step %d/%d (level %d)" % (ent["r_chain_over_E_rec"], DGK["chain_tol"], j + 1, nsub, level)   # v36k chart
                break   # v36k chart
            cols, base, f, Ecur, anchor, prev = cols_n, base_n, f_n, E_n, anchor_n, nodes_n   # v36k chart
        attempts.append({"level": level, "nsub": nsub, "tau_sub": tau_s, "converged": ok, "why": why, "iterations": sum(q["iterations"] for q in subs), "substeps": [dict((k_, v_) for k_, v_ in q.items() if k_ != "history") for q in subs]})   # v36k chart
        if ok:   # v36k chart
            conv = True; nodes_last = prev; E_end = Ecur   # v36k chart
            break   # v36k chart
        print("T/rs %.4f v36k chart: %s%s" % (T / rs, why, (("; the sub-step halved (level %d)" % (level + 1)) if level < DGK["sub_max"] else ("; the deepest level (%d) reached" % level))), flush=True)   # v36k chart
        if level >= DGK["sub_max"]:   # v36k chart
            break   # v36k chart
        level += 1   # v36k chart
    anchor0 = None; anchor = None   # v36k chart
    dsum = (sum(q["energy_defect_final"] for q in subs if q.get("energy_defect_final") is not None) if subs else None)   # v36k chart
    rsum = (sum(q["r_chain_over_E_rec"] for q in subs if q.get("r_chain_over_E_rec") is not None) if (conv and subs) else None)   # v36k chart
    if conv:   # v36k chart
        for ch in chs:   # v36k chart
            st[ch.kappa]["sea"], st[ch.kappa]["dust"] = cols[ch.kappa]   # v36k chart
        DG["nodes"] = nodes_last; DGK["sub_len"] = tau_s; DG["steps"] += 1   # v36k chart
        quiet = all((q["iterations"] <= max(3, DG["cap"] // 3)) and (max(q["changed_hist"] or [0]) == 0) and (abs(q["r_chain_over_E_rec"] or 0.0) <= DGK["chain_tol"] / 16.0) for q in subs)   # v36k chart
        DGK["level"] = max(int(DGK["level_min"]), level - 1) if (quiet and level > int(DGK["level_min"])) else level   # v36k chart
    last = subs[-1] if subs else {}   # v36k chart
    ent = {"from_T": T, "from_T_over_rs": T / rs, "to_T_over_rs": (T + a.tau) / rs, "tau": a.tau, "converged": conv, "iterations": sum(q["iterations"] for q in subs), "why": (None if conv else why),   # v36k chart
           "start": (subs[0]["start"] if subs else None), "gauss_nodes": [q[0] for q in GL3_V36F], "gauss_weights": [q[1] for q in GL3_V36F], "tolerance_energy_relative": DG["tol"], "tolerance_fields": a.stop_fields,   # v36k chart
           "cap": DG["cap"], "mix": DGK["mix"], "chart": "frozen", "level": level, "nsub": nsub, "tau_sub": tau_s, "next_level": DGK["level"], "attempts": len(attempts), "attempt_list": attempts, "substeps": subs,   # v36k chart
           "energy_defect_final": dsum, "energy_change_between_iterates_final": last.get("energy_change_between_iterates_final"), "fields_change_final": last.get("fields_change_final"),   # v36k chart
           "node_residuals_final": last.get("node_residuals_final"), "node_E": last.get("node_E"), "node_E_over_E_rec": last.get("node_E_over_E_rec"), "E_rec": E_rec,   # v36k chart
           "E_pred": ((E_rec * (1.0 + dsum)) if dsum is not None else None), "E_end": E_end, "r_chain_over_E_rec": rsum, "chart_disagreement_total": last.get("chart_disagreement"),   # v36k chart
           "tracked_set_changes": sum(sum(q["changed_hist"]) for q in subs), "continuation_nodes_final": last.get("track_nodes_final"), "track_end": last.get("track_end"), "history": last.get("history"),   # v36k chart
           "anchor": anchor_info, "eigendecompositions": TM["eigh_n"] - n_eigh0, "solves": sum(q.get("solves", 0) for q in subs)}   # v36k chart
    sync_dev()   # v36k chart
    ent["seconds"] = time.time() - t0   # v36k chart
    print("T/rs %.4f v36k chart step to T/rs %.4f: %s; level %d (%d sub-steps of tau %.4g; %d attempts; next level %d); %d iterations; energy defect %s of E; gate 1 r_chain %s of E; E_end/E_rec - 1 = %s; E_pred/E_rec - 1 = %s; the convention would reassign %s levels; anchor: %s; eigh %d, solves %d, %.1f s%s" % (   # v36k chart
        T / rs, (T + a.tau) / rs, "converged" if conv else "NOT CONVERGED", level, nsub, tau_s, len(attempts), DGK["level"], ent["iterations"], fmt(dsum, "%+.3e"), fmt(rsum, "%+.3e"),   # v36k chart
        fmt(((E_end / E_rec - 1.0) if (E_end is not None and E_rec) else None), "%+.3e"), fmt(dsum, "%+.3e"), ent["chart_disagreement_total"] if ent["chart_disagreement_total"] is not None else "-",   # v36k chart
        dgk_anchor_summary_v36k(anchor_info), ent["eigendecompositions"], ent["solves"], ent["seconds"], ("; " + why) if (why and not conv) else ""), flush=True)   # v36k chart
    return ent   # v36k chart


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)'''),
("run label",
r'''    if DG["on"]:   # v36f dg
        label += "; v36f: " + dg_desc_v36f(a)   # v36f dg''',
r'''    if DG["on"]:   # v36f dg
        label += "; v36f: " + dg_desc_v36f(a)   # v36f dg
    if DGK["on"]:   # v36k chart
        label += "; v36k: " + dgk_desc_v36k(a)   # v36k chart'''),
("resume slice step",
r'''        DG["nodes"] = (np.asarray(Z["dg_nodes_N"], float), np.asarray(Z["dg_nodes_v"], float)) if ("dg_nodes_N" in Z.files and "dg_nodes_v" in Z.files) else None   # v36f dg
        dgr = dg_step_v36f(a, chs, st, R, base, (N_s.copy(), v_s.copy()), float(rec_s["E"]), Ts, rs)   # v36f dg''',
r'''        DG["nodes"] = (np.asarray(Z["dg_nodes_N"], float), np.asarray(Z["dg_nodes_v"], float)) if ("dg_nodes_N" in Z.files and "dg_nodes_v" in Z.files) else None   # v36f dg
        if DGK["on"]:   # v36k chart: the sub-step length and level of the step that produced the slice's state (a v36f slice: full steps, the flags' level)
            DGK["sub_len"] = float(Z["dg_sub_len_v36k"]) if "dg_sub_len_v36k" in Z.files else (float(a.tau) if DG["nodes"] is not None else None)   # v36k chart
            DGK["level"] = int(Z["dg_level_v36k"]) if "dg_level_v36k" in Z.files else int(DGK["level"])   # v36k chart
        dgr = dg_step_v36k(a, chs, st, R, base, (N_s.copy(), v_s.copy()), float(rec_s["E"]), Ts, rs) if DGK["on"] else dg_step_v36f(a, chs, st, R, base, (N_s.copy(), v_s.copy()), float(rec_s["E"]), Ts, rs)   # v36k chart'''),
("slice arrays",
r'''                arrs["dg_nodes_N"] = np.asarray(DG["nodes"][0], float); arrs["dg_nodes_v"] = np.asarray(DG["nodes"][1], float)   # v36f dg''',
r'''                arrs["dg_nodes_N"] = np.asarray(DG["nodes"][0], float); arrs["dg_nodes_v"] = np.asarray(DG["nodes"][1], float)   # v36f dg
            if DGK["on"] and DGK["sub_len"] is not None:   # v36k chart: the sub-step length and the level of the next step (a restart continues as the uninterrupted run)
                arrs["dg_sub_len_v36k"] = np.asarray(float(DGK["sub_len"])); arrs["dg_level_v36k"] = np.asarray(int(DGK["level"]))   # v36k chart'''),
("record entry",
r'''                d_["E_minus_E_prev_over_E0"] = ((E - recs[-1]["E"]) / E0) if (recs and recs[-1].get("E") is not None) else None   # v36f dg''',
r'''                d_["E_minus_E_prev_over_E0"] = ((E - recs[-1]["E"]) / E0) if (recs and recs[-1].get("E") is not None) else None   # v36f dg
                if DGK["on"]:   # v36k chart: this record's charge against the step's chart endpoint: the record solve's reassignment move (the readout's third bin)
                    d_["E_minus_E_end_over_E0"] = ((E - d_["E_end"]) / E0) if d_.get("E_end") is not None else None   # v36k chart
                    d_["E_minus_E_end_over_E_rec_prev"] = ((E - d_["E_end"]) / d_["E_rec"]) if (d_.get("E_end") is not None and d_.get("E_rec")) else None   # v36k chart'''),
("record line",
r'''                "; ".join("beyond %s r_s: number %.5f, energy %.5f, Killing split below 0 %.5f, 0 to m %.5f, m to b %.5f, above b %.5f" % (k_, q["number"], q["energy_density_route"], q["killing_split"]["below_0"], q["killing_split"]["0_to_m"], q["killing_split"]["m_to_b"], q["killing_split"]["above_b"]) for k_, q in f_.items()) or "-"), flush=True)   # v36f dg''',
r'''                "; ".join("beyond %s r_s: number %.5f, energy %.5f, Killing split below 0 %.5f, 0 to m %.5f, m to b %.5f, above b %.5f" % (k_, q["number"], q["energy_density_route"], q["killing_split"]["below_0"], q["killing_split"]["0_to_m"], q["killing_split"]["m_to_b"], q["killing_split"]["above_b"]) for k_, q in f_.items()) or "-"), flush=True)   # v36f dg
            if DGK["on"] and d_ is not None:   # v36k chart
                print("T/rs %.4f v36k chart: reassignment move (E - E_end)/E0 = %s; the step: level %d (%d sub-steps of tau %.4g, %d attempts), gate 1 r_chain %s of E, gate 2 defect %s of E, tracked-set changes %s, the convention would reassign %s levels at the step's end, anchor %s; the record solve converged: %s" % (   # v36k chart
                    T / rs, fmt(d_.get("E_minus_E_end_over_E0"), "%+.3e"), d_.get("level", 0), d_.get("nsub", 1), d_.get("tau_sub", a.tau), d_.get("attempts", 1), fmt(d_.get("r_chain_over_E_rec"), "%+.3e"), fmt(d_.get("energy_defect_final"), "%+.3e"),   # v36k chart
                    d_.get("tracked_set_changes"), d_.get("chart_disagreement_total"), dgk_anchor_summary_v36k(d_.get("anchor")), rec.get("converged", "-")), flush=True)   # v36k chart'''),
("propagation",
r'''            dgr = dg_step_v36f(a, chs, st, R, base, f_ref, E, T, rs)   # v36f dg''',
r'''            dgr = dg_step_v36k(a, chs, st, R, base, f_ref, E, T, rs) if DGK["on"] else dg_step_v36f(a, chs, st, R, base, f_ref, E, T, rs)   # v36k chart'''),
("arguments",
r'''    ap.add_argument("--dg-cap", dest="dg_cap", type=int, default=30, help="(v36f dg) at most this many iterations of the fixed point; the run ends at the record otherwise")   # v36f dg''',
r'''    ap.add_argument("--dg-cap", dest="dg_cap", type=int, default=30, help="(v36f dg) at most this many iterations of the fixed point; the run ends at the record otherwise")   # v36f dg
    ap.add_argument("--dg-chart", dest="dg_chart", default="continued", choices=["continued", "frozen"], help="(v36k) continued: the reference re-continued at every iterate (v36f); frozen: the record's accepted set held along the chord, with gates 1 and 2 and sub-steps")   # v36k chart
    ap.add_argument("--dg-mix", dest="dg_mix", default="inherit", choices=["inherit", "plain", "anderson"], help="(v36k) the dg fixed point's mixer under --dg-chart frozen (inherit: --mix)")   # v36k chart
    ap.add_argument("--dg-anderson-m", dest="dg_anderson_m", type=int, default=4, help="(v36k) the Anderson history length")   # v36k chart
    ap.add_argument("--dg-beta", dest="dg_beta", type=float, default=1.0, help="(v36k) the mixing parameter of the dg fixed point (the damping of plain, Anderson's beta)")   # v36k chart
    ap.add_argument("--dg-max-change", dest="dg_max_change", type=float, default=2e-2, help="(v36k) the trust region: no node's fields move by more than this per iterate (0: none)")   # v36k chart
    ap.add_argument("--dg-sub-max", dest="dg_sub_max", type=int, default=3, help="(v36k) the deepest sub-step level L (2^L sub-steps of tau/2^L); 0: no sub-steps")   # v36k chart
    ap.add_argument("--dg-sub-start", dest="dg_sub_start", type=int, default=0, help="(v36k) the level the steps start at and never go below (a restart reads the slice's level)")   # v36k chart
    ap.add_argument("--dg-chain-tol", dest="dg_chain_tol", type=float, default=1e-8, help="(v36k) gate 1: |E(C_1) - E(C_0)| on the chart above this (relative to the record's charge) halves the sub-step")   # v36k chart
    ap.add_argument("--dg-end-passes", dest="dg_end_passes", type=int, default=2, help="(v36k) the quasi-static evaluations at the step's end (gate 1 and the next anchor)")   # v36k chart'''),
("validation",
r'''    if DG["on"] and (a.dg_tol <= 0.0 or a.dg_cap < 1):   # v36f dg
        ap.error("version 36f: --dg-tol must be positive and --dg-cap at least 1")   # v36f dg''',
r'''    if DG["on"] and (a.dg_tol <= 0.0 or a.dg_cap < 1):   # v36f dg
        ap.error("version 36f: --dg-tol must be positive and --dg-cap at least 1")   # v36f dg
    DGK["on"] = bool(DG["on"] and a.dg_chart == "frozen"); DGK["mix"] = (a.mix if a.dg_mix == "inherit" else a.dg_mix); DGK["m"] = int(a.dg_anderson_m); DGK["beta"] = float(a.dg_beta)   # v36k chart
    DGK["max_change"] = float(a.dg_max_change); DGK["sub_max"] = int(a.dg_sub_max); DGK["chain_tol"] = float(a.dg_chain_tol); DGK["end_passes"] = int(a.dg_end_passes); DGK["level"] = int(a.dg_sub_start); DGK["level_min"] = int(a.dg_sub_start)   # v36k chart
    if a.dg_chart == "frozen" and not DG["on"]:   # v36k chart
        ap.error("version 36k: --dg-chart frozen needs --step dg")   # v36k chart
    if DGK["on"] and DGK["mix"] not in ("plain", "anderson"):   # v36k chart
        ap.error("version 36k: --dg-mix must resolve to plain or anderson")   # v36k chart
    if DGK["on"] and (DGK["m"] < 1 or not (0.0 < DGK["beta"] <= 1.0) or DGK["max_change"] < 0.0 or DGK["sub_max"] < 0 or not (0 <= DGK["level"] <= DGK["sub_max"]) or DGK["chain_tol"] <= 0.0 or DGK["end_passes"] < 1):   # v36k chart
        ap.error("version 36k: --dg-anderson-m >= 1, 0 < --dg-beta <= 1, --dg-max-change >= 0, 0 <= --dg-sub-start <= --dg-sub-max, --dg-chain-tol > 0, --dg-end-passes >= 1")   # v36k chart'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36k: the OLD string of '%s' occurs %d times in the source (1 required): the source is not input 01's engine_collapse_v36f.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36k: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36f.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36k.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36k: the reverse replacements do not restore the source (a NEW text overlaps another)")
    marked = sum(1 for l_ in out.splitlines() if ("# v36k chart" in l_ or "[v36k chart]" in l_))
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
