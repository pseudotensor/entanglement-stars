#!/usr/bin/env python
"""engine_ops_v3.py (builder analyst; engine spec 25.1, red team R27c's code points ruled): engine_ops_v2's continuation of R2 with
(1) the sliding window retired: past the Weyl test the reference is assigned in one step from the previous accepted record (m = 1; the
window's consistency number was 1 on every stored record, so no stored mask changes; the nonlocal assignment and the ||dh||_G = 0 branch
go with it); the Landau-Zener sub-steps (tau_LZ <= tau) as before; (2) a partly occupied block (occupied and empty vectors in one block)
gives the next record a Weyl gap of zero in both tests: info["weyl_gap"] = 0 there, else the occupied-to-empty gap between blocks, and
the test uses it; where only the inter-block gap passes, info["isolation_verified"] = True with "verified" False ("inter-block isolation
verified"); (3) a block is numerically degenerate only when its whole width e[j-1] - e[i] is below tol_rel times the band scale (no
transitive joining of successive near-equalities); only such blocks are rotated to their weight eigenbasis; (4) a rotated block whose
eigenvalues straddle zero gives its mixed vectors no energy sign: they enter neither pocket count (counted in info["straddling_blocks"]
and info["sign_free_vectors"]); (5) an exact tie at the occupation cut (the n_sea-th and the next weight within 1e-12) is broken by the
predecessor's assignment where the window's last entry carries a mask over the same energy-ordered indices and it decides the tie, else
by energy and flagged (info["ambiguous_tie"] = True); (6) inversions by block order: occupied vectors in a block above the block of the
lowest empty vector. The returned mask is over the (rotated) eigenbasis in energy order as before; the window list keeps its entries
{"ref", "f", "gap"} and may carry "mask"; continue_sea_v3 reads only its last entry."""
import math

import numpy as np

from engine_ops_v2 import gershgorin  # noqa: F401  (the same Gershgorin bound)


def blocks_of_v3(e, tol_rel):
    """blocks of eigenvalues whose whole width is below tol_rel times the band scale (each block starts at its first element)."""
    e = np.asarray(e, float)
    scale = max(float(np.max(np.abs(e))), 1e-300)
    out, i = [], 0
    while i < len(e):
        j = i + 1
        while j < len(e) and (e[j] - e[i]) < tol_rel * scale:
            j += 1
        out.append((i, j))
        i = j
    return out


def _assign_v3(torch, A, e_np, n_sea, tol_block):
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy().astype(float)
    blocks = blocks_of_v3(e_np, tol_block)
    rot = {}
    blk_id = np.empty(len(e_np), dtype=np.int64)
    for b, (i, j) in enumerate(blocks):
        blk_id[i:j] = b
        if j - i > 1:
            M = A[:, i:j].conj().T @ A[:, i:j]
            lam, V = torch.linalg.eigh(M)
            w[i:j] = lam.real.cpu().numpy()
            rot[(i, j)] = V
    n_inter = int(((w > 0.1) & (w < 0.9)).sum())
    return w, blocks, rot, blk_id, n_inter


def assign_max_overlap_v3(ref_prev, e, U, n_sea, tol_block=1e-8, tol_fallback=1e-6, prev_mask=None):
    """the assignment with (2) to (6) of the docstring. Returns (ref, info, mask)."""
    import torch
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    A = ref_prev.conj().T @ U
    w, blocks, rot, blk_id, n_inter = _assign_v3(torch, A, e_np, n_sea, tol_block)
    tol_used = tol_block
    if n_inter > 0 and tol_fallback and tol_fallback > tol_block:
        w2, blocks2, rot2, blk2, n2 = _assign_v3(torch, A, e_np, n_sea, tol_fallback)
        if n2 < n_inter:
            w, blocks, rot, blk_id, n_inter, tol_used = w2, blocks2, rot2, blk2, n2, tol_fallback
    order = list(np.lexsort((e_np, -w)))                    # weight descending, ties by energy ascending
    ambiguous = False; tie_by_pred = False
    if 0 < n_sea < len(order):
        wc = w[order[n_sea - 1]]
        tied = [k for k in order if abs(w[k] - wc) < 1e-12]
        if len(tied) > 1 and any(k in order[n_sea:] for k in tied):
            above = [k for k in order[:n_sea] if k not in tied]
            need = n_sea - len(above)
            pm = np.asarray(prev_mask, bool) if prev_mask is not None else None
            chosen = None
            if pm is not None and len(pm) == len(e_np):
                pref = [k for k in tied if pm[k]]
                if len(pref) == need:
                    chosen = pref; tie_by_pred = True
            if chosen is None:
                ambiguous = True
                chosen = sorted(tied, key=lambda k: e_np[k])[:need]
            order = above + chosen + [k for k in order if k not in above and k not in chosen]
    occ = np.zeros(len(e_np), bool); occ[np.asarray(order[:n_sea], dtype=np.int64)] = True
    bsize = np.array([j - i for (i, j) in blocks])
    plain = np.where(occ & (bsize[blk_id] == 1))[0]
    cols = []
    if len(plain):
        cols.append(U[:, torch.as_tensor(plain.astype(np.int64), device=U.device)])
    for (i, j), V in rot.items():
        sel = np.where(occ[i:j])[0]
        if len(sel):
            cols.append(U[:, i:j] @ V[:, torch.as_tensor(sel, dtype=torch.int64, device=U.device)])
    ref = torch.cat(cols, dim=1) if cols else U[:, :0]
    wo = w[occ]; we = w[~occ]
    # (4) sign-free vectors: those of rotated blocks straddling zero
    signless = np.zeros(len(e_np), bool); straddle = 0
    for (i, j) in rot:
        if e_np[i] < 0.0 < e_np[j - 1]:
            signless[i:j] = True; straddle += 1
    pockets_occ_pos = int(np.sum(occ & (e_np > 0.0) & ~signless)); holes_emp_neg = int(np.sum(~occ & (e_np < 0.0) & ~signless))
    # (6) inversions by block order
    inv = 0
    if (~occ).any() and occ.any():
        k_low = int(np.argmin(np.where(~occ, e_np, np.inf)))
        inv = int(np.sum(occ & (blk_id > blk_id[k_low])))
    gb = None
    has_o = np.zeros(len(blocks), bool); has_e = np.zeros(len(blocks), bool)
    has_o[np.unique(blk_id[occ])] = True; has_e[np.unique(blk_id[~occ])] = True
    for b in range(len(blocks) - 1):
        if (has_o[b] and has_e[b + 1]) or (has_e[b] and has_o[b + 1]):
            d_ = float(e_np[blocks[b + 1][0]] - e_np[blocks[b][1] - 1])
            gb = d_ if gb is None else min(gb, d_)
    partial = int(sum(1 for b in range(len(blocks)) if has_o[b] and has_e[b]))
    info = {"n_occ": int(occ.sum()), "n_sea": n_sea, "n_intermediate": n_inter, "min_occupied_weight": float(wo.min()) if len(wo) else None,
            "max_empty_weight": float(we.max()) if len(we) else None, "inversions": inv, "n_pocket": pockets_occ_pos + holes_emp_neg,
            "pockets_occupied_positive": pockets_occ_pos, "holes_empty_negative": holes_emp_neg, "nneg": int((e_np < 0).sum()),
            "gap_between_blocks": gb, "weyl_gap": (0.0 if partial else gb), "partly_occupied_blocks": partial, "rotated_blocks": len(rot),
            "straddling_blocks": straddle, "sign_free_vectors": int(signless.sum()), "tol_block": tol_used, "ambiguous_tie": ambiguous,
            "tie_by_predecessor": tie_by_pred, "empty_weight_sum": float(n_sea - wo.sum()) if len(wo) else None}
    return ref, info, occ


def continue_sea_v3(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of, window_max=1):
    """the continuation of the docstring from the window's last entry. Returns (ref, info, mask)."""
    prev = window[-1]
    dH = h_of(ch, f_cur[0] - prev["f"][0], f_cur[1] - prev["f"][1])
    nG = gershgorin(dH)
    s_T = nG / tau
    tau_LZ = (1.0 / (c_LZ * math.sqrt(s_T))) if s_T > 0.0 else math.inf
    info_extra = {"norm_dH_G": nG, "s_T": s_T, "tau_LZ": tau_LZ, "c_LZ": c_LZ, "m": 1, "m_eff": 1, "window_consistency": None}
    g_prev = prev.get("weyl_gap", prev.get("gap"))
    g_iso = prev.get("gap")
    if tau_LZ <= tau:
        nsub = max(1, int(round(c_LZ * math.sqrt(tau * nG))))
        ref = prev["ref"]; g0 = g_prev; gi = g_iso; verified = True; iso = True; margin = math.inf
        dh = nG / nsub
        pm = prev.get("mask")
        for k in range(1, nsub + 1):
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                s_ = k / nsub
                e_s, U_s = eig_at(ch, prev["f"][0] + s_ * (f_cur[0] - prev["f"][0]), prev["f"][1] + s_ * (f_cur[1] - prev["f"][1]))
            if g0 is None or 2.0 * dh >= g0:
                verified = False
            if gi is None or 2.0 * dh >= gi:
                iso = False
            if g0 is not None:
                margin = min(margin, g0 - 2.0 * dh)
            ref, info, mask = assign_max_overlap_v3(ref, e_s, U_s, n_sea, prev_mask=pm)
            g0 = info["weyl_gap"]; gi = info["gap_between_blocks"]; pm = mask
        info.update(info_extra)
        info.update({"mode": "LZ sub-steps (nsub %d)" % nsub, "nsub": nsub, "verified": verified, "isolation_verified": iso,
                     "margin_min": (margin if math.isfinite(margin) else None)})
        return ref, info, mask
    verified = bool(g_prev is not None and 2.0 * nG < g_prev)
    iso = bool(g_iso is not None and 2.0 * nG < g_iso)
    ref, info, mask = assign_max_overlap_v3(prev["ref"], e_cur, U_cur, n_sea, prev_mask=prev.get("mask"))
    info.update(info_extra)
    info.update({"mode": "one step", "nsub": 1, "verified": verified, "isolation_verified": iso,
                 "margin_min": ((g_prev - 2.0 * nG) if g_prev is not None else None)})
    return ref, info, mask
