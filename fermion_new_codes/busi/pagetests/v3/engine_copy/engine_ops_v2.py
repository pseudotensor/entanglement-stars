#!/usr/bin/env python
"""engine_ops_v2.py (builder analyst; engine spec 21.2, the lead's "engine_ops.continue_sea_v2"; the operators of engine_ops_v1 are
unchanged and not repeated here): R2's continuation, Landau-Zener matched, by maximum overlap, never rejected [a declaration of this lane,
the owner's confirmation asked].
assign_max_overlap: at one configuration (the current eigenvectors U, eigenvalues e) from a previous reference V (orthonormal columns):
the weights w_k = <phi_k|V V^dag|phi_k>; eigenvalues within tol_block of the band scale form a block, whose weight sub-matrix
(V^dag U_b)^dag (V^dag U_b) is diagonalized and the block's eigenvectors rotated to its eigenbasis (the eigh basis of a degenerate block
is arbitrary); the n_sea largest weights occupied, the count exact (ties by energy); when the 1e-8 blocks leave intermediate weights (in
(0.1, 0.9)) the blocks at 1e-6 are tried and kept if they leave fewer (BUILDER, 21.2); returned with the count of intermediate weights,
the smallest occupied and largest empty weight, the inversions (occupied levels above the lowest empty level, outside its block), the
pockets (occupied levels above zero, empty levels below it), the occupied-to-empty gap between blocks (the next Weyl test's), the mask
over the (rotated) eigenbasis and the block tolerance used.
continue_sea_v2: per channel and re-solve, with dh the change of h_m from the previous accepted record to the trial fields and ||.||_G
its Gershgorin bound (the largest absolute row sum): s_T = ||dh||_G/tau, tau_LZ = 1/(c_LZ sqrt(s_T)); if tau_LZ <= tau, nsub =
max(1, round(c_LZ sqrt(tau ||dh||_G))) sub-steps along the linear path of the fields (the eigendecompositions at the intermediate
configurations by eig_at), the assignment at each; else m = round(tau_LZ/tau), at most 4 (and at most the records held), and the
reference assigned in one step from the accepted R2 m records back (a sliding window: every record keeps its own continued R2), the
weight of the result on the one-step assignment from the previous record printed for m > 1 (the window's consistency). "verified
(Weyl)" where every sub-step's (or the window's) 2 ||dh_sub||_G is below the occupied-to-empty gap between blocks at its start; else the
LZ convention. The window: a list (oldest first) of the accepted records' {"ref", "f": (N, v), "gap"}, the last the previous record."""
import math

import numpy as np


def blocks_of(e, tol_rel):
    e = np.asarray(e, float)
    scale = max(float(np.max(np.abs(e))), 1e-300)
    out, i = [], 0
    while i < len(e):
        j = i + 1
        while j < len(e) and abs(e[j] - e[j - 1]) < tol_rel * scale:
            j += 1
        out.append((i, j))
        i = j
    return out


def _assign(torch, A, e_np, U, n_sea, tol_block):
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy().astype(float)
    blocks = blocks_of(e_np, tol_block)
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


def assign_max_overlap(ref_prev, e, U, n_sea, tol_block=1e-8, tol_fallback=1e-6):
    """21.2's assignment (docstring). Returns (ref, info, mask)."""
    import torch
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    A = ref_prev.conj().T @ U
    w, blocks, rot, blk_id, n_inter = _assign(torch, A, e_np, U, n_sea, tol_block)
    tol_used = tol_block
    if n_inter > 0 and tol_fallback and tol_fallback > tol_block:
        w2, blocks2, rot2, blk2, n2 = _assign(torch, A, e_np, U, n_sea, tol_fallback)
        if n2 < n_inter:
            w, blocks, rot, blk_id, n_inter, tol_used = w2, blocks2, rot2, blk2, n2, tol_fallback
    order = np.lexsort((e_np, -w))                     # weight descending, ties by energy ascending
    occ = np.zeros(len(e_np), bool); occ[order[:n_sea]] = True
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
    eo = e_np[occ]; ee = e_np[~occ]
    scale = max(float(np.max(np.abs(e_np))), 1e-300)
    inv = 0
    if len(ee) and len(eo):
        k_low = int(np.argmin(np.where(~occ, e_np, np.inf)))
        inv = int(np.sum((eo > e_np[k_low] + tol_used * scale) & (blk_id[occ] != blk_id[k_low])))
    pockets_occ_pos = int(np.sum(eo > 0.0)); holes_emp_neg = int(np.sum(ee < 0.0))
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
            "gap_between_blocks": gb, "partly_occupied_blocks": partial, "rotated_blocks": len(rot), "tol_block": tol_used,
            "empty_weight_sum": float(n_sea - wo.sum()) if len(wo) else None}
    return ref, info, occ


def gershgorin(dH):
    """the largest absolute row sum of a host sparse matrix."""
    return float(np.max(np.asarray(abs(dH).sum(axis=1)).ravel())) if dH.nnz else 0.0


def continue_sea_v2(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of, window_max=4):
    """docstring. h_of(ch, dN, dv): the host sparse h_m[dN, dv]; eig_at(ch, N, v) -> (e, U). Returns (ref, info, mask)."""
    prev = window[-1]
    dH = h_of(ch, f_cur[0] - prev["f"][0], f_cur[1] - prev["f"][1])
    nG = gershgorin(dH)
    s_T = nG / tau
    tau_LZ = (1.0 / (c_LZ * math.sqrt(s_T))) if s_T > 0.0 else math.inf
    info_extra = {"norm_dH_G": nG, "s_T": s_T, "tau_LZ": tau_LZ, "c_LZ": c_LZ}
    if tau_LZ <= tau:
        nsub = max(1, int(round(c_LZ * math.sqrt(tau * nG))))
        ref = prev["ref"]; g0 = prev["gap"]; verified = True; margin = math.inf
        dh = nG / nsub
        for k in range(1, nsub + 1):
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                s_ = k / nsub
                e_s, U_s = eig_at(ch, prev["f"][0] + s_ * (f_cur[0] - prev["f"][0]), prev["f"][1] + s_ * (f_cur[1] - prev["f"][1]))
            if g0 is None or 2.0 * dh >= g0:
                verified = False
            if g0 is not None:
                margin = min(margin, g0 - 2.0 * dh)
            ref, info, mask = assign_max_overlap(ref, e_s, U_s, n_sea)
            g0 = info["gap_between_blocks"]
        info.update(info_extra)
        info.update({"mode": "LZ sub-steps (nsub %d)" % nsub, "nsub": nsub, "m": 1, "m_eff": 1, "verified": verified,
                     "margin_min": (margin if math.isfinite(margin) else None), "window_consistency": None})
        return ref, info, mask
    m = min(window_max, max(1, int(round(tau_LZ / tau)))) if math.isfinite(tau_LZ) else window_max   # no change of h_m: the longest window
    m_eff = min(m, len(window))
    base = window[-m_eff]
    if m_eff > 1:
        nW = gershgorin(h_of(ch, f_cur[0] - base["f"][0], f_cur[1] - base["f"][1]))
    else:
        nW = nG
    verified = bool(base["gap"] is not None and 2.0 * nW < base["gap"])
    ref, info, mask = assign_max_overlap(base["ref"], e_cur, U_cur, n_sea)
    cons = None
    if m_eff > 1:
        ref1, _i1, _m1 = assign_max_overlap(prev["ref"], e_cur, U_cur, n_sea)
        cons = float(((ref1.conj().T @ ref).abs() ** 2).sum()) / max(n_sea, 1)
        del ref1
    info.update(info_extra)
    info.update({"mode": "window (m %d of %d)" % (m_eff, m), "nsub": 1, "m": m, "m_eff": m_eff, "verified": verified,
                 "margin_min": ((base["gap"] - 2.0 * nW) if base["gap"] is not None else None), "norm_dH_window_G": nW, "window_consistency": cons})
    return ref, info, mask
