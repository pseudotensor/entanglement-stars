#!/usr/bin/env python
"""engine_ops_v34_cert.py (builder analyst, the HORIZON version 2 builder; engine spec 39.4 (2), the certified S11 segment): the
reference's continuation with the Weyl certificate on every sub-step, for engine_collapse_v34.py --weyl-certified (never used otherwise).
The lead's rule (39.4 (2), verbatim): "with ceil(2 ||dh||/gap) sub-steps per channel so that the Weyl certificate passes on every channel
(a channel that still fails marked)". Per channel and evaluation, from the window's last entry (the previous accepted record: its reference,
its applied fields, its Weyl gap and its occupation mask in energy order) to the trial fields:
- ||dh|| = the Gershgorin bound of h_m[Delta N, Delta v] (engine_ops_v3.gershgorin, the same bound version 3 uses), gap = the previous
  record's Weyl gap (version 3's weyl_gap: the smallest distance between adjacent blocks one of which holds an occupied and the other an empty
  level, zero when a block is partly occupied); n_lead = ceil(2 ||dh||/gap) uniform sub-steps of the linear path in (N, v) (h_m is linear in
  them); each sub-step tested at its start, 2 ||dh||/n < the Weyl gap there (Weyl's inequality: no occupied level meets an empty one inside
  the sub-step); a sub-step whose start gap is too small is halved until it passes, at most --weyl-refine-max times (the pieces printed); a
  piece that still fails marks the channel ("still fails");
- route "full": at every piece's end the eigendecomposition of h_m there and version 3's assignment by maximum overlap
  (engine_ops_v3.assign_max_overlap_v3, the engine's own rule), its Weyl gap the next piece's start gap: version 3's sub-step loop with the
  lead's count;
- route "eigvals": at every piece's start only the eigenvalues of h_m there (scipy's banded Hermitian solver on the host matrix, no vectors),
  the Weyl gap of the occupation carried by sorted index: where every piece passes, Weyl's inequality keeps the occupied cluster separated from
  the empty one along the whole path, so the continued occupation is the same set of sorted indices and the continued reference is the
  spectral projector onto it at the end (the trial fields' eigenvectors, which the engine computes anyway); the endpoint's columns are built
  from that occupation by version 3's own post-processing (degenerate blocks rotated by their weight on the previous reference, pockets,
  inversions, gaps); at a piece that still fails, the full route takes over from that piece's start (the eigenvectors there, the occupation
  continued so far), marked;
- the oracle (oracle_v34): at given fields (the record's accepted applied fields) the full route through the same pieces and version 3's
  one-step assignment, against the run's reference: the masks' equality and the subspace residuals ||(1 - V V^dag) W||_F.
Every quantity it returns is printed by engine_collapse_v34.py into the record."""
import math
import time

import numpy as np
import scipy.linalg as sla

import engine_ops_v3 as OPS3


def band_lower(Hcsr):
    """the lower band storage of a Hermitian host CSR matrix for scipy.linalg.eigvals_banded (lower=True): ab[d, j] = H[j + d, j]."""
    C = Hcsr.tocoo()
    off = C.row - C.col
    b = int(max(0, off.max())) if off.size else 0
    n = Hcsr.shape[0]
    ab = np.zeros((b + 1, n), dtype=complex)
    sel = off >= 0
    ab[off[sel], C.col[sel]] += C.data[sel]
    return ab


def eigvals_host(Hcsr):
    return np.sort(sla.eigvals_banded(band_lower(Hcsr), lower=True))


def weyl_gap_of(e, occ, tol_block=1e-8):
    """version 3's Weyl gap of the occupation occ (bool over the levels in energy order): blocks by engine_ops_v3.blocks_of_v3; the smallest
    distance between adjacent blocks one of which holds occupied and the other empty levels; 0 when a block holds both."""
    e = np.asarray(e, float)
    blocks = OPS3.blocks_of_v3(e, tol_block)
    blk_id = np.empty(len(e), dtype=np.int64)
    for b, (i, j) in enumerate(blocks):
        blk_id[i:j] = b
    has_o = np.zeros(len(blocks), bool); has_e = np.zeros(len(blocks), bool)
    has_o[np.unique(blk_id[occ])] = True; has_e[np.unique(blk_id[~occ])] = True
    gb = None
    for b in range(len(blocks) - 1):
        if (has_o[b] and has_e[b + 1]) or (has_e[b] and has_o[b + 1]):
            d_ = float(e[blocks[b + 1][0]] - e[blocks[b][1] - 1])
            gb = d_ if gb is None else min(gb, d_)
    partial = int(sum(1 for b in range(len(blocks)) if has_o[b] and has_e[b]))
    return (0.0 if partial else gb), gb, partial


def assign_occupation_v34(ref_prev, e, U, n_sea, occ_given, tol_block=1e-8):
    """version 3's assignment (engine_ops_v3.assign_max_overlap_v3) with the occupation given instead of chosen by weight: the same weights,
    blocks, rotations, columns and diagnostics. Returns (ref, info, mask)."""
    import torch
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    A = ref_prev.conj().T @ U
    w, blocks, rot, blk_id, n_inter = OPS3._assign_v3(torch, A, e_np, n_sea, tol_block)
    occ = np.asarray(occ_given, bool).copy()
    bsize = np.array([j - i for (i, j) in blocks])
    plain = np.where(occ & (bsize[blk_id] == 1))[0]
    cols = []
    if len(plain):
        cols.append(U[:, torch.as_tensor(plain.astype(np.int64), device=U.device)])
    for (i, j), V in rot.items():
        sel = np.where(occ[i:j])[0]
        if len(sel):
            # a rotated (degenerate) block: as many of its rotated vectors as the occupation gives it, the ones of largest weight on the previous
            # reference (version 3's order inside a block); the mask marks those rotated vectors
            wl = w[i:j]
            pick = np.sort(np.argsort(-wl, kind="stable")[: len(sel)])
            occ[i:j] = False; occ[i + pick] = True
            cols.append(U[:, i:j] @ V[:, torch.as_tensor(pick, dtype=torch.int64, device=U.device)])
    ref = torch.cat(cols, dim=1) if cols else U[:, :0]
    wo = w[occ]; we = w[~occ]
    signless = np.zeros(len(e_np), bool); straddle = 0
    for (i, j) in rot:
        if e_np[i] < 0.0 < e_np[j - 1]:
            signless[i:j] = True; straddle += 1
    pockets_occ_pos = int(np.sum(occ & (e_np > 0.0) & ~signless)); holes_emp_neg = int(np.sum(~occ & (e_np < 0.0) & ~signless))
    inv = 0
    if (~occ).any() and occ.any():
        k_low = int(np.argmin(np.where(~occ, e_np, np.inf)))
        inv = int(np.sum(occ & (blk_id > blk_id[k_low])))
    wg, gb, partial = weyl_gap_of(e_np, occ, tol_block)
    info = {"n_occ": int(occ.sum()), "n_sea": n_sea, "n_intermediate": n_inter, "min_occupied_weight": float(wo.min()) if len(wo) else None,
            "max_empty_weight": float(we.max()) if len(we) else None, "inversions": inv, "n_pocket": pockets_occ_pos + holes_emp_neg,
            "pockets_occupied_positive": pockets_occ_pos, "holes_empty_negative": holes_emp_neg, "nneg": int((e_np < 0).sum()),
            "gap_between_blocks": gb, "weyl_gap": wg, "partly_occupied_blocks": partial, "rotated_blocks": len(rot),
            "straddling_blocks": straddle, "sign_free_vectors": int(signless.sum()), "tol_block": tol_block, "ambiguous_tie": False,
            "tie_by_predecessor": False, "empty_weight_sum": float(n_sea - wo.sum()) if len(wo) else None}
    return ref, info, occ


def _prev_mask(prev, n_levels, n_sea):
    m = prev.get("mask")
    if m is not None:
        return np.asarray(m, bool)
    occ = np.zeros(n_levels, bool); occ[:n_sea] = True          # the T = 0 window holds the n_sea lowest levels (no mask stored there)
    return occ


def pieces_eigvals(ch, prev, f_cur, nG, n_lead, refine_max, h_of, occ0, stats):
    """the eigvals route's walk: returns (pieces [(s0, s1, gap at s0, 2 dh, passed)], all passed, the first failing piece index or None)."""
    dN = f_cur[0] - prev["f"][0]; dv = f_cur[1] - prev["f"][1]
    g_start = prev.get("weyl_gap", prev.get("gap"))
    out = []
    queue = [(k / n_lead, (k + 1) / n_lead, 0) for k in range(n_lead)]
    gap_cache = {0.0: g_start}
    ok_all = True; first_fail = None
    while queue:
        s0, s1, depth = queue.pop(0)
        if s0 in gap_cache:
            g0 = gap_cache[s0]
        else:
            t_ = time.time()
            ev = eigvals_host(h_of(ch, prev["f"][0] + s0 * dN, prev["f"][1] + s0 * dv))
            stats["eigvals_n"] += 1; stats["eigvals_s"] += time.time() - t_
            g0, _, _ = weyl_gap_of(ev, occ0)
            gap_cache[s0] = g0
        thr = 2.0 * nG * (s1 - s0)
        if g0 is not None and thr < g0:
            out.append((s0, s1, g0, thr, True))
        elif depth < refine_max:
            sm = 0.5 * (s0 + s1)
            queue[0:0] = [(s0, sm, depth + 1), (sm, s1, depth + 1)]
            stats["refinements"] += 1
        else:
            out.append((s0, s1, g0, thr, False)); ok_all = False
            if first_fail is None:
                first_fail = len(out) - 1
    return out, ok_all, first_fail


def continue_full_pieces(ch, prev, f_cur, e_cur, U_cur, n_sea, pieces, eig_at, ref0, g0, pm0, stats):
    """the full route over given pieces from (ref0 at pieces[0]'s start, its gap g0, its mask pm0): at each piece's end the eigendecomposition
    and version 3's maximum-overlap assignment; the test at each piece's start with the gap carried by the assignments. Returns (ref, info, mask,
    tests [(s0, s1, gap, 2dh, passed)])."""
    dN = f_cur[0] - prev["f"][0]; dv = f_cur[1] - prev["f"][1]
    ref = ref0; g = g0; pm = pm0; tests = []; info = None; mask = pm0
    for (s0, s1, _g, thr, _p) in pieces:
        tests.append((s0, s1, g, thr, bool(g is not None and thr < g)))
        if abs(s1 - 1.0) < 1e-15:
            e_s, U_s = e_cur, U_cur
        else:
            t_ = time.time()
            e_s, U_s = eig_at(ch, prev["f"][0] + s1 * dN, prev["f"][1] + s1 * dv)
            stats["eigh_n"] += 1; stats["eigh_s"] += time.time() - t_
        ref, info, mask = OPS3.assign_max_overlap_v3(ref, e_s, U_s, n_sea, prev_mask=pm)
        g = info["weyl_gap"]; pm = mask
    return ref, info, mask, tests


def continue_sea_v34(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of, route, refine_max=10):
    """the certified continuation of the docstring. Returns (ref, info, mask) with version 3's info keys and info["certified"]."""
    t0 = time.time()
    prev = window[-1]
    dH = h_of(ch, f_cur[0] - prev["f"][0], f_cur[1] - prev["f"][1])
    nG = OPS3.gershgorin(dH)
    g_prev = prev.get("weyl_gap", prev.get("gap"))
    s_T = nG / tau
    tau_LZ = (1.0 / (c_LZ * math.sqrt(s_T))) if s_T > 0.0 else math.inf
    extra = {"norm_dH_G": nG, "s_T": s_T, "tau_LZ": tau_LZ, "c_LZ": c_LZ, "m": 1, "m_eff": 1, "window_consistency": None}
    stats = {"eigvals_n": 0, "eigvals_s": 0.0, "eigh_n": 0, "eigh_s": 0.0, "refinements": 0}
    n_levels = int(e_cur.shape[0])
    occ0 = _prev_mask(prev, n_levels, n_sea)
    if g_prev is None or not (g_prev > 0.0) or nG == 0.0:
        ref, info, mask = OPS3.continue_sea_v3(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of)
        info["certified"] = {"route": route, "status": ("no Weyl gap at the start (%s): version 3's convention, marked" % g_prev) if nG > 0.0 else "no field change",
                             "verified": bool(nG == 0.0), "seconds": time.time() - t0}
        return ref, info, mask
    n_lead = max(1, int(math.ceil(2.0 * nG / g_prev)))
    pieces, ok_all, first_fail = pieces_eigvals(ch, prev, f_cur, nG, n_lead, refine_max, h_of, occ0, stats)
    status = "every piece passes" if ok_all else ("still fails from piece %d of %d" % (first_fail + 1, len(pieces)))
    if route == "full":
        ref, info, mask, tests = continue_full_pieces(ch, prev, f_cur, e_cur, U_cur, n_sea, pieces, eig_at, prev["ref"], g_prev, prev.get("mask"), stats)
        ok_run = all(q[4] for q in tests)
    else:
        if ok_all:
            ref, info, mask = assign_occupation_v34(prev["ref"], e_cur, U_cur, n_sea, occ0)
            tests = pieces; ok_run = True
        else:
            # the full route from the failing piece's start, the occupation continued by sorted index up to there (all earlier pieces passed)
            s_f = pieces[first_fail][0]
            dN = f_cur[0] - prev["f"][0]; dv = f_cur[1] - prev["f"][1]
            if s_f > 0.0:
                t_ = time.time()
                e_f, U_f = eig_at(ch, prev["f"][0] + s_f * dN, prev["f"][1] + s_f * dv)
                stats["eigh_n"] += 1; stats["eigh_s"] += time.time() - t_
                ref_f, info_f, mask_f = assign_occupation_v34(prev["ref"], e_f, U_f, n_sea, occ0)
                g_f = info_f["weyl_gap"]
            else:
                ref_f, g_f, mask_f = prev["ref"], g_prev, prev.get("mask")
            ref, info, mask, tests_b = continue_full_pieces(ch, prev, f_cur, e_cur, U_cur, n_sea, pieces[first_fail:], eig_at, ref_f, g_f, mask_f, stats)
            tests = pieces[:first_fail] + tests_b
            ok_run = False
    margin = min((q[2] - q[3]) for q in tests if q[2] is not None) if any(q[2] is not None for q in tests) else None
    info.update(extra)
    info.update({"mode": "Weyl-certified sub-steps (route %s; the lead's count %d; pieces %d)" % (route, n_lead, len(tests)), "nsub": len(tests), "verified": bool(ok_run),
                 "isolation_verified": bool(ok_run), "margin_min": margin})
    info["certified"] = {"route": route, "n_lead": n_lead, "gap_start": g_prev, "norm_dH_G": nG, "pieces": len(tests), "refinements": stats["refinements"],
                         "failed_pieces": int(sum(1 for q in tests if not q[4])), "status": status, "verified": bool(ok_run), "margin_min": margin,
                         "smallest_piece": min(q[1] - q[0] for q in tests), "eigvals_calls": stats["eigvals_n"], "eigvals_seconds": stats["eigvals_s"],
                         "eigh_calls": stats["eigh_n"], "eigh_seconds": stats["eigh_s"], "seconds": time.time() - t0,
                         "piece_list": [[float(q[0]), float(q[1])] for q in tests] if len(tests) <= 4096 else None}
    return ref, info, mask


def oracle_v34(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of, ref_run, mask_run, refine_max=10):
    """at the given fields: the full route through the eigvals route's pieces, and version 3's one-step continuation, against the run's
    reference and mask. Returns the comparison (no arrays)."""
    t0 = time.time()
    prev = window[-1]
    dH = h_of(ch, f_cur[0] - prev["f"][0], f_cur[1] - prev["f"][1])
    nG = OPS3.gershgorin(dH)
    g_prev = prev.get("weyl_gap", prev.get("gap"))
    stats = {"eigvals_n": 0, "eigvals_s": 0.0, "eigh_n": 0, "eigh_s": 0.0, "refinements": 0}
    occ0 = _prev_mask(prev, int(e_cur.shape[0]), n_sea)
    out = {"norm_dH_G": nG, "gap_start": g_prev}

    def resid(V, W):
        return float((W - V @ (V.conj().T @ W)).norm().item()) if V.shape[1] and W.shape[1] else None

    if g_prev is not None and g_prev > 0.0 and nG > 0.0:
        n_lead = max(1, int(math.ceil(2.0 * nG / g_prev)))
        pieces, ok_all, first_fail = pieces_eigvals(ch, prev, f_cur, nG, n_lead, refine_max, h_of, occ0, stats)
        ref_f, info_f, mask_f, tests = continue_full_pieces(ch, prev, f_cur, e_cur, U_cur, n_sea, pieces, eig_at, prev["ref"], g_prev, prev.get("mask"), stats)
        out["full"] = {"n_lead": n_lead, "pieces": len(pieces), "eigvals_pieces_pass": bool(ok_all), "full_tests_pass": bool(all(q[4] for q in tests)),
                       "tests_agree": bool(all(p[4] == q[4] for p, q in zip(pieces, tests))),
                       "largest_gap_difference": float(max(abs(p[2] - q[2]) for p, q in zip(pieces, tests) if p[2] is not None and q[2] is not None)) if pieces else None,
                       "mask_equals_run": bool(np.array_equal(np.asarray(mask_f, bool), np.asarray(mask_run, bool))),
                       "residual_run_on_full": resid(ref_f, ref_run), "residual_full_on_run": resid(ref_run, ref_f),
                       "min_occupied_weight": info_f.get("min_occupied_weight"), "max_empty_weight": info_f.get("max_empty_weight"), "n_intermediate": info_f.get("n_intermediate"),
                       "eigh_calls": stats["eigh_n"], "eigh_seconds": stats["eigh_s"], "eigvals_calls": stats["eigvals_n"], "eigvals_seconds": stats["eigvals_s"]}
    ref_l, info_l, mask_l = OPS3.continue_sea_v3(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of)
    out["v3_one_step"] = {"mode": info_l.get("mode"), "verified": info_l.get("verified"), "mask_equals_run": bool(np.array_equal(np.asarray(mask_l, bool), np.asarray(mask_run, bool))),
                          "residual_run_on_v3": resid(ref_l, ref_run), "inversions": info_l.get("inversions"), "n_pocket": info_l.get("n_pocket")}
    out["seconds"] = time.time() - t0
    return out
