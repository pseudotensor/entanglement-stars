#!/usr/bin/env python
"""engine_ops_v4.py (builder analyst; engine spec 28.5, S4 (d) of REVIEWA_REVIEW_web40e: "for isolated real zero pairs use oriented-plane
transport (P = (Pi + iJ)/2 with J transported by the plane's real orthogonal transport) instead of maximum overlap"; the brief
BRIEF_v26_builder_S4_S9: "engine_ops_v4.py = engine_ops_v3 with oriented-plane transport for isolated real zero pairs at m = 0 in place of
the maximum overlap; ops v3's rule everywhere else; per record the pairs so transported and the plane's transport residual printed").

The structure used (re-derived here, P07 sec. 2.1 as the idea): at m = 0 the clocked channel matrix is purely imaginary Hermitian in the
engine's radial basis (h^* = -h, engine_realskew_unit_v2), h = iA with A real skew; if u has the eigenvalue eps, u^* has -eps, so the sorted
spectrum is +- paired (e_j = -e_(n-1-j)) and each pair spans a real two-plane Q (the span of Re u, Im u). The only lines in Q with P + P^* = Q
(neutral: one member of the pair occupied) are u and u^*. Transport: with B_0, B real orthonormal bases of the previous and the current
plane, O = polar(B^T B_0) (real orthogonal 2 x 2; the closed form of P07's W = polar{Q Q_0 + (1 - Q)(1 - Q_0)} on the plane), the previous
occupied line phi_0 goes to phi = B O B_0^T phi_0, which is neutral, hence u or u^* of the current pair: the orientation J = -i(2P - Q) is
carried continuously, through eps = 0 included (no 1/eps anywhere). On an isolated pair the overlap rule picks the same line whenever the
weights are resolvable (|u^dag phi_0|^2 - |u^T phi_0|^2 = det(B^T B_0)); the rules differ where the pair is numerically degenerate (eps
below the block tolerance: ops v3 rotates the block to its weight eigenbasis, a line that need not be neutral) or where the plane turned by
more than 45 degrees in one step.

Rule [the builder's construction of the review's sentence; the thresholds are listed open]: per (sub-)step of engine_ops_v3's continuation,
with eta the step's Gershgorin bound of the field increment (per sub-step) and the spectrum +- paired to --pair-tol (1e-12 of max |e|):
a pair p (levels +-eps_p) is a zero pair when its own gap fails the ordinary certificate, 2 eps_p <= 2 eta; it is isolated when its levels
are separated from every other level by sep_p > 2 eta (Weyl: along the linear path no other level reaches the plane, so the plane is a
continuous spectral plane over the sub-step; isolation "weyl", the default) or, as the labelled uncertified variant "angle", when its
matched previous plane has every principal cosine at least --angle-cos (0.9); it is transported when a stored neutral occupied line of the
previous record (one of the tracked pairs' lines, --n-track 16 smallest pairs) matches its plane (mean squared principal cosine above 1/2,
the smallest principal cosine above 0.1) and that line is neutral (|phi_0^T phi_0| <= 1e-8). Every other level is assigned by
engine_ops_v3's rule (assign_max_overlap_v3's code, the same blocks, ties and inversions) over the remaining eigenvectors with n_sea less the
transported lines; with no pair transported the step is engine_ops_v3's bitwise (the same function call on the same arrays). The transported
line is the reference's column itself (projected on the computed pair's complex span and normalized); the mask marks the member with the
larger weight, and a line whose two weights are both below 3/4 (eps below the eigensolver's resolution) is "sign free" (no pocket count).
Per transported pair the record keeps eps, sep, eta, the smallest principal cosine, det O (+1 rotation, -1 reflection), the weights on
u_+ and u_-, the signed energy Tr(h P) = eps (w_+ - w_-), the neutrality defect |phi^T phi|, the projection residual (the transported line's
part outside the computed span) and the commutator residual 2 eps sqrt(w_+ w_-); the plane's transport residual is the largest of the last
three. The protected gap (the certificate's g_n of S4 (e)): the smallest |e_o - e_u| over occupied o and empty u, the transported pairs' own
two levels excluded, zero where a degenerate block (ops v3's width rule) is partly occupied; it is info["weyl_gap"] for the next step's
test, so "verified" is the certificate 2 eta_n < g_n at every sub-step. The window's last entry carries "lines" (the tracked pairs'
occupied lines: per pair the occupied eigenvector, or the transported line; none where both or neither member is occupied or the member
sits in a rotated block)."""
import math

import numpy as np

import engine_ops_v3 as OPS3
from engine_ops_v2 import gershgorin  # noqa: F401  (the same Gershgorin bound)

OPTS = {"oriented": True, "isolation": "weyl", "angle_cos": 0.9, "pair_tol": 1e-12, "n_track": 16, "neutral_tol": 1e-8,
        "match_min": 0.5, "polar_min": 0.1, "snap": 0.75}


def pairing_defect(e_np):
    """max |e_j + e_(n-1-j)| over max |e| for the sorted spectrum (zero for a +- paired h)."""
    e_np = np.asarray(e_np, float)
    sc = max(float(np.max(np.abs(e_np))), 1e-300)
    return float(np.max(np.abs(e_np + e_np[::-1]))) / sc


def pair_index(e_np, n_track):
    """the n_track pairs nearest zero of a +- paired sorted spectrum: arrays (jm, jp, eps) with jm the negative member's index, jp = n - 1 - jm,
    eps = (e[jp] - e[jm])/2, ordered by eps (p = 0 nearest zero)."""
    n = len(e_np)
    h = n // 2
    k = min(int(n_track), h)
    jm = np.arange(h - 1, h - 1 - k, -1)
    jp = n - 1 - jm
    eps = 0.5 * (np.asarray(e_np, float)[jp] - np.asarray(e_np, float)[jm])
    return jm, jp, eps


def pair_separations(e_np, n_track):
    """per tracked pair the separation of its levels {+-eps_p} from every other level: min over q != p of |eps_p - eps_q| (all eps >= 0),
    with eps over all pairs of the spectrum (the neighbours in eps order)."""
    n = len(e_np)
    h = n // 2
    e_np = np.asarray(e_np, float)
    jm_all = np.arange(h - 1, -1, -1)
    eps_all = 0.5 * (e_np[n - 1 - jm_all] - e_np[jm_all])
    k = min(int(n_track), h)
    sep = np.empty(k)
    for p in range(k):
        d = np.abs(eps_all - eps_all[p])
        d[p] = np.inf
        sep[p] = float(d.min()) if len(d) > 1 else math.inf
    return sep


def real_plane(torch, Z):
    """the real orthonormal basis B (n x 2, real) of the real span of the complex columns Z (n x 2): the top two left singular vectors of
    [Re Z, Im Z]; returns (B, s) with s the four singular values (s_3, s_4 zero for a conjugation-invariant span)."""
    R4 = torch.cat([Z.real, Z.imag], dim=1)
    Ub, s, _ = torch.linalg.svd(R4, full_matrices=False)
    return Ub[:, :2].contiguous(), s.cpu().numpy().astype(float)


def line_plane(torch, phi):
    """the real orthonormal basis of the plane of a neutral line phi (the span of Re phi, Im phi; orthogonal and of equal norm for a
    neutral line, phi^T phi = 0)."""
    R2 = torch.stack([phi.real, phi.imag], dim=1)
    Ub, s, _ = torch.linalg.svd(R2, full_matrices=False)
    return Ub[:, :2].contiguous()


def polar2(M):
    """the polar factor of a real 2 x 2 matrix (U V^T of its SVD), its singular values and det."""
    u, s, vt = np.linalg.svd(M)
    O = u @ vt
    return O, s, float(np.linalg.det(O))


def _assign_v3_full(ref_prev, e, U, n_sea, tol_block=1e-8, tol_fallback=1e-6, prev_mask=None):
    """engine_ops_v3.assign_max_overlap_v3 verbatim (the same operations in the same order), returning also its blocks, the rotated blocks'
    rotations and the block ids, for the lines' bookkeeping. Returns (ref, info, occ, blocks, rot, blk_id)."""
    import torch
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    A = ref_prev.conj().T @ U
    w, blocks, rot, blk_id, n_inter = OPS3._assign_v3(torch, A, e_np, n_sea, tol_block)
    tol_used = tol_block
    if n_inter > 0 and tol_fallback and tol_fallback > tol_block:
        w2, blocks2, rot2, blk2, n2 = OPS3._assign_v3(torch, A, e_np, n_sea, tol_fallback)
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
    signless = np.zeros(len(e_np), bool); straddle = 0
    for (i, j) in rot:
        if e_np[i] < 0.0 < e_np[j - 1]:
            signless[i:j] = True; straddle += 1
    pockets_occ_pos = int(np.sum(occ & (e_np > 0.0) & ~signless)); holes_emp_neg = int(np.sum(~occ & (e_np < 0.0) & ~signless))
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
    return ref, info, occ, blocks, rot, blk_id


def protected_gap(e_np, occ, excluded_pairs, blocks=None):
    """the smallest |e_o - e_u| over occupied o and empty u, each transported pair's own two levels excluded (excluded_pairs: (jm, jp)
    tuples); zero where a block of `blocks` (ops v3's degenerate blocks, (i, j) ranges) is partly occupied."""
    e_np = np.asarray(e_np, float); occ = np.asarray(occ, bool)
    if blocks is not None:
        for (i, j) in blocks:
            if j - i > 1 and occ[i:j].any() and (~occ[i:j]).any():
                return 0.0
    io = np.where(occ)[0]; iu = np.where(~occ)[0]
    if not len(io) or not len(iu):
        return None
    partner = np.full(len(e_np), -1, dtype=np.int64)
    for (a, b) in excluded_pairs:
        partner[int(a)] = int(b); partner[int(b)] = int(a)
    eu = e_np[iu]
    order = np.argsort(eu, kind="stable"); eus = eu[order]; ius = iu[order]
    x = e_np[io]
    pos = np.searchsorted(eus, x)
    best = math.inf
    for off in (-3, -2, -1, 0, 1, 2):
        q = pos + off
        ok = (q >= 0) & (q < len(eus))
        if not ok.any():
            continue
        qq = np.clip(q, 0, len(eus) - 1)
        d = np.abs(eus[qq] - x)
        d = np.where(ok & (ius[qq] != partner[io]), d, np.inf)
        best = min(best, float(d.min()))
    return best if math.isfinite(best) else None


def select_transport(torch, e_np, U, eta, lines_prev, opts):
    """the zero pairs transported at this (sub-)step (docstring's rule) and their lines. Returns (list of dicts, counts)."""
    jm, jp, eps = pair_index(e_np, opts["n_track"])
    sep = pair_separations(e_np, opts["n_track"])
    cnt = {"zero_pairs": 0, "not_isolated": 0, "unmatched": 0, "transported": 0}
    cands = [p for p in range(len(eps)) if eps[p] <= eta]
    cnt["zero_pairs"] = len(cands)
    if not cands or not lines_prev:
        cnt["unmatched"] = len(cands) if cands else 0
        return [], cnt
    prev = []
    for L in lines_prev:
        if L.get("phi") is None or (L.get("neutral_defect") or 0.0) > opts["neutral_tol"]:
            continue
        prev.append((L, line_plane(torch, L["phi"])))
    used = set()
    out = []
    for p in cands:
        if opts["isolation"] == "weyl" and not (sep[p] > 2.0 * eta):
            cnt["not_isolated"] += 1
            continue
        idx = torch.as_tensor([int(jm[p]), int(jp[p])], dtype=torch.int64, device=U.device)
        Z = U[:, idx]
        B, s4 = real_plane(torch, Z)
        best = None
        for i, (L, B0) in enumerate(prev):
            if i in used:
                continue
            M = (B.T @ B0).cpu().numpy().astype(float)
            sv = np.linalg.svd(M, compute_uv=False)
            o = 0.5 * float((sv ** 2).sum())
            if best is None or o > best[0]:
                best = (o, i, M, sv)
        if best is None or best[0] <= opts["match_min"] or float(best[3].min()) <= opts["polar_min"]:
            cnt["unmatched"] += 1
            continue
        if opts["isolation"] == "angle" and float(best[3].min()) < opts["angle_cos"]:
            cnt["not_isolated"] += 1
            continue
        o, i, M, sv = best
        L, B0 = prev[i]
        O, _sv, detO = polar2(M)
        phi0 = L["phi"]
        c0 = B0.to(phi0.dtype).T @ phi0
        phi = B.to(phi0.dtype) @ (torch.as_tensor(O, dtype=phi0.dtype, device=phi0.device) @ c0)
        coef = Z.conj().T @ phi
        phi_p = Z @ coef
        proj_res = float(torch.linalg.norm(phi - phi_p))
        nrm = float(torch.linalg.norm(phi_p))
        phi_p = phi_p / nrm
        coef = Z.conj().T @ phi_p
        w_m = float(abs(complex(coef[0])) ** 2); w_p = float(abs(complex(coef[1])) ** 2)
        neutral = float(abs(complex((phi_p * phi_p).sum())))
        e_p = float(eps[p])
        comm = 2.0 * e_p * math.sqrt(max(w_p * w_m, 0.0))
        used.add(i)
        cnt["transported"] += 1
        out.append({"p": int(p), "jm": int(jm[p]), "jp": int(jp[p]), "eps": e_p, "sep": float(sep[p]), "eta": float(eta), "phi": phi_p,
                    "cos_min": float(sv.min()), "cos_max": float(sv.max()), "det_O": detO, "w_plus": w_p, "w_minus": w_m,
                    "signed_energy": e_p * (w_p - w_m), "neutrality_defect": neutral, "projection_residual": proj_res,
                    "commutator_residual": comm, "plane_reality_defect": float(s4[2] / max(s4[0], 1e-300)),
                    "transport_residual": max(neutral, proj_res, comm), "sign_free": bool(max(w_p, w_m) < opts["snap"]),
                    "occupied": ("sign free" if max(w_p, w_m) < opts["snap"] else ("+" if w_p >= w_m else "-")),
                    "prev_eps": L.get("eps")})
    return out, cnt


def lines_from(torch, e_np, U, occ, blk_size_full, trans, n_track):
    """the tracked pairs' occupied lines after an assignment: the transported line, else the occupied member's eigenvector when exactly one
    member is occupied and neither sits in a block of more than one level; with |phi^T phi| (the neutrality defect)."""
    jm, jp, eps = pair_index(e_np, n_track)
    tmap = {t["p"]: t for t in trans}
    out = []
    for p in range(len(eps)):
        t = tmap.get(p)
        if t is not None:
            phi = t["phi"]
        else:
            a, b = int(jm[p]), int(jp[p])
            if occ[a] == occ[b] or blk_size_full[a] > 1 or blk_size_full[b] > 1:
                out.append({"p": p, "eps": float(eps[p]), "phi": None, "neutral_defect": None})
                continue
            phi = U[:, a if occ[a] else b]
        out.append({"p": p, "eps": float(eps[p]), "phi": phi.clone(), "neutral_defect": float(abs(complex((phi * phi).sum())))})
    return out


def assign_v4(ref_prev, e, U, n_sea, eta, lines_prev, prev_mask=None, opts=None):
    """one (sub-)step's assignment (docstring). Returns (ref, info, mask, lines)."""
    import torch
    opts = OPTS if opts is None else opts
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    n = len(e_np)
    pdef = pairing_defect(e_np)
    paired = bool(pdef <= opts["pair_tol"])
    trans, cnt = [], {"zero_pairs": 0, "not_isolated": 0, "unmatched": 0, "transported": 0}
    if opts["oriented"] and paired and lines_prev:
        trans, cnt = select_transport(torch, e_np, U, eta, lines_prev, opts)
    if not trans:
        ref, info, occ, blocks, rot, blk_id = _assign_v3_full(ref_prev, e, U, n_sea, prev_mask=prev_mask)
        bsize = np.array([j - i for (i, j) in blocks])
        blk_size_full = bsize[blk_id]
        pg = protected_gap(e_np, occ, [], [(i, j) for (i, j) in blocks])
        info["protected_gap"] = pg
    else:
        excl = sorted({t["jm"] for t in trans} | {t["jp"] for t in trans})
        keep = np.setdiff1d(np.arange(n), np.asarray(excl, dtype=np.int64))
        keep_t = torch.as_tensor(keep, dtype=torch.int64, device=U.device)
        pm = np.asarray(prev_mask, bool)[keep] if (prev_mask is not None and len(prev_mask) == n) else None
        e_k = e[keep_t] if hasattr(e, "cpu") else e_np[keep]
        ref_r, info, occ_r, blocks_r, rot_r, blk_r = _assign_v3_full(ref_prev, e_k, U[:, keep_t], n_sea - len(trans), prev_mask=pm)
        phis = torch.stack([t["phi"] for t in trans], dim=1)
        ref = torch.cat([ref_r, phis], dim=1)
        occ = np.zeros(n, bool); occ[keep] = occ_r
        sign_free = np.zeros(n, bool)
        for t in trans:
            j_occ = t["jm"] if (t["sign_free"] or t["w_minus"] > t["w_plus"]) else t["jp"]
            occ[j_occ] = True
            if t["sign_free"]:
                sign_free[t["jm"]] = sign_free[t["jp"]] = True
        blk_size_full = np.ones(n, dtype=np.int64)
        blocks_full = []
        e_kept = e_np[keep]
        for (i, j) in blocks_r:
            idx = keep[i:j]
            blk_size_full[idx] = j - i
            blocks_full.append(idx)
            if j - i > 1 and e_kept[i] < 0.0 < e_kept[j - 1]:
                sign_free[idx] = True
        bstart = np.empty(n)
        for idx in blocks_full:
            bstart[idx] = e_np[idx].min()
        for t in trans:
            bstart[t["jm"]] = e_np[t["jm"]]; bstart[t["jp"]] = e_np[t["jp"]]
        pairs_ex = [(t["jm"], t["jp"]) for t in trans]
        partial = int(sum(1 for idx in blocks_full if len(idx) > 1 and occ[idx].any() and (~occ[idx]).any()))
        pg = protected_gap(e_np, occ, pairs_ex)
        inv = 0
        if (~occ).any() and occ.any():
            k_low = int(np.argmin(np.where(~occ, e_np, np.inf)))
            inv = int(np.sum(occ & (bstart > bstart[k_low])))
        po = int(np.sum(occ & (e_np > 0.0) & ~sign_free)); he = int(np.sum(~occ & (e_np < 0.0) & ~sign_free))
        info.update({"n_occ": int(occ.sum()), "n_sea": n_sea, "inversions": inv, "n_pocket": po + he, "pockets_occupied_positive": po,
                     "holes_empty_negative": he, "nneg": int((e_np < 0).sum()), "gap_between_blocks": pg, "weyl_gap": (0.0 if partial else pg),
                     "partly_occupied_blocks": partial, "sign_free_vectors": int(sign_free.sum()), "protected_gap": (0.0 if partial else pg),
                     "rest_assignment": "engine_ops_v3's rule over the %d remaining eigenvectors (n_sea %d)" % (len(keep), n_sea - len(trans))})
    lines = lines_from(torch, e_np, U, occ, blk_size_full, trans, opts["n_track"])
    info["oriented"] = {"pairing_defect": pdef, "paired": paired, "isolation": opts["isolation"], "counts": cnt,
                        "pairs": [{k_: v_ for k_, v_ in t.items() if k_ != "phi"} for t in trans]}
    return ref, info, occ, lines


def continue_sea_v4(ch, window, f_cur, e_cur, U_cur, n_sea, tau, c_LZ, eig_at, h_of, window_max=1, opts=None):
    """engine_ops_v3.continue_sea_v3 with assign_v4 at every (sub-)step (the oriented pairs, the protected gap, the lines carried through
    the sub-steps from the window's last entry). Returns (ref, info, mask); info["lines"] the lines for the next window entry,
    info["certificate"] S4 (e)'s per-step log: eta_n (the Gershgorin bound per sub-step: an interval bound for the linear path), the
    smallest protected gap at a sub-step's start g_n, passes = 2 eta_n < g_n at every sub-step."""
    opts = OPTS if opts is None else opts
    prev = window[-1]
    dH = h_of(ch, f_cur[0] - prev["f"][0], f_cur[1] - prev["f"][1])
    nG = gershgorin(dH)
    s_T = nG / tau
    tau_LZ = (1.0 / (c_LZ * math.sqrt(s_T))) if s_T > 0.0 else math.inf
    info_extra = {"norm_dH_G": nG, "s_T": s_T, "tau_LZ": tau_LZ, "c_LZ": c_LZ, "m": 1, "m_eff": 1, "window_consistency": None}
    g_prev = prev.get("weyl_gap", prev.get("gap"))
    g_iso = prev.get("gap")
    lines = prev.get("lines")
    plog = []
    if tau_LZ <= tau:
        nsub = max(1, int(round(c_LZ * math.sqrt(tau * nG))))
        ref = prev["ref"]; g0 = g_prev; gi = g_iso; verified = True; iso = True; margin = math.inf
        dh = nG / nsub
        pm = prev.get("mask")
        gmin = None
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
                gmin = g0 if gmin is None else min(gmin, g0)
            ref, info, mask, lines = assign_v4(ref, e_s, U_s, n_sea, dh, lines, prev_mask=pm, opts=opts)
            g0 = info["weyl_gap"]; gi = info["gap_between_blocks"]; pm = mask
            plog += [dict(q, substep=k) for q in info["oriented"]["pairs"]]
        info.update(info_extra)
        info.update({"mode": "LZ sub-steps (nsub %d)" % nsub, "nsub": nsub, "verified": verified, "isolation_verified": iso,
                     "margin_min": (margin if math.isfinite(margin) else None)})
        info["oriented"]["pairs_all_substeps"] = plog
        info["lines"] = lines
        info["certificate"] = {"eta_n": dh, "g_n": gmin, "passes": bool(verified), "substeps": nsub}
        return ref, info, mask
    verified = bool(g_prev is not None and 2.0 * nG < g_prev)
    iso = bool(g_iso is not None and 2.0 * nG < g_iso)
    ref, info, mask, lines = assign_v4(prev["ref"], e_cur, U_cur, n_sea, nG, lines, prev_mask=prev.get("mask"), opts=opts)
    info.update(info_extra)
    info.update({"mode": "one step", "nsub": 1, "verified": verified, "isolation_verified": iso,
                 "margin_min": ((g_prev - 2.0 * nG) if g_prev is not None else None)})
    info["oriented"]["pairs_all_substeps"] = [dict(q, substep=1) for q in info["oriented"]["pairs"]]
    info["lines"] = lines
    info["certificate"] = {"eta_n": nG, "g_n": g_prev, "passes": bool(verified), "substeps": 1}
    return ref, info, mask


def initial_lines(e, U, n_track=None):
    """the tracked lines of a fresh negative-energy sea (the window's first entry): per pair nearest zero its negative member."""
    import torch
    n_track = OPTS["n_track"] if n_track is None else n_track
    e_np = e.cpu().numpy().astype(float) if hasattr(e, "cpu") else np.asarray(e, float)
    occ = e_np < 0
    if pairing_defect(e_np) > OPTS["pair_tol"]:
        return None
    return lines_from(torch, e_np, U, occ, np.ones(len(e_np), dtype=np.int64), [], n_track)
