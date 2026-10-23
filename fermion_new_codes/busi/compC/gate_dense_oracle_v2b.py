#!/usr/bin/env python
"""gate_dense_oracle_v2b.py (busi lane, Computation C): gate_dense_oracle_v2.py with criterion (5)'s Z identity measured against its purity floor (the
ruling on (5) of gate D); otherwise the dense oracle on the 2-field 256-site time-dependent fixture, gate_dense_oracle_v1b.py
with the gate D judgment applied (busi/notes/C_gateD_judgment_v1.md); the engine under test is engine_twofield_v3 (v1b tested engine_twofield_v1).
The fixture (unchanged): R_n = 1 .. 256; prescribed fields of a collapsing marginally bound ball, 2gE = 48 sites, N(r, t) = 1 - 0.25 (1 - (r/R_b)^2)_+
and v(r, t) = -sqrt(2gE/r) for r >= R_b(t), -sqrt(2gE/R_b^3) r inside, R_b from 80 to 32 sites over 40 steps of tau 2, so that a trapping zero of
N - |v| appears at 48 sites once R_b < 48; the heavy channel kappa 1 at ma 0.2 (its sea and 6 occupied positive orbitals as dust), the light
channels kappa 1, 2, 3 at m = 0; every state propagated exactly by the step's eigendecomposition; the heavy reference continued by engine_ops_v3's
convention, the light by the certified mu S walk (engine_twofield_v3.light_continue_v36) and, beside it, by the declared convention
(engine_ops_v3.continue_sea_v3 on its own chain) and by version 34's eigenvalue route (its own chain). The fields outside R_b do not change with
t, so beyond 160 sites the exterior operator is the same at every step.
Oracled:
(1) [h_m, S] = 0 and Re h_m = 0 at the fixture's fields (1e-14); the light channel operator against engine_vacflow_v1's within 4 eps kappa (v1b
    demanded 0: page_ops_v5's vectorized complex division multiplies by the reciprocal of R_n, vacflow's scalar division divides; at kappa 3 and
    R_n = 5, 3 fl(1/5) and fl(3/5) differ by 1 ulp = 1.11e-16; the same operator to rounding);
(2) the certificate's claims, piece by piece (the certificate keeps its full piece list in v3): at every piece start s0 > 0 the true gap of K(s0)
    (dense eigvalsh) satisfies gap^2 >= d^2 - F with F = 64 eps ||K||_G^2 the Cholesky's rounding floor the certificate states; at s0 = 0,
    gap >= d - 16 eps ||K||_G; the pieces tile [0, 1] with lengths at most sqrt(d^2 - F)/||dK||_G (d itself on the first piece); ||dK||_2 <= ||dK||_G;
    derived from these, on 65 values of s the true gap is at least the Weyl bound d_rig - (s - s0) ||dK||_G and the count of negative eigenvalues of
    K(s) equals n_sea (v1b compared d with the true gap across the piece, a claim the certificate never makes); the smallest certified d^2/F printed;
(3) the light reference against the declared convention: the walk chain's mask (certified or fallen back) must equal the convention chain's at
    every record and the subspace residual be below 1e-10; the 64-sub-step maximum-overlap continuation (v1b's "truth": the diabatic branch of an
    avoided crossing when its sub-steps are coarser than the avoided gap, the adiabatic one when finer) and version 34's route are information,
    with the walk's certified count, first fallback and cut crossings per channel and the convention's first pocket per channel;
(4) as v1b: the light sources by the engine's sparse route against the dense Re diag(h (C - P_ref)) per site to 1e-12, E_inv by the coefficient
    route against the dense trace to 1e-10;
(5) S_esc every 4 steps by 3 routes (dense; the retained complement, the Z identity their difference; the retained channel): dense against retained
    channel to 1e-10; the Z identity, exact only for a projector C, against its purity floor 2n H((||C^2 - C||_1 + 16 n eps)/2n) with H the binary
    entropy and ||C^2 - C||_1 the measured trace-norm purity defect of the propagated state (v2 held it to 1e-10 and read 1.04e-10 at step 40: the
    floor, 1.25e-11 + 1.45e-11 sqrt(step) on this fixture, the per-step rounding of the unitary accumulating as a random walk and amplified by
    -x ln x at the eigenvalues near 1, not a defect); the identity on the spectral projector C0 of C is printed as the control; the archive
    the all-history collector of outgoing positive-frequency exterior modes beyond 160 sites projected against the archive (classical Gram-Schmidt
    twice) with page_ops's drop rule on the block's Gram eigenvalues, max(1e-12, 1e-14 g_max) on unit columns (v1b's rule, relative to the block's own
    largest singular value, kept 51 rounding directions per block on the static exterior: the archive outgrew the exterior's 192 dimensions and the
    Z identity broke at step 16, with no pocket involved); the Gram defect below 1e-12 and the column count at most the exterior dimension are
    criteria.
Usage: python gate_dense_oracle_v2b.py [--gpu 0] [--steps 40] [--out gateD_v2b]; prints PASS or the failing items."""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_twofield_v3 as E36  # noqa: E402
import engine_collapse_v5 as C5  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402
import engine_ops_v34_cert as CERT  # noqa: E402

BK = E36.BK
EPS = float(np.finfo(float).eps)


def fields_at(R, t, g2E=48.0, Rb0=80.0, Rb1=32.0, t_end=80.0):
    Rb = Rb0 + (Rb1 - Rb0) * min(max(t / t_end, 0.0), 1.0)
    N = 1.0 - 0.25 * np.clip(1.0 - (R / Rb) ** 2, 0.0, None)
    v = -np.sqrt(g2E / R)
    inside = R < Rb
    v[inside] = -np.sqrt(g2E / Rb ** 3) * R[inside]
    return N, v


def binary_entropy(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    q = x[(x > 1e-300) & (x < 1.0 - 1e-16)]
    return float(np.sum(-q * np.log(q) - (1.0 - q) * np.log1p(-q)))


def f_(x, f="%.2e"):
    """a number in format f; inf, None and nan printed as they are."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return (f % xf) if np.isfinite(xf) else str(x)


def fine_continuation(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, prev_mask, nsub=64):
    """version 3's maximum-overlap assignment at nsub uniform sub-steps of the panel (v1b's "truth"). Information in v2: at an avoided crossing
    it is the diabatic branch when the sub-steps' level motion exceeds the avoided gap and the adiabatic branch when it does not."""
    r = ref_prev; pm = prev_mask
    for k in range(1, nsub + 1):
        if k == nsub:
            e_s, U_s = e_cur, U_cur
        else:
            s = k / nsub
            e_s, U_s = E36.eigT(E36.clocked_gpu(ch, f_prev[0] + s * (f_cur[0] - f_prev[0]), f_prev[1] + s * (f_cur[1] - f_prev[1])))
        r, inf_, pm = OPS3.assign_max_overlap_v3(r, e_s, U_s, n_sea, prev_mask=pm)
    return r, np.asarray(pm, bool)


def resid(V, W):
    """||(1 - V V^dagger) W||_F for V with orthonormal columns."""
    return float((W - V @ (V.conj().T @ W)).norm().item())


def cert_oracle(ch, f_prev, f_cur, cert, S_host, n_sea, ns=65):
    """criterion (2): the certificate's claims against dense eigenvalues of K(s) = (1 - s) K0 + s K1. Returns margins (no arrays):
    norm_dK_2_minus_G (criterion <= 1e-12); start_margin_over_tol = (gap(0) - d)/(16 eps ||K||_G) on the first piece (criterion >= -1);
    cholesky_margin_d2_over_F = (gap(s0)^2 - d^2)/F at every Cholesky-certified start (criterion >= -1), with the margin in d and the smallest
    d^2/F beside it; tiling_defect (criterion <= 1e-12); the derived grid checks grid_weyl_margin (>= -1e-12) and grid_count_defect (0)."""
    K0 = (E36.clocked_host_sparse(ch, f_prev[0], f_prev[1]) - cert["mu0"] * S_host).tocsr()
    K1 = (E36.clocked_host_sparse(ch, f_cur[0], f_cur[1]) - cert["mu1"] * S_host).tocsr()
    GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))
    K0d = K0.toarray(); dKd = K1.toarray() - K0d
    nG = float(cert["norm_dK_G"])
    F = float(cert["rounding_F"]) if cert.get("rounding_F") is not None else 64.0 * EPS * GK * GK
    out = {"pieces": 0, "norm_dK_2_minus_G": float(np.max(np.abs(np.linalg.eigvalsh(dKd)))) - nG, "rounding_F": F, "norm_K_G": GK,
           "start_margin_over_tol": np.inf, "cholesky_margin_d2_over_F": np.inf, "cholesky_margin_d": np.inf, "soundness_min_d2_over_F": np.inf,
           "tiling_defect": 0.0, "grid_weyl_margin": np.inf, "grid_count_defect": 0, "grid_gap_min": np.inf}
    pieces = [tuple(float(x) for x in p) for p in (cert.get("piece_list") or [])]
    out["pieces"] = len(pieces)
    if not pieces or nG <= 0.0:
        return out
    rig = lambda s0, d: d if s0 == 0.0 else math.sqrt(max(d * d - F, 0.0))
    td = max(abs(pieces[0][0]), abs(pieces[-1][1] - 1.0))
    for i, (s0, s1, d) in enumerate(pieces):
        td = max(td, (s1 - s0) - rig(s0, d) / nG)
        if i:
            td = max(td, abs(s0 - pieces[i - 1][1]))
        g = float(np.min(np.abs(np.linalg.eigvalsh(K0d + s0 * dKd))))
        if s0 == 0.0:
            out["start_margin_over_tol"] = min(out["start_margin_over_tol"], (g - d) / (16.0 * EPS * GK))
        else:
            out["cholesky_margin_d2_over_F"] = min(out["cholesky_margin_d2_over_F"], (g * g - d * d) / F)
            out["cholesky_margin_d"] = min(out["cholesky_margin_d"], g - d)
            out["soundness_min_d2_over_F"] = min(out["soundness_min_d2_over_F"], d * d / F)
    out["tiling_defect"] = float(td)
    starts = np.array([p[0] for p in pieces], float)
    for s in np.linspace(0.0, 1.0, ns):
        ev = np.linalg.eigvalsh(K0d + s * dKd); g = float(np.min(np.abs(ev)))
        out["grid_gap_min"] = min(out["grid_gap_min"], g)
        if int(np.sum(ev < 0.0)) != n_sea:
            out["grid_count_defect"] += 1
        k = max(int(np.searchsorted(starts, s, side="right")) - 1, 0)
        s0, s1, d = pieces[k]
        out["grid_weyl_margin"] = min(out["grid_weyl_margin"], g - max(0.0, rig(s0, d) - (s - s0) * nG))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--nsites", type=int, default=256); ap.add_argument("--out", default="gateD_v2b"); ap.add_argument("--R-cut", type=float, default=160.0)
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--nsub-fine", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C5.backend(a.gpu)
    E36.tm_reset(); E36.tml_reset_v36()
    torch = BK["torch"]
    n = a.nsites
    args = SimpleNamespace(light_transport="banded", light_tol_block=1e-8, light_sigma_tol=1e-6, light_d_min=1e-7, light_max_pieces=4096, tau=a.tau, c_LZ=1.0,
                           weyl_refine_max=10, nsites=n)
    heavy = C5.Chan(1, 0.2, n)
    lights = [E36.LightChan(k, n) for k in (1, 2, 3)]
    R = heavy.R
    sgn_dev, S_host = E36.light_S_v36(n)
    chans = [("heavy", heavy)] + [("light", c) for c in lights]
    fails = []; notes = {}
    t_gate = time.time()
    # ---- (1) the symmetry at the fixture's fields and the operator against vacflow's (to rounding)
    N0, v0 = fields_at(R, 0.0)
    for c in lights:
        Hm = E36.clocked_host_sparse(c, N0, v0)
        comm = (Hm @ S_host - S_host @ Hm).tocsr(); re = Hm.real
        cm = float(abs(comm).max()) if comm.nnz else 0.0; rm = float(abs(re).max()) if re.nnz else 0.0
        oc = c.operator_check
        tol_op = 4.0 * EPS * c.kappa                       # v2: a few ulp of the largest on-site entry kappa/R_1 = kappa
        notes["symmetry_k%d" % c.kappa] = {"commutator_max": cm, "real_part_max": rm, "operator_check": oc, "operator_tolerance": tol_op}
        if cm > 1e-14 or rm > 1e-14 or max(oc["h_max_abs_diff_vs_vacflow"], oc["Q_max_abs_diff_vs_vacflow"]) > tol_op:
            fails.append("(1) symmetry or operator check failed on kappa %d: %s" % (c.kappa, notes["symmetry_k%d" % c.kappa]))
    print("(1) [h_m, S] and Re h_m at m = 0, the operator against vacflow's within 4 eps kappa: %s" % json.dumps(notes, default=float), flush=True)
    # ---- the states at t = 0
    st = {}
    e, U = E36.eigT(E36.clocked_gpu(heavy, N0, v0))
    neg = e < 0; kneg = int(neg.sum()); en = e.cpu().numpy()
    pos_idx = torch.nonzero(~neg).ravel()[:6]
    dust = U[:, pos_idx].clone()
    st["heavy"] = {"sea": U[:, neg].clone(), "dust": dust, "f": np.ones(6), "ref": U[:, neg].clone(), "n_sea": kneg,
                   "window": [{"ref": U[:, neg].clone(), "f": (N0.copy(), v0.copy()), "gap": float(en[kneg] - en[kneg - 1]), "weyl_gap": float(en[kneg] - en[kneg - 1]), "mask": neg.cpu().numpy()}]}
    stL, headL = E36.light_init_v36(args, lights, N0, v0, sgn_dev, S_host)
    for c in lights:
        st[("light", c.kappa)] = stL[c.key]
        st[("light3", c.kappa)] = {"ref": stL[c.key]["ref"].clone(), "window": [dict(stL[c.key]["window"][0])]}      # v2: the convention's own chain
        st[("light34", c.kappa)] = {"ref": stL[c.key]["ref"].clone(), "window": [dict(stL[c.key]["window"][0])]}
    print("the light seas at t = 0: %s" % json.dumps(headL, default=float), flush=True)
    archive = {}
    worst = {"cert_dK_2_minus_G": -np.inf, "cert_start_margin_over_tol": np.inf, "cert_cholesky_margin_d2_over_F": np.inf, "cert_cholesky_margin_d": np.inf,
             "cert_soundness_min_d2_over_F": np.inf, "cert_tiling_defect": 0.0, "cert_grid_weyl_margin": np.inf, "cert_grid_count_defect": 0, "cert_panels": 0,
             "cert_pieces": 0, "conv_mask_diff": 0, "conv_resid": 0.0, "fine_vs_conv_diff": 0, "fine_vs_conv_panels": 0, "src": 0.0, "einv": 0.0, "sesc": 0.0,
             "z": 0.0, "z_over_floor": 0.0, "purity_1": 0.0, "purity_2": 0.0, "z_purified": 0.0, "sesc_purity_shift": 0.0, "gram": 0.0, "archive_over_dim": 0,
             "heavy_fine_vs_v3_diff": 0}
    panels = []; sesc_rows = []; first_trapped = None
    first_pocket_conv = {}; first_fallback = {}; certified_panels = {c.kappa: 0 for c in lights}; cut_crossings = {c.kappa: 0 for c in lights}
    Rc = a.R_cut
    ext = np.asarray(R, float) > Rc
    ext_rows = np.repeat(ext, 2); n_ext2 = int(2 * ext.sum())
    for step in range(a.steps + 1):
        t = step * a.tau
        N, v = fields_at(R, t)
        trapped = bool(np.any(N - np.abs(v) < 0.0))
        if trapped and first_trapped is None:
            first_trapped = step
        f_cur = (N, v)
        eigs = {}
        for name, c in chans:
            key = "heavy" if name == "heavy" else ("light", c.kappa)
            e, U = E36.eigT(E36.clocked_gpu(c, N, v))
            s_ = st[key]
            if step > 0:
                prev = s_["window"][-1]
                if name == "heavy":
                    ref, info, mask = OPS3.continue_sea_v3(c, s_["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse)
                    mask = np.asarray(mask, bool)
                    r_f, m_f = fine_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"), a.nsub_fine)
                    worst["heavy_fine_vs_v3_diff"] = max(worst["heavy_fine_vs_v3_diff"], int(np.sum(mask != m_f)))
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap_between_blocks"), "weyl_gap": info.get("weyl_gap"), "mask": mask}]
                else:
                    # the engine's light continuation (the walk; U rotated in place to S's eigenbasis on degenerate blocks)
                    ref, info, mask = E36.light_continue_v36(args, c, s_["window"], f_cur, e, U, s_["n_sea"], sgn_dev, S_host)
                    mask = np.asarray(mask, bool)
                    cert = info.get("certified")
                    # ---- (3) the declared convention on its own chain: engine_ops_v3.continue_sea_v3 (one step at this fixture's rates)
                    s3 = st[("light3", c.kappa)]
                    ref3, inf3, mask3 = OPS3.continue_sea_v3(c, s3["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse)
                    mask3 = np.asarray(mask3, bool)
                    md3 = int(np.sum(mask != mask3)); rr3 = max(resid(ref, ref3), resid(ref3, ref))
                    worst["conv_mask_diff"] = max(worst["conv_mask_diff"], md3); worst["conv_resid"] = max(worst["conv_resid"], rr3)
                    if (inf3.get("n_pocket") or 0) > 0 and c.kappa not in first_pocket_conv:
                        first_pocket_conv[c.kappa] = step
                    # information: v1b's finely sub-stepped continuation from the walk chain's previous state, against the convention
                    r_f, m_f = fine_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"), a.nsub_fine)
                    mdf = int(np.sum(m_f != mask3))
                    worst["fine_vs_conv_diff"] = max(worst["fine_vs_conv_diff"], mdf); worst["fine_vs_conv_panels"] += int(mdf > 0)
                    # information: version 34's eigenvalue route on its own chain
                    s34 = st[("light34", c.kappa)]
                    ref34, info34, mask34 = CERT.continue_sea_v34(c, s34["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse, "eigvals", 10)
                    md34 = int(np.sum(np.asarray(mask34, bool) != mask3))
                    # ---- (2) the certificate's claims, piece by piece (also on a panel that handed over after a valid walk)
                    co = None
                    if isinstance(cert, dict) and cert.get("certified"):
                        co = cert_oracle(c, prev["f"], f_cur, cert, S_host, s_["n_sea"])
                        worst["cert_panels"] += 1; worst["cert_pieces"] += int(co["pieces"])
                        worst["cert_dK_2_minus_G"] = max(worst["cert_dK_2_minus_G"], co["norm_dK_2_minus_G"])
                        if co["pieces"]:
                            worst["cert_start_margin_over_tol"] = min(worst["cert_start_margin_over_tol"], co["start_margin_over_tol"])
                            worst["cert_cholesky_margin_d2_over_F"] = min(worst["cert_cholesky_margin_d2_over_F"], co["cholesky_margin_d2_over_F"])
                            worst["cert_cholesky_margin_d"] = min(worst["cert_cholesky_margin_d"], co["cholesky_margin_d"])
                            worst["cert_soundness_min_d2_over_F"] = min(worst["cert_soundness_min_d2_over_F"], co["soundness_min_d2_over_F"])
                            worst["cert_tiling_defect"] = max(worst["cert_tiling_defect"], co["tiling_defect"])
                            worst["cert_grid_weyl_margin"] = min(worst["cert_grid_weyl_margin"], co["grid_weyl_margin"])
                            worst["cert_grid_count_defect"] += int(co["grid_count_defect"])
                    if info.get("verified"):
                        certified_panels[c.kappa] += 1
                    elif c.kappa not in first_fallback:
                        first_fallback[c.kappa] = {"step": step, "why": info.get("why")}
                    if info.get("cut_crossing"):
                        cut_crossings[c.kappa] += 1
                    panels.append({"step": step, "kappa": c.kappa, "trapped": trapped, "walk_verified": bool(info.get("verified")), "walk_fallback": bool(info.get("fallback")),
                                   "walk_why": info.get("why"), "cut_crossing": bool(info.get("cut_crossing")), "cut_check": info.get("cut_check"), "walk_nsub": info.get("nsub"),
                                   "walk_pieces": (cert.get("pieces") if isinstance(cert, dict) else None), "walk_d_min": (cert.get("d_min_certified") if isinstance(cert, dict) else None),
                                   "walk_d_floor": (cert.get("d_floor") if isinstance(cert, dict) else None), "walk_certificate_valid": (bool(cert.get("certified")) if isinstance(cert, dict) else None),
                                   "conv_mask_diff": md3, "conv_resid": rr3, "conv_n_pocket": inf3.get("n_pocket"), "conv_inversions": inf3.get("inversions"),
                                   "fine_vs_conv_diff": mdf, "v34_verified": bool(info34.get("certified", {}).get("verified")), "v34_pieces": info34.get("nsub"),
                                   "v34_mask_vs_conv": md34, "cert_oracle": co})
                    if md3 or info.get("cut_crossing") or mdf or md34:
                        print("  step %d kappa %d: walk %s%s; convention mask differences %d (residual %.1e); fine %d-sub-step vs convention %d; v34 %s vs convention %d; convention pockets %s; %s" % (
                            step, c.kappa, "certified" if info.get("verified") else "FALLBACK", " at a cut crossing" if info.get("cut_crossing") else "", md3, rr3, a.nsub_fine, mdf,
                            "verified" if panels[-1]["v34_verified"] else "not verified", md34, inf3.get("n_pocket"), (info.get("why") or "")[:160]), flush=True)
                    s34["ref"] = ref34
                    s34["window"] = [{"ref": ref34, "f": (N.copy(), v.copy()), "gap": info34.get("gap_between_blocks"), "weyl_gap": info34.get("weyl_gap"), "mask": np.asarray(mask34, bool)}]
                    s3["ref"] = ref3
                    s3["window"] = [{"ref": ref3, "f": (N.copy(), v.copy()), "gap": inf3.get("gap_between_blocks"), "weyl_gap": inf3.get("weyl_gap"), "mask": mask3}]
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap"), "weyl_gap": info.get("weyl_gap", info.get("gap")), "mask": mask, "mu": info.get("mu"), "n_plus": info.get("n_plus")}]
                s_["ref"] = ref
            else:
                mask = np.asarray(s_["window"][0]["mask"], bool); ref = s_["ref"]
            eigs[key] = (e, U, ref, mask)
            # ---- (4) the sources and E_inv against dense arithmetic (as v1b)
            sea = s_["sea"]
            if name == "heavy":
                C = sea @ sea.conj().T + (s_["dust"] * torch.as_tensor(s_["f"], dtype=torch.complex128, device=sea.device)[None, :]) @ s_["dust"].conj().T
            else:
                C = sea @ sea.conj().T
            P = ref @ ref.conj().T
            hd = E36.clocked_gpu(c, N, v)
            eps_sparse = E36.dens(c.h, sea) - E36.dens(c.h, ref) + (E36.dens(c.h, s_["dust"], s_["f"]) if name == "heavy" else 0.0)
            eps_dense = torch.diagonal(E36.V35.dev_csr(c.hs) @ (C - P)).real.reshape(-1, 2).sum(dim=1).cpu().numpy()
            worst["src"] = max(worst["src"], float(np.max(np.abs(eps_sparse - eps_dense))))
            A = U.conj().T @ sea
            nocc = (A.abs() ** 2).sum(dim=1)
            e64 = e.to(torch.float64)
            E_coef = float((e64 * nocc).sum()) - float(e64[torch.as_tensor(mask, device=U.device)].sum())
            if name == "heavy":
                Ad = U.conj().T @ s_["dust"]
                E_coef += float((e64 * ((Ad.abs() ** 2) * torch.as_tensor(s_["f"], dtype=torch.float64, device=U.device)[None, :]).sum(dim=1)).sum())
            E_dense = float(torch.trace(hd @ (C - P)).real)
            worst["einv"] = max(worst["einv"], abs(E_coef - E_dense))
            # ---- (5) S_esc at the checkpoints; the archive with page_ops's drop rule
            if step % a.every == 0:
                hext = hd[ext_rows][:, ext_rows]
                ee, Ue = torch.linalg.eigh(hext)
                Xr = torch.as_tensor(np.repeat(np.asarray(R, float)[ext], 2), dtype=torch.complex128, device=hd.device)
                vel = (1j * (hext @ torch.diag(Xr) - torch.diag(Xr) @ hext))
                vexp = torch.einsum("ij,ji->i", Ue.conj().T, vel @ Ue).real
                sel = (ee > 1e-6) & (vexp > 0.0)
                new = torch.zeros((2 * n, int(sel.sum())), dtype=torch.complex128, device=hd.device)
                new[torch.as_tensor(ext_rows, device=hd.device)] = Ue[:, sel]
                Aold = archive.get(key)
                if Aold is not None:
                    new = new - Aold @ (Aold.conj().T @ new); new = new - Aold @ (Aold.conj().T @ new)
                n_cand = int(new.shape[1]); n_kept = 0
                if n_cand:
                    Q_, sv, _ = torch.linalg.svd(new, full_matrices=False)
                    gsv = (sv * sv).cpu().numpy()
                    keep = gsv > max(1e-12, 1e-14 * float(gsv.max()))        # v2: page_ops's rule on unit columns; a block already in the archive adds nothing
                    n_kept = int(keep.sum())
                    if n_kept:
                        newQ = Q_[:, torch.as_tensor(np.where(keep)[0], dtype=torch.int64, device=Q_.device)]
                        archive[key] = newQ if Aold is None else torch.cat([Aold, newQ], dim=1)
                    elif Aold is None:
                        archive[key] = new[:, :0]
                Ak = archive[key]
                gram = float((Ak.conj().T @ Ak - torch.eye(Ak.shape[1], dtype=torch.complex128, device=Ak.device)).abs().max()) if Ak.shape[1] else 0.0
                worst["gram"] = max(worst["gram"], gram)
                if Ak.shape[1] > n_ext2:
                    worst["archive_over_dim"] += 1
                M = Ak.conj().T @ C @ Ak
                S_dense = binary_entropy(torch.linalg.eigvalsh(0.5 * (M + M.conj().T)).cpu().numpy())
                Pi = Ak @ Ak.conj().T
                lam, V = torch.linalg.eigh(0.5 * (Pi + Pi.conj().T))
                Qc = V[:, lam < 0.5]
                Mc = Qc.conj().T @ C @ Qc
                S_comp = binary_entropy(torch.linalg.eigvalsh(0.5 * (Mc + Mc.conj().T)).cpu().numpy())
                if name == "heavy":
                    cols = torch.cat([sea, s_["dust"]], dim=1)
                else:
                    cols = sea
                B = Ak.conj().T @ cols
                svB = torch.linalg.svdvals(B).cpu().numpy()
                S_mem = binary_entropy(svB * svB)
                # v2b: the Z identity S(A) = S(A^perp) is exact for a projector C; the propagated C is a projector only to rounding, and a defect d in an
                # eigenvalue near 1 costs H(d) = d (1 + ln 1/d) of entropy, so the identity's floor is 2n H((||C^2 - C||_1 + 2n rho)/(2n)) with rho = 8 eps
                # the per-eigenvalue slack of the matrix products and the eigensolvers (C = C0 + Delta with C0 the spectral projector of C, for which the
                # identity is exact; pinching, Mirsky, |H(x) - H(y)| <= H(|x - y|), Jensen over the 2n eigenvalues); the identity on C0 is the control
                cC, VC = torch.linalg.eigh(0.5 * (C + C.conj().T))
                cc = cC.cpu().numpy(); pur = np.abs(cc * (cc - 1.0))          # the moduli of the eigenvalues of C^2 - C
                purity_1 = float(pur.sum()); purity_2 = float(pur.max())
                hz = (purity_1 + 2.0 * n * 8.0 * EPS) / (2.0 * n)
                z_floor = 2.0 * n * (-hz * math.log(hz) - (1.0 - hz) * math.log1p(-hz))
                V1 = VC[:, cC >= 0.5]; C0 = V1 @ V1.conj().T
                M0 = Ak.conj().T @ C0 @ Ak; Mc0 = Qc.conj().T @ C0 @ Qc
                S0_dense = binary_entropy(torch.linalg.eigvalsh(0.5 * (M0 + M0.conj().T)).cpu().numpy())
                S0_comp = binary_entropy(torch.linalg.eigvalsh(0.5 * (Mc0 + Mc0.conj().T)).cpu().numpy())
                z_ = abs(S_dense - S_comp); z0_ = abs(S0_dense - S0_comp)
                worst["sesc"] = max(worst["sesc"], abs(S_dense - S_mem)); worst["z"] = max(worst["z"], z_); worst["z_over_floor"] = max(worst["z_over_floor"], z_ / z_floor)
                worst["purity_1"] = max(worst["purity_1"], purity_1); worst["purity_2"] = max(worst["purity_2"], purity_2)
                worst["z_purified"] = max(worst["z_purified"], z0_); worst["sesc_purity_shift"] = max(worst["sesc_purity_shift"], abs(S_dense - S0_dense))
                sesc_rows.append({"step": step, "t": t, "key": str(key), "w": c.w, "archive_columns": int(Ak.shape[1]), "exterior_dimension": n_ext2, "block_candidates": n_cand,
                                  "block_kept": n_kept, "gram_defect": gram, "S_dense": S_dense, "S_complement": S_comp, "S_retained_channel": S_mem, "weighted": c.w * S_dense,
                                  "purity_defect_1": purity_1, "purity_defect_2": purity_2, "Z_identity": z_, "Z_floor": z_floor, "Z_over_floor": z_ / z_floor,
                                  "Z_identity_purified": z0_, "S_dense_purified": S0_dense})
        # ---- the propagation
        for name, c in chans:
            key = "heavy" if name == "heavy" else ("light", c.kappa)
            e, U, ref, mask = eigs[key]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[key]
            s_["sea"] = U @ (ph[:, None] * (U.conj().T @ s_["sea"]))
            if name == "heavy":
                s_["dust"] = U @ (ph[:, None] * (U.conj().T @ s_["dust"]))
        if step % a.every == 0:
            tot = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step)
            guard = sum(r_["weighted"] for r_ in sesc_rows if r_["step"] == step and r_["key"] == str(("light", 3)))
            arch = {r_["key"]: r_["archive_columns"] for r_ in sesc_rows if r_["step"] == step}
            print("step %d t %.1f trapped region %s: S_esc weighted total %.6f (guard kappa 3 share %.3e; archive columns %s, Gram defect %.1e); worst so far: certificate start %s tol, Cholesky %s F (in d %s), tiling %s, grid Weyl %s, counts %d; convention mask differences %d residual %.1e; fine vs convention %d; sources %.2e, E_inv %.2e, S_esc routes %.2e, Z identity %.2e at %.2f of its purity floor (||C^2 - C||_1 %.1e, purified %.1e); heavy fine vs v3 %d (%.0f s)" % (
                step, t, trapped, tot, guard / tot if tot else 0.0, arch, worst["gram"], f_(worst["cert_start_margin_over_tol"], "%.2f"), f_(worst["cert_cholesky_margin_d2_over_F"], "%.2f"),
                f_(worst["cert_cholesky_margin_d"]), f_(worst["cert_tiling_defect"], "%.1e"), f_(worst["cert_grid_weyl_margin"]), worst["cert_grid_count_defect"], worst["conv_mask_diff"],
                worst["conv_resid"], worst["fine_vs_conv_diff"], worst["src"], worst["einv"], worst["sesc"], worst["z"], worst["z_over_floor"], worst["purity_1"], worst["z_purified"],
                worst["heavy_fine_vs_v3_diff"], time.time() - t_gate), flush=True)
    # ---- the verdicts
    if worst["cert_dK_2_minus_G"] > 1e-12:
        fails.append("(2) ||dK||_2 exceeds the certificate's Gershgorin bound by %.2e" % worst["cert_dK_2_minus_G"])
    if worst["cert_start_margin_over_tol"] < -1.0:
        fails.append("(2) a walk's start d exceeds the true gap of K(0) by %.2f times the eigensolver's tolerance 16 eps ||K||_G" % (-worst["cert_start_margin_over_tol"]))
    if worst["cert_cholesky_margin_d2_over_F"] < -1.0:
        fails.append("(2) a Cholesky-certified piece start has gap^2 - d^2 = %.2f F, below the stated floor -F (in d: %.2e)" % (worst["cert_cholesky_margin_d2_over_F"], worst["cert_cholesky_margin_d"]))
    if worst["cert_tiling_defect"] > 1e-12:
        fails.append("(2) the pieces do not tile [0, 1] within their Weyl lengths (defect %.2e)" % worst["cert_tiling_defect"])
    if worst["cert_grid_weyl_margin"] < -1e-12:
        fails.append("(2) the true gap falls below the Weyl bound on the grid by %.2e" % (-worst["cert_grid_weyl_margin"]))
    if worst["cert_grid_count_defect"] > 0:
        fails.append("(2) the count of negative eigenvalues of K(s) differs from n_sea at %d grid points" % worst["cert_grid_count_defect"])
    print("(2) certificate: %d certified walks (%d pieces) oracled; ||dK||_2 - ||dK||_G at most %s; start margin %s tolerances (criterion >= -1); Cholesky margin (gap^2 - d^2)/F at least %s (criterion >= -1; in d %s); tiling defect %s; grid Weyl margin %s (criterion >= -1e-12); grid count defects %d; the smallest certified d^2/F %s (4 by construction)" % (
        worst["cert_panels"], worst["cert_pieces"], f_(worst["cert_dK_2_minus_G"]), f_(worst["cert_start_margin_over_tol"], "%.2f"), f_(worst["cert_cholesky_margin_d2_over_F"], "%.3f"),
        f_(worst["cert_cholesky_margin_d"]), f_(worst["cert_tiling_defect"], "%.1e"), f_(worst["cert_grid_weyl_margin"]), worst["cert_grid_count_defect"], f_(worst["cert_soundness_min_d2_over_F"], "%.1f")), flush=True)
    if worst["conv_mask_diff"] > 0 or worst["conv_resid"] > 1e-10:
        fails.append("(3) the light reference differs from the declared convention (engine_ops_v3's continuation): mask differences %d, residual %.2e" % (worst["conv_mask_diff"], worst["conv_resid"]))
    n_light_panels = a.steps * len(lights)
    print("(3) convention: mask differences at most %d, residual %.2e over %d panels; the walk certified %s panels per channel, first fallback %s, cut crossings %s; the convention's first pocket per channel %s; the trapped region from step %s; the %d-sub-step continuation differs from the convention on %d panels (at most %d levels)" % (
        worst["conv_mask_diff"], worst["conv_resid"], n_light_panels, json.dumps(certified_panels), json.dumps(first_fallback, default=str), json.dumps(cut_crossings), json.dumps(first_pocket_conv),
        first_trapped, a.nsub_fine, worst["fine_vs_conv_panels"], worst["fine_vs_conv_diff"]), flush=True)
    if worst["src"] > 1e-12:
        fails.append("(4) the sources differ from the dense route by %.2e" % worst["src"])
    if worst["einv"] > 1e-10:
        fails.append("(4) E_inv differs from the dense trace by %.2e" % worst["einv"])
    if worst["sesc"] > 1e-10:
        fails.append("(5) S_esc's dense and retained-channel routes differ by %.2e" % worst["sesc"])
    if worst["z_over_floor"] > 1.0:
        fails.append("(5) the Z identity's defect %.2e is %.2f times its purity floor 2n H((||C^2 - C||_1 + 16 n eps)/2n) (||C^2 - C||_1 at most %.2e; on the purified C0 the identity holds to %.2e)" % (
            worst["z"], worst["z_over_floor"], worst["purity_1"], worst["z_purified"]))
    if worst["gram"] > 1e-12:
        fails.append("(5) the archive's Gram defect reached %.2e" % worst["gram"])
    if worst["archive_over_dim"] > 0:
        fails.append("(5) the archive exceeded the exterior dimension %d at %d checkpoints" % (n_ext2, worst["archive_over_dim"]))
    print("(5) archive: Gram defect at most %.1e (criterion 1e-12), columns %s against the exterior dimension %d; S_esc routes %.2e (criterion 1e-10); Z identity %.2e at %.2f of its purity floor 2n H((||C^2 - C||_1 + 16 n eps)/2n) (criterion <= 1; ||C^2 - C||_1 at most %.2e, ||C^2 - C||_2 at most %.2e); on the purified C0 the identity holds to %.2e and S_esc shifts by at most %.2e (information)" % (
        worst["gram"], sorted({r_["archive_columns"] for r_ in sesc_rows}), n_ext2, worst["sesc"], worst["z"], worst["z_over_floor"], worst["purity_1"], worst["purity_2"],
        worst["z_purified"], worst["sesc_purity_shift"]), flush=True)
    n_cert_before = sum(1 for q in panels if not q["trapped"] and q["v34_verified"]); n_before = sum(1 for q in panels if not q["trapped"])
    n_cert_after = sum(1 for q in panels if q["trapped"] and q["v34_verified"]); n_after = sum(1 for q in panels if q["trapped"])
    n_walk_after = sum(1 for q in panels if q["trapped"] and q["walk_verified"])
    print("version 34's eigenvalue route (information): before the trapped region certified %d of %d panels, after %d of %d; the walk certified %d of %d after; v34 mask differences from the convention: %s; heavy channel: the %d-sub-step continuation differs from version 3's one step on at most %d levels" % (
        n_cert_before, n_before, n_cert_after, n_after, n_walk_after, n_after, sorted({q["v34_mask_vs_conv"] for q in panels}), a.nsub_fine, worst["heavy_fine_vs_v3_diff"]), flush=True)
    with open(os.path.join(a.out, "gate_dense_oracle_v2b_result.json"), "w") as fh:
        json.dump({"worst": worst, "symmetry": notes, "panels": panels, "S_esc": sesc_rows, "first_trapped_step": first_trapped, "first_pocket_convention": first_pocket_conv,
                   "first_fallback": first_fallback, "certified_panels": certified_panels, "cut_crossings": cut_crossings, "exterior_dimension": n_ext2, "fails": fails,
                   "pass": not fails, "seconds": time.time() - t_gate}, fh, default=float, indent=1)
    print("GATE D: %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)), flush=True)


if __name__ == "__main__":
    main()
