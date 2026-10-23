#!/usr/bin/env python
"""gate_dense_oracle_v1.py (busi lane, Computation C): the dense oracle on a 2-field 256-site time-dependent fixture with a trapping zero.
The fixture: R_n = 1 .. 256; prescribed fields of a collapsing marginally bound ball, 2gE = 48 sites, N(r, t) = 1 - 0.25 (1 - (r/R_b)^2)_+ and
v(r, t) = -sqrt(2gE/r) for r >= R_b(t), -sqrt(2gE/R_b^3) r inside, R_b from 80 to 32 sites over 40 steps of tau 2, so that a trapping zero of
N - |v| appears at 48 sites once R_b < 48 and a pocket region inside it; the heavy channel kappa 1 at ma 0.2 (its sea and 6 occupied positive
orbitals as dust), the light channels kappa 1, 2, 3 at m = 0 built by engine_twofield_v1.LightChan; every state propagated exactly by the
step's eigendecomposition (the engine's convention); the references continued by the engine's own code: the heavy by engine_ops_v3's
convention, the light by the certified mu S walk (engine_twofield_v1.light_continue_v36) and, beside it, by version 34's eigenvalue route.
Oracled: (1) [h_m, S] = 0 and the imaginary-ness of the massless h_m at the fixture's fields; (2) the certificate's validity: on every certified
piece the certified d against the true gap of K(s) on 65 s values per panel (margin at least -1e-12); (3) the walk's continued mask against the
fine-sub-step maximum-overlap ground truth (64 sub-steps per panel) and the subspace residual below 1e-10; version 34's route on the same
panels (certified before the pockets, failing after, its mask against the truth); (4) the light sources by the engine's sparse route against the
dense Re diag(h (C - P_ref)) per site to 1e-12, E_inv by the coefficient route against the dense trace to 1e-10; (5) S_esc every 4 steps by 3
routes to 1e-10: dense (the full covariance restricted to the archive), the retained complement (the orthogonal complement of the archive;
the Z identity printed as their difference), the retained-channel route (the state's columns restricted to the archive, no 2n x 2n covariance);
the archive the all-history collector of outgoing positive-frequency exterior modes beyond 160 sites, projected against the archive and
Gram-checked every block (drop rule 1e-12), weighted 16 kappa per channel, the guard kappa 3's share printed.
Usage: python gate_dense_oracle_v1.py [--gpu 0] [--steps 40] [--out gateD] ; prints PASS or the failing item."""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_twofield_v1 as E36  # noqa: E402
import engine_collapse_v5 as C5  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402
import engine_ops_v34_cert as CERT  # noqa: E402

BK = E36.BK


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


def truth_continuation(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, prev_mask, nsub=64):
    """the adiabatic ground truth: version 3's maximum-overlap assignment at nsub sub-steps of the panel."""
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
    return float((W - V @ (V.conj().T @ W)).norm().item())


def true_gap_grid(ch, f_prev, f_cur, mu0, mu1, S, ns=65):
    K0 = (E36.clocked_host_sparse(ch, f_prev[0], f_prev[1]) - mu0 * S).toarray(); K1 = (E36.clocked_host_sparse(ch, f_cur[0], f_cur[1]) - mu1 * S).toarray()
    ss = np.linspace(0.0, 1.0, ns); g = np.empty(ns)
    for i, s in enumerate(ss):
        g[i] = float(np.min(np.abs(np.linalg.eigvalsh((1.0 - s) * K0 + s * K1))))
    return ss, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--nsites", type=int, default=256); ap.add_argument("--out", default="gateD"); ap.add_argument("--R-cut", type=float, default=160.0)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C5.backend(a.gpu)
    E36.tm_reset(); E36.tml_reset_v36()
    torch = BK["torch"]
    args = SimpleNamespace(light_transport="banded", light_tol_block=1e-8, light_sigma_tol=1e-6, light_d_min=1e-7, light_max_pieces=4096, tau=a.tau, c_LZ=1.0, weyl_refine_max=10)
    n = a.nsites
    heavy = C5.Chan(1, 0.2, n)
    lights = [E36.LightChan(k, n) for k in (1, 2, 3)]
    R = heavy.R
    sgn_dev, S_host = E36.light_S_v36(n)
    chans = [("heavy", heavy)] + [("light", c) for c in lights]
    fails = []; notes = {}
    # ---- (1) the symmetry at the fixture's fields
    N0, v0 = fields_at(R, 0.0)
    for c in lights:
        Hm = E36.clocked_host_sparse(c, N0, v0)
        comm = (Hm @ S_host - S_host @ Hm).tocsr(); re = Hm.real
        cm = float(abs(comm).max()) if comm.nnz else 0.0; rm = float(abs(re).max()) if re.nnz else 0.0
        notes["symmetry_k%d" % c.kappa] = {"commutator_max": cm, "real_part_max": rm, "operator_check": c.operator_check}
        if cm > 1e-14 or rm > 1e-14 or max(c.operator_check["h_max_abs_diff_vs_vacflow"], c.operator_check["Q_max_abs_diff_vs_vacflow"]) > 0.0:
            fails.append("(1) symmetry or operator check failed on kappa %d: %s" % (c.kappa, notes["symmetry_k%d" % c.kappa]))
    print("(1) [h_m, S] and Re h_m at m = 0: %s" % json.dumps(notes, default=float), flush=True)
    # ---- the states at t = 0
    st = {}
    e, U = E36.eigT(E36.clocked_gpu(heavy, N0, v0))
    neg = e < 0; kneg = int(neg.sum()); en = e.cpu().numpy()
    pos_idx = torch.nonzero(~neg).ravel()[:6]
    dust = U[:, pos_idx].clone()
    st["heavy"] = {"sea": U[:, neg].clone(), "dust": dust, "f": np.ones(6), "ref": U[:, neg].clone(), "n_sea": kneg,
                   "window": [{"ref": U[:, neg].clone(), "f": (N0.copy(), v0.copy()), "gap": float(en[kneg] - en[kneg - 1]), "weyl_gap": float(en[kneg] - en[kneg - 1]), "mask": neg.cpu().numpy()}]}
    stL, headL = E36.light_init_v36(args_ns(args, n), lights, N0, v0, sgn_dev, S_host)
    for c in lights:
        st[("light", c.kappa)] = stL[c.key]
        st[("light34", c.kappa)] = {"ref": stL[c.key]["ref"].clone(), "window": [dict(stL[c.key]["window"][0])]}
    print("the light seas at t = 0: %s" % json.dumps(headL, default=float), flush=True)
    archive = {}
    worst = {"cert_margin": np.inf, "walk_resid": 0.0, "walk_mask_diff": 0, "src": 0.0, "einv": 0.0, "sesc": 0.0, "z": 0.0, "v3_mask_diff": 0}
    v34_log = []; first_pocket = None; sesc_rows = []
    Rc = a.R_cut
    ext = np.asarray(R, float) > Rc
    ext_rows = np.repeat(ext, 2)
    for step in range(a.steps + 1):
        t = step * a.tau
        N, v = fields_at(R, t)
        pockets_now = bool(np.any(N - np.abs(v) < 0.0))
        if pockets_now and first_pocket is None:
            first_pocket = step
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
                    r_t, m_t = truth_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"))
                    worst["v3_mask_diff"] = max(worst["v3_mask_diff"], int(np.sum(np.asarray(mask, bool) != m_t)))
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap_between_blocks"), "weyl_gap": info.get("weyl_gap"), "mask": np.asarray(mask, bool)}]
                else:
                    ref, info, mask = E36.light_continue_v36(args_ns(args, n), c, s_["window"], f_cur, e, U, s_["n_sea"], sgn_dev, S_host)
                    mask = np.asarray(mask, bool)
                    r_t, m_t = truth_continuation(c, prev["ref"], prev["f"], f_cur, e, U, s_["n_sea"], prev.get("mask"))
                    rr = resid(ref, r_t); md = int(np.sum(mask != m_t))
                    cert = info.get("certified")
                    if info.get("verified"):
                        worst["walk_resid"] = max(worst["walk_resid"], rr); worst["walk_mask_diff"] = max(worst["walk_mask_diff"], md)
                        ss, gtrue = true_gap_grid(c, prev["f"], f_cur, cert["mu0"], cert["mu1"], S_host)
                        margin = np.inf
                        if cert["piece_list"]:
                            for (s0, s1, d) in cert["piece_list"]:
                                sel = (ss >= s0 - 1e-15) & (ss <= s1 + 1e-15)
                                if sel.any():
                                    margin = min(margin, float(gtrue[sel].min() - d))
                        else:
                            margin = min(margin, float(gtrue.min() - cert["d_min_certified"]))
                        worst["cert_margin"] = min(worst["cert_margin"], margin)
                    # version 34's eigenvalue route on its own chain
                    s34 = st[("light34", c.kappa)]
                    ref34, info34, mask34 = CERT.continue_sea_v34(c, s34["window"], f_cur, e, U, s_["n_sea"], a.tau, 1.0, E36.eig_at, E36.clocked_host_sparse, "eigvals", 10)
                    v34_log.append({"step": step, "kappa": c.kappa, "pockets": pockets_now, "v34_verified": bool(info34.get("certified", {}).get("verified")), "v34_pieces": info34.get("nsub"),
                                    "v34_mask_vs_truth": int(np.sum(np.asarray(mask34, bool) != m_t)), "walk_verified": bool(info.get("verified")), "walk_pieces": info.get("nsub"),
                                    "walk_d_min": (cert or {}).get("d_min_certified") if isinstance(cert, dict) else None, "walk_mask_vs_truth": md, "walk_resid_vs_truth": rr,
                                    "pockets_in_mask": info.get("n_pocket")})
                    s34["ref"] = ref34
                    s34["window"] = [{"ref": ref34, "f": (N.copy(), v.copy()), "gap": info34.get("gap_between_blocks"), "weyl_gap": info34.get("weyl_gap"), "mask": np.asarray(mask34, bool)}]
                    s_["window"] = [{"ref": ref, "f": (N.copy(), v.copy()), "gap": info.get("gap"), "weyl_gap": info.get("weyl_gap", info.get("gap")), "mask": mask, "mu": info.get("mu"), "n_plus": info.get("n_plus")}]
                s_["ref"] = ref
            else:
                mask = np.asarray(s_["window"][0]["mask"], bool); ref = s_["ref"]
            eigs[key] = (e, U, ref, mask)
            # ---- (4) the sources and E_inv against dense arithmetic
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
            # ---- (5) S_esc at the checkpoints
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
                if new.shape[1]:
                    Q_, sv, _ = torch.linalg.svd(new, full_matrices=False)
                    keep = (sv * sv) > 1e-12 * max(float((sv * sv).max()), 1e-300)
                    newQ = Q_[:, keep]
                    archive[key] = newQ if Aold is None else torch.cat([Aold, newQ], dim=1)
                Ak = archive[key]
                gram = float((Ak.conj().T @ Ak - torch.eye(Ak.shape[1], dtype=torch.complex128, device=Ak.device)).abs().max())
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
                worst["sesc"] = max(worst["sesc"], abs(S_dense - S_mem)); worst["z"] = max(worst["z"], abs(S_dense - S_comp))
                sesc_rows.append({"step": step, "t": t, "key": str(key), "w": c.w, "archive_columns": int(Ak.shape[1]), "gram_defect": gram, "S_dense": S_dense, "S_complement": S_comp,
                                  "S_retained_channel": S_mem, "weighted": c.w * S_dense})
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
            print("step %d t %.1f pockets %s: S_esc weighted total %.6f (guard kappa 3 share %.3e); worst so far: certificate margin %.2e, walk residual %.2e, walk mask differences %d, sources %.2e, E_inv %.2e, S_esc routes %.2e, Z identity %.2e, heavy v3 mask differences %d" % (
                step, t, pockets_now, tot, guard / tot if tot else 0.0, worst["cert_margin"], worst["walk_resid"], worst["walk_mask_diff"], worst["src"], worst["einv"], worst["sesc"], worst["z"], worst["v3_mask_diff"]), flush=True)
    # ---- the verdicts
    if worst["cert_margin"] < -1e-12:
        fails.append("(2) a certified piece's d exceeds the true gap by %.2e" % (-worst["cert_margin"]))
    if worst["walk_resid"] > 1e-10 or worst["walk_mask_diff"] > 0:
        fails.append("(3) the walk's continuation differs from the fine-sub-step truth: residual %.2e, mask differences %d" % (worst["walk_resid"], worst["walk_mask_diff"]))
    if worst["src"] > 1e-12:
        fails.append("(4) the sources differ from the dense route by %.2e" % worst["src"])
    if worst["einv"] > 1e-10:
        fails.append("(4) E_inv differs from the dense trace by %.2e" % worst["einv"])
    if worst["sesc"] > 1e-10 or worst["z"] > 1e-10:
        fails.append("(5) S_esc's routes differ by %.2e (retained channel) and %.2e (the Z identity)" % (worst["sesc"], worst["z"]))
    n_cert_before = sum(1 for q in v34_log if not q["pockets"] and q["v34_verified"]); n_before = sum(1 for q in v34_log if not q["pockets"])
    n_cert_after = sum(1 for q in v34_log if q["pockets"] and q["v34_verified"]); n_after = sum(1 for q in v34_log if q["pockets"])
    n_walk_after = sum(1 for q in v34_log if q["pockets"] and q["walk_verified"])
    print("version 34's eigenvalue route against the walk: before the pockets v34 certified %d of %d panels, after %d of %d; the walk certified %d of %d after; the first pocket at step %s; v34 mask differences from the truth after the pockets: %s" % (
        n_cert_before, n_before, n_cert_after, n_after, n_walk_after, n_after, first_pocket, sorted({q["v34_mask_vs_truth"] for q in v34_log if q["pockets"]})), flush=True)
    if n_after and n_walk_after < n_after:
        print("note: the walk fell back on %d of %d panels after the pockets (the fallback is the engine's marked convention; the certified panels passed their oracle)" % (n_after - n_walk_after, n_after), flush=True)
    with open(os.path.join(a.out, "gate_dense_oracle_v1_result.json"), "w") as fh:
        json.dump({"worst": worst, "symmetry": notes, "v34_vs_walk": v34_log, "S_esc": sesc_rows, "first_pocket_step": first_pocket, "fails": fails, "pass": not fails}, fh, default=float, indent=1)
    print("GATE D: %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)), flush=True)


def args_ns(args, n):
    return args


if __name__ == "__main__":
    main()
