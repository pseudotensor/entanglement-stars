#!/usr/bin/env python
"""page_graft_posbound_v1.py (the solve lane's analyst builder 12; engine spec 54.2 (2), (3)): the graft's positive bound on its join
defect, from the stored hold and the frozen record alone (no propagation), on the box CPU.

The graft's state is P_long + Delta0 with Delta0 = Phi Phi^dagger - P_-^short (version 2's certified truncation with --trunc 1, the chains'
construction; the full Delta0 beside). Its purity defect is -(D Delta0 + Delta0 D), D = P_walled - P_long; the rows' readouts use the
polynomial D_sigma = p(H_walled) - p(H_connected) (page_graft_pullback_v2's step_coeffs and cheb_filter_pair, imported read-only). For a
detector O(t) = W O W^dagger (W an isometry, so ||O(t)|| <= ||O||): |Tr(O(t)(D Delta0 + Delta0 D))| <= 2 ||O|| ||D Delta0||_1, and the
ruled bound (54.2 (2), lead's arithmetic) is 4 ||O|| ||D_sigma Delta0||_1 per detector. D_sigma is exactly local: p has degree M and
p(H_w) - p(H_c) is nonzero only between rows and columns within M index ranges of the wall, so D_sigma Delta0 = D_sigma[win, win] Y with
Y = Delta0[win_s, :] (win_s the window's rows on the short chain). Its singular values are those of D_sigma[win, win] R with R R^dagger =
Y Y^dagger = (Delta0^2)[win_s, win_s] (R from that block's eigendecomposition), so the Frobenius and trace norms come from one filter of R.
Printed per hold and channel, at sigma and at 2 sigma and for their difference (the filter's term): ||D_s Delta0||_F, ||D_s Delta0||_1, and
4 ||O|| ||D_s Delta0||_1 for O = J_SS (L), the windowed split at 32 sites (E_flux_out(32), its operator-norm factor split_weight), the
inside energy (I, h's Gershgorin bound), against the digit 1e-3 L_H; beside: the hold's own ||C_S^2 - C_S||_F (C_S = Phi Phi^dagger), the
window's reach (2 (M + 1) ranges against M), and the measured sup-norm error of p against the step on a fine grid at sigma and at 2 sigma
(the 2 sigma polynomial truncated at M as the readouts use it) beside step_coeffs' coefficient tail eps.
Outputs (rule 10): <out>/<tag>_posbound.json (every number above, the singular values of D_s Delta0 at sigma and 2 sigma) and
<tag>_posbound.md.
Usage: --record --fields --init-state --freeze-at-rs --LH --tag --out [--append 8192] [--channels 1,2] [--threads 16] [--trunc 1]."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_graft_pullback_v2 as G2   # noqa: E402  (read-only)
G1 = G2.G1; PK = G2.PK; OPS = G2.OPS; CD = G2.CD
T0 = time.time()


def log(m):
    print("[%7.1f s] %s" % (time.time() - T0, m), flush=True)


def cheb_eval(c, x):
    """sum_n c_n T_n(x) by Clenshaw."""
    b1 = np.zeros_like(x); b2 = np.zeros_like(x)
    for cn in c[:0:-1]:
        b1, b2 = 2.0 * x * b1 - b2 + cn, b1
    return x * b1 - b2 + c[0]


def sup_error(c, sigma, lam, npts=2000001):
    from scipy.special import erfc
    x = np.cos(np.linspace(0.0, math.pi, npts))
    f = 0.5 * erfc(lam * x / (2.0 * sigma))
    return float(np.max(np.abs(cheb_eval(np.real(np.asarray(c)), x) - f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True); ap.add_argument("--fields-key", default="applied")
    ap.add_argument("--init-state", required=True); ap.add_argument("--nsites-short", type=int, default=8192); ap.add_argument("--append", type=int, default=8192)
    ap.add_argument("--channels", default="1,2"); ap.add_argument("--freeze-at-rs", type=float, required=True)
    ap.add_argument("--R-collar-rs", type=float, default=16.0); ap.add_argument("--windows", default="32,16,64")
    ap.add_argument("--null-sigma-TH", type=float, default=8.0); ap.add_argument("--null-tol", type=float, default=1e-12)
    ap.add_argument("--trunc", type=int, default=1); ap.add_argument("--trunc-drop-sum", type=float, default=0.0)
    ap.add_argument("--trunc-digit-frac", type=float, default=1e-3); ap.add_argument("--LH", type=float, required=True)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    os.makedirs(a.out, exist_ok=True); pre = os.path.join(a.out, a.tag)
    Hs = PK.History(a.record, a.fields, a.nsites_short, None, a.fields_key)
    Hl = PK.History(a.record, a.fields, a.nsites_short + a.append, None, a.fields_key)
    rs = float(Hs.rs_nom or 2.0 * Hs.g * Hs.E[0])
    kf = int(np.argmin(np.abs(Hs.T - a.freeze_at_rs * rs))); tf = float(Hs.T[kf])
    Rc_abs = a.R_collar_rs * rs
    wins = [int(x) for x in a.windows.split(",") if x.strip()]
    st = torch.load(a.init_state, map_location="cpu", weights_only=False)
    Nl, vl = Hl.fields(kf); Ns, vs = Hs.fields(kf)
    Rl = Hl.R; ns = a.nsites_short; n2s = 2 * ns
    ic = int(np.searchsorted(Rl, Rc_abs, side="right"))
    T_H = 1.0 / (4.0 * math.pi * rs)
    digit = a.trunc_digit_frac * a.LH
    log("graft positive bound v1: record kf %d T/r_s %.4f, short %d, appended %d; digit %.4e; trunc %d" % (kf, tf / rs, ns, a.append, digit, a.trunc))
    res = {"settings": vars(a), "kf": kf, "T_freeze": tf, "rs": rs, "T_H": T_H, "digit": digit, "channels": {},
           "label": "the graft's positive bound on its join defect (54.2 (2), (3)): 4 ||O|| ||D_sigma Delta0||_1 per detector"}
    for kap in [int(x) for x in a.channels.split(",") if x.strip()]:
        t_c = time.time()
        ops_s = PK.ChannelOps(kap, Hs.R, dev); ops_l = PK.ChannelOps(kap, Rl, dev)
        hs = ops_s.h_csr(Ns, vs); hl = ops_l.h_csr(Nl, vl)
        e, V = OPS.eigh_csr(hs, dev)
        Vn = V[:, e < 0].clone(); del V, e
        log("kappa %d: short chain diagonalized (%d occupied)" % (kap, Vn.shape[1]))
        Phi = st["channels"][str(kap)]["Phi"].to(dev)
        # the hold's own purity: ||C_S^2 - C_S||_F with C_S = Phi Phi^dagger, from the Gram matrix g: ||Phi (g - 1) Phi^dagger||_F^2 = Tr(g (g-1) g (g-1))
        g = Phi.mH @ Phi; gm = g - torch.eye(g.shape[0], dtype=g.dtype)
        hold_pur = float(torch.sqrt(torch.clamp(torch.trace(g @ gm @ g @ gm).real, min=0.0)))
        J = OPS.current_out_csr(hl, Rl, Rc_abs)
        J_norm_full = None
        Jc = J.tocoo(); jsites = np.unique(np.concatenate([Jc.row // 2, Jc.col // 2]))
        inside = np.nonzero(Rl[:ns] <= Rc_abs)[0]
        lo_w = max(0, ic - max(wins) // 2); hi_w = min(ns, lo_w + max(wins))
        S = np.unique(np.concatenate([inside, jsites, np.arange(lo_w, hi_w)]))
        idx = np.sort(np.concatenate([2 * S, 2 * S + 1]))
        J_norm = float(np.linalg.norm(J.tocsr()[idx][:, idx].toarray(), 2))       # the readouts' ||J_SS||
        Bh = OPS.Banded(hl, dev)
        wsplit32 = G2.split_weight(Rl, Nl, vl, ic, kap, 32)
        wsplit_max = G2.split_weight(Rl, Nl, vl, ic, kap, max(wins))
        dsets = {}
        Dfull = Phi @ Phi.mH - Vn @ Vn.mH; Dfull = 0.5 * (Dfull + Dfull.mH)
        trunc = {"taken": False}
        if a.trunc:
            lam, U = torch.linalg.eigh(Dfull)
            lam_np = lam.numpy(); order = np.argsort(np.abs(lam_np)); cs = np.cumsum(np.abs(lam_np[order]))
            budget = a.trunc_drop_sum if a.trunc_drop_sum > 0 else 0.5 * digit / max(J_norm, Bh.bound, wsplit_max)
            ndrop = int(np.searchsorted(cs, budget, side="right")); sdrop = float(cs[ndrop - 1]) if ndrop else 0.0
            keep = np.sort(order[ndrop:])
            cert = {"L_at_t": sdrop * J_norm, "I_inside": sdrop * Bh.bound, "split_E_flux": sdrop * wsplit_max}
            ok = all(x < digit for x in cert.values())
            trunc = {"taken": bool(ok), "kept_pairs": int(len(keep)), "dropped_pairs": int(ndrop), "dropped_sum": sdrop, "certificate": cert}
            if ok:
                kt = torch.as_tensor(keep); Vr = U[:, kt].contiguous(); Lr = lam[kt].to(CD)
                dsets["trunc"] = (Vr, Lr)
            del U, lam
        log("kappa %d: trunc %s (dropped %s pairs, sum %.3e)" % (kap, trunc.get("taken"), trunc.get("dropped_pairs"), trunc.get("dropped_sum", 0.0)))
        # the filter (version 2's construction)
        sig = a.null_sigma_TH * T_H; lam_b = Bh.bound * (1.0 + 1e-6)
        c1, M1, eps1 = G2.step_coeffs(sig, lam_b, a.null_tol)
        c2, M2, eps2 = G2.step_coeffs(2.0 * sig, lam_b, a.null_tol)
        c2p = np.zeros(M1 + 1, complex); c2p[:min(M1, M2) + 1] = c2[:min(M1, M2) + 1]
        eps2 = eps2 + (float(np.sum(np.abs(c2[M1 + 1:]))) if M2 > M1 else 0.0)
        sup1 = sup_error(c1, sig, lam_b); sup2 = sup_error(c2p, 2.0 * sig, lam_b)
        A = hl.tocsr(); rr_ = max(abs(d) for d, _ in Bh.diags)
        half = 2 * (M1 + 1) * rr_
        lo_j = max(0, n2s - half); hi_j = min(A.shape[0], n2s + half); hs_j = n2s - lo_j
        Aw = A[lo_j:hi_j, lo_j:hi_j]
        Bc = OPS.Banded(Aw, dev, dmax=max(16, rr_)); Bwl = OPS.Banded(G2.walled(Aw, ns - lo_j // 2), dev, dmax=max(16, rr_))
        filt = {"sigma": sig, "sigma_TH": a.null_sigma_TH, "Lambda": lam_b, "M": M1, "M_2sigma": M2, "eps_coef_tail": eps1, "eps_coef_tail_2sigma": eps2,
                "sup_error": sup1, "sup_error_2sigma_truncated_at_M": sup2, "index_range": rr_, "window_rows": [int(lo_j), int(hi_j)],
                "window_half_rows": half, "reach_rows_M_ranges": M1 * rr_, "locality_exact": bool(half >= M1 * rr_ and lo_j > 0 and hi_j < A.shape[0])}
        log("kappa %d: sigma %.3e M %d (2s: %d); eps tail %.2e / %.2e, sup error %.2e / %.2e; window rows %d..%d (short to %d), reach %d of %d" % (
            kap, sig, M1, M2, eps1, eps2, sup1, sup2, lo_j, hi_j, n2s, M1 * rr_, half))
        out_c = {"J_SS_norm": J_norm, "h_bound": Bh.bound, "split_weight_32": wsplit32, "split_weight_max": wsplit_max,
                 "hold_C2_minus_C_fro": hold_pur, "n_Phi": int(Phi.shape[1]), "n_occupied_short": int(Vn.shape[1]), "trunc": trunc, "filter": filt, "sets": {}}
        Dsets = {"full": Dfull}
        if "trunc" in dsets:
            Vr, Lr = dsets["trunc"]; Dsets["trunc"] = None
        for name in (["trunc"] if "trunc" in dsets else []) + ["full"]:
            t_s = time.time()
            if name == "trunc":
                Vr, Lr = dsets["trunc"]
                Dcol = Vr @ (Lr[:, None] * Vr[lo_j:n2s].mH)                       # Delta0[:, win_s]
            else:
                Dcol = Dfull[:, lo_j:n2s].clone()
            G = Dcol.mH @ Dcol; G = 0.5 * (G + G.mH)                                 # (Delta0^2)[win_s, win_s]
            del Dcol
            gl, gU = torch.linalg.eigh(G); del G
            keepg = gl > 0
            R = gU[:, keepg] * torch.sqrt(gl[keepg]).to(CD)[None, :]
            del gU
            Rp = torch.zeros((hi_j - lo_j, R.shape[1]), dtype=CD); Rp[:hs_j] = R
            K1 = torch.zeros_like(Rp); K2 = torch.zeros_like(Rp)
            for j0 in range(0, Rp.shape[1], a.chunk):
                r_ = G2.cheb_filter_pair(Bwl, Bc, Rp[:, j0:j0 + a.chunk], [c1, c2p], lam_b)
                K1[:, j0:j0 + a.chunk] = r_[0]; K2[:, j0:j0 + a.chunk] = r_[1]
            s1 = torch.linalg.svdvals(K1).numpy(); s2 = torch.linalg.svdvals(K2).numpy(); sd = torch.linalg.svdvals(K1 - K2).numpy()
            del K1, K2, Rp, R
            ent = {"rank_Y": int(keepg.sum()), "Delta0_window_rows_fro": float(math.sqrt(max(float(gl.sum()), 0.0)))}
            for lab, s_ in (("sigma", s1), ("2sigma", s2), ("difference", sd)):
                fro = float(np.sqrt(np.sum(s_ ** 2))); tr = float(np.sum(s_))
                b = {"L": 4 * J_norm * tr, "E_flux_out_32": 4 * wsplit32 * tr, "I_inside": 4 * Bh.bound * tr}
                ent[lab] = {"fro": fro, "trace": tr, "op": float(s_.max()) if len(s_) else 0.0, "bound": b,
                            "under_digit": {k: bool(v < digit) for k, v in b.items()}}
            ent["seconds"] = time.time() - t_s
            np.savez_compressed(pre + "_k%d_%s_svals.npz" % (kap, name), sigma=s1, two_sigma=s2, difference=sd, G_eigs=gl.numpy())
            out_c["sets"][name] = ent
            log("kappa %d %s: ||D_s Delta0||_F %.3e, ||.||_1 %.3e -> bound L %.3e, E_flux(32) %.3e (2 sigma: %.3e, %.3e; difference %.3e) against digit %.3e (%.0f s)" % (
                kap, name, ent["sigma"]["fro"], ent["sigma"]["trace"], ent["sigma"]["bound"]["L"], ent["sigma"]["bound"]["E_flux_out_32"],
                ent["2sigma"]["bound"]["L"], ent["2sigma"]["bound"]["E_flux_out_32"], ent["difference"]["bound"]["L"], digit, ent["seconds"]))
        out_c["seconds"] = time.time() - t_c
        res["channels"][str(kap)] = out_c
        json.dump(res, open(pre + "_posbound.json", "w"), indent=1)
        del Dfull, Vn, Phi, dsets
    # the summary
    L_ = ["# %s: the graft's positive bound on its join defect (engine spec 54.2 (2), (3))" % a.tag, "",
          "Hold %s frozen at T/r_s %.4f; digit 1e-3 L_H = %.4e; bound = 4 ||O|| ||D_sigma Delta0||_1 per detector (O: J_SS for L, the 32-site split for "
          "E_flux_out(32), h for I); D_sigma = p(H_walled) - p(H_connected) on the window (exactly local). Computed on the box CPU from the stored hold "
          "and the frozen record, no propagation. 'trunc' is the chains' Delta0 (version 2's certified truncation); 'full' the untruncated beside." % (
              a.init_state, tf / rs, digit), ""]
    verdict = True
    for k, c in res["channels"].items():
        f = c["filter"]
        L_.append("## kappa %s" % k)
        L_.append("- hold ||C_S^2 - C_S||_F %.3e (Phi %d columns); ||J_SS|| %.4e, split weight(32) %.4e, ||h|| bound %.4e" % (
            c["hold_C2_minus_C_fro"], c["n_Phi"], c["J_SS_norm"], c["split_weight_32"], c["h_bound"]))
        L_.append("- filter: sigma %.4e (%.0f T_H), M %d; coefficient tail %.2e (2 sigma %.2e); sup-norm error on 2,000,001 points %.2e (2 sigma truncated at M: %.2e); "
                  "window rows %d..%d, reach %d rows within %d: locality exact %s" % (f["sigma"], f["sigma_TH"], f["M"], f["eps_coef_tail"], f["eps_coef_tail_2sigma"],
                                                                                    f["sup_error"], f["sup_error_2sigma_truncated_at_M"], f["window_rows"][0], f["window_rows"][1],
                                                                                    f["reach_rows_M_ranges"], f["window_half_rows"], f["locality_exact"]))
        tr = c["trunc"]
        L_.append("- truncation: taken %s, dropped %s pairs, dropped sum %.3e" % (tr.get("taken"), tr.get("dropped_pairs"), tr.get("dropped_sum", 0.0)))
        for name, e in c["sets"].items():
            for lab in ("sigma", "2sigma", "difference"):
                b = e[lab]["bound"]
                L_.append("- %s, %s: ||D Delta0||_F %.4e, ||D Delta0||_1 %.4e; bound L %.4e, E_flux_out(32) %.4e, I %.4e; under the digit: L %s, E_flux %s" % (
                    name, lab, e[lab]["fro"], e[lab]["trace"], b["L"], b["E_flux_out_32"], b["I_inside"], e[lab]["under_digit"]["L"], e[lab]["under_digit"]["E_flux_out_32"]))
            if name == ("trunc" if "trunc" in c["sets"] else "full"):
                for lab in ("sigma", "2sigma"):
                    verdict = verdict and e[lab]["under_digit"]["L"] and e[lab]["under_digit"]["E_flux_out_32"]
        L_.append("")
    L_.append("VERDICT (the chains' Delta0, L and E_flux_out(32), at sigma and 2 sigma, every channel): %s" % (
        "BOUNDED: join defect bounded below the digit" if verdict else "NOT BOUNDED: measured, not bounded"))
    res["verdict_bounded"] = bool(verdict)
    json.dump(res, open(pre + "_posbound.json", "w"), indent=1)
    open(pre + "_posbound.md", "w").write("\n".join(L_) + "\n")
    log("VERDICT %s; written %s_posbound.json and .md" % ("BOUNDED" if verdict else "NOT BOUNDED", pre))


if __name__ == "__main__":
    main()
