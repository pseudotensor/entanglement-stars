#!/usr/bin/env python
"""s23_spectral_v1.py (busi, inbox #420, NOTCOMP item 2.3: the drain's end state, the accumulated output entropy and the held object's Page curve;
ReviewA_notcomp_v1 sec. 2.3, with ReviewB_notcomp_v1 sec. 4.9's conditions: bound modes, resonance widths, no dephasing shortcut).

Spectral, no time stepping. The held operator is the production's own (page_history_v6s.History + page_interior_v6s.hold_extend +
ChannelOps.h_csr + page_w1_sector_v1d.sector_plus_csr, md5 611d190f / 15e08e28, read-only), kappa 1, + sector, B2's record 800 (T/r_s 100)
held. The infinite exterior is represented by a complex absorbing potential (CAP) W on sites j0 ... L-1 of the chain's leading L sites:
H_cap = h_+[:L, :L] - i W, W_j = eta ((j - j0)/(L - j0))^2. Its eigen-decomposition H_cap = R diag(lam) R^-1 gives the interior propagator
U_IJ(t) = R_I e^{-i lam t} (R^-1)_J for any t (no stepping). For a state C at the hold clock t_s with deviation D_JJ = C_JJ - P_JJ on the
region J = sites 0 ... nJ-1 (P_JJ the held infinite-chain vacuum's block, the solve lane's contour block m8_PII_3074_v1.npy, read-only),
     C_II(t_s + t) = P_II + U_IJ(t) D_JJ U_IJ(t)^dag,
exact for the infinite chain when (i) no content outside J returns into I (outgoing radiation beyond J) and (ii) the CAP is reflectionless at
the energies that matter; both are tested (CAP strength and length scans; t = 0 identity; the control state at 100 r_s propagated to 500 r_s
against the stored hold's readouts and the solve lane's back-propagated shell occupations).
The excess at a cut is 2 [S(C_II) - S(P_II)] per chain (the lean readout's interior side, by purity).
Bound levels: CAP eigenvalues with |Im lam| below a tolerance that are stable under the CAP scan, and the full held spectrum's levels outside the
exterior band [-1, 1] (banded eigensolver), classified by their weight on the chain's last sites; the retained entropy is the bound block's.

Usage (box): CUDA_VISIBLE_DEVICES=2 python s23_spectral_v1.py --L 12288 --j0 4096 --eta 0.01 --out DIR [--state istate|ctrl] ...
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import torch

T0W = time.time()


def log(msg):
    print("[%8.1f s] %s" % (time.time() - T0W, msg), flush=True)


def md5_file(p, chunk=1 << 24):
    m = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def ent_ev(ev):
    x = ev.clamp(0.0, 1.0)
    return float(-(torch.special.xlogy(x, x) + torch.special.xlogy(1.0 - x, 1.0 - x)).sum())


def S_block(M):
    ev = torch.linalg.eigvalsh(0.5 * (M + M.mH))
    return ent_ev(ev), float(ev.min()), float(ev.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="<internal>/code_d13")
    ap.add_argument("--code-v6", default="<internal>/code_v6")
    ap.add_argument("--record", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json")
    ap.add_argument("--fields", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz")
    ap.add_argument("--istate", default="<internal>/v6s_B2hold_k1_istate.pt")
    ap.add_argument("--ctrl100", default="<internal>/v6o_ctrl_B2_k1_state.pt")
    ap.add_argument("--pii", default="<internal>/m8_PII_3074_v1.npy")
    ap.add_argument("--levels", default="<internal>/m8_levels_v1.npz")
    ap.add_argument("--states", default="istate,ctrl", help="comma list: istate (the hold's stored state at 500 r_s), ctrl (the interp-none control at 100 r_s)")
    ap.add_argument("--L", type=int, default=12288)
    ap.add_argument("--j0", type=int, default=4096)
    ap.add_argument("--eta", type=float, default=0.01)
    ap.add_argument("--cap-power", type=float, default=2.0)
    ap.add_argument("--times-rs", default="0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,600,800,1000,1200,1500,1800,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12000,15000,20000,25000,30000,40000,50000,100000,1000000")
    ap.add_argument("--cuts", default="128,1024,1536,1920,3072", help="interior site counts (2, 16, 24, 30, 48 r_s)")
    ap.add_argument("--bound-tol", type=float, default=1e-9, help="|Im lam| below this (per a) counts as a bound candidate")
    ap.add_argument("--full-bound", action="store_true", help="also classify the full held spectrum's levels outside [-1, 1] (banded, eigenvectors)")
    ap.add_argument("--keep-eig", action="store_true", help="cache the CAP eigen-decomposition on disk (resume after a crash)")
    ap.add_argument("--device", default="cpu", help="device of the propagated products (cpu default: the box GPU is shared)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code_v6); sys.path.insert(0, a.code)
    import page_history_v6s as PH
    import page_interior_v6s as IR
    import page_w1_sector_v1d as W1S
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or "s23_L%d_j%d_eta%g" % (a.L, a.j0, a.eta)
    CD = torch.complex128
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = {"tag": tag, "args": vars(a), "md5": {}}
    for p in (os.path.join(a.code, "page_history_v6s.py"), os.path.join(a.code, "page_interior_v6s.py"), os.path.join(a.code, "page_w1_sector_v1d.py"), a.record, a.fields, a.pii):
        out["md5"][p] = md5_file(p); log("md5 %s %s" % (out["md5"][p], p))

    # ---------------------------------------------------------------- the held operator (record 800 = T/r_s 100)
    H = PH.History(a.record, a.fields, 32768, None, "applied")
    rs = float(H.rs_nom)
    hold = IR.hold_extend(H, 100.0 * rs, 500.0 * rs, H.tau, log)
    ops = PH.ChannelOps(1, H.R, torch.device("cpu"))
    kh = int(hold["k_hold"])
    N, v = H.fields(kh)
    hp = W1S.sector_plus_csr(ops.h_csr(N, v)).tocsr()
    out["hp_md5"] = IR._hp_md5(hp); log("held h_+ md5 %s (solve's M8: 0ddcdce1...), n %d, nnz %d; r_s %.1f a" % (out["hp_md5"], hp.shape[0], hp.nnz, rs))
    n = hp.shape[0]
    Rarr = np.asarray(H.R, float)
    shell = np.where(N < np.abs(v))[0]
    log("trapped shell N < |v|: sites %d ... %d (R %g ... %g)" % (shell.min(), shell.max(), Rarr[shell.min()], Rarr[shell.max()]))

    # ---------------------------------------------------------------- the vacuum block
    P = torch.as_tensor(np.load(a.pii), dtype=CD)
    nJ = P.shape[0]
    cuts = [int(x) for x in a.cuts.split(",")]

    def load_D(which):
        if which == "istate":
            st = torch.load(a.istate, map_location="cpu", weights_only=False)
            Phi = st["channels"]["1"]["Phi"]; t_s = 500.0
            log("state istate: the hold's stored state, k %s, Phi %s (T/r_s 500)" % (st["k"], tuple(Phi.shape)))
        else:
            st = torch.load(a.ctrl100, map_location="cpu", weights_only=False)
            Phi = st["channels"]["1"]["Phi"]; t_s = 100.0
            log("state ctrl: the interp-none control at 100 r_s, Phi %s" % (tuple(Phi.shape),))
        del st
        PhJ = Phi[:nJ].to(CD)
        C = PhJ @ PhJ.mH
        del Phi, PhJ
        exc0 = []
        for m in cuts:
            sC = S_block(C[:m, :m]); sP = S_block(P[:m, :m])
            exc0.append(2.0 * (sC[0] - sP[0]))
            log("t = 0 (T/r_s %.0f): cut %d sites: S_I(C) %.9f [%.2e, 1%+.2e], S_I(P) %.9f; excess per chain %.9f" % (t_s, m, sC[0], sC[1], sC[2] - 1, sP[0], exc0[-1]))
        return C - P, t_s, exc0

    # ---------------------------------------------------------------- the CAP operator and its eigen-decomposition
    L, j0 = a.L, a.j0
    hL = torch.as_tensor(hp[:L, :L].toarray(), dtype=CD)
    x = np.clip((np.arange(L) - j0) / float(L - j0), 0.0, None)
    W = a.eta * x ** a.cap_power
    Hc = hL - 1j * torch.as_tensor(np.diag(W), dtype=CD)
    del hL
    t1 = time.time()
    cache = os.path.join(a.out, "eigcache_L%d_j%d_eta%g_p%g.pt" % (L, j0, a.eta, a.cap_power))
    if os.path.exists(cache):
        ce = torch.load(cache, weights_only=False); lam, R, Rinv = ce["lam"], ce["R"], ce["Rinv"]
        log("eig of H_cap loaded from %s" % cache)
    else:
        lam, R = torch.linalg.eig(Hc)
        log("eig of H_cap (L %d, j0 %d, eta %g, power %g): %.1f s; Im lam in [%.3e, %.3e]" % (L, j0, a.eta, a.cap_power, time.time() - t1, float(lam.imag.min()), float(lam.imag.max())))
        Rinv = torch.linalg.inv(R)
        if a.keep_eig:
            torch.save({"lam": lam, "R": R, "Rinv": Rinv}, cache); log("eig cached to %s (delete after the chain)" % cache)
    idres = float((R @ Rinv - torch.eye(L, dtype=CD)).abs().max())
    resid = float(((Hc @ R) - R * lam[None, :]).norm(dim=0).max())
    log("max |R R^-1 - I| %.2e; max eigen-residual %.2e" % (idres, resid))
    out["cap"] = {"L": L, "j0": j0, "eta": a.eta, "power": a.cap_power, "eig_s": time.time() - t1, "RRinv_err": idres, "eig_resid": resid}
    del Hc

    # ---------------------------------------------------------------- bound candidates and resonances near zero
    im = lam.imag.numpy(); re = lam.real.numpy()
    wts = R.abs().pow(2); wts = wts / wts.sum(dim=0, keepdim=True)
    w_in = wts[:j0].sum(dim=0).numpy(); w128 = wts[:128].sum(dim=0).numpy(); wsh = wts[shell].sum(dim=0).numpy()
    bnd = np.where(np.abs(im) < a.bound_tol)[0]
    log("bound candidates |Im lam| < %.1e: %d; real parts %s" % (a.bound_tol, len(bnd), np.array2string(np.sort(re[bnd])[:20], precision=5)))
    near = np.where((np.abs(re) < 0.02) & (wsh > 0.05))[0]
    near = near[np.argsort(re[near])]
    res_rows = []
    for i in near:
        g_rs = -2.0 * im[i] * rs
        res_rows.append({"re": float(re[i]), "im": float(im[i]), "Gamma_per_rs": float(g_rs), "w_shell": float(wsh[i]), "w_128": float(w128[i]), "w_noCAP": float(w_in[i])})
        log("  resonance e %+.6f  Im %.3e  Gamma %.3e per r_s (lifetime %.0f r_s)  w(shell) %.3f  w(<128) %.3f  w(<j0) %.3f" % (re[i], im[i], g_rs, 1.0 / max(g_rs, 1e-30), wsh[i], w128[i], w_in[i]))
    out["resonances"] = res_rows
    out["bound"] = [{"re": float(re[i]), "im": float(im[i]), "w_noCAP": float(w_in[i]), "w_shell": float(wsh[i])} for i in bnd]

    # ---------------------------------------------------------------- the propagated interior blocks
    # U_JJ(t) = R_J e^{-i lam t} Z, Z = (R^-1)_{., J}; C_JJ(t) = P + U_JJ D U_JJ^dag; the cuts' blocks are its leading m x m blocks.
    RJ = R[:nJ].to(dev); Zd = Rinv[:, :nJ].to(dev); Pd = P.to(dev); lamd = lam.to(dev)
    bm = torch.as_tensor(np.abs(im) < a.bound_tol)
    Vlev = None
    if os.path.exists(a.levels):
        z = np.load(a.levels)
        Vfull = z["V"]
        Vlev = torch.as_tensor(Vfull[:nJ], dtype=CD).to(dev)
        out["levels_e"] = [float(e) for e in z["eV"]]
        out["levels_weight_beyond_J"] = [float(np.sum(np.abs(Vfull[nJ:, i]) ** 2)) for i in range(Vfull.shape[1])]
        log("shell levels %s; their weight beyond J (truncated here) %s" % (np.array2string(z["eV"], precision=6), out["levels_weight_beyond_J"]))
    sP = {m: S_block(Pd[:m, :m])[0] for m in cuts}

    def readout(U, D_):
        CJ = Pd + U @ D_ @ U.mH
        row = {"excess": [], "Cmin": [], "Cmax": []}
        for m in cuts:
            sC = S_block(CJ[:m, :m])
            row["excess"].append(2.0 * (sC[0] - sP[m])); row["Cmin"].append(sC[1]); row["Cmax"].append(sC[2])
        if Vlev is not None:
            M = Vlev.mH @ CJ @ Vlev
            row["occ"] = [float(M[i, i].real) for i in range(M.shape[0])]
            row["block_S_per_chain"] = 2.0 * S_block(M)[0]
        return row

    out["states"] = {}
    for which in [x for x in a.states.split(",") if x]:
        D, t_s, exc0 = load_D(which)
        Dd = D.to(dev)
        rows = []
        for trs in [float(s) for s in a.times_rs.split(",")]:
            t1 = time.time()
            ph = torch.exp(-1j * lamd * (trs * rs))
            U = RJ @ (ph[:, None] * Zd)
            row = readout(U, Dd); row.update({"t_after_rs": trs, "T_rs": t_s + trs})
            rows.append(row)
            log("T/r_s %8.0f (+%7.0f): excess per chain %s  occ %s  [C eig %.2e .. 1%+.2e] (%.1f s)" % (
                t_s + trs, trs, " ".join("%.6f" % e for e in row["excess"]), " ".join("%.5f" % o for o in row.get("occ", [])), min(row["Cmin"]), max(row["Cmax"]) - 1, time.time() - t1))
            del U
        # the infinite-time limit: only the bound terms survive (their phases kept at t = 1e7 r_s, and dephased: the time average)
        bmd = bm.to(dev)
        U_inf = RJ[:, bmd] @ (torch.exp(-1j * lamd[bmd] * (1e7 * rs))[:, None] * Zd[bmd])
        lim = {"bound_block_t1e7": readout(U_inf, Dd)["excess"]}
        X = Zd[bmd] @ Dd @ Zd[bmd].mH
        CJd = Pd + RJ[:, bmd] @ torch.diag(torch.diagonal(X)) @ RJ[:, bmd].mH
        lim["dephased"] = [2.0 * (S_block(CJd[:m, :m])[0] - sP[m]) for m in cuts]
        dvec = torch.real(torch.diagonal(X))
        lim["n_bound"] = int(bmd.sum()); lim["bound_dev_sum_abs"] = float(dvec.abs().sum()); lim["bound_dev_max_abs"] = float(dvec.abs().max()) if dvec.numel() else 0.0
        log("state %s: the infinite-time limit (the %d bound terms only): excess per chain %s; dephased %s; sum |bound deviation| %.3e, max %.3e" % (
            which, lim["n_bound"], " ".join("%.3e" % e for e in lim["bound_block_t1e7"]), " ".join("%.3e" % e for e in lim["dephased"]), lim["bound_dev_sum_abs"], lim["bound_dev_max_abs"]))
        out["states"][which] = {"t_s": t_s, "excess_t0": exc0, "rows": rows, "limit": lim}
        json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
        del D, Dd, X, CJd, U_inf
    # ---------------------------------------------------------------- the full held spectrum outside the exterior band
    if a.full_bound:
        ab, _u = IR._herm_band_upper(hp)
        t1 = time.time()
        for side, rng in (("above", (1.0, 1e3)), ("below", (-1e3, -1.0))):
            e_b, U_b = sla.eig_banded(ab, lower=False, select="v", select_range=rng, check_finite=False)
            wl = (np.abs(U_b[-4096:]) ** 2).sum(axis=0) if U_b.size else np.zeros(0)
            wJ = (np.abs(U_b[:nJ]) ** 2).sum(axis=0) if U_b.size else np.zeros(0)
            loc = wl < 1e-6
            out["outside_band_" + side] = {"count": int(len(e_b)), "localized_last4096_lt_1e-6": int(loc.sum()), "e_range": [float(e_b.min()), float(e_b.max())] if len(e_b) else None,
                                            "weight_in_J_max": float(wJ.max()) if len(wJ) else None}
            log("levels %s the band: %d (%.1f s), localized (weight on the last 4096 sites < 1e-6): %d, e in %s, max weight in J %.3e" % (
                side, len(e_b), time.time() - t1, int(loc.sum()), out["outside_band_" + side]["e_range"], out["outside_band_" + side]["weight_in_J_max"]))
            np.save(os.path.join(a.out, tag + "_outside_%s_e.npy" % side), e_b)
            np.save(os.path.join(a.out, tag + "_outside_%s_wlast.npy" % side), wl)
    json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    log("wrote %s" % os.path.join(a.out, tag + ".json"))


if __name__ == "__main__":
    main()
