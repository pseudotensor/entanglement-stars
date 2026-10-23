#!/usr/bin/env python
"""m8_b2hold_levels_v1.py (solve lane, the second route of the B2 hold-drain reading's M8 and its section 8 checks A and B).

Independent of the checking lane's own_b2_*.py (not read). The held operator h_+ is built by the production's own code
(page_history_v6s.History + page_interior_v6s.hold_extend + ChannelOps.h_csr + page_w1_sector_v1d.sector_plus_csr, md5 611d190f / 15e08e28);
the stored final lean state Phi_+ (record 4000, T/r_s 500) is read from the drain's istate.pt.

Stage 1 (this script), all on the host with torch (8 threads):
  (1) h_+ at T0 and at the held record 800; md5 against the init cache's; the full spectrum by eigvals_banded (E_ref, min|e|);
  (2) Phi_+ from istate.pt: E_hC = 2 Tr[Phi^dag h Phi] - Tr h, E_inv, the site energy density of C over the whole chain;
  (3) the leading 4096-site block of h_+ by torch eigh; its near-zero levels; the shell (N < |v|) weights, the weights inside 128 a, the
      occupations n_a = ||Phi_blk^dag u_a||^2, the residuals against the full h_+; selection: shell weight > 0.3;
  (4) the vacuum's interior block P_II (sites 0 ... 3073) by the imaginary-axis contour of sign(h_+) with the Schur complement (the exterior's
      2 x 2 self-energy by 2-column banded solves, the interior resolvent by a dense torch inverse), the production's grid (hs 0.3, U 37);
      sign(h_+) V for the selected levels by full-chain banded solves on the same grid (M0 = V^dag P_- V exactly);
  (5) S_I(C), S_I(P) at the five cuts, the excess; the end state P_II + V_I D V_I^dag (D = V^dag (C - P_-) V) and its dephased form;
  (6) check A at 500: the energy inside each cut, 2 Re sum_{i < m} [h (C - P_-)]_ii, and the whole-chain sums;
  (7) check B on the stored 8192-site interp-none control state at 100 r_s (the only kappa 1 state at 100 r_s on the box): the zero-padded
      levels' occupations.
Outputs under --out (npz + log). Per chain = two sectors = 2 x the + sector value throughout.
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


def ent(ev):
    x = ev.clamp(0.0, 1.0)
    return float(-(torch.special.xlogy(x, x) + torch.special.xlogy(1.0 - x, 1.0 - x)).sum())


def S_block(M):
    ev = torch.linalg.eigvalsh(0.5 * (M + M.mH))
    return ent(ev), float(ev.min()), float(ev.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="<internal>/code_d13")
    ap.add_argument("--code-v6", default="<internal>/code_v6")
    ap.add_argument("--record", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json")
    ap.add_argument("--fields", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz")
    ap.add_argument("--istate", default="<internal>/v6s_B2hold_k1_istate.pt")
    ap.add_argument("--summary", default="<internal>/v6s_B2hold_k1_summary.json")
    ap.add_argument("--init-cache", default="<internal>/B2_n32768_k1_T0_Phiplus_mrrr_lowdin.pt")
    ap.add_argument("--ctrl100", default="<internal>/v6o_ctrl_B2_k1_state.pt")
    ap.add_argument("--out", default="<internal>/page_b2hold_m8")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nblk", type=int, default=4096)
    ap.add_argument("--win", type=float, default=0.02)
    ap.add_argument("--shell-min", type=float, default=0.3)
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code_v6); sys.path.insert(0, a.code)
    import page_history_v6s as PH
    import page_interior_v6s as IR
    import page_w1_sector_v1d as W1S
    import page_ops_v6 as OPS
    os.makedirs(a.out, exist_ok=True)
    CD = torch.complex128
    for p in (os.path.join(a.code, "page_history_v6s.py"), os.path.join(a.code, "page_interior_v6s.py"), a.record, a.fields):
        log("md5 %s %s" % (md5_file(p), p))

    # ---------------------------------------------------------------- (1) the operator
    H = PH.History(a.record, a.fields, 32768, None, "applied")
    rs = float(H.rs_nom)
    hold = IR.hold_extend(H, 100.0 * rs, 500.0 * rs, H.tau, log)
    ops = PH.ChannelOps(1, H.R, torch.device("cpu"))
    N0, v0 = H.fields(0)
    hp0 = W1S.sector_plus_csr(ops.h_csr(N0, v0))
    kh = int(hold["k_hold"])
    N, v = H.fields(kh)
    hp = W1S.sector_plus_csr(ops.h_csr(N, v)).tocsr()
    N4000, v4000 = H.fields(len(H.T) - 1)
    log("held record %d (T %.1f = %.3f r_s); record 4000's fields equal the held ones: %s" % (kh, H.T[kh], H.T[kh] / rs,
        bool(np.array_equal(N4000, N) and np.array_equal(v4000, v))))
    md5_0 = IR._hp_md5(hp0); md5_h = IR._hp_md5(hp)
    ic = torch.load(a.init_cache, map_location="cpu", weights_only=False)
    log("h_+ md5 at T0 %s (init cache %s: %s); held h_+ md5 %s; n %d, nnz %d" % (md5_0, ic.get("md5_h_plus"), md5_0 == ic.get("md5_h_plus"), md5_h, hp.shape[0], hp.nnz))
    del ic
    n = hp.shape[0]
    ab, _u = IR._herm_band_upper(hp)
    e_all = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
    E_ref = float(e_all[e_all < 0].sum() - e_all[e_all > 0].sum())
    log("held spectrum: %d negative, min|e| %.9e, E_ref %.10f, max e %.6f, %d above 1, %d within 0.02 of zero" % (
        int((e_all < 0).sum()), np.min(np.abs(e_all)), E_ref, e_all.max(), int((e_all > 1).sum()), int((np.abs(e_all) < 0.02).sum())))
    Rarr = np.asarray(H.R, float)
    shell = np.where(N < np.abs(v))[0]
    log("trapped shell N < |v|: sites j %d ... %d (R %g ... %g), %d sites; min(N - |v|) %.5f at R %g" % (
        shell.min(), shell.max(), Rarr[shell.min()], Rarr[shell.max()], len(shell), np.min(N - np.abs(v)), Rarr[np.argmin(N - np.abs(v))]))
    trh = W1S.trace_diag(hp)

    # ---------------------------------------------------------------- (2) the state
    if not a.skip_md5:
        log("istate md5 %s" % md5_file(a.istate))
    st = torch.load(a.istate, map_location="cpu", weights_only=False)
    ch = st["channels"]["1"]
    Phi = ch["Phi"]
    log("istate: k %s, n_ck %s, rows %d, Phi %s %s, n_substeps %s, cheb_K_max %s, imp_last %s at %s" % (
        st["k"], st["n_ck"], len(st["rows"]), tuple(Phi.shape), Phi.dtype, ch["n_substeps"], ch["cheb_K_max"], ch["imp_last"], ch["imp_last_t"]))
    last = st["rows"][-1]["per_channel"]["1"]
    rows_ = st["rows"]
    del st
    B = IR.BandedLean(hp, torch.device("cpu"))
    eC_site = np.zeros(n)
    trhC = 0.0
    ch_ = 1024
    buf = torch.empty((n, ch_), dtype=CD)
    for j0 in range(0, Phi.shape[1], ch_):
        X = Phi[:, j0:j0 + ch_]
        Y = B.apply_into(X, buf[:, :X.shape[1]])
        d = (X.conj() * Y).real.sum(dim=1)
        eC_site += d.numpy()
    del buf
    trhC = float(eC_site.sum())
    E_hC = 2.0 * trhC - trh
    log("Tr[h_+ C_+] %.13f (site sum), Tr h_+ %.13f, E_hC %.13f (stored %.13f), E_inv %.10e (stored %.10e)" % (
        trhC, trh, E_hC, last["E_hC"], E_hC - E_ref, last["E_inv"]))

    # ---------------------------------------------------------------- (3) the block levels
    nb = a.nblk
    A = torch.as_tensor(hp[:nb, :nb].toarray(), dtype=CD)
    t1 = time.time()
    eb, Ub = torch.linalg.eigh(A)
    log("block eigh %d: %.1f s; %d levels within %.3f of zero" % (nb, time.time() - t1, int((eb.abs() < a.win).sum()), a.win))
    sel_win = torch.where(eb.abs() < a.win)[0]
    Pb = Phi[:nb]
    Uw = Ub[:, sel_win]
    occ = (Pb.mH @ Uw).abs().pow(2).sum(dim=0).numpy()
    ew = eb[sel_win].numpy()
    wt = Uw.abs().pow(2).numpy()
    w128 = wt[:128].sum(0); wsh = wt[shell[shell < nb]].sum(0); w1024 = wt[1024:].sum(0); wlast = wt[nb - 256:].sum(0)
    dev_theta = occ - (ew < 0)
    # residual against the full operator
    Upad = torch.zeros((n, len(sel_win)), dtype=CD); Upad[:nb] = Uw
    HU = torch.as_tensor((hp @ Upad.numpy()), dtype=CD)
    res = (HU - Upad * torch.as_tensor(ew, dtype=CD)[None, :]).norm(dim=0).numpy()
    rows = []
    for i in range(len(sel_win)):
        rows.append((float(ew[i]), float(occ[i]), float(dev_theta[i]), float(w128[i]), float(wsh[i]), float(w1024[i]), float(wlast[i]), float(res[i])))
    log("levels with |e| < %.3f: %d; those with shell weight > %.2f:" % (a.win, len(rows), a.shell_min))
    keep = [i for i in range(len(rows)) if rows[i][4] > a.shell_min]
    for i in keep:
        r = rows[i]
        log("  SHELL e %+.6f  n %.5f  dev %+.5f  w(<128) %.3f  w(shell) %.3f  w(>=1024) %.3f  w(last 256) %.2e  resid %.2e" % r)
    log("other window levels with |dev| > 0.01 (standing-wave candidates):")
    for i in range(len(rows)):
        if i not in keep and abs(rows[i][2]) > 0.01:
            log("  other e %+.6f  n %.5f  dev %+.5f  w(<128) %.3f  w(shell) %.3f  w(>=1024) %.3f  w(last 256) %.2e  resid %.2e" % rows[i])
    V = Upad[:, keep].clone()                                                   # n x nk, zero-padded block eigenvectors
    eV = ew[keep]
    nk = V.shape[1]
    del A, Ub, Uw, HU, Upad

    # ---------------------------------------------------------------- (4) the vacuum by the contour
    nI = 3074
    w = OPS.gershgorin(hp)
    delta = float(np.min(np.abs(e_all)))
    hs, U = 0.3, 37.0
    s_lo = math.log(delta) - U; s_hi = math.log(w) + U
    m = int(math.ceil((s_hi - s_lo) / hs)) + 1
    s_nodes = s_lo + hs * np.arange(m)
    l = u = 2
    Ad = torch.as_tensor(hp[:nI, :nI].toarray(), dtype=CD)
    D = hp[nI:, nI:]
    Bc = hp[nI - 2:nI, nI:nI + 2].toarray()
    Rr = np.zeros((n - nI, 2), dtype=complex); Rr[:2, :] = Bc.conj().T
    abD0 = IR.band_general(D, l, u); abF0 = IR.band_general(hp, l, u)
    Vn = V.numpy()
    S = torch.zeros((nI, nI), dtype=CD); Se = torch.zeros((nI, nI), dtype=CD)
    SV = np.zeros((n, nk), dtype=complex); SVe = np.zeros((n, nk), dtype=complex)
    I_n = torch.eye(nI, dtype=CD)
    t1 = time.time()
    for j in range(m):
        tj = math.exp(s_nodes[j]); z = 1j * tj
        abD = abD0.copy(); abD[u, :] -= z
        Y = sla.solve_banded((l, u), abD, Rr, check_finite=False)
        Sig = Bc @ Y[:2, :]
        Mz = Ad - z * I_n
        Mz[nI - 2:, nI - 2:] -= torch.as_tensor(Sig, dtype=CD)
        G = torch.linalg.inv(Mz)
        Gh = 0.5 * (G + G.mH) * tj
        S += Gh
        abF = abF0.copy(); abF[u, :] -= z
        Xf = sla.solve_banded((l, u), abF, Vn, check_finite=False)              # (h - z)^-1 V
        abF2 = abF0.copy(); abF2[u, :] += z
        Xf2 = sla.solve_banded((l, u), abF2, Vn, check_finite=False)            # (h + z)^-1 V = [(h - z)^-1]^dag V
        T = 0.5 * (Xf + Xf2) * tj
        SV += T
        if j % 2 == 0:
            Se += Gh; SVe += T
        if j == 2:
            log("contour: %d nodes, %.2f s per node" % (m, (time.time() - t1) / 3))
    sgn = (2.0 / math.pi) * hs * S; sgn = 0.5 * (sgn + sgn.mH)
    sgn_e = (2.0 / math.pi) * 2 * hs * Se; sgn_e = 0.5 * (sgn_e + sgn_e.mH)
    PII = 0.5 * (I_n - sgn)
    sgnV = (2.0 / math.pi) * hs * SV; sgnV_e = (2.0 / math.pi) * 2 * hs * SVe
    PV = 0.5 * (Vn - sgnV)
    log("contour done (%.1f s): nodes %d, delta %.6e, w %.6f; quad check 2hs: block %.2e, P V %.2e" % (
        time.time() - t1, m, delta, w, float((sgn - sgn_e).abs().max()), float(np.abs(sgnV - sgnV_e).max())))
    M0 = torch.as_tensor(Vn.conj().T @ PV, dtype=CD)                            # V^dag P_- V
    log("V^dag P_- V diagonal %s; largest off-diagonal %.2e" % (np.round(M0.diagonal().real.numpy(), 6).tolist(),
        float((M0 - torch.diag(M0.diagonal())).abs().max())))

    # ---------------------------------------------------------------- (5) the entropies and the end state
    PhiI = Phi[:nI]
    Gc = PhiI @ PhiI.mH                                                         # C_ab = sum_k Phi_ak conj(Phi_bk)
    PV_t = Phi.mH @ V                                                           # 16384 x nk
    MC = PV_t.mH @ PV_t                                                         # V^dag C V
    Dm = MC - M0
    log("V^dag C V diagonal (occupations) %s; |off-diagonal| Frobenius %.4e" % (np.round(MC.diagonal().real.numpy(), 6).tolist(),
        float(torch.linalg.norm(MC - torch.diag(MC.diagonal())))))
    log("D = V^dag (C - P_-) V diagonal %s; trace %.5f; off-diagonal Frobenius %.4e" % (np.round(Dm.diagonal().real.numpy(), 6).tolist(),
        float(Dm.diagonal().real.sum()), float(torch.linalg.norm(Dm - torch.diag(Dm.diagonal())))))
    eRay = (V.mH @ torch.as_tensor(hp @ Vn, dtype=CD)).diagonal().real.numpy()
    E_lev = 2.0 * float((eRay * Dm.diagonal().real.numpy()).sum())
    E_lev_full = 2.0 * float((torch.as_tensor(Vn.conj().T @ (hp @ Vn), dtype=CD) * Dm.T).sum().real)
    log("the deviation's energy in the kept levels per chain: 2 sum e_a D_aa %.6e, 2 Tr[(V^dag h V) D] %.6e; E_inv per chain %.6e (fraction %.4f)" % (
        E_lev, E_lev_full, E_hC - E_ref, E_lev / (E_hC - E_ref)))
    VI = V[:nI]
    cuts = [("2", 128), ("16", 1024), ("24", 1536), ("30", 1920), ("48", 3072)]
    stored = last["interior"]["cuts"]
    out_cuts = {}
    for lab, mm in cuts:
        SC, cmin, cmax = S_block(Gc[:mm, :mm]); SP, pmin, pmax = S_block(PII[:mm, :mm])
        Vm = VI[:mm]
        Bend = PII[:mm, :mm] + Vm @ Dm @ Vm.mH
        Bdeph = PII[:mm, :mm] + Vm @ torch.diag(Dm.diagonal()) @ Vm.mH
        Dth = MC - torch.diag(torch.as_tensor((eV < 0).astype(float), dtype=CD))
        Bth = PII[:mm, :mm] + Vm @ Dth @ Vm.mH
        Se_, emin, emax = S_block(Bend); Sd_, dmin, dmax = S_block(Bdeph); Sth, _, _ = S_block(Bth)
        exc = 2 * (SC - SP)
        o = {"S_I_C": 2 * SC, "S_I_P": 2 * SP, "excess": exc, "stored_S_I_C": stored[lab]["S_I_C"], "stored_S_I_P": stored[lab]["S_I_P"],
             "stored_excess": stored[lab]["excess"], "end_excess": 2 * (Se_ - SP), "end_excess_dephased": 2 * (Sd_ - SP),
             "end_excess_theta_ref": 2 * (Sth - SP), "end_eig_range": (emin, emax), "C_eig_range": (cmin, cmax), "P_eig_range": (pmin, pmax)}
        out_cuts[lab] = o
        log("cut %s r_s (%d sites): S_I(C) %.9f (stored %.9f, diff %.1e); S_I(P) %.9f (stored %.9f, diff %.1e); excess %.9f (stored %.9f); "
            "end state %.4f, dephased %.4f (theta reference %.4f); end - measured %.4f; end-block eigenvalues %.1e .. 1 + %.1e" % (
                lab, mm, 2 * SC, stored[lab]["S_I_C"], 2 * SC - stored[lab]["S_I_C"], 2 * SP, stored[lab]["S_I_P"], 2 * SP - stored[lab]["S_I_P"],
                exc, stored[lab]["excess"], o["end_excess"], o["end_excess_dephased"], o["end_excess_theta_ref"], o["end_excess"] - exc, emin, emax - 1))

    # ---------------------------------------------------------------- (6) check A: the energy inside the cuts
    hI = torch.as_tensor(hp[:nI, :nI].toarray(), dtype=CD)
    Dev = Gc - PII
    eC_I = (hI * Gc.T).sum(dim=1).real                                          # sum_j h_ij C_ji
    eP_I = (hI * PII.T).sum(dim=1).real
    log("site density of C from the Gram block equals the whole-chain density on 0 ... %d to %.1e" % (nI - 3, float(np.abs(eC_I.numpy()[:nI - 2] - eC_site[:nI - 2]).max())))
    trhP = float(e_all[e_all < 0].sum())
    log("whole chain: 2 sum_i eC_i - Tr h %.13f = E_hC; 2 Tr[h P_-] - Tr h = E_ref %.13f; E_inv %.10e" % (2 * trhC - trh, 2 * trhP - trh, 2 * (trhC - trhP)))
    devd = (eC_I - eP_I).numpy()
    Ein = {}
    for lab, mm in cuts:
        Ein[lab] = 2.0 * float(devd[:mm].sum())
        log("check A at 500: energy of the deviation inside the %s r_s cut (sites < %d) per chain %.6e = %.4f E_inv; outside %.6e" % (
            lab, mm, Ein[lab], Ein[lab] / (E_hC - E_ref), (E_hC - E_ref) - Ein[lab]))
    regs = [("R <= 10 (inside the shell)", 0, shell.min()), ("the shell", shell.min(), shell.max() + 1), ("shell to 128", shell.max() + 1, 128),
            ("128 to 1024", 128, 1024), ("1024 to 3072", 1024, 3072)]
    for lab, i0, i1 in regs:
        log("  region %s: %.6e per chain" % (lab, 2.0 * float(devd[i0:i1].sum())))
    log("  beyond 3072 (E_inv minus the inside of 3072): %.6e per chain" % ((E_hC - E_ref) - 2.0 * float(devd[:3072].sum())))

    # ---------------------------------------------------------------- (7) check B on the stored 8192-site control state at 100 r_s
    ctrl = None
    if a.ctrl100 and os.path.exists(a.ctrl100):
        log("ctrl100 md5 %s" % md5_file(a.ctrl100))
        sc = torch.load(a.ctrl100, map_location="cpu", weights_only=False)
        keys = list(sc.keys())
        Pc = None
        if "channels" in sc:
            Pc = sc["channels"]["1"].get("Phi")
        if Pc is None and "Phi" in sc:
            Pc = sc["Phi"]
        if Pc is None:
            for kk, vv in sc.items():
                if isinstance(vv, dict) and "Phi" in vv:
                    Pc = vv["Phi"]
        log("ctrl100 keys %s; k %s; Phi %s" % (keys, sc.get("k"), None if Pc is None else tuple(Pc.shape)))
        if Pc is not None:
            Pc = Pc.to(CD)
            if Pc.shape[0] == 16384:
                log("ctrl100 Phi has 2n rows: projecting to the + sector by W_+")
                Wp = W1S.W_plus_sparse(8192)
                Pc = torch.as_tensor(Wp.conj().T @ Pc.numpy(), dtype=CD)
            Vc = V[:Pc.shape[0]]
            Mc = (Pc.mH @ Vc).mH @ (Pc.mH @ Vc)
            ctrl = Mc.diagonal().real.numpy()
            log("check B (control state, 8192 sites, interp none, T/r_s 100): occupations %s (500 r_s: %s); weight of the levels beyond site %d: %s" % (
                np.round(ctrl, 5).tolist(), np.round(MC.diagonal().real.numpy(), 5).tolist(), Pc.shape[0],
                np.round(V[Pc.shape[0]:].abs().pow(2).sum(0).numpy(), 8).tolist()))
            del Pc, sc

    np.savez(os.path.join(a.out, "m8_levels_v1.npz"), V=Vn, eV=eV, eRay=eRay, MC=MC.numpy(), M0=M0.numpy(), D=Dm.numpy(),
             rows=np.array(rows), keep=np.array(keep), shell=shell, eC_site=eC_site, eP_I=eP_I.numpy(), eC_I=eC_I.numpy(),
             E_hC=E_hC, E_ref=E_ref, trh=trh, trhP=trhP, ctrl_occ=(ctrl if ctrl is not None else np.zeros(0)),
             cuts=json.dumps(out_cuts), Ein=json.dumps(Ein))
    np.save(os.path.join(a.out, "m8_PII_3074_v1.npy"), PII.numpy())
    log("saved; wall %.1f s" % (time.time() - T0W))


if __name__ == "__main__":
    main()
