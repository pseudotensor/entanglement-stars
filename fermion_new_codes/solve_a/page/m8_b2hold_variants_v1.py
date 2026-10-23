#!/usr/bin/env python
"""m8_b2hold_variants_v1.py (solve lane, B2 hold-drain M8): the robustness of the bound-block reading to the block's size and to the selection.

For block sizes --nblks the leading block of the held h_+ is diagonalized (torch eigh); levels with |e| < --win and shell weight > --shell-min
are kept; their occupations in the stored final Phi_+; D = V^dag (C - P_-) V with P_- V by the full-chain contour (the production's grid) and,
beside it, with the step reference theta(-e); the end-state excess at the five cuts (exact-P reference, its dephased form, the theta reference)
from stage 1's vacuum block P_II. Variant 'plus': the kept set plus every window level with shell weight > 0.1.
"""
import argparse
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


def ent(M):
    ev = torch.linalg.eigvalsh(0.5 * (M + M.mH)).clamp(0.0, 1.0)
    return float(-(torch.special.xlogy(ev, ev) + torch.special.xlogy(1.0 - ev, 1.0 - ev)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="<internal>/code_d13")
    ap.add_argument("--code-v6", default="<internal>/code_v6")
    ap.add_argument("--record", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json")
    ap.add_argument("--fields", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz")
    ap.add_argument("--istate", default="<internal>/v6s_B2hold_k1_istate.pt")
    ap.add_argument("--pii", default="<internal>/m8_PII_3074_v1.npy")
    ap.add_argument("--nblks", default="2048,3072,4096,6144,8192")
    ap.add_argument("--win", type=float, default=0.02)
    ap.add_argument("--shell-min", type=float, default=0.3)
    ap.add_argument("--out", default="<internal>/m8_variants_v1.json")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code_v6); sys.path.insert(0, a.code)
    import page_history_v6s as PH
    import page_interior_v6s as IR
    import page_w1_sector_v1d as W1S
    import page_ops_v6 as OPS
    CD = torch.complex128
    H = PH.History(a.record, a.fields, 32768, None, "applied")
    rs = float(H.rs_nom)
    hold = IR.hold_extend(H, 100.0 * rs, 500.0 * rs, H.tau, log)
    ops = PH.ChannelOps(1, H.R, torch.device("cpu"))
    N, v = H.fields(int(hold["k_hold"]))
    hp = W1S.sector_plus_csr(ops.h_csr(N, v)).tocsr()
    n = hp.shape[0]
    shell = np.where(N < np.abs(v))[0]
    ab, _u = IR._herm_band_upper(hp)
    e_all = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
    delta = float(np.min(np.abs(e_all))); w = OPS.gershgorin(hp)
    hs, U = 0.3, 37.0
    s_lo = math.log(delta) - U; s_hi = math.log(w) + U
    m = int(math.ceil((s_hi - s_lo) / hs)) + 1
    s_nodes = s_lo + hs * np.arange(m)
    abF0 = IR.band_general(hp, 2, 2)

    def P_apply(Vn):
        SV = np.zeros_like(Vn)
        for j in range(m):
            tj = math.exp(s_nodes[j]); z = 1j * tj
            abF = abF0.copy(); abF[2, :] -= z
            X1 = sla.solve_banded((2, 2), abF, Vn, check_finite=False)
            abF = abF0.copy(); abF[2, :] += z
            X2 = sla.solve_banded((2, 2), abF, Vn, check_finite=False)
            SV += 0.5 * (X1 + X2) * tj
        return 0.5 * (Vn - (2.0 / math.pi) * hs * SV)

    PII = torch.as_tensor(np.load(a.pii), dtype=CD)
    st = torch.load(a.istate, map_location="cpu", weights_only=False)
    Phi = st["channels"]["1"]["Phi"]; del st
    cuts = [("2", 128), ("16", 1024), ("24", 1536), ("30", 1920), ("48", 3072)]
    SP = {lab: ent(PII[:mm, :mm]) for lab, mm in cuts}
    res = {}
    for nb in [int(x) for x in a.nblks.split(",")]:
        t1 = time.time()
        A = torch.as_tensor(hp[:nb, :nb].toarray(), dtype=CD)
        eb, Ub = torch.linalg.eigh(A); del A
        sw = torch.where(eb.abs() < a.win)[0]
        Uw = Ub[:, sw]; ew = eb[sw].numpy(); del Ub
        wt = Uw.abs().pow(2).numpy()
        wsh = wt[shell].sum(0); w128 = wt[:128].sum(0)
        Vpad = torch.zeros((n, len(sw)), dtype=CD); Vpad[:nb] = Uw
        Y = Phi.mH @ Vpad
        occ = (Y.abs() ** 2).sum(0).numpy()
        resid = np.linalg.norm(hp @ Vpad.numpy() - Vpad.numpy() * ew[None, :], axis=0)
        log("nblk %d: eigh %.1f s; %d window levels" % (nb, time.time() - t1, len(sw)))
        sets = {"kept": [i for i in range(len(sw)) if wsh[i] > a.shell_min], "plus": [i for i in range(len(sw)) if wsh[i] > 0.1]}
        rv = {"levels": [], "sets": {}}
        for i in range(len(sw)):
            if wsh[i] > 0.1 or abs(occ[i] - (ew[i] < 0)) > 0.05:
                rv["levels"].append({"e": float(ew[i]), "n": float(occ[i]), "w_shell": float(wsh[i]), "w_128": float(w128[i]), "resid": float(resid[i])})
                log("   e %+.6f  n %.5f  w(shell) %.3f  w(<128) %.3f  resid %.2e" % (ew[i], occ[i], wsh[i], w128[i], resid[i]))
        for sname, idx in sets.items():
            if not idx:
                continue
            V = Vpad[:, idx]; Vn = V.numpy()
            M0 = torch.as_tensor(Vn.conj().T @ P_apply(Vn), dtype=CD)
            MC = Y[:, idx].mH @ Y[:, idx]
            Dx = MC - M0
            Dt = MC - torch.diag(torch.as_tensor((ew[idx] < 0).astype(float), dtype=CD))
            eR = np.real(np.diag(Vn.conj().T @ (hp @ Vn)))
            row = {"idx_e": [float(ew[i]) for i in idx], "occ": [float(occ[i]) for i in idx], "M0_diag": M0.diagonal().real.tolist(),
                   "D_diag": Dx.diagonal().real.tolist(), "E_levels_per_chain": 2.0 * float((eR * Dx.diagonal().real.numpy()).sum()), "cuts": {}}
            for lab, mm in cuts:
                Vm = V[:mm]
                Sx = ent(PII[:mm, :mm] + Vm @ Dx @ Vm.mH); Sd = ent(PII[:mm, :mm] + Vm @ torch.diag(Dx.diagonal()) @ Vm.mH)
                St = ent(PII[:mm, :mm] + Vm @ Dt @ Vm.mH); Std = ent(PII[:mm, :mm] + Vm @ torch.diag(Dt.diagonal()) @ Vm.mH)
                row["cuts"][lab] = {"end": 2 * (Sx - SP[lab]), "end_deph": 2 * (Sd - SP[lab]), "end_theta": 2 * (St - SP[lab]), "end_theta_deph": 2 * (Std - SP[lab])}
            rv["sets"][sname] = row
            log("  set %s (%d levels, e %s): occ %s; D diag %s; E in levels %.4e; end (exact P) %s; dephased %s; theta %s; theta dephased %s" % (
                sname, len(idx), np.round(row["idx_e"], 6).tolist(), np.round(row["occ"], 5).tolist(), np.round(row["D_diag"], 5).tolist(), row["E_levels_per_chain"],
                [round(row["cuts"][l]["end"], 4) for l, _ in cuts], [round(row["cuts"][l]["end_deph"], 4) for l, _ in cuts],
                [round(row["cuts"][l]["end_theta"], 4) for l, _ in cuts], [round(row["cuts"][l]["end_theta_deph"], 4) for l, _ in cuts]))
        res[str(nb)] = rv
        json.dump(res, open(a.out, "w"), indent=1)
    log("done; wall %.1f s" % (time.time() - T0W))


if __name__ == "__main__":
    main()
