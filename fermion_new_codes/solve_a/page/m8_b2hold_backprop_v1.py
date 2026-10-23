#!/usr/bin/env python
"""m8_b2hold_backprop_v1.py (solve lane, B2 hold-drain M8, checks A and B on the hold's own clocks).

On the hold the state evolves by the one held operator: Phi(t) = exp(-i h (t - t_100)) Phi(t_100) for t >= 100 r_s, so for s = t_500 - t
    C(t)_ab = e_a^dag exp(+i h s) C_500 exp(-i h s) e_b = y_a(s)^dag y_b(s),   y_a(s) = Phi_500^dag exp(-i h s) e_a.
A few vectors x (the block levels of stage 1, zero-padded, and the unit vectors of the sites 0 ... nbs - 1) are propagated forward by the
production's own lean propagator (page_interior_v6s.BandedLean + propagate_lean, page_ops_v6.cheb_order at --cheb-tol 1e-16) in steps of
1 r_s, and projected on the stored final Phi_500 every 10 r_s. This reconstructs, without any stored intermediate state, at t = 500, 490, ..., 100:
the levels' occupation matrix (check B: the inversion at 100 against 500), the interior block C_II at the cuts inside nbs - 2 sites
(S_I(C), against the run's stored readouts), and the energy of the deviation inside those cuts (check A on the hold's readouts).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
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
    ap.add_argument("--summary", default="<internal>/v6s_B2hold_k1_summary.json")
    ap.add_argument("--levels", default="<internal>/m8_levels_v1.npz")
    ap.add_argument("--pii", default="<internal>/m8_PII_3074_v1.npy")
    ap.add_argument("--nbs", type=int, default=130)
    ap.add_argument("--nblk", type=int, default=4096)
    ap.add_argument("--win", type=float, default=0.02)
    ap.add_argument("--cheb-tol", type=float, default=1e-16)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--every-rs", type=int, default=10)
    ap.add_argument("--span-rs", type=int, default=400)
    ap.add_argument("--out", default="<internal>/m8_backprop_v1.npz")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.code_v6); sys.path.insert(0, a.code)
    import page_history_v6s as PH
    import page_interior_v6s as IR
    import page_w1_sector_v1d as W1S
    CD = torch.complex128

    H = PH.History(a.record, a.fields, 32768, None, "applied")
    rs = float(H.rs_nom)
    hold = IR.hold_extend(H, 100.0 * rs, 500.0 * rs, H.tau, log)
    ops = PH.ChannelOps(1, H.R, torch.device("cpu"))
    N, v = H.fields(int(hold["k_hold"]))
    hp = W1S.sector_plus_csr(ops.h_csr(N, v)).tocsr()
    log("held h_+ md5 %s" % IR._hp_md5(hp))
    n = hp.shape[0]

    # the block's window levels (all of them, the kept four among them), as stage 1
    A = torch.as_tensor(hp[:a.nblk, :a.nblk].toarray(), dtype=CD)
    eb, Ub = torch.linalg.eigh(A); del A
    sw = torch.where(eb.abs() < a.win)[0]
    Lv = torch.zeros((n, len(sw)), dtype=CD); Lv[:a.nblk] = Ub[:, sw]
    eL = eb[sw].numpy(); del Ub
    Z = np.load(a.levels)
    Vk = torch.as_tensor(Z["V"], dtype=CD)
    keep = [int(i) for i in Z["keep"]]
    ov = (Lv[:, keep].mH @ Vk).abs()
    log("%d window levels; the kept %s reproduce stage 1's vectors (|overlap| diag %s)" % (len(sw), keep, np.round(ov.diagonal().numpy(), 12).tolist()))
    PII = torch.as_tensor(np.load(a.pii), dtype=CD)
    hI = torch.as_tensor(hp[:a.nbs, :a.nbs].toarray(), dtype=CD)
    eP = (hI * PII[:a.nbs, :a.nbs].T).sum(dim=1).real
    cuts = [(lab, m) for lab, m in [("2", 128), ("16", 1024), ("24", 1536), ("30", 1920), ("48", 3072)] if m + 2 <= a.nbs]
    SP = {lab: 2.0 * ent(PII[:m, :m]) for lab, m in cuts}

    summ = json.load(open(a.summary))
    stored = {round(r["t_over_rs"], 3): r["per_channel"]["1"] for r in summ["rows"]}

    st = torch.load(a.istate, map_location="cpu", weights_only=False)
    Phi = st["channels"]["1"]["Phi"]; del st
    log("Phi_500 %s" % (tuple(Phi.shape),))

    nL = Lv.shape[1]
    X = torch.zeros((n, nL + a.nbs), dtype=CD)
    X[:, :nL] = Lv
    X[torch.arange(a.nbs), nL + torch.arange(a.nbs)] = 1.0
    del Lv
    B = IR.BandedLean(hp, torch.device("cpu"))
    tau = rs                                                                   # 1 r_s per step
    out = {"t_rs": [], "occ": [], "Mkeep": [], "S_I_C": {lab: [] for lab, _ in cuts}, "excess": {lab: [] for lab, _ in cuts},
           "stored_excess": {lab: [] for lab, _ in cuts}, "E_in": {lab: [] for lab, _ in cuts}, "norm_drift": []}
    n0 = X.norm(dim=0)
    Kall = 0
    for step in range(a.span_rs + 1):
        if step % a.every_rs == 0:
            t_rs = 500.0 - step
            Y = Phi.mH @ X                                                     # y_a = Phi^dag x_a
            YL = Y[:, :nL]; Ye = Y[:, nL:]
            occ = (YL.abs() ** 2).sum(dim=0).numpy()
            Mk = (YL[:, keep].mH @ YL[:, keep]).numpy()
            Cb = Ye.mH @ Ye                                                    # C(t)_ab = y_a^dag y_b
            eC = (hI * Cb.T).sum(dim=1).real
            dev = (eC - eP).numpy()
            out["t_rs"].append(t_rs); out["occ"].append(occ); out["Mkeep"].append(Mk)
            out["norm_drift"].append(float((X.norm(dim=0) - n0).abs().max()))
            parts = []
            for lab, m in cuts:
                SC = 2.0 * ent(Cb[:m, :m]); ex = SC - SP[lab]; Ein = 2.0 * float(dev[:m].sum())
                se = stored.get(round(t_rs, 3), {}).get("interior", {}).get("cuts", {}).get(lab, {}).get("excess", float("nan"))
                out["S_I_C"][lab].append(SC); out["excess"][lab].append(ex); out["stored_excess"][lab].append(se); out["E_in"][lab].append(Ein)
                parts.append("%s r_s: excess %.9f (stored %.9f, diff %.1e), E_in %.6e" % (lab, ex, se, ex - se, Ein))
            log("t %.0f r_s: kept occupations %s; %s; K so far %d; norm drift %.1e" % (
                t_rs, np.round(occ[keep], 5).tolist(), "; ".join(parts), Kall, out["norm_drift"][-1]))
            np.savez(a.out, t_rs=np.array(out["t_rs"]), occ=np.array(out["occ"]), Mkeep=np.array(out["Mkeep"]), eL=eL, keep=np.array(keep),
                     cuts=json.dumps(cuts), S_I_C=json.dumps(out["S_I_C"]), excess=json.dumps(out["excess"]),
                     stored_excess=json.dumps(out["stored_excess"]), E_in=json.dumps(out["E_in"]), SP=json.dumps(SP),
                     norm_drift=np.array(out["norm_drift"]), nbs=a.nbs)
        if step == a.span_rs:
            break
        t1 = time.time()
        X, K, tail = IR.propagate_lean(B, X, tau, a.cheb_tol, a.chunk)
        Kall += int(K)
        if step == 0:
            log("one 1 r_s step: K %d, tail %.1e, %.1f s" % (K, tail, time.time() - t1))
    log("done; wall %.1f s" % (time.time() - T0W))


if __name__ == "__main__":
    main()
