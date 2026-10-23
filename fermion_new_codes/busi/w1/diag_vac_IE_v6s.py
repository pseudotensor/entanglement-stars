#!/usr/bin/env python
"""diag_vac_IE_v6s.py (busi lane, #309 gate C follow-up): the dense held vacuum's interior and exterior blocks at one B2 record on the host (CPU):
S_I(P_{-,+}) against S_E(P_{-,+}) per cut, both Gram routes, and the exterior Gram's eigenvalue range. Usage: python diag_vac_IE_v6s.py REC FIELDS T_rs"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import page_history_v6s as V6S, page_w1_sector_v1d as W1S, page_ops_v6 as OPS
torch.set_num_threads(8)
H = V6S.History(sys.argv[1], sys.argv[2], 8192, None, "applied"); rs = float(H.rs_nom)
k = int(np.argmin(np.abs(H.T - float(sys.argv[3]) * rs)))
ops = V6S.ChannelOps(1, H.R, torch.device("cpu")); hp = W1S.sector_plus_csr(ops.h_csr(*H.fields(k)))
e, V = OPS.eigh_csr(hp, torch.device("cpu")); Vp = V[:, e < 0]
for x in (2, 16, 24, 30, 48):
    m = int(np.sum(np.asarray(H.R) <= x * rs))
    evI, rI = W1S.gram_eigs(Vp[:m]); evE, rE = W1S.gram_eigs(Vp[m:])
    G = Vp[m:] @ Vp[m:].mH
    evE2 = torch.linalg.eigvalsh(0.5 * (G + G.mH))
    SI = 2 * float(OPS.binary_entropy_t(evI).sum()); SE = 2 * float(OPS.binary_entropy_t(evE).sum()); SE2 = 2 * float(OPS.binary_entropy_t(evE2).sum())
    print("T/r_s %g cut %g: S_I %.12f (%s) S_E %.12f (%s) S_E big-gram %.12f; I-E %.2e; exterior Gram eigenvalues [%.3e, %.3e]" % (
        H.T[k] / rs, x, SI, rI, SE, rE, SE2, SI - SE, float(evE.min()), float(evE.max())), flush=True)
