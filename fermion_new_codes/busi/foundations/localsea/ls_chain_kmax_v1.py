#!/usr/bin/env python
"""ls_chain_kmax_v1.py (busi, inbox #455; ReviewA_found7_v1.md sec. 3, the one new computation; v1.1 sec. 8 item 5): the
channel-truncation test of the 3D/chain gap. On one record (default s8_T16, where R2 = P_neg = P_lb on channels 1..10), the
instantaneous negative set of h_m = {N, h}/2 + {v, Q}/2 at the record's applied fields, channels kappa = 1..K (K = 40), weight
16 kappa (the engine's), with per-channel polarization eps_pol = dens(h, P_neg) - dens(h, P_0), J_pol likewise, and each channel's
band character against the clock-only sea (count of eigenvectors with character in (0.05, 0.95); count of negative-set vectors of
upper character and vice versa). The partial sums K = 10, 20, 40 come from one run (channels are independent). The applied fields
are read from the audit npz of ls_chain_audit_v1 mode slice (R, N, v unchanged from the slice), and the K = 10 sum is checked
against that npz's eps_R2 - eps_0 and J_R2 - J_0. Window tables as v1.1 sec. 6.3 (R_t from --Rt): production polarization at
K = 10, 20, 40, the 3D cubic second-order sea (the npz's rho3d, J3d) and their ratios. Box CPU, torch complex128, at most 8 threads.
Outputs <out>/ls_chain_kmax_v1_<tag>.npz (per-channel arrays) and .json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_chain_audit_v1 as L   # noqa: E402  (sets the thread count from --threads, default 8)
import argparse   # noqa: E402
import json   # noqa: E402
import time   # noqa: E402

import numpy as np   # noqa: E402
import torch   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--audit_npz", required=True); ap.add_argument("--K", type=int, default=40)
ap.add_argument("--Rt", type=float, required=True)
ap.add_argument("--m", type=float, default=0.2)
ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--threads", default="8")
a = ap.parse_args()
Z = np.load(a.audit_npz)
R, N, v = np.asarray(Z["R"], float), np.asarray(Z["N"], float), np.asarray(Z["v"], float)
n = len(R)
pe = np.zeros((a.K, n)); pJ = np.zeros((a.K, n)); info = {}; t0 = time.time()
for kap in range(1, a.K + 1):
    w = 16.0 * kap
    h, Q = L.chain_dense(R, a.m, kap)
    e, U, Uneg = L.neg_cols(L.clocked(h, Q, N, v))
    _, _, U0 = L.neg_cols(h)
    _, _, UN = L.neg_cols(L.clocked(h, Q, N, np.zeros(n)))
    bN = ((UN.conj().T @ U).abs() ** 2).sum(0).numpy(); neg = (e < 0).numpy()
    pe[kap - 1] = w * (L.dens(h, Uneg) - L.dens(h, U0)); pJ[kap - 1] = w * (L.dens(Q, Uneg) - L.dens(Q, U0))
    info[kap] = {"min_abs_e": float(e.abs().min()), "n_neg": int(Uneg.shape[1]), "mixed": int(np.sum((bN > 0.05) & (bN < 0.95))),
                 "neg_upper_character": int(np.sum(neg & (bN < 0.5))), "pos_lower_character": int(np.sum(~neg & (bN > 0.5))),
                 "eps_pol_total": float(pe[kap - 1].sum()), "J_pol_total": float(pJ[kap - 1].sum())}
    print("kappa %d" % kap, json.dumps(info[kap]), "%.1fs" % (time.time() - t0), flush=True)
    del h, Q, e, U, Uneg, U0, UN
st_e = Z["eps_R2"] - Z["eps_0"]; st_J = Z["J_R2"] - Z["J_0"]
chk = {"K10_eps_vs_stored_maxabs": float(np.max(np.abs(pe[:10].sum(0) - st_e))), "K10_J_vs_stored_maxabs": float(np.max(np.abs(pJ[:10].sum(0) - st_J))),
       "stored_eps_pol_maxabs": float(np.max(np.abs(st_e)))}
print("check", json.dumps(chk), flush=True)
Rt = a.Rt
win = [(1, 4), (4, 8), (8, 16), (16, 32), (32, Rt), (Rt, 2 * Rt), (2 * Rt, 4 * Rt), (4 * Rt, n + 1)]
rows = []
for lo, hi in win:
    m = (R >= lo) & (R < hi)
    r_ = {"r": [lo, hi], "rho3d": float(Z["rho3d"][m].sum()), "J3d": float(Z["J3d"][m].sum())}
    for K in (10, 20, 30, 40):
        if K <= a.K:
            r_["eps_K%d" % K] = float(pe[:K, m].sum()); r_["J_K%d" % K] = float(pJ[:K, m].sum())
    rows.append(r_)
    print("r %7.1f to %7.1f | eps K10 %.4e K20 %.4e K40 %.4e | 3D %.4e | 3D/K10 %.3g 3D/K40 %.3g | K40/K10 %.3f | J K10 %.4e K40 %.4e 3D %.4e" % (
        lo, hi, r_["eps_K10"], r_.get("eps_K20", np.nan), r_.get("eps_K40", np.nan), r_["rho3d"], r_["rho3d"] / r_["eps_K10"],
        r_["rho3d"] / r_.get("eps_K40", np.nan), r_.get("eps_K40", np.nan) / r_["eps_K10"], r_["J_K10"], r_.get("J_K40", np.nan), r_["J3d"]), flush=True)
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "ls_chain_kmax_v1_%s.npz" % a.tag), R=R, eps_pol_k=pe, J_pol_k=pJ)
meta = {"script": "ls_chain_kmax_v1.py", "audit_npz": a.audit_npz, "K": a.K, "m": a.m, "Rt": Rt, "check_K10": chk, "per_channel": info,
        "windows": rows, "seconds": time.time() - t0}
with open(os.path.join(a.out, "ls_chain_kmax_v1_%s.json" % a.tag), "w") as f:
    json.dump(meta, f, indent=1, default=float)
print("DONE %.1fs" % meta["seconds"])
