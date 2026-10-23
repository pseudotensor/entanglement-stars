#!/usr/bin/env python
"""ls_chain_scale_v2.py (busi, inbox #449): ls_chain_audit_v1 mode scale, with the applied fields read from the lambda = 1 output
(ls_chain_audit_v1_s4_T15_lam1.npz holds R, N_applied, v_applied of the slice unchanged) instead of the slice file, which left the box
at 14:00 UTC 22 Oct (<internal>/slices emptied). Channels kappa = 1..10 with weight 16 kappa (the engine's,
ls_chain_audit_v1 docstring). --check recomputes lambda = 1 and compares with the stored lambda = 1 polarization (validates the
weights and the field read). Box CPU, torch complex128, at most 8 threads. Outputs <out>/ls_chain_audit_v1_<tag>.npz/.json in the
same format as mode scale."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_chain_audit_v1 as L   # noqa: E402  (sets the thread count from --threads, default 8)
import argparse   # noqa: E402
import json   # noqa: E402
import time   # noqa: E402

import numpy as np   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fields_npz", required=True); ap.add_argument("--lam", type=int, required=True)
ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--threads", default="8")
a = ap.parse_args()
Z = np.load(a.fields_npz)
R, N, v = np.asarray(Z["R"], float), np.asarray(Z["N"], float), np.asarray(Z["v"], float)
Rp, Np, vp = (R, N, v) if a.lam == 1 else L.interp_fields(R, N, v, a.lam)
acc = {k: np.zeros(len(Rp)) for k in ("eps_pol", "J_pol")}
info = {}; t0 = time.time()
for kap in range(1, a.K + 1):
    w = 16.0 * kap
    h, Q = L.chain_dense(Rp, a.m, kap)
    e, U, Uneg = L.neg_cols(L.clocked(h, Q, Np, vp))
    _, _, U0 = L.neg_cols(h)
    acc["eps_pol"] += w * (L.dens(h, Uneg) - L.dens(h, U0)); acc["J_pol"] += w * (L.dens(Q, Uneg) - L.dens(Q, U0))
    info[kap] = {"min_abs_e": float(e.abs().min()), "n_neg": int(Uneg.shape[1]), "n_half": len(Rp)}
    print("lambda %d kappa %d" % (a.lam, kap), json.dumps(info[kap]), "%.1fs" % (time.time() - t0), flush=True)
    del h, Q, e, U, Uneg, U0
meta = {"script": "ls_chain_scale_v2.py", "mode": "scale", "m": a.m, "fields_npz": a.fields_npz, "lam": a.lam, "per_channel": info}
if a.lam == 1:
    meta["check_vs_stored"] = {"eps_pol_maxabs_diff": float(np.max(np.abs(acc["eps_pol"] - Z["eps_pol"]))),
                               "J_pol_maxabs_diff": float(np.max(np.abs(acc["J_pol"] - Z["J_pol"]))),
                               "eps_pol_maxabs": float(np.max(np.abs(Z["eps_pol"])))}
    print("check", json.dumps(meta["check_vs_stored"]), flush=True)
os.makedirs(a.out, exist_ok=True)
np.savez_compressed(os.path.join(a.out, "ls_chain_audit_v1_%s.npz" % a.tag), R=Rp, N=Np, v=vp, **acc)
meta["seconds"] = time.time() - t0
with open(os.path.join(a.out, "ls_chain_audit_v1_%s.json" % a.tag), "w") as f:
    json.dump(meta, f, indent=1, default=float)
print("DONE %.1fs" % meta["seconds"])
