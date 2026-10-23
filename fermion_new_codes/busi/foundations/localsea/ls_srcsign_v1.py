#!/usr/bin/env python
"""ls_srcsign_v1.py (busi, inbox #455; ReviewB_found7_v1.md sec. 3 item 1, sec. 2 'A sign inconsistency'): which sign does the
record's source carry? On each audited slice record, compare the field module's input sources F["eps_tot"][k], F["J_tot"][k] and
F["eps_pairs"][k], F["J_pairs"][k] (the stored history fields file) with the audit's site densities of the propagated sea minus the
production reference, eps_sea - eps_R2 and J_sea - J_R2 (ls_chain_audit_v1 mode slice outputs). If eps_pairs = +(sea - R2), then
replacing the reference R2 by P_0 + b changes the source by Delta s = +(eps_R2 - eps_0) - b (v1.1 sec. 6.1's text), and
ls_reviewbA_3d_v2.py, which added b - (eps_R2 - eps_0), solved with the reversed sign. Post-processing only, seconds, one thread.
Prints a json and writes <out>/ls_srcsign_v1.json."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse   # noqa: E402
import json   # noqa: E402

import numpy as np   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="<internal>/audit")
ap.add_argument("--fields4", required=True); ap.add_argument("--fields8", required=True)
ap.add_argument("--out", default="<internal>/localsea")
a = ap.parse_args()
res = {}
for tag, hist, T in (("s4_T07", "s4", 7.0), ("s4_T13", "s4", 13.0), ("s4_T14", "s4", 14.0), ("s4_T15", "s4", 15.0),
                     ("s4_T16", "s4", 16.0), ("s8_T14", "s8", 14.0), ("s8_T16", "s8", 16.0)):
    p = os.path.join(a.dir, "ls_chain_audit_v1_%s.npz" % tag)
    if not os.path.exists(p):
        continue
    Z = np.load(p)
    F = np.load(a.fields4 if hist == "s4" else a.fields8, allow_pickle=True)
    k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - T) < 1e-9)[0])
    ep, Jp = np.asarray(F["eps_pairs"][k], float), np.asarray(F["J_pairs"][k], float)
    et, Jt = np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float)
    de, dJ = Z["eps_sea"] - Z["eps_R2"], Z["J_sea"] - Z["J_R2"]
    sc = max(float(np.max(np.abs(ep))), 1e-300); scJ = max(float(np.max(np.abs(Jp))), 1e-300)
    out = {"k": k, "pairs_eps_maxabs": sc, "pairs_J_maxabs": scJ,
           "eps_pairs_minus_(sea-R2)_maxabs": float(np.max(np.abs(ep - de))), "eps_pairs_plus_(sea-R2)_maxabs": float(np.max(np.abs(ep + de))),
           "J_pairs_minus_(sea-R2)_maxabs": float(np.max(np.abs(Jp - dJ))), "J_pairs_plus_(sea-R2)_maxabs": float(np.max(np.abs(Jp + dJ))),
           "eps_tot_total": float(et.sum()), "eps_pairs_total": float(ep.sum()), "eps_dust_total(tot-pairs)": float((et - ep).sum()),
           "(sea-R2)_eps_total": float(de.sum()), "(sea-R2)_J_total": float(dJ.sum()),
           "pol_eps_total(R2-0)": float((Z["eps_R2"] - Z["eps_0"]).sum()), "pol_J_total(R2-0)": float((Z["J_R2"] - Z["J_0"]).sum()),
           "rho3d_total": float(Z["rho3d"].sum()), "J3d_total": float(Z["J3d"].sum())}
    res[tag] = out
    print(tag, json.dumps(out), flush=True)
with open(os.path.join(a.out, "ls_srcsign_v1.json"), "w") as f:
    json.dump(res, f, indent=1)
