#!/usr/bin/env python
"""ls_reviewbA_3d_v2.py (busi, inbox #449; v1 plus --tags, run on the sigma/a = 8 records only after the sigma/a = 4 _to40 record files left the box at 14:00 UTC 22 Oct; ReviewB_found4_v1.md sec. 5.2 A with reviewer B's candidate b[u] = the 3D cubic local sea at
second order): on each audited slice record, one field solve with the record's sources changed by
    Delta s = b[u] - W[(P_prod - P_0)]  per site,  i.e.  eps' = eps_tot - (eps_R2 - eps_0) + rho3d,  J' = J_tot - (J_R2 - J_0) + J3d,
(the reference replaced, the state held), with the reader's field module (engine_fields_v16, md5 7a69faad), against the record's own
solve: the roots, min(N - |v|) and the collar charge E = r w^2/(2 g). Variants: the replacement everywhere; only outside r = 4a
(where the gradient expansion is nearer its regime); only the removal of the production polarization (b = 0, the fixed flat P_0).
Post-processing only (no eigensolves). Writes <out>/ls_reviewbA_3d_v1.json and prints the table."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ls_audit_read_v1 import field_module, solve, E_of, record_of   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="<internal>/audit")
ap.add_argument("--json4", default=""); ap.add_argument("--fields4", default="")
ap.add_argument("--json8", required=True); ap.add_argument("--fields8", required=True)
ap.add_argument("--out", default="<internal>/localsea")
ap.add_argument("--tags", default="s4_T07,s4_T13,s4_T14,s4_T15,s4_T16,s8_T14,s8_T16")
a = ap.parse_args()
FM = field_module()
src = {"s4": (a.json4, a.fields4), "s8": (a.json8, a.fields8)}
res = {}
for tag, hist, T in (("s4_T07", "s4", 7.0), ("s4_T13", "s4", 13.0), ("s4_T14", "s4", 14.0), ("s4_T15", "s4", 15.0),
                     ("s4_T16", "s4", 16.0), ("s8_T14", "s8", 14.0), ("s8_T16", "s8", 16.0)):
    p = os.path.join(a.dir, "ls_chain_audit_v1_%s.npz" % tag)
    if tag not in a.tags.split(",") or not os.path.exists(p):
        continue
    Z = np.load(p)
    rec, _ = record_of(src[hist][0], T)
    F = np.load(src[hist][1], allow_pickle=True)
    k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - T) < 1e-9)[0])
    g, ell = float(F["g"]), float(F["ell"]); Rs = np.asarray(F["R"], float)
    eps0, J0 = np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float)
    R = Z["R"]
    pol_e = Z["eps_R2"] - Z["eps_0"]; pol_J = Z["J_R2"] - Z["J_0"]
    de_all, dj_all = Z["rho3d"] - pol_e, Z["J3d"] - pol_J
    out = {"T_over_rs": T, "record_zeros": rec.get("trapping_zeros"), "record_min": rec.get("min_N_minus_absv"), "record_E": rec.get("E"),
           "sources_eps_total": float(eps0.sum()), "sources_J_total": float(J0.sum())}
    for lab, de, dj in (("record", 0 * pol_e, 0 * pol_J), ("LS2_3d", de_all, dj_all),
                        ("LS2_3d_outside_4a", np.where(R >= 4, de_all, 0.0), np.where(R >= 4, dj_all, 0.0)),
                        ("flat_P0", -pol_e, -pol_J)):
        sol = solve(FM, eps0 + de, J0 + dj, g, Rs, ell)
        if sol is None:
            out[lab] = {"failed": True}
            print(tag, lab, "solve failed", flush=True)
            continue
        rc, Nc, vc = E_of(sol, Rs)
        out[lab] = {"zeros": sol["zeros"], "min": sol["min"], "r_min": sol["r_min"], "E_collar": rc * (vc / Nc) ** 2 / (2 * g),
                    "r_collar": rc, "d_eps_total": float(np.sum(de)), "d_J_total": float(np.sum(dj))}
        print(tag, lab, json.dumps(out[lab], default=float), flush=True)
    res[tag] = out
with open(os.path.join(a.out, "ls_reviewbA_3d_v2.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
