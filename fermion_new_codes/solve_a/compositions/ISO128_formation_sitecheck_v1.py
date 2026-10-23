#!/usr/bin/env python
"""ISO128_formation_sitecheck_v1.py (solve lane): which stored fields carry the records' min(N - |v|), and what the T0011.50 slice's
site fields give at 11.5. Stored files only (box CPU, light). Usage: python ISO128_formation_sitecheck_v1.py"""
import json

import numpy as np

B = "<internal>/solve_a_bulk"
T = "lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"
R = json.load(open(B + "/iso128_h2/run_h2/collapse_%s.json" % T))["records"]
F = np.load(B + "/iso128_h2/run_h2/collapse_%s_fields.npz" % T, allow_pickle=True)  # own run file; r_nodes is ragged
S = np.load(B + "/iso128_h2/states_h2/orbitals_%s_T0011.50.npz" % T)
for t in (5.0, 11.1875, 11.25):
    i = [k for k, r in enumerate(R) if r["T_over_rs"] == t][0]
    fs = F["N"][i] - abs(F["v"][i])
    rn = np.asarray(F["r_nodes"][i])
    fn = np.asarray(F["N_nodes"][i]) - abs(np.asarray(F["v_nodes"][i]))
    print("T/rs %.4f json min %.6f at %.3f | site fields min %.6f at %.1f | node fields min %.6f at %.1f (%d nodes)"
          % (t, R[i]["min_N_minus_absv"], R[i]["r_min_N_minus_absv"], fs.min(), F["R"][fs.argmin()], fn.min(), rn[fn.argmin()], len(rn)))
f = S["N_applied"] - abs(S["v_applied"])
print("slice T/rs %.4f (T %.1f, rs %.1f): site N_applied, v_applied: min %.6f at %.1f; sign changes %d; no node fields stored in the slice"
      % (float(S["T"]) / float(S["rs"]), float(S["T"]), float(S["rs"]), f.min(), S["R"][f.argmin()], int(np.sum(np.sign(f[:-1]) != np.sign(f[1:])))))
