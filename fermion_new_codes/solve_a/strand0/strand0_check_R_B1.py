#!/usr/bin/env python
"""Strand 0 check requested in the lead's judgment (revision 1, sec. 0): at B1_J_W6_q0, sector N = 4, from the saved npz,
R = sum_x (N4_rhoM_x - N4_enat0_x); report Tr R/70, ||R|| and ||R - (Tr R/70) 1|| for m in {0.1, 0.2}, U = 0 (and U = 0.1)."""
import json
import os
import sys
import numpy as np

mats = sys.argv[1]
out = []
for m in (0.1, 0.2):
    for U in (0.0, 0.1):
        f = os.path.join(mats, "law_B1_J_W6_q0_m%g_U%g.npz" % (m, U))
        if not os.path.exists(f):
            continue
        z = np.load(f)
        R = sum(z["N4_rhoM_x%d" % x] - z["N4_enat0_x%d" % x] for x in range(4))
        d = R.shape[0]
        t = np.trace(R).real / d
        out.append({"m": m, "U": U, "dim": d, "TrR_over_dim": float(t), "norm_R": float(np.linalg.norm(R)),
                    "norm_R_minus_trace_part": float(np.linalg.norm(R - t * np.eye(d)))})
print(json.dumps(out, indent=1))
