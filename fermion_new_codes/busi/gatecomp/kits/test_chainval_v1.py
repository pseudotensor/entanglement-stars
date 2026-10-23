#!/usr/bin/env python
"""test_chainval_v1.py (busi, GATECOMP kit (i)): the comparator's self-test on stored data. From job (b)'s run files it writes two X
runs in OUT: 'crop' = P's json with args nsites 2048 and P's fields rows cropped to the first 2048 sites (and the dense nodes to r <= 2048):
the comparator must print PASS with zero differences; 'pert' = crop with E_over_E0 moved by 1e-6 at T/r_s 6.0 and v moved by 1e-4 at the
first 10 sites at 8.0: the comparator must FAIL at exactly those two records (E_over_E0, fields). Box CPU, small files.
  test_chainval_v1.py P_DIR/P_TAG OUT_DIR
"""
import json
import os
import sys

import numpy as np

pspec, out = sys.argv[1:3]
pd, pt = os.path.split(pspec)
J = json.load(open(os.path.join(pd, "collapse_%s.json" % pt)))
F = np.load(os.path.join(pd, "collapse_%s_fields.npz" % pt), allow_pickle=True)
n = 2048
rs = float(J["args"]["rs"])
for case in ("crop", "pert"):
    d = os.path.join(out, case); os.makedirs(d, exist_ok=True)
    Jx = json.loads(json.dumps(J)); Jx["args"]["nsites"] = n
    arrs = {}
    for k in F.files:
        x = F[k]
        if k == "R":
            arrs[k] = x[:n]
        elif x.dtype.kind != "O" and x.ndim == 2 and x.shape[1] == len(F["R"]):
            arrs[k] = x[:, :n].copy()
        elif x.dtype.kind == "O" and x.ndim == 2 and k in ("r_nodes", "N_nodes", "v_nodes"):
            keep = np.asarray(F["r_nodes"][0], float) <= n
            arrs[k] = x[:, keep]
        else:
            arrs[k] = x
    if case == "pert":
        for r in Jx["records"]:
            if abs(r["T"] / rs - 6.0) < 1e-9:
                r["E_over_E0"] = r["E_over_E0"] + 1e-6
        i8 = int(np.argmin(np.abs(np.asarray(F["T"], float) / rs - 8.0)))
        arrs["v"][i8, :10] += 1e-4
    json.dump(Jx, open(os.path.join(d, "collapse_%s.json" % pt), "w"))
    np.savez(os.path.join(d, "collapse_%s_fields.npz" % pt), **arrs)
    print("written %s" % d)
