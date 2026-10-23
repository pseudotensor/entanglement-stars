#!/usr/bin/env python
"""ls_finefit_read_v1.py (busi, inbox #449): the fine-K fits of the linear rate response rho1(K) (ls_zone_v1 mode D, the *_Dfine.json
files). Two forms, least squares on the seven K values per direction:
  analytic : rho1 = c1 K + c3 K^3 + c5 K^5
  log      : rho1 = c1 K + K^3 (a ln K + b) + c5 K^5
c1 is the two-derivative (a^-2) rate coefficient per unit d_t v. Prints the table and writes <out>/ls_finefit_read_v1.json."""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="<internal>/zone")
ap.add_argument("--out", default="<internal>/localsea")
a = ap.parse_args()
res = {}
for tag in ("m0p2", "m0"):
    r = json.load(open(os.path.join(a.dir, "ls_zone_v1_%s_Dfine.json" % tag)))
    res[tag] = {}
    for d, x in r["D"].items():
        k = np.asarray(x["k"], float); y = np.asarray(x["rho1"], float); ab = np.asarray(x["abs_integrand"], float)
        if np.max(np.abs(y)) < 1e-15 * np.max(ab):
            res[tag][d] = {"identically_zero": True, "max_abs_rho1": float(np.max(np.abs(y))), "max_abs_integrand": float(np.max(ab))}
            print(tag, d, "rho1 identically zero (max |rho1| %.1e against int |integrand| %.1e)" % (np.max(np.abs(y)), np.max(ab)))
            continue
        An = np.stack([k, k ** 3, k ** 5], 1); cn, *_ = np.linalg.lstsq(An, y, rcond=None)
        Al = np.stack([k, k ** 3 * np.log(k), k ** 3, k ** 5], 1); cl, *_ = np.linalg.lstsq(Al, y, rcond=None)
        rn = float(np.max(np.abs(An @ cn - y) / np.abs(y))); rl = float(np.max(np.abs(Al @ cl - y) / np.abs(y)))
        res[tag][d] = {"k": k.tolist(), "rho1_over_k3": (y / k ** 3).tolist(), "abs_over_int": (ab / np.abs(y)).tolist(),
                       "analytic_c1_c3_c5": cn.tolist(), "analytic_max_rel_resid": rn,
                       "log_c1_a_b_c5": cl.tolist(), "log_max_rel_resid": rl}
        print(tag, d, "rho1/K^3", " ".join("%.6f" % v for v in y / k ** 3))
        print("   analytic c1 %.3e c3 %.6f c5 %.4f (max rel resid %.1e)" % (cn[0], cn[1], cn[2], rn))
        print("   log      c1 %.3e a(ln K) %.6f b %.6f c5 %.4f (max rel resid %.1e)" % (cl[0], cl[1], cl[2], cl[3], rl))
        print("   int |integrand| / |int| max %.6f" % float(np.max(ab / np.abs(y))))
with open(os.path.join(a.out, "ls_finefit_read_v1.json"), "w") as f:
    json.dump(res, f, indent=1)
