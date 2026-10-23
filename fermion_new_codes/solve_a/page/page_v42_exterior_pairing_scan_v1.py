#!/usr/bin/env python
"""PAGE v4.2 issue 1 companion (inbox #205): for every stored exterior spectrum eig_X_{plus,minus}_cut<c> of the B2 production channels
kappa 1..18 (inside [0, 1] or not), the deviation from the Gram pairing with the same checkpoint's interior spectrum (eig_X = 1 - [eig_Xc, 0...]
for orthonormal columns), the trace defect, and the entropy difference S(eig_X) - S(eig_Xc); lists every spectrum whose pairing deviation
exceeds 1e-6. Stored data only. Usage: python page_v42_exterior_pairing_scan_v1.py PROD_DIR OUT_JSON"""
import sys, json, re, os
import numpy as np

PROD = sys.argv[1]; OUT = sys.argv[2]


def S(x):
    x = np.clip(np.asarray(x, float), 0, 1); m = (x > 0) & (x < 1); y = x[m]
    return float((-y * np.log(y) - (1 - y) * np.log1p(-y)).sum())


rows = []
for K in range(1, 19):
    d = np.load(os.path.join(PROD, "v6o_B2_k%d_d13.npz" % K))
    for key in d.files:
        m_ = re.match(r"T(\d+\.\d+)_k(\d+)_eig_X_(plus|minus)_cut(\d+)$", key)
        if not m_:
            continue
        ev = d[key]; evc = d["T%s_k%d_eig_Xc_%s_cut%s" % (m_.group(1), K, m_.group(3), m_.group(4))]
        paired = np.sort(1.0 - np.concatenate([evc, np.zeros(ev.size - evc.size)]))
        dev = float(np.abs(np.sort(ev) - paired).max())
        rows.append({"K": K, "T": float(m_.group(1)), "sector": m_.group(3), "cut": m_.group(4), "pair_dev": dev,
                     "trace_defect": float(ev.sum() + evc.sum() - ev.size), "dS": S(ev) - S(evc), "max": float(ev.max()), "min": float(ev.min())})
bad = [r for r in rows if r["pair_dev"] > 1e-6]
devs = np.array([r["pair_dev"] for r in rows if r["pair_dev"] <= 1e-6])
out = {"n_spectra": len(rows), "n_pair_dev_gt_1e-6": len(bad), "bad": bad,
       "good_pair_dev_max": float(devs.max()), "good_pair_dev_median": float(np.median(devs)),
       "good_abs_dS_max": float(max(abs(r["dS"]) for r in rows if r["pair_dev"] <= 1e-6)),
       "clocks": sorted(set(r["T"] for r in rows))}
json.dump(out, open(OUT, "w"), indent=1)
print("spectra %d; pairing deviation > 1e-6: %d; others: max dev %.3e median %.3e max |dS| %.3e; clocks %s" % (
    len(rows), len(bad), out["good_pair_dev_max"], out["good_pair_dev_median"], out["good_abs_dS_max"], out["clocks"]))
for r in sorted(bad, key=lambda r: (r["T"], r["cut"], r["K"])):
    print("K %2d T %5.1f %s cut %s pair_dev %.3e trace_defect %.3e dS %.6e max %.17g" % (r["K"], r["T"], r["sector"], r["cut"], r["pair_dev"], r["trace_defect"], r["dS"], r["max"]))
