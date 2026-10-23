#!/usr/bin/env python
"""nodes_gate_v1.py (busi, GATECOMP (iii)). From gate_profiles_v1's JSONs: the gate on the six-term code of record's own nodes r = sigma,
2 sigma, ... (engine_core_slices_v3's convention; the profile_every_sigma arrays), over [sigma, R_c] and [sigma, 2 R_t] (R_t the birth
radius before the first root), beside the fine-grid maxima that include the regular centre r < sigma. T_gate = the first crossing of 0.1,
linear between the bracketing records. Also: the re-solved (ivp) dense outer root and min(N - |v|) against the record's stored values (the
records were written on the collocation route for job (b)), the largest differences. Prints text and writes one JSON."""
import json
import sys

import numpy as np


def cross(T, y):
    i = np.nonzero(y > 0.1)[0]
    if not len(i):
        return None
    i = int(i[0])
    return float(T[0]) if i == 0 else float(T[i - 1] + (T[i] - T[i - 1]) * (0.1 - y[i - 1]) / (y[i] - y[i - 1]))


out = {}
for arg in sys.argv[2:]:
    lab, path = arg.split("=")
    G = json.load(open(path)); rows = sorted(G["rows"], key=lambda w: w["T_over_rs"]); sig = G["meta"]["sigma"]
    T = []; res = {k: [] for k in ("six_c", "six_f", "Kop_c", "Kop_f", "H_c", "H_f", "fine_six_f", "fine_six_f_noc")}
    dRt = 0.0; dmin = 0.0
    for w in rows:
        p = w["results"]["h32"]["profile_every_sigma"]; r = np.array(p["r"]); Rc = w["record"]["R_c"]; Rt = w["R_t_used"]
        mc = (r >= sig - 1e-9) & (r <= Rc + 1e-9); mf = (r >= sig - 1e-9) & (r <= 2 * Rt + 1e-9)
        T.append(w["T_over_rs"])
        for q, key in (("sigma_six", "six"), ("sigma_Kop", "Kop"), ("sigma_H", "H")):
            a = np.array(p[q]); res[key + "_c"].append(float(a[mc].max())); res[key + "_f"].append(float(a[mf].max()))
        z = w["results"]["h32"]["zones"]["reviewa"]; res["fine_six_f"].append(z["sigma_six"][0])
        res["fine_six_f_noc"].append(max(z["six_terms"][t][0] for t in z["six_terms"]))
        if w["dense_zeros"] and w["record"]["R_t"]:
            dRt = max(dRt, abs(w["dense_zeros"][-1] - w["record"]["R_t"]))
        if w["dense_min"] is not None and w["record"]["min_N_minus_absv"] is not None:
            dmin = max(dmin, abs(w["dense_min"] - w["record"]["min_N_minus_absv"]))
    T = np.array(T)
    o = {"T": T.tolist(), **res, "T_gate": {k: cross(T, np.array(v)) for k, v in res.items()}, "max_abs_dRt_vs_record": dRt, "max_abs_dmin_vs_record": dmin}
    out[lab] = o
    print("%s: T_gate on nodes r >= sigma: six [sigma,R_c] %s, six [sigma,2R_t] %s, Kop %s / %s, H %s / %s; fine grid incl. centre: six %s; |R_t ivp - record| max %.2e a, |min| max %.2e" % (
        lab, o["T_gate"]["six_c"], o["T_gate"]["six_f"], o["T_gate"]["Kop_c"], o["T_gate"]["Kop_f"], o["T_gate"]["H_c"], o["T_gate"]["H_f"], o["T_gate"]["fine_six_f"], dRt, dmin))
    for t, a, b, c in zip(T, res["six_f"], res["Kop_f"], res["H_f"]):
        if 10.9 <= t <= 12.7:
            print("   %.4f  nodes: six %.4f  Kop %.4f  H %.4f" % (t, a, b, c))
json.dump(out, open(sys.argv[1], "x"), indent=1)
