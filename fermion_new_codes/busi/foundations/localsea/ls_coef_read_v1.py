#!/usr/bin/env python
"""ls_coef_read_v1.py (busi, inbox #447): reads the ls_zone_v1 and ls_exact_v1 jsons and prints the derived numbers with their
error estimates (box; light post-processing only). Writes <out>/ls_coef_read_v1.json and a markdown block to stdout.

Derived (per mass; reviewer A's basis A_sea^(2) = (N/a^2)[s1 (trK)^2 + s2 trK^2 + s3 sum K_ii^2 + sb tr Om^T Om + sc |grad ln N|^2]):
 route A (closed form + the exact identity s1 = TB; see the delivery): s1 = TB, s2 = TA + TB, s3 = T1111 - TA - 2 TB, sb = TA - TB,
          sc = 0 (the clock two-point is O(q^4));
 route C (three-point, Richardson in t^2): s1 = beta1/2, s2 = (beta2 + beta4)/2, sb = (beta4 - beta2)/2, s3 = beta6/2;
          the undifferentiated-flow structures beta3, beta5, beta7 and the identity's residual beta1 - beta2;
 kappa/g = 4 pi (3 s1 - s2 + s3/5) (reviewer A found3 sec. 2.3 item 7, reviewer B found4 sec. 2.2);
 the sea's induced coupling: A_eff = A_laws - A_sea^(2), so 1/(16 pi g_eff) = 1/(16 pi g) + s_EH with s_EH the coefficient of
          +N Phi/a^2, Phi = (trK)^2 - trK^2, in A_sea (reviewer A's sigma = -s_EH/a^2); conventions for s_EH: s1; -s2; (s1 - s2)/2
          (least squares on (1, -1, 0)); and the cubic channels A1g (3/2)(s1 + (s2 + s3)/3), Eg -(s2 + s3), T2g -s2;
 kappa' = kappa - 16 pi g s_EH (the remainder after the Einstein part is put in G).
The entropy-match window 1/g = 4 eta in (3.27, 4.08) and the match point 1/g = 1.917^2 (paper Sec. III F)."""
import argparse
import glob
import json
import math
import os

import numpy as np


def last_A(r):
    ns = sorted(int(k) for k in r["A"])
    a = r["A"][str(ns[-1])]
    if len(ns) < 2:
        return a, {k: float("nan") for k in a}, [ns[-1], ns[-1]]
    b = r["A"][str(ns[-2])]
    err = {k: abs(a[k] - b[k]) for k in a}
    return a, err, ns


def derived(A):
    TA, TB, T4 = A["TA"], A["TB"], A["T1111"]
    return {"s1": TB, "s2": TA + TB, "s3": T4 - TA - 2 * TB, "sb": TA - TB, "sc": 0.0, "X_L": 2 * TB, "X_T": TA}


def derived_err(Aerr):
    eA, eB, e4 = Aerr["TA"], Aerr["TB"], Aerr["T1111"]
    return {"s1": eB, "s2": eA + eB, "s3": e4 + eA + 2 * eB, "sb": eA + eB}


def from_C(beta):
    b = np.asarray(beta)
    return {"s1": b[0] / 2, "s2": (b[1] + b[3]) / 2, "sb": (b[3] - b[1]) / 2, "s3": b[5] / 2,
            "beta3": b[2], "beta5": b[4], "beta7": b[6], "beta1_minus_beta2": b[0] - b[1]}


def quantities(s, g):
    k_over_g = 4 * math.pi * (3 * s["s1"] - s["s2"] + s["s3"] / 5)
    conv = {"s1": s["s1"], "minus_s2": -s["s2"], "ls_half_s1_minus_s2": (s["s1"] - s["s2"]) / 2,
            "chan_A1g": 1.5 * (s["s1"] + (s["s2"] + s["s3"]) / 3), "chan_Eg": -(s["s2"] + s["s3"]), "chan_T2g": -s["s2"]}
    out = {"kappa_over_g": k_over_g, "kappa_at_g": {str(x): k_over_g * x for x in g},
           "s_EH": conv, "16pi_s_EH": {k: 16 * math.pi * v for k, v in conv.items()},
           "kappa_prime_over_g": {k: k_over_g - 16 * math.pi * v for k, v in conv.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="<internal>/localsea")
    ap.add_argument("--out", default="<internal>/localsea")
    a = ap.parse_args()
    Z = os.path.join(a.dir, "zone")
    g_list = [0.25, 1 / 1.917 ** 2]
    res = {"g_list": g_list, "window_inv_g": [3.27, 4.08], "masses": {}}
    files = sorted(glob.glob(os.path.join(Z, "ls_zone_v1_*.json")))
    by_m = {}
    for f in files:
        r = json.load(open(f))
        by_m.setdefault(r["m"], []).append((os.path.basename(f), r))
    for m, lst in sorted(by_m.items()):
        rec = {"files": [x[0] for x in lst]}
        # route A at the finest grid available among the files
        bestA = None
        for name, r in lst:
            if "A" in r:
                A, Aerr, ns = last_A(r)
                if bestA is None or (ns[-1] > bestA[2][-1] and ns[-2] < ns[-1]) or (bestA[2][-2] == bestA[2][-1] and ns[-2] < ns[-1]):
                    bestA = (A, Aerr, ns, name)
        if bestA:
            A, Aerr, ns, name = bestA
            s = derived(A); se = derived_err(Aerr)
            rec["A"] = {"source": name, "n": ns[-1], "n_prev": ns[-2], "tensors": A, "tensor_err": Aerr, "s": s, "s_err": se,
                        "derived": quantities(s, g_list)}
        # route B (finite q fits) against route A
        for name, r in lst:
            if "B" in r and bestA:
                dev = {k: abs(v["fit_c2_c4_c6"][0] - v["pred_A"]) / abs(v["pred_A"]) for k, v in r["B"].items() if v.get("pred_A")}
                rec["B"] = {"source": name, "max_rel_dev_q2_coefficient": max(dev.values()), "per_direction": dev,
                            "clock_E2_over_t2": r["B"]["clock100"]["E2_over_t2"], "clock_t": r["B"]["clock100"]["t"]}
        # route C (three-point), every n3 with its Richardson value
        cs = []
        for name, r in lst:
            if "C" in r:
                fits = r["C"]["fits"]
                ts = r["C"]["t"]
                cs.append({"source": name, "n3": r["C"]["n3"], "t": ts,
                           "richardson": from_C(fits["richardson_t01"]),
                           "smallest_t": from_C(fits[str(ts[0])]["beta"]), "rel_resid_smallest_t": fits[str(ts[0])]["rel_resid"],
                           "K0_check_rel": r["C"]["K0_check"][0]["rel"]})
        if cs:
            cs.sort(key=lambda x: x["n3"])
            rec["C"] = cs
        # route D (linear rate)
        for name, r in lst:
            if "D" in r:
                ks = None
                dd = {}
                for lab, v in r["D"].items():
                    k = np.asarray(v["k"]); y = np.asarray(v["rho1"]); yabs = np.asarray(v["abs_integrand"])
                    ks = k.tolist()
                    if np.max(np.abs(y)) < 1e-18:
                        dd[lab] = {"max_abs": float(np.max(np.abs(y))), "note": "transverse: zero by symmetry"}
                        continue
                    # models: y = c1 k + c3 k^3 + c5 k^5 (m > 0), y = c1 k + k^3 (a ln k + b) + c5 k^5 (m = 0)
                    A1 = np.stack([k, k ** 3, k ** 5], 1)
                    c_an = np.linalg.lstsq(A1, y, rcond=None)[0]
                    A2 = np.stack([k, k ** 3 * np.log(k), k ** 3, k ** 5], 1)
                    c_log = np.linalg.lstsq(A2, y, rcond=None)[0] if len(k) >= 5 else None
                    dd[lab] = {"k": ks, "rho1": y.tolist(), "rho1_over_k3": (y / k ** 3).tolist(),
                               "abs_over_signed": (yabs / np.abs(y)).tolist(), "fit_analytic_c1_c3_c5": c_an.tolist(),
                               "fit_log_c1_a_b_c5": (c_log.tolist() if c_log is not None else None),
                               "c1_over_y_smallest_k_times_k": float(c_an[0] * k[0] / y[0])}
                rec.setdefault("D", {})[name] = dd
        res["masses"][str(m)] = rec
    # the two fields together (heavy ma = 0.2 and light m = 0) where both have route A
    if "0.2" in res["masses"] and "0.0" in res["masses"] and "A" in res["masses"]["0.2"] and "A" in res["masses"]["0.0"]:
        s2f = {k: res["masses"]["0.2"]["A"]["s"][k] + res["masses"]["0.0"]["A"]["s"][k] for k in ("s1", "s2", "s3", "sb", "sc")}
        res["two_fields"] = {"s": s2f, "derived": quantities(s2f, g_list)}
    # exact-diagonalization second route
    E = os.path.join(a.dir, "exact")
    ex = {}
    for f in sorted(glob.glob(os.path.join(E, "ls_exact_v1_*.json"))):
        r = json.load(open(f))
        md = r["mode"]
        if md in ("two", "three"):
            ex[os.path.basename(f)] = {"max_rel": max(x["rel"] for x in r[md]), "n": len(r[md])}
        elif md == "rate1":
            ex[os.path.basename(f)] = {"max_rel_im": max(x["rel_im"] for x in r[md] if abs(x["i_R_formula"][1]) > 1e-12),
                                       "onsager_rho1_minus_vh_dot_J1": max(abs(sum(x["vh"][i] * x["J1_K_exact_from_Ndot"][i][1] for i in range(3))
                                                                                   - x["rho1_K_exact"][1]) for x in r[md])}
        elif md == "evolve":
            ex[os.path.basename(f)] = [{"K": x["K"], "vh": x["vh"], "R": x["R"], "ratio_im": [y["ratio_im"] for y in x["records"]],
                                        "dev_K": [y["dev_K"] for y in x["records"]]} for x in r[md]]
        elif md == "rate2":
            red = {}
            for j, v in r["rate2"].items():
                red[j] = {"q": v["q"], "Lx": r["Lx"], "ny": r["ny"], "nz": r["nz"]}
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        red[j][kk] = {k3: v3 for k3, v3 in vv.items() if k3 != "harm"}
            ex[os.path.basename(f)] = red
    res["exact"] = ex
    with open(os.path.join(a.out, "ls_coef_read_v1.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    # markdown summary
    for m, rec in res["masses"].items():
        if "A" not in rec:
            continue
        s, se = rec["A"]["s"], rec["A"]["s_err"]
        print("## m = %s (route A, n = %d vs %d)" % (m, rec["A"]["n"], rec["A"]["n_prev"]))
        for k in ("s1", "s2", "s3", "sb"):
            print("  %s = %.10e  +- %.1e" % (k, s[k], se[k]))
        if "C" in rec:
            c = rec["C"][-1]["richardson"]
            print("  route C (n3 %d): s1 %.9e s2 %.9e s3 %.9e sb %.9e | beta3 %.1e beta5 %.1e beta7 %.1e beta1-beta2 %.1e" % (
                rec["C"][-1]["n3"], c["s1"], c["s2"], c["s3"], c["sb"], c["beta3"], c["beta5"], c["beta7"], c["beta1_minus_beta2"]))
        d = rec["A"]["derived"]
        print("  kappa/g %.6f ; kappa(g=1/4) %.6f ; 16 pi s_EH:" % (d["kappa_over_g"], d["kappa_at_g"]["0.25"]),
              " ".join("%s %.4f" % (k, v) for k, v in d["16pi_s_EH"].items()))
    if "two_fields" in res:
        d = res["two_fields"]["derived"]
        print("## both fields: kappa/g %.6f ; 16 pi s_EH:" % d["kappa_over_g"], " ".join("%s %.4f" % (k, v) for k, v in d["16pi_s_EH"].items()))
    print(json.dumps(res["exact"], indent=0, default=float)[:3000])


if __name__ == "__main__":
    main()
