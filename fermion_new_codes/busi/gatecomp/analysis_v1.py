#!/usr/bin/env python
"""analysis_v1.py (busi, GATECOMP (iii)). Reads gate_profiles_v1's JSONs, birth_v1's JSON and the dust comparator JSON/NPZs; computes
(1) T_gate per member, diagnostic and zone: the first crossing of 0.1, linear between the bracketing records (labelled '<' when the first
record read is already above); the window T_gate - T_birth; (2) T_c on the stored records three ways: the caustic law
T_gate(sigma|H|) + (20/3) sigma/r_s; a linear fit of 1/max|H| against T over the last four records at or before the birth (its slope over
the homogeneous free fall's -(3/2) r_s printed); the per-record asymptote T + 1/((3/2) r_s max|H|) at the last record before the birth;
(3) T_c - T_birth on the dust comparators (the comparator's caustic = first pin at a trapped centre, and the clocks at which 10 and 50
percent of the count is pinned); (4) the caustic law across the r_s/a 64 triple, T_gate = T_c - c sigma/r_s, c against 20/3; (5) the sigma^2
law at r_s/a 64: T and R against (sigma/r_s)^2, R in units of s = 2GE_0 (and of the record's 2GE), three fits; (6) the derivative-step
check h32 against h64. Prints text tables and writes one JSON. Nothing is ruled by the script."""
import argparse
import json

import numpy as np

DIAGS = [("sigma_H", "centroid"), ("sigma_H", "reviewa"), ("sigma_Kop", "centroid"), ("sigma_Kop", "reviewa"), ("sigma_six", "centroid"),
         ("sigma_six", "reviewa"), ("sigma_Kop", "root"), ("sigma_six", "root"), ("sigma_supp", "reviewa")]


def series(rows, q, zone, h="h32"):
    T, y, r = [], [], []
    for w in rows:
        z = w["results"][h]["zones"].get(zone)
        if z:
            T.append(w["T_over_rs"]); y.append(z[q][0]); r.append(z[q][1])
    return np.array(T), np.array(y), np.array(r)


def crossing(T, y, level=0.1):
    above = np.nonzero(y > level)[0]
    if not len(above):
        return None, "never above in the records read"
    i = int(above[0])
    if i == 0:
        return float(T[0]), "already above at the first record read (<)"
    return float(T[i - 1] + (T[i] - T[i - 1]) * (level - y[i - 1]) / (y[i] - y[i - 1])), "linear between %.4f and %.4f" % (T[i - 1], T[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True, help="label=path,...")
    ap.add_argument("--birth", required=True, help="label=index into birth_v1.json list,... plus --birth-json")
    ap.add_argument("--birth-json", required=True)
    ap.add_argument("--dust", required=True, help="label=path-stem (json+npz),...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = json.load(open(a.birth_json)); births = {k: B[int(v)] for k, v in (x.split("=") for x in a.birth.split(","))}
    res = {"members": {}, "dust": {}}
    for lab, path in (x.split("=") for x in a.gates.split(",")):
        G = json.load(open(path)); rows = sorted(G["rows"], key=lambda w: w["T_over_rs"]); meta = G["meta"]
        rs, sig = meta["rs"], meta["sigma"]; b = births[lab]; Tb = b["T_birth"]
        m = {"rs": rs, "sigma": sig, "sigma_over_rs": sig / rs, "T_birth": Tb, "T_birth_err": b["T_birth_err"], "R_birth": b["R_birth"],
             "R_birth_err": b["R_birth_err"], "two_gE0": b["two_gE0"], "R_birth_over_s": b["R_birth"] / b["two_gE0"], "R_birth_over_2gE": b["R_birth_over_2gE"],
             "first_root": b["first_root_record"], "gates": {}}
        for q, zn in DIAGS:
            T, y, r = series(rows, q, zn)
            tg, how = crossing(T, y)
            m["gates"]["%s|%s" % (q, zn)] = {"T_gate": tg, "how": how, "window": (tg - Tb) if tg is not None else None}
        # T_c on the stored records
        T, yH, rH = series(rows, "sigma_H", "reviewa")
        Hmax = yH / sig
        tgH = m["gates"]["sigma_H|centroid"]["T_gate"]; tgHf = m["gates"]["sigma_H|reviewa"]["T_gate"]
        tc = {"law_centroid": tgH + (20.0 / 3.0) * sig / rs if tgH is not None else None,
              "law_reviewa": tgHf + (20.0 / 3.0) * sig / rs if tgHf is not None else None}
        sel = np.nonzero(T <= Tb + 1e-9)[0][-4:]
        p = np.polyfit(T[sel], 1.0 / Hmax[sel], 1)
        tc["fit4_maxH"] = float(-p[1] / p[0]); tc["fit4_slope_over_minus_1p5rs"] = float(p[0] / (-1.5 * rs)); tc["fit4_records"] = T[sel].tolist()
        i = sel[-1]; tc["asymptote_last_before_birth"] = float(T[i] + 1.0 / (1.5 * rs * Hmax[i])); tc["asymptote_record"] = float(T[i])
        # the centre alone
        H0 = np.array([w["results"]["h32"]["centre"]["H0"] for w in rows]); T0 = np.array([w["T_over_rs"] for w in rows])
        sel0 = np.nonzero(T0 <= Tb + 1e-9)[0][-4:]
        p0 = np.polyfit(T0[sel0], 1.0 / np.abs(H0[sel0]), 1)
        tc["fit4_centreH"] = float(-p0[1] / p0[0]); tc["fit4_centre_slope_over_minus_1p5rs"] = float(p0[0] / (-1.5 * rs))
        m["T_c"] = tc; m["T_c_minus_T_birth"] = {k: (v - Tb) for k, v in tc.items() if k in ("law_centroid", "law_reviewa", "fit4_maxH", "asymptote_last_before_birth", "fit4_centreH") and v is not None}
        # h32 against h64
        worst = 0.0
        for w in rows:
            for zn in ("centroid", "reviewa", "root"):
                z1 = w["results"]["h32"]["zones"].get(zn); z2 = w["results"]["h64"]["zones"].get(zn)
                if z1 and z2:
                    for q in ("sigma_H", "sigma_Kop", "sigma_six"):
                        worst = max(worst, abs(z1[q][0] - z2[q][0]) / abs(z1[q][0]))
        m["h32_vs_h64_max_rel"] = worst
        # the record table near birth
        tab = []
        for w in rows:
            if Tb - 1.05 <= w["T_over_rs"] <= Tb + 0.65:
                zz = w["results"]["h32"]["zones"]
                tab.append({"T": w["T_over_rs"], "certified": w["record_certified"], "R_t": (w["dense_zeros"][-1] if w["dense_zeros"] else None), "R_c": w["record"]["R_c"],
                            "min": w["dense_min"], **{"%s|%s" % (q, zn): (zz[zn][q] if zz.get(zn) else None) for q, zn in DIAGS},
                            "six_dom_reviewa": zz["reviewa"]["six_dominant"] if zz.get("reviewa") else None, "centre_six": w["results"]["h32"]["centre"]["sigma_six_r0"],
                            "centre_H0": w["results"]["h32"]["centre"]["H0"], "fine_min": w["results"]["h32"]["fine_min_N_minus_absv"]})
        m["table"] = tab
        res["members"][lab] = m
    for lab, stem in (x.split("=") for x in a.dust.split(",")):
        j = json.load(open(stem + ".json")); z = np.load(stem + ".npz"); rs = float(j["meta"].get("rs", 0) or 0) or None
        rs = 128.0 if "128" in stem or "iso" in lab.lower() else 64.0
        lt = z["ledger_t"] / rs; o = np.argsort(z["shell_R0"]); n = z["shell_n"][o]; c = np.cumsum(n) / np.sum(n)
        qs = {str(x): float(lt[min(int(np.searchsorted(c, x)), len(lt) - 1)]) for x in (0.1, 0.5)}
        Tf = j["fold"]["T_over_rs"]; Tc = j["centre"]["caustic"]["T_over_rs"]
        res["dust"][lab] = {"sigma": j["spec"]["sigma"], "rs": rs, "T_birth": Tf, "R_birth": j["fold"]["R"], "R_birth_over_s": j["fold"]["R_over_2gE0"],
                            "two_gE0": j["two_gE0"], "T_c_caustic": Tc, "T_c_pinned_10pct": qs["0.1"], "T_c_pinned_50pct": qs["0.5"],
                            "T_c_minus_T_birth": Tc - Tf, "T_c10_minus_T_birth": qs["0.1"] - Tf, "T_c50_minus_T_birth": qs["0.5"] - Tf,
                            "first_record": j["first_record"]["T_over_rs"], "first_record_root": j["first_record"]["outer"]}
    # (4) the caustic law across the 64 triple
    trip = [k for k in ("P64_s2", "P64_s4", "P64_s8") if k in res["members"]]
    if len(trip) == 3:
        law = {}
        for key in ("sigma_H|centroid", "sigma_H|reviewa", "sigma_Kop|reviewa", "sigma_six|reviewa"):
            x = np.array([res["members"][k]["sigma_over_rs"] for k in trip]); y = np.array([res["members"][k]["gates"][key]["T_gate"] for k in trip])
            p = np.polyfit(x, y, 1); law[key] = {"T_c": float(p[1]), "c": float(-p[0]), "c_over_20_3": float(-p[0] / (20.0 / 3.0)), "T_gate": y.tolist(),
                                                 "resid": (y - np.polyval(p, x)).tolist()}
        res["caustic_law_64"] = law
        # (5) the sigma^2 law
        sq = {}
        x = np.array([res["members"][k]["sigma_over_rs"] ** 2 for k in trip])
        for name, ys in (("T_birth", [res["members"][k]["T_birth"] for k in trip]), ("R_birth_over_s", [res["members"][k]["R_birth_over_s"] for k in trip]),
                         ("R_birth_over_2gE", [res["members"][k]["R_birth_over_2gE"] for k in trip]),
                         ("T_first_root", [res["members"][k]["first_root"]["T"] for k in trip]),
                         ("R_first_root_over_s", [res["members"][k]["first_root"]["R_t"] / res["members"][k]["two_gE0"] for k in trip]),
                         ("R_first_root_over_2gE", [res["members"][k]["first_root"]["R_t"] / res["members"][k]["first_root"]["two_gE"] for k in trip])):
            y = np.array(ys, float)
            c24 = (y[1] - y[0]) / (x[1] - x[0]); y0_24 = y[0] - c24 * x[0]; dep8 = y[2] - (y0_24 + c24 * x[2])
            c28 = (y[2] - y[0]) / (x[2] - x[0]); y0_28 = y[0] - c28 * x[0]; dep4 = y[1] - (y0_28 + c28 * x[1])
            p = np.polyfit(x, y, 1)
            sq[name] = {"values": y.tolist(), "x": x.tolist(), "fit_2a4a": {"c": c24, "y0": y0_24, "departure_8a": dep8},
                        "fit_2a8a": {"c": c28, "y0": y0_28, "departure_4a": dep4}, "lsq3": {"c": float(p[0]), "y0": float(p[1]), "resid": (y - np.polyval(p, x)).tolist()}}
        res["sigma2_64"] = sq
    json.dump(res, open(a.out, "x"), indent=1, default=float)
    # text
    for lab, m in res["members"].items():
        print("\n== %s: r_s %g, sigma %g (sigma/r_s %.5f); T_birth %.4f +- %.4f, R_birth %.3f +- %.3f a (%.4f of s = 2GE_0 %.3f; %.4f of the record's 2GE); first root %.4f at %.3f a" % (
            lab, m["rs"], m["sigma"], m["sigma_over_rs"], m["T_birth"], m["T_birth_err"], m["R_birth"], m["R_birth_err"], m["R_birth_over_s"], m["two_gE0"],
            m["R_birth_over_2gE"], m["first_root"]["T"], m["first_root"]["R_t"]))
        for k, v in m["gates"].items():
            print("  T_gate %-22s %s  window %s  (%s)" % (k, ("%.4f" % v["T_gate"]) if v["T_gate"] is not None else "-", ("%+.4f" % v["window"]) if v["window"] is not None else "-", v["how"]))
        print("  T_c: %s" % json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m["T_c"].items()}))
        print("  T_c - T_birth: %s" % json.dumps({k: round(v, 4) for k, v in m["T_c_minus_T_birth"].items()}))
        print("  h32 vs h64 max rel %.2e" % m["h32_vs_h64_max_rel"])
    for lab, d in res["dust"].items():
        print("dust %s: %s" % (lab, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})))
    if "caustic_law_64" in res:
        print("caustic law 64: %s" % json.dumps(res["caustic_law_64"], default=float))
        print("sigma^2 64: %s" % json.dumps(res["sigma2_64"], default=float))


if __name__ == "__main__":
    main()
