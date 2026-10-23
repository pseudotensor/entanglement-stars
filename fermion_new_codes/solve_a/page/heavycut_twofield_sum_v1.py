#!/usr/bin/env python3
"""heavycut_twofield_sum_v1.py: compose the heavy field's exterior-cut excess and the light channel's K = 18 excess at
matched clocks, and their sum, FROM STORED FILES ONLY (no propagation, no spectrum, no slice read).

REORIENT_6 item 9 / PAGE v4.1 open item (v). Runs on the box CPU (light parsing; seconds).

Inputs (read only):
  heavy, B2 history (4096 sites, chain end 64 r_s): busi page_own_object_v1  slices144/out/page/page_B2_T*.json  (cuts 16, 24 r_s)
  heavy, prod chain (2048 sites, chain end 32 r_s): busi W8 heavy_cut_entropy_v1  slices144/out/heavycut/HEAVYCUT_W8_v1_all.json  (cuts 2, 16, 24)
  heavy, s8 chain (2048 sites): the same W8 json
  heavy, prod (page_own_object on prod): slices144/out/page/page_prod_T*.json (cross-check of W8)
  light, B2 and s8: page_v4/out_v41/page_v4_data_v1.json (PAGE v4.1 producer output; K = 18, weights 16 kappa)
  B2 record: B2_final/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json (md5 417a4018...), the light's history

Definitions carried unchanged from the producers:
  heavy excess  = sum_kappa 16 kappa [S_X(C) - S_X(P_-)]  over the slice's channels (10), cut X = {R > x}, R = n + 1, x = cut * 64
  light excess  = sum_{kappa=1..18} 16 kappa [S_X(C) - S_X(P_-(h))], same cut set X = {R > x}, x = cut * 64 (cut_R_abs)
  two-field sum = heavy + light (the product-state identity: the light is first order, its sources not fed back)
Floors:
  heavy F_H = sum_kappa 16 kappa |excess_kappa - excess_c_kappa| (the complement route; both states pure so they agree up to numerics),
            plus, on a complex64 slice, the slice's stated relative floor (8.7e-6) times the raw S_X (the subtraction's scale)
  heavy tail diagnostic: the kappa 10 weighted term (the last stored channel), not a bound
  light F_meas (propagation floor) and K_c = 18 c (common mode's share), copied from the v4.1 readout rows
"""
import glob, hashlib, json, math, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "<internal>/HEAVYCUT_v1"
PAGE = "<internal>/page"
W8 = "<internal>/HEAVYCUT_W8_v1_all.json"
LIGHT = "<internal>/page_v4_data_v1.json"
REC = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"
CUTS = ["2", "16", "24", "30", "48"]
TOL = 0.0625
C64_REL = 8.7e-6


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fl(x):
    return None if x is None else float(x)


# ---------------- light ----------------
L = json.load(open(LIGHT))
cut_order = [str(c) for c in L["cuts"]]
light = {}
for hist in ("B2", "s8"):
    B = L[hist]
    clocks = [float(t) for t in B["clocks"]]
    SUM = np.array(B["SUM"], float)
    rows = {}
    for i, t in enumerate(clocks):
        for j, c in enumerate(cut_order):
            rows[(t, c)] = {"S": float(SUM[i, j]), "F_meas": None, "K_c": None, "verdict": None, "source": "SUM array (no readout row)"}
    for r in B["readout"]:
        k = (float(r["t"]), str(r["cut"]))
        sumv = rows[k]["S"] if k in rows else None
        rows[k] = {"S": float(r["S"]), "F_meas": fl(r["F_meas"]), "K_c": 18.0 * float(r["c"]) if r.get("c") is not None else None,
                   "verdict": r.get("verdict"), "source": r.get("source"), "readout_minus_SUM": None if sumv is None else float(r["S"]) - sumv}
    light[hist] = {"clocks": clocks, "rows": rows,
                   "E": B.get("E"), "R_t": B.get("R_t")}


def light_at(hist, t, c):
    best = None
    for (tt, cc), v in light[hist]["rows"].items():
        if cc == c and abs(tt - t) <= TOL + 1e-9:
            if best is None or abs(tt - t) < abs(best[0] - t):
                best = (tt, v)
    return best


# ---------------- heavy: B2 (page_own_object) ----------------
heavy = {"B2": {}, "prod": {}, "s8": {}}
meta = {"B2": {}, "prod": {}, "s8": {}}
inputs = []
for p in sorted(glob.glob(os.path.join(PAGE, "page_B2_T*.json"))):
    d = json.load(open(p)); inputs.append(p)
    t = float(d["T_over_rs"]); c64 = bool(d.get("complex64"))
    meta["B2"][t] = {"E_over_E0": d["record"]["E_over_E0"], "R_t": d["record"]["R_t"], "dtype": d["dtype"], "tag": d["tag"],
                     "first_return": {k: v["first_return_rs"] for k, v in d["first_return"].items()}}
    for c, w in d["weighted"].items():
        F = 0.0; k10 = None; F_vac = 0.0
        for ch in d["channels"]:
            cc = ch["cuts"][c]; wt = float(ch["weight"])
            F += wt * abs(float(cc["excess"]) - float(cc["excess_c"]))
            if int(ch["kappa"]) == max(int(x["kappa"]) for x in d["channels"]):
                k10 = wt * float(cc["excess"])
        if c64:
            F += C64_REL * float(w["S_X"])
        heavy["B2"][(t, c)] = {"excess": float(w["excess"]), "S_X": float(w["S_X"]), "S_X_vac": float(w["S_X_vac"]),
                               "F_H": F, "k_last_term": k10, "purity_gap": float(w["max_abs_purity_gap"]),
                               "past_first_return": not d["first_return"][c]["before_first_return"],
                               "first_return_rs": d["first_return"][c]["first_return_rs"], "dtype": d["dtype"],
                               "n_channels": len(d["channels"])}

# ---------------- heavy: prod and s8 (W8) ----------------
W = json.load(open(W8)); inputs.append(W8)
for r in W["rows"]:
    h = r["history"]; t = float(r["T_over_rs"]); c64 = bool(r.get("complex64"))
    meta[h][t] = {"E_over_E0": r["record"]["E_over_E0"], "R_t": r["record"]["R_t"], "dtype": r["dtype"], "tag": r["tag"],
                  "formation": r["formation"], "first_return": {k: v.get("first_return_rs") for k, v in r["first_return"].items()}}
    kmax = max(int(x["kappa"]) for x in r["channels"])
    for c, w in r["weighted"].items():
        F = 0.0; k10 = None
        for ch in r["channels"]:
            cc = ch["cuts"][c]; wt = float(ch["weight"])
            F += wt * abs(float(cc["heavy_excess"]) - float(cc["heavy_excess_c"]))
            if int(ch["kappa"]) == kmax:
                k10 = wt * float(cc["heavy_excess"])
        if c64:
            F += C64_REL * float(w["S_H"])
        fr = r["first_return"][c]
        heavy[h][(t, c)] = {"excess": float(w["heavy_excess"]), "S_X": float(w["S_H"]), "S_X_vac": float(w["S_P"]),
                            "excess_sea": float(w["excess_sea"]), "S_d": float(w["S_d"]), "K_sd": float(w["K_sd"]),
                            "F_H": F, "k_last_term": k10, "purity_gap": float(w["max_abs_purity_gap_H"]),
                            "past_first_return": not fr.get("before_first_return", True),
                            "first_return_rs": fr.get("first_return_rs"), "dtype": r["dtype"], "n_channels": len(r["channels"]),
                            "all_bounds_hold": w.get("all_bounds_hold")}

# ---------------- cross-check 1: W8 prod vs page_own_object prod ----------------
chk_w8_page = []
for p in sorted(glob.glob(os.path.join(PAGE, "page_prod_T*.json"))):
    d = json.load(open(p)); inputs.append(p); t = float(d["T_over_rs"])
    for c, w in d["weighted"].items():
        hv = heavy["prod"].get((t, c))
        if hv:
            chk_w8_page.append({"t": t, "cut": c, "W8": hv["excess"], "page_own_object": float(w["excess"]),
                                "diff": hv["excess"] - float(w["excess"])})

# ---------------- cross-check 2: prod vs B2 (heavy, overlap) ----------------
chk_prod_B2 = []
for (t, c), hv in sorted(heavy["prod"].items()):
    b = heavy["B2"].get((t, c))
    if b:
        chk_prod_B2.append({"t": t, "cut": c, "prod": hv["excess"], "B2": b["excess"], "diff": hv["excess"] - b["excess"],
                            "rel": (hv["excess"] - b["excess"]) / b["excess"] if b["excess"] else None,
                            "prod_past_return": hv["past_first_return"], "B2_past_return": b["past_first_return"],
                            "floors_prod_B2": [hv["F_H"], b["F_H"]],
                            "E_over_E0_prod_B2": [meta["prod"][t]["E_over_E0"], meta["B2"][t]["E_over_E0"]],
                            "R_t_prod_B2": [meta["prod"][t]["R_t"], meta["B2"][t]["R_t"]]})

# ---------------- cross-check 3: B2 heavy slices vs the light's record (same history) ----------------
R = json.load(open(REC)); inputs.append(REC)
recs = R["records"]
chk_hist = []
for t, m in sorted(meta["B2"].items()):
    k = int(np.argmin([abs(float(x["T_over_rs"]) - t) for x in recs])); x = recs[k]
    chk_hist.append({"t": t, "record_t": float(x["T_over_rs"]), "E_over_E0_slice": m["E_over_E0"], "E_over_E0_record": x["E_over_E0"],
                     "dE": m["E_over_E0"] - float(x["E_over_E0"]), "R_t_slice": m["R_t"], "R_t_record": x["R_t"],
                     "dR_t": None if x["R_t"] is None or m["R_t"] is None else m["R_t"] - float(x["R_t"])})

# ---------------- cross-check 4: s8 heavy (W8) vs the light's s8 history ----------------
chk_s8 = []
Ls8 = light["s8"]
for t, m in sorted(meta["s8"].items()):
    if t in Ls8["clocks"]:
        i = Ls8["clocks"].index(t)
        chk_s8.append({"t": t, "R_t_heavy": m["R_t"], "R_t_light": None if Ls8["R_t"] is None else Ls8["R_t"][i],
                       "E_heavy_over_E0": m["E_over_E0"], "E_light": None if Ls8["E"] is None else Ls8["E"][i],
                       "formation_heavy": m["formation"]})

# ---------------- composition ----------------
table = []
heavy_clocks = {"B2": sorted({t for (t, c) in heavy["B2"]}), "prod": sorted({t for (t, c) in heavy["prod"]}),
                "s8": sorted({t for (t, c) in heavy["s8"]})}
for hh, lh in (("B2", "B2"), ("prod", "B2"), ("s8", "s8")):
    for t in heavy_clocks[hh]:
        for c in CUTS:
            hv = heavy[hh].get((t, c))
            lt = light_at(lh, t, c)
            row = {"heavy_history": hh, "light_history": lh, "same_chain": hh == lh, "t": t, "cut": c,
                   "heavy": hv, "light_clock": None if lt is None else lt[0], "light": None if lt is None else lt[1]}
            row["sum"] = hv["excess"] + lt[1]["S"] if (hv is not None and lt is not None) else None
            if row["sum"] is not None:
                fl_ = (hv["F_H"] or 0.0) + (lt[1]["F_meas"] or 0.0) if lt[1]["F_meas"] is not None else None
                row["sum_floor"] = fl_
            table.append(row)
# light-only rows on B2 at clocks where no heavy exists (for the record of gaps)
light_only = sorted({t for (t, c) in light["B2"]["rows"]} - set(heavy_clocks["B2"]))

os.makedirs(OUT, exist_ok=True)
res = {"script": os.path.abspath(__file__), "inputs": {p: md5(p) for p in [LIGHT, W8, REC]}, "n_heavy_json": len(inputs) - 2,
       "cut_order_light": cut_order, "cut_R_abs_light": L["cut_R_abs"], "tol_rs": TOL, "c64_rel_floor": C64_REL,
       "heavy_clocks": heavy_clocks, "light_clocks_B2": light["B2"]["clocks"], "light_only_clocks_B2": light_only,
       "table": table, "check_W8_vs_page_prod": chk_w8_page, "check_prod_vs_B2": chk_prod_B2,
       "check_B2_slices_vs_record": chk_hist, "check_s8_heavy_vs_light": chk_s8,
       "light_readout_minus_SUM_max": max(abs(v.get("readout_minus_SUM") or 0.0) for v in light["B2"]["rows"].values())}
json.dump(res, open(os.path.join(OUT, "heavycut_twofield_sum_v1.json"), "w"), indent=1, default=str)

# tsv
with open(os.path.join(OUT, "heavycut_twofield_sum_v1.tsv"), "w") as f:
    f.write("heavy_history\tlight_history\tT_over_rs\tcut_rs\theavy_excess\tF_H\theavy_k_last_term\theavy_dtype\theavy_past_first_return\t"
            "light_clock\tlight_S_K18\tlight_F_meas\tlight_K_c\tlight_verdict\tsum\tsum_floor\n")
    for r in table:
        hv = r["heavy"]; lv = r["light"]
        f.write("\t".join(str(x) for x in [r["heavy_history"], r["light_history"], r["t"], r["cut"],
                hv["excess"] if hv else "absent", hv["F_H"] if hv else "", hv["k_last_term"] if hv else "", hv["dtype"] if hv else "",
                hv["past_first_return"] if hv else "", r["light_clock"] if lv else "absent", lv["S"] if lv else "",
                lv["F_meas"] if lv else "", lv["K_c"] if lv else "", lv["verdict"] if lv else "",
                r["sum"] if r["sum"] is not None else "", r.get("sum_floor", "")]) + "\n")
print("wrote", OUT, "rows", len(table))
print("W8 vs page prod max |diff|", max(abs(x["diff"]) for x in chk_w8_page), "n", len(chk_w8_page))
for x in chk_prod_B2:
    print("prod-B2", x["t"], x["cut"], "%.6f %.6f diff %.3e rel %s ret %s/%s" % (x["prod"], x["B2"], x["diff"], x["rel"], x["prod_past_return"], x["B2_past_return"]))
print("B2 slices vs record max |dE/E0|", max(abs(x["dE"]) for x in chk_hist), "max |dR_t|", max(abs(x["dR_t"]) for x in chk_hist if x["dR_t"] is not None),
      "max |dt|", max(abs(x["t"] - x["record_t"]) for x in chk_hist))
for x in chk_s8:
    print("s8", x)
print("light readout - SUM max", res["light_readout_minus_SUM_max"])
