#!/usr/bin/env python3
"""heavycut_check_v1.py: cold recomputation of every number printed by
deliveries/solve_a/HEAVYCUT_twofield_sum_v1.1.md from the stored inputs it names.

Own code (checking lane). The delivery's scripts were read only to locate the inputs and to name the
convention behind each column; nothing of them is imported or called here.

Inputs (read only, on the box):
  heavy B2   : <internal>/page_B2_T*.json   (page_own_object_v1; cuts 16, 24)
  heavy prod : <internal>/HEAVYCUT_W8_v1_all.json (W8; cuts 2, 16, 24; prod and s8)
               and the 18 per-slice heavycut_{prod,s8}_T*.json beside it
  heavy prod (page_own_object, the W8 cross-check): page_prod_T*.json
  light      : <internal>/page_v4_data_v1.json (PAGE v4.1 producer output)
  record     : <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json (+ the w1 copy)
  delivery   : <internal>/{json,tsv,tables md,MD5SUMS,code}
               and a copy of the v1.1 md (scp'd into OUT/inputs/) for the table-line diff

Outputs (OUT): heavycut_check_v1.out (this log), heavycut_check_v1.json (every recomputed value),
  heavycut_check_v1_regenerated_tables.md (the 8 tables + check table in the delivery's format, from my values),
  heavycut_check_v1_tables.diff (against the delivery's embedded table lines), heavycut_check_v1_md5.txt.

No linear algebra is needed (sums of stored scalars); the weighted sums are formed with torch float64 dot
products (8 threads) and the rest in plain Python. Nothing is propagated and no slice is opened.
"""
import glob, hashlib, json, math, os, re, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
T0 = time.time()
OUT = sys.argv[1] if len(sys.argv) > 1 else "<internal>/heavycut_v1_1"
os.makedirs(OUT, exist_ok=True)
ROOT = "<internal>/solve_a_bulk"
PAGE_DIR = ROOT + "/slices144/out/page"
HC_DIR = ROOT + "/slices144/out/heavycut"
W8 = HC_DIR + "/HEAVYCUT_W8_v1_all.json"
LIGHT = ROOT + "/page_v4/out_v41/page_v4_data_v1.json"
REC = ROOT + "/B2_final/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"
REC_W1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"
DELIV = ROOT + "/compositions/HEAVYCUT_v1"
DELIV_MD = os.path.join(OUT, "inputs", "HEAVYCUT_twofield_sum_v1.1.md")
C64_REL = 8.7e-6          # the delivery's complex64 relative floor (slice_restart_check_v1: 8.742e-6 production at 40)
CUTS_ALL = ["2", "16", "24", "30", "48"]

logf = open(os.path.join(OUT, "heavycut_check_v1.out"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")
    logf.flush()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wdot(w, v):
    """weighted sum with torch float64 (the only 'linear algebra' here)."""
    return float(torch.dot(torch.tensor(w, dtype=torch.float64), torch.tensor(v, dtype=torch.float64)))


RES = {"script": os.path.abspath(__file__), "inputs_md5": {}, "notes": []}

# ----------------------------------------------------------------------------------------------------------------
P("# heavycut_check_v1: cold recomputation of HEAVYCUT_twofield_sum_v1.1 (box, CPU, torch threads 8)")
P("## 1. inputs and md5")
page_B2 = sorted(glob.glob(PAGE_DIR + "/page_B2_T*.json"))
page_prod = sorted(glob.glob(PAGE_DIR + "/page_prod_T*.json"))
page_s8 = sorted(glob.glob(PAGE_DIR + "/page_s8_T*.json"))
hc_slices = sorted(glob.glob(HC_DIR + "/heavycut_*_T*.json"))
named = {"LIGHT": LIGHT, "W8_all": W8, "REC_B2_final": REC, "REC_w1_copy": REC_W1,
         "deliv_json": DELIV + "/heavycut_twofield_sum_v1.json", "deliv_tsv": DELIV + "/heavycut_twofield_sum_v1.tsv",
         "deliv_tables": DELIV + "/heavycut_twofield_sum_v1_tables.md",
         "deliv_code": DELIV + "/code/heavycut_twofield_sum_v1.py", "deliv_code_tables": DELIV + "/code/heavycut_twofield_sum_tables_v1.py",
         "deliv_md_v1.1_copy": DELIV_MD}
for k, p in named.items():
    RES["inputs_md5"][k] = (p, md5(p) if os.path.exists(p) else None)
    P("  %-22s %s  %s" % (k, RES["inputs_md5"][k][1], p))
RES["inputs_md5"]["page_B2"] = {os.path.basename(p): md5(p) for p in page_B2}
RES["inputs_md5"]["page_prod"] = {os.path.basename(p): md5(p) for p in page_prod}
RES["inputs_md5"]["hc_slices"] = {os.path.basename(p): md5(p) for p in hc_slices}
P("  page_B2 jsons: %d, page_prod jsons: %d, page_s8 jsons: %d, W8 per-slice jsons: %d" % (len(page_B2), len(page_prod), len(page_s8), len(hc_slices)))
deliv_md5sums = open(DELIV + "/MD5SUMS").read().split()
RES["deliv_MD5SUMS_text"] = open(DELIV + "/MD5SUMS").read()
P("  delivery MD5SUMS file says:", RES["deliv_MD5SUMS_text"].strip().replace("\n", " | "))
with open(os.path.join(OUT, "heavycut_check_v1_md5.txt"), "w") as f:
    for k, v in RES["inputs_md5"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                f.write("%s  %s/%s\n" % (vv, k, kk))
        else:
            f.write("%s  %s\n" % (v[1], v[0]))

# ----------------------------------------------------------------------------------------------------------------
P("\n## 2. light (PAGE v4.1 producer json)")
L = json.load(open(LIGHT))
cut_order = [str(c) for c in L["cuts"]]
RES["light"] = {"cuts": cut_order, "cut_R_abs": L["cut_R_abs"], "version": L.get("version"), "script": L.get("script")}
P("  version", L.get("version"), "cuts", cut_order, "cut_R_abs", L["cut_R_abs"])
light = {}
for hist in ("B2", "s8"):
    B = L[hist]
    clocks = [float(t) for t in B["clocks"]]
    W = [float(w) for w in B["W"]] if "W" in B else [16.0 * k for k in B["kappas"]]
    kap = [int(k) for k in B["kappas"]]
    x = np.array(B["x"], float)        # [kappa][clock][cut]
    WX = np.array(B["WX"], float)
    SUM = np.array(B["SUM"], float)    # [clock][cut]
    # my own recomposition of the K = 18 weighted sum
    assert kap == list(range(1, 19)), kap
    assert W == [16.0 * k for k in kap], W
    mySUM = np.zeros_like(SUM)
    for i in range(len(clocks)):
        for j in range(len(cut_order)):
            mySUM[i, j] = wdot(W, x[:, i, j])
    dWX = np.max(np.abs(WX - np.array(W)[:, None, None] * x))
    dSUM = np.max(np.abs(SUM - mySUM))
    P("  %s: clocks %d (%s), kappas 1..18, weights 16 kappa; max|WX - W x| %.2e; max|SUM - sum_k W_k x_k| %.2e" % (
        hist, len(clocks), ", ".join("%g" % t for t in clocks), dWX, dSUM))
    rows = {}
    for i, t in enumerate(clocks):
        for j, c in enumerate(cut_order):
            rows[(t, c)] = {"S": float(SUM[i, j]), "S_mine": float(mySUM[i, j]), "F_meas": None, "K_c": None, "c": None,
                            "verdict": None, "source": "SUM array (no readout row)", "readout": False}
    ro = B["readout"]
    keys = [(float(r["t"]), str(r["cut"])) for r in ro]
    assert len(keys) == len(set(keys)), "duplicate readout pairs"
    dRO = 0.0
    for r in ro:
        k = (float(r["t"]), str(r["cut"]))
        assert k in rows, k
        dRO = max(dRO, abs(float(r["S"]) - rows[k]["S"]))
        rows[k].update({"S": float(r["S"]), "F_meas": float(r["F_meas"]) if r.get("F_meas") is not None else None,
                        "c": float(r["c"]) if r.get("c") is not None else None,
                        "K_c": 18.0 * float(r["c"]) if r.get("c") is not None else None,
                        "verdict": r.get("verdict"), "source": r.get("source"), "readout": True})
    srcs = sorted(set(r["source"] for r in ro))
    P("  %s: readout rows %d, distinct pairs %d, sources %s, readout t range [%g, %g]; max|readout S - SUM| %.2e" % (
        hist, len(ro), len(set(keys)), srcs, min(k[0] for k in keys), max(k[0] for k in keys), dRO))
    light[hist] = {"clocks": clocks, "rows": rows, "E": B.get("E"), "R_t": B.get("R_t"), "dRO": dRO, "dSUM": dSUM, "dWX": dWX}
RES["light"]["B2_clocks"] = light["B2"]["clocks"]
RES["light"]["s8_clocks"] = light["s8"]["clocks"]
RES["light"]["checks"] = {h: {"readout_minus_SUM_max": light[h]["dRO"], "SUM_recomposed_max": light[h]["dSUM"], "WX_max": light[h]["dWX"]} for h in light}

# ----------------------------------------------------------------------------------------------------------------
P("\n## 3. heavy B2 (page_own_object_v1 jsons)")
heavy = {"B2": {}, "prod": {}, "s8": {}}
meta = {"B2": {}, "prod": {}, "s8": {}}
b2_tags = set(); b2_print = set(); b2_floor_applies = set(); b2_nd = {}
for p in page_B2:
    d = json.load(open(p))
    t = float(d["T_over_rs"])
    assert abs(t - float(re.search(r"_T(\d+\.\d+)\.json", p).group(1))) == 0.0
    b2_tags.add(d["tag"]); b2_print.add(d["print_status"]); b2_floor_applies.add(bool(d["floor"]["applies"]))
    assert d["history"] == "B2" and d["record_found"] is True
    meta["B2"][t] = {"E_over_E0": float(d["record"]["E_over_E0"]), "R_t": d["record"]["R_t"], "dtype": d["dtype"], "complex64": bool(d["complex64"]),
                     "record_converged": d["record_converged"], "nsites": d["nsites"], "chain_end": d["chain_end_over_rs"], "slice": d["slice"],
                     "formation": d["formation"]}
    chans = d["channels"]
    kappas = [int(c["kappa"]) for c in chans]
    weights = [float(c["weight"]) for c in chans]
    assert kappas == list(range(1, 11)) and weights == [16.0 * k for k in kappas], (kappas, weights)
    b2_nd[t] = [int(c["n_dust"]) for c in chans]
    for c, w in d["weighted"].items():
        ex = [float(ch["cuts"][c]["excess"]) for ch in chans]
        exc = [float(ch["cuts"][c]["excess_c"]) for ch in chans]
        SX = [float(ch["cuts"][c]["S_X"]) for ch in chans]
        SXv = [float(ch["cuts"][c]["S_X_vac"]) for ch in chans]
        Nout = [float(ch["cuts"][c]["dust"]["N_out"]) if ch["cuts"][c].get("dust") else 0.0 for ch in chans]   # kappa 6..10: no dust columns, dust None
        my_excess = wdot(weights, ex)
        my_SX = wdot(weights, SX); my_SXv = wdot(weights, SXv)
        F_H = wdot(weights, [abs(a - b) for a, b in zip(ex, exc)])
        if d["complex64"]:
            F_H += C64_REL * float(w["S_X"])
        heavy["B2"][(t, c)] = {
            "excess": float(w["excess"]), "excess_mine": my_excess, "excess_from_SX": my_SX - my_SXv,
            "S_X": float(w["S_X"]), "S_X_mine": my_SX, "S_X_vac": float(w["S_X_vac"]), "S_X_vac_mine": my_SXv, "S_Xc": float(w["S_Xc"]),
            "F_H": F_H, "k10_term": weights[-1] * ex[-1], "dtype": d["dtype"], "complex64": bool(d["complex64"]),
            "first_return_rs": float(d["first_return"][c]["first_return_rs"]), "before_first_return": bool(d["first_return"][c]["before_first_return"]),
            "R_abs": float(chans[0]["cuts"][c]["R_abs"]), "dust_N_out": float(w["dust_N_out"]), "dust_N_out_mine": wdot(weights, Nout),
            "purity_gap": float(w["max_abs_purity_gap"]), "n_channels": len(chans),
            "per_channel_excess": ex, "per_channel_excess_c": exc}
P("  B2 jsons: %d clocks: %s" % (len(meta["B2"]), ", ".join("%g" % t for t in sorted(meta["B2"]))))
P("  tags:", b2_tags, "| print_status:", b2_print, "| floor.applies:", b2_floor_applies)
P("  dtype set:", sorted(set(m["dtype"] for m in meta["B2"].values())), "complex64 flags:", sorted(set(m["complex64"] for m in meta["B2"].values())),
  "nsites:", sorted(set(m["nsites"] for m in meta["B2"].values())), "chain_end:", sorted(set(m["chain_end"] for m in meta["B2"].values())))
P("  n_dust per channel (all clocks identical: %s): %s" % (len(set(tuple(v) for v in b2_nd.values())) == 1, b2_nd[sorted(b2_nd)[0]]))
P("  cuts stored on B2: %s, R_abs: %s" % (sorted(set(c for (_, c) in heavy["B2"])), sorted(set(v["R_abs"] for v in heavy["B2"].values()))))
dmax = max(abs(v["excess"] - v["excess_mine"]) for v in heavy["B2"].values())
dmax2 = max(abs(v["excess"] - v["excess_from_SX"]) for v in heavy["B2"].values())
dN = max(abs(v["dust_N_out"] - v["dust_N_out_mine"]) for v in heavy["B2"].values())
P("  weighted excess vs my sum_k 16k excess_k: max|diff| %.2e; vs (S_X - S_X_vac) resummed: %.2e; dust_N_out vs my sum: %.2e" % (dmax, dmax2, dN))
P("  F_H (complex128, sum_k 16k|excess_k - excess_c_k|): min %.2e max %.2e" % (min(v["F_H"] for v in heavy["B2"].values()), max(v["F_H"] for v in heavy["B2"].values())))

# ----------------------------------------------------------------------------------------------------------------
P("\n## 4. heavy prod and s8 (W8 all json; the per-slice jsons checked identical)")
Wj = json.load(open(W8))
P("  W8 title", Wj["title"], "code", Wj["code"], "histories", Wj["histories"], "cuts_rs", Wj["cuts_rs"], "rows", len(Wj["rows"]))
P("  W8 print_status:", Wj["print_status"])
RES["W8_print_status"] = Wj["print_status"]
w8_rows = {(r["history"], float(r["T_over_rs"])): r for r in Wj["rows"]}
nsame = 0
for p in hc_slices:
    d = json.load(open(p)); k = (d["history"], float(d["T_over_rs"]))
    same = json.dumps(d["weighted"], sort_keys=True) == json.dumps(w8_rows[k]["weighted"], sort_keys=True) and \
        json.dumps(d["channels"], sort_keys=True) == json.dumps(w8_rows[k]["channels"], sort_keys=True)
    nsame += same
P("  per-slice jsons identical (weighted + channels) to the all json: %d of %d" % (nsame, len(hc_slices)))
w8_nd = {}
for (h, t), r in sorted(w8_rows.items()):
    meta[h][t] = {"E_over_E0": float(r["record"]["E_over_E0"]), "R_t": r["record"]["R_t"], "dtype": r["dtype"], "complex64": bool(r["complex64"]),
                  "tag": r["tag"], "nsites": r["nsites"], "chain_end": r["chain_end_over_rs"], "formation": r["formation"], "slice": r["slice"],
                  "floor_applies": bool(r["floor"]["applies"]), "device": r.get("device")}
    chans = r["channels"]
    kappas = [int(c["kappa"]) for c in chans]; weights = [float(c["weight"]) for c in chans]
    assert kappas == list(range(1, 11)) and weights == [16.0 * k for k in kappas]
    w8_nd[(h, t)] = [int(c["n_dust"]) for c in chans]
    for c, w in r["weighted"].items():
        ex = [float(ch["cuts"][c]["heavy_excess"]) for ch in chans]
        exc = [float(ch["cuts"][c]["heavy_excess_c"]) for ch in chans]
        SH = [float(ch["cuts"][c]["S_H"]) for ch in chans]; SP = [float(ch["cuts"][c]["S_P"]) for ch in chans]
        SQ = [float(ch["cuts"][c]["S_Q"]) for ch in chans]; Sd = [float(ch["cuts"][c]["S_d"]) for ch in chans]
        Ksd = [float(ch["cuts"][c]["K_sd"]) for ch in chans]
        my = {"heavy_excess": wdot(weights, ex), "S_H": wdot(weights, SH), "S_P": wdot(weights, SP), "S_Q": wdot(weights, SQ),
              "S_d": wdot(weights, Sd), "K_sd": wdot(weights, Ksd)}
        F_H = wdot(weights, [abs(a - b) for a, b in zip(ex, exc)])
        F_H_c64 = F_H + (C64_REL * float(w["S_H"]) if r["complex64"] else 0.0)
        fr = r["first_return"][c]
        heavy[h][(t, c)] = {
            "excess": float(w["heavy_excess"]), "excess_mine": my["heavy_excess"], "excess_from_SH": my["S_H"] - my["S_P"],
            "S_X": float(w["S_H"]), "S_X_mine": my["S_H"], "S_P": float(w["S_P"]), "S_Q": float(w["S_Q"]), "S_d": float(w["S_d"]),
            "K_sd": float(w["K_sd"]), "excess_sea": float(w["excess_sea"]), "mine": my,
            "identity_residual": (float(w["S_Q"]) - float(w["S_P"])) + float(w["S_d"]) - float(w["K_sd"]) - float(w["heavy_excess"]),
            "F_H_pure": F_H, "F_H": F_H_c64, "k10_term": weights[-1] * ex[-1], "dtype": r["dtype"], "complex64": bool(r["complex64"]),
            "first_return_rs": float(fr["first_return_rs"]), "before_first_return": bool(fr["before_first_return"]),
            "R_abs": float(chans[0]["cuts"][c]["R_abs"]), "dust_N_out": float(w["dust_number_beyond_cut"]),
            "purity_gap": float(w["max_abs_purity_gap_H"]), "n_channels": len(chans), "all_bounds_hold": w.get("all_bounds_hold"),
            "per_channel_excess": ex, "per_channel_excess_c": exc}
for h in ("prod", "s8"):
    vals = [v for (hh, _), v in [((k[0], k[1]), v) for k, v in heavy[h].items()]]
    P("  %s: clocks %s; cuts %s; tags %s; nsites %s; chain_end %s" % (
        h, ", ".join("%g" % t for t in sorted(meta[h])), sorted(set(c for (_, c) in heavy[h]), key=int), sorted(set(m["tag"] for m in meta[h].values())),
        sorted(set(m["nsites"] for m in meta[h].values())), sorted(set(m["chain_end"] for m in meta[h].values()))))
    P("     dtype per clock: %s" % ", ".join("%g:%s" % (t, m["dtype"]) for t, m in sorted(meta[h].items())))
    P("     floor.applies per clock: %s" % ", ".join("%g:%s" % (t, m["floor_applies"]) for t, m in sorted(meta[h].items())))
    P("     n_dust per channel (identical over clocks: %s): %s" % (len(set(tuple(v) for (hh, t), v in w8_nd.items() if hh == h)) == 1, [v for (hh, t), v in sorted(w8_nd.items()) if hh == h][0]))
    P("     weighted heavy_excess vs my sum: max|diff| %.2e; vs S_H - S_P resummed: %.2e; identity residual max %.2e; all_bounds_hold: %s" % (
        max(abs(v["excess"] - v["excess_mine"]) for v in vals), max(abs(v["excess"] - v["excess_from_SH"]) for v in vals),
        max(abs(v["identity_residual"]) for v in vals), sorted(set(str(v["all_bounds_hold"]) for v in vals))))
    P("     F_H pure part range: %.2e to %.2e; complex64 addition 8.7e-6 S_H range: %.3e to %.3e (over complex64 rows)" % (
        min(v["F_H_pure"] for v in vals), max(v["F_H_pure"] for v in vals),
        min([C64_REL * v["S_X"] for v in vals if v["complex64"]] or [float("nan")]), max([C64_REL * v["S_X"] for v in vals if v["complex64"]] or [float("nan")])))
RES["W8_dust_count_check"] = Wj["rows"][0].get("dust_count_check")
P("  W8 dust_count_check (row 0):", json.dumps(RES["W8_dust_count_check"]))

# ----------------------------------------------------------------------------------------------------------------
P("\n## 5. cross-checks")
# 5a W8 prod vs page_own_object prod
chk_w8_page = []
for p in page_prod:
    d = json.load(open(p)); t = float(d["T_over_rs"])
    for c, w in d["weighted"].items():
        if (t, c) in heavy["prod"]:
            chk_w8_page.append({"t": t, "cut": c, "W8": heavy["prod"][(t, c)]["excess"], "page": float(w["excess"]),
                                "diff": heavy["prod"][(t, c)]["excess"] - float(w["excess"])})
P("  5a W8 vs page_own_object on prod: %d pairs, max|diff| %.3e (page_prod jsons: %d, cuts %s)" % (
    len(chk_w8_page), max(abs(x["diff"]) for x in chk_w8_page), len(page_prod), sorted(set(x["cut"] for x in chk_w8_page), key=int)))
# 5b prod vs B2 at the overlap
chk_prod_B2 = []
for (t, c), hv in sorted(heavy["prod"].items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
    b = heavy["B2"].get((t, c))
    if b:
        chk_prod_B2.append({"t": t, "cut": c, "prod": hv["excess"], "B2": b["excess"], "diff": hv["excess"] - b["excess"],
                            "rel": (hv["excess"] - b["excess"]) / b["excess"] if b["excess"] else None,
                            "prod_past": not hv["before_first_return"], "B2_past": not b["before_first_return"],
                            "F_prod": hv["F_H"], "F_B2": b["F_H"], "E_prod": meta["prod"][t]["E_over_E0"], "E_B2": meta["B2"][t]["E_over_E0"],
                            "Rt_prod": meta["prod"][t]["R_t"], "Rt_B2": meta["B2"][t]["R_t"]})
P("  5b prod vs B2 overlap pairs: %d" % len(chk_prod_B2))
for x in chk_prod_B2:
    P("     t %3g cut %2s prod %+.6e B2 %+.6e diff %+.3e rel %s past %s/%s | E/E0 %.10f %.10f (|d| %.1e) | R_t %s %s" % (
        x["t"], x["cut"], x["prod"], x["B2"], x["diff"], "-" if x["rel"] is None else "%+.3e" % x["rel"], x["prod_past"], x["B2_past"],
        x["E_prod"], x["E_B2"], abs(x["E_prod"] - x["E_B2"]), x["Rt_prod"], x["Rt_B2"]))
# 5c B2 slices vs the light's record
R = json.load(open(REC))
recs = {float(r["T_over_rs"]): r for r in R["records"]}
P("  5c record json: %d records, T/r_s %g to %g, tag %s, slices_c128 arg: %s" % (
    len(recs), min(recs), max(recs), R["args"].get("tag"), R["args"].get("slices_c128")))
slices_c128 = [float(s) for s in R["args"].get("slices_c128", "").split(",") if s]
P("     slices_c128 arg clocks == the 37 page_B2 clocks: %s" % (slices_c128 == sorted(meta["B2"])))
chk_hist = []
for t, m in sorted(meta["B2"].items()):
    x = recs[t]
    chk_hist.append({"t": t, "dE": m["E_over_E0"] - float(x["E_over_E0"]), "dR": None if (x["R_t"] is None or m["R_t"] is None) else m["R_t"] - float(x["R_t"]),
                     "R_slice": m["R_t"], "R_rec": x["R_t"], "conv": m["record_converged"]})
P("     B2 slice record values vs record json at the same clock: max|dE/E0| %.1e, max|dR_t| %.1e, R_t None on both: %d, record_converged all: %s" % (
    max(abs(x["dE"]) for x in chk_hist), max(abs(x["dR"]) for x in chk_hist if x["dR"] is not None),
    sum(1 for x in chk_hist if x["R_slice"] is None and x["R_rec"] is None), all(x["conv"] for x in chk_hist)))
P("     B2 tag on slices %s vs record args tag %s: %s" % (b2_tags, R["args"].get("tag"), b2_tags == {R["args"].get("tag")}))
# 5d s8 heavy vs light
Ls8 = light["s8"]
chk_s8 = []
for t, m in sorted(meta["s8"].items()):
    if t in Ls8["clocks"]:
        i = Ls8["clocks"].index(t)
        chk_s8.append({"t": t, "R_heavy": m["R_t"], "R_light": Ls8["R_t"][i], "E_heavy_over_E0": m["E_over_E0"], "E_light": Ls8["E"][i],
                       "E_light_over_E0": Ls8["E"][i] / Ls8["E"][0]})
P("  5d s8 heavy (W8) vs light s8 at common clocks:")
for x in chk_s8:
    P("     t %3g R_t heavy %s light %s equal %s | E/E0 heavy %.10f light %.10f (|d| %.1e)" % (
        x["t"], x["R_heavy"], x["R_light"], (x["R_heavy"] == x["R_light"]), x["E_heavy_over_E0"], x["E_light_over_E0"], abs(x["E_heavy_over_E0"] - x["E_light_over_E0"])))
P("     s8 formation heavy (W8): %s | light marks.s8: %s" % (json.dumps(meta["s8"][12.0]["formation"]),
  json.dumps({k: L["marks"]["s8"].get(k) for k in ("formation_record_clock_over_rs", "formation_R_t_a")})))
RES["checks"] = {"W8_vs_page_prod": chk_w8_page, "prod_vs_B2": chk_prod_B2, "B2_slices_vs_record": chk_hist, "s8": chk_s8}

# ----------------------------------------------------------------------------------------------------------------
P("\n## 6. composition (heavy + light at the same clock; exact clock match, and the delivery's 0.0625 tolerance gives the same pairs)")
table = []
for hh, lh in (("B2", "B2"), ("prod", "B2"), ("s8", "s8")):
    for t in sorted(meta[hh]):
        for c in CUTS_ALL:
            hv = heavy[hh].get((t, c))
            if hv is None:
                continue
            lt = light[lh]["rows"].get((t, c))
            near = [tt for (tt, cc) in light[lh]["rows"] if cc == c and abs(tt - t) <= 0.0625 + 1e-9]
            assert (lt is None and not near) or (lt is not None and near == [t]), (hh, t, c, near)
            s = None if lt is None else hv["excess"] + lt["S"]
            fl = None if (lt is None or lt["F_meas"] is None) else hv["F_H"] + lt["F_meas"]
            table.append({"hh": hh, "lh": lh, "t": t, "cut": c, "heavy": hv, "light": lt, "sum": s, "floor": fl})
P("  rows: %d; sums formed: %d" % (len(table), sum(1 for r in table if r["sum"] is not None)))

# ----------------------------------------------------------------------------------------------------------------
P("\n## 7. regenerated tables (the delivery's format, my values) and the line diff against the delivery md")


def e1(x):
    return "-" if x is None else "%.1e" % x


def row_fmt(r):
    hv, lv = r["heavy"], r["light"]
    ret = ("yes (%.1f)" if not hv["before_first_return"] else "no (%.1f)") % hv["first_return_rs"]
    return "| %.3f | %+.6f | %s | %+.2e | %s; %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["t"], hv["excess"], e1(hv["F_H"]), hv["k10_term"], hv["dtype"], ret,
        ("%.3f" % r["t"]) if lv else "absent", ("%+.6f" % lv["S"]) if lv else "absent",
        e1(lv["F_meas"]) if lv else "-", e1(lv["K_c"]) if lv else "-", (lv["verdict"] or "(no readout row)") if lv else "-",
        ("%+.6f" % r["sum"]) if r["sum"] is not None else "-", e1(r["floor"]))


HDR = "| T/r_s | heavy excess | F_H | kappa-10 term | heavy dtype; past first return | light clock | light K=18 excess | F_meas | K_c | light verdict | two-field sum | floor (F_H + F_meas) |"
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"
titles = [("B2", "16", "B2, 16 r_s cut (heavy: page_own_object_v1 on the B2 slices; light: PAGE v4.1; same chain)"),
          ("B2", "24", "B2, 24 r_s cut (same chain)"),
          ("prod", "2", "2 r_s cut: heavy on the prod chain (W8, 2048 sites, chain end 32 r_s), light on B2 (CROSS-CHAIN sum)"),
          ("prod", "16", "16 r_s cut: heavy prod (W8), light B2 (cross-chain; B2's own heavy rows above are the same-chain sum)"),
          ("prod", "24", "24 r_s cut: heavy prod (W8), light B2 (cross-chain; prod past its first return from 20 r_s)"),
          ("s8", "2", "s8 history, 2 r_s cut (heavy: W8 s8; light: PAGE v4.1 s8; same history by R_t)"),
          ("s8", "16", "s8 history, 16 r_s cut (heavy: W8 s8; light: PAGE v4.1 s8; same history by R_t)"),
          ("s8", "24", "s8 history, 24 r_s cut (heavy: W8 s8; light: PAGE v4.1 s8; same history by R_t)")]
out = []
for hh, c, title in titles:
    out += ["", "### " + title, "", HDR, SEP]
    out += [row_fmt(r) for r in table if r["hh"] == hh and r["cut"] == c]
out += ["", "### Check: prod vs B2 heavy excess at the overlap", "",
        "| T/r_s | cut | prod (W8) | B2 (page_own_object) | prod - B2 | rel | prod past return; B2 past return | F_H prod; B2 | E/E0 prod; B2 | R_t prod; B2 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for x in chk_prod_B2:
    out.append("| %.0f | %s | %+.6e | %+.6e | %+.2e | %s | %s; %s | %s; %s | %.6f; %.6f | %s; %s |" % (
        x["t"], x["cut"], x["prod"], x["B2"], x["diff"], "-" if x["rel"] is None else "%+.2e" % x["rel"], x["prod_past"], x["B2_past"],
        e1(x["F_prod"]), e1(x["F_B2"]), x["E_prod"], x["E_B2"], "-" if x["Rt_prod"] is None else "%.4f" % x["Rt_prod"], "-" if x["Rt_B2"] is None else "%.4f" % x["Rt_B2"]))
out += ["", "W8 vs page_own_object on prod: max |diff| %.2e over %d clock-cut pairs." % (max(abs(x["diff"]) for x in chk_w8_page), len(chk_w8_page)),
        "B2 slices' record values vs the light's record json (md5 %s): max |dE/E0| %.1e, max |dR_t| %.1e over %d clocks." % (
            RES["inputs_md5"]["REC_B2_final"][1], max(abs(x["dE"]) for x in chk_hist), max(abs(x["dR"]) for x in chk_hist if x["dR"] is not None), len(chk_hist)),
        "Light readout S vs the SUM array at the same pair: max |diff| %.1e." % light["B2"]["dRO"],
        "s8 heavy (W8) vs light s8 R_t at common clocks: " + "; ".join("%.0f: %s vs %s" % (
            x["t"], "-" if x["R_heavy"] is None else "%.10f" % x["R_heavy"], "-" if x["R_light"] is None else "%.10f" % x["R_light"]) for x in chk_s8) + "."]
regen = os.path.join(OUT, "heavycut_check_v1_regenerated_tables.md")
open(regen, "w").write("\n".join(out) + "\n")
mine_lines = [l for l in out if l.startswith("| ") or l.startswith("W8 vs") or l.startswith("B2 slices'") or l.startswith("Light readout") or l.startswith("s8 heavy")]
deliv_lines = [l.rstrip("\n") for l in open(DELIV_MD) if l.startswith("| ") or l.startswith("W8 vs") or l.startswith("B2 slices'") or l.startswith("Light readout") or l.startswith("s8 heavy")]
tabl_lines = [l.rstrip("\n") for l in open(DELIV + "/heavycut_twofield_sum_v1_tables.md") if l.startswith("| ") or l.startswith("W8 vs") or l.startswith("B2 slices'") or l.startswith("Light readout") or l.startswith("s8 heavy")]
import difflib
diff_md = list(difflib.unified_diff(deliv_lines, mine_lines, "delivery_v1.1_md_table_lines", "my_regenerated_table_lines", lineterm="", n=0))
diff_tb = list(difflib.unified_diff(tabl_lines, mine_lines, "delivery_tables_md_lines", "my_regenerated_table_lines", lineterm="", n=0))
open(os.path.join(OUT, "heavycut_check_v1_tables.diff"), "w").write("\n".join(diff_md) + ("\n" if diff_md else ""))
P("  table/check lines: delivery md %d, delivery tables.md %d, mine %d; md vs tables.md identical: %s; diff(md, mine) lines: %d; diff(tables.md, mine) lines: %d" % (
    len(deliv_lines), len(tabl_lines), len(mine_lines), deliv_lines == tabl_lines, len(diff_md), len(diff_tb)))
for l in diff_md[:40]:
    P("    " + l)

# ----------------------------------------------------------------------------------------------------------------
P("\n## 8. the prose numbers, one by one (my value from the stored inputs)")
pr = {}


def B2row(t, c):
    return next(r for r in table if r["hh"] == "B2" and r["t"] == t and r["cut"] == c)


def PRrow(t, c):
    return next(r for r in table if r["hh"] == "prod" and r["t"] == t and r["cut"] == c)


def S8row(t, c):
    return next(r for r in table if r["hh"] == "s8" and r["t"] == t and r["cut"] == c)


# summary 1
b2clocks = sorted(meta["B2"])
pr["B2_heavy_clocks"] = b2clocks
expect = [12, 16, 20, 24, 28, 36] + list(range(40, 101, 2))
P("  S1 B2 heavy clocks: %d; equal to {12,16,20,24,28,36} + 40..100 step 2: %s" % (len(b2clocks), b2clocks == [float(x) for x in expect]))
for c in ("16", "24"):
    r = B2row(100.0, c)
    P("  S1 at 100 r_s, cut %s: heavy %+.6f light %+.6f sum %+.6f; light share %.4f (%.2f percent)" % (
        c, r["heavy"]["excess"], r["light"]["S"], r["sum"], r["light"]["S"] / r["sum"], 100 * r["light"]["S"] / r["sum"]))
    pr["at100_" + c] = (r["heavy"]["excess"], r["light"]["S"], r["sum"], r["light"]["S"] / r["sum"])
    fr = [(x["t"], x["heavy"]["excess"] / x["sum"]) for x in table if x["hh"] == "B2" and x["cut"] == c and x["sum"] and x["sum"] > 0]
    above = [t for t, f in fr if f > 0.95]
    first_persist = None
    for i, (t, f) in enumerate(fr):
        if all(ff > 0.95 for _, ff in fr[i:]):
            first_persist = t; break
    P("  S1 heavy/sum > 0.95 from (persisting to 100): %s r_s at cut %s; fractions near it: %s" % (
        first_persist, c, ", ".join("%g:%.3f" % (t, f) for t, f in fr if abs(t - first_persist) <= 4)))
    pr["heavy_share_from_" + c] = first_persist
# summary 2
ex16 = {t: heavy["B2"][(t, "16")]["excess"] for t in b2clocks}
N16 = {t: heavy["B2"][(t, "16")]["dust_N_out"] for t in b2clocks}
N24 = {t: heavy["B2"][(t, "24")]["dust_N_out"] for t in b2clocks}
P("  S2 16 r_s heavy excess over 36..46: %s" % ", ".join("%g:%.6f" % (t, ex16[t]) for t in (36, 40, 42, 44, 46)))
P("  S2 16 r_s weighted dust count beyond the cut 36..48: %s" % ", ".join("%g:%.4f" % (t, N16[t]) for t in (36, 40, 42, 44, 46, 48)))
P("  S2 dust beyond 16 r_s at 40, 48, 50, 56, 100: %s" % ", ".join("%g:%.4f" % (t, N16[t]) for t in (40, 48, 50, 56, 100)))
P("  S2 dust beyond 24 r_s at 62, 64, 100: %s" % ", ".join("%g:%.4f" % (t, N24[t]) for t in (62, 64, 100)))
P("  S2 nats per weighted particle at 40 (16 r_s): %.1f; max heavy excess at 16/24 before 36 r_s: %.6f" % (
    ex16[40.0] / N16[40.0], max(heavy["B2"][(t, c)]["excess"] for t in b2clocks if t < 36 for c in ("16", "24"))))
pr["dust16"] = N16; pr["dust24"] = N24
h = heavy["prod"][(40.0, "2")]
P("  S2 prod 2 r_s cut at 40: heavy %.4f, S_d %.4f, excess_sea %.4f, K_sd %.4f, dust beyond %.4f; identity residual %.1e" % (
    h["excess"], h["S_d"], h["excess_sea"], h["K_sd"], h["dust_N_out"], h["identity_residual"]))
pr["prod40_2"] = {k: h[k] for k in ("excess", "S_d", "excess_sea", "K_sd", "dust_N_out")}
lt16 = L["marks"]["light_travel"]["16"]
P("  S2 light burst arrival at the 16 r_s cut (marks.light_travel, collar clock): c2.0 %.2f, c1.5 %.2f; record clock %.2f, %.2f" % (
    lt16["2.0"]["arrival_collar_clock_over_rs"], lt16["1.5"]["arrival_collar_clock_over_rs"], lt16["2.0"]["arrival_record_clock_over_rs"], lt16["1.5"]["arrival_record_clock_over_rs"]))
# summary 3
pclocks = sorted(meta["prod"])
fr2 = [heavy["prod"][(t, "2")]["first_return_rs"] for t in pclocks]
P("  S3 prod clocks %s; 2 r_s first return %.2f to %.2f; all before: %s" % (pclocks, min(fr2), max(fr2), all(heavy["prod"][(t, "2")]["before_first_return"] for t in pclocks)))
m2 = [r["t"] for r in table if r["hh"] == "prod" and r["cut"] == "2" and r["sum"] is not None]
r = PRrow(40.0, "2")
P("  S3 2 r_s cross-chain sum: matched clocks %s (%d); at 40: heavy %.6f light %.6f sum %.6f floor %.2e" % (m2, len(m2), r["heavy"]["excess"], r["light"]["S"], r["sum"], r["floor"]))
# summary 5
s8m = [r["t"] for r in table if r["hh"] == "s8" and r["cut"] == "2" and r["sum"] is not None]
r = S8row(40.0, "2")
P("  S5 s8 matched clocks %s; at 40 the 2 r_s cut: heavy %.6f light %.6f sum %.6f" % (s8m, r["heavy"]["excess"], r["light"]["S"], r["sum"]))
# summary 6
gaps = sorted(set(light["B2"]["clocks"]) - set(b2clocks))
P("  S6 light-only B2 clocks (no heavy): %s" % gaps)
P("  S6 light readout min clock: B2 %g, s8 %g" % (min(k[0] for k, v in light["B2"]["rows"].items() if v["readout"]), min(k[0] for k, v in light["s8"]["rows"].items() if v["readout"])))
# section 2
P("  Sec2 16 r_s |prod - B2| for 12..24: %s; 28: %.2e (%.2f percent of %.4f); 40: %.2e (rel %.2e), prod first return at 40: %.2f" % (
    ", ".join("%g:%.2e" % (x["t"], abs(x["diff"])) for x in chk_prod_B2 if x["cut"] == "16" and x["t"] <= 24),
    abs(next(x["diff"] for x in chk_prod_B2 if x["cut"] == "16" and x["t"] == 28)), 100 * abs(next(x["rel"] for x in chk_prod_B2 if x["cut"] == "16" and x["t"] == 28)),
    next(x["B2"] for x in chk_prod_B2 if x["cut"] == "16" and x["t"] == 28),
    abs(next(x["diff"] for x in chk_prod_B2 if x["cut"] == "16" and x["t"] == 40)), abs(next(x["rel"] for x in chk_prod_B2 if x["cut"] == "16" and x["t"] == 40)),
    heavy["prod"][(40.0, "16")]["first_return_rs"]))
P("  Sec2 24 r_s prod first return: %.2f; prod values from 20 on: %s; B2 values from 20 on: %s; at 40 diff %.2e rel %.2e" % (
    heavy["prod"][(20.0, "24")]["first_return_rs"],
    ", ".join("%g:%+.2e" % (x["t"], x["prod"]) for x in chk_prod_B2 if x["cut"] == "24" and x["t"] >= 20),
    ", ".join("%g:%+.2e" % (x["t"], x["B2"]) for x in chk_prod_B2 if x["cut"] == "24" and x["t"] >= 20),
    abs(next(x["diff"] for x in chk_prod_B2 if x["cut"] == "24" and x["t"] == 40)), abs(next(x["rel"] for x in chk_prod_B2 if x["cut"] == "24" and x["t"] == 40))))
P("  Sec2 E/E0 prod vs B2 at the overlap, 10 digits: %s" % "; ".join("%g: %.10f / %.10f" % (x["t"], x["E_prod"], x["E_B2"]) for x in chk_prod_B2 if x["cut"] == "16"))
P("  Sec2 R_t prod vs B2: %s" % "; ".join("%g: %s / %s" % (x["t"], x["Rt_prod"], x["Rt_B2"]) for x in chk_prod_B2 if x["cut"] == "16"))
# section 3
k10_16 = {t: heavy["B2"][(t, "16")]["k10_term"] for t in b2clocks}
k10_24 = {t: heavy["B2"][(t, "24")]["k10_term"] for t in b2clocks}
P("  Sec3 B2 kappa-10 term, 16 r_s cut: t>=48 min %.3f max %.3f; t>=46 min %.3f; 24 r_s cut: t>=64 min %.3f max %.3f; t>=62 min %.3f" % (
    min(v for t, v in k10_16.items() if t >= 48), max(v for t, v in k10_16.items() if t >= 48), min(v for t, v in k10_16.items() if t >= 46),
    min(v for t, v in k10_24.items() if t >= 64), max(v for t, v in k10_24.items() if t >= 64), min(v for t, v in k10_24.items() if t >= 62)))
P("  Sec3 kappa-10 share: 16 r_s at 100: %.4f (%.2f pct), at 50: %.4f (%.2f pct); 24 r_s at 100: %.4f (%.2f pct), at 64: %.4f (%.2f pct)" % (
    k10_16[100.0] / ex16[100.0], 100 * k10_16[100.0] / ex16[100.0], k10_16[50.0] / ex16[50.0], 100 * k10_16[50.0] / ex16[50.0],
    k10_24[100.0] / heavy["B2"][(100.0, "24")]["excess"], 100 * k10_24[100.0] / heavy["B2"][(100.0, "24")]["excess"],
    k10_24[64.0] / heavy["B2"][(64.0, "24")]["excess"], 100 * k10_24[64.0] / heavy["B2"][(64.0, "24")]["excess"]))
k10p = {t: heavy["prod"][(t, "2")]["k10_term"] for t in pclocks}
P("  Sec3 prod 2 r_s kappa-10 term from 20 on: %s; shares: %s" % (
    ", ".join("%g:%.3f" % (t, v) for t, v in k10p.items() if t >= 20),
    ", ".join("%g:%.3f pct" % (t, 100 * v / heavy["prod"][(t, "2")]["excess"]) for t, v in k10p.items() if t >= 20)))
c64F = sorted(v["F_H"] for h in ("prod", "s8") for v in heavy[h].values() if v["complex64"])
P("  Sec3 complex64 F_H range over all complex64 rows: %.2e to %.2e; complex128 F_H range (all histories): %.1e to %.1e" % (
    c64F[0], c64F[-1], min(v["F_H"] for h in heavy for v in heavy[h].values() if not v["complex64"]), max(v["F_H"] for h in heavy for v in heavy[h].values() if not v["complex64"])))
# section 4
P("  Sec4 B2 first-return estimates (page_own_object): 16 r_s %s; 24 r_s %s" % (
    sorted(set(round(heavy["B2"][(t, "16")]["first_return_rs"], 2) for t in b2clocks)), sorted(set(round(heavy["B2"][(t, "24")]["first_return_rs"], 2) for t in b2clocks))))
P("  Sec4 B2 rows past first return: 16 r_s %s; 24 r_s %s" % (
    [t for t in b2clocks if not heavy["B2"][(t, "16")]["before_first_return"]], [t for t in b2clocks if not heavy["B2"][(t, "24")]["before_first_return"]]))
# section 5
d36 = json.load(open(PAGE_DIR + "/page_B2_T0036.00.json"))
tfs = d36["two_fields_sum"]
P("  Sec5 T0036 two_fields_sum raw: 16: heavy %.3f light %.3f sum %.3f; 24: heavy %.3f light %.3f sum %.3f; light source: %s" % (
    tfs["16"]["heavy"], tfs["16"]["light"], tfs["16"]["sum"], tfs["24"]["heavy"], tfs["24"]["light"], tfs["24"]["sum"], d36["light_channel"].get("files")))
P("  Sec5 excess sums at 36: 16 r_s %+.6f, 24 r_s %+.6f" % (B2row(36.0, "16")["sum"], B2row(36.0, "24")["sum"]))
# status section
P("  Status 'below 0.03 nats before 36': max heavy excess at 16 and 24 for t < 36: %.6f" % max(heavy["B2"][(t, c)]["excess"] for t in b2clocks if t < 36 for c in ("16", "24")))
P("  Status '89 and 69 of 640': %.1f and %.1f; '232 and 201': %.1f and %.1f; '8.7 and 8.5 (3.6 and 4.0 pct)': %.2f %.2f (%.2f %.2f pct)" % (
    N16[100.0], N24[100.0], ex16[100.0], heavy["B2"][(100.0, "24")]["excess"], B2row(100.0, "16")["light"]["S"], B2row(100.0, "24")["light"]["S"],
    100 * B2row(100.0, "16")["light"]["S"] / B2row(100.0, "16")["sum"], 100 * B2row(100.0, "24")["light"]["S"] / B2row(100.0, "24")["sum"]))
P("  Status '0.9 to 6 nats over 36 to 46 at the 16 r_s cut': %.3f to %.3f" % (min(ex16[t] for t in (36, 40, 42, 44, 46)), max(ex16[t] for t in (36, 40, 42, 44, 46))))
# light verdict counts per table
for hh, c, _ in titles:
    vs = [r["light"]["verdict"] if r["light"] else "absent" for r in table if r["hh"] == hh and r["cut"] == c]
    P("  verdicts %s/%s: %s" % (hh, c, {v: vs.count(v) for v in sorted(set(str(x) for x in vs))}))
RES["prose"] = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in pr.items()}

# ----------------------------------------------------------------------------------------------------------------
# full values to json
RES["table"] = [{"hh": r["hh"], "lh": r["lh"], "t": r["t"], "cut": r["cut"],
                 "heavy": {k: v for k, v in r["heavy"].items() if k not in ("mine",)}, "light": r["light"], "sum": r["sum"], "floor": r["floor"]} for r in table]
RES["meta"] = {h: {str(t): m for t, m in meta[h].items()} for h in meta}
RES["seconds"] = time.time() - T0
json.dump(RES, open(os.path.join(OUT, "heavycut_check_v1.json"), "w"), indent=1, default=str)
P("\nwall %.2f s; outputs in %s" % (RES["seconds"], OUT))
