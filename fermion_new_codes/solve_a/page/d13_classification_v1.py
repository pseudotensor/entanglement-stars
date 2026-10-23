#!/usr/bin/env python
"""D13 classification composition (NOTES_D13_classification_v1): the four observables of REVIEWA_RULING_O1_D13_protocol_v1 section 1
read from the stored inline D13 outputs (page_d13_readout_v1d) of the B2 production (kappa 1, 2; kappa 3 the guard), the B2
record-interpolation control (interp none; kappa 1, 2) and the s8 production (kappa 1, 2, 3; to 40 r_s), with the protocol's error
budget: the total fluence allowance per pre-return window, every error source included, 1e-4 of the comparator's luminosity
L_H(kappa_T) (kappa 1 + 2, weighted) times the window.

Box CPU, numpy/json only; reads json only. Writes d13_class_v1.json and D13_class_v1.md to --out.
"""
import argparse, json, math, os, time, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--b2", default="<internal>/prod")
ap.add_argument("--ctrl", default="<internal>/v6o_ctrl_B2")
ap.add_argument("--s8", default="<internal>/prod")
ap.add_argument("--s8b", default="<internal>/s8_prod_K18")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t00 = time.time()
RS = 64.0  # a per r_s (r_s/a 64)
ALLOW = 1e-4
DARK = 1e-3


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(dirn, tag, k):
    pj = os.path.join(dirn, "%s_k%d_d13.json" % (tag, k))
    ps = os.path.join(dirn, "%s_k%d_summary.json" % (tag, k))
    D = json.load(open(pj))
    S = json.load(open(ps)) if os.path.exists(ps) else None
    return D, S, {"d13": (os.path.realpath(pj), md5(pj)), "summary": (os.path.realpath(ps), md5(ps)) if S else None}


def w_of(k):
    return 16.0 * k


SETS = {"B2": (a.b2, "v6o_B2", [1, 2, 3]), "ctrl": (a.ctrl, "v6o_ctrl_B2", [1, 2]), "s8": (a.s8, "v6o_s8", [1, 2, 3])}
data, prov = {}, {}
for name, (d, tag, ks) in SETS.items():
    data[name] = {}
    for k in ks:
        D, S, p = load(d, tag, k)
        data[name][k] = (D, S)
        prov["%s_k%d" % (name, k)] = p
# the s8 copy under page_v4 (is it the same file?)
s8b_same = {}
for k in (1, 2, 3):
    pb = os.path.join(a.s8b, "v6o_s8_k%d_d13.json" % k)
    if os.path.exists(pb):
        s8b_same[k] = md5(pb) == prov["s8_k%d" % k]["d13"][1]


def summary_rows(S):
    return {round(float(r["t_over_rs"]), 4): r for r in (S or {}).get("rows", [])}


def clock(D, t, k):
    c = D["clocks"].get("%010.4f" % t)
    return ((c or {}).get("per_channel") or {}).get(str(k))


# --------------------------------------------------------------------------------------------- comparator L_H(kappa_T), weighted
def LH(name, t, ks=(1, 2)):
    v = 0.0
    for k in ks:
        r = summary_rows(data[name][k][1]).get(round(t, 4))
        if r is None or r.get("L_H_kappaT") is None:
            return None
        v += r["L_H_kappaT"]
    return v


def LH_window(name, t0, t1, ks=(1, 2)):
    """the comparator luminosity's time integral over the window (trapezoid on the stored checkpoints), energy units (L per a x a)."""
    rows = summary_rows(data[name][ks[0]][1])
    ts = sorted(t for t in rows if t0 - 1e-9 <= t <= t1 + 1e-9)
    vals = [LH(name, t, ks) for t in ts]
    if len(ts) < 2 or any(v is None for v in vals):
        return None, None
    I = float(np.trapz(vals, np.array(ts) * RS))
    return I, float(np.mean(vals))


# ---------------------------------------------------------------------------------------------------------- (a) the ledger
def ledger(name, ks):
    out = {}
    for k in ks:
        D = data[name][k][0]
        for w in D["windows"][str(k)]:
            if w.get("second_term_D") is None or w["kind"].startswith("between"):
                continue
            key = (w["t0_over_rs"], w["t1_over_rs"], w["cut"])
            e = out.setdefault(key, {"t0": w["t0_over_rs"], "t1": w["t1_over_rs"], "cut": w["cut"], "u": (w.get("u_start_rs"), w.get("u_end_rs")), "per_k": {}})
            wt = w_of(k)
            e["per_k"][k] = {q: wt * w[q] for q in ("second_term_D", "crossing_current", "crossing_jumps", "vacuum_change", "telescoping_residual", "inventory_D_start", "inventory_D_end")}
    return out


def inside_inventory(name, k, t, cut="16"):
    c = clock(data[name][k][0], t, k)
    if c is None or c.get("E_inv") is None:
        return None
    inv = (c.get("inventory") or {}).get(cut)
    if inv is None:
        return None
    return w_of(k) * (c["E_inv"] - inv["E_D"])


def between_inventory(name, k, t):
    c = clock(data[name][k][0], t, k)
    if c is None:
        return None
    i16, i30 = (c.get("inventory") or {}).get("16"), (c.get("inventory") or {}).get("30")
    if i16 is None or i30 is None:
        return None
    return w_of(k) * (i16["E_D"] - i30["E_D"])


def sumk(e, ks, q):
    if not all(k in e["per_k"] for k in ks):
        return None
    return sum(e["per_k"][k][q] for k in ks)


rows_a = []
L_B2 = ledger("B2", [1, 2, 3])
L_ct = ledger("ctrl", [1, 2])
L_s8 = ledger("s8", [1, 2, 3])
for key in sorted(L_B2):
    e = L_B2[key]
    t0, t1, cut = key
    if t1 - t0 < 13.0 - 1e-9:
        continue
    st12 = sumk(e, (1, 2), "second_term_D")
    st123 = sumk(e, (1, 2, 3), "second_term_D")
    cur12 = sumk(e, (1, 2), "crossing_current")
    jmp12 = sumk(e, (1, 2), "crossing_jumps")
    vac12 = sumk(e, (1, 2), "vacuum_change")
    tel12 = sumk(e, (1, 2), "telescoping_residual")
    ec = L_ct.get(key)
    st_ct = sumk(ec, (1, 2), "second_term_D") if ec else None
    cur_ct = sumk(ec, (1, 2), "crossing_current") if ec else None
    Iw, LHm = LH_window("B2", t0, t1)
    allow = ALLOW * Iw if Iw is not None else None
    r = {"t0": t0, "t1": t1, "cut": cut, "u": e["u"], "second_term_12": st12, "crossing_current_12": cur12, "creation_12": (jmp12 - vac12) if jmp12 is not None else None,
         "telescoping_residual_12": tel12, "kappa3_change": (st123 - st12) if st123 is not None else None,
         "ctrl_second_term_12": st_ct, "interp_change": (st12 - st_ct) if st_ct is not None else None, "ctrl_crossing_current_12": cur_ct,
         "LH_window_integral": Iw, "LH_mean": LHm, "allowance": allow}
    if cut == "16":
        din = [inside_inventory("B2", k, t1) - inside_inventory("B2", k, t0) for k in (1, 2)] if all(inside_inventory("B2", k, t) is not None for k in (1, 2) for t in (t0, t1)) else None
        r["inside16_change_12"] = sum(din) if din else None
        r["replenished_16"] = (st12 + r["inside16_change_12"]) if din else None  # energy beyond 16 gained minus energy inside 16 lost: >0 the window's work
        if all(between_inventory("B2", k, t) is not None for k in (1, 2) for t in (t0, t1)):
            r["between_change_12"] = sum(between_inventory("B2", k, t1) - between_inventory("B2", k, t0) for k in (1, 2))
        # the energy identity from the production summary: E_inv change = W - collar flow over the window
        dinv = 0.0; ok = True
        for k in (1, 2):
            rs_ = summary_rows(data["B2"][k][1])
            if round(t0, 4) in rs_ and round(t1, 4) in rs_:
                dinv += rs_[round(t1, 4)]["E_inv_weighted"] - rs_[round(t0, 4)]["E_inv_weighted"]
            else:
                ok = False
        r["E_inv_change_12_summary"] = dinv if ok else None
    errs = {k_: abs(v) for k_, v in (("telescoping", tel12), ("kappa3_guard", r["kappa3_change"]), ("record_interpolation", r["interp_change"])) if v is not None}
    r["error_terms"] = errs
    r["error_total"] = sum(errs.values()) if errs else None
    r["budget_ratio"] = (r["error_total"] / allow) if (allow and r["error_total"] is not None) else None
    rows_a.append(r)

# ------------------------------------------------------------------------------------------------- (b) the probes, L_old/L_sea
def probes(name, ks):
    out = {}
    for k in ks:
        for p in data[name][k][0]["probes"]:
            key = (p["t_over_rs"], p["cut"])
            e = out.setdefault(key, {"t": p["t_over_rs"], "cut": p["cut"], "per_k": {}, "first_return": p.get("first_return_rs"), "before": p.get("before_first_return")})
            refs = {round(q["t_ref_over_rs"], 4): {"L_old": w_of(k) * q["L_old"], "L_sea": w_of(k) * q["L_sea"], "check": w_of(k) * q["check_L_minus_direct"]} for q in p["refs"]}
            e["per_k"][k] = {"L_direct": w_of(k) * p["L_direct"], "refs": refs}
    return out


P_B2 = probes("B2", [1, 2, 3])
P_ct = probes("ctrl", [1, 2])
P_s8 = probes("s8", [1, 2, 3])
rows_b = []
for key in sorted(P_B2):
    e = P_B2[key]
    t, cut = key
    for tref in (12.0, 20.0, 40.0):
        if not all(k in e["per_k"] and tref in e["per_k"][k]["refs"] for k in (1, 2)):
            continue
        Ld = sum(e["per_k"][k]["L_direct"] for k in (1, 2))
        Lo = sum(e["per_k"][k]["refs"][tref]["L_old"] for k in (1, 2))
        Ls = sum(e["per_k"][k]["refs"][tref]["L_sea"] for k in (1, 2))
        ch = sum(abs(e["per_k"][k]["refs"][tref]["check"]) for k in (1, 2))
        k3 = e["per_k"].get(3, {}).get("refs", {}).get(tref, {}).get("L_sea") if 3 in e["per_k"] else None
        ec = P_ct.get(key)
        Ls_ct = sum(ec["per_k"][k]["refs"][tref]["L_sea"] for k in (1, 2)) if (ec and all(k in ec["per_k"] and tref in ec["per_k"][k]["refs"] for k in (1, 2))) else None
        Ld_ct = sum(ec["per_k"][k]["L_direct"] for k in (1, 2)) if ec else None
        rows_b.append({"t": t, "cut": cut, "t_ref": tref, "L_direct": Ld, "L_old": Lo, "L_sea": Ls, "check": ch, "kappa3_L_sea": k3,
                       "ctrl_L_sea": Ls_ct, "ctrl_L_direct": Ld_ct, "LH": LH("B2", t), "before_first_return": e["before"], "first_return": e["first_return"]})

# windowed L_sea fluence over probe windows of 16 r_s (5 probes, trapezoid), per cut and t_ref
rows_bw = []
for cut in ("16", "30"):
    for tref in (20.0, 40.0, 12.0):
        pts = sorted((r for r in rows_b if r["cut"] == cut and r["t_ref"] == tref), key=lambda r: r["t"])
        ts = [r["t"] for r in pts]
        for i in range(len(pts)):
            j = [jj for jj in range(i, len(pts)) if ts[jj] - ts[i] >= 13.0 - 1e-9]
            if not j:
                continue
            j = j[0]
            seg = pts[i:j + 1]
            tt = np.array([r["t"] for r in seg]) * RS

            def I(key_, seg=seg, tt=tt):
                v = [r[key_] for r in seg]
                return None if any(x is None for x in v) else float(np.trapz(v, tt))
            Lsea, Lold, Ldir = I("L_sea"), I("L_old"), I("L_direct")
            k3 = I("kappa3_L_sea")
            ct = I("ctrl_L_sea")
            chk = I("check")
            lh = I("LH")
            errs = {k_: abs(v) for k_, v in (("backprop_check", chk), ("kappa3_guard", k3), ("record_interpolation", (Lsea - ct) if ct is not None else None)) if v is not None}
            rows_bw.append({"cut": cut, "t_ref": tref, "t0": ts[i], "t1": ts[j], "L_sea_fluence": Lsea, "L_old_fluence": Lold, "L_direct_fluence": Ldir,
                            "LH_fluence": lh, "allowance": ALLOW * lh if lh else None, "dark_threshold": DARK * lh if lh else None,
                            "error_terms": errs, "error_total": sum(errs.values()) if errs else None,
                            "L_sea_over_LH": (Lsea / lh) if lh else None, "L_old_over_direct": (Lold / Ldir) if Ldir else None})

# ------------------------------------------------------------------------------------------------- (c) hard and soft per window
rows_c = []
for w0 in data["B2"][1][0]["hard_windows"]["1"]:
    t0, t1 = w0["t0_over_rs"], w0["t1_over_rs"]
    if t1 - t0 < 13.0 - 1e-9:
        continue
    agg = {}
    for ks, lab in (((1, 2), "12"), ((1, 2, 3), "123")):
        for q in ("delta_hard_E_out", "delta_hard_E_in", "delta_soft_E_out", "delta_soft_E_in", "delta_hard_E_out_P", "delta_soft_E_out_P", "delta_hard_E_in_P", "delta_soft_E_in_P"):
            vals = []
            for k in ks:
                ww = [x for x in data["B2"][k][0]["hard_windows"][str(k)] if x["t0_over_rs"] == t0 and x["t1_over_rs"] == t1]
                vals.append(w_of(k) * ww[0][q] if (ww and ww[0].get(q) is not None) else None)
            agg[q + "_" + lab] = sum(vals) if all(v is not None for v in vals) else None
    ctrl = {}
    for q in ("delta_hard_E_out", "delta_soft_E_out"):
        vals = []
        for k in (1, 2):
            ww = [x for x in data["ctrl"][k][0]["hard_windows"][str(k)] if x["t0_over_rs"] == t0 and x["t1_over_rs"] == t1]
            vals.append(w_of(k) * ww[0][q] if ww else None)
        ctrl[q] = sum(vals) if all(v is not None for v in vals) else None
    Iw, _ = LH_window("B2", t0, t1)
    rows_c.append({"t0": t0, "t1": t1, "u": (w0.get("u_start_rs"), w0.get("u_end_rs")), **agg, "ctrl_hard_E_out_12": ctrl["delta_hard_E_out"], "ctrl_soft_E_out_12": ctrl["delta_soft_E_out"],
                   "LH_window_integral": Iw})

# ------------------------------------------------------------------------- the collar's signed split per checkpoint (soft/hard, in/out)
rows_col = []
for t in sorted(summary_rows(data["B2"][1][1])):
    vals = {}
    ok = True
    for k in (1, 2):
        c = clock(data["B2"][k][0], t, k)
        cs = (c or {}).get("collar_split")
        if not cs:
            ok = False; break
        for q in ("E_flux_out_soft", "E_flux_out_hard", "E_flux_in_soft", "E_flux_in_hard"):
            vals[q] = vals.get(q, 0.0) + w_of(k) * cs[q]
        vals["join_max_rel_dv"] = c["join"]["max_rel_dv"]
        fr = (c.get("first_return") or {}).get("16") or {}
        vals["before_first_return_16"] = fr.get("before_first_return")
    if not ok:
        continue
    out_ = vals["E_flux_out_soft"] + vals["E_flux_out_hard"]
    in_ = vals["E_flux_in_soft"] + vals["E_flux_in_hard"]
    vals.update({"t": t, "out": out_, "in": in_, "in_over_out": abs(in_) / out_ if out_ > 0 else None,
                 "soft_share_out": vals["E_flux_out_soft"] / out_ if out_ > 0 else None, "LH": LH("B2", t),
                 "L_interval": sum(summary_rows(data["B2"][k][1])[t].get("L_interval") or 0.0 for k in (1, 2))})
    rows_col.append(vals)

# ----------------------------------------------------------------------------------------------- the units check (L x a = energy)
units = []
for k in (1, 2):
    rs_ = summary_rows(data["B2"][k][1])
    ts = sorted(rs_)
    for i in range(1, len(ts)):
        r0, r1 = rs_[ts[i - 1]], rs_[ts[i]]
        if r1.get("L_interval") and abs(r1["L_interval"]) > 1e-8:
            units.append((k, ts[i], (r1["collar_flow_acc_weighted"] - r0["collar_flow_acc_weighted"]) / (r1["L_interval"] * (ts[i] - ts[i - 1]) * RS)))

# ---------------------------------------------------------------------------------------------------------------- s8 (to 40)
rows_s8 = []
for key in sorted(L_s8):
    e = L_s8[key]
    t0, t1, cut = key
    if t1 - t0 < 13.0 - 1e-9:
        continue
    rows_s8.append({"t0": t0, "t1": t1, "cut": cut, "second_term_12": sumk(e, (1, 2), "second_term_D"), "second_term_123": sumk(e, (1, 2, 3), "second_term_D"),
                    "crossing_current_12": sumk(e, (1, 2), "crossing_current"), "LH_window_integral": LH_window("s8", t0, t1)[0]})
rows_s8b = []
for key in sorted(P_s8):
    e = P_s8[key]
    for tref in (12.0, 20.0, 40.0):
        if all(k in e["per_k"] and tref in e["per_k"][k]["refs"] for k in (1, 2)):
            rows_s8b.append({"t": key[0], "cut": key[1], "t_ref": tref, "L_direct": sum(e["per_k"][k]["L_direct"] for k in (1, 2)),
                             "L_old": sum(e["per_k"][k]["refs"][tref]["L_old"] for k in (1, 2)), "L_sea": sum(e["per_k"][k]["refs"][tref]["L_sea"] for k in (1, 2)), "LH": LH("s8", key[0])})

OUT = {"code": "d13_classification_v1.py", "protocol": "REVIEWA_RULING_O1_D13_protocol_v1 section 1; allowance 1e-4 L_H(kappa_T) x window (REVIEWA_web40i_x1fix_v1 Y15/(f)); dark threshold 1e-3",
       "provenance": prov, "s8_copy_same_as_composed": s8b_same, "units_check_collar_flow_over_L_interval_dt": units[:20], "ledger": rows_a, "probes": rows_b, "probe_windows": rows_bw,
       "hard_soft": rows_c, "collar": rows_col, "s8_ledger": rows_s8, "s8_probes": rows_s8b, "seconds": time.time() - t00}
json.dump(OUT, open(os.path.join(a.out, "d13_class_v1.json"), "w"), indent=1, default=float)

f = lambda x, s="%.3e": "-" if x is None else (s % x)
L = ["# d13_classification_v1 (box CPU; composed from the stored inline D13 outputs)", ""]
L.append("units check (accumulated collar flow change over L_interval x dt in a; 1 means L is energy per a): " + ", ".join("k%d T%g %.4f" % u for u in units[:12]))
L.append("s8 copy under page_v4 identical to composed_s8: %s" % s8b_same)
L.append("")
L.append("## (a) ledger, kappa 1+2 weighted, windows >= 13 r_s, B2 production; errors: telescoping, kappa-3 guard change, record interpolation (prod - ctrl)")
L.append("| window | cut | second term | crossing current | creation (jumps - vac change) | inside-16 change | replenished (second term + inside change) | E_inv change (summary) | allowance 1e-4 L_H dt | kappa3 change | interp change | telescoping | error total / allowance |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows_a:
    L.append("| %g-%g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["t0"], r["t1"], r["cut"], f(r["second_term_12"]), f(r["crossing_current_12"]), f(r["creation_12"]), f(r.get("inside16_change_12")),
                                                                         f(r.get("replenished_16")), f(r.get("E_inv_change_12_summary")), f(r["allowance"]), f(r["kappa3_change"]), f(r["interp_change"]), f(r["telescoping_residual_12"]), f(r["budget_ratio"], "%.2e")))
L.append("")
L.append("## (b) probe windows (trapezoid over probes every 4 r_s, >= 13 r_s), kappa 1+2 weighted")
L.append("| cut | t_ref | window | L_sea fluence | L_old fluence | direct fluence | L_H fluence | L_sea/L_H fluence | dark threshold 1e-3 | allowance 1e-4 | errors (check; kappa3; interp) | error/allowance |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows_bw:
    e = r["error_terms"]
    L.append("| %s | %g | %g-%g | %s | %s | %s | %s | %s | %s | %s | %s; %s; %s | %s |" % (r["cut"], r["t_ref"], r["t0"], r["t1"], f(r["L_sea_fluence"]), f(r["L_old_fluence"]), f(r["L_direct_fluence"]), f(r["LH_fluence"]),
                                                                                f(r["L_sea_over_LH"], "%.3g"), f(r["dark_threshold"]), f(r["allowance"]), f(e.get("backprop_check")), f(e.get("kappa3_guard")), f(e.get("record_interpolation")),
                                                                                f((r["error_total"] / r["allowance"]) if r["allowance"] and r["error_total"] is not None else None, "%.2e")))
L.append("")
L.append("## (c) exterior shells beyond R_cut 24 r_s, per window (kappa 1+2 weighted; with kappa 3; held vacuum _P; ctrl)")
L.append("| window | hard out | hard in | soft out | soft in | hard out P | soft out P | hard out (123) | soft out (123) | ctrl hard out | ctrl soft out | L_H dt |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows_c:
    L.append("| %g-%g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["t0"], r["t1"], f(r["delta_hard_E_out_12"]), f(r["delta_hard_E_in_12"]), f(r["delta_soft_E_out_12"]), f(r["delta_soft_E_in_12"]),
                                                                       f(r["delta_hard_E_out_P_12"]), f(r["delta_soft_E_out_P_12"]), f(r["delta_hard_E_out_123"]), f(r["delta_soft_E_out_123"]),
                                                                       f(r["ctrl_hard_E_out_12"]), f(r["ctrl_soft_E_out_12"]), f(r["LH_window_integral"])))
L.append("")
L.append("## collar signed split per checkpoint (kappa 1+2 weighted): soft/hard out, in, in/out, soft share, L_H, join")
L.append("| T | out soft | out hard | in soft | in hard | in/out | soft share of out | L_H | L_interval | join max rel dv | before first return |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows_col:
    L.append("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["t"], f(r["E_flux_out_soft"]), f(r["E_flux_out_hard"]), f(r["E_flux_in_soft"]), f(r["E_flux_in_hard"]), f(r["in_over_out"], "%.3g"),
                                                                     f(r["soft_share_out"], "%.3g"), f(r["LH"]), f(r["L_interval"]), f(r["join_max_rel_dv"], "%.2e"), r["before_first_return_16"]))
L.append("")
L.append("## s8 (to 40 r_s): ledger windows and probes")
for r in rows_s8:
    L.append("- %g-%g cut %s: second term 12 %s (123 %s), crossing current %s, L_H dt %s" % (r["t0"], r["t1"], r["cut"], f(r["second_term_12"]), f(r["second_term_123"]), f(r["crossing_current_12"]), f(r["LH_window_integral"])))
for r in rows_s8b:
    L.append("- probe T %g cut %s t_ref %g: L %s, L_old %s, L_sea %s, L_H %s" % (r["t"], r["cut"], r["t_ref"], f(r["L_direct"]), f(r["L_old"]), f(r["L_sea"]), f(r["LH"])))
L.append("")
L.append("seconds %.1f" % (time.time() - t00))
open(os.path.join(a.out, "D13_class_v1.md"), "w").write("\n".join(L) + "\n")
print("wrote", a.out, round(time.time() - t00, 1))
