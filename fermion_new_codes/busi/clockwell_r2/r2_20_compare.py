"""Second reader: compare the two .wip reads' printed tables and judgment numbers with the second reader's own values
(r2_*.json from r2_10/r2_12/r2_13/r2_14/r2_15). Read-only on inputs. Output: r2_20_compare.json and printed summary.
argv: filtered_wip baseline_wip out_json"""
import json, re, sys, os, math
import numpy as np
O = "<internal>/clockwell_r2"
FW, BW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
G = {"0": "0", "1/2": "0p5", "1": "1", "2": "2"}
def J(arm, n, g): return json.load(open(os.path.join(O, "r2_%s_even_n%s_g%s_v1.json" % (arm, n, G[g]))))
D = {(arm, n, g): J(arm, n, g) for arm in ("filtered", "onsite") for n in ("2", "3") for g in G}

def tables(path):
    lines = open(path).read().splitlines(); out = {}; cur = None; i = 0
    while i < len(lines):
        m = re.match(r"\*\*Table ([FB]\d+b?)\.", lines[i])
        if m: cur = m.group(1); out[cur] = []; sub = 0
        elif lines[i].startswith("Computed primary-versus-protected"): cur = (cur or "") + "_dist"; out[cur] = []
        elif lines[i].startswith("|") and cur:
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not set(lines[i].replace("|", "").strip()) <= set("-"): out[cur].append(cells)
        elif lines[i].startswith("## ") : cur = None
        i += 1
    return {k: v[1:] for k, v in out.items()}, {k: v[0] if v else None for k, v in out.items()}
def iv(s):
    m = re.match(r"\[([^,]+), ([^\]]+)\]", s.strip()); return float(m.group(1)), float(m.group(2))
def inside(x, s, rel=1e-9, ab=1e-12):
    a, b = iv(s); sl = ab + rel * max(abs(a), abs(b)); return a - sl <= x <= b + sl
def gnorm(s): return s.strip()
res = {"defects": [], "fails": []}
FT, FH = tables(FW); BT, BH = tables(BW)

def fail(tag, msg): res["fails"].append("%s: %s" % (tag, msg))

# ---------- conversion (F7, B3) ----------
def conv_check(rows, tag):
    for r in rows:
        arm = "filtered" if r[0].startswith("filtered") else "onsite"; n, g = r[1], gnorm(r[2])
        d = D[(arm, n, g)]; c = d["conversion"]
        cnt = int(r[3].split()[0])
        if cnt != c["count_flag"]: fail(tag, "%s n%s g%s count %d vs %d" % (arm, n, g, cnt, c["count_flag"]))
        own = d["own_pLL_by_clock"]; best = max(own, key=lambda k: own[k][0])
        mclk = float(re.search(r"\(([^)]+)\)", r[4]).group(1)); mval = float(r[4].split()[0])
        if float(best) != mclk and max(o[0] for o in own.values()) > 1e-9: fail(tag, "%s n%s g%s argmax %s vs %s" % (arm, n, g, best, mclk))
        if abs(mval - own[best][0]) > 5e-6 and mval != 0: fail(tag, "%s n%s g%s max %s vs %s" % (arm, n, g, mval, own[best][0]))
        mk = "%s" % float(mclk)
        if not inside(own[mk][0], r[5]): fail(tag, "pLL@max not inside %s %s" % (r[5], own[mk][0]))
        if not inside(own[mk][1], r[6]): fail(tag, "CHL@max not inside %s %s" % (r[6], own[mk][1]))
        if not inside(own["4.0"][0], r[7]): fail(tag, "pLL@4 not inside %s %s %s" % (arm, n, g))
        if not inside(own["4.0"][1], r[8]): fail(tag, "CHL@4 not inside %s %s %s" % (arm, n, g))
conv_check(FT["F7"], "F7"); conv_check(BT["B3"], "B3")

# ---------- populations (F8, B4) ----------
def pop_check(rows, arm, tag):
    for r in rows:
        n, g = r[0], gnorm(r[1]); d = D[(arm, n, g)]; p4 = d["own_pops"]["4.0"]
        for col, key in ((2, "core_H"), (3, "core_L"), (4, "lead_H"), (5, "lead_L")):
            v = float(r[col])
            if abs(v - p4[key]) > 5e-6 * max(1, abs(v)) + 1e-9: fail(tag, "%s n%s g%s %s %s vs %s" % (arm, n, g, key, v, p4[key]))
        if n == "3":
            h, l, tot = [float(x) for x in r[7].split(",")[0].split("/")]
            p1 = d["own_pops"]["1.0"]; p075 = d["own_pops"]["0.75"]
            m1 = abs(h - p1["1_H"]) < 5e-6 and abs(l - p1["1_L"]) < 5e-6
            m075 = abs(h - p075["1_H"]) < 5e-6 and abs(l - p075["1_L"]) < 5e-6
            res.setdefault("B2_label", []).append({"arm": arm, "g": g, "printed": [h, l, tot], "label_in_wip": "t = 0.75",
                                                   "matches_t1": m1, "matches_t075": m075, "record_clock": d["B2"]["record_clock"],
                                                   "own_t1": [p1["1_H"], p1["1_L"]], "own_t075": [p075["1_H"], p075["1_L"]]})
pop_check(FT["F8"], "filtered", "F8"); pop_check(BT["B4"], "onsite", "B4")

# ---------- purification (F9, B5) ----------
def pur_check(rows, tag):
    for r in rows:
        arm = "filtered" if r[0].startswith("filtered") else "onsite"; g = gnorm(r[1]); p = D[(arm, "3", g)]["purification"]
        for col, key in ((2, "c2"), (3, "D2"), (4, "Cq")):
            a, b = iv(r[col]); A, B = p[key]
            if abs(a - A) > 1e-7 * max(1, abs(A)) or abs(b - B) > 1e-7 * max(1, abs(B)): fail(tag, "%s g%s %s %s vs %s" % (arm, g, key, r[col], p[key]))
            if not a > 0: fail(tag, "%s nonpositive" % key)
        for col, key in ((6, "B_second"), (7, "Bwall_first"), (8, "Bwall_second")):
            if abs(float(r[col]) - p[key]) > 5e-6 * max(1, p[key]): fail(tag, "%s g%s %s" % (arm, g, key))
        for col, key in ((9, "Ucond_fin"), (10, "Ucum_fin"), (11, "Ucond_ext"), (12, "Ucum_ext")):
            if abs(float(r[col]) - p["mine"][key]) > 5e-5: fail(tag, "%s g%s %s %s vs %s" % (arm, g, key, r[col], p["mine"][key]))
        if r[14] != "no" or p["exclusion_fin"] or p["exclusion_ext"]: fail(tag, "exclusion verdict")
pur_check(FT["F9"], "F9"); pur_check(BT["B5"], "B5")

# ---------- wall (F10, B6) ----------
def wall_check(rows, tag):
    for r in rows:
        arm = "filtered" if r[0].startswith("filtered") else "onsite"; n, g = r[1], gnorm(r[2]); b = D[(arm, n, g)]["wall"]["bounds"]
        if b is None: res.setdefault("wall_missing", []).append([arm, n, g]); continue
        for k, col in zip((4, 8, 12, 16), (3, 4, 5, 6)):
            if abs(float(r[col]) - b[k]) > 5e-6 * max(1, b[k]): fail(tag, "%s n%s g%s Bwall(%d) %s vs %s" % (arm, n, g, k // 4, r[col], b[k]))
wall_check(FT["F10"], "F10"); wall_check(BT["B6"], "B6")

# ---------- return history (F11, B7) ----------
cond = {}
def ret_check(rows, tag):
    for r in rows:
        arm = "filtered" if r[0].startswith("filtered") else "onsite"; n, g = r[1], gnorm(r[2]); d = D[(arm, n, g)]
        m = d["return_mine"]; pr = {float(k): v for k, v in m["p_return"].items()}
        if not inside(m["p_escape"], r[3]): fail(tag, "%s n%s g%s escape" % (arm, n, g))
        for t, col in ((2.0, 4), (3.0, 5), (4.0, 6)):
            if not inside(pr[t], r[col]): fail(tag, "%s n%s g%s p_return(%s) %s vs %s" % (arm, n, g, t, pr[t], r[col]))
        tb = max(pr, key=lambda t: pr[t]); mclk = float(re.search(r"\(([^)]+)\)\s*$", r[7]).group(1))
        if tb != mclk: fail(tag, "%s n%s g%s argmax %s vs %s" % (arm, n, g, tb, mclk))
        if not inside(pr[tb], r[7].split("(")[0]): fail(tag, "%s n%s g%s max value" % (arm, n, g))
        ro = d["readout"]["returns"]
        lows = [x["pj_int"][0] for x in ro if x["t"] > 1.0]
        cond[(arm, n, g)] = {"max_cond": max(pr[t] / m["p_escape"] for t in pr if t > 1.0), "min_joint_lower_after_te": min(lows),
                             "max_joint_lower": max(lows), "max_joint": pr[tb], "argmax": tb, "p_escape": m["p_escape"]}
ret_check(FT["F11"], "F11"); ret_check(BT["B7"], "B7")
res["return_summary"] = {"%s n%s g%s" % k: v for k, v in cond.items()}

# ---------- register / receiver (F12, B8) ----------
lab = {"whole lead light only": ("light", False), "whole lead heavy + light": ("all", False),
       "first block (site 2) light only": ("light", True), "first block (site 2) heavy + light": ("all", True)}
def reg_check(rows, arm, tag):
    fe_max = 0.0
    for r in rows:
        n, g, clk = r[0], gnorm(r[1]), float(r[2]); role, first = lab[r[3]]
        regs = [x for x in D[(arm, n, g)]["register"]["rows"] if x["clock"] == clk and x["role"] == role and bool(x["first"]) == first]
        if len(regs) != 1: fail(tag, "row match %s n%s g%s %s %s (%d)" % (arm, n, g, clk, r[3], len(regs))); continue
        x = regs[0]
        if not inside(x["D"], r[4]): fail(tag, "D %s n%s g%s %s %s: %s vs %s" % (arm, n, g, clk, r[3], x["D"], r[4]))
        if not inside(x["Fe"], r[6], rel=1e-9, ab=1e-10): fail(tag, "Fe %s n%s g%s %s %s: %s vs %s" % (arm, n, g, clk, r[3], x["Fe"], r[6]))
        if r[5] != x["status"] or x["D_int"][1] <= 1e-4: fail(tag, "status")
        fe_max = max(fe_max, iv(r[6])[1])
        # source label check: icloud readout exists for this cohort?
        has_ro = any(abs(f["clock"] - clk) < 1e-12 for f in D[(arm, n, g)].get("readout", {}).get("receivers", []))
        res.setdefault("Fe_source", []).append([arm, n, g, clk, r[3], r[7], has_ro])
    res.setdefault("Fe_max_printed", {})[tag] = fe_max
reg_check(FT["F12"], "filtered", "F12"); reg_check(BT["B8"], "onsite", "B8")
# also own maxima
for arm in ("filtered", "onsite"):
    res.setdefault("Fe_max_own", {})[arm] = max(x["Fe"] for n in "23" for g in G for x in D[(arm, n, g)]["register"]["rows"])
    res.setdefault("Fe_max_readout_upper", {})[arm] = max(f["Fe_int_saved"][1] for n in "23" for g in G for f in D[(arm, n, g)]["readout"]["receivers"])
    res.setdefault("all_not_decoupled", {})[arm] = all(D[(arm, n, g)]["register"]["all_not_decoupled"] for n in "23" for g in G)

# ---------- purities (F13, B9) ----------
cutmap = {"core (site 0)": "core", "whole lead": "lead", "B1 (site 2)": "B1"}
def purity_check(rows, arm, tag):
    for r in rows:
        n, g, cut = r[0], gnorm(r[1]), cutmap[r[2]]; own = D[(arm, n, g)]["own_purity"][cut]
        for col, t in ((3, "1.0"), (4, "2.0"), (5, "4.0")):
            if not inside(own[t], r[col], rel=0, ab=1.1e-8): fail(tag, "%s n%s g%s %s t%s %s vs %s" % (arm, n, g, cut, t, own[t], r[col]))
purity_check(FT["F13"], "filtered", "F13"); purity_check(BT["B9"], "onsite", "B9")

# ---------- odd (B10) ----------
odd = {n: json.load(open(os.path.join(O, "r2_odd_n%s.json" % n))) for n in "23"}
olab = {"whole lead light": ("light", False), "whole lead all": ("all", False), "first block light": ("light", True), "first block all": ("all", True)}
for r in BT["B10"]:
    n, clk = r[0], float(r[1]); role, first = olab[r[2]]
    xs = [x for x in odd[n]["rows"] if x["clock"] == clk and x["role"] == role and bool(x["first"]) == first]
    if len(xs) != 1: fail("B10", "match %s" % r); continue
    if not inside(xs[0]["D"], r[3]): fail("B10", "D %s %s" % (r, xs[0]["D"]))

# ---------- cohort deviation tables (F5, B1): own max deviations ----------
dev_rows = []
for arm in ("filtered", "onsite"):
    for n in "23":
        for g in G:
            d = D[(arm, n, g)]
            dev_rows.append({"arm": arm, "n": n, "g": g, "md5": d["report_md5"], "term_ok": d["termination"]["ok"],
                             "checks": d["termination"]["check_rows"], "dev_pLL": max(d["conversion"]["dev"]["pLL_density"], d["conversion"]["dev"]["pLL_vector"]),
                             "dev_CHL": max(d["conversion"]["dev"]["CHL_density"], d["conversion"]["dev"]["CHL_vector"]),
                             "dev_S": d["entropy"]["max_dev"], "S_rows": d["entropy"]["rows"], "dev_D": d["register"]["max_dev_D"],
                             "dev_U": d.get("purification", {}).get("max_dev"), "traj": d["traj_check_t1"],
                             "halo": d.get("halo_audit"), "md5_mismatch": len(d["md5_log"]["mismatch"]),
                             "ro_md5": d["readout"]["report_md5"], "ro_term": d["readout"]["termination"]["ok"],
                             "ro_dev_Fe": d["readout"]["receiver_max_dev"], "ro_dev_pur": d["readout"]["purity_max_dev"],
                             "ro_ret_inside": d["readout"]["return_all_inside"], "ro_rows": [d["readout"]["receiver_rows"], d["readout"]["purity_rows"], d["readout"]["return_rows"]]})
res["cohorts"] = dev_rows

# ---------- conversion claims ----------
cc = {}
for n in "23":
    for g in ("1/2", "1", "2"):
        f = D[("filtered", n, g)]["own_pLL_by_clock"]; o = D[("onsite", n, g)]["own_pLL_by_clock"]
        above = [c for c in f if f[c][0] > o[c][0] + 1e-12]
        cc["n%s g%s" % (n, g)] = {"max_f": max(v[0] for v in f.values()), "max_o": max(v[0] for v in o.values()), "clocks_filtered_above_onsite": above}
    for g in G:
        for arm in ("filtered", "onsite"):
            rows = D[(arm, n, g)]["conversion"]["rows"]
            cc.setdefault("flags", {})["%s n%s g%s" % (arm, n, g)] = [r["clock"] for r in rows if r["flag"]]
res["conversion_claims"] = cc

# ---------- g = 0 identity control ----------
gz = {}
for n in "23":
    a = D[("filtered", n, "0")]; b = D[("onsite", n, "0")]
    gz[n] = {"pops": max(abs(a["own_pops"][c][k] - b["own_pops"][c][k]) for c in a["own_pops"] for k in a["own_pops"][c]),
             "pLL": max(abs(a["own_pLL_by_clock"][c][0] - b["own_pLL_by_clock"][c][0]) for c in a["own_pLL_by_clock"]),
             "D": max(abs(x["D"] - y["D"]) for x, y in zip(a["register"]["rows"], b["register"]["rows"])),
             "Fe": max(abs(x["Fe"] - y["Fe"]) for x, y in zip(a["register"]["rows"], b["register"]["rows"]))}
res["g0_identity"] = gz

# ---------- halo header audit (any record kind) ----------
HALO = "endpoint-copy halo; N_-1=N_0=1/2; N_n=N_(n-1)=1; declared convention, not a measurement"
ha = {}
for n in "23":
    for g in G:
        p = "<internal>/written_report.md" % (n, G[g])
        tot = good = 0; kinds = {}
        for l in open(p):
            if '"header"' not in l: continue
            d = json.loads(l)["data"]
            if isinstance(d, dict) and "header" in d:
                tot += 1; good += d["header"] == HALO; kinds[json.loads(l)["kind"]] = kinds.get(json.loads(l)["kind"], 0) + 1
        ha["n%s g%s" % (n, g)] = {"top_level_header_rows": tot, "correct": good, "kinds": kinds}
res["halo_all_kinds"] = ha

# ---------- table header labels in the filtered .wip (#145) ----------
txt = open(FW).read().splitlines()
hdr = []
for i, l in enumerate(txt):
    if l.startswith("**Table F"):
        hdr.append([l.split(".")[0].strip("*"), "endpoint-copy halo" in l and "N_-1 = 1/2" in l and "N_n = 1" in l])
    if l.startswith("Computed primary-versus-protected"): hdr.append(["F9 distance/switch-work sub-table (line %d)" % (i + 1), "endpoint-copy halo" in l])
res["wip_table_headers"] = hdr
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({"fails": res["fails"], "B2_label": res.get("B2_label"), "Fe_max_printed": res["Fe_max_printed"], "Fe_max_own": res["Fe_max_own"],
                  "Fe_max_readout_upper": res["Fe_max_readout_upper"], "conv": res["conversion_claims"], "g0": gz, "halo": ha, "hdr": hdr,
                  "wall_missing": res.get("wall_missing")}, indent=0, default=str)[:12000])
