#!/usr/bin/env python
"""Cold correctness reading of D13_classification_v1.1 (solve_a): an independent recomputation of every number group
the delivery prints, from the stored d13/summary jsons only (B2 production kappa 1, 2, 3; the record-interpolation control
v6o_ctrl_B2 kappa 1, 2; s8 kappa 1, 2, 3), with code written for this reading (the delivery's script was read only to locate
its inputs). No linear algebra: sums, differences and a trapezoid rule on stored scalars (own implementation, no numpy.trapz).

Box CPU, nice 10, json only. Writes d13_v1_1_recompute.json and d13_v1_1_recompute.md to --out.
"""
import argparse, hashlib, json, math, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="<internal>/d13_v1_1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t00 = time.time()

RS = 64.0            # a per r_s
ALLOW = 1e-4         # the total fluence allowance per window, of L_H(kappa_T) x the window
DARK = 1e-3          # the darkness threshold
B2 = "<internal>/prod"
CT = "<internal>/v6o_ctrl_B2"
S8 = "<internal>/prod"
S8B = "<internal>/s8_prod_K18"
READOUT = "<internal>/page_d13_readout_v1d.py"
DELIV = "<internal>/D13_class_v1"

# the md5s the delivery states (its section 1 / Files, and its json's provenance block, read on the box)
EXPECT = {
    "B2_k1_d13": "50fe59a03edc93837980127cbbcec51b", "B2_k1_summary": "b8c7edc73dc90619561674eb6b1e726f",
    "B2_k2_d13": "d19ccb67b023255cf99afbac22d0eb4a", "B2_k2_summary": "207cc3146a31f307d7931b7a0d624378",
    "B2_k3_d13": "78f26e42db7d7b1e6443c25e66a4ba90", "B2_k3_summary": "936565a2ffbf367e17fc642a8667552d",
    "ctrl_k1_d13": "dbe908308319809f6b4a70085d4cb792", "ctrl_k1_summary": "d7b14de1490d339def8db60397764c3e",
    "ctrl_k2_d13": "5a8b1b878d01556981521d7b5e952ead", "ctrl_k2_summary": "efd5871ecbab764bd290d854aaf98f82",
    "s8_k1_d13": "3ff51aa06e5e0d9b3ee50bcd9980491f", "s8_k1_summary": "2c5f1d65ca72877e883c71ca4afe8b8f",
    "s8_k2_d13": "c5fb6e1ec732371ea9d753369c2c64ff", "s8_k2_summary": "21badaf597d9233b81e039df949a059c",
    "s8_k3_d13": "615e5e625ed9de7f49ada483e6a88150", "s8_k3_summary": "ccefb50b9a9721cc637a39d1150559d0",
    "readout": "70383bb15fa5c642a1ac7d570e2f923e", "deliv_script": "2ad93fc8d10377168600a7262f632ee3",
    "deliv_json": "fd8846161d58704399ae1665436b2101", "deliv_md": "84164990830c418d2bea91bc4f74775e",
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def w(k):
    return 16.0 * k


def trap(ts, vs):
    """own trapezoid rule; ts in a (not r_s)."""
    s = 0.0
    for i in range(1, len(ts)):
        s += 0.5 * (vs[i] + vs[i - 1]) * (ts[i] - ts[i - 1])
    return s


def rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


# ----------------------------------------------------------------------------------------------------------------- inputs
files = {}
for k in (1, 2, 3):
    files["B2_k%d_d13" % k] = os.path.join(B2, "v6o_B2_k%d_d13.json" % k); files["B2_k%d_summary" % k] = os.path.join(B2, "v6o_B2_k%d_summary.json" % k)
    files["s8_k%d_d13" % k] = os.path.join(S8, "v6o_s8_k%d_d13.json" % k); files["s8_k%d_summary" % k] = os.path.join(S8, "v6o_s8_k%d_summary.json" % k)
    files["s8b_k%d_d13" % k] = os.path.join(S8B, "v6o_s8_k%d_d13.json" % k)
for k in (1, 2):
    files["ctrl_k%d_d13" % k] = os.path.join(CT, "v6o_ctrl_B2_k%d_d13.json" % k); files["ctrl_k%d_summary" % k] = os.path.join(CT, "v6o_ctrl_B2_k%d_summary.json" % k)
files["readout"] = READOUT
files["deliv_script"] = os.path.join(DELIV, "code", "d13_classification_v1.py"); files["deliv_json"] = os.path.join(DELIV, "d13_class_v1.json"); files["deliv_md"] = os.path.join(DELIV, "D13_class_v1.md")
MD5 = {k: (os.path.realpath(p), md5(p)) for k, p in files.items()}
md5_ok = {k: (MD5[k][1] == v) for k, v in EXPECT.items()}
s8b_same = {k: MD5["s8b_k%d_d13" % k][1] == MD5["s8_k%d_d13" % k][1] for k in (1, 2, 3)}
n_d13 = {"B2": len([f for f in os.listdir(B2) if f.endswith("_d13.json")]), "ctrl": len([f for f in os.listdir(CT) if f.endswith("_d13.json")]), "s8": len([f for f in os.listdir(S8) if f.endswith("_d13.json")])}

J = {("B2", k): json.load(open(files["B2_k%d_d13" % k])) for k in (1, 2, 3)}
J.update({("ctrl", k): json.load(open(files["ctrl_k%d_d13" % k])) for k in (1, 2)})
J.update({("s8", k): json.load(open(files["s8_k%d_d13" % k])) for k in (1, 2, 3)})
S = {("B2", k): json.load(open(files["B2_k%d_summary" % k])) for k in (1, 2, 3)}
S.update({("ctrl", k): json.load(open(files["ctrl_k%d_summary" % k])) for k in (1, 2)})
S.update({("s8", k): json.load(open(files["s8_k%d_summary" % k])) for k in (1, 2, 3)})


def rows(name, k):
    return {float(r["t_over_rs"]): r for r in S[(name, k)]["rows"]}


def pc(name, k, t):
    c = J[(name, k)]["clocks"].get("%010.4f" % t)
    return (c or {}).get("per_channel", {}).get(str(k))


def LH(name, t, ks=(1, 2)):
    v = 0.0
    for k in ks:
        r = rows(name, k).get(float(t))
        if r is None or r.get("L_H_kappaT") is None:
            return None
        v += r["L_H_kappaT"]
    return v


def LH_int(name, t0, t1):
    """integral of the two-channel comparator over the window, energy units (L per a times a); None if L_H is undefined on a checkpoint inside."""
    ts = sorted(t for t in rows(name, 1) if t0 - 1e-9 <= t <= t1 + 1e-9)
    vs = [LH(name, t) for t in ts]
    if len(ts) < 2 or any(v is None for v in vs):
        return None, None, None
    return trap([t * RS for t in ts], vs), sum(vs) / len(vs), ts


# ------------------------------------------------------------------------------------------------------- (a) the ledger
def windows(name, k):
    out = {}
    for x in J[(name, k)]["windows"][str(k)]:
        if x["kind"].startswith("between") or x.get("second_term_D") is None:
            continue
        out[(float(x["t0_over_rs"]), float(x["t1_over_rs"]), x["cut"])] = x
    return out


WB = {k: windows("B2", k) for k in (1, 2, 3)}
WC = {k: windows("ctrl", k) for k in (1, 2)}
Q = ("second_term_D", "crossing_current", "crossing_jumps", "vacuum_change", "telescoping_residual", "inventory_D_start", "inventory_D_end")


def wsum(W, ks, key, q):
    if not all(key in W[k] for k in ks):
        return None
    return sum(w(k) * W[k][key][q] for k in ks)


def inside16(k, t):
    c = pc("B2", k, t)
    return w(k) * (c["E_inv"] - c["inventory"]["16"]["E_D"])


A = []
for key in sorted(WB[1]):
    t0, t1, cut = key
    if t1 - t0 < 13.0 - 1e-9:
        continue
    r = {"t0": t0, "t1": t1, "cut": cut, "window_rs": t1 - t0}
    for q in Q:
        r[q + "_12"] = wsum(WB, (1, 2), key, q)
    r["second_term_k3"] = wsum(WB, (3,), key, "second_term_D")
    r["creation_12"] = r["crossing_jumps_12"] - r["vacuum_change_12"]
    r["ctrl_second_term_12"] = wsum(WC, (1, 2), key, "second_term_D")
    r["interp_change"] = (r["second_term_D_12"] - r["ctrl_second_term_12"]) if r["ctrl_second_term_12"] is not None else None
    I, mean, ts = LH_int("B2", t0, t1)
    r["LH_int"] = I; r["LH_mean_checkpoints"] = mean; r["LH_mean_trapezoid"] = (I / ((t1 - t0) * RS)) if I else None
    r["allowance"] = ALLOW * I if I else None
    r["dark"] = DARK * I if I else None
    e = {"telescoping": abs(r["telescoping_residual_12"]), "kappa3": abs(r["second_term_k3"]), "interp": abs(r["interp_change"]) if r["interp_change"] is not None else None}
    r["err"] = e; r["err_total"] = sum(v for v in e.values() if v is not None)
    if r["allowance"]:
        r["ratio"] = {q_: (v / r["allowance"] if v is not None else None) for q_, v in e.items()}; r["ratio_total"] = r["err_total"] / r["allowance"]
        r["st_over_LHdt"] = r["second_term_D_12"] / I
    else:
        r["ratio"] = None; r["ratio_total"] = None; r["st_over_LHdt"] = None
    st = r["second_term_D_12"]
    r["creation_over_st"] = r["creation_12"] / st; r["kappa3_over_st"] = r["second_term_k3"] / st
    r["interp_over_st"] = (r["interp_change"] / st) if r["interp_change"] is not None else None
    r["inv_start_over_st"] = r["inventory_D_start_12"] / st
    if cut == "16":
        d_in = sum(inside16(k, t1) - inside16(k, t0) for k in (1, 2))
        r["inside16_change_12"] = d_in
        r["inside16_over_st"] = d_in / st
        r["E_inv_change_summary_12"] = sum(rows("B2", k)[t1]["E_inv_weighted"] - rows("B2", k)[t0]["E_inv_weighted"] for k in (1, 2))
        r["E_inv_change_direct_12"] = sum(w(k) * (pc("B2", k, t1)["E_inv"] - pc("B2", k, t0)["E_inv"]) for k in (1, 2))
        r["identity_residual"] = r["E_inv_change_summary_12"] - (d_in + st)   # E_inv = inside + beyond 16, so dE_inv = d inside + second term exactly
    A.append(r)

# ------------------------------------------------------------------------------------------------------- (b) the probes
def probes(name, k):
    out = {}
    for p in J[(name, k)]["probes"]:
        refs = {float(q["t_ref_over_rs"]): q for q in p["refs"]}
        out[(float(p["t_over_rs"]), p["cut"])] = {"L": w(k) * p["L_direct"], "refs": {tr: {"L_old": w(k) * q["L_old"], "L_sea": w(k) * q["L_sea"], "check": w(k) * q["check_L_minus_direct"]} for tr, q in refs.items()},
                                                  "first_return": p.get("first_return_rs"), "before": p.get("before_first_return")}
    return out


PB = {k: probes("B2", k) for k in (1, 2, 3)}
PS = {k: probes("s8", k) for k in (1, 2, 3)}
B = []
for key in sorted(PB[1]):
    t, cut = key
    for tr in (12.0, 20.0, 40.0):
        L = sum(PB[k][key]["L"] for k in (1, 2))
        Lo = sum(PB[k][key]["refs"][tr]["L_old"] for k in (1, 2)); Ls = sum(PB[k][key]["refs"][tr]["L_sea"] for k in (1, 2))
        ch = sum(abs(PB[k][key]["refs"][tr]["check"]) for k in (1, 2))
        k3 = PB[3][key]["refs"][tr]["L_sea"]
        B.append({"t": t, "cut": cut, "t_ref": tr, "L": L, "L_old": Lo, "L_sea": Ls, "check": ch, "k3_L_sea": k3, "LH": LH("B2", t), "abs_Lold_over_L": abs(Lo) / abs(L),
                  "first_return": PB[1][key]["first_return"], "before": PB[1][key]["before"], "join_max_rel_dv": pc("B2", 1, t)["join"]["max_rel_dv"]})
BW = []
for cut in ("16", "30"):
    for tr in (12.0, 20.0, 40.0):
        pts = sorted([r for r in B if r["cut"] == cut and r["t_ref"] == tr], key=lambda r: r["t"])
        for i in range(len(pts)):
            js = [j for j in range(i, len(pts)) if pts[j]["t"] - pts[i]["t"] >= 13.0 - 1e-9]
            if not js:
                continue
            seg = pts[i:js[0] + 1]
            ts = [r["t"] * RS for r in seg]
            F = {q: trap(ts, [r[q] for r in seg]) for q in ("L", "L_old", "L_sea", "check", "k3_L_sea", "LH")}
            al = ALLOW * F["LH"]
            e = {"backprop": abs(F["check"]), "kappa3": abs(F["k3_L_sea"])}
            BW.append({"cut": cut, "t_ref": tr, "t0": seg[0]["t"], "t1": seg[-1]["t"], **{q + "_fl": v for q, v in F.items()}, "allowance": al, "dark": DARK * F["LH"],
                       "err": e, "err_total": sum(e.values()), "ratio_total": sum(e.values()) / al, "ratio_kappa3": e["kappa3"] / al,
                       "abs_Lsea_over_L": abs(F["L_sea"]) / abs(F["L"]), "abs_Lold_over_L": abs(F["L_old"]) / abs(F["L"]), "Lsea_over_LHdt": F["L_sea"] / F["LH"],
                       "k3_over_Lsea": F["k3_L_sea"] / F["L_sea"], "check_over_absL": abs(F["check"]) / abs(F["L"]), "check_over_absLsea": abs(F["check"]) / abs(F["L_sea"])})

# ------------------------------------------------------------------------------------------- (c) the exterior shells per window
HK = ("delta_hard_E_out", "delta_hard_E_in", "delta_soft_E_out", "delta_soft_E_in", "delta_hard_E_out_P", "delta_hard_E_in_P", "delta_soft_E_out_P", "delta_soft_E_in_P")
HW = {k: {(float(x["t0_over_rs"]), float(x["t1_over_rs"])): x for x in J[("B2", k)]["hard_windows"][str(k)]} for k in (1, 2, 3)}
C = []
for key in sorted(HW[1]):
    t0, t1 = key
    if t1 - t0 < 13.0 - 1e-9:
        continue
    r = {"t0": t0, "t1": t1}
    for q in HK:
        r[q + "_12"] = sum(w(k) * HW[k][key][q] for k in (1, 2)); r[q + "_k3"] = w(3) * HW[3][key][q]; r[q + "_123"] = r[q + "_12"] + r[q + "_k3"]
    I, _, _ = LH_int("B2", t0, t1)
    r["LH_int"] = I
    r["softP_over_LHdt"] = (r["delta_soft_E_out_P_12"] / I) if I else None
    r["hardP_over_LHdt"] = (r["delta_hard_E_out_P_12"] / I) if I else None
    r["bg_hard_out_123_over_12"] = r["delta_hard_E_out_123"] / r["delta_hard_E_out_12"]
    r["bg_hard_out_k3_over_12"] = r["delta_hard_E_out_k3"] / r["delta_hard_E_out_12"]
    r["P_hard_out_k3_over_12"] = r["delta_hard_E_out_P_k3"] / r["delta_hard_E_out_P_12"]
    r["P_soft_out_k3_over_12"] = r["delta_soft_E_out_P_k3"] / r["delta_soft_E_out_P_12"]
    r["P_hard_out_k3_over_allow"] = (abs(r["delta_hard_E_out_P_k3"]) / (ALLOW * I)) if I else None
    r["P_soft_out_k3_over_allow"] = (abs(r["delta_soft_E_out_P_k3"]) / (ALLOW * I)) if I else None
    C.append(r)
# the shells' totals per clock: the block basis (block_clock_over_rs) against the content; the jumps of the block-ground totals at the block clocks
SH = []
clocks = sorted(float(t) for t in J[("B2", 1)]["clocks"])
for t in clocks:
    row = {"t": t}
    for k in (1, 2, 3):
        c = pc("B2", k, t); sh = c.get("shells") or {}; tot = sh.get("totals") or {}
        row["block_clock_k%d" % k] = sh.get("block_clock_over_rs")
        for q in ("hard_E_out", "hard_E_out_P", "soft_E_out", "soft_E_out_P", "hard_E_in", "hard_E_in_P"):
            row["%s_k%d" % (q, k)] = tot.get(q)
    SH.append(row)
jumps = []
for i in range(1, len(SH)):
    p, r = SH[i - 1], SH[i]
    for k in (1, 2, 3):
        if r["block_clock_k%d" % k] != p["block_clock_k%d" % k]:
            jumps.append({"t": r["t"], "k": k, "d_hard_E_out_bg": w(k) * (r["hard_E_out_k%d" % k] - p["hard_E_out_k%d" % k]), "d_hard_E_out_P": w(k) * (r["hard_E_out_P_k%d" % k] - p["hard_E_out_P_k%d" % k]),
                          "d_hard_E_in_bg": w(k) * (r["hard_E_in_k%d" % k] - p["hard_E_in_k%d" % k])})
# in-decade steps (no basis change) of the block-ground hard out, weighted, kappa 1+2: the typical 2 r_s step
steps_in = []
for i in range(1, len(SH)):
    p, r = SH[i - 1], SH[i]
    if all(r["block_clock_k%d" % k] == p["block_clock_k%d" % k] for k in (1, 2)):
        steps_in.append((r["t"], sum(w(k) * (r["hard_E_out_k%d" % k] - p["hard_E_out_k%d" % k]) for k in (1, 2))))

# ------------------------------------------------------------------------------------- the collar's signed split per checkpoint
COL = []
for t in clocks:
    row = {"t": t}
    ok = True
    for q in ("E_flux_out_soft", "E_flux_out_hard", "E_flux_in_soft", "E_flux_in_hard"):
        row[q] = 0.0
    for k in (1, 2):
        c = pc("B2", k, t); cs = c.get("collar_split")
        if not cs:
            ok = False; break
        for q in ("E_flux_out_soft", "E_flux_out_hard", "E_flux_in_soft", "E_flux_in_hard"):
            row[q] += w(k) * cs[q]
        row["soft_out_k%d" % k] = cs["E_flux_out_soft"]; row["L_k%d" % k] = cs["L"]; row["sites_valid_k%d" % k] = cs["sites_valid"]
        # the first nonzero FFT bin against the threshold: q1 = 2 pi / (4 L); omega_+(q1) = N_c E(q1) + v_c sin(2 q1)/2 (E = sqrt(sin^2 q + (kappa/R_c)^2)),
        # |omega_-(-q1)| the same by symmetry; the q = 0 bin: omega_+(0) = N_c kappa/R_c with group velocity v_c (the frame-drag term alone)
        Lw = cs["L"]; q1 = 2.0 * math.pi / (4 * Lw); kr = k / cs["R_c"]; E1 = math.sqrt(math.sin(q1) ** 2 + kr ** 2)
        om1 = cs["N_c"] * E1 + 0.5 * cs["v_c"] * math.sin(2 * q1)
        vg1 = cs["N_c"] * math.sin(q1) * math.cos(q1) / E1 + cs["v_c"] * math.cos(2 * q1)
        row["om1_k%d" % k] = om1; row["om1_margin_k%d" % k] = om1 / cs["omega_th"] - 1.0; row["vg1_k%d" % k] = vg1
        row["om0_k%d" % k] = cs["N_c"] * kr; row["vg0_k%d" % k] = cs["v_c"]; row["omega_th"] = cs["omega_th"]; row["v_c"] = cs["v_c"]; row["N_c"] = cs["N_c"]
    if not ok:
        continue
    row["out"] = row["E_flux_out_soft"] + row["E_flux_out_hard"]; row["in"] = row["E_flux_in_soft"] + row["E_flux_in_hard"]
    row["in_over_out_signed"] = (abs(row["in"]) / row["out"]) if row["out"] > 0 else None     # the delivery's ratio (undefined where out <= 0)
    row["abs_in_over_abs_out"] = abs(row["in"]) / abs(row["out"]) if row["out"] != 0 else None  # PAGE v4.1 T5's ratio
    row["join_max_rel_dv"] = pc("B2", 1, t)["join"]["max_rel_dv"]
    fr = (pc("B2", 1, t).get("first_return") or {})
    row["first_return_16"] = (fr.get("16") or {}).get("first_return_rs"); row["first_return_30"] = (fr.get("30") or {}).get("first_return_rs")
    COL.append(row)

# ------------------------------------------------------------------------------------------------------------ units check
units = []
for k in (1, 2):
    R = rows("B2", k); ts = sorted(R)
    for i in range(1, len(ts)):
        r0, r1 = R[ts[i - 1]], R[ts[i]]
        if r1.get("L_interval") and abs(r1["L_interval"]) > 1e-8:
            units.append((k, ts[i], (r1["collar_flow_acc_weighted"] - r0["collar_flow_acc_weighted"]) / (r1["L_interval"] * (ts[i] - ts[i - 1]) * RS)))

# ----------------------------------------------------------------------------------------------------------------- s8
s8_win = sorted(set((float(x["t0_over_rs"]), float(x["t1_over_rs"])) for k in (1, 2) for x in J[("s8", k)]["windows"][str(k)] if not x["kind"].startswith("between")))
s8_probe_clocks = sorted(set(key for k in (1, 2) for key in PS[k]))
s8_rows = []
for key in s8_probe_clocks:
    for tr in (12.0, 20.0, 40.0):
        L = sum(PS[k][key]["L"] for k in (1, 2)); Lo = sum(PS[k][key]["refs"][tr]["L_old"] for k in (1, 2)); Ls = sum(PS[k][key]["refs"][tr]["L_sea"] for k in (1, 2))
        s8_rows.append({"t": key[0], "cut": key[1], "t_ref": tr, "L": L, "L_old": Lo, "L_sea": Ls, "abs_Lold_over_L": abs(Lo) / abs(L)})
s8_clocks = sorted(float(t) for t in J[("s8", 1)]["clocks"])

# ------------------------------------------------------------------------------- the stored floors named in the delivery's section 4
def find_keys(o, pat, path="", acc=None):
    acc = [] if acc is None else acc
    if isinstance(o, dict):
        for k_, v in o.items():
            if pat in k_.lower() and not isinstance(v, (dict, list)):
                acc.append((path + "/" + k_, v))
            find_keys(v, pat, path + "/" + k_, acc)
    return acc


cheb40 = find_keys(pc("B2", 1, 40.0), "cheb") + find_keys(rows("B2", 1)[40.0], "cheb")

# ================================================================================================= the delivery's claims, checked
A16 = [r for r in A if r["cut"] == "16" and r["allowance"]]; A30 = [r for r in A if r["cut"] == "30" and r["allowance"]]
late16 = [r for r in A16 if r["t0"] >= 60]; late30 = [r for r in A30 if r["t0"] >= 60]
tail16 = [r for r in A16 if r["t0"] >= 76]
pre40_16 = [r for r in A16 if r["t0"] < 40]


def byw(rowsA, t0):
    return [r for r in rowsA if r["t0"] == t0][0]


CL = []   # (claim id, delivery's statement, my value(s))
CL.append(("S1.windows", "34 windows per cut from 20 to 100; ledger windows 13 to 14 r_s; probe windows 16 r_s", {"n_allow_16": len(A16), "n_allow_30": len(A30), "t0_first_allow": A16[0]["t0"], "window_rs": rng([r["window_rs"] for r in A]), "n_windows_total_per_cut": len([r for r in A if r["cut"] == "16"]), "probe_window_rs": rng([(r["t1"] - r["t0"]) for r in BW])}))
CL.append(("S1.allowance", "allowance 2.6e-7 to 5.9e-7 per window; L_H dt 2.6e-3 to 5.9e-3", {"allow_a": rng([r["allowance"] for r in A16 + A30]), "allow_b": rng([r["allowance"] for r in BW]), "LHdt_a": rng([r["LH_int"] for r in A16]), "LHdt_b": rng([r["LH_fl"] for r in BW])}))
CL.append(("S1.LH_20_34", "L_H 2.95e-6 mean over 20 to 34 (allowance 2.57e-7)", {"mean_checkpoints": byw(A16, 20.0)["LH_mean_checkpoints"], "mean_trapezoid": byw(A16, 20.0)["LH_mean_trapezoid"], "allowance_20_34": byw(A16, 20.0)["allowance"], "LH_at_20_to_34": [(t, LH("B2", t)) for t in (20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0)]}))
CL.append(("S1.LH_40_100", "L_H 4.8e-6 to 6.1e-6 from 40 to 100", {"per_clock_40_100": rng([LH("B2", t) for t in clocks if t >= 40]), "clocks_below_4.8e-6": [(t, LH("B2", t)) for t in clocks if t >= 40 and LH("B2", t) < 4.8e-6], "window_mean_a_t0ge40": rng([r["LH_mean_trapezoid"] for r in A16 if r["t0"] >= 40]), "window_mean_b": rng([r["LH_fl"] / ((r["t1"] - r["t0"]) * RS) for r in BW])}))
CL.append(("S1.T7xchecks", "26-40 at 16: 8.000e-4 (T7 8.0001e-4); 36-50: 0.3123 (0.31232); with kappa 3 8.000e-4 + 2.038e-4 = 1.0038e-3 (1.004e-3); (c) 46-60 hard out 0.3807 and _P 0.334", {"st_26_40_16": byw(A16, 26.0)["second_term_D_12"], "st_36_50_16": byw(A16, 36.0)["second_term_D_12"], "k3_26_40_16": byw(A16, 26.0)["second_term_k3"], "st123_26_40_16": byw(A16, 26.0)["second_term_D_12"] + byw(A16, 26.0)["second_term_k3"], "hard_out_46_60": [c["delta_hard_E_out_12"] for c in C if c["t0"] == 46.0][0], "hard_out_P_46_60": [c["delta_hard_E_out_P_12"] for c in C if c["t0"] == 46.0][0]}))
CL.append(("S1.units", "collar flow change over L_interval dt = 1.0000 at every checkpoint (12 of 12)", {"n": len(units), "range": rng([u[2] for u in units])}))
CL.append(("S2.pre.first_return", "every row 0 to 100 before the first return: 227.5 at 16, 198.5 at 30", {"probe_first_return_16": rng([r["first_return"] for r in B if r["cut"] == "16"]), "probe_first_return_30": rng([r["first_return"] for r in B if r["cut"] == "30"]), "clock_first_return_16": rng([r["first_return_16"] for r in COL]), "clock_first_return_30": rng([r["first_return_30"] for r in COL]), "all_before": all(r["before"] for r in B)}))
jd = {r["t"]: r["join_max_rel_dv"] for r in COL}
CL.append(("S2.pre.join", "join max rel dv 4.4e-6 to 7.1e-6 through 56, 5.8e-5 at 58, 4.6e-4 at 60, above 1e-3 from 66, 7.2e-2 at 100", {"through_56": rng([v for t, v in jd.items() if t <= 56]), "58": jd[58.0], "60": jd[60.0], "62": jd[62.0], "64": jd[64.0], "66": jd[66.0], "first_above_1e-3": min(t for t, v in jd.items() if v > 1e-3), "100": jd[100.0]}))
CL.append(("S2a.st_over_LHdt", "second term over L_H dt, 60 to 100: cut 16 7.3 (60) falling to 0.60 to 0.80 (76 to 100); cut 30 32.8 (60) falling to 3.2 (86 to 100)", {"16": [(r["t0"], round(r["st_over_LHdt"], 3)) for r in late16], "30": [(r["t0"], round(r["st_over_LHdt"], 3)) for r in late30], "16_76_100": rng([r["st_over_LHdt"] for r in tail16])}))
CL.append(("S2a.creation", "creation over second term, 60 to 100: 16: 0.007 to 0.051; 30: 0.000 to 0.006", {"16": rng([r["creation_over_st"] for r in late16]), "30": rng([r["creation_over_st"] for r in late30])}))
CL.append(("S2a.kappa3_over_st", "kappa 3 guard over second term, 60 to 100: 16: 0.011 to 0.100; 30: 0.014 to 0.096; (76 to 100 at 16: 5.5 to 10 percent)", {"16": rng([r["kappa3_over_st"] for r in late16]), "30": rng([r["kappa3_over_st"] for r in late30]), "16_76_100": rng([r["kappa3_over_st"] for r in tail16])}))
CL.append(("S2a.interp_over_st", "interp change over second term, 60 to 100: 16: -0.035 to +0.011; 30: -0.058 to +0.028; (76 to 100 at 16: 1 to 3.5 percent; 'from 60 on 1 to 3.5 percent at 16, 1 to 6 percent at 30'; 'up to 100 percent or more in 20 to 40')", {"16": rng([r["interp_over_st"] for r in late16]), "30": rng([r["interp_over_st"] for r in late30]), "abs16_76_100": rng([abs(r["interp_over_st"]) for r in tail16]), "abs16_60_100": rng([abs(r["interp_over_st"]) for r in late16]), "abs30_60_100": rng([abs(r["interp_over_st"]) for r in late30]), "abs16_20_40": rng([abs(r["interp_over_st"]) for r in pre40_16]), "abs30_20_40": rng([abs(r["interp_over_st"]) for r in A30 if r["t0"] < 40])}))
CL.append(("S2a.tel_over_allow", "telescoping over allowance: 16: 3e-4 to 0.019; 30: 4e-4 to 0.016", {"16": rng([r["ratio"]["telescoping"] for r in A16]), "30": rng([r["ratio"]["telescoping"] for r in A30])}))
CL.append(("S2a.kappa3_over_allow", "kappa 3 guard over allowance, 20 to 100: 16: 1.85 to 6.5e4; 30: 9.0 to 5.5e4", {"16": rng([r["ratio"]["kappa3"] for r in A16]), "30": rng([r["ratio"]["kappa3"] for r in A30]), "argmin16": min(A16, key=lambda r: r["ratio"]["kappa3"])["t0"], "argmin30": min(A30, key=lambda r: r["ratio"]["kappa3"])["t0"]}))
CL.append(("S2a.interp_over_allow", "interp over allowance: 16: 17.6 to 1.9e4; 30: 0.018 to 1.9e4", {"16": rng([r["ratio"]["interp"] for r in A16]), "30": rng([r["ratio"]["interp"] for r in A30])}))
CL.append(("S2a.total_over_allow", "total error over allowance: 16: 59 to 8.2e4; 30: 10.1 to 6.8e4", {"16": rng([r["ratio_total"] for r in A16]), "30": rng([r["ratio_total"] for r in A30])}))
CL.append(("S2a.inventory_107", "inventory beyond 16 at the window's start over the second term: 107 at 76 and 86 (0.399/3.72e-3; 0.402/3.77e-3), 20 at 66", {t0: (byw(A16, t0)["inventory_D_start_12"], byw(A16, t0)["second_term_D_12"], byw(A16, t0)["inv_start_over_st"], byw(A16, t0)["crossing_current_12"], byw(A16, t0)["inventory_D_start_12"] / byw(A16, t0)["crossing_current_12"]) for t0 in (66.0, 76.0, 86.0)}))
CL.append(("S2a.inside16", "inside-16 inventory change -2.0 to +1.1 per window from 20 to 100; 1.0 to 18 times the second term over 60 to 100; up to 5e5 before 40; section 3: '10 to 500 times'", {"change_20_100": rng([r["inside16_change_12"] for r in A16]), "abs_ratio_60_100": rng([abs(r["inside16_over_st"]) for r in late16]), "abs_ratio_before_40": rng([abs(r["inside16_over_st"]) for r in pre40_16]), "abs_ratio_20_100": rng([abs(r["inside16_over_st"]) for r in A16]), "identity_residual_max": max(abs(r["identity_residual"]) for r in A16), "E_inv_summary_vs_direct_max": max(abs(r["E_inv_change_summary_12"] - r["E_inv_change_direct_12"]) for r in A16)}))
b16_20 = [r for r in BW if r["cut"] == "16" and r["t_ref"] == 20.0]
CL.append(("S2b.table", "the six (cut, t_ref) rows: |L_sea|/|L|; L_sea/L_H dt; error/allowance", {"%s_%g" % (c_, tr): {"abs_Lsea_over_L": rng([r["abs_Lsea_over_L"] for r in BW if r["cut"] == c_ and r["t_ref"] == tr]), "Lsea_over_LHdt": rng([r["Lsea_over_LHdt"] for r in BW if r["cut"] == c_ and r["t_ref"] == tr]), "ratio_total": rng([r["ratio_total"] for r in BW if r["cut"] == c_ and r["t_ref"] == tr]), "abs_Lold_over_L": rng([r["abs_Lold_over_L"] for r in BW if r["cut"] == c_ and r["t_ref"] == tr])} for c_ in ("16", "30") for tr in (12.0, 20.0, 40.0)}))
CL.append(("S2b.backprop", "back-propagation check 1e-11 of the window fluence", {"check_over_absL": rng([r["check_over_absL"] for r in BW]), "check_over_absLsea": rng([r["check_over_absLsea"] for r in BW]), "check_fl": rng([r["check_fl"] for r in BW])}))
CL.append(("S2b.kappa3_every_row", "the kappa 3 guard alone breaks the allowance in every row (b); overall least 1.85 (a) / 2.7 (b, 30 t_ref 40 40-56: L_sea -5.7e-7, allowance 5.0e-7, guard 1.3e-6)", {"min_ratio_kappa3_b": min(r["ratio_kappa3"] for r in BW), "argmin": [(r["cut"], r["t_ref"], r["t0"]) for r in BW if r["ratio_kappa3"] == min(x["ratio_kappa3"] for x in BW)], "min_total_b": min(r["ratio_total"] for r in BW), "max_total_b": max(r["ratio_total"] for r in BW), "best_row": [(r["L_sea_fl"], r["allowance"], r["err"]["kappa3"], r["ratio_total"]) for r in BW if r["cut"] == "30" and r["t_ref"] == 40.0 and r["t0"] == 40.0], "min_kappa3_a": min(r["ratio"]["kappa3"] for r in A16 + A30), "min_total_a": min(r["ratio_total"] for r in A16 + A30), "max_total_all": max([r["ratio_total"] for r in A16 + A30] + [r["ratio_total"] for r in BW])}))
CL.append(("S2b.ill_conditioned", "+5.6 against -5.4 for 0.22 (16, t_ref 20, 40-56); |L_sea| and |L_old| up to 197 times the direct current", {"row": [(r["L_sea_fl"], r["L_old_fl"], r["L_fl"]) for r in b16_20 if r["t0"] == 40.0], "max_abs_Lsea_over_L": max(r["abs_Lsea_over_L"] for r in BW), "max_abs_Lold_over_L": max(r["abs_Lold_over_L"] for r in BW), "argmax": [(r["cut"], r["t_ref"], r["t0"], r["abs_Lsea_over_L"], r["abs_Lold_over_L"]) for r in BW if r["abs_Lsea_over_L"] == max(x["abs_Lsea_over_L"] for x in BW)]}))
CL.append(("S0.sign_robust", "the total L_sea's sign is robust against the guard (its change is 1e-3 of the value at 16 r_s, t_ref 20)", {"k3_over_Lsea_16_20": rng([r["k3_over_Lsea"] for r in b16_20]), "per_window": [(r["t0"], r["k3_over_Lsea"]) for r in b16_20], "sign_flips_any_row": [(r["cut"], r["t_ref"], r["t0"]) for r in BW if abs(r["k3_over_Lsea"]) >= 1.0]}))
CL.append(("S2b.old_40_52", "at t_ref 40 over 40 to 52 the flux at 16 r_s is old (0.96 to 1.0 with t_ref 40, 0.91 to 1.0 with t_ref 20)", {"t_ref40": [(r["t"], round(r["abs_Lold_over_L"], 3)) for r in B if r["cut"] == "16" and r["t_ref"] == 40.0 and r["t"] <= 52], "t_ref20": [(r["t"], round(r["abs_Lold_over_L"], 3)) for r in B if r["cut"] == "16" and r["t_ref"] == 20.0 and r["t"] <= 52]}))
c38 = [c for c in C if c["t0"] >= 38]; c56 = [c for c in C if c["t0"] >= 56]
CL.append(("S2c.P_ranges", "_P hard out 4.7e-3 to 3.5e-1 per window from 38 to 100; _P soft out 1.2e-2 to 2.2e-2 from 56 to 100 (2.4 to 4.3 L_H dt); later '_P 4.7e-3 to 3.8e-2 over 56 to 100, comparable to the soft 1.2e-2 to 2.1e-2'", {"hardP_38_100": rng([c["delta_hard_E_out_P_12"] for c in c38]), "hardP_56_100": rng([c["delta_hard_E_out_P_12"] for c in c56]), "softP_56_100": rng([c["delta_soft_E_out_P_12"] for c in c56]), "softP_over_LHdt_56_100": rng([c["softP_over_LHdt"] for c in c56]), "hardP_over_LHdt_56_100": rng([c["hardP_over_LHdt"] for c in c56]), "hardP_positive_every_window_56_100": all(c["delta_hard_E_out_P_12"] > 0 for c in c56)}))
CL.append(("S2c.bg_guard", "with kappa 3 the hard outgoing change (block-ground column) moves by factors up to 5 and changes sign in 50 to 68 (0.075 against -0.344 in 54 to 68)", {"bg_54_68": [(c["delta_hard_E_out_12"], c["delta_hard_E_out_123"]) for c in C if c["t0"] == 54.0], "sign_changes_t0": [c["t0"] for c in C if c["delta_hard_E_out_12"] * c["delta_hard_E_out_123"] < 0], "abs_k3_over_12_bg_38_100": rng([abs(c["bg_hard_out_k3_over_12"]) for c in c38]), "abs_123_over_12_bg_38_100": rng([abs(c["bg_hard_out_123_over_12"]) for c in c38])}))
CL.append(("S2c.P_guard (not in the delivery)", "the guard on the _P (deviation) columns: kappa 3's _P hard and soft outgoing change over the kappa 1+2 value, and over the allowance", {"P_hard_k3_over_12_56_100": rng([c["P_hard_out_k3_over_12"] for c in c56]), "P_soft_k3_over_12_56_100": rng([c["P_soft_out_k3_over_12"] for c in c56]), "P_hard_k3_over_allow_56_100": rng([c["P_hard_out_k3_over_allow"] for c in c56]), "P_soft_k3_over_allow_56_100": rng([c["P_soft_out_k3_over_allow"] for c in c56]), "P_hard_sign_changes_56_100": [c["t0"] for c in c56 if c["delta_hard_E_out_P_12"] * c["delta_hard_E_out_P_123"] < 0], "per_window_56_100": [(c["t0"], c["delta_hard_E_out_P_12"], c["delta_hard_E_out_P_k3"], c["delta_soft_E_out_P_12"], c["delta_soft_E_out_P_k3"]) for c in c56]}))
CL.append(("S2c.shells_recomputed", "'recomputed only at the baseline clocks every 10 r_s, so consecutive 14 r_s windows share endpoints and repeat'", {"block_clocks_k1": sorted(set(r["block_clock_k1"] for r in SH)), "hard_E_out_k1_at_0_2_4": [r["hard_E_out_k1"] for r in SH if r["t"] in (0.0, 2.0, 4.0)], "hard_E_out_P_k1_at_0_2_4": [r["hard_E_out_P_k1"] for r in SH if r["t"] in (0.0, 2.0, 4.0)], "bg_hard_out_jumps_at_block_clocks_weighted_k1_k2_k3": jumps, "in_decade_2rs_steps_bg_hard_out_12_abs_max": max(abs(s[1]) for s in steps_in), "in_decade_steps_before_50_abs_max": max(abs(s[1]) for s in steps_in if s[0] < 50)}))
col14 = [r for r in COL if r["t"] >= 14]; col40 = [r for r in COL if r["t"] >= 40]
CL.append(("S2c.collar_soft_zero", "soft outgoing flux exactly 0 at every checkpoint from 14 to 100; L = 32, sites_valid 32; grid 2 pi/128 = 0.049; 2 pi/32 = 0.196; soft band 25 T_H^ref = 0.037; 'only the q = 0 bin (zero group velocity) is soft'", {"soft_out_exactly_zero_from_14_k1": all(r["soft_out_k1"] == 0.0 for r in col14), "soft_out_exactly_zero_from_14_k2": all(r["soft_out_k2"] == 0.0 for r in col14), "soft_out_nonzero_clocks_k1": [(r["t"], r["soft_out_k1"]) for r in COL if r["soft_out_k1"] != 0.0], "soft_out_nonzero_clocks_k2": [(r["t"], r["soft_out_k2"]) for r in COL if r["soft_out_k2"] != 0.0], "L_values": sorted(set(r["L_k1"] for r in COL) | set(r["L_k2"] for r in COL)), "sites_valid": sorted(set(r["sites_valid_k1"] for r in COL) | set(r["sites_valid_k2"] for r in COL)), "omega_th": sorted(set(r["omega_th"] for r in COL)), "grid_2pi_over_128": 2 * math.pi / 128, "2pi_over_32": 2 * math.pi / 32, "om1_margin_k1_from_14": rng([r["om1_margin_k1"] for r in col14]), "om1_margin_k2_from_14": rng([r["om1_margin_k2"] for r in col14]), "om1_margin_k1_0_12": [(r["t"], r["om1_margin_k1"]) for r in COL if r["t"] < 14], "om1_k1_range_from_14": rng([r["om1_k1"] for r in col14]), "vg1_k1_from_14": rng([r["vg1_k1"] for r in col14]), "om0_k1": rng([r["om0_k1"] for r in COL]), "vg0_is_v_c": rng([r["vg0_k1"] for r in COL]), "v_c_range_from_14": rng([r["v_c"] for r in col14]), "v_c_0_12": rng([r["v_c"] for r in COL if r["t"] < 14]), "v_c_critical_k1_at_T14_Nc": None}))
# the critical v_c at which the first bin would turn soft: N_c E1 + v_c sin(2 q1)/2 = omega_th -> v_c = 2 (omega_th - N_c E1) / sin(2 q1)
q1 = 2 * math.pi / 128
for r in COL:
    E1 = math.sqrt(math.sin(q1) ** 2 + (1.0 / 1025.0) ** 2)
    r["v_c_crit_k1"] = 2.0 * (r["omega_th"] - r["N_c"] * E1) / math.sin(2 * q1)
CL[-1][2]["v_c_critical_k1_range"] = rng([r["v_c_crit_k1"] for r in COL]); CL[-1][2]["v_c_minus_crit_from_14_min"] = min(r["v_c"] - r["v_c_crit_k1"] for r in col14)
neg = [r["t"] for r in col40 if r["out"] <= 0]
CL.append(("S5.collar_in_out", "from 40: the outgoing part negative at 9 of 31 checkpoints (46, 56, 62, 68, 76, 78, 84, 92, 96); the largest ratio elsewhere 2.7 percent (70); PAGE v4.1's 3.3 percent at 78 'comes from T5's split, a different stored column'", {"n_from_40": len(col40), "out_negative_at": neg, "n_negative": len(neg), "max_in_over_out_where_out_positive": max((r["in_over_out_signed"], r["t"]) for r in col40 if r["in_over_out_signed"] is not None), "T5_abs_ratio_at_78": [(r["out"], r["in"], r["abs_in_over_abs_out"]) for r in col40 if r["t"] == 78.0], "T5_abs_ratio_above_0.01": [(r["t"], round(r["abs_in_over_abs_out"], 3)) for r in col40 if r["abs_in_over_abs_out"] > 0.01], "max_abs_ratio_from_40": max((r["abs_in_over_abs_out"], r["t"]) for r in col40), "out_in_at_T5_rows": [(r["t"], r["out"], r["in"]) for r in col40 if r["t"] in (40.0, 46.0, 56.0, 70.0, 78.0, 92.0, 100.0)]}))
CL.append(("S2.s8", "s8 ends at 40; ledger windows end at 40; probes at 40 only; |L_old/L| 1.0 at t_ref 20 and 40 at 16 r_s; s8 copy identical", {"s8_last_clock": s8_clocks[-1], "s8_window_t1_max": max(t1 for t0, t1 in s8_win), "s8_probe_clocks": s8_probe_clocks, "s8_probes": [(r["t"], r["cut"], r["t_ref"], round(r["abs_Lold_over_L"], 4)) for r in s8_rows], "s8b_same": s8b_same, "n_d13_files": n_d13}))
CL.append(("S1.ctrl", "control: probes [] and hard_windows [] (ledger only); 12 channels", {"ctrl_probes": {k: len(J[("ctrl", k)]["probes"]) for k in (1, 2)}, "ctrl_hard_windows": {k: len(J[("ctrl", k)]["hard_windows"][str(k)]) for k in (1, 2)}, "ctrl_windows": {k: len(J[("ctrl", k)]["windows"][str(k)]) for k in (1, 2)}, "ctrl_shells_at_60": {k: bool((pc("ctrl", k, 60.0) or {}).get("shells")) for k in (1, 2)}, "n_ctrl_d13": n_d13["ctrl"]}))
CL.append(("S4.floors", "T_cheb 6.3e-7 at 40 for kappa 1 (stored trace-norm floors per clock)", {"cheb_keys_at_40_k1": cheb40}))
CL.append(("S0.ranges", "total counted error 2.7 to 1.9e5 times the allowance; the kappa 3 guard alone exceeds it at every window by 1.85 at the least", {"min_total_any": min([r["ratio_total"] for r in A16 + A30] + [r["ratio_total"] for r in BW]), "max_total_any": max([r["ratio_total"] for r in A16 + A30] + [r["ratio_total"] for r in BW]), "min_kappa3_any": min([r["ratio"]["kappa3"] for r in A16 + A30] + [r["ratio_kappa3"] for r in BW]), "n_windows_with_allowance": len(A16) + len(A30) + len(BW)}))

OUT = {"md5": MD5, "md5_matches_delivery": md5_ok, "s8b_same": s8b_same, "n_d13": n_d13, "ledger": A, "probes": B, "probe_windows": BW, "hard_soft": C, "shell_totals_per_clock": SH, "block_jumps": jumps,
       "collar": COL, "units": units, "s8_probes": s8_rows, "s8_windows": s8_win, "claims": [{"id": c[0], "delivery": c[1], "mine": c[2]} for c in CL], "seconds": time.time() - t00}
json.dump(OUT, open(os.path.join(a.out, "d13_v1_1_recompute.json"), "w"), indent=1, default=float)

f = lambda x, s="%.4e": "-" if x is None else (s % x)
L = ["# d13_v1_1_recompute (busi reading of D13_classification_v1.1; own code; box CPU, json only)", ""]
L.append("## md5 of the inputs against the delivery's provenance")
for k, v in EXPECT.items():
    L.append("- %s: %s %s (%s)" % (k, MD5[k][1], "MATCH" if md5_ok[k] else "MISMATCH, delivery says " + v, MD5[k][0]))
L.append("- s8 copy under page_v4 identical to composed_s8: %s; d13 json counts B2 %d, ctrl %d, s8 %d" % (s8b_same, n_d13["B2"], n_d13["ctrl"], n_d13["s8"]))
L.append("")
L.append("## the delivery's claims against this recomputation")
for c in CL:
    L.append("### %s" % c[0]); L.append("delivery: %s" % c[1]); L.append("mine: %s" % json.dumps(c[2], default=float)); L.append("")
L.append("## (a) ledger rows with an allowance (kappa 1+2 weighted; errors: telescoping | kappa 3 | interp; ratios over the allowance)")
L.append("| window | cut | second term | st/L_H dt | creation | k3 change | interp change | telescoping | allowance | k3/allow | interp/allow | tel/allow | total/allow | inside-16 change | inside/st |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in A:
    if not r["allowance"]:
        continue
    L.append("| %g-%g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["t0"], r["t1"], r["cut"], f(r["second_term_D_12"]), f(r["st_over_LHdt"], "%.3f"), f(r["creation_12"]), f(r["second_term_k3"]), f(r["interp_change"]), f(r["telescoping_residual_12"]), f(r["allowance"]), f(r["ratio"]["kappa3"], "%.3g"), f(r["ratio"]["interp"], "%.3g"), f(r["ratio"]["telescoping"], "%.3g"), f(r["ratio_total"], "%.3g"), f(r.get("inside16_change_12")), f(r.get("inside16_over_st"), "%.3g")))
L.append("")
L.append("## (b) probe windows (16 r_s, 5 probes, own trapezoid)")
L.append("| cut | t_ref | window | L_sea fl | L_old fl | L fl | L_H fl | L_sea/L_H dt | |L_sea|/|L| | allowance | kappa3 guard | k3/allow | total/allow | k3/L_sea |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in BW:
    L.append("| %s | %g | %g-%g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["cut"], r["t_ref"], r["t0"], r["t1"], f(r["L_sea_fl"]), f(r["L_old_fl"]), f(r["L_fl"]), f(r["LH_fl"]), f(r["Lsea_over_LHdt"], "%.3g"), f(r["abs_Lsea_over_L"], "%.3g"), f(r["allowance"]), f(r["err"]["kappa3"]), f(r["ratio_kappa3"], "%.3g"), f(r["ratio_total"], "%.3g"), f(r["k3_over_Lsea"], "%.2e")))
L.append("")
L.append("## (c) shells per window: block-ground and _P columns, kappa 1+2 and the kappa 3 part")
L.append("| window | hard out bg 12 | hard out bg k3 | hard out P 12 | hard out P k3 | soft out P 12 | soft out P k3 | L_H dt | softP/L_H dt | hardP/L_H dt | k3 P hard/allow | k3 P soft/allow |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in C:
    L.append("| %g-%g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (c["t0"], c["t1"], f(c["delta_hard_E_out_12"]), f(c["delta_hard_E_out_k3"]), f(c["delta_hard_E_out_P_12"]), f(c["delta_hard_E_out_P_k3"]), f(c["delta_soft_E_out_P_12"]), f(c["delta_soft_E_out_P_k3"]), f(c["LH_int"]), f(c["softP_over_LHdt"], "%.3g"), f(c["hardP_over_LHdt"], "%.3g"), f(c["P_hard_out_k3_over_allow"], "%.3g"), f(c["P_soft_out_k3_over_allow"], "%.3g")))
L.append("")
L.append("## block-basis jumps of the shells' block-ground hard-out totals at the block clocks (weighted per channel) against the _P totals' change over the same 2 r_s step")
for j in jumps:
    L.append("- T %g kappa %d: block-ground hard out jump %s (hard in %s); _P hard out change %s" % (j["t"], j["k"], f(j["d_hard_E_out_bg"]), f(j["d_hard_E_in_bg"]), f(j["d_hard_E_out_P"])))
L.append("")
L.append("## collar split per checkpoint (kappa 1+2 weighted) with the first-bin margin")
L.append("| T | out soft | out hard | in soft | in hard | out | in | |in|/out (out>0) | |in|/|out| | v_c | omega_+(q1) k1 | margin k1 | margin k2 | join |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in COL:
    L.append("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %.5f | %.5f | %+.4f | %+.4f | %s |" % (r["t"], f(r["E_flux_out_soft"], "%.3e"), f(r["E_flux_out_hard"], "%.3e"), f(r["E_flux_in_soft"], "%.3e"), f(r["E_flux_in_hard"], "%.3e"), f(r["out"], "%.3e"), f(r["in"], "%.3e"), f(r["in_over_out_signed"], "%.3f"), f(r["abs_in_over_abs_out"], "%.3f"), r["v_c"], r["om1_k1"], r["om1_margin_k1"], r["om1_margin_k2"], f(r["join_max_rel_dv"], "%.2e")))
L.append("")
L.append("units check (collar flow change over L_interval dt): n %d, range %s to %s" % (len(units), f(rng([u[2] for u in units])[0], "%.6f"), f(rng([u[2] for u in units])[1], "%.6f")))
L.append("seconds %.1f" % (time.time() - t00))
open(os.path.join(a.out, "d13_v1_1_recompute.md"), "w").write("\n".join(L) + "\n")
print("wrote", a.out, "%.1f s" % (time.time() - t00))
