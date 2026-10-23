#!/usr/bin/env python
"""ISO64_bridge_checks_v1_1.py (COMP4 v1.1, M9 and M10, R5's read): (M9) the 8 Newton records of collapse_isoB.json (13.625 to 14.5) against the
two card logs' printed E/E0 (7 digits) and their printed kept/confirming residuals; (M10) the 18 B2 channel files' md5s (summary and d13) against
PAGE_solved_history_v4.1's SOURCES lines; (R5) the isotropic catalog block of the isoB head. Read-only. Usage: python ... OUT_LOG"""
import hashlib, json, re, sys
OUT = sys.argv[1]
ISO = "<internal>/collapse_isoB.json"
LOGS = {"p8": "<internal>/lad_ISO_64_s4_nk_ep_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36n.log",
        "r3": "<internal>/lad_ISO_64_s4_nk_ep_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36n.log"}
V41 = sys.argv[2]
L = []
def md5(p):
    h = hashlib.md5(); f = open(p, "rb")
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
J = json.load(open(ISO))
recs = {round(r["T_over_rs"], 4): r for r in J["records"] if "T_over_rs" in r}
newton = [t for t, r in sorted(recs.items()) if isinstance(r.get("solve_v32"), dict) or (r.get("solve_v32") is not None)]
import engine_ladder_compare_v5 as LC
newton = [t for t, r in sorted(recs.items()) if LC.v32_solve(r) is not None]
for name, p in LOGS.items():
    txt = open(p).read()
    ee = {round(float(a), 4): b for a, b in re.findall(r"^T/rs ([0-9.]+) E/E0 ([0-9.]+)", txt, re.M)}
    kc = {round(float(a), 4): (float(b), float(c)) for a, b, c in re.findall(r"^T/rs ([0-9.]+) newton .*?kept residual ([0-9.e+-]+) \(evaluation \d+\), the confirming plain step's residual ([0-9.e+-]+)", txt, re.M)}
    ok_e, ok_r, nres = 0, 0, 0
    bad = []
    for t in newton:
        r = recs[t]; sv = LC.v32_solve(r)
        if t in ee and ee[t] == "%.7f" % r["E_over_E0"]: ok_e += 1
        else: bad.append(("E", t, ee.get(t), "%.7f" % r["E_over_E0"]))
        if t in kc:
            nres += 1
            k, c = kc[t]
            if ("%.2e" % k) == ("%.2e" % sv["kept_residual"]) and ("%.2e" % c) == ("%.2e" % sv["confirming_residual"]): ok_r += 1
            else: bad.append(("res", t, kc[t], (sv["kept_residual"], sv["confirming_residual"])))
    L.append("M9 %s log (md5 %s): the %d Newton records %s; E/E0 equal to 7 digits at %d of %d; kept and confirming residuals printed at %d records, equal to 3 digits at %d; mismatches %s"
             % (name, md5(p), len(newton), ",".join("%g" % t for t in newton), ok_e, len(newton), nres, ok_r, bad))
# M10
src = open(V41).read()
want = {}
for m in re.finditer(r"- ([0-9a-f]{32})  (\S+v6o_B2\S*_k(\d+)_(summary|d13)\.json)", src):
    want[m.group(2)] = m.group(1)
dirs = {k: ("<internal>/v6o_B2", "v6o_B2") for k in range(1, 7)}
for a, b, tag in ((7, 9, "v6o_B2_k7to9"), (10, 12, "v6o_B2_k10to12"), (13, 16, "v6o_B2_k13to16"), (17, 18, "v6o_B2_k17to18")):
    for k in range(a, b + 1): dirs[k] = ("<internal>/runs" + tag, tag)
eq, ne, missing = 0, [], []
for k in range(1, 19):
    for suf in ("summary", "d13"):
        p = "%s/%s_k%d_%s.json" % (dirs[k][0], dirs[k][1], k, suf)
        w = want.get(p)
        if w is None: missing.append(p); continue
        if md5(p) == w: eq += 1
        else: ne.append(p)
L.append("M10 B2 channel files (kappa 1 to 18, summary and d13, 36 files) against PAGE_solved_history_v4.1 SOURCES: %d equal, %d differ %s, %d not listed in SOURCES %s" % (eq, len(ne), ne, len(missing), missing[:6]))
# R5
cat = J["head"]["catalog"]
ch = cat["channels"]
L.append("R5 isoB head (md5 %s): args dust_set %s, dust_kappa_max %s, dust_multiplicity %s, kappa_sea %s; catalog label: %s; top_channel %s; columns per kappa %s; particles per kappa %s (sum %s); j per kappa %s; f all 1: %s; kappa_max_g16_rule %s; kappa_max_rule %s; k_F_R0_catalog %s; E_F %s; lowest level of kappa 7, 8: %s, %s"
         % (md5(ISO), J["args"].get("dust_set"), J["args"].get("dust_kappa_max"), J["args"].get("dust_multiplicity"), J["args"].get("kappa_sea"), cat["label"], cat["top_channel"],
            [ch[k]["columns"] for k in sorted(ch, key=int)], [ch[k]["particles"] for k in sorted(ch, key=int)], sum(ch[k]["particles"] for k in ch),
            [ch[k]["j"] for k in sorted(ch, key=int)], all(x == 1.0 for k in ch for x in ch[k]["f"]), cat["kappa_max_g16_rule"], cat["kappa_max_rule"],
            cat["k_F_R0_catalog"], cat["E_F"], cat["key_info"]["7"]["lowest"], cat["key_info"]["8"]["lowest"]))
import math
f = lambda x: 1 - (1 - x * x) ** 0.5 * (1 - 4 * x * x) - 3 * x ** 3 * math.acos(x)
K = int(cat["top_channel"]); NB = J["head"]["N_B"]
for lab, c in (("0.96 N_B^(1/3)", 0.96 * NB ** (1 / 3.)), ("engine g16 rule", cat["kappa_max_g16_rule"])):
    L.append("R5 f_cap (P14 hard-cutoff capacity f(x) = 1 - (1 - x^2)^(1/2)(1 - 4x^2) - 3x^3 arccos x) at K %d, x = K/%s = %.4f: %.4f; production check K 5: %.4f; K 8: %.4f" % (K, lab, K / c, f(K / c), f(5 / c), f(min(8 / c, 1))))
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
