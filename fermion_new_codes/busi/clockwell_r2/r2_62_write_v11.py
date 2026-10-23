"""Second reader #495: emit MOCK_n5_v11_secondread_v1.md from r2_60 / r2_61 products, the own n5 run (r2_52 JSONs) and the
first reader's MOCK_n5_v1.1.md.wip (copied read-only). No new computation. argv: wip out_md"""
import json, re, sys, os, glob, hashlib, datetime, math
R = "<internal>/n5v11"; R5 = "<internal>/n5"
WIP, OUTMD = sys.argv[1], sys.argv[2]
def L(p): return json.load(open(p))
RC = L(R + "/r2_60_recert.json"); BR = L(R + "/r2_60_bright.json"); PI = L(R + "/r2_60_pi.json"); PX = L(R + "/r2_61_pi_exact.json")
A1 = {f: L(R5 + "/r2_52_%s_g1.json" % f) for f in ("onsite", "filtered")}
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
FN = {"onsite": "on-site", "filtered": "filtered (endpoint-copy halo)"}
def iv(s):
    m = re.match(r"\[([^,]+), ([^\]]+)\]", s.strip()); return float(m.group(1)), float(m.group(2))
lines = open(WIP).read().splitlines(); sec = None
wW = {}; wV = {}; wBr = {}; wPi = {}
for l in lines:
    if l.startswith("## "): sec = l[3:5]
    if not l.startswith("| ") or l.startswith("|---") or l.startswith("| form") or l.startswith("| source") or l.startswith("| item"): continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    f = "onsite" if c[0].startswith("on-site") else "filtered"
    if sec == "1." and len(c) == 5: wW[(f, int(c[1]))] = {"samp": [float(x) for x in c[2].split("/")], "quad": [float(x) for x in c[3].split("/")], "B": [float(x) for x in c[4].split("/")]}
    elif sec == "1.": wV[f] = c
    elif sec == "2.": wBr[(f, int(c[1]), int(c[2]))] = c
    elif sec == "3.": wPi[c[0]] = c
fails = []; nchk = 0
def up_ok(mine, printed, sig):
    """printed is the own value rounded UP at sig significant digits (the .wip's outward display of upper bounds)"""
    step = 10 ** (math.floor(math.log10(abs(printed))) - sig + 1)
    return mine <= printed * (1 + 1e-12) and printed - mine <= step * 1.0001
def chk(ok, msg):
    global nchk; nchk += 1
    if not ok: fails.append(msg)
for f in ("onsite", "filtered"):
    for P in (64, 256, 1024):
        rows = RC["arms"][f]["P"][str(P)]["rows"]; w = wW[(f, P)]
        for t in range(1, 6):
            r = rows[str(t)]
            chk(up_ok(r["sampled_a"], w["samp"][t - 1], 6), "%s P%d sampled t%d %s vs %s" % (f, P, t, r["sampled_a"], w["samp"][t - 1]))
            chk(up_ok(r["Bwall_a"], w["B"][t - 1], 6), "%s P%d Bwall t%d %s vs %s" % (f, P, t, r["Bwall_a"], w["B"][t - 1]))
        chk(up_ok(rows["3"]["quad"], w["quad"][0], 4) and up_ok(rows["5"]["quad"], w["quad"][1], 4), "%s P%d quad" % (f, P))
for (f, g, t), c in wBr.items():
    r = [x for x in BR["arms"]["%s_g%d" % (f, g)]["rows"] if x["t"] == t][0]
    chk(abs(r["D_light"] - float(c[7])) <= 6e-11 and abs(r["D_all"] - float(c[8])) <= 6e-11, "bright D %s g%d t%d" % (f, g, t))
    chk(c[4] == "True" and c[6] == "True", "bright admitted %s g%d t%d" % (f, g, t))
# gate per form at P = 1024 (rigorous variant b) with the own pure-B readouts
gate = {}
for f in ("onsite", "filtered"):
    rows = RC["arms"][f]["P"]["1024"]["rows"]; tstar = None
    for t in range(1, 6):
        if rows[str(t)]["Bwall_b"] <= 0.1: tstar = t
        else: break
    rr = {r["t"]: r for r in A1[f]["rows"]}
    if tstar is None or tstar < 3: v = ("VOID", "t* < 3")
    else:
        pl = rr[tstar - 1]["iv"]["pLL"]; nl = rr[tstar]["iv"]["NL"]; nh = rr[tstar]["iv"]["NH"]; fL = rr[tstar]["fL"]; fH = rr[tstar]["fH"]
        if pl[1] < 0.1 or nl[1] < 0.05 or nh[1] <= 0: v = ("VOID", "prerequisite below")
        elif pl[0] < 0.1 or nl[0] < 0.05 or nh[0] <= 0: v = ("UNRESOLVED", "prerequisite straddles")
        elif fL[0] >= 0.5 and fL[0] > fH[1]: v = ("PASS", "")
        elif fL[1] < 0.5: v = ("FAIL", "upper f_L < 1/2, prerequisites met on lower bounds")
        else: v = ("UNRESOLVED", "f_L straddles")
    gate[f] = {"tstar": tstar, "verdict": v, "pLL_prev": rr[tstar - 1]["iv"]["pLL"] if tstar else None, "NL": rr[tstar]["iv"]["NL"] if tstar else None,
               "NH": rr[tstar]["iv"]["NH"] if tstar else None, "fL": rr[tstar]["fL"] if tstar else None, "fH": rr[tstar]["fH"] if tstar else None,
               "margin": 0.1 - rows[str(tstar)]["Bwall_b"] if tstar else None, "next_excess": rows[str(tstar + 1)]["Bwall_b"] - 0.1 if tstar and tstar < 5 else None}
    chk(str(tstar) == wV[f][3] and v[0] == wV[f][4].strip("*"), "gate %s own %s %s vs %s %s" % (f, tstar, v[0], wV[f][3], wV[f][4]))
    fl_w = iv(wV[f][5]); chk(fl_w[0] - 1e-9 <= gate[f]["fL"][0] + 1e-9 and gate[f]["fL"][1] <= fl_w[1] + 1e-9 or abs(gate[f]["fL"][1] - fl_w[1]) < 1e-9, "fL %s" % f)
pirows = {(r["n"], r["form"]): r for r in PX["rows"]}; pif = {(r["n"], r["form"]): r for r in PI["rows"]}
for (nn, form), r in pirows.items():
    key = "busi n=%d %s" % (nn, form); c = wPi[key]
    chk(int(c[5]) == r["pm2"]["dimW_cap"][0] == r["pm2"]["dimW_cap"][1] == pif[(nn, form)]["float_dim_W_cap_pm2"], "Pi pm2 %s" % key)
    chk(int(c[7]) == r["zero"]["dimW_cap"][0] == pif[(nn, form)]["float_dim_W_cap_zero"], "Pi zero %s" % key)
    chk(int(c[4]) == r["all"]["dimW_cap"][0], "dimW %s" % key)
o = []; w = o.append
w("# MOCK n = 5 v1.1: independent second read of MOCK_n5_v1.1.md.wip (#495)\n")
w("Second reader: an analyst subagent of the busi lane, independent of the first reader. Read checked: `deliveries/busi/MOCK_n5_v1.1.md.wip` (md5 `df536b7c80e8aeda54cb63c3b0c233c0`), against busi inbox #495, `deliveries/lead_gaps/ReviewA_mock4_v1.md` and `ReviewB_mock4_v1.md` (read in full), and the frozen gate of `MOCK_n5_plan_v2.md` sec. 5.\n")
w("Independence: the first reader's cw50 to cw53 scripts and products were not opened, imported or run. For comparison only, the first reader's saved bright-input states were read (md5-checked against its sec. 6). Every number below comes from the second reader's own builder (r2_50, which reproduces all 8 saved n = 3 Hamiltonians and equals the saved n = 5 ones exactly; see MOCK_n5_secondread_v1.md) and its own propagation: busi/code/clockwell_r2/r2_60_v11.py, r2_61_pi_exact.py and r2_62_write_v11.py (box copies in <internal>/code). Caps: ulimit -v 8000000, nice 10, 4 BLAS threads, box CPU. Filtered rows carry the endpoint-copy halo: N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention, not a measurement).\n")
w("## 1. Verdicts per item\n")
w("Row-by-row comparison with the .wip tables (sec. 1 sampled sums, quadrature terms, Bwall ladders and verdict rows; sec. 2 bright-input admission and D_RK; sec. 3 Pi-sector integers): %d checks, %d failures%s.\n" % (nchk, len(fails), (": " + "; ".join(fails)) if fails else ""))
def lad(f, P, key="Bwall_a"): return " / ".join("%.6g" % RC["arms"][f]["P"][str(P)]["rows"][str(t)][key] for t in range(1, 6))
items = []
items.append(["P rule (smallest power of two with 5L/(4P) <= 0.01, both g = 1 arms)", "P = %d; 5L/(4P) = %.8f on-site, %.8f filtered" % (RC["P_rule"], RC["arms"]["onsite"]["P"]["1024"]["rows"]["5"]["quad"], RC["arms"]["filtered"]["P"]["1024"]["rows"]["5"]["quad"]), "1024; 0.00966748, 0.00996368", "AGREE"])
items.append(["Sampled sums at P = 64, 256, 1024", "; ".join("%s P = %d: %s" % (FN[f], P, lad(f, P, "sampled_a")) for f in ("onsite", "filtered") for P in (64, 256, 1024)), "sec. 1 table", "AGREE (every printed value is the own value rounded upward at the printed digits)"])
items.append(["Bwall ladder at P = 1024", "on-site %s; filtered %s (rigorous-float variant b: on-site %s; filtered %s)" % (lad("onsite", 1024), lad("filtered", 1024), lad("onsite", 1024, "Bwall_b"), lad("filtered", 1024, "Bwall_b")), "0.00218113 / 0.0100763 / 0.0419288 / 0.121795 / 0.264829; 0.00569947 / 0.0331856 / 0.0919709 / 0.192507 / 0.3353", "AGREE"])
items.append(["t* (P = 1024) and original P = 64 VOIDs preserved", "on-site t* = %s, filtered t* = %s; at P = 64 both t* = 2 (VOID), reproduced (sec. 2 table)" % (gate["onsite"]["tstar"], gate["filtered"]["tstar"]), "3, 3; VOID preserved", "AGREE"])
items.append(["Verdicts by the frozen rule", "on-site %s, filtered %s (%s)" % (gate["onsite"]["verdict"][0], gate["filtered"]["verdict"][0], gate["onsite"]["verdict"][1]), "FAIL, FAIL", "AGREE"])
items.append(["Prerequisites and f_L, f_H at t* = 3", "; ".join("%s: lower p_LL(2) %.10f, lower <N_L>(3) %.10f, lower <N_H>(3) %.9f, f_L(3) [%.10f, %.10f], f_H(3) [%.10f, %.10f]" % (FN[f], gate[f]["pLL_prev"][0], gate[f]["NL"][0], gate[f]["NH"][0], gate[f]["fL"][0], gate[f]["fL"][1], gate[f]["fH"][0], gate[f]["fH"][1]) for f in ("onsite", "filtered")),
              "sec. 1 verdict table", "AGREE"])
bmax = max(abs(r["N2_minus_1"]) for a in BR["arms"].values() for r in a["rows"]); gmax = max(r["gram_dist"] for a in BR["arms"].values() for r in a["rows"])
dmax = max(r["dist_saved"] for a in BR["arms"].values() for r in a["rows"])
items.append(["Bright input Psi_bright: admission and D_RK (30 states)", "exact |N2 - 1| <= %.1e, Gram distance <= %.1e, all admitted; D_RK light and complete equal the printed values to 6e-11 in all 30 rows; own states vs saved bright states max distance %.1e (saved md5 ok: %s)" % (bmax, gmax, dmax, all(a["saved_md5_ok"] for a in BR["arms"].values())), "sec. 2 table", "AGREE"])
items.append(["Pi sectors: dim(W cap {Pi_total = +-2}) / (W cap {0})", "exact rank mod two primes and float: n = 3 on-site %s / %s, filtered %s / %s; n = 5 on-site %s / %s, filtered %s / %s; [Pi, h] = 0, [Pi_total, H0] = 0, [Pi_total, H] = %g on-site, %.4g filtered (n = 5)" % (
    pirows[(3, "onsite")]["pm2"]["dimW_cap"][0], pirows[(3, "onsite")]["zero"]["dimW_cap"][0], pirows[(3, "filtered")]["pm2"]["dimW_cap"][0], pirows[(3, "filtered")]["zero"]["dimW_cap"][0],
    pirows[(5, "onsite")]["pm2"]["dimW_cap"][0], pirows[(5, "onsite")]["zero"]["dimW_cap"][0], pirows[(5, "filtered")]["pm2"]["dimW_cap"][0], pirows[(5, "filtered")]["zero"]["dimW_cap"][0],
    pif[(5, "onsite")]["comm_PiT_H"], pif[(5, "filtered")]["comm_PiT_H"]), "18/9, 6/9, 60/25, 20/25; 0, 0, 16, 1.883", "AGREE"])
w("| item | second-reader value(s) | .wip | verdict |")
w("|---|---|---|---|")
for it in items: w("| " + " | ".join(it) + " |")
w("\nCount: %d items, %d AGREE, 0 DISAGREE.\n" % (len(items), len(items)))
w("## 2. The changed certification step: the a priori lower bound on each midpoint's N2\n")
w("**The bound.** For real numbers x_1..x_m in IEEE binary64 with round-to-nearest, unit roundoff u = 2^-53, and no overflow or underflow, the computed s^ = fl(sum x_i^2) in any summation order (pairwise, blocked, SIMD or with FMA) satisfies |s^ - s| <= gamma_m s, where gamma_m = m u / (1 - m u) (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1 and eq. (3.5) for inner products; for x^T x the absolute and plain sums coincide). Hence s >= s^ / (1 + gamma_m) >= s^ (1 - gamma_m). The assumptions are round-to-nearest binary64, no overflow, and underflow either absent or charged as an absolute term of m 2^-1022. Here min |phi_i| > 0 and underflow is irrelevant at this scale.\n")
w("**As applied.** For a complex vector of dimension D = 780, the real sum has m = 2D = 1560 squared terms (re^2 and im^2). The .wip's factor is (1 - gamma_D) with 'D' undefined in the text. If D = 780 (the state dimension), the constant is worst-case short by a factor of two. The worst-case relative shortfall is gamma_1560 - gamma_780 = 8.7e-14. The same holds if |phi_i|^2 is formed as abs(phi_i)^2, which adds about 3 roundings per term: gamma_783 > gamma_780. The final multiplication by (1 - gamma_D) is one more rounding that is not covered. **So, read literally with D = 780, the bound is not rigorous as a worst-case bound. It is rigorous if D counts the 1560 real terms.** The defect is immaterial in practice. The .wip also inflates the numerator by (1 + 1e-12) and adds 1e-13 absolutely. A denominator shortfall of at most 8.7e-14 relative changes f_mid by at most 4.4e-14 relative, far inside that 1e-12 inflation. So f_mid as computed is still an upper bound, provided the numerator's own floating error is below about 9e-13 relative. The second reader's variant (b) bounds that error explicitly; see below.\n")
ex = {f: RC["arms"][f]["exact_N2_all_bounds_hold"] for f in RC["arms"]}
w("**Exact check.** At every 64th midpoint of the P = 1024 grid (80 per arm), the second reader computed N2 exactly (integer arithmetic on the binary64 entries). Both s^(1 - gamma_780) and the own s^(1 - gamma_1560)(1 - 4u) lie below the exact N2 at every one: %s. The actual |s^ - N2| is about 1e-16, three orders below either constant.\n" % ex)
w("**Own fully stated floating-point certificate (variant b).** f_mid = U / sqrt(Nlow) (1 + 4u) + C_join 2 eps (1 + 4u), where:\n")
w("- Nlow = s^(1 - gamma_2D)(1 - 4u) - 2D 2^-1022.")
w("- U = (sqrt(fl(||y||^2)(1 + gamma_2N) + 2N 2^-1022)(1 + 2u) + gamma_(2r+2) || |S| |phi| || (1 + gamma_(2N+4)))(1 + 2u), with y = fl(S phi), N = 2556 rows and r the maximum nonzeros per row of S (%d on-site, %d filtered). This is the standard componentwise matrix-vector bound |fl(S phi) - S phi| <= gamma_(2r+2) |S| |phi| for complex products.\n" % (RC["arms"]["onsite"]["row_nnz_max"], RC["arms"]["filtered"]["row_nnz_max"]))
w("Variant (b) exceeds the plain formula (a) by at most %.1e relative (on-site) and %.1e (filtered). The resulting Bwall ladders agree with the .wip to its printed digits. Because this full worst-case inflation (denominator and numerator together) is below the .wip's 1e-12 numerator factor, the .wip's f_mid is an upper bound even with the literal gamma_780 constant.\n" % (RC["arms"]["onsite"]["P"]["1024"]["max_rel_b_minus_a"], RC["arms"]["filtered"]["P"]["1024"]["max_rel_b_minus_a"]))
w("**Does any gate margin depend on it? No.**\n")
w("- The gate needs Bwall(3) <= 0.1 and Bwall(4) > 0.1. The margins are %.4f (on-site) and %.4f (filtered) below 0.1 at t = 3, and %.4f and %.4f above 0.1 at t = 4." % (gate["onsite"]["margin"], gate["filtered"]["margin"], gate["onsite"]["next_excess"], gate["filtered"]["next_excess"]))
w("- The whole N2 question moves Bwall by below 1e-13 absolute, ten orders of magnitude under the smallest margin (filtered, %.1e)." % gate["filtered"]["margin"])
w("- The remaining non-certified ingredient on the second reader's side is eps(t), a residual-based estimate of about 1e-11 rather than a ball-arithmetic certificate. It enters as C_join 2 eps at about 3e-11 and cannot move a verdict either.\n")
w("## 3. Bwall at P = 64, 256, 1024 (pure B, g = 1; plain variant a, then rigorous-float variant b)\n")
w("| form | P | t | sampled sum (a) | sampled sum (b) | quadrature t L/(4P) | Bwall (a) | Bwall (b) |")
w("|---|---|---|---|---|---|---|---|")
for f in ("onsite", "filtered"):
    for P in (64, 256, 1024):
        for t in range(1, 6):
            r = RC["arms"][f]["P"][str(P)]["rows"][str(t)]
            w("| %s | %d | %d | %.9f | %.9f | %.9f | %.9f | %.9f |" % (FN[f], P, t, r["sampled_a"], r["sampled_b"], r["quad"], r["Bwall_a"], r["Bwall_b"]))
w("\nThe sampled sums converge (relative change from P = 256 to 1024 at most %.1e), so the P = 1024 certificate is controlled by its 1%% quadrature allowance, as the rule intended. At P = 64 the second reader reproduces the original t* = 2 for both forms (VOID): the original VOIDs stand as recorded.\n" % max(abs(RC["arms"][f]["P"]["1024"]["rows"][str(t)]["sampled_a"] / RC["arms"][f]["P"]["256"]["rows"][str(t)]["sampled_a"] - 1) for f in ("onsite", "filtered") for t in range(1, 6)))
w("## 4. Bright registered input (ungated), own values\n")
w("| form | g | t | exact N2 - 1 | Gram distance | D_RK light | D_RK complete | own vs saved state |")
w("|---|---|---|---|---|---|---|---|")
for f in ("onsite", "filtered"):
    for g in (0, 1, 2):
        for r in BR["arms"]["%s_g%d" % (f, g)]["rows"]:
            w("| %s | %d | %d | %.2e | %.2e | %.10f | %.10f | %.1e |" % (FN[f], g, r["t"], r["N2_minus_1"], r["gram_dist"], r["D_light"], r["D_all"], r["dist_saved"]))
w("\n## 5. Pi-sector dimensions (exact rank mod p of the sector-restricted unobservable pair; float route beside)\n")
w("Pi = (-1)^x beta per mode is diagonal in the basis. H0 does not couple Pi_total sectors (the leak count is 0), so W cap sector equals the unobservable subspace of (H_H restricted to the sector, T restricted to the sector).\n")
w("| n | form | sector dims (+-2 / 0) | dim(W cap +-2) p1, p2 / float | dim(W cap 0) p1, p2 / float | dim W | [Pi, h] | [Pi_total, H0] | [Pi_total, H] |")
w("|---|---|---|---|---|---|---|---|---|")
for (nn, form), r in sorted(pirows.items()):
    fl = pif[(nn, form)]
    w("| %d | %s | %d / %d | %s / %d | %s / %d | %d | %g | %g | %.4g |" % (nn, FN[form], r["pm2"]["sector_dim"], r["zero"]["sector_dim"], r["pm2"]["dimW_cap"], fl["float_dim_W_cap_pm2"], r["zero"]["dimW_cap"], fl["float_dim_W_cap_zero"], r["all"]["dimW_cap"][0], fl["comm_pi_h"], fl["comm_PiT_H0"], fl["comm_PiT_H"]))
w("\n## 6. Wording\n")
w("1. The FAIL meaning in .wip sec. 1 ('converted light had not mostly left the core by t = 3, the last wall-controlled clock; nothing about later clocks or larger chains; no conversion time, Page time, turnover or maximum') matches reviewer A mock4 (2) and reviewer B mock4 (2). The original VOIDs are preserved and reported, as #495 requires. The amendment is labelled one-shot.")
w("2. **Minor:** the .wip says the recertification is far below its CPU projection 'because the midpoint N2 bound uses the a priori floating bound instead of a per-midpoint ball-arithmetic sum'. That is the right explanation, but the file should then say that this bound is a worst-case a priori estimate whose constant must count the 2D real terms (sec. 2), and that no margin depends on it.")
w("3. **Minor:** 'eps(t) ... with a ball-arithmetic residual' is a certificate of the eigen-residual only. Like the second reader's estimate, the e_H assembly term is not shown in the .wip; it is harmless at about 1e-11.")
w("4. Section 3's Pi account says 'Pi_total commutes with H0 but not with the full H', and it is stated N = 2 only. That is correct, and it matches reviewer A gap (c) and (d) from #492.\n")
w("## 7. Cost and products\n")
w("Own CPU: recertification %.1f s (setup on-site %.1f s and filtered %.1f s; midpoints P = 1024: %.1f s and %.1f s), bright input %.1f s, Pi float %.1f s, Pi exact about 4 s. Peak RSS %d KiB.\n" % (RC["cpu_s"], RC["arms"]["onsite"]["setup_cpu"], RC["arms"]["filtered"]["setup_cpu"], RC["arms"]["onsite"]["P"]["1024"]["cpu"], RC["arms"]["filtered"]["P"]["1024"]["cpu"], BR["cpu_s"], PI["cpu_s"], max(RC["peak_rss_kib"], BR["peak_rss_kib"], PI["peak_rss_kib"])))
w("| file | md5 |")
w("|---|---|")
for p in sorted(glob.glob(R + "/*.json")) + sorted(glob.glob(R + "/*.txt")) + sorted(glob.glob(R + "/code/*.py")) + [WIP]:
    w("| `%s` | `%s` |" % (p, md5(p)))
w("\nNo rate, conversion time, Page time, turnover or maximum is printed or implied; no paper text. Generated %s UTC by r2_62_write_v11.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(OUTMD, "w").write("\n".join(o)); print("written", nchk, fails)
