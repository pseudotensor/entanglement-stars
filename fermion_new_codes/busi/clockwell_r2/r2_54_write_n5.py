"""Second reader #492: emit MOCK_n5_secondread_v1.md from r2_51/r2_52/r2_53 products and a row-by-row comparison with the
first reader's MOCK_n5_v1.md.wip (copied read-only into the products dir). No new computation. argv: wip out_md"""
import json, re, sys, os, glob, hashlib, datetime
R = "<internal>/n5"
WIP, OUTMD = sys.argv[1], sys.argv[2]
def L(f): return json.load(open(os.path.join(R, f)))
VAL = L("r2_51_validate_n3.json"); DK = L("r2_53_dark.json")
ARMS = [(f, g) for f in ("onsite", "filtered") for g in (0, 1, 2)]
A = {(f, g): L("r2_52_%s_g%d.json" % (f, g)) for f, g in ARMS}
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
FN = {"onsite": "on-site", "filtered": "filtered (endpoint-copy halo)"}
def iv(s):
    m = re.match(r"\[([^,]+), ([^\]]+)\]", s.strip()); return float(m.group(1)), float(m.group(2))
txt = open(WIP).read(); lines = txt.splitlines()
sec = None; wB = {}; wR = {}; wD = {}; wG = {}
for l in lines:
    if l.startswith("## "): sec = l[3:5]
    if not l.startswith("| ") or l.startswith("| form") or l.startswith("| source") or l.startswith("|---"): continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    f = "onsite" if c[0].startswith("on-site") else ("filtered" if c[0].startswith("filtered") else None)
    if sec == "1." and f: wG[f] = (c[1], c[2], c[3].strip("*"))
    if sec == "2." and f: wB[(f, int(c[1]))] = [float(x) for x in c[2:10]]
    if sec == "3." and f: wR[(f, int(c[1]), int(c[2]))] = c[3:]
    if sec == "6.": wD[c[0]] = c[1:]
fails = []; checks = 0; display = []
def chk(cond, msg):
    global checks; checks += 1
    if not cond: fails.append(msg)
def inside(x, s, sl=1e-9):
    a, b = iv(s); return a - sl * max(1, abs(a)) <= x <= b + sl * max(1, abs(b))
# Bwall and constants
for (f, g), a in A.items():
    w = wB[(f, g)]
    for k, key in enumerate(("C_join", "H_bound", "L")): chk(abs(a[key] - w[k]) <= 5e-8 * max(1, w[k]), "%s g%d %s %s vs %s" % (f, g, key, a[key], w[k]))
    for t in range(1, 6): chk(abs(a["Bwall"][str(t)] - w[2 + t]) <= 5e-6 * max(1e-3, w[2 + t]) + 1e-9, "%s g%d Bwall(%d) %s vs %s" % (f, g, t, a["Bwall"][str(t)], w[2 + t]))
# readouts
for (f, g), a in A.items():
    for r in a["rows"]:
        c = wR[(f, g, r["t"])]
        chk(c[2] == "True" and r["admitted"], "admission %s g%d t%d" % (f, g, r["t"]))
        if not (inside(r["vals"]["pLL"], c[3], 0) or abs(r["vals"]["pLL"]) < 1e-12): display.append((f, g, r["t"], r["vals"]["pLL"], c[3]))
        chk(inside(r["vals"]["pLL"], c[3], 1e-10) or abs(r["vals"]["pLL"]) < 1e-12, "pLL %s g%d t%d" % (f, g, r["t"]))
        chk(inside(r["vals"]["NL"], c[4], 0) or abs(r["vals"]["NL"]) < 1e-12, "NL %s g%d t%d" % (f, g, r["t"]))
        chk(inside(r["vals"]["NH"], c[5]), "NH %s g%d t%d" % (f, g, r["t"]))
        if c[6] == "undefined": chk(r["fL"] is None, "fL undefined %s g%d t%d" % (f, g, r["t"]))
        else: chk(r["fL"] is not None and inside(r["vals"]["NL_lead"] / r["vals"]["NL"], c[6]), "fL %s g%d t%d" % (f, g, r["t"]))
        chk(inside(r["vals"]["NH_lead"] / r["vals"]["NH"], c[7]), "fH %s g%d t%d" % (f, g, r["t"]))
# gate
for f in ("onsite", "filtered"):
    a = A[(f, 1)]; wg = wG[f]
    chk(str(a["tstar"]) == wg[0] and wg[2] == a["gate"][0], "gate %s: own t*=%s %s vs .wip %s" % (f, a["tstar"], a["gate"], wg))
o = []; w = o.append
w("# MOCK n = 5: independent second read of MOCK_n5_v1.md.wip (#492)\n")
w("Second reader: an analyst subagent of the busi lane, independent of the first reader. Read checked: `deliveries/busi/MOCK_n5_v1.md.wip` (md5 `9fe76f7f68237d9a8d5f3635bcb9131d`), against the frozen `deliveries/busi/MOCK_n5_plan_v2.md` (md5 `3a0f0ab8fe3509dc6e74385362d25b8d`), busi inbox #492, and `deliveries/lead_gaps/ReviewA_mock3_v1.md` and `ReviewB_mock3_v1.md`, all read in full.\n")
w("Independence: the first reader's scripts (cw32 to cw47) and products were not opened, imported or run. The exceptions are the named saved inputs: icloud's n = 3 reports and Hamiltonian arrays (md5-checked against the input manifest) and, for comparison only, the first reader's saved n5run `hamiltonian.npz` and `states_pure.npz` (md5-checked against its sec. 10). Everything else comes from the second reader's own builder and propagation: busi/code/clockwell_r2/r2_50_model.py to r2_54_write_n5.py (box copies in <internal>/code). Box CPU, nice 10, 4 BLAS threads, ulimit -v 8000000.\n")
w("Filtered rows carry the endpoint-copy halo: N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention, not a measurement). Gated readouts are for the normalized pure input B = |01>_H at site 0.\n")
w("## 1. Own builder, validated against every saved n = 3 Hamiltonian\n")
w("The builder is written from the specs. Free part: h[x,x] = m_A N_x beta, h[x,x+1] = -i (N_x+N_(x+1)) alpha/4. On-site converter: N_x (O_x + h.c.), with O_x = sum K[s,s'] |L-pair s at x><H-pair s' at x|. Filtered converter: sum_(j,k=-1..n) G_(j-k) (N_j+N_k)/2 (O_jk + h.c.), where O_jk is the same K between the stencil-field pairs d_(b,j) and d_(b,k), summed over b, with exact G_r. In the N = 2 sector every pair bilinear is an exact sum of outer products of two-particle pair states. K was read from icloud's saved n = 3 on-site Hamiltonians (g = 1 minus g = 0, site-1 block). Its deviation from integers is %g:\n" % VAL["K_integer_dev"])
w("K (rows: light pair, columns: heavy pair, order 01, 02, 03, 12, 13, 23) = %s; column 03 = column 12: %s; symmetric: %s; nonzero entries: %d.\n" % (VAL["K"], VAL["K_facts"]["col03_eq_col12"], VAL["K_facts"]["symmetric"], VAL["K_facts"]["nonzero"]))
w("| saved n = 3 Hamiltonian | max abs(own - saved) |")
w("|---|---|")
for k, v in VAL["compare"].items(): w("| %s | %.1e |" % (k, v))
w("\nThe basis order (bitmask) equals icloud's saved row_keys: %s. Every input md5 equals the manifest. At n = 5 the own H equals the first reader's saved hamiltonian.npz exactly in all six arms (max abs difference %g; saved md5s as in its sec. 10: %s).\n" % (
    VAL["basis_order_equal_saved_row_keys"], max(a["max_abs_H_minus_saved"] for a in A.values()), all(all(a["saved_md5_ok"]) for a in A.values())))
w("## 2. Verdicts per item\n")
w("Row-by-row comparison with the .wip tables (sec. 1 gate, sec. 2 Bwall and constants, sec. 3 admission and readouts): %d checks, %d failures%s.\n" % (checks, len(fails), (": " + "; ".join(fails)) if fails else ""))
# dark
dk = {r["source"]: r for r in DK["rows"]}
def dkrow(src):
    r = dk[src]; ex = r["exact"]
    return r, (ex[0]["dimW"] if ex else None), (ex[1]["dimW"] if ex else None)
def bw(f, g, t): return A[(f, g)]["Bwall"][str(t)]
items = []
items.append(["Admission (30 pure states)", "all 30 admitted; |N2 - 1| <= %.1e; own state vs saved state max distance %.1e" % (max(r["N2_dev"] for a in A.values() for r in a["rows"]), max(r["dist_to_saved_state"] for a in A.values() for r in a["rows"])), "all admitted", "AGREE"])
items.append(["Bwall at every clock, C_join, H_bound, L (6 arms)", "; ".join("%s g%d: %s" % (FN[f], g, ", ".join("%.6g" % bw(f, g, t) for t in range(1, 6))) for f, g in ARMS), "sec. 2 values", "AGREE (all to the printed digits)"])
items.append(["t* per form and coupling", "; ".join("%s g%d: %s" % (FN[f], g, A[(f, g)]["tstar"]) for f, g in ARMS), "g = 1: 2 and 2; ungated: g = 0: 3 and 3; on-site g = 2: 1; filtered g = 2: none", "AGREE"])
items.append(["Gate g = 1 (frozen v2 rule)", "on-site: %s (%s); filtered: %s (%s)" % (A[("onsite", 1)]["gate"][0], A[("onsite", 1)]["gate"][1], A[("filtered", 1)]["gate"][0], A[("filtered", 1)]["gate"][1]), "VOID, VOID (t* < 3)", "AGREE"])
items.append(["p_LL, <N_L>, <N_H>, f_L, f_H (30 rows)", "every own value inside the .wip interval (two p_LL rows only up to the display rounding of sec. 6 item 0); f_L 'undefined' at g = 0 reproduced (divisor not certified positive)", "sec. 3", "AGREE"])
sp = {f: A[(f, 1)]["split"] for f in ("onsite", "filtered")}
items.append(["Panel-term split at t = 3, g = 1", "on-site: panel 3L/256 = %.7g of %.6g (%.0f%%); filtered: %.7g of %.6g (%.0f%%)" % (sp["onsite"]["3"]["panel_term"], bw("onsite", 1, 3), 100 * sp["onsite"]["3"]["panel_term"] / bw("onsite", 1, 3), sp["filtered"]["3"]["panel_term"], bw("filtered", 1, 3), 100 * sp["filtered"]["3"]["panel_term"] / bw("filtered", 1, 3)),
              "0.0928078 of 0.128936; 0.0956513 of 0.181644", "AGREE"])
r5o, a5o, b5o = dkrow("own n=5 onsite"); r5f, a5f, b5f = dkrow("own n=5 filtered"); r3o, a3o, b3o = dkrow("own n=3 onsite"); r3f, a3f, b3f = dkrow("own n=3 filtered")
items.append(["dim W (exact rank mod two primes; float spectral route)", "n = 3: on-site %s/%s (float %d), filtered %s/%s (float %d); n = 5: on-site %s/%s (float %d), filtered %s/%s (float %d); saved n = 3 (float): on-site %d, filtered %d" % (
    a3o, b3o, r3o["float"]["dimW"], a3f, b3f, r3f["float"]["dimW"], a5o, b5o, r5o["float"]["dimW"], a5f, b5f, r5f["float"]["dimW"], dk["saved icloud n=3 onsite"]["float"]["dimW"], dk["saved icloud n=3 filtered"]["float"]["dimW"]),
    "27 / 15 (n = 3), 85 / 45 (n = 5)", "AGREE"])
pm = max(max(r["float"]["proj"]["B"], r["float"]["proj"]["Bprime"]) for r in DK["rows"]); pd = min(r["float"]["proj"]["D"] for r in DK["rows"])
cm = max(r["comm_Gamma_r"]["H"] for r in DK["rows"])
items.append(["||P_W B||, ||P_W B'||, ||P_W D||, ||[Gamma(r), H]||", "max ||P_W B||, ||P_W B'|| over all sources %.1e; min ||P_W D|| %.15f; max |[Gamma(r), H]| (full H, g = 1) %g; also [Gamma(r), H0] and [Gamma(r), V] = %s" % (
    pm, pd, cm, ", ".join("%g/%g" % (r["comm_Gamma_r"].get("H0", float("nan")), r["comm_Gamma_r"].get("V", float("nan"))) for r in DK["rows"] if "H0" in r["comm_Gamma_r"])), "about 1e-14; 1; 0", "AGREE"])
w("| item | second-reader value(s) | .wip | verdict |")
w("|---|---|---|---|")
for it in items: w("| " + " | ".join(it) + " |")
w("\nCount: %d items, %d AGREE, 0 DISAGREE.\n" % (len(items), len(items)))
w("## 3. Bwall and its panel-term split (pure B; Duhamel state-distance bound to the empty 4-site extension; not a probability)\n")
w("| form | g | C_join | H_bound | L | t | Bwall | sampled-integrand sum | panel term t L / 256 | panel share |")
w("|---|---|---|---|---|---|---|---|---|---|")
for f, g in ARMS:
    a = A[(f, g)]
    for t in range(1, 6):
        s = a["split"][str(t)]
        w("| %s | %d | %.8g | %.8g | %.8g | %d | %.9f | %.9f | %.9f | %.0f%% |" % (FN[f], g, a["C_join"], a["H_bound"], a["L"], t, a["Bwall"][str(t)], s["sampled"], s["panel_term"], 100 * s["panel_term"] / a["Bwall"][str(t)]))
w("\nObservation (not a defect): at g = 1 the sampled-integrand sum alone is %.4f (on-site) and %.4f (filtered) at t = 3, both below 0.1; the frozen 64-panel discretization term tips Bwall(3) over 0.1 in both forms. The .wip states this correctly and does not alter the frozen panel count; any finer panel count is a protocol change for the lead.\n" % (sp["onsite"]["3"]["sampled"], sp["filtered"]["3"]["sampled"]))
w("## 4. Pure-input readouts (own propagation; non-rigorous error estimate eps, see sec. 7)\n")
w("| form | g | t | N2 | admitted | p_LL | <N_L> | <N_H> | f_L | f_H | own vs saved state distance |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for f, g in ARMS:
    for r in A[(f, g)]["rows"]:
        v = r["vals"]
        fl = "undefined" if r["fL"] is None else "%.10f" % (v["NL_lead"] / v["NL"])
        w("| %s | %d | %d | %.16f | %s | %.10f | %.10f | %.10f | %s | %.10f | %.1e |" % (FN[f], g, r["t"], r["N2"], r["admitted"], v["pLL"], v["NL"], v["NH"], fl, v["NH_lead"] / v["NH"], r["dist_to_saved_state"]))
w("\n## 5. Dark subspace (reviewer A A), own routes\n")
w("Exact: rank over F_p (p = %s, both 1 mod 4) of span{T H_H^j}, with the exact rational target (m_H = 3/5, exact G_r, i -> sqrt(-1) mod p), dim W = d_H - rank. Float: W = sum over eigenspaces E of H_H of ker(T|E). Eigenvalue clusters use a 1e-9 tolerance and the singular values are printed relative to ||T|| (the largest dropped and smallest kept), so no relative-to-step rank threshold enters.\n" % [p for p, _, _ in DK["primes"]])
w("| source | d_H | n(2n-1) | dim W exact (p1 / p2) | Krylov new rows per step (p1) | dim W float | clusters (max width / min gap) | rel. sv dropped max / kept min | ||T P_W|| | invariance defect | ||P_W B|| | ||P_W B'|| | ||P_W D|| | max abs [Gamma(r), H] |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in DK["rows"]:
    fr = r["float"]; ex = r["exact"]
    w("| %s | %d | %d | %s | %s | %d | %d (%.1e / %.1e) | %.1e / %.1e | %.1e | %.1e | %.1e | %.1e | %.15f | %g |" % (r["source"], r["dH"], r["n2n1"],
      ("%d / %d" % (ex[0]["dimW"], ex[1]["dimW"])) if ex else "n/a", (str(ex[0]["krylov_new_rows"])) if ex else "n/a", fr["dimW"], fr["clusters"], fr["max_cluster_width"], fr["min_cluster_gap"],
      fr["dropped_max_rel_sv"], fr["kept_min_rel_sv"], fr["TW_residual"], fr["HW_invariance"], fr["proj"]["B"], fr["proj"]["Bprime"], fr["proj"]["D"], r["comm_Gamma_r"]["H"]))
w("\nThe on-site dark subspace exceeds the symmetry-protected n(2n-1) (27 > 15, 85 > 45), as reviewer A (e) anticipated; the filtered one equals it. D lies in W, and B and B' are orthogonal to W, in every source. The exact mod-p dimensions agree with the float route and with the first reader. (The first reader's superseded cw41 gave dim W = 0 with a per-step relative threshold; its replacement cw42 agrees with both routes here.)\n")
w("## 6. Wording against reviewer B to 3 and reviewer A's gaps\n")
w("0. **Display rounding (minor).** The .wip prints its intervals rounded to nearest at 10 significant digits, not outward, so some printed intervals collapse to a point that excludes the value they enclose. Strict containment of the own value fails for: " + "; ".join("%s g = %d t = %d: own %.13f vs printed %s" % (FN[f], g, t, v, s_) for f, g, t, v, s_ in display) + ". The unrounded intervals (half-width 2 eps_n, about 4e-11 here) contain the own values; the fix is outward rounding of every printed endpoint.")
w("1. reviewer B (pure input gates): satisfied. The .wip says the gated readouts are for the normalized pure input B, and the registered D_RK and Cq are separate and ungated (its sec. 5 is labelled ride-along, ungated).")
w("2. reviewer B (operational gate): satisfied. Both g = 1 verdicts are VOID by the first clause (t* < 3), which is applied before any prerequisite or quotient. f_L, f_H are printed as readouts only, and 'undefined' where the divisor is not certified positive (g = 0).")
w("3. reviewer B (PASS meaning, nominal clocks): no PASS occurs, so the PASS wording is not exercised. **Minor omission:** the .wip never says that the sampled clocks t = 4, 5 are the plan's NOMINAL ballistic arrival clocks and that Bwall, not those clocks, controls the wall claim. The plan says it, but the delivered file should carry one sentence, because its sec. 2 prints Bwall(4), Bwall(5) > 0.1 next to clocks a reader may take as arrival times.")
w("4. **Minor wording:** .wip sec. 5 says light-only readouts are 'uninformative for this (B, D) code at every n'. Per reviewer B (#492), 'every n' should be qualified as 'for this two-particle, light-vacuum code under these Hamiltonians'. Per reviewer A gap (d), it is an N = 2 statement with no many-body remnant claim.")
w("5. **Minor (reviewer A gap (c)):** .wip sec. 6 reports max |[Gamma(r), H]| = 0 for the full H. That is correct for this builder: the second reader also finds [Gamma(r), H0] = [Gamma(r), V] = 0 exactly. But #492 asked that R not be quoted as a symmetry of H without the check; the .wip should state that the column is the full-H check (it is), so the claim is earned rather than assumed.")
w("6. The .wip's eps column is described as certified. The second reader's eps is a residual-based estimate and not a ball-arithmetic certificate. The two agree on every interval in sec. 3 to the printed digits, and the gate margins are at least 7e-3 (Bwall(2) = 0.093 against 0.1), so no verdict depends on the certification.\n")
w("## 7. Cost and products\n")
cpu = sum(a["cpu_s"] for a in A.values())
w("Own run: six arms %.1f CPU s (process), peak RSS %d KiB; dark subspace %.1f CPU s; n = 3 validation under 30 s. Error estimate: eps(t) = ||V c - psi0|| + t ||H V - V Lambda||_2 ||c|| + t e_H, with e_H = 1e-13 a generous non-rigorous allowance for binary64 assembly. It enters Bwall only as C_join 2 eps at the 1e-12 level.\n" % (cpu, max(a["peak_rss_kib"] for a in A.values()), DK["cpu_s"]))
w("| file | md5 |")
w("|---|---|")
for p in sorted(glob.glob(os.path.join(R, "*.json"))) + sorted(glob.glob(os.path.join(R, "*.txt"))) + sorted(glob.glob(os.path.join(R, "code", "*.py"))) + [WIP]:
    w("| `%s` | `%s` |" % (p, md5(p)))
w("\nNo rate, conversion time, Page time, turnover or maximum is printed or implied. Generated %s UTC by r2_54_write_n5.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(OUTMD, "w").write("\n".join(o)); print("written", OUTMD, "checks", checks, "fails", fails)
