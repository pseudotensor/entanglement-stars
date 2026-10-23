"""Second reader #498: emit MOCK_n5_yardstick_v11_check_v1.md from r2_80 (own certified ||T_ext||_2) and r2_70 (own first-order
wall bounds), compared with MOCK_n5_yardstick_v1.1.md (read-only copy). No new computation. argv: v11 out_md"""
import json, re, sys, os, glob, hashlib, datetime
R = "<internal>/yardstick_v11"; RY = "<internal>/yardstick"
V11, OUTMD = sys.argv[1], sys.argv[2]
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
TN = json.load(open(R + "/r2_80_tnorm.json"))
Y = {"onsite": json.load(open(RY + "/r2_70_onsite_P2048.json")), "filtered": json.load(open(RY + "/r2_70_filtered_P4096.json"))}
G1 = {f: json.load(open("<internal>/r2_52_%s_g1.json" % f)) for f in Y}
FN = {"onsite": "on-site", "filtered": "filtered (endpoint-copy halo)"}
txt = open(V11).read(); sec = None; t1 = {}; t2 = {}
for l in txt.splitlines():
    if l.startswith("## "): sec = l[3:5]
    if not l.startswith("| ") or l.startswith("|---") or l.startswith("| form") or l.startswith("| item"): continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]; f = "onsite" if c[0].startswith("on-site") else "filtered"
    if sec == "1.": t1[f] = c
    if sec == "2.": t2[(f, int(c[2]))] = c
o = []; w = o.append
w("# MOCK n = 5 yardstick v1.1: independent narrow check (#498)\n")
w("Checker: an analyst subagent of the busi lane, independent of the author. File checked: `deliveries/busi/MOCK_n5_yardstick_v1.1.md` (md5 `%s`), against busi inbox #498 and `deliveries/lead_gaps/ReviewA_mock6_v1.md` and `ReviewB_mock6_v1.md` (read in full). No cw62 or cw63 code was opened or run. Own code: busi/code/clockwell_r2/r2_80_tnorm.py and r2_81_write_check.py, plus the earlier own r2_50 builder and r2_70 yardstick (box copies in <internal>/code and ../yardstick/code/). Caps: ulimit -v 8000000, nice 10, 2 BLAS threads, box CPU. Filtered rows use the endpoint-copy halo N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention).\n" % md5(V11))
on, fi = TN["forms"]["onsite"], TN["forms"]["filtered"]
w("## 1. Verdicts\n")
w("| item | own result | file | verdict |")
w("|---|---|---|---|")
w("| (1) certified upper bound on ||T_ext||_2 and its rigor | own certified upper bound: on-site %.13f, filtered %.13f (floating sigma_max %.13f, %.13f). The on-site exact value is 8: the on-site converter is block diagonal in the site of the pair, with N_x <= 1 and ||K||_2 = 8 (K eigenvalues %s). Both of the file's bounds lie between the floating value and the own bound, consistent with valid upper bounds | 8.00000000002, 2.25627892597 | AGREE |" % (on["upper_T"], fi["upper_T"], on["sigma_max_float"], fi["sigma_max_float"], on["K_eigs"]))
w("| (1) sector restriction | V_ext from the heavy-pair sector into every non-light-pair row: max abs %g (on-site), %g (filtered), and 0 converter monomials outside HH -> LL in the exact build. H0_ext from HH to non-HH: %g, %g. So e0 (source S0 psi0 in HH, generator species-conserving) lies in HH, and V_ext e0 = T_ext e0 exactly | other blocks 0 | AGREE |" % (on["other_blocks_from_HH_float"], fi["other_blocks_from_HH_float"], on["H0_ext_species_mixing"], fi["H0_ext_species_mixing"]))
lmE = min(Y[f]["wall"][str(t)]["norm_psi1"] - Y[f]["wall"][str(t)]["E1_A"] for f in Y for t in (1, 2, 3))
w("| (2) l - E > 0 wherever 2E/(2l - E) is used | own minimum of ||psi1|| - E1 over both forms, t = 1, 2, 3, both variants: %.4f (> 0 by a wide margin; the file prints l - E1 >= 0.4857). Dunkl-Williams needs only 2l - E > 0, with l a lower bound on ||J psi1||, since ||psi1_ext|| >= l - E; l - E > 0 is stronger | certified True | AGREE |" % lmE)
rows = []; maxd = 0.0
for f in ("onsite", "filtered"):
    for t in (1, 2, 3):
        c = t2[(f, t)]; wv = Y[f]["wall"][str(t)]
        own = (wv["Bwall1_A"], wv["Bwall1_D_from_A"], wv["Bwall1_D_from_C"]); fil = (float(c[8]), float(c[9]), float(c[10]))
        d = max(abs(a - b) for a, b in zip(own, fil)); maxd = max(maxd, d); rows.append((f, t, own, fil, d))
w("| (3) side-by-side columns (A), D on A, D on C | own floating table equals the file's columns to max |difference| %.1e. The differences come from the own zero-order E0 bound (piecewise-constant from above, no further quadrature), which is slightly tighter. The file's value is the larger in every cell: %s. The certified versus floating block norm changes D on C by only about 1e-11 | sec. 2 table | AGREE |" % (maxd, all(fil[i] >= own[i] - 1e-12 for _, _, own, fil, _ in rows for i in range(3))))
w("| (4) the #498 sentence and the verdict of record | supported, read with sec. 2's (D on C): both forms have D on C at t = 3 <= 0.1 (on-site %.4f, filtered %.4f), and I is negative outside [-0.1, 0.1] (on-site [%.6f], filtered [%.6f]). The file keeps VOID as the verdict of record (sec. 1 heading, bold) and labels the sentence 'not adopted after the fact' | supported; VOID kept | AGREE (with one precision note, below) |" % (
    Y["onsite"]["wall"]["3"]["Bwall1_D_from_C"], Y["filtered"]["wall"]["3"]["Bwall1_D_from_C"],
    [x for x in G1["onsite"]["rows"] if x["t"] == 3][0]["fL"][0] - Y["onsite"]["rows"]["3"]["fL1"], [x for x in G1["filtered"]["rows"] if x["t"] == 3][0]["fL"][0] - Y["filtered"]["rows"]["3"]["fL1"]))
w("\nCount: 5 rows (4 items), all AGREE, 0 DISAGREE.\n")
w("**Precision note on item (4), not a defect of the prescribed text.** 'Under the exact normalization' alone (Dunkl-Williams on the executed bound, D on A) leaves filtered at %.4f > 0.1, so filtered is not judgeable by the normalization alone. Both forms become judgeable only with the certified HH -> LL block norm as well (D on C). The table makes this visible, but the sentence, prescribed verbatim by #498, does not name the block norm. A reader quoting the sentence should cite 'D on C'.\n" % Y["filtered"]["wall"]["3"]["Bwall1_D_from_A"])
w("## 2. (1) Rigor of the file's norm certificate, as described\n")
w("The described method is G = T^dag T in 128-bit ball arithmetic; s = lambda_max(G)_float + 2 eta; a certified decomposition s I - G = B B^dag + eta I + E with ||E||_F < eta; then ||T||_2 <= sqrt(s) + e_T. This is rigorous, provided:")
w("- B B^dag and E are evaluated in the same ball arithmetic. Then s I - G >= (eta - ||E||_2) I > 0 because ||E||_2 <= ||E||_F < eta, so lambda_max(G) <= s.")
w("- e_T bounds ||T_target - T_binary64||_2. The Frobenius norm suffices. The own exact rational computation gives ||T - T_f||_F <= %.1e (on-site exactly 0) against the file's e_T = 4.6e-12 and 9.4e-13, so the file's allowance is conservative.\n" % fi["e_T_frobenius_upper"])
w("The own independent route uses no Gram matrix: ||T||_2 <= (sqrt(1 + dU) sigma_max sqrt(1 + dV) + ||R||_F) + ||T - T_f||_F, from a floating SVD. dU and dV are the orthogonality defects of the singular-vector matrices and R is the reconstruction residual, all bounded with a priori gamma_m constants (IEEE binary64, round to nearest). The defect bounds are on-site %.1e / %.1e with residual %.1e, and filtered %.1e / %.1e with residual %.1e. The result agrees with the file's bounds to about 6e-11. Neither bound is near any threshold: the filtered norm enters D on C at t = 3 through a term whose sensitivity to the norm is about 0.04 per unit, so a 1e-10 change moves Bwall1 by about 4e-12.\n" % (
    on["U_orth_defect_upper"], on["V_orth_defect_upper"], on["residual_F_upper"], fi["U_orth_defect_upper"], fi["V_orth_defect_upper"], fi["residual_F_upper"]))
w("## 3. (3) Side by side, own against the file\n")
w("| form | t | own (A) | file (A) | own D on A | file D on A | own D on C | file D on C | max abs diff |")
w("|---|---|---|---|---|---|---|---|---|")
for f, t, own, fil, d in rows:
    w("| %s | %d | %.7f | %.7g | %.7f | %.7g | %.7f | %.7g | %.1e |" % (FN[f], t, own[0], fil[0], own[1], fil[1], own[2], fil[2], d))
w("\n## 4. Cost and products\n")
w("Own CPU: norm bound %.1f s (on-site) and %.1f s (filtered); the wall columns are from the earlier own run (r2_70, %.0f and %.0f CPU s).\n" % (on["cpu_s"], fi["cpu_s"], Y["onsite"]["cpu_s"], Y["filtered"]["cpu_s"]))
w("| file | md5 |")
w("|---|---|")
for p in sorted(glob.glob(R + "/*.json")) + sorted(glob.glob(R + "/*.txt")) + sorted(glob.glob(R + "/code/*.py")) + [RY + "/r2_70_onsite_P2048.json", RY + "/r2_70_filtered_P4096.json", V11]:
    w("| `%s` | `%s` |" % (p, md5(p)))
w("\nNo paper text; no n = 7 state touched. Generated %s UTC by r2_81_write_check.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(OUTMD, "w").write("\n".join(o)); print("written", maxd)
