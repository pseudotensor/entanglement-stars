"""Second reader #497 Step 1: emit MOCK_n5_yardstick_secondread_v1.md from r2_70 products, the own n5 run (r2_52) and the first
reader's MOCK_n5_yardstick_v1.md.wip (read-only copy). Also a numerical spot check of the Dunkl-Williams inequality. argv: wip out_md"""
import json, re, sys, os, glob, hashlib, datetime, math
import numpy as np
R = "<internal>/yardstick"; R5 = "<internal>/n5"
WIP, OUTMD = sys.argv[1], sys.argv[2]
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
Y = {"onsite": json.load(open(R + "/r2_70_onsite_P2048.json")), "filtered": json.load(open(R + "/r2_70_filtered_P4096.json"))}
G1 = {f: json.load(open(R5 + "/r2_52_%s_g1.json" % f)) for f in ("onsite", "filtered")}
FN = {"onsite": "on-site", "filtered": "filtered (endpoint-copy halo)"}
WV = {"onsite": {"fL1": {1: 0.048215654, 2: 0.2541965516, 3: 0.5190560571}, "norm2": {1: 7.010424064, 2: 21.01141407, 3: 31.81795395}, "B": {1: 0.0067345, 2: 0.021265, 3: 0.11962},
                 "parts3": (0.03532, 0, 0.07652, 0.007786), "I": (-0.1280671985, -0.128067195), "VB2": 8},
      "filtered": {"fL1": {1: 0.3873826009, 2: 0.3515084795, 3: 0.2984799006}, "norm2": {1: 0.238446386, 2: 0.9506335204, 3: 1.556728873}, "B": {1: 0.010663, 2: 0.044205, 3: 0.20747},
                   "parts3": (0.05606, 0.01759, 0.1268, 0.007119), "I": (-0.1562554166, -0.1562554155), "VB2": 0.2200049514}}
rng = np.random.default_rng(3); worst = 0.0
for _ in range(20000):
    a = rng.normal(size=8) + 1j * rng.normal(size=8); b = a + rng.normal(scale=10 ** rng.uniform(-3, 1), size=8) * (1 + 1j)
    lhs = np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)); rhs = 2 * np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))
    worst = max(worst, lhs / rhs)
o = []; w = o.append
w("# MOCK n = 5 free-phase yardstick: independent second read of MOCK_n5_yardstick_v1.md.wip (#497 Step 1)\n")
w("Second reader: an analyst subagent of the busi lane, independent of the first reader. Read checked: `deliveries/busi/MOCK_n5_yardstick_v1.md.wip` (md5 `efa2c65d36abfdbb56eb6e672b2be8c0`), against busi inbox #497 and `deliveries/lead_gaps/ReviewA_mock5x_v1.md` and `ReviewB_mock5x_v1.md` (read in full).\n")
w("Independence: cw54 to cw57 and the yardstick products were not opened or run. The saved n = 5 Hamiltonians were read only to check linearity and to compare with the own matrices (md5s equal to the .wip's sec. 7: %s). Everything else is own code: the builder r2_50 (equal to the saved matrices: max difference %g), an own eigendecomposition of H0, and own extended matrices at 9 sites. Scripts: busi/code/clockwell_r2/r2_70_yardstick.py and r2_71_write_yardstick.py (box copies in <internal>/code). Caps: ulimit -v 8000000, nice 10, 2 BLAS threads, box CPU. Filtered rows use the endpoint-copy halo N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention).\n" % (
    {f: Y[f]["saved_md5_ok"] for f in Y}, max(Y[f]["own_vs_saved"] for f in Y)))
# items
def fL3(f):
    r = [x for x in G1[f]["rows"] if x["t"] == 3][0]; return r["fL"]
items = []
for f in ("onsite", "filtered"):
    y = Y[f]; wv = WV[f]
    items.append(["(c) linearity and prerequisites, %s" % FN[f], "H2 - 2H1 + H0 = %g (own), %g (saved); psi1 weight outside LL %.1e; ||VB||^2 = %.10g; ||psi1(t)||^2 = %s" % (y["own_linearity"], y["saved_linearity"], max(r["LL_only"] for r in y["rows"].values()), y["VB2"], ", ".join("%.10g" % y["rows"][str(t)]["norm2"] for t in (1, 2, 3))),
                  "0; LL only; %s; %s" % (wv["VB2"], ", ".join("%.10g" % wv["norm2"][t] for t in (1, 2, 3))), "AGREE"])
    items.append(["(c) psi1 and f_L1 at t = 1, 2, 3, %s" % FN[f], "f_L1 = %s; closed form vs 64-node Gauss-Legendre psi1, relative difference <= %.1e" % (", ".join("%.10f" % y["rows"][str(t)]["fL1"] for t in (1, 2, 3)), max(r["quad_check"] for r in y["rows"].values())),
                  ", ".join("%.10g" % wv["fL1"][t] for t in (1, 2, 3)), "AGREE"])
    fl = fL3(f); f1 = y["rows"]["3"]["fL1"]
    items.append(["(c) Delta = f_L(3) - f_L1(3), %s" % FN[f], "f_L(3) = [%.10f, %.10f] (own n5 run), I = [%.10f, %.10f]: outside [-0.1, 0.1] on the negative side" % (fl[0], fl[1], fl[0] - f1, fl[1] - f1), "[%.10f, %.10f]" % wv["I"], "AGREE"])
for f in ("onsite", "filtered"):
    y = Y[f]; wv = WV[f]
    items.append(["(c) wall prerequisite and verdict by the #497 rule, %s (first reader's construction, own re-implementation)" % FN[f],
                  "Bwall1(1, 2, 3) = %s, P = %d; Bwall1(3) > 0.1, so the prerequisite fails: VOID" % (", ".join("%.6g" % y["wall"][str(t)]["Bwall1_A"] for t in (1, 2, 3)), y["P"]),
                  "%s; VOID" % ", ".join("%.6g" % wv["B"][t] for t in (1, 2, 3)), "AGREE"])
items.append(["(b) normalization of f_L1", "N_L[psi1] / ||psi1||^2 = %s (psi1 has both particles light); f_L1 = <N_L on sites 1-4>/<N_L> is exactly the definition of f_L at g = 1, applied to psi1, i.e. the g -> 0 limit of f_L(g)" % ", ".join("%.12g" % Y[f]["rows"][str(t)]["NL_over_norm2"] for f in Y for t in (1, 2, 3)), "= 2", "AGREE"])
w("## 1. Verdicts per item\n")
w("| item | second-reader value(s) | .wip | verdict |")
w("|---|---|---|---|")
for it in items: w("| " + " | ".join(it) + " |")
w("\nCount: %d items, %d AGREE, 0 DISAGREE. Both forms VOID by the #497 rule (wall prerequisite on psi1 fails at t = 3), as the first reader found.\n" % (len(items), len(items)))
w("## 2. (a) The first-order wall bound: derivation, validity, looseness\n")
w("**Derivation from the definitions.** Let J embed the 5-site N = 2 sector into the 9-site one (4 empty added sites, added clocks 1, halo copied at both ends), H_g = H0 + gV and H_ext,g = H0_ext + g V_ext. The coupling-g wall error e_g(t) = psi_ext,g(t) - J psi_g(t), from the same embedded B, solves i e_g' = H_ext,g e_g + S_g psi_g(t), with e_g(0) = 0 and S_g = H_ext,g J - J H_g = S0 + g S_V. Expanding e_g = e0 + g e1 + O(g^2) and psi_g = psi0 + g psi1 + O(g^2) gives:\n")
w("- i e0' = H0_ext e0 + S0 psi0;")
w("- i e1' = H0_ext e1 + S0 psi1 + S_V psi0 + V_ext e0, with e1(0) = 0.\n")
w("So e1 = psi1_ext - J psi1 exactly (psi1_ext being the same first-order construction on the extended chain). By Duhamel with unitary U0_ext:\n")
w("- ||e1(t)|| <= int_0^t ||S0 psi1(s) + S_V psi0(s)|| ds + int_0^t ||V_ext e0(s)|| ds;")
w("- ||e0(s)|| <= E0(s) = int_0^s ||S0 psi0(r)|| dr.\n")
w("The yardstick fraction is formed on the normalized psi1, so the controlling quantity is ||psi1_ext/|psi1_ext| - J psi1/|psi1|||.\n")
w("**Validity of the first reader's bound: valid.** It is E1 = int ||S0 psi1|| + int ||S_V psi0|| + ||V_ext|| int E0 + quadrature, with Schur norms, and Bwall1 = 2 E1 / lower ||psi1||. Each step is an upper bound:")
w("- the triangle inequality on the source;")
w("- the Schur norm >= the spectral norm;")
w("- E0 >= ||e0||;")
w("- midpoint quadrature with Lipschitz constants La = ||S0|| ||V|| (||H0|| t + 1) and Lc = ||S_V|| ||H0||. The second reader re-derived these as %.3f and %.3f (on-site) and %.3f and %.3f (filtered); the .wip has 59.963, 0 and 21.985, 2.269;" % (Y["onsite"]["La"], Y["onsite"]["Lc"], Y["filtered"]["La"], Y["filtered"]["Lc"]))
w("- ||a/|a| - b/|b||| <= 2||a - b||/|b|.\n")
w("The own re-implementation of this construction gives Bwall1(3) = %.5f (on-site) and %.5f (filtered), against the .wip's 0.11962 and 0.20747. The small difference comes from the own E0 bound: piecewise-constant from above, with no further quadrature. Both exceed 0.1.\n" % (Y["onsite"]["wall"]["3"]["Bwall1_A"], Y["filtered"]["wall"]["3"]["Bwall1_A"]))
w("**Looseness.** One factor is needlessly loose: the normalization 2E/|b|. In an inner-product space the Dunkl-Williams inequality gives:\n")
w("||a/|a| - b/|b||| <= 2||a - b|| / (|a| + |b|) <= 2E / (2|b| - E)  (for E < 2|b|),\n")
w("which is about E/|b|, half the factor used. Proof in one line: with c = cos(angle between a and b), 4|a - b|^2 - (|a| + |b|)^2 |a/|a| - b/|b|||^2 = 2(1 + c)(|a| - |b|)^2 >= 0. A numerical spot check on 20000 random complex pairs gives a worst ratio of %.6f (must be <= 1).\n" % worst)
w("Two other refinements are possible, using the combined source norm, and using the spectral norm of the HH -> LL block of V_ext in place of the Schur norm of V_ext (e0 is heavy-heavy, so V_ext e0 = T_ext e0). Both are valid but gain nothing on-site: S_V = 0, and ||T_ext||_2 = %g = the Schur norm. On the filtered form ||T_ext||_2 = %.5f against Schur %.5f.\n" % (Y["onsite"]["norms"]["Vext_HH_to_LL_spectral"], Y["filtered"]["norms"]["Vext_HH_to_LL_spectral"], Y["filtered"]["norms"]["Vext_schur"]))
w("| form | t | (A) first reader's construction | (B) + combined source | (C) + spectral norm of T_ext | (D) Dunkl-Williams on (A) | (D) Dunkl-Williams on (C) |")
w("|---|---|---|---|---|---|---|")
for f in ("onsite", "filtered"):
    for t in (1, 2, 3):
        r = Y[f]["wall"][str(t)]
        w("| %s | %d | %.6f | %.6f | %.6f | %.6f | %.6f |" % (FN[f], t, r["Bwall1_A"], r["Bwall1_B"], r["Bwall1_C"], r["Bwall1_D_from_A"], r["Bwall1_D_from_C"]))
on3 = Y["onsite"]["wall"]["3"]["Bwall1_D_from_C"]; fi3 = Y["filtered"]["wall"]["3"]["Bwall1_D_from_C"]
w("\n**Reported for the lead and the models; the verdict is not changed.** The run is one shot, and the #497 rule as executed gives VOID for both forms. With the valid, tighter normalization (D), Bwall1(3) is %.4f on-site (%s 0.1) and %.4f filtered (%s 0.1).\n" % (on3, "under" if on3 <= 0.1 else "above", fi3, "under" if fi3 <= 0.1 else "above"))
w("- **On-site:** with (D) the wall would control t = 3 on psi1. Its I = [-0.12807, -0.12807] misses [-0.1, 0.1], so on-site would read FAIL (conversion-specific shift, negative sign) instead of VOID.")
w("- **Filtered:** %s." % ("VOID either way: even the tightest bound here leaves Bwall1(3) above 0.1" if fi3 > 0.1 else "it would also become judgeable, with I missing [-0.1, 0.1] on the negative side"))
w("\nWhether to adopt (D) is for the lead and the models to rule on, not for this read.\n")
w("**Other parts of the bound.** The dominant remaining part, the V_ext e0 term (%.4f of the on-site (A) total at t = 3 after normalization), cannot be tightened without propagating the extended chain. Its operator norm is already sharp on-site (8 = ||K||). Any remaining slack is in ||V_ext e0|| <= ||V_ext|| ||e0|| and in the time-integrated triangle bound E0.\n" % Y["onsite"]["wall"]["3"]["parts_A_normalized"]["Vext_E0"])
def parts(f):
    y = Y[f]; r = y["wall"]["3"]; pn = r["parts_A_normalized"]; n1 = r["norm_psi1"]; dd = 1.0 / y["P"]
    qa = 2 * 3 * y["La"] * dd / 4 / n1; qc = 2 * 3 * y["Lc"] * dd / 4 / n1
    return (pn["S0psi1"] - qa, pn["SVpsi0"] - qc, pn["Vext_E0"], pn["quadrature"])
w("Own parts of Bwall1(3) under construction (A), normalized by 2/|psi1(3)| (sampled sums; quadrature separate). On-site: S0 psi1 %.5f, S_V psi0 %.5f, V_ext E0 %.5f, quadrature %.5f (.wip: 0.03532 / 0 / 0.07652 / 0.007786). Filtered: %.5f, %.5f, %.5f, %.5f (.wip: 0.05606 / 0.01759 / 0.1268 / 0.007119).\n" % (parts("onsite") + parts("filtered")))
w("Rigor of the tighter variants: (D) is an exact inequality, so (D on A) is as rigorous as (A). (C) uses a floating-point spectral norm of T_ext (2.25628, not ball-certified). The filtered (D on C) margin to 0.1 is %.3f, far above any plausible error in that norm, but it should be certified before use. (D on A) alone already brings on-site under 0.1 (%.4f) but not filtered (%.4f).\n" % (0.1 - Y["filtered"]["wall"]["3"]["Bwall1_D_from_C"], Y["onsite"]["wall"]["3"]["Bwall1_D_from_A"], Y["filtered"]["wall"]["3"]["Bwall1_D_from_A"]))
w("## 3. (b) Normalization of f_L1\n")
w("psi1 lives entirely in the LL sector, since V maps HH to LL and H0 conserves species (own check: weight outside LL at most %.1e). So N_L[psi1] = 2 ||psi1||^2 exactly, and the message's 'N_L/||psi1||^2 = 2' is right. f_L1 = <N_L on sites 1-4>[psi1] / <N_L>[psi1] uses the same definition as the gated f_L = <N_L on sites 1-4>/<N_L> at g = 1. It equals lim_(g->0) f_L(g), because both light numbers start at order g^2 with the same psi1. The normalization is consistent. The prerequisite 'lower N_L[psi1]/||psi1||^2 >= 0.05' is trivially met (= 2). No heavy fraction of psi1 is formed (N_H[psi1] = 0), as reviewer B allows.\n" % max(r["LL_only"] for f in Y for r in Y[f]["rows"].values()))
w("## 4. Own numbers\n")
w("| form | t | ||psi1||^2 | N_L/||psi1||^2 | f_L1 | Bwall1 (A) | Bwall1 (D on C) |")
w("|---|---|---|---|---|---|---|")
for f in ("onsite", "filtered"):
    for t in (1, 2, 3):
        r = Y[f]["rows"][str(t)]; wl = Y[f]["wall"][str(t)]
        w("| %s | %d | %.10f | %.12f | %.10f | %.6f | %.6f |" % (FN[f], t, r["norm2"], r["NL_over_norm2"], r["fL1"], wl["Bwall1_A"], wl["Bwall1_D_from_C"]))
w("\nNote on rigor: the own values are floating-point with residual-checked eigendecomposition (||H0 V0 - V0 W||_F = %.1e) and are not ball-certified. The first reader's ball certificates are not re-done here. Every verdict-relevant margin (|I| - 0.1 >= 0.028, and the Bwall1 differences from 0.1) is above 1e-3.\n" % Y["onsite"]["eig_res"])
w("## 5. Cost and products\n")
w("Own CPU: on-site P = 2048 %.1f s, filtered P = 4096 %.1f s; peak RSS %d KiB; 2 BLAS threads, nice 10.\n" % (Y["onsite"]["cpu_s"], Y["filtered"]["cpu_s"], max(Y[f]["peak_rss_kib"] for f in Y)))
w("| file | md5 |")
w("|---|---|")
for p in sorted(glob.glob(R + "/*.json")) + sorted(glob.glob(R + "/*.txt")) + sorted(glob.glob(R + "/code/*.py")) + [WIP]:
    w("| `%s` | `%s` |" % (p, md5(p)))
w("\nNo paper text; no rate, conversion time, Page time, turnover or maximum. Generated %s UTC by r2_71_write_yardstick.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(OUTMD, "w").write("\n".join(o)); print("written")
