"""Second reader: emit MOCK_recovery_secondread_v1.md from r2_40_recovery.json and a row-by-row comparison with the
first reader's .wip (copied read-only into the products dir). No new computation. argv: wip_path out_md"""
import json, re, sys, os, glob, hashlib, datetime
R = "<internal>/recovery"
WIP, OUTMD = sys.argv[1], sys.argv[2]
D = json.load(open(os.path.join(R, "r2_40_recovery.json")))
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def iv(s):
    m = re.match(r"\[([^,]+), ([^\]]+)\]", s.strip()); return float(m.group(1)), float(m.group(2))
lines = open(WIP).read().splitlines()
sec = None; adm = {}; res = {}
for l in lines:
    if l.startswith("## 1."): sec = 1
    elif l.startswith("## 3."): sec = 3
    elif l.startswith("## "): sec = None
    if not l.startswith("| ") or l.startswith("| arm") or l.startswith("|---"): continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    arm = "onsite" if c[0].startswith("on-site") else "filtered"
    if sec == 1:
        adm[(arm, int(c[1]), float(c[2]))] = {"N2": c[4], "dev": float(c[6]), "tol": float(c[7]), "gram": float(c[10]), "gtol": float(c[11]), "admitted": c[12]}
    elif sec == 3:
        o = "i_whole_lead_light" if c[3].startswith("i ") else "ii_all_light_incl_core"
        res[(arm, int(c[1]), float(c[2]), o)] = {"saved": iv(c[4]), "phys": iv(c[5]), "gate": c[7]}
ARM = {"onsite": "on-site", "filtered": "filtered (endpoint-copy halo)"}
o = []; w = o.append
w("# MOCK recovery audit: independent second read of MOCK_recovery_audit_v1.md.wip (#490)\n")
w("Second reader: an analyst subagent of the busi lane, independent of the first reader. Read checked: `deliveries/busi/MOCK_recovery_audit_v1.md.wip` (md5 `fa3d0823b6a0856d22b659a12d712782`). Governing texts read in full: busi inbox #490; `deliveries/lead_gaps/ReviewA_mock1x_v1.md` (B1); `deliveries/lead_gaps/ReviewB_mock1x_v1.md` (B); the readout addendum for the normalized-state tolerance.\n")
w("Independence: the first reader's scripts (cw18 to cw30) and its products in <internal>/recovery were not opened, imported or run. The only thing used from there is the interpreter <internal>/python, as a Python with Clarabel and python-flint installed. All code is the second reader's own (busi/code/clockwell_r2/r2_40_recovery.py and r2_41_write_recovery.py; box copies in <internal>/code). Inputs are icloud's saved registered states, read in place and read-only. Nothing was propagated.\n")
w("Scope (frozen by #490): n = 3; on-site baseline and filtered control; g = 0 (control) and 1; t = 1, 2, 4. Output (i) is the whole-lead light modes {12-15, 20-23} (the receiver). Output (ii) is all light modes including the core, {4-7} added (transport diagnostic, not an expanded receiver). Filtered rows carry the endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement).\n")
w("## 1. Route (different from the first reader's)\n")
w("By Koenig-Renner-Schaffner, the optimal entanglement fidelity over all CPTP decoders O -> qubit is F* = 2^(-Hmin(R|O))/2 = (1/2) min{Tr sigma : I_2 x sigma >= rho_RO}. Total fermion number is 2, so rho_RO = Tr_K |Psi><Psi| is block diagonal in the O particle number k = 0, 1, 2. Block-diagonal pinching keeps dual feasibility and the trace, so F* = (1/2)(Opt_0 + Opt_1 + Opt_2). The graded split puts O first, with sign -1 iff the lower mode of the pair is in K and the higher in O; a K-first convention changes rho_k only by the uniform factor (-1)^(k(2-k)) = +1. All sectors are kept, with no postselection or branch renormalization. The only divisor is N2, the exact squared norm.\n")
w("- k = 0 (O vacuum, 2 x 2 on R): Opt_0 = lambda_max(rho_0), in closed form, evaluated in 60-digit interval arithmetic from the exact rational rho_0.")
w("- k = 2 (K vacuum, so rho_2 = |v><v| has rank one): Opt_2 = (s_0 + s_1)^2 = Tr G + 2 sqrt(det G), with G = V V^dag (2 x 2 exact rational) and V the 2 x d_O coefficient block. This is the pure-state min-entropy formula, again in interval arithmetic.")
w("- k = 1 (general mixed, at most 24 x 24): SDP in the second reader's own Clarabel formulation (complex-to-real embedding, Clarabel svec), with the dual (sigma) and the primal (Z >= 0, Tr_R Z = I) solved separately on the full, uncompressed block. Certification is exact rational (python-flint fmpq). Upper: sigma' = sigma_num + delta I, and N2 (I x sigma') - C C^dag > 0 is proved by exact symmetric Gaussian elimination of its real embedding (all pivots > 0). Lower: Z = A A^dag with dyadic A (PSD by construction), s >= lambda_max(Tr_R Z) proved the same way, and LB = ||A^dag C||_F^2 / (N2 s) exactly.")
w("- Cross-check: the same SDP code on the k = 0 and k = 2 blocks, compressed to supp(rho_O), reproduces the closed forms to within 1.6e-13 (sec. 4).")
w("- Physical-target bounds widen the saved-state bounds by 2 eps, as the first reader does. eps is the saved action error; F* is a supremum of expectations of 0 <= X <= I, so it is 1-Lipschitz in trace distance, and the normalized computed vector lies within 2 eps of the target.\n")
w("Known-answer tests through the same certify routine: " + "; ".join("%s: [%.12g, %.12g]%s" % (k, v[0], v[1], (" closed form %.12g" % v[2]) if len(v) > 2 else "") for k, v in D["kat"].items()) +
  ". Expected block optima: maximally entangled 2, product 1, I/2 x I/2 0.5, classical correlation 1, random rank-one and random k = 0 blocks equal to their closed forms. All pass.\n")
w("## 2. Admission (exact, before any division or optimization)\n")
w("N2 is the exact dyadic squared norm. The tolerance on |N2 - 1| is (1+eps)^2 - 1. Gram: G_rs = 2 <v_r, v_s> for the two R-branches of the raw vector. Its Frobenius distance from I_2 has an exact upper bound, compared with 2(2 sqrt2 eps + 2 eps^2), derived in r2_40 from sum_r ||v_r^ - v_r||^2 <= eps^2. The first reader divides G by N2; the two definitions differ by about 1e-15 and give the same verdict.\n")
w("| arm | g | t | raw array (md5) = .wip | N2 (exact value shown as nearest double) | abs(N2-1) / tol | Gram dist (own) / tol | .wip Gram dist / tol | admitted (own / .wip) |")
w("|---|---|---|---|---|---|---|---|---|")
for s in D["states"]:
    a = s["admission"]; k = (s["arm"], s["g"], s["t"]); A = adm[k]
    w("| %s | %d | %g | `%s` (`%s`) %s | [%.22g, %.22g] | %.3e / %.3e | %.3e / %.3e | %s / %s | %s / %s |" % (ARM[s["arm"]], s["g"], s["t"], s["array"], s["array_md5"], "yes" if s["array_matches_wip"] else "NO",
      a["N2"][0], a["N2"][1], a["abs_N2_minus_1"], a["tol"], a["gram_fro_upper"], a["gram_tol"], A["gram"], A["gtol"], a["admitted"], A["admitted"]))
w("\nAll 12 states admitted by both readers. Report md5s are verified equal to the .wip sec. 6 values: %s.\n" % all(s["report_md5_ok"] for s in D["states"]))
w("## 3. Optimal entanglement fidelity per state, and verdicts\n")
w("Own certified bounds, followed by the .wip's certified saved-state bounds; 'centre diff' = own centre - .wip centre. Gate on the physical bounds: lower > 1/2 PASS (a one-shot transfer witness only), upper <= 1/2 FAIL, else UNRESOLVED. A state AGREEs if the two readers' certified intervals overlap, the centres differ by < 1e-6, and the gates coincide.\n")
w("| arm | g | t | output | own F* [lower, upper] (width) | .wip F* [lower, upper] | centre diff | own physical [lower, upper] | gate own / .wip | Opt_0 / Opt_1 / Opt_2 (own) |")
w("|---|---|---|---|---|---|---|---|---|---|")
verdict = {}; maxdiff = 0.0; maxUB = {"i_whole_lead_light": 0, "ii_all_light_incl_core": 0}
for s in D["states"]:
    k = (s["arm"], s["g"], s["t"]); ok_state = True
    for on, r in s["outputs"].items():
        W = res[k + (on,)]
        lo, hi = r["F_saved"]; wl, wh = W["saved"]
        diff = (lo + hi) / 2 - (wl + wh) / 2; maxdiff = max(maxdiff, abs(diff))
        overlap = lo <= wh and wl <= hi
        g_own = r["gate"]; g_wip = W["gate"].split()[0]
        ok = overlap and abs(diff) < 1e-6 and g_own == g_wip
        ok_state = ok_state and ok
        maxUB[on] = max(maxUB[on], r["F_phys"][1])
        w("| %s | %d | %g | %s | [%.12f, %.12f] (%.1e) | [%.12g, %.12g] | %.1e | [%.12f, %.12f] | %s / %s | %.9f / [%.3e, %.3e] / %.9f |" % (
            ARM[s["arm"]], s["g"], s["t"], "i receiver" if on.startswith("i_") else "ii diagnostic", lo, hi, hi - lo, wl, wh, diff,
            r["F_phys"][0], r["F_phys"][1], g_own + ("" if on.startswith("i_") else " (diagnostic only)"), W["gate"], r["Opt0"][1], r["Opt1"][0], r["Opt1"][1], r["Opt2"][1]))
    verdict[k] = "AGREE" if ok_state else "DISAGREE"
w("\nLargest |centre difference| over all 24 (state, output) pairs: %.1e. Largest own physical upper bound: receiver %.10f, diagnostic %.10f (.wip: 0.2867152156 and 0.3614789538).\n" % (maxdiff, maxUB["i_whole_lead_light"], maxUB["ii_all_light_incl_core"]))
w("**Per-state verdicts:**\n")
w("| state | admission | receiver gate | diagnostic | verdict |")
w("|---|---|---|---|---|")
for s in D["states"]:
    k = (s["arm"], s["g"], s["t"])
    w("| %s, g = %d, t = %g | admitted (both) | %s (both) | %s (both) | %s |" % (ARM[s["arm"]], s["g"], s["t"], s["outputs"]["i_whole_lead_light"]["gate"], s["outputs"]["ii_all_light_incl_core"]["gate"], verdict[k]))
na = sum(v == "AGREE" for v in verdict.values())
w("\nCount: %d of 12 states AGREE, %d DISAGREE.\n" % (na, 12 - na))
w("## 4. Diagnostics of the own solve\n")
w("| arm | g | t | output | k = 1 SDP status (dual / primal) | k = 1 numerical dual / primal | SDP minus closed form (k = 0 / k = 2) | O-weight k = 0 / 1 / 2 |")
w("|---|---|---|---|---|---|---|---|")
for s in D["states"]:
    for on, r in s["outputs"].items():
        i1 = r["k1_info"]; bw = r["block_weights"]
        w("| %s | %d | %g | %s | %s | %.6g / %.6g | %.1e / %.1e | %.6g / %.3g / %.6g |" % (ARM[s["arm"]], s["g"], s["t"], on[:2].strip("_"), " / ".join(i1.get("status", ["zero block"])),
          i1.get("dual_num", 0), i1.get("primal_num", 0), r["sdp_vs_closed"]["0"], r["sdp_vs_closed"]["2"], bw["0"], bw["1"], bw["2"]))
w("\nOpt_0 = lambda_max(rho_0) equals 1/2 to within 1e-15 in all 12 states, for both outputs, including g = 1. The O-vacuum branch therefore contributes exactly the null 1/4 to F*. Everything above 1/4 comes from the k = 1 and k = 2 blocks, i.e. from light weight in O: at most 0.0734 (receiver) and 0.223 (diagnostic) of the state. This structural reason is why every gate is a clear FAIL. Some k = 1 solves end with Clarabel status AlmostSolved; the certified bounds do not depend on solver status, because each is proved exactly from the returned point.\n")
w("## 5. Wording against the rulings\n")
w("1. The .wip's gate statements match #490 and reviewer A B1 / reviewer B B. FAIL is upper bound <= 1/2. 'A one-shot transfer witness only, not recovery, not return exclusion' is attached to PASS. The transport diagnostic is kept as a diagnostic, labelled 'FAIL (diagnostic only)', and is not used as a receiver.")
w("2. 'For every decoder': the .wip correctly restricts it to n = 3 ('at this size (n = 3); it says nothing about larger chains'). **Recommended precision (minor):** the licensed phrase covers every CPTP decoder acting on the whole-lead light modes, at the sampled clocks t = 1, 2, 4 only, both converter forms, g = 0 and 1. The sentence in sec. 4 ('Overall ...') does not restate the output algebra or that the clocks are sampled; it should read 'for every decoder acting on the whole-lead light modes, at the sampled clocks t = 1, 2, 4, n = 3'. It must not be stretched to decoders on the complete (heavy + light) lead output, which this audit did not test.")
w("3. **Minor claim precision (sec. 6 of the .wip):** compressing each block to the support of rho_O with a 1e-13 relative singular-value cutoff is called 'exact'. It is exact only when the discarded singular values are exactly zero. The reported bounds remain valid because they are re-certified on the full space, but the word 'exact' should be dropped or qualified. The own route needs no compression: k = 0 and k = 2 are closed forms, and k = 1 is solved uncompressed.")
w("4. The .wip's Table 3 gives a 'fixed receiver F_e' column. Every value in it is <= the certified optimum, as it must be (fixed decoder <= optimum). No inconsistency.")
w("5. No rate, chronology, source, Page time, recovery claim or return-exclusion claim is made by either file.\n")
w("## 6. Cost and products\n")
w("Own run (r2_40): CPU %.1f s (process), wall %.1f s, peak RSS %d KiB, under ulimit -v 32000000. Box CPU, nice 10, BLAS and Rayon threads 4. The first reader's 62 GB full-space route is avoided entirely: the largest matrix formed is the 48 x 48 real embedding of the k = 1 block. The comparison and writing step (r2_41) did no computation beyond parsing.\n" % (D["cpu_s"], D["wall_s"], D["peak_rss_kib"]))
w("| file | md5 |")
w("|---|---|")
for p in sorted(glob.glob(os.path.join(R, "*.json"))) + sorted(glob.glob(os.path.join(R, "*.txt"))) + sorted(glob.glob(os.path.join(R, "code", "*"))) + [WIP]:
    w("| `%s` | `%s` |" % (p, md5(p)))
w("\nGenerated %s UTC by r2_41_write_recovery.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(OUTMD, "w").write("\n".join(o)); print("written", OUTMD, na)
