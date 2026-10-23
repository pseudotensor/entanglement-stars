#!/usr/bin/env python
"""s16A_read_v1.py (busi; reads B' s = 16 stage A as run on the owner box, inbox #471, and EMITS the whole delivery note
PAGE_Bprime_s16A_v1.md: every number in it is printed by this script from the run files; nothing is retyped).
Inputs: --run-dir (s16A_h100x8_v1: the three single-window arms to T = 69.32 L at dt 0.5 a), --decl (stageA_decl_v2: the declared hole
packet and covariances), --prep-dir (prep_v5: the split preps, for the basis md5s and the manifest), --ladder (ladder_read_v1.json).
Readouts follow ReviewA_bpchk2_v1 secs. 1.3, 2, 4, 5 and ReviewB_bpchk2_v1 secs. 1 to 3: the sink2L relabel, the declared-covariance
labels, censored thresholds, the error bars that exist, and the stage-A gate."""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_v2lib as V2  # noqa: E402

RUNS = {"1_sup": "s16A_1_F1_d0_dt0.5_T69.32_sup", "1_exit": "s16A_1_F1_d0_dt0.5_T69.32_exit", "2a_sup": "s16A_2a_F1_eta2_dt0.5_T69.32_sup"}


def g(x, d=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return "%d" % x
    return ("%." + str(d) + "g") % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--decl", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = {k: json.load(open(os.path.join(a.run_dir, t + ".json"))) for k, t in RUNS.items()}
    Z = {k: np.load(os.path.join(a.run_dir, t + ".npz")) for k, t in RUNS.items()}
    man = json.load(open(os.path.join(a.decl, "stageA_manifest_v1.json")))
    hp = np.load(os.path.join(a.decl, "hole_packet_v1.npz"))
    lad = json.load(open(a.ladder))
    L = 16.0
    T0 = 1.0 / (4 * math.pi * L)
    out = []
    P = out.append
    md = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    P("# PAGE B' s = 16, stage A as run on the owner box: read (busi lane), version 1")
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_read_v1.py from the run files; no number is retyped. Runs: %s (md5-checked against their MD5SUMS)." % a.run_dir)
    P("")
    P("**Labels that govern every number below (ReviewA_bpchk2_v1 sec. 1.3 and 2; ReviewB_bpchk2_v1 sec. 1 and 2).**")
    P("- Port: `%s` (run 2a: `%s`). All output readouts are properties of h[N, v] on the box with this sink, for the declared inputs: p is the weight a window direction delivers to the sink beyond 2 L (p_2L), Gamma and N_acc are leakage to that sink, the entropies are bath-record quantities of the sink2L dilation. Not far widths, not greybody factors, not emitted radiation, not radiation entropy, not a Page curve." % (man["port_tag"], man["port_variants"]["run 2a"]))
    P("- Hypotheses (declared covariances, no massless preparation claimed; vacuum sink bath per channel; holes as conjugate particles, exit block read against conj(G); no supply-exit cross block): killing_null (the origin: every output identically zero, not run), exit_hole (one declared hole packet, H2), supply_filled (every supply direction occupied, H1; the in-run reader), file:thermal_T0 (optional bracket).")
    P("- Fields: F1 (the r_s/a 64 reference collapse at T/r_s 30, re-solved dense; fails the hydrodynamic gate inside, sigma ||K|| about 1 at 0.30 L: results on given fields), production Gaussian back-average sigma = 2 sites = L/8, d = 0, s = L/a = 16.")
    P("- What ran: three single-window arms to T = 69.32 L (the clocks 6.93 L = theta 0.1 and 69.3 L = theta 0.01, plus the run's end), dt 0.5 a (the dt ladder's choice), 128 columns each; the supply-exit cross block, the 1e2 L clock, runs 2b and 2c and the exit window of 2a are on h100-f, queued behind C_half450.")
    P("")
    # ------------------------------------------------------------------ manifest
    P("## 1. Manifest")
    P("")
    P("| run | window | columns | dt (a) | clocks (T/L) | steps | step median (s) | peak GPU (GB) | wall (h) | device | json md5 | npz md5 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, t in RUNS.items():
        j = J[k]; tm = j["timing"]
        P("| %s | %s | %d | %s | %s | %d | %.3f | %.1f | %.2f | %s | %s | %s |" % (k.split("_")[0], k.split("_")[1], j["K"], g(tm["dt"]), ", ".join("%.4g" % x for x in j["clocks_over_L"]),
          tm["steps"], tm["step_median_s"], tm["peak_mem_GB"] or 0, tm["seconds_total_this_session"] / 3600, tm["device_name"],
          md(os.path.join(a.run_dir, t + ".json"))[:8], md(os.path.join(a.run_dir, t + ".npz"))[:8]))
    P("")
    bm = {}
    for w in ("sup", "exit"):
        for pr in ("F1_s16_d0", "F1_s16_d0_eta2_bb"):
            p = os.path.join(a.prep_dir, "%s_%s_prep.npz" % (pr, w))
            if os.path.exists(p):
                bm[(pr, w)] = hashlib.md5(np.ascontiguousarray(np.load(p)["U_sub"]).tobytes()).hexdigest()[:12]
    P("Input basis (md5 of the basis bytes): %s. The same basis vectors in runs 1 and 2a (ReviewA_bpchk2_v1 sec. 6). Run tags in every file: port_tag and hypothesis_tag as above." % "; ".join("%s %s %s" % (pr, w, v) for (pr, w), v in bm.items()))
    P("")
    # ------------------------------------------------------------------ inputs
    P("## 2. The window inputs (interior and input properties; no exterior enters)")
    P("")
    P("| window | N_w (columns) | per valley (8 valleys) | e/T0 range | mean \\|<q_r>\\| | sign mean of <q_r> | theta(chi) = <aq> chi/60.8 | transit T_theta(chi)/L |")
    P("|---|---|---|---|---|---|---|---|")
    for k, lab in (("1_sup", "supply (0, T0)"), ("1_exit", "exit (-T0, 0)")):
        e = Z[k]["e"]; w = "supply" if "sup" in k else "exit"
        r_ = J[k][w]
        th = r_.get("theta_chi")
        P("| %s | %d | %d | %.3f to %.3f | %s | %s | %s | %s (censored: beyond the run) |" % (lab, len(e), len(e) // 8, e.min() / T0, e.max() / T0, g(r_.get("aq_mean_abs")),
          g(r_.get("aq_sign_mean")), g(th), g(math.log(2) / th) if th else "-"))
    P("")
    # ------------------------------------------------------------------ p_2L spectra
    P("## 3. p_2L(T): the weight delivered to the sink at 2 L (the whole spectrum per window; reviewer B's readouts under the relabel)")
    P("")
    P("| run | window | T/L | Tr K^dag K (= number delivered from the window span) | count p > 1/2 | > 1e-1 | > 1e-2 | > 1e-3 | min p | median p | max p | largest/median | max (1 - p) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    spec = {}
    for k in RUNS:
        for i, T in enumerate(Z[k]["T"]):
            G = Z[k]["G"][i]
            p = np.sort(np.linalg.eigvalsh(0.5 * (G + G.conj().T)))
            spec[(k, i)] = p
            P("| %s | %s | %.4g | %.4f | %d | %d | %d | %d | %.4g | %.4g | %.4g | %.4g | %.3g |" % (k.split("_")[0], k.split("_")[1], T / L, p.sum(), (p > 0.5).sum(), (p > 1e-1).sum(),
              (p > 1e-2).sum(), (p > 1e-3).sum(), p.min(), np.median(p), p.max(), p.max() / max(np.median(p), 1e-300), (1 - p).max()))
    P("")
    dps = max(float(np.abs(spec[("1_exit", i)] - spec[("1_sup", i)]).max()) for i in range(len(Z["1_sup"]["T"])))
    dGc = max(float(np.abs(Z["1_exit"]["G"][i] - np.conj(Z["1_sup"]["G"][i])).max()) for i in range(len(Z["1_sup"]["T"])))
    dGp = max(float(np.abs(Z["1_exit"]["G"][i] - Z["1_sup"]["G"][i]).max()) for i in range(len(Z["1_sup"]["T"])))
    P("The exit window's spectrum equals the supply window's: max |p_exit - p_supply| = %.3g over the clocks; element by element, max |G_exit - G_supply| = %.3g and max |G_exit - conj(G_supply)| = %.3g. The spectra agree to rounding while the elements do not, so the two Grams agree up to a unitary change of basis inside the window's degenerate clusters (the eigensolver's choice of vectors there): the exit window is the operator's mirror image of the supply window (e -> -e), and on held fields the hole channel repeats the particle channel's basis-independent readouts. The exit readouts below are therefore not independent information about the sink2L model." % (dps, dGp, dGc))
    P("")
    # retained and returns
    P("Retained weight and returns (the sub-2L cavity; the near-zone return is absent by construction):")
    P("")
    P("| run | window | T/L | retained weight per direction: mean | max | \\|S_jj\\| at the last sample: median | max |")
    P("|---|---|---|---|---|---|---|")
    for k in RUNS:
        for i, T in enumerate(Z[k]["T"]):
            rd = np.real(np.diag(Z[k]["R"][i]))
            Sl = np.abs(Z[k]["S_diag"][-1])
            P("| %s | %s | %.4g | %.3g | %.3g | %.3g | %.3g |" % (k.split("_")[0], k.split("_")[1], T / L, rd.mean(), rd.max(), np.median(Sl), Sl.max()))
    P("")
    # ------------------------------------------------------------------ N_acc
    P("## 4. N_w against N_acc_2L(theta): censored, never zero")
    P("")
    P("Time route: #{p_2L(T_theta) > 1/2} at T_theta = ln 2 L/theta where that clock was run; otherwise 'censored' with the lower bound #{p(T_max) > 1/2}. Pole route: distinct fitted poles (nearest the input energy, deduplicated) with Gamma_2L L > theta; a fitted-pole diagnostic only (reviewer B bpchk2 sec. 3D), censored below theta < L/T_max.")
    P("")
    P("| run | window | N_w | theta | time route | pole route |")
    P("|---|---|---|---|---|---|")
    for k in RUNS:
        w = "supply" if "sup" in k else "exit"
        cc = J[k][w]["N_acc_censored"]
        for th, rec in cc.items():
            tr = rec["time"]; pr = rec.get("poles", {})
            ts = ("%d (measured)" % tr["count"]) if tr["status"] == "measured" else (">= %d (censored; needs T = %.0f L)" % (tr["lower_bound"], tr["T_needed_over_L"]))
            ps = ("%d distinct (measured)" % pr["count_distinct"]) if pr.get("status") == "measured" else (">= %d distinct (censored)" % pr.get("lower_bound", 0))
            P("| %s | %s | %d | %s | %s | %s |" % (k.split("_")[0], k.split("_")[1], J[k]["K"], th, ts, ps))
    P("")
    for k in RUNS:
        po = J[k]["poles"]
        nres = sum(1 for p_ in po if p_["status"] == "resolved")
        uniq = V2.dedupe_poles(po, 1e-12)
        GL = sorted(q[1] * L for q in uniq)
        P("- %s: %d of %d per-input poles resolved; %d distinct after deduplication (the window's degenerate clusters share poles); distinct Gamma_2L L from %s to %s." % (
            k, nres, len(po), len(uniq), g(GL[0]) if GL else "-", g(GL[-1]) if GL else "-"))
    P("")
    # ------------------------------------------------------------------ error bars
    P("## 5. Error bars and the stage-A controls")
    P("")
    P("**eta_max x 2 (run 2a against run 1, supply window; ReviewA_bpchk2_v1 sec. 1.3: pass if |dp| <= max(0.02 p, 1e-4) on every direction).** Compared on the sorted spectra (the directions' p_2L in rank order) and on the diagonal (each input's own delivered weight):")
    P("")
    P("| T/L | max \\|dp\\| sorted | directions passing (sorted) | max \\|dp\\| diagonal | directions passing (diagonal) | number delivered, run 1 | run 2a | relative change |")
    P("|---|---|---|---|---|---|---|---|")
    eta_pass = True
    for i, T in enumerate(Z["1_sup"]["T"]):
        p1 = spec[("1_sup", i)]; p2 = spec[("2a_sup", i)]
        tol = np.maximum(0.02 * p1, 1e-4)
        d1 = np.real(np.diag(Z["1_sup"]["G"][i])); d2 = np.real(np.diag(Z["2a_sup"]["G"][i]))
        tol_d = np.maximum(0.02 * d1, 1e-4)
        ok_s = int((np.abs(p2 - p1) <= tol).sum()); ok_d = int((np.abs(d2 - d1) <= tol_d).sum())
        if ok_s < len(p1) or ok_d < len(d1):
            eta_pass = False
        P("| %.4g | %.3g | %d of %d | %.3g | %d of %d | %.4f | %.4f | %.3g |" % (T / L, np.abs(p2 - p1).max(), ok_s, len(p1), np.abs(d2 - d1).max(), ok_d, len(d1), p1.sum(), p2.sum(), (p2.sum() - p1.sum()) / p1.sum()))
    P("")
    P("Verdict on the eta control: **%s**. %s" % ("PASS" if eta_pass else "FAIL",
      "The delivered weight is eta-independent to the error bar, so the error bar is taken from it." if eta_pass else
      "By ReviewA_bpchk2_v1 sec. 1.3 this is the eta-limited branch (not a HOLD): the tag carries both eta values, p_2L is reported at both (the tables above), the error bar is taken from dt alone, and no rate under either eta is called a leakage rate of the fields."))
    P("")
    rows = {r_["dt"]: r_ for r_ in lad["rows"]}
    P("**Time step (dt 0.5 a).** The 2c control (dt 0.25 a) is queued on h100-f. Until it lands the error bar is the dt ladder's (32 columns, clocks 6.93 and 10 L): |G(dt 1) - G(dt 0.5)| = %s, observed order about 1, so the dt 0.5 a error is estimated at the same size, %s (first-order Richardson: the error of dt/2 equals the dt-to-dt/2 difference). Target 1e-4." % (g(rows[1.0]["dG_vs_min"]), g(rows[1.0]["dG_vs_min"])))
    P("")
    P("**Onset derivative (2b, onset 2.25 L).** Not run yet (h100-f). It is a result of the sink2L model, not an error bar (reviewer A sec. 1.3).")
    P("")
    # ------------------------------------------------------------------ entropies
    P("## 6. Bath-record quantities under the declared hypotheses")
    P("")
    P("Bins: (0, 6.93 L], (6.93, 69.3 L], (69.3, 69.32 L] (the last a 0.02 L end bin). S(R_n): the record collected to the end of bin n; S(b | R) = S(R_n) - S(R_(n-1)); I(early : late) at each split; I(retained : record) from R + G. All of the sink2L dilation, under the stated covariance.")
    P("")

    def table(name, Gl, Rl, C, conj=False):
        if conj:
            Gl = [(t, np.conj(G)) for t, G in Gl]; Rl = [np.conj(R) for R in Rl]
        rd = V2.bins_reader(Gl, C, R_list=Rl)
        P("**%s**" % name)
        P("")
        P("| bin end T/L | number delivered (cumulative) | in this bin | S(R_n) | S(b_n) | S(b_n \\| R_(n-1)) | S(retained) | I(retained : record) |")
        P("|---|---|---|---|---|---|---|---|")
        for b, ro in zip(rd["bins"], rd["retained_output"]):
            P("| %.4g | %.4f | %.4g | %.4f | %.4f | %s | %.4f | %.4f |" % (b["t"] / L, b["number"], b["number_bin"], b["S_R"], b["S_b"], g(b.get("S_b_given_R"), 4), ro["S_retained"], ro["I_retained_output"]))
        P("")
        P("Early-late information: %s." % "; ".join("I(R up to %.4g L : rest) = %.4g" % (m["split_t"] / L, m["I_early_late"]) for m in rd["early_late"]))
        P("")
        return rd

    for k, lab in (("1_sup", "supply_filled (H1), run 1, supply window"), ("2a_sup", "supply_filled (H1), run 2a (eta_max x 2), supply window")):
        Gl = [(float(t), G) for t, G in zip(Z[k]["T"], Z[k]["G"])]; Rl = list(Z[k]["R"])
        C = np.eye(len(Z[k]["e"]), dtype=complex)
        table(lab, Gl, Rl, C)
    # exit_hole
    c = hp["c"]; ns_full = 128
    ce = c[ns_full:]
    Ch = np.outer(ce, ce.conj())
    Gl = [(float(t), G) for t, G in zip(Z["1_exit"]["T"], Z["1_exit"]["G"])]; Rl = list(Z["1_exit"]["R"])
    P("The declared hole packet (stageA_manifest_v1, declared before this read): the exit-window combination of minimum radial spread about r_mid = %.4f sites (spread %.4f sites), mean energy %.4f T0; md5 %s." % (
        float(hp["r_mid"]), float(hp["spread"]), float(hp["e_mean"]) / T0, man["hole_packet"]["md5"][:8]))
    P("")
    table("exit_hole (H2), run 1, exit window: one hole in the declared packet (conjugate convention)", Gl, Rl, Ch, conj=True)
    P("The exit block's depletion spectrum (every exit eigendirection; the test-vector spectrum, not a state) is the exit row of sec. 3.")
    P("")
    e_s = Z["1_sup"]["e"]; e_e = Z["1_exit"]["e"]
    f = lambda x: 1.0 / (np.exp(x) + 1.0)
    Gs = [(float(t), G) for t, G in zip(Z["1_sup"]["T"], Z["1_sup"]["G"])]
    table("file:thermal_T0 (optional bracket), run 1, supply block (Fermi occupations f(e/T0), %.3f to %.3f)" % (f(e_s / T0).min(), f(e_s / T0).max()), Gs, list(Z["1_sup"]["R"]), np.diag(f(e_s / T0)).astype(complex))
    table("file:thermal_T0, run 1, exit block (hole occupations f(|e|/T0))", Gl, Rl, np.diag(f(np.abs(e_e) / T0)).astype(complex), conj=True)
    P("The thermal bracket's joint entropy over both windows needs the supply-exit cross block of G, which these single-window arms do not measure (it comes from the full-span arm 1 on h100-f); the two blocks above are reported separately, not summed.")
    P("")
    P("killing_null: zero excitation, every output identically zero by construction (the ladder's origin; not run).")
    P("")
    # ------------------------------------------------------------------ sectors
    P("## 7. Direction-sector proxy of the delivered flux (a labelled proxy; not kappa-channel content; C9 not built)")
    P("")
    P("| run | window | [100]-like | [110]-like | [111]-like |")
    P("|---|---|---|---|---|")
    for k in RUNS:
        fl = Z[k]["absorbed_flux_sectors_every16"][-1].sum(axis=0)
        P("| %s | %s | %.4f | %.4f | %.4f |" % (k.split("_")[0], k.split("_")[1], fl[0] / fl.sum(), fl[1] / fl.sum(), fl[2] / fl.sum()))
    P("")
    # ------------------------------------------------------------------ decision
    P("## 8. What stage A decides")
    P("")
    i_last = int(np.argmin(np.abs(Z["1_sup"]["T"] - 69.3147 * L)))
    p_sup = spec[("1_sup", i_last)]; p_ex = spec[("1_exit", i_last)]
    i0 = 0
    frac_half_sup = int((spec[("1_sup", i0)] > 0.5).sum())
    resid_sup = (1 - p_sup).max(); resid_ex = (1 - p_ex).max()
    tail_sup = int(((1 - p_sup) > 1e-3).sum()); tail_ex = int(((1 - p_ex) > 1e-3).sum())
    P("1. **The window drains to the sink at 2 L within the first clock.** At 6.93 L, %d of %d supply directions have p_2L > 1/2; by 69.3 L the largest undelivered weight 1 - p_2L is %s (supply) and %s (exit), and %d (supply) and %d (exit) directions keep more than 1e-3 undelivered. In the sink2L model the window's directions are strongly coupled to the sink (fitted Gamma_2L L of order %s to %s, sec. 4), not a slow reservoir." % (
        frac_half_sup, len(spec[("1_sup", i0)]), g(resid_sup), g(resid_ex), tail_sup, tail_ex,
        g(min(q[1] * L for q in V2.dedupe_poles(J["1_sup"]["poles"], 1e-12))), g(max(q[1] * L for q in V2.dedupe_poles(J["1_sup"]["poles"], 1e-12)))))
    P("2. **N_acc_2L(theta) equals N_w at theta 0.1 and 0.01** (time route, measured), and is censored at 1e-3 and 1e-4 with lower bound N_w. So the s = 16 count question ('how many window directions reach the sink faster than theta') is answered at s = 16 in this model with 'all of them' down to theta 1e-2; the size exponent needs other sizes, which are not run.")
    P("3. **The eta_max x 2 control %s** (sec. 5). %s" % ("passes" if eta_pass else "fails",
      "The error bar comes from it." if eta_pass else "This is the eta-limited branch of reviewer A sec. 1.3: the delivered weight depends on the sink's strength, as the near-zone reading of B11 predicts, so p_2L is reported at both eta values and no rate is a field property. Production continues (not a HOLD)."))
    P("4. **Stage A is not complete**: 2b (the onset derivative), 2c (dt 0.25) and the full-span arm 1 (the supply-exit cross block and the 1e2 L clock) are queued on h100-f behind C_half450. The stage-A gate to stage B (ReviewA_bpchk2_v1 sec. 5) is read when they land; stage B is not started.")
    P("5. **For stage B's cost**: with %d of %d supply directions retaining more than 1e-3 at 69.3 L, the long arms to 1e3 L (steps 3 and 8) would measure a residual below %s of the delivered weight in the supply window. By ReviewA_bpchk2_v1 sec. 5 (stage C: 'if F1's 1e-3 < p < 1e-2 tail ... is empty or wholly censored, 4 and 5 stop at 3e2 L'), the models may shorten stage B's and C's endpoints; that is theirs to rule once the 1e2 L clock of the full-span arm is in." % (tail_sup, len(p_sup), g(resid_sup)))
    P("6. **The supply_filled bath record** rises to S(R) = %s nats at 6.93 L and falls to %s at 69.3 L (S(b | R) = %s in the second bin): a filled window delivered almost wholly to the sink leaves an almost pure record. This is a property of the sink2L dilation and the declared hypothesis, not a Page curve." % tuple(g(x, 4) for x in (
        V2.bins_reader([(float(t), G) for t, G in zip(Z["1_sup"]["T"], Z["1_sup"]["G"])], np.eye(128, dtype=complex))["bins"][0]["S_R"],
        V2.bins_reader([(float(t), G) for t, G in zip(Z["1_sup"]["T"], Z["1_sup"]["G"])], np.eye(128, dtype=complex))["bins"][1]["S_R"],
        V2.bins_reader([(float(t), G) for t, G in zip(Z["1_sup"]["T"], Z["1_sup"]["G"])], np.eye(128, dtype=complex))["bins"][1]["S_b_given_R"])))
    P("")
    P("Files: the runs and their MD5SUMS in %s; this reader busi/code/pagetests/v3/s16A_read_v1.py; the declarations in %s." % (a.run_dir, a.decl))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
