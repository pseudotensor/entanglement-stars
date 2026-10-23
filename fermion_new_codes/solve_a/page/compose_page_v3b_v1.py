#!/usr/bin/env python
"""compose_page_v3b_v1.py (solve builder 6, analyst; engine spec 47.1 (1), (2); inbox #81: the cold reading of PAGE_solved_history_v3,
corrections 2 to 8): PAGE_solved_history version 3b. Light parsing only: every number is the delivered v3 summary json's (or arithmetic on it,
printed by this script) or a string of the delivered v3 text asserted present. Version 3 stands as delivered (RELABEL_table_v1.25); 3b
WITHHOLDS every quantity that passes through the collector (S_col and its spread, the outside-weight range, the collected and archive counts,
S_B, S(M) - S(M'), S_vac(archive), the E_K/E_F/E_cross/E_remove/M_chi split, the ledgers' W_R and partners, the held-field entropies, the
first-chain S_col rows, the figure's top panel) until the corrected selector's checks (47.1 (2) (i) to (vii)) and the variants' reruns land,
and carries corrections 2 to 8 on the collector-independent readouts. Outputs: PAGE_solved_history_v3b.md, ITEM7_solved_history_paragraph_v3.md,
ITEM7_fig_page_v5.png with _SOURCES.txt (the flux panel only), PAGE_solved_history_v3b_summary.json.
Args: --v3md --v3json --outdir --checks (results/page/V6J/gen_checks_v6jA_s4gen_v1.json)."""
import argparse, datetime, hashlib, json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = ("The light channel of the extension at first order (owner decision D16): the massless 4-component Dirac field of the same lattice operator, "
         "by which the model is extended for the Page curve and the radiation seen from infinity only, propagated at first order (D16) on the solved "
         "history of the heavy field's collapse, its sources not fed back; the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75 "
         "(the gate's wording in the owner's O2 form once ruled).")
WITHHELD = ("withheld until the corrected selector's checks (engine spec 47.1 (2) (i) to (vii)) and the variants' reruns land: S_col and its spread, "
            "the collected modes' outside-weight range, the collected and archive counts, S_B, S(M) - S(M'), S_vac(archive), the E_K, E_F, E_cross, "
            "E_remove and M_chi split, the ledgers' W_R and partner modes, the held-field entropies (34.03 and 6.15), the first chain's S_col rows 42 to 56, "
            "S_col against A/(4G) and eta A/a^2, and the figure's top panel (inbox #81 correction 1)")
FEEDBACK = ("E_rad^(0)/E = %s: the net signed collar flow through T/r_s 40 over E(40), a diagnostic; the omitted sources' energy relative to the channel's "
            "continued sea is not printed; the channel's excitation above the instantaneous Killing ground state is %s to %s percent of E over 14 to 40 "
            "(part of it an occupation offset at the pocket phases), and the formation's emission reaches the collar after 40 (the first chain's net collar "
            "flow %s, %s of E, by 52)")


def f(v, fm="%.4g"):
    return (fm % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"


def main():
    ap = argparse.ArgumentParser()
    for k in ("v3md", "v3json", "outdir", "checks"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t3 = open(a.v3md).read(); raw = open(a.v3json, "rb").read(); J = json.loads(raw); md5 = hashlib.md5(raw).hexdigest()
    CK = json.load(open(a.checks))
    for q in ("about 2227 r_s after the hold's start", "about 285 r_s after the hold's start", "the linear emptying clock T/r_s 3547",
              "the linear emptying clock T/r_s 16182", "37 PASS, 0 FAIL, 42 tests", "its barrier is 1e-4 to 1e-6"):
        assert q in t3, q
    s4 = J["A_s4"]["rows"]; ext = J["A_to100"]["rows"]; MARK = float(J["A_to100"]["marker_rs"])
    E40 = [r for r in s4 if abs(r["t_over_rs"] - 40) < 1e-9][0]
    fr = E40["collar_flow_acc_weighted"] / E40["E"]
    pe = [100 * r["E_inv_weighted"] / r["E"] for r in s4 if 14 - 1e-9 <= r["t_over_rs"] <= 40 + 1e-9]
    r52 = [r for r in ext if abs(r["t_over_rs"] - 52) < 1e-9][0]
    fb = FEEDBACK % ("%.4e" % fr, "%.2g" % min(pe), "%.2g" % max(pe), "%.3g" % r52["collar_flow_acc_weighted"], "%.2g" % (r52["collar_flow_acc_weighted"] / r52["E"]))
    # flux clause numbers
    mid = [r for r in s4 if 14 - 1e-9 <= r["t_over_rs"] <= 32 + 1e-9 and r.get("L_interval") is not None]
    nneg = sum(1 for r in mid if r["L_interval"] < 0)
    pre = [r for r in s4 if r["t_over_rs"] < 36 - 1e-9 and r.get("L_over_LH_kappaT") is not None]
    pmax = max(max(abs(r["L_over_LH_kappaT"]), abs(r["L_over_LH_TH"])) for r in pre)
    late = [r for r in s4 if 36 - 1e-9 <= r["t_over_rs"] <= 40 + 1e-9]
    lk = [r["L_over_LH_kappaT"] for r in late]; lt = [r["L_over_LH_TH"] for r in late]
    flux = ("the collar energy flow per coordinate interval Delta T, placed at the collar's recordwise stationary-chart clock and compared with the two-channel "
            "continuum greybody at the record's dense surface slope and at 1/(8 pi G E), not an observed thermal flux, is at most %.2g of either comparator in "
            "magnitude before 36, negative at %d of the %d checkpoints from 14 to 32, and %.2g to %.2g times the slope comparator and %.3g to %.3g times the "
            "1/(8 pi G E) comparator over 36 to 40") % (pmax, nneg, len(mid), min(lk), max(lk), min(lt), max(lt))
    C = J["C"]; h15 = C["c15"]["hold_readouts"]; h30 = C["c30"]["hold_readouts"]
    d15 = (h15["inside_start"] - h15["inside_clean_end"]) / h15["inside_start"]; d30 = (h30["inside_start"] - h30["inside_clean_end"]) / h30["inside_start"]
    hold = ("with the fields held at the records of T/r_s 15 and 30 the whole-chain excitation is conserved to the printed digits and leaves through the collar "
            "at 5.4 to 6.3 times the comparator from T/r_s 77 to the reflection clock %.1f after the burst's passage, and at -0.017 to 0.076 from 148 to %.1f, "
            "the drain rule met on neither record within the chain's clean reading: drains in progress; no emptying time beyond linear extrapolations, "
            "2,227 and 285 r_s after the hold's start on the whole-window chord, T/r_s 3,547 and 16,182 on the plateau's slope") % (h15["T_refl"], h30["T_refl"])
    para = ("The massless channel of the lattice Dirac operator propagated at first order (D16) on the solved history of the massive collapse (r_s/a 64, G/a^2 1/4, "
            "m a 0.2 for the matter, sigma/a 4, tau 8), its sources not fed back, the interior beyond the hydrodynamic gate from T/r_s 11.75, on a chain whose exterior "
            "beyond the engine's domain is the vacuum form at the record's charge, read through T/r_s 40, before the formation's emission reaches R_cut 24: the "
            "channel's excitation above the instantaneous Killing ground state at each clock, Tr[h(C - P_-)], reads %.2g percent of the charge at T/r_s 14 and "
            "%.2g to %.2g percent of it afterwards, relative to the instantaneous Killing ground state, whose refilling at the pocket phases is part of the change "
            "between 14 and 16 (the same-clock energy decomposition closes to %.1e); %s. At the collar %s. %s. [the selected archive's entropy: withheld until the "
            "corrected selector's checks and the variants' reruns land (version 3b; engine spec 47.1)] [the light channel's reading from T/r_s 40 to 100 on the "
            "doubled chain, with the no-return clause of the drain rule read on the reruns carrying the signed collar split and the observables of O1 for D13, "
            "once ruled: version 3b's extension]") % (
        100 * [r for r in s4 if abs(r["t_over_rs"] - 14) < 1e-9][0]["E_inv_weighted"] / [r for r in s4 if abs(r["t_over_rs"] - 14) < 1e-9][0]["E"],
        min(p for p, r in zip(pe, [r for r in s4 if 14 - 1e-9 <= r["t_over_rs"] <= 40 + 1e-9]) if r["t_over_rs"] > 14 + 1e-9),
        max(p for p, r in zip(pe, [r for r in s4 if 14 - 1e-9 <= r["t_over_rs"] <= 40 + 1e-9]) if r["t_over_rs"] > 14 + 1e-9),
        max(abs(r["energy_identity_residual_weighted"]) for r in s4), fb, flux, hold[0].upper() + hold[1:])
    os.makedirs(a.outdir, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# Computation A on the solved history: PAGE_solved_history version 3b (engine spec 47.1; inbox #81, the cold reading's corrections 2 to 8)", "",
         "Composed %s by code/page/compose_page_v3b_v1.py from PAGE_solved_history_v3_summary.json (md5 %s) and PAGE_solved_history_v3.md; version 3 stands as "
         "delivered with RELABEL_table_v1.25's rows. Label: %s" % (now, md5, LABEL), "",
         "WITHHELD: " + WITHHELD + ". The corrected collector is the generalized selector (page_history_v6j/v6l --collector-select gen); its checks so far: "
         "(i) the history oracle with gen PASS (dense S_esc to 4.02e-11) and with gen and split admission PASS (9.49e-11); (ii) to (iv) on v6jA_s4gen %s "
         "(||F^dag F - I||_F at most %.2e, the energy identity at most %.2e, S_col per channel at most %.3g of its archive size x ln 2); (v) to (vii) running or with the lead." % (
             "PASS" if CK["verdict"]["pass"] else "FAIL", CK["verdict"]["ii_gram_dev_max"], CK["verdict"]["iii_energy_identity_max_abs"], CK["verdict"]["iv_S_over_bound_max"]), "",
         "## 1. The collector-independent readouts of A (v6bA_s4, R_cut 24 r_s, the collar 16 r_s), with the join check per record (S1 A.1)", "",
         "| T/r_s | t_inf/r_s (collar) | 2gE | E_inv | E_inv/E | inside the collar | beyond | W (inside increment per Delta T + L) | Tr[h_dot (C - P)] | L (per Delta T) | L/L_H(kappa_T) | L/L_H(T_H) | energy identity | collar flow (acc.) | join max dN | join max rel dv |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s4:
        L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["t_over_rs"], f((r.get("t_inf_collar") or 0) / 64.0, "%.3f"), f(r.get("two_gE"), "%.4f"), f(r.get("E_inv_weighted"), "%.5g"), f(100 * r["E_inv_weighted"] / r["E"], "%.3g%%"),
            f(r.get("E_inv_inside_weighted"), "%.5g"), f(r.get("E_out_weighted"), "%.5g"), f(r.get("W_creation"), "%.4g"), f(r.get("W_hdot_weighted"), "%.4g"), f(r.get("L_interval"), "%.4g"),
            f(r.get("L_over_LH_kappaT"), "%.4g"), f(r.get("L_over_LH_TH"), "%.4g"), f(r.get("energy_identity_residual_weighted"), "%.2e"), f(r.get("collar_flow_acc_weighted"), "%.5g"),
            f(r.get("join_max_dN"), "%.2e"), f(r.get("join_max_rel_dv"), "%.2e")))
    L += ["", "Readings (corrections 2, 4, 5): " + fb + ". The excitation is relative to the instantaneous Killing ground state, whose refilling at the pocket phases is part "
          "of the change between 14 and 16; the changes are booked by the record-jump ledger (sec. 3), not by a matched work identity (W and Tr[h_dot (C - P)] are not one). "
          "The flux: " + flux + ".", "",
          "## 2. The first chain's rows after 40 and before the return marker %.0f (A to 100, page_history_v6i on B's records; correction 3)" % MARK, "",
          "Through T/r_s 40 the reading precedes the formation's arrival at R_cut 24. Each row below is before the chain end's first return; the join departs from the vacuum form as printed; S_col and the anchored ledger's W_R are withheld (they read the archive; web40g P09: W_R near 1 certifies capture of the selected partners, not their passage through the core).", "",
          "| T/r_s | E_inv/E | collar flow (acc.) | L/L_H(kappa_T) | L/L_H(T_H) | join max dN | join max rel dv | after first return |", "|---|---|---|---|---|---|---|---|"]
    for r in ext:
        if 40 + 1e-9 < r["t_over_rs"] < MARK - 1e-9:
            L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (r["t_over_rs"], f(r["E_inv_weighted"] / r["E"], "%.4g"), f(r.get("collar_flow_acc_weighted"), "%.4g"),
                     f(r.get("L_over_LH_kappaT"), "%.4g"), f(r.get("L_over_LH_TH"), "%.4g"), f(r.get("join_max_dN"), "%.2e"), f(r.get("join_max_rel_dv"), "%.2e"), r.get("after_first_return")))
    L += ["", "## 3. The record-jump ledger on the s4 settings (A to 100's json; correction 4)", "",
          "| T/r_s | W (s4 table) | W_acc_interval/Delta T | jump check |", "|---|---|---|---|"]
    w4 = {round(r["t_over_rs"], 6): r.get("W_creation") for r in s4}
    for r in ext:
        if r["t_over_rs"] <= 40 + 1e-9 and r.get("W_acc_interval_weighted") is not None:
            dT = r.get("_dT")
            L.append("| %.3f | %s | %s | %s |" % (r["t_over_rs"], f(w4.get(round(r["t_over_rs"], 6)), "%.5g"), f(r.get("W_acc_rate"), "%.5g"), f(r.get("jump_check_weighted"), "%.2e")))
    L += ["", "Sum Q_j (the reset term web40g item 4 names) beside the trapezoid is not in the stored records; it enters the producer before its next run (rule 10), and no matched-work sentence prints meanwhile.", "",
          "## 4. The held fields (correction 6)", "",
          "- v6eA_c15 (the record of T/r_s 15): the clean window from T/r_s 77 (after the burst's passage, ratios to 164 before it) to the reflection clock %.3f; L/L_H 5.4 to 6.3 there; the pre-reflection interior change %.4f of the inside start (%.6g to %.6g, script); the fraction %.6f is the whole-hold signed total, the 37 rows after the reflection clock included (label); the emptying extrapolations, both linear: 2,227 r_s after the hold's start on the whole-window chord and T/r_s 3,547 on the plateau's slope (version 3 secs. 4 and 10), extrapolations only." % (
              h15["T_refl"], d15, h15["inside_start"], h15["inside_clean_end"], C["c15"]["fraction_leaving_script"]),
          "- v6eA_c30 (the record of T/r_s 30): the clean window from 148 to the reflection clock %.3f (ratios 0.033 to 0.37 over 74 to 146); L/L_H -0.017 to 0.076 there; the pre-reflection interior change %.4f (%.6g to %.6g, script); the fraction %.6f is the whole-hold signed total; its end-of-hold W_R is read after the reflection clock (labelled, not printed); extrapolations 285 r_s after the hold's start (chord) and T/r_s 16,182 (plateau's slope)." % (
              h30["T_refl"], d30, h30["inside_start"], h30["inside_clean_end"], C["c30"]["fraction_leaving_script"]),
          "- Version 3 sec. 10's clause-4 medians and clause-3 negative rows used the json's L_at_t, a third convention (neither the tables' L nor defined there); they are records and do not print. The absorber-free reruns with the signed collar split: v6kA_c15 ended (results/page/v6bA_runs/v6kA_c15), v6kA_c30 running on pod 2; clause 3 is read on their split rows before the reflection clock in the extension.", "",
          "## 5. The band top and the hard share (correction 7)", "",
          "The hard share column is dropped (its overflow bin [40, 1e30) T_H is weighted by its midpoint, so a unit share establishes nothing). Archive label: its outside weight beyond R_cut and its velocity selection certify no escape; weight above the band top (1/a, 667 T_H at 40) is bound content. The per-block collected weight above 1/a prints with the corrected archive.", "",
          "## 6. Consistency (correction 9)", "",
          "- The self-test v6b: 37 PASS, 0 FAIL, 5 diagnostics with no pass flag (2 of them duplicated entries), 42 entries.",
          "- The regression (results/page/page_v6_regression_v1.md): the ruling's target S_max 42.956064 at 76 r_s; repair 1: dS_max -9.05; repair 3, the active-subspace collector: dS_max -16.05, dE_rad -0.789 (that is R3).",
          "- Readout 9's kappa 2 barrier: Gamma_2 from 2.9e-10 to 8.5e-4 over the soft bins, 2.7e-6 to 1.7e-3 of Gamma_1 (replacing \"1e-4 to 1e-6\").",
          "- The doubled-chain extension is version 3b's extension in every text (the caption's \"4b\" read as this).", "",
          "## 7. Item 7's paragraph (ITEM7_solved_history_paragraph_v3.md)", "", "Printable after the second route reads it: \"" + para + "\"", ""]
    open(os.path.join(a.outdir, "PAGE_solved_history_v3b.md"), "w").write("\n".join(L) + "\n")
    open(os.path.join(a.outdir, "ITEM7_solved_history_paragraph_v3.md"), "w").write(
        "# Item 7's finite-chain paragraph, version 3 (PAGE_solved_history v3b; engine spec 47.1 (1); inbox #81 corrections 2 to 8)\n\nComposed %s by code/page/compose_page_v3b_v1.py; prints only after the second route has read it. S_col withheld (47.1).\n\nPrintable: \"%s\"\n" % (now, para))
    # the figure: the flux panel only (the top panel withheld)
    F = J["figure"]; x = F["t_inf_over_rs"]
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2))
    for key, lab, col in (("L_over_LH_kappaT", "L/L_H at the record's surface slope", "r"), ("L_over_LH_TH", "L/L_H at 1/(8 pi G E)", "b")):
        pts = [(xx, abs(v), v < 0) for xx, v in zip(x, F[key]) if v is not None and v != 0]
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], col + "-", lw=1.0, label=lab + " (|.|; open: negative)")
        ax.semilogy([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], col + "o", ms=3)
        ax.semilogy([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], col + "o", ms=4, mfc="none")
    ax.set_ylabel("|L| / L_H\n(L per coordinate interval Delta T;\nL_H comparators)", fontsize=7)
    ax.set_xlabel("t_inf / r_s at the collar: a recordwise stationary-chart substitution", fontsize=7)
    ax.legend(fontsize=6.5, loc="lower right")
    cap = textwrap.wrap(LABEL + " " + fb + ". The selected archive's entropy (the top panel of version 4) is withheld until the corrected selector's checks land (engine spec 47.1).", 170)
    fig.text(0.01, 0.005, "\n".join(cap), fontsize=4.8, va="bottom")
    fig.tight_layout(rect=(0, 0.012 + 0.021 * len(cap), 1, 1))
    png = os.path.join(a.outdir, "ITEM7_fig_page_v5.png"); fig.savefig(png, dpi=160)
    open(os.path.join(a.outdir, "ITEM7_fig_page_v5_SOURCES.txt"), "w").write(
        "ITEM7 figure version 5 (written %s by code/page/compose_page_v3b_v1.py): version 4's bottom panel alone, the top panel withheld (inbox #81 correction 1; engine spec 47.1); data PAGE_solved_history_v3_summary.json (md5 %s), its figure block: x t_inf_collar/64, y |L_interval|/L_H for both comparators; colours red and blue.\ncaption: %s\n" % (now, md5, " ".join(cap)))
    json.dump({"written": now, "script": "code/page/compose_page_v3b_v1.py", "source_json_md5": md5, "label": LABEL, "withheld": WITHHELD, "feedback_clause": fb, "flux_clause": flux,
               "hold_clause": hold, "paragraph": para, "checks_ii_iv": CK["verdict"]}, open(os.path.join(a.outdir, "PAGE_solved_history_v3b_summary.json"), "w"), indent=1)
    txt = (open(os.path.join(a.outdir, "PAGE_solved_history_v3b.md")).read() + para).lower()
    bad = [w for w in ("mutual information", "frozen", "prediction", "event horizon", "singularity forms", "toy model", "semiclassical", "mean-field") if w in txt]
    print("composed; guard words:", bad or "none")
    if bad:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
