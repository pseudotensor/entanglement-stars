#!/usr/bin/env python
"""obs_p26_slope_ratios_v1.py (builder analyst; engine spec 36.4 (4): "P26: a table of 2 pi T_fit over the stored dense slope kappa_T and
over the applied-field root slope kappa_m at both phases from obs_late_phase_v1.md and the J3' records; no sentence changes (version 1.3
stands, cleared by #22); the attribution candidate goes to the ledger"). Record-only.

For each J3' record (an obs_raymap_solved json: its frozen record's T/rs and its budget rows, one per packet omega r_s, m/omega and hold, with
the fitted T_fit/T_H, T_H = 1/(8 pi G E) at the frozen record's E, the frozen-stage flag and the withheld reason) the frozen record's E, the
stored dense surface slope kappa_T (d_R(N + v) at R_t of its dense solve) and the applied-field root slope kappa_m are read from
obs_late_phase_v1.json's row at that T/rs (the J3' json's own frozen entry printed beside, a check), and per budget row
  2 pi T_fit/kappa_T = (T_fit/T_H)/(4 G E kappa_T),   2 pi T_fit/kappa_m = (T_fit/T_H)/(4 G E kappa_m),
since 2 pi T_H = 1/(4 G E). Withheld fits print their reason and no ratio. Nothing is attributed; WEB40F P26's factorization
T/T_H(E) = (1 + D_1)(1 + D_2)(1 + delta_conn) with D_2 = 4 G E kappa_s - 1 is the return's reading, quoted, not this lane's.
Args: --late obs_late_phase_v1.json --rec label=history=json (repeated; history is obs_late_phase's label) --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--late", required=True)
    ap.add_argument("--rec", action="append", required=True, help="label=history=J3P json")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    LP = json.load(open(a.late))
    hist = {h["label"]: h for h in LP["histories"]}
    L = ["## 2 pi T_fit against the stored surface slopes at the J3' records (obs_p26_slope_ratios_v1.py; engine spec 36.4 (4), WEB40F P26; record-only)", "",
         "- per J3' record: the frozen record's E, kappa_T (the stored dense d_R(N + v) at R_t) and kappa_m (the applied-field root slope) from obs_late_phase_v1.json at its T/rs (the J3' json's own frozen values beside); per packet and hold the stored T_fit/T_H (T_H = 1/(8 pi G E)) and the two ratios 2 pi T_fit/kappa_T = (T_fit/T_H)/(4 G E kappa_T) and 2 pi T_fit/kappa_m = (T_fit/T_H)/(4 G E kappa_m); \"frozen stage\" the J3' record's own flag; a withheld fit prints its reason and no ratio; G = g of the J3' run (printed per record).",
         "- quoted, not this lane's reading: WEB40F P26's exact factorization T/T_H(E) = (1 + D_1)(1 + D_2)(1 + delta_conn), D_2 = 4 G E kappa_s - 1 (the stored slope's factor); TEMPERATURE_solved_exterior_v1.3's sentence stands unchanged (36.4 (4)).", ""]
    for spec in a.rec:
        lab, hname, path = spec.split("=", 2)
        J = json.load(open(path))
        fr = J["frozen"]; T = fr["T_over_rs"]
        g = float(J["args"]["g"])
        rows = hist[hname]["rows"]
        lr = min(rows, key=lambda r: abs(r["T_over_rs"] - T))
        if abs(lr["T_over_rs"] - T) > 1e-9:
            L.append("### %s: obs_late_phase_v1 holds no %s row at T/rs %.4f; skipped" % (lab, hname, T)); continue
        E, kT, km = lr["E"], lr["kappa_T"], lr["kappa_m"]
        fT, fm = 4.0 * g * E * kT, 4.0 * g * E * km
        L.append("### %s (%s; frozen at T/rs %.4f)" % (lab, path.split("/")[-1], T))
        L.append("")
        L.append("- obs_late_phase_v1 (%s) at T/rs %.4f: E %.6f, kappa_T %.6e, kappa_m %.6e; with g %g: 4 G E kappa_T %.6f, 4 G E kappa_m %.6f. The J3' json's frozen entry: E %.6f, kappa_T %.6e, kappa_m %.6e, four_gE_kappa_T %.6f." % (
            hname, lr["T_over_rs"], E, kT, km, g, fT, fm, fr["E"], fr["kappa_T"], fr["kappa_m"], fr.get("four_gE_kappa_T", float("nan"))))
        L.append("")
        L.append("| omega r_s | m/omega | hold (r_s) | frozen stage | T_fit/T_H | 2 pi T_fit/kappa_T | 2 pi T_fit/kappa_m | withheld |")
        L.append("|---|---|---|---|---|---|---|---|")
        for b in J["budget"]:
            t = b.get("T_over_T_H")
            L.append("| %g | %g | %g | %s | %s | %s | %s | %s |" % (b["omega_rs"], b["m_over_omega"], b["hold_over_rs"], "yes" if b.get("frozen_stage") else "no",
                                                               ("%.4f" % t) if t is not None else "-", ("%.4f" % (t / fT)) if t is not None else "-",
                                                               ("%.4f" % (t / fm)) if t is not None else "-", b.get("T_withheld") or ""))
        L.append("")
    L.append("- every number above is printed by the script from obs_late_phase_v1.json and the J3' records' jsons.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
