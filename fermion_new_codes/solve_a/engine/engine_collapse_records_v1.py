#!/usr/bin/env python
"""engine_collapse_records_v1.py (builder analyst; engine spec 20.5 delivery (1), COLLAPSE_records_v1): per stored collapse record the
handoff quantities assembled from the run's json and fields npz, the replay's handoff npz (engine_replay_v2 or engine_collapse_v19's own)
and the rows' json (engine_rows_v2): E/E0; the Gauss identity E = sum_n eps_n <N^2>_n (its error per record); per interval the Gauss
balance dE against sum_n <N^2>_n Delta eps_n + sum_n Delta<N^2>_n eps_n (the product rule with the interval's means, exact), Delta eps_n
split into the propagated state's part (the sea and the dust under the frozen h_m: "evolution") and the reference's (-Delta Tr[R2 h_n]:
"reset"), the lapse part, and the closure dE minus their sum (the change of the identity's error); beside them the Killing cut current at
the collar integrated over the interval (the record's M1 trapezoid, the transport share); the stress checks (the balance identity's
largest difference for the dust and the pairs, the virial residual relative, the smoothed trace identity); the evolution rows' residuals
(support, exterior, at R_t, R_sh and r_in, the stepping floor); whether the occupation mask equals the negative-energy set; the record's
label. Writes the md and a data npz (per record and shell: the stresses S_r and S_t of the dust and of the pairs, b, the shell totals
eps and J of the total, the pairs and the dust, <N^2>_n, the sea's and R2's shell energies, the dense-solve and applied site fields; the
smoothed S_r,ell and S_t,ell at the coarse nodes). Args: --run dir/tag --handoff npz --spectra npz --rows json --replay-json json --label
text --md --npz."""
import argparse
import json
import math
import os

import numpy as np


def shell_stress(H, k, part, R):
    F = np.asarray(H["F_face_%s" % part][k], float); fc = np.asarray(H["fc_%s" % part][k], float)
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    Srf = F / (4.0 * math.pi * Rf ** 2)
    return 0.5 * (np.concatenate([[Srf[0]], Srf[:-1]]) + Srf), fc / (8.0 * math.pi * R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--spectra", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--replay-json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--md", required=True)
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(a.handoff, allow_pickle=True)
    rs = float(J["args"]["rs"]); R = np.asarray(Z["R"], float)
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    E0 = float(J["head"]["E0"])
    Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)
    rows = {}
    if a.rows and os.path.exists(a.rows):
        for r_ in json.load(open(a.rows))["rows"]:
            rows[round(r_["T_over_rs"], 6)] = r_
    rep = {}
    if a.replay_json and os.path.exists(a.replay_json):
        for r_ in json.load(open(a.replay_json))["rows"]:
            rep[round(r_["T_over_rs"], 6)] = r_
    L = ["## COLLAPSE_records_v1: %s (engine_collapse_records_v1.py)" % tag, "",
         "%s Run: %s (%s), r_s %g, tau %g, ell %g, site fields %s; %d records with the handoff quantities (the stresses: the lane's Noether flux of the declared current, "
         "dust and pairs apart; J_ell = <P_r> = -J_r of the goal's display; the rows: flat-slice ADM, K^ADM = -K)." % (
             a.label, tag, J.get("script"), rs, float(J["args"]["tau"]), float(J["args"]["ell"]), J["args"].get("site_fields"), len(TH)), "",
         "### The Gauss balance per interval (dE against the evolution, reset and lapse parts; the collar's Killing cut current over the interval beside)", "",
         "| T/rs | E/E0 | Gauss identity error | dE | evolution | reset | lapse | closure | collar current x dT |", "|---|---|---|---|---|---|---|---|---|"]
    data = {k: [] for k in ("T", "S_r_dust", "S_t_dust", "S_r_pairs", "S_t_pairs", "b_dust", "b_pairs", "eps_tot", "J_tot", "eps_pairs", "J_pairs",
                            "N2avg", "eps_sea", "eps_R2", "eps_dust", "N", "v", "N_applied", "v_applied", "S_r_ell", "S_t_ell")}
    prev = None
    for k in range(len(TH)):
        T = float(TH[k]); kz = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[kz] - T) > 1e-9:
            continue
        rec = recs.get(round(T, 6), {})
        eps = np.asarray(Z["eps_tot"][kz], float)
        N2 = np.asarray(H["N2avg"][k], float)
        eC = np.asarray(H["eps_sea"][k], float) + np.asarray(H["eps_dust"][k], float); eR = np.asarray(H["eps_R2"][k], float)
        E = float(rec.get("E", np.nan))
        G = float(np.sum(eps * N2))
        gerr = (E - G) / abs(E) if E == E else float("nan")
        line = "| %.3f | %.7f | %+.2e |" % (T / rs, E / E0, gerr)
        if prev is not None:
            a_ = 0.5 * (prev["N2"] + N2)
            evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eR - prev["eR"])))
            lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + eps)))
            dE = E - prev["E"]
            mb = rec.get("M1_balance") or {}
            line += " %+.4e | %+.4e | %+.4e | %+.4e | %+.2e | %s |" % (dE, evo, rst, lap, dE - (evo + rst + lap), ("%+.4e" % mb["evolution_trapezoid"]) if "evolution_trapezoid" in mb else "-")
        else:
            line += " - | - | - | - | - | - |"
        L.append(line)
        prev = {"N2": N2, "eC": eC, "eR": eR, "eps": eps, "E": E}
        Srd, Std = shell_stress(H, k, "dust", R); Srp, Stp = shell_stress(H, k, "pairs", R)
        for key, val in (("T", T), ("S_r_dust", Srd), ("S_t_dust", Std), ("S_r_pairs", Srp), ("S_t_pairs", Stp), ("b_dust", H["b_dust"][k]), ("b_pairs", H["b_pairs"][k]),
                         ("eps_tot", eps), ("J_tot", Z["J_tot"][kz]), ("eps_pairs", Z["eps_pairs"][kz]), ("J_pairs", Z["J_pairs"][kz]), ("N2avg", N2),
                         ("eps_sea", H["eps_sea"][k]), ("eps_R2", eR), ("eps_dust", H["eps_dust"][k]), ("N", Z["N"][kz]), ("v", Z["v"][kz]),
                         ("N_applied", Z["N_applied"][kz]), ("v_applied", Z["v_applied"][kz]), ("S_r_ell", H["S_r_ell"][k]), ("S_t_ell", H["S_t_ell"][k])):
            data[key].append(np.asarray(val, float) if key != "T" else val)
    L += ["", "- evolution: sum_n <N^2>_n Delta(eps of the propagated sea and dust); reset: -sum_n <N^2>_n Delta Tr[R2 h_n]; lapse: sum_n Delta<N^2>_n eps_n "
          "(the interval's means: the three parts sum exactly to the change of sum_n eps_n <N^2>_n); closure: dE minus their sum (the change of the Gauss "
          "identity's error, the collar tail); collar current x dT: the inside Killing energy's cut current at the collar by the trapezoid (the record's M1).", "",
          "### The stress checks and the evolution rows (residual over the largest term; the matter's support and the exterior; at the radii)", "",
          "| T/rs | identity dust, pairs | virial dust, pairs | replay eps, J | res_rr support | res_tt support | res_rr, res_tt exterior | at R_t | at R_sh | at r_in | stepping floor | mask = negative set |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f_ = lambda x: ("%.1e" % x) if isinstance(x, (int, float)) else "-"
    for k in range(len(TH)):
        T = float(TH[k]); key = round(T / rs, 6)
        rp = rep.get(key, {}); rw = rows.get(key, {})
        rad = rw.get("radii", {})
        g2 = lambda lab: ("%.1e, %.1e" % tuple(rad[lab])) if lab in rad else "-"
        sf = rw.get("stepping_floor") or [None, None]
        idn = rp.get("identity_max_abs") or [None, None]; vir = rp.get("virial_rel") or [None, None]
        L.append("| %.3f | %s, %s | %s, %s | %s, %s | %s | %s | %s, %s | %s | %s | %s | %s, %s | %s |" % (
            T / rs, f_(idn[0]), f_(idn[1]), f_(vir[0]), f_(vir[1]), f_(rp.get("eps_rel")), f_(rp.get("J_rel")), f_(rw.get("res_rr_support")), f_(rw.get("res_tt_support")),
            f_(rw.get("res_rr_exterior")), f_(rw.get("res_tt_exterior")), g2("R_t"), g2("R_sh"), g2("r_in"), f_(sf[0]), f_(sf[1]), rp.get("mask_equals_negative_set", "-")))
    L += ["", "- identity: the lattice momentum balance d<Q_in>/dt = -F + sum f (exact at unit clock and zero flow; the largest absolute difference); virial: "
          "S_r + 2 S_t against rho - m b relative (exact for the continuum Dirac field, order (pa)^2 on the lattice and order one at lattice momenta, so large for the "
          "pairs); replay: the replayed sources against the stored shell totals (acceptance 1e-8); the rows' d_t k by the centred difference of successive records.",
          "- the data file %s holds per record and shell S_r, S_t (dust, pairs), b, eps and J (total, pairs), <N^2>_n, the sea's, R2's and the dust's shell energies, "
          "N and v (the record's solve) and the applied fields, with the smoothed S_r,ell and S_t,ell at the coarse nodes r_nodes; R, T, g, ell, tau, m beside." % os.path.basename(a.npz),
          "- every number above is printed by the script from the run's json and npz, the replay's handoff file and the rows' json."]
    np.savez_compressed(a.npz, R=R, r_nodes=np.asarray(H["r_nodes"], float), g=float(J["args"]["g"]), ell=float(J["args"]["ell"]), tau=float(J["args"]["tau"]),
                        m=float(J["args"]["m"]), rs=rs, **{k: np.array(v) for k, v in data.items()})
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
