#!/usr/bin/env python
"""dust_tail_table_v1.py (busi lane, reviewer A): the markdown tables of dust_tail_v1.py, composed from the rows it computed (every
number printed from the rows; the first route's numbers copied from its jsons where given and printed beside, never averaged)."""
import time


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def fmt(x, f="%.6e"):
    return "-" if x is None else (f % x)


def compose_table(rows, first, cuts_rs, prof_rs, m, prov):
    c1 = str(cuts_rs[0]); c2 = str(cuts_rs[1]) if len(cuts_rs) > 1 else c1
    L = ["# The shed dust's second route (dust_tail_v1.py, busi lane): number, Killing energy and number current beyond the collars on the stored slices", "",
         "Composed %s; every number printed by the script. Weighted over the channels (w = 16 kappa). The dust = the stored dust columns with their occupations (the tagged quantities); "
         "h_m = (1/2){N, h_0} + (1/2){v, Q} at each slice's applied fields; beyond R_c = the sites with R > R_c; E_sym = the symmetrized Killing density summed beyond R_c (the first "
         "route's E_out_dust); particles and holes = the dust's occupation of the exterior block's Killing levels with omega > 0 and omega < 0 (holes printed as a positive weight); "
         "E/particle in units of m = %g; F = the dust's number current through the cut (outward positive). The first route's numbers (solve lane: D_dust_tail_v1 and "
         "engine_massive_channel_D_v2's jsons) stand beside where given, never averaged; its holes_dust is the negative of the weight printed here." % (utc(), m), ""]
    L += ["## A. This route, weighted", "",
          "| T/r_s | N beyond %s | E_sym beyond %s | particles; E beyond %s | E/particle (m): E_p/N_p; E_sym/N | holes beyond %s | N beyond %s | E_sym beyond %s | particles; E beyond %s | E/particle (m) beyond %s | F through %s r_s | F through %s r_s |" % (
              c1, c1, c1, c1, c2, c2, c2, c2, c1, c2), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["weighted"][c1], r["weighted"][c2]
        L.append("| %.3f | %.6e | %.6e | %.6e; %.6e | %s; %s | %.4e | %.6e | %.6e | %.6e; %.6e | %s | %+.6e | %+.6e |" % (
            r["T_over_rs"], a["N"], a["E_sym"], a["particles"], a["E_particles"], fmt(a["E_per_particle_over_m"], "%.4f"), fmt(a["E_sym_per_N_over_m"], "%.4f"), a["holes"], b["N"], b["E_sym"],
            b["particles"], b["E_particles"], fmt(b["E_per_particle_over_m"], "%.4f"), a["F"], b["F"]))
    if first["tail"] or first["dv2"]:
        L += ["", "## B. Beside the first route (its numbers copied from its jsons; the difference this minus first printed; nothing averaged)", "",
              "| T/r_s | quantity | beyond %s: this route | first route | difference | beyond %s: this route | first route | difference |" % (c1, c2), "|---|---|---|---|---|---|---|---|"]
        pairs_ = (("N (dust weight)", "N", "dust_weight"), ("particles", "particles", "particles_dust"), ("E particles", "E_particles", "E_particles_dust"),
                  ("E/particle (m)", "E_per_particle_over_m", "E_per_particle_over_m"), ("holes (weight)", "holes", "-holes_dust"), ("E_sym (E_out_dust)", "E_sym", "E_out_dust"))
        for r in rows:
            ft = first["tail"].get(round(r["T_over_rs"], 6))
            if ft is None:
                L.append("| %.3f | (no first-route row at this clock) | | | | | | |" % r["T_over_rs"]); continue
            for lab, mine, theirs in pairs_:
                cells = []
                for c in (c1, c2):
                    fb = ft.get("beyond_%s" % c.rstrip("0").rstrip(".")) or ft.get("beyond_%s" % c) or {}
                    tv = fb.get(theirs.lstrip("-"))
                    if tv is not None and theirs.startswith("-"):
                        tv = -tv
                    mv = r["weighted"][c].get(mine)
                    cells += [fmt(mv), fmt(tv), fmt((mv - tv) if (mv is not None and tv is not None) else None, "%+.3e")]
                L.append("| %.3f | %s | %s |" % (r["T_over_rs"], lab, " | ".join(cells)))
            mv = r["weighted"][c1]["F"]; tv = ft.get("F_dust_16_weighted")
            L.append("| %.3f | F through 16 r_s | %s | %s | %s | | | |" % (r["T_over_rs"], fmt(mv, "%+.6e"), fmt(tv, "%+.6e"), fmt((mv - tv) if tv is not None else None, "%+.3e")))
    L += ["", "## C. Checks per slice (maxima over the channels; the structural fact: the sea's density 1, no excess number and no current beyond any cut)", "",
          "| T/r_s | sea density - 1 | sea excess beyond %s; %s | sea current through %s; %s | dust Gram - 1 | dust-sea overlap | spectrum vs stored e_k | spectrum symmetry | sum n - N | block identity | F bond - F sparse | total dust - nominal | density route - Killing (total) | K_density - stored K_dust beyond %s; %s (rel) | bare energy - E_dust_shells | applied fields vs fields npz | seconds |" % (
              c1, c2, c1, c2, c1, c2), "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ck = r["checks"]; a, b = r["weighted"][c1], r["weighted"][c2]; rc = r.get("run_checks") or {}; rb = rc.get("beyond") or {}
        L.append("| %.3f | %.1e | %+.1e; %+.1e | %+.1e; %+.1e | %.1e | %.1e | %s | %s | %.1e | %.1e | %.1e | %+.1e | %+.1e | %s; %s | %s | %s | %.0f |" % (
            r["T_over_rs"], ck["max_sea_density_dev"], a["max_sea_excess"], b["max_sea_excess"], a["max_sea_current"], b["max_sea_current"], ck["max_dust_gram_dev"], ck["max_dust_sea_overlap"],
            fmt(ck["max_spectrum_vs_stored"], "%.1e"), fmt(ck["max_spectrum_symmetry"], "%.1e"), max(a["max_sum_n_minus_N"], b["max_sum_n_minus_N"]), max(a["max_block_identity"], b["max_block_identity"]),
            max(a["max_F_minus_F_sparse"], b["max_F_minus_F_sparse"]), ck["total_number_minus_nominal"], ck["density_route_minus_killing_total"],
            ("%+.1e (%s)" % (rb[c1]["diff"], fmt(rb[c1]["rel"], "%.1e"))) if c1 in rb else "-", ("%+.1e (%s)" % (rb[c2]["diff"], fmt(rb[c2]["rel"], "%.1e"))) if c2 in rb else "-",
            fmt(rc.get("E_dust_shells_diff"), "%+.1e"), ("%.1e, %.1e" % tuple(rc["applied_fields_max_dev"])) if rc.get("applied_fields_max_dev") else "-", r["seconds"]))
    L += ["", "## D. Continuity between slices: the central difference of N beyond the cut against the current F at the middle slice (a coarse-clock check on slices 2 r_s apart, not an identity)", "",
          "| T/r_s | N beyond %s | dN/dT (central difference, per unit lattice time) | F through %s | ratio dN/dT : F | N beyond %s | dN/dT | F through %s | ratio |" % (c1, c1, c2, c2), "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        cells = ["%.3f" % r["T_over_rs"]]
        for c in (c1, c2):
            Nc = r["weighted"][c]["N"]; F = r["weighted"][c]["F"]
            if 0 < i < len(rows) - 1:
                dN = (rows[i + 1]["weighted"][c]["N"] - rows[i - 1]["weighted"][c]["N"]) / (rows[i + 1]["T"] - rows[i - 1]["T"])
                cells += ["%.6e" % Nc, "%+.6e" % dN, "%+.6e" % F, fmt((dN / F) if F else None, "%.3f")]
            else:
                cells += ["%.6e" % Nc, "-", "%+.6e" % F, "-"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## E. The dust's number beyond R and its current through R (weighted), R/r_s = %s" % ", ".join("%g" % c for c in prof_rs), "",
          "| T/r_s | " + " | ".join("N beyond %g; F" % c for c in prof_rs) + " |", "|---|" + "---|" * len(prof_rs)]
    for r in rows:
        L.append("| %.3f | " % r["T_over_rs"] + " | ".join("%.4e; %+.3e" % (r["profile_N_weighted"][str(c)], r["profile_F_weighted"][str(c)]) for c in prof_rs) + " |")
    bands = rows[0]["weighted"][c1]["bands_m"]
    L += ["", "## F. The dust's occupation of the exterior Killing levels beyond %s r_s by omega/m band (weighted): n; E" % c1, "",
          "| T/r_s | " + " | ".join("[%g, %g)" % (b_["lo"], b_["hi"]) for b_ in bands) + " |", "|---|" + "---|" * len(bands)]
    for r in rows:
        L.append("| %.3f | " % r["T_over_rs"] + " | ".join("%.4e; %.4e" % (b_["n"], b_["E"]) for b_ in r["weighted"][c1]["bands_m"]) + " |")
    L += ["", "## G. Per channel (this route; the first route's per-channel numbers beside where its D v2 jsons are given)", "",
          "| T/r_s | kappa | w | dust columns | N beyond %s | E_sym | particles; E | E/particle (m) | holes | F through %s | N beyond %s | F through %s | first route beyond %s: dust_weight; particles_dust; E_particles_dust; E_out_dust | first route F through 16 |" % (c1, c1, c2, c2, c1),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fd = first["dv2"].get(round(r["T_over_rs"], 6)) or {}; ft = first["tail"].get(round(r["T_over_rs"], 6)) or {}
        for ch in r["channels"]:
            if not ch["n_dust"]:
                continue
            a, b = ch["beyond"][c1], ch["beyond"][c2]
            regs = (fd.get(ch["kappa"]) or {}).get("regions") or {}
            fr = regs.get(c1.rstrip("0").rstrip(".")) or regs.get(c1) or {}
            fF = (ft.get("F_dust_16_per_channel") or {}).get(str(ch["kappa"]))
            L.append("| %.3f | %d | %g | %d | %.6e | %.6e | %.6e; %.6e | %s | %.4e | %+.6e | %.6e | %+.6e | %s | %s |" % (
                r["T_over_rs"], ch["kappa"], ch["w"], ch["n_dust"], a["N"], a["E_sym"], a["particles"], a["E_particles"], fmt(a["E_per_particle_over_m"], "%.4f"), a["holes"], a["F"], b["N"], b["F"],
                ("%.6e; %.6e; %.6e; %.6e" % (fr["dust_weight"], fr["particles_dust"], fr["E_particles_dust"], fr["E_out_dust"])) if fr else "-", fmt(fF, "%+.6e")))
    if any((r.get("run_checks") or {}).get("E_G") for r in rows):
        L += ["", "## H. The run's records at the slices (from the run json and fields npz)", "",
              "| T/r_s | E_G | 2gE | T_H | m/T_H | E_dust_shells (stored) | this route's bare energy total | E_pairs_shells | R_t | min(N - abs v) | labels |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            rc = r.get("run_checks") or {}
            L.append("| %.3f | %s | %s | %s | %s | %s | %.6f | %s | %s | %s | %s; %s |" % (
                r["T_over_rs"], fmt(rc.get("E_G"), "%.6f"), fmt(rc.get("two_gE"), "%.4f"), fmt(rc.get("T_H"), "%.6e"), fmt(rc.get("m_over_TH"), "%.3f"), fmt(rc.get("E_dust_shells"), "%.6f"),
                r["dust_bare_energy_total_weighted"], fmt(rc.get("E_pairs_shells"), "%.6f"), fmt(rc.get("R_t"), "%.4f"), fmt(rc.get("min_N_minus_absv"), "%.4f"), rc.get("continuation_label"), rc.get("record_label")))
    if any("pairs_energy" in r["weighted"][c1] for r in rows):
        L += ["", "## I. The pairs (--pairs): the sea-subtracted Killing energy beyond the cuts, Re Tr[Pi_Y h_m (C_sea - P_-)], weighted", "", "| T/r_s | beyond %s | beyond %s | max vacuum density - 1 |" % (c1, c2), "|---|---|---|---|"]
        for r in rows:
            L.append("| %.3f | %s | %s | %s |" % (r["T_over_rs"], fmt(r["weighted"][c1].get("pairs_energy")), fmt(r["weighted"][c2].get("pairs_energy")), fmt(r["checks"].get("max_vacuum_density_dev"), "%.1e")))
    L += ["", "## Provenance", ""] + ["- %s: %s" % (k, v) for k, v in prov.items()]
    return "\n".join(L) + "\n"
