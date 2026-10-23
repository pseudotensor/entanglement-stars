#!/usr/bin/env python
"""dust_tail_table_v2.py (busi lane, reviewer A): the delivery's tables of the dust's second route, version 2, composed from the outputs
only: <out-dir>/<tag>_all.json written by dust_tail_v2.py (the per-slice npz hold every array and are not needed here) and the first
route's jsons (D_dust_tail_v1.json's weighted rows and the D v2 batch jsons' per-channel regions, read by dust_tail_v1.load_first_route),
whose numbers stand beside this route's and are never averaged with them. The first route's entropy and count keys (S_hull, S_F, E_F, the
Killing-basis particles and holes of the whole deviation, the region excess, the sector's dimension) are looked up under the names in CAND;
what is found is named in the provenance, what is not prints "-". Usage: python dust_tail_table_v2.py --out-dir DIR --tag TAG
[--first-route-tail-json F] [--first-route-dv2-jsons a.json,b.json] [--out FILE]; dust_tail_v2.run_main calls compose_from_outputs."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_tail_v1 as V1  # noqa: E402  (load_first_route, fmt, utc; no slice is read here)

fmt = V1.fmt
utc = V1.utc
LABELS = ("(-inf,-b)", "[-b,-m)", "[-m,m)", "[m,b)", "[b,inf)")
CAND = {"S_hull": ("S_hull", "S_hull_eps1e-09", "S_hull_eps1e-9", "S_hull_1e-9", "S_esc", "S_esc_hull"),
        "S_F": ("S_F", "S_F_escaped", "S_escaped", "S_escaped_sector", "S_F_eps_loc_0.01"),
        "E_F": ("E_F", "E_escaped", "E_escaped_sector"),
        "particles": ("particles", "particles_Pm", "particles_P-", "killing_particles", "particles_total", "n_particles"),
        "holes": ("holes", "holes_Pm", "holes_P-", "killing_holes", "holes_total", "n_holes"),
        "region_S_excess": ("region_S_excess", "S_region_excess", "region_excess", "S_excess", "region_entropy_excess"),
        "escaped_dim": ("escaped_dim", "escaped_sector_dim", "sector_dim", "d_F", "F_dim", "dim_F")}
FOUND = {}


def ck(key):
    return str(key).rstrip("0").rstrip(".")


def pick(dicts, name):
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k in CAND[name]:
            if k in d and d[k] is not None and not isinstance(d[k], (dict, list)):
                try:
                    return float(d[k]), k
                except (TypeError, ValueError):
                    pass
    return None, None


def first_region(fd, kappa, cut_key):
    chd = fd.get(kappa) or fd.get(int(kappa)) or {}
    regs = chd.get("regions") or {}
    return chd, (regs.get(cut_key) or regs.get(cut_key + ".0") or {})


def first_weighted(fd, cut_key, name):
    """sum over the first route's channels of w x the per-channel value of `name` (found under its candidate keys); None if any channel lacks it."""
    if not fd:
        return None
    tot = 0.0
    for kappa in fd:
        chd, reg = first_region(fd, kappa, cut_key)
        v, kname = pick([reg, chd], name)
        if v is None:
            return None
        FOUND[name] = kname
        tot += float(chd.get("w", 16 * int(kappa))) * v
    return tot


def diffs(x, y, f="%+.3e"):
    return fmt((x - y) if (x is not None and y is not None) else None, f)


def compose_v2(J, first, all_json_name="<tag>_all.json"):
    rows = sorted([r for r in J.get("rows", []) if "v2" in r], key=lambda r: r["T_over_rs"])
    prov = J.get("provenance") or {}
    if not rows:
        return "# dust_tail_table_v2: no rows with v2 readouts in the outputs\n"
    cuts_rs = J.get("cuts_rs") or [16.0, 24.0]
    m = float(prov.get("m", rows[0].get("m", 0.2)))
    keys = [str(float(c)) for c in cuts_rs]
    k1 = keys[0]
    v20 = rows[0]["v2"]; b = float(v20["b"]); dev = any(r["v2"].get("deviation") for r in rows); eps = v20.get("eps_loc")
    e0 = next(((r.get("run_checks_v2") or {}) for r in rows if (r.get("run_checks_v2") or {}).get("E_0") is not None), {})
    nominal = rows[0].get("dust_number_nominal")
    L = ["# The shed dust's second route, version 2 (dust_tail_v2.py, busi lane; web40g task 1): the 5-bin split, the counts, the current through 16 r_s and the entropy subsystems, beside the first route", "",
         "Composed %s by dust_tail_table_v2.py from %s (the outputs of dust_tail_v2.py) and the first route's jsons; every number printed by the script; the first route's numbers stand beside this "
         "route's and are never averaged with them. Weighted = summed over the channels with w = 16 kappa (the multiplicity); unweighted = summed over the 10 chains without the copies. The dust = the "
         "stored dust columns with their occupations f; h_m = (1/2){N, h_0} + (1/2){v, Q} at the slice's applied fields; beyond R_c = the sites with R > R_c; m = %g; b = sqrt(m^2 + 1/a^2) = %.6f "
         "(b/m = %.4f); the bins (-inf,-b), [-b,-m), [-m,m), [m,b), [b,inf) with inclusive lower edges; E_0 = %s (%s), E_0/m = %s; eps_loc = %s; s(x) = -x ln x - (1 - x) ln(1 - x), entropies in nats; "
         "energies in units of 1/a and in units of m; the number current in fermions per unit lattice time; the nominal dust number %s = sum 16 kappa n_kappa." % (
             utc(), all_json_name, m, b, b / m, fmt(e0.get("E_0"), "%.6f"), e0.get("E_0_source") or "no source", fmt(e0.get("E_0_over_m"), "%.4f"), fmt(eps, "%g"), fmt(nominal, "%g")), ""]
    # A, B: the 5 bins per cut
    for letter, key in zip("AB", keys):
        L += ["## V2-%s. The dust's number and Killing energy beyond %s r_s in the 5 bins (weighted): per bin n; E/m (E = sum omega n); the [m,b) and [b,inf) weights separately with their shares of N" % (letter, ck(key)), "",
              "| T/r_s | N beyond | " + " | ".join("%s: n; E/m" % lab for lab in LABELS) + " | mean omega/m in [m,b); in [b,inf) | [m,b)/N; [b,inf)/N | at or above m: n; share | partition: max abs(sum n - N); max abs(sum E - E_block) |",
              "|---|---|" + "---|" * 5 + "---|---|---|---|"]
        for r in rows:
            q = r["v2"]["weighted"][key]; bb = q["bins5"]
            L.append("| %.3f | %.6e | %s | %s; %s | %s; %s | %.6e; %s | %.1e; %.1e |" % (
                r["T_over_rs"], q["N"], " | ".join("%.6e; %.6e" % (x["n"], x["E_over_m"]) for x in bb), fmt(bb[3]["mean_omega_over_m"], "%.4f"), fmt(bb[4]["mean_omega_over_m"], "%.4f"),
                fmt(q["frac_m_b"], "%.4e"), fmt(q["frac_b_inf"], "%.4e"), bb[3]["n"] + bb[4]["n"], fmt(q["frac_above_m"], "%.4f"), q["max_bins5_sum_n_minus_N"], q["max_bins5_sum_E_minus_E_block"]))
        L.append("")
    # C: the counts
    L += ["## V2-C. The counts and the energy per particle (weighted): N is a count of fermions (the fraction N/%s beside it); the Killing energy's count is E_sym/m; a fraction of E_0 converts to a count by E_0/m = %s, and the same fraction times %s is printed only as the form the review rejects" % (
        fmt(nominal, "%g"), fmt(e0.get("E_0_over_m"), "%.4f"), fmt(nominal, "%g")), "",
          "| T/r_s | cut r_s | N (fermions) | N/%s | E_sym | E_sym/m (count) | E_sym/E_0 | that fraction x E_0/m | that fraction x %s | particles (positive branch); E_particles/m | E_particles/(particles m) | E_sym/(N m) | first route: dust_weight; particles_dust; E_particles_dust/m; E/particle (m) |" % (fmt(nominal, "%g"), fmt(nominal, "%g")),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ft = first["tail"].get(round(r["T_over_rs"], 6)) or {}
        for key in keys:
            q = r["v2"]["weighted"][key]
            fb = ft.get("beyond_%s" % ck(key)) or {}
            fcell = ("%.6e; %.6e; %.6e; %s" % (fb["dust_weight"], fb["particles_dust"], fb["E_particles_dust"] / m, fmt(fb.get("E_per_particle_over_m"), "%.4f"))) if fb.get("dust_weight") is not None else "-"
            L.append("| %.3f | %s | %.6e | %s | %.6e | %.6e | %s | %s | %s | %.6e; %.6e | %s | %s | %s |" % (
                r["T_over_rs"], ck(key), q["N"], fmt(q["N_frac_nominal"], "%.4e"), q["E_sym"], q["E_sym_over_m"], fmt(q.get("E_sym_frac_E0"), "%.4e"), fmt(q.get("E_sym_count_from_frac_E0"), "%.6e"),
                fmt(q.get("E_sym_frac_E0_times_nominal"), "%.6e"), q["particles"], q["E_particles_over_m"], fmt(q["E_per_particle_over_m"], "%.4f"), fmt(q["E_sym_per_N_over_m"], "%.4f"), fcell))
    # D: the current
    L += ["", "## V2-D. The dust's number current through %s r_s (outward positive; version 1's bond-block form with its sparse-block check) beside the first route; the continuity between slices (the central difference of N, a coarse-clock check on slices 2 r_s apart) and the site continuity identity at the slice" % ck(k1), "",
          "| T/r_s | F (bond blocks) | F (sparse block) | first route F | this - first | dN/dT (central difference) | dN/dT : F | max abs(d rho_x/dt - (F(x) - F(x+1))) | sum_x d rho_x/dt | per channel kappa: F this; first |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        ft = first["tail"].get(round(r["T_over_rs"], 6)) or {}
        F = r["weighted"][k1]["F"]
        Fsp = float(sum(ch["w"] * ch["beyond"][k1]["F_sparse"] for ch in r["channels"] if ch["n_dust"]))
        fF = ft.get("F_dust_16_weighted")
        if 0 < i < len(rows) - 1:
            dN = (rows[i + 1]["weighted"][k1]["N"] - rows[i - 1]["weighted"][k1]["N"]) / (rows[i + 1]["T"] - rows[i - 1]["T"]); dNs = "%+.6e" % dN; rat = fmt((dN / F) if F else None, "%.3f")
        else:
            dNs = "-"; rat = "-"
        per = "; ".join("%d: %+.4e; %s" % (ch["kappa"], ch["beyond"][k1]["F"], fmt((ft.get("F_dust_16_per_channel") or {}).get(str(ch["kappa"])), "%+.4e")) for ch in r["channels"] if ch["n_dust"])
        L.append("| %.3f | %+.6e | %+.6e | %s | %s | %s | %s | %.1e | %+.1e | %s |" % (
            r["T_over_rs"], F, Fsp, fmt(fF, "%+.6e"), diffs(F, fF), dNs, rat, r["v2"]["max_continuity_residual"], r["v2"]["max_abs_drho_total"], per))
    # E: the dust-only subsystem
    L += ["", "## V2-E. The entropy subsystems and the bound d h2(n/d), (i) the dust-only exterior block Pi_Y D f D^dag Pi_Y (its rank is n_kappa, the dust columns per chain; weighted with 16 kappa; the dimension d_Y = 2 x the sites beyond)", "",
          "| T/r_s | cut r_s | N | S_dust (nats) | rank: sum 16k n_k; distinct columns | bound sum 16k n_k h2(N_k/n_k) | S_dust/bound (max over channels) | respected | bound sum 16k d_Y h2(N_k/d_Y) (d_Y) | the weighted count against one dimension: d_w h2(N/d_w) (d_w = sum 16k d_Y over the 10 chains); the review's form d_unw h2(N/d_unw) (d_unw = 10 d_Y) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for key in keys:
            q = r["v2"]["weighted"][key]
            L.append("| %.3f | %s | %.6e | %.6e | %g; %d | %.6e | %s (%s) | %s | %.6e (%d) | %.6e (%g); %.6e (%d) |" % (
                r["T_over_rs"], ck(key), q["N"], q["S_dust"], q["rank_w"], q["rank_unw"], q["bound_rank"], fmt((q["S_dust"] / q["bound_rank"]) if q["bound_rank"] > 0 else None, "%.4f"),
                fmt(q["max_S_dust_over_bound_rank"], "%.4f"), "yes" if q["all_bound_rank_respected"] else "NO", q["bound_dim"], q["d_Y"], q["bound_dim_w_form"], q["d_Y_w_all"], q["bound_review_form"], q["d_Y_unw_all"]))
    # F: the escaped sector
    L += ["", "## V2-F. (ii) The escaped sector F (eps_loc %s; d_F = d_+ + d_- per chain; weighted with 16 kappa): the whole state C = S S^dag + D f D^dag in F, its counts split into the dust's and the pairs', S_F and its bound; beside it the first route's S_hull, S_F, Killing-basis particles and holes where its jsons carry them%s" % (
        fmt(eps, "%g"), "" if dev else " (not computed on this run: --deviation was off)"), ""]
    if dev:
        L += ["| T/r_s | cut r_s | d_F: unweighted; weighted | n_+ = Tr F_+^dag C F_+: total (dust; pairs) | n_- = Tr F_-^dag (1 - C) F_-: total (pairs; the dust's occupation of F_-, subtracted) | n_F = n_+ + n_-; the pairs' share | S_F (nats) | bound sum 16k d_F h2(n_F,k/d_F,k) | S_F/bound (max over channels) | respected | the dust in F: its count; S_F,dust; its rank bound; what its count alone permits in d_F: per-channel sum; d_w form; d_unw form | E_F | region: S(C_YY) - S((P_-)_YY) | Killing block of C - P_-: particles; holes (of which the dust: p; h) | first route: S_hull (this S_F - S_hull); S_F; particles; holes; dust_weight |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            fd = first["dv2"].get(round(r["T_over_rs"], 6)) or {}; ft = first["tail"].get(round(r["T_over_rs"], 6)) or {}
            for key in keys:
                q = r["v2"]["weighted"][key]; d = q["dev"]; ckk = ck(key)
                fsh = first_weighted(fd, ckk, "S_hull"); fsf = first_weighted(fd, ckk, "S_F"); fpa = first_weighted(fd, ckk, "particles"); fho = first_weighted(fd, ckk, "holes")
                fdw = (ft.get("beyond_%s" % ckk) or {}).get("dust_weight")
                fcell = "%s (%s); %s; %s; %s; %s" % (fmt(fsh), diffs(d["S_F"], fsh), fmt(fsf), fmt(fpa), fmt(fho), fmt(fdw))
                L.append("| %.3f | %s | %d; %g | %.6e (%.6e; %.6e) | %.6e (%.6e; %.6e) | %.6e; %s | %.6e | %.6e | %s (%s) | %s | %.6e; %.6e; %.6e; %.6e; %.6e; %.6e | %+.6e | %+.6e | %.6e; %.6e (%.6e; %.6e) | %s |" % (
                    r["T_over_rs"], ckk, d["d_F_unw"], d["d_F_w"], d["n_plus"], d["n_plus_dust"], d["n_plus_pairs"], d["n_minus"], d["n_minus_pairs"], d["n_minus_dust_occ"], d["n_F"], fmt(d["pairs_share_of_n_F"], "%.4f"),
                    d["S_F"], d["bound_F"], fmt((d["S_F"] / d["bound_F"]) if d["bound_F"] > 0 else None, "%.4f"), fmt(d["max_S_F_over_bound_F"], "%.4f"), "yes" if d["all_bound_F_respected"] else "NO",
                    d["w_F_dust"], d["S_F_dust"], d["bound_F_dust_rank"], d["bound_F_dustcount"], d["bound_F_dustcount_w_form"], d["bound_F_dustcount_review_form"], d["E_F"], d["region_excess"],
                    d["particles_dev"], d["holes_dev"], d["particles_dust_block"], d["holes_dust_block"], fcell))
        L += ["", "The tighter block form sum 16k [d_+ h2(n_+/d_+) + d_- h2(n_-/d_-)] and the one-dimension forms d_w h2(n_F/d_w), d_unw h2(n_F/d_unw) are in the json (dev.bound_F_blocks, bound_F_w_form, bound_F_review_form)."]
    # G: per channel
    L += ["", "## V2-G. Per channel (kappa, w = 16 kappa): the [m,b) and [b,inf) weights, S_dust with its rank bound%s; the first route's per-channel numbers beside where its D v2 jsons carry them" % (
        ", the escaped sector's counts, S_F and bound, the region excess" if dev else ""), "",
          "| T/r_s | kappa | w | dust columns | cut r_s | N | [m,b): n; E/m | [b,inf): n; E/m | S_dust | rank bound n_k h2(N/n_k) | " + ("d_F | n_+ (dust; pairs) | n_- | S_F | bound d_F h2(n_F/d_F) | region excess | Killing block p; h | " if dev else "") +
          "first route: dust_weight; S_hull; particles; holes |", "|---|---|---|---|---|---|---|---|---|---|" + ("---|" * 7 if dev else "") + "---|"]
    for r in rows:
        fd = first["dv2"].get(round(r["T_over_rs"], 6)) or {}
        for c2 in r["v2"]["channels"]:
            if not c2["n_dust"] and not dev:
                continue
            for key in keys:
                ckk = ck(key); reg = c2["beyond"].get(key); d = c2["dev"].get(key)
                chd, fr = first_region(fd, c2["kappa"], ckk)
                fsh, _ = pick([fr, chd], "S_hull"); fpa, _ = pick([fr, chd], "particles"); fho, _ = pick([fr, chd], "holes")
                cells = ["%.3f" % r["T_over_rs"], "%d" % c2["kappa"], "%g" % c2["w"], "%d" % c2["n_dust"], ckk]
                if reg:
                    cells += ["%.6e" % reg["N"], "%.6e; %.6e" % (reg["bins5"][3]["n"], reg["bins5"][3]["E_over_m"]), "%.6e; %.6e" % (reg["bins5"][4]["n"], reg["bins5"][4]["E_over_m"]), "%.6e" % reg["S_dust"], "%.6e" % reg["bound_rank"]]
                else:
                    cells += ["-", "-", "-", "-", "-"]
                if dev:
                    if d:
                        cells += ["%d" % d["d_F"], "%.6e (%.6e; %.6e)" % (d["n_plus"], d["n_plus_dust"], d["n_plus_pairs"]), "%.6e" % d["n_minus"], "%.6e" % d["S_F"], "%.6e" % d["bound_F"], "%+.6e" % d["region_excess"], "%.6e; %.6e" % (d["particles_dev"], d["holes_dev"])]
                    else:
                        cells += ["-"] * 7
                cells.append("%s; %s; %s; %s" % (fmt(fr.get("dust_weight")), fmt(fsh), fmt(fpa), fmt(fho)))
                L.append("| " + " | ".join(cells) + " |")
    # H: the deviation's bins
    L += ["", "## V2-H. The 5-bin split of the whole deviation (C - P_-)_YY in the exterior block's Killing basis (weighted; signed occupations): the deviation; of which the pairs; the dust (per bin n; E/m)%s" % ("" if dev else " (not computed: --deviation was off)"), ""]
    if dev:
        L += ["| T/r_s | cut r_s | part | " + " | ".join(LABELS) + " | sum n | sum E/m |", "|---|---|---|" + "---|" * 7]
        for r in rows:
            for key in keys:
                d = r["v2"]["weighted"][key]["dev"]
                for name, lab in (("bins5_dev", "deviation (P_- reference)"), ("bins5_pairs", "pairs"), ("bins5_dust_block", "dust")):
                    L.append("| %.3f | %s | %s | %s | %+.6e | %+.6e |" % (r["T_over_rs"], ck(key), lab, " | ".join("%+.6e; %+.6e" % (x["n"], x["E_over_m"]) for x in d[name]),
                                                                        sum(x["n"] for x in d[name]), sum(x["E_over_m"] for x in d[name])))
    # I: checks
    L += ["", "## V2-I. Checks per slice (maxima over channels and cuts)", "",
          "| T/r_s | 5-bin partition: abs(sum n - N); abs(sum E - E_block) | nu sum - N | S_dust <= rank bound (all) | site continuity residual; sum d rho/dt | " +
          ("F: F_+^dag P_- F_+ max; F_-^dag P_- F_- - 1 max; n_C - n_P - N max; sum n_dev - (n_C - n_P) max; S_F <= bound (all); vacuum density - 1 | " if dev else "") +
          "version 1: sea density - 1; spectrum vs stored e_k; total dust - nominal; K_density - stored K_dust beyond %s | seconds (v1 pass + v2 pass) |" % ck(k1),
          "|---|---|---|---|---|" + ("---|" if dev else "") + "---|---|"]
    for r in rows:
        qs = [r["v2"]["weighted"][k] for k in keys]; ckv = r["checks"]; rb = (r.get("run_checks") or {}).get("beyond") or {}
        cells = ["%.3f" % r["T_over_rs"], "%.1e; %.1e" % (max(q["max_bins5_sum_n_minus_N"] for q in qs), max(q["max_bins5_sum_E_minus_E_block"] for q in qs)), "%.1e" % max(q["max_nu_sum_minus_N"] for q in qs),
                 "yes" if all(q["all_bound_rank_respected"] for q in qs) else "NO", "%.1e; %+.1e" % (r["v2"]["max_continuity_residual"], r["v2"]["max_abs_drho_total"])]
        if dev:
            ds = [q["dev"] for q in qs]
            cells.append("%.1e; %.1e; %.1e; %.1e; %s; %s" % (max(d["max_vacuum_plus_overlap"] for d in ds), max(d["max_vacuum_minus_identity_dev"] for d in ds), max(d["max_n_C_minus_n_P_minus_N"] for d in ds),
                                                             max(d["max_sum_ndev_minus_excess"] for d in ds), "yes" if all(d["all_bound_F_respected"] for d in ds) else "NO", fmt(ckv.get("max_vacuum_density_dev"), "%.1e")))
        cells.append("%.1e; %s; %+.1e; %s" % (ckv["max_sea_density_dev"], fmt(ckv.get("max_spectrum_vs_stored"), "%.1e"), ckv["total_number_minus_nominal"], ("%+.1e" % rb[k1]["diff"]) if k1 in rb else "-"))
        cells.append("%.0f + %.0f" % (r["seconds"], r["v2"]["seconds"]))
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Provenance", ""] + ["- %s: %s" % (k, v) for k, v in prov.items()]
    L.append("- first route's per-channel keys used for the entropy and count columns (as found in its D v2 jsons): %s" % (", ".join("%s <- %s" % (k, v) for k, v in sorted(FOUND.items())) if FOUND else "none found (the columns print -)"))
    fd0 = next((first["dv2"][t] for t in sorted(first["dv2"])), None)
    if fd0:
        chd0 = next(iter(fd0.values()))
        reg0 = (chd0.get("regions") or {}).get(ck(k1)) or {}
        L.append("- first route's per-channel region keys present (first slice, first channel, region %s): %s" % (ck(k1), ", ".join(sorted(reg0.keys())) if reg0 else "none"))
    return "\n".join(L) + "\n"


def compose_from_outputs(out_dir, tag, tail_json, dv2_jsons, out_path=None):
    FOUND.clear()
    J = json.load(open(os.path.join(out_dir, "%s_all.json" % tag)))
    first = V1.load_first_route(tail_json, dv2_jsons)
    md = compose_v2(J, first, all_json_name="%s_all.json" % tag)
    out_path = out_path or os.path.join(out_dir, "%s_table_v2.md" % tag)
    open(out_path, "w").write(md)
    return out_path, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--first-route-tail-json", default=""); ap.add_argument("--first-route-dv2-jsons", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out_path, _ = compose_from_outputs(a.out_dir, a.tag, a.first_route_tail_json, a.first_route_dv2_jsons, a.out or None)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
