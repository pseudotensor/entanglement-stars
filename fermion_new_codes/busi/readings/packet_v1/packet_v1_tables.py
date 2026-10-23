#!/usr/bin/env python
"""packet_v1_tables.py (busi lane, reviewer A slot, inbox #354): the markdown tables of the packet completeness reading from the stored outputs
(packet_v1_summary.json and packet_v1_series.npz of packet_v1_propagate.py, packet_v1_bounds.json of packet_v1_bounds.py, the covariance jsons
of packet_v1_covariance.py); nothing is recomputed here beyond sums of stored arrays.
Usage: python packet_v1_tables.py --summary S.json --series S.npz --bounds B.json --cov DIR --out TABLES.md
"""
import argparse
import glob
import json
import os

import numpy as np


def f(x, p=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (abs(x) < 1e-3 or abs(x) >= 1e4) and x != 0:
        return "%.1e" % x
    return ("%%.%df" % p) % x if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True); ap.add_argument("--series", required=True); ap.add_argument("--bounds", required=True)
    ap.add_argument("--cov", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary)); Z = np.load(a.series, allow_pickle=True); B = json.load(open(a.bounds))
    meta = S["meta"]
    L = []
    L.append("## Tables of the packet reading (packet_v1_tables.py; inputs: fields %s, json %s)" % (meta["fields_md5"][:8], meta["json_md5"][:8]))
    L.append("")
    L.append("Window: records %d to %d (T/r_s %.3f to %.3f), tau %g, nsub %d (rule I), packet width sigma_p %g a, device for the dense eigendecomposition %s." % (
        meta["rec0"], meta["rec1"], meta["T_over_rs"][0], meta["T_over_rs"][1], meta["tau"], meta["nsub"], meta["sigma_p"], meta["device"]))
    L.append("Formation zeros %s. Over the window max |v|/N = %.4f at T/r_s %.3f, r = %.1f a; min N_applied = %.4f." % (
        meta["formation_zeros"], meta["vN_max_over_window"]["max"], meta["vN_max_over_window"]["T_over_rs_at_max"], meta["vN_max_over_window"]["r_at_max"], meta["vN_max_over_window"]["min_N_applied_over_window"]))
    ec = meta["engine_check"]
    L.append("Engine check (own chain against engine_vacflow_v1.chain_sparse, md5 %s): max |dh0| %.1e, max |dQ| %.1e over the channels run." % (
        ec["md5"][:8], max(v["dh0"] for k, v in ec.items() if k not in ("module", "md5")), max(v["dQ"] for k, v in ec.items() if k not in ("module", "md5"))))
    L.append("")
    for kap, ch in S["channels"].items():
        b = ch["band_top_free"]
        L.append("### Channel kappa = %s (weight %g): ||h0|| = %.4f, ||Q|| = %.4f, radial band top b = sqrt(m^2 + 1) = %.4f; positive band %d levels, lowest %.4f" % (
            kap, ch["weight"], ch["norm_h0"], ch["norm_Q"], b, ch["positive_band"]["n_positive"], ch["positive_band"]["lowest_positive"]))
        L.append("")
        L.append("Per packet, rule E (the engine's frozen step); P+ = the share of the launched Gaussian kept by the positive-band projection; v_g and q' = the free group velocity and the flow coupling cos 2k0 at k0; thr = the lattice trapping threshold |v|/N = v_g/q' (none: untrappable); E0 = <h0>, Em = <h_m>; shares at the window end; 'in' = inside the inner root; ray = the classical null ray of the same launch and direction (log10 eps and the affine length at the window end, and its affine growth over the last 5 r_s).")
        L.append("")
        L.append("| launch | r0 | k0 | v_g | q' | thr | P+ | norm dev | E0 launch | E0 max | E0 max/b | max abs Em/||h_m|| | <r> launch -> end (min, max) | spread end | share in / shell / out at end | share core max | mom top max | N_min along | clock end (rate last 5 r_s) | ray: r end, log10 eps end, lambda end (growth last 5 r_s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in ch["packets"]:
            E = p["E"]; d = p["dispersion"]
            ray = p.get("ray")
            rays = ("%.1f, %.1f, %.2f (%.1e)" % (ray["r_end"], ray["ln_eps_end"] / np.log(10.0), ray["lambda_end"], ray["lambda_growth_last5rs"]) + (" escaped" if ray["escaped"] else "")) if ray else "-"
            L.append("| %s | %.1f | %+.2f | %.2f | %+.2f | %s | %.3f | %.1e | %.3f | %.3f | %.3f | %.3f | %.1f -> %.1f (%.1f, %.1f) | %.1f | %s / %s / %s | %.2f | %.3f | %.3f | %.1f (%.3f) | %s |" % (
                p["launch"], p["r0"], p["k0"], d["group_velocity"], d["flow_coupling"], (f(d["trap_threshold_v_over_N"], 2) if d["trap_threshold_v_over_N"] is not None else "none"),
                p["projected_fraction"], E["norm_dev_max"], E["E0_launch"], E["E0_max"], E["E0_max_over_b"], E["absEm_over_normHm_max"],
                E["rmean_launch"], E["rmean_end"], E["rmean_min"], E["rmean_max"], E["rspread_end"],
                f(E["share_in_end"], 2), f(E["share_shell_end"], 2), f(E["share_out_end"], 2), E["share_core_max"], E["mom_top_max"], E["Nm_min"], E["clock_end"], E["clock_rate_last5rs"], rays))
        L.append("")
        L.append("Rule I (interpolated fields, %d sub-steps per record) against rule E, per packet: max |<r>_E - <r>_I| over the window, max |E0_E - E0_I|, and |<psi_E|psi_I>|^2 at the window end." % meta["nsub"])
        L.append("")
        L.append("| launch | k0 | max d<r> | max dE0 | overlap end | I: E0 max/b | I: <r> end | I: share in end |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in ch["packets"]:
            q = p["E_vs_I"]; I = p["I"]
            L.append("| %s | %+.2f | %.2f | %.4f | %.4f | %.3f | %.1f | %s |" % (p["launch"], p["k0"], q["rmean_max_abs_diff"], q["E0_max_abs_diff"], q["overlap_end"], I["E0_max_over_b"], I["rmean_end"], f(I["share_in_end"], 2)))
        L.append("")
    # the per-record series of a few packets of channel 1: <r>, E0, share in, with the ray
    kap = list(S["channels"].keys())[0]
    labels = Z["launch_label_k%s" % kap]; k0s = Z["launch_k0_k%s" % kap]; x = Z["T_over_rs"]
    pick = [i for i in range(len(k0s)) if (labels[i] in ("r_in", "r_out", "centre", "mid")) and abs(k0s[i]) in (0.3, 0.45) and k0s[i] != 0]
    cols = [0, 7, 14, 23, 32, 47, 63, 95, 127, 159, 191, 223]
    cols = [c for c in cols if c < len(x)]
    L.append("### Channel %s time series (rule E) at selected records: <r> / E0 / share inside the inner root, and the ray's r / log10 eps" % kap)
    L.append("")
    L.append("| launch, k0 | " + " | ".join("T/r_s %.2f" % x[c] for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for i in pick:
        sfx = "out" if k0s[i] > 0 else "in"
        rm = Z["E_k%s_rmean" % kap][i]; e0 = Z["E_k%s_E0" % kap][i]; si = Z["E_k%s_share_in" % kap][i]
        L.append("| %s %+.2f packet | " % (labels[i], k0s[i]) + " | ".join("%.1f / %.3f / %s" % (rm[c], e0[c], f(float(si[c]), 2) if not np.isnan(si[c]) else "open") for c in cols) + " |")
        key = "ray_k%s_%s_%s_r" % (kap, labels[i], sfx)
        if key in Z.files:
            rr = Z[key]; le = Z["ray_k%s_%s_%s_lneps" % (kap, labels[i], sfx)] / np.log(10.0)
            L.append("| %s %s ray | " % (labels[i], sfx) + " | ".join("%.1f / %.1f" % (rr[c], le[c]) for c in cols) + " |")
    L.append("")
    L.append("| roots and fields | " + " | ".join("T/r_s %.2f" % x[c] for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    L.append("| r_in / r_out | " + " | ".join("%s / %s" % (f(float(Z["r_in"][c]), 1) if not np.isnan(Z["r_in"][c]) else "open", f(float(Z["r_out"][c]), 1) if not np.isnan(Z["r_out"][c]) else "open") for c in cols) + " |")
    L.append("| N_min / max abs v/N | " + " | ".join("%.3f / %.3f" % (Z["N_min"][c], Z["vN_max"][c]) for c in cols) + " |")
    L.append("| max ||h_m|| kappa %s | " % kap + " | ".join("%.3f" % Z["normHm_k%s" % kap][c] for c in cols) + " |")
    L.append("")
    # bounds
    sb = B["source_bound"]; nb = B["norm_bound"]
    L.append("### The bounds on every record (packet_v1_bounds.py; fields %s, spectra %s)" % (B["meta"]["fields_md5"][:8], (B["meta"]["spectra_md5"] or "-")[:8]))
    L.append("")
    L.append("- cubic constant at W = 0: 4 sqrt(m^2 + 3/2) = %.4f / a^4; the chain's own constant over the cubic one at W = 0: %s (the chain's constant is the larger for r < %s a, the cubic one beyond)." % (
        sb["cubic_constant_at_W0"], ", ".join("%s: %.3g" % (k, v) for k, v in sb["chain_constant_over_cubic_at_W0"].items() if k != "crossing_radius"), sb["chain_constant_over_cubic_at_W0"]["crossing_radius"]))
    L.append("- max over records and sites of |rho + v J| / cubic cap: %.4f (record %d, site R = %d a); with the sign flipped %.4f; sites r >= 4a: %.4f; r >= 12a: %.4f." % (
        sb["max_ratio_cubic_plus"][0], sb["max_ratio_cubic_plus"][1][0], sb["max_ratio_cubic_plus"][1][1] + 1, sb["max_ratio_cubic_minus"][0], sb["max_ratio_cubic_plus_sites_ge_4"][0], sb["max_ratio_cubic_plus_sites_ge_12"][0]))
    L.append("- max |rho + v J| / chain constant: %.4f (record %d, site R = %d a); sign flipped %.4f." % (sb["max_ratio_chain_plus"][0], sb["max_ratio_chain_plus"][1][0], sb["max_ratio_chain_plus"][1][1] + 1, sb["max_ratio_chain_minus"][0]))
    L.append("- max site density |rho| a^4 = %.4f (record %d, R = %d a); max |v J| a^4 = %.4f; kernel-averaged rho_ell max %.4f / a^4 = %.4f of the cap (K_0 weight sums %s)." % (
        sb["max_abs_rho"][0], sb["max_abs_rho"][1][0], sb["max_abs_rho"][1][1] + 1, sb["max_abs_WJ"][0], sb["rho_ell_max_a4"][0], sb["rho_ell_max_over_cubic_W0"], ", ".join("%.3f" % w for w in sb.get("K0_weight_sum_check", []))))
    pr = sb["per_record_max_ratio_cubic_plus"]; pm = sb["per_record_max_abs_rho_a4"]; pc = sb["per_record_max_ratio_chain_plus"]
    sel = [0, 80, 94, 97, 104, 106, 108, 120, 160, 240, 320]
    L.append("")
    L.append("| T/r_s | " + " | ".join("%.3f" % (k * 0.125) for k in sel) + " |")
    L.append("|---|" + "---|" * len(sel))
    L.append("| max_n abs(rho + vJ)/cubic cap | " + " | ".join("%.1e" % pr[k] for k in sel) + " |")
    L.append("| max_n abs(rho + vJ)/chain constant | " + " | ".join("%.1e" % pc[k] for k in sel) + " |")
    L.append("| max_n abs(rho) a^4 | " + " | ".join("%.1e" % pm[k] for k in sel) + " |")
    L.append("| max_kappa ||h_m|| | " + " | ".join("%.3f" % nb["per_record_max_normHm"][k] for k in sel) + " |")
    L.append("| max_kappa bound | " + " | ".join("%.3f" % nb["per_record_max_bound"][k] for k in sel) + " |")
    L.append("| max N_applied / max abs v_applied | " + " | ".join("%.3f / %.3f" % (nb["per_record_Nmax"][k], nb["per_record_absvmax"][k]) for k in sel) + " |")
    L.append("")
    L.append("- ||h_m|| / (max N ||h0|| + max |v| ||Q||) over all records and channels: between %.3f and %.3f; ||h_m|| against the engine's stored eigenvalues on the %d records that hold them: max relative difference %.1e." % (
        nb["min_normHm_over_bound"], nb["max_normHm_over_bound"], nb["stored_spectra_records"], nb["max_rel_diff_normHm_vs_stored"]))
    L.append("- ||h0_kappa|| per channel: %s; ||Q|| = %.4f; radial band top b = %.4f." % (", ".join("%s: %.3f" % (k, v) for k, v in nb["norm_h0"].items()), nb["norm_Q"], nb["band_top_free_radial"]))
    L.append("")
    # covariance jsons
    for fn in sorted(glob.glob(os.path.join(a.cov, "packet_v1_covariance_*.json"))):
        C = json.load(open(fn)); m_ = C["meta"]
        L.append("### The stored covariance at T/r_s %.2f (record %d, %s; slice md5 %s): packet_v1_covariance.py" % (m_["T_over_rs"], m_["record"], m_["dtype"], m_["slice_md5"][:8]))
        L.append("")
        sc = C.get("sources_check")
        if sc:
            L.append("- the stored eps_tot and J_tot of the record recomputed from the slice's orbitals with the own chain: max relative differences %.1e (eps) and %.1e (J); sum of eps %.6f stored, %.6f recomputed." % (
                sc["max_abs_diff_eps_over_max_abs_eps"], sc["max_abs_diff_J_over_max_abs_J"], sc["sum_eps_stored"], sc["sum_eps_mine"]))
        L.append("")
        L.append("| kappa | n_dust (sum f) | h_m R2 - R2 e (rel) | abs(C - P) norm | tr(C - P) (expected) | lambda_min(C - P) | Hoelder max ratio (site) | max site abs(rho + vJ) a^4 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in C["channels"].items():
            L.append("| %s | %d (%.1f) | %.1e | %.6f | %.3f (%.3f) | %.4f | %.4f (R = %.0f) | %.1e |" % (k, v["n_dust"], v["f_sum"], v["operator_check_resid_rel"], v["normD"], v["trD"], v["trD_expected"], v["lam_min"], v["hoelder_max_ratio"], v["site_of_max_ratio"], v["max_lhs_site_density_a4"]))
        L.append("")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
