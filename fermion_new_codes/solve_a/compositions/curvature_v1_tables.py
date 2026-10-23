"""curvature_v1_tables.py: markdown tables for CURVATURE_readout_v1 from the per-member json and npz written by curvature_v1_data.py.
Usage: python curvature_v1_tables.py [outdir]   (prints to stdout)."""
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "<internal>/curvature_v1"
NAMES = ["B2", "P64_s4_c19", "P64_s4", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2"]
S = {n: json.load(open(os.path.join(D, "curvature_v1_%s.json" % n))) for n in NAMES}
Z = {n: np.load(os.path.join(D, "curvature_v1_%s.npz" % n)) for n in NAMES}
allj = [f for f in os.listdir(D) if f.startswith("curvature_v1_all_")][0]
ALL = json.load(open(os.path.join(D, allj)))


def g(x, f="%.3g"):
    return "-" if x is None else (f % x)


def pk(d, f="%.3g"):
    return "-" if d is None else ("%s at %.3f" % (f % d["value"], d["T_over_rs"]))


print("## inputs\n\n| member | file | md5 |\n|---|---|---|")
for n in NAMES:
    for k, v in ALL["inputs"][n].items():
        print("| %s | %s | %s |" % (n, v["path"], v["md5"]))
print("\nscript md5 %s\n" % ALL["script_md5"])

print("## members, gate and formation\n\n| member | sigma/a | tau | records (T/r_s) | converged | certified extent (v1.2) | gate sigma max|v'| > 0.1, barred / unbarred | strain gate sigma max(|K_r|,|K_t|) > 0.1 | formation (zeros) |\n|---|---|---|---|---|---|---|---|---|")
for n in NAMES:
    s = S[n]
    gt = s["gate"]
    print("| %s | %g | %g | %d (%g to %g) | %d | %s | %s / %s | %s | %s (%s) |" % (
        n, s["sigma"], s["tau"], s["records"], s["T_over_rs_range"][0], s["T_over_rs_range"][1], s["n_converged"],
        "not listed" if s["cert_end"] is None else "%g" % s["cert_end"],
        g(gt["onet_barred"] and gt["onet_barred"]["T_over_rs"], "%g"), g(gt["onet_unbarred"] and gt["onet_unbarred"]["T_over_rs"], "%g"),
        g(gt["strain_unbarred"] and gt["strain_unbarred"]["T_over_rs"], "%g"),
        g(s["formation"]["T_over_rs"], "%g"), ", ".join("%.2f" % z for z in s["formation"]["zeros"])))

print("\n## (a) spheres' curvature K_t^2 = 2GM/r^3 = (v/(rN))^2, max over r (unbarred unless labelled), units 1/a^2\n")
print("| member | cap | gate scale | max all records (at T/r_s; r) | pre-gate max | post-gate max | max on certified records | at formation (r) | nodes only | barred sites | centre identity (8/3) pi g rho_ell(0), max | spline centre vs identity, max rel | max / cap | max / gate scale | records above cap | records above gate scale |")
print("|" + "---|" * 16)
for n in NAMES:
    s = S[n]
    a = s["Kt2_max_all"]
    rmax = float(Z[n]["Kt2_rmax"][a["k"]])
    print("| %s | %.3f | %.3g | %.4g at %.3f; r %.2f | %s | %s | %s | %.4g (%.2f) | %s | %s | %s | %.2g | %.4f | %.0f | %d | %d of %d |" % (
        n, s["cap"], s["gate_scale"], a["value"], a["T_over_rs"], rmax, pk(s["Kt2_max_pre"], "%.4g"), pk(s["Kt2_max_post"], "%.4g"),
        pk(s["Kt2_max_certified"], "%.4g"), s["at_formation"]["Kt2_max"], s["at_formation"]["r"], pk(s["Kt2_nodes_max_all"], "%.4g"),
        pk(s["Kt2_bar_max_all"], "%.4g"), pk(s["Kt2_centre_series_max"], "%.4g"), s["centre_spline_vs_series_rel_max"], s["frac_cap_max"],
        s["over_gate_scale_max"], s["records_Kt2_above_cap"], s["records_Kt2_above_gate_scale"], s["records"]))

print("\n### time series of max_r K_t^2 (unbarred), r of the max, and centre value; selected records\n")
for n in ["B2", "P64_s4", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2"]:
    z = Z[n]
    Trs = z["Trs"]
    marks = [0, 5, 10, 11, 11.5, 11.75, 12, 12.125, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 14, 15, 17, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100]
    print("%s (cap %.2f, gate scale %.3g):\n" % (n, float(z["cap"]), float(z["gate_scale"])))
    print("| T/r_s | max K_t^2 | r of max | K_t^2 at r -> 0 | max / cap | frame norm max (cadence 1) | r of norm max | certified |\n|---|---|---|---|---|---|---|---|")
    for m in marks:
        j = int(np.argmin(np.abs(Trs - m)))
        if abs(Trs[j] - m) > 1e-6:
            continue
        print("| %g | %.4g | %.2f | %.4g | %.2e | %s | %s | %s |" % (m, z["Kt2_max"][j], z["Kt2_rmax"][j], z["Kt2_centre"][j], z["Kt2_max"][j] / float(z["cap"]),
                                                             g(z["norm_max_c1"][j], "%.4g"), g(z["norm_rmax_c1"][j], "%.2f"), "yes" if z["certified"][j] else "no"))
    print()

print("## trapping-zero episodes (records holding a zero; from the run json) and gauge-intact pre-gate maxima\n")
print("| member | episodes (first to last record with a zero, T/r_s) | non-intact gauge records | pre-gate max K_t^2 on intact records | pre-gate max frame norm on intact records (T/r_s) |\n|---|---|---|---|---|")
for n in NAMES:
    z = Z[n]; Trs = z["Trs"]; has = z["nzeros"] > 0
    eps_ = []
    k = 0
    while k < len(has):
        if has[k]:
            j = k
            while j + 1 < len(has) and has[j + 1]:
                j += 1
            eps_.append("%g to %g" % (Trs[k], Trs[j])); k = j + 1
        else:
            k += 1
    kg = int(z["kgate"]); pre = np.arange(len(Trs)) < kg; ok = pre & z["gauge_intact"]
    jn = int(np.nanargmax(np.where(ok, z["norm_max_c1"], -np.inf)))
    print("| %s | %s | %d | %.4g | %.4g (%g) |" % (n, "; ".join(eps_), int((~z["gauge_intact"]).sum()), np.max(z["Kt2_max"][ok]), z["norm_max_c1"][jn], Trs[jn]))
print()

print("## reviewer B's slice-frame orthonormal Riemann norm (sum_abcd R_abcd^2)^(1/2), unbarred, max over r <= 256 a; pre-gate / post-gate (gate = barred one-term test)\n")
print("| member | cadence | pre-gate max (T/r_s; r) | post-gate max (T/r_s; r) | |Kretschmann| pre | |Kretschmann| post | components at the post max: R0101, R0202, R0212, R1212, R2323 |\n|---|---|---|---|---|---|---|")
for n in NAMES:
    for c in ("1", "2", "4"):
        q = S[n]["norm"][c]
        comp = q.get("components_at_post_max") or {}
        print("| %s | %s | %s; %s | %s; %s | %s | %s | %s |" % (n, c, pk(q["pre"], "%.4g"), g(q.get("r_at_pre_max"), "%.2f"), pk(q["post"], "%.4g"),
                                                           g(q.get("r_at_post_max"), "%.2f"), pk(q["kret_pre"], "%.3g"), pk(q["kret_post"], "%.3g"),
                                                           ", ".join("%.3g" % comp[k] for k in ("R0101", "R0202", "R0212", "R1212", "R2323")) if comp else "-"))

print("\n## stencil variants of the frame norm (cadence 1): pre-gate max / post-gate max\n\n| member | quintic spline on the fine grid | spline at the nodes | FD2 at the nodes | FD4 at the nodes | barred site fields, FD2 |\n|---|---|---|---|---|---|")
for n in NAMES:
    st = S[n]["stencil"]; q = S[n]["norm"]["1"]
    print("| %s | %s / %s | %s | %s | %s | %s |" % (n, g(q["pre"]["value"], "%.4g"), g(q["post"]["value"], "%.4g"),
                                                *["%s / %s" % (g(st[k]["pre"]["value"], "%.4g"), g(st[k]["post"]["value"], "%.4g")) for k in ("spline@nodes", "FD2@nodes", "FD4@nodes", "barred_sites_FD2")]))

print("\n## reviewer A's second route: Riemann from (M, rho_ell, J_ell, S_r_ell, S_t_ell) by the Einstein identities, against the (N, v) route at the nodes\n")
print("| member | r range | T_nr = s J_ell, s | records | constraint rows rel. max pre (G_nn, G_nr) | post | resid/norm pre: R0101, R0202, R0212, R1212 | post | norm from sources pre / post | norm from (N, v) at nodes pre / post |\n|---|---|---|---|---|---|---|---|---|---|")
for n in NAMES:
    f = S[n].get("reviewa_route")
    if f is None:
        print("| %s | no stresses stored (no handoff npz) | | | | | | | | |" % n)
        continue
    rp, ro = f["resid_over_norm_pre"], f["resid_over_norm_post"]
    print("| %s | %g to %g | %+g | %d | %s, %s | %s, %s | %s | %s | %s / %s | %s / %s |" % (
        n, f["r_range"][0], f["r_range"][1], f["sign_T01_vs_Jell"], f["records"], g(f["constraint_rel_max_pre"][0]), g(f["constraint_rel_max_pre"][1]),
        g(f["constraint_rel_max_post"][0]), g(f["constraint_rel_max_post"][1]), ", ".join(g(rp[k]) for k in ("R0101", "R0202", "R0212", "R1212")),
        ", ".join(g(ro[k]) for k in ("R0101", "R0202", "R0212", "R1212")), g(f["norm_src_pre"], "%.4g"), g(f["norm_src_post"], "%.4g"),
        g(f["norm_Nv_nodes_pre"], "%.4g"), g(f["norm_Nv_nodes_post"], "%.4g")))

print("\nmedian over records of resid/norm (R0101, R0202), pre-gate gauge-intact / post-gate:\n")
for n in NAMES:
    if "reviewa_resid" not in Z[n].files:
        continue
    z = Z[n]; rel = z["reviewa_resid"] / z["reviewa_scale"][:, None]; kg = int(z["kgate"])
    pre = (np.arange(len(z["Trs"])) < kg) & z["gauge_intact"] & ~np.isnan(z["reviewa_scale"]); post = (np.arange(len(z["Trs"])) >= kg) & ~np.isnan(z["reviewa_scale"])
    print("- %s: pre %.3f, %.3f; post %.3f, %.3f" % (n, np.median(rel[pre, 0]) if pre.any() else np.nan, np.median(rel[pre, 1]) if pre.any() else np.nan,
                                                 np.median(rel[post, 0]), np.median(rel[post, 1])))
print("\n## (b) radial null rays from the formation record\n")
for n in NAMES:
    R = S[n]["rays"]["1"]
    print("### %s: launch T/r_s %.4g (zeros %s), window to %.4g (zeros at the end %s), dT step %.3g a\n" % (
        n, R["T_start_over_rs"], ", ".join("%.2f" % z for z in S[n]["formation"]["zeros"]), R["T_end_over_rs"], ", ".join("%.2f" % z for z in R["zeros_end"]), R["dt"]))
    print("| launch | r0 | dir | reaches r = 0 at T/r_s (affine length there) | r end | d|r|/dT end | eps end | eps max | eps min | affine length to the window's end | its growth over the last 5 r_s | d ln eps/dT over the last 5 r_s | K_t^2 at end | frame norm at end | end state |")
    print("|" + "---|" * 15)
    for row in R["rows"]:
        if not row["label"]:
            continue
        print("| %s | %.2f | %s | %s | %.2f | %s | %.3g | %.3g | %.3g | %.4g | %.3g | %.3g | %s | %s | %s |" % (
            row["label"], row["r0"], row["dir"], "-" if row["T_centre_over_rs"] is None else "%.3f (%.3g)" % (row["T_centre_over_rs"], row["lam_centre"]),
            row["r_end"], g(row["drdT_end"]), row["eps_end"], row["eps_max"], row["eps_min"], row["lam_end"], row["dlam_last_5rs"], row["dlneps_dT_last_5rs"],
            g(row["Kt2_at_end"], "%.3g"), g(row["norm_at_end"], "%.3g"), row["status"]))
    rows = R["rows"]
    nin = sum(1 for r_ in rows if r_["dir"] == "in"); nout = len(rows) - nin
    reach = sum(1 for r_ in rows if r_["dir"] == "in" and r_["T_centre_over_rs"] is not None)
    esc = sum(1 for r_ in rows if "escap" in r_["status"])
    trap = sum(1 for r_ in rows if r_["status"].startswith("between"))
    core = sum(1 for r_ in rows if "untrapped core" in r_["status"])
    out_esc = [r_["r0"] for r_ in rows if r_["dir"] == "out" and "escap" in r_["status"]]
    out_not = [r_["r0"] for r_ in rows if r_["dir"] == "out" and "escap" not in r_["status"]]
    lam_c = [r_["lam_centre"] for r_ in rows if r_["dir"] == "in" and r_["lam_centre"] is not None]
    print("\nall launches (every node to 3 r_out plus the labelled radii): %d ingoing, %d outgoing; ingoing reaching r = 0: %d of %d (affine length there %.3g to %.3g); "
          "ending escaping or off the grid: %d; ending in the trapped shell: %d; ending inside the inner root: %d; largest eps %.3g; smallest eps %.3g\n" % (
              nin, nout, reach, nin, min(lam_c) if lam_c else float("nan"), max(lam_c) if lam_c else float("nan"), esc, trap, core,
              max(r_["eps_max"] for r_ in rows), min(r_["eps_min"] for r_ in rows)))
    print("outgoing launches: escaping from r0 >= %s; not escaping for r0 <= %s (the outer root at launch %.2f)\n" % (
        g(min(out_esc) if out_esc else None, "%.2f"), g(max(out_not) if out_not else None, "%.2f"), R["r_out0"]))

print("## ray sensitivity: cadence (records kept every 1, 2, 4) and half step\n")
print("| member | variant | in from r_out: T/r_s at r = 0 (affine) | in from r_out: r end, eps end, affine end | out from mid: r end, eps end, affine end | out from r_out: r end, state | out from 2 r_out: r end, state |\n|---|---|---|---|---|---|---|")
for n in NAMES:
    for c in ("1", "2", "4", "dt_half"):
        R = S[n]["rays"].get(c)
        if R is None:
            continue
        d = {(r_["label"], r_["dir"]): r_ for r_ in R["rows"] if r_["label"]}
        a_ = d[("r_out", "in")]; b_ = d[("mid", "out")]; c_ = d[("r_out", "out")]; e_ = d[("2 r_out", "out")]
        print("| %s | %s | %s | %.2f, %.3g, %.4g | %.2f, %.3g, %.4g | %.2f, %s | %.1f, %s |" % (
            n, c, "-" if a_["T_centre_over_rs"] is None else "%.3f (%.4g)" % (a_["T_centre_over_rs"], a_["lam_centre"]), a_["r_end"], a_["eps_end"], a_["lam_end"],
            b_["r_end"], b_["eps_end"], b_["lam_end"], c_["r_end"], c_["status"], e_["r_end"], e_["status"]))
