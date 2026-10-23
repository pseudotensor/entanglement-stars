#!/usr/bin/env python
"""engine_result_rev1_v1.py (builder analyst): assembles the tables of RESULT_final4b_engine.md revision 1 from the records (every number
is read from a record by this script; the builder's prose around them is in the RESULT file itself, never a retyped number).
Inputs (local paths under solve/results/engine/): J0p/J0p_checks.md, diag/diag_pass0_count_v2_*.json, V3D/V3D_summary_v1.md,
V3D/valley_degeneracy_v1.log, S2test/star_*.json (the zero-temperature star tests), HF/hf_cluster_v2_scan1.md, S2test/massfluct_*.json.
Args: <results/engine folder> <out.md>."""
import glob
import json
import os
import sys

d, out = sys.argv[1], sys.argv[2]
L = []


def add_file(title, path):
    L.append("## " + title)
    L.append("")
    if os.path.exists(path):
        L.append(open(path).read().rstrip("\n"))
    else:
        L.append("(missing: %s)" % os.path.basename(path))
    L.append("")


add_file("J0': the hydrodynamic solver's checks (engine_J0p_v1.py)", os.path.join(d, "J0p", "J0p_checks.md"))
L.append("## The smearing diagnosis (diag_pass0_count_v2.py): counts on the continuum start profile")
L.append("")
L.append("| (m a, g, E_F/m) | continuum N_B (proper volume) | WKB sum | chain sharp count | Fermi-weighted at T_s/m = 0.01 | 0.005 | 0.0025 | 0.00125 |")
L.append("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(d, "diag", "diag_pass0_count_v2_*.json"))):
    D = json.load(open(f)); t = D["totals"]; a = D["args"]
    L.append("| (%g, %g, %g) | %.3f | %.3f | %.0f | %.3f | %.3f | %.3f | %.3f |" % (a["m"], a["g"], a["EF_over_m"], D["NB_cont_proper"], t["wkb"], t["chain_sharp"],
             t["chain_fermi_Ts0.01"], t["chain_fermi_Ts0.005"], t["chain_fermi_Ts0.0025"], t["chain_fermi_Ts0.00125"]))
L.append("")
L.append("## The zero-temperature star tests (engine_star_v2/v3 at fixed N_B, the continuum start)")
L.append("")
L.append("| test | script | m a | g | ell | N_B | E | M_cont (continuum, same E_F start) | E/M_cont - 1 | E_F used/m | r_star | compactness | N_min | min(N - abs v) | passes | outer dN | flags |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(d, "S2test", "star_t*.json"))):
    D = json.load(open(f))
    st = D.get("start") or {}
    for mm in D["members"]:
        Mc = st.get("M_cont")
        L.append("| %s | %s | %g | %g | %d | %.2f | %.6f | %s | %s | %.5f | %s | %s | %.5f | %.5f | %d | %.1e | %s |" % (
            os.path.basename(f)[5:-5], D["script"], mm["m"], mm["g"], mm["ell"], mm["N_B"], mm["E"], "%.6f" % Mc if Mc else "-",
            "%.2e" % (mm["E"] / Mc - 1) if Mc else "-", mm["EF_used"] / mm["m"], mm["r_star"], "%.4f" % mm["compactness"] if mm["compactness"] else "-",
            mm["N_min"], mm["min_N_minus_absv"], mm["passes"], mm["outer_residual"]["max_abs_dN"], "; ".join(mm["flags"]) or "none"))
    for ec in (D["summary"].get("ell_control") or []):
        L.append("| %s ell-control | | | | %s | | %s | | | | | | | | | | spread E %.3e |" % (
            os.path.basename(f)[5:-5], ", ".join(ec["rows"].keys()), ", ".join("%.6f" % ec["rows"][k]["E"] for k in ec["rows"]), ec["spread"]["E"]))
L.append("")
add_file("V3D: the open-cube records (engine_V3D_summary_v1.py)", os.path.join(d, "V3D", "V3D_summary_v1.md"))
add_file("The valley-degeneracy test (engine_valley_degeneracy_v1.py)", os.path.join(d, "V3D", "valley_degeneracy_v1.log"))
add_file("Task 8: the cluster control (engine_hf_cluster_v2.py)", os.path.join(d, "HF", "hf_cluster_v2_scan1.md"))
L.append("## The Hartree star's mass-function fluctuation and monopole exchange (engine_massfluct_v1.py)")
L.append("")
L.append("| record | N_B | E | sqrt(Var_mat m(r*))/E | 1/sqrt(N_B) | sqrt(Var_sea m(r*))/E (channels of the matter only) | E_x^(0) matter | E_x^(0) sea |")
L.append("|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(d, "S2test", "massfluct_*.json"))):
    D = json.load(open(f))
    L.append("| %s | %.2f | %.6f | %.5f | %.5f | %.5f | %.4e | %.4e |" % (os.path.basename(f)[10:-5], D["N_B"], D["E"], D["fluct_mat_at_rstar_over_E"], D["one_over_sqrt_NB"],
             D["fluct_sea_at_rstar_over_E"], D["Ex0_matter"], D["Ex0_sea"]))
L.append("")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
