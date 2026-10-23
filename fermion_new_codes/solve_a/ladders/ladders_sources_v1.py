#!/usr/bin/env python
"""ladders_sources_v1.py: writes the SOURCES and open-items json the composer reads (md5 of every script and output; the open items
generated from the producer's json; the r_s/a 32 preparation line from the catalog's head comparison). Usage:
  python ladders_sources_v1.py --readout-dir DIR --code-dir DIR --catalog CAT.json ... --engine PATH --local-results DIR --out SOURCES.json"""
import argparse
import hashlib
import json
import os


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout-dir", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--catalog", action="append", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--local-code", default="<internal>/ladders")
    ap.add_argument("--local-results", default="<internal>/ladders")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(os.path.join(a.readout_dir, "ladders_readout_v1.json")))
    M = D["members"]
    jp = os.path.join(a.readout_dir, "ladders_readout_v1.json"); npz = os.path.join(a.readout_dir, "ladders_readout_v1.npz")
    lc, lr = a.local_code, a.local_results
    S = {"catalogs": [(c, md5(c)) for c in a.catalog], "catalog_script": os.path.join(lc, "ladders_catalog_v1.py"),
         "producer": os.path.join(lc, "ladders_readout_v1.py"), "producer_md5": md5(os.path.join(a.code_dir, "ladders_readout_v1.py")),
         "lc_md5": md5(os.path.join(a.code_dir, "engine_ladder_compare_v5.py")),
         "composer": os.path.join(lc, "ladders_compose_v1.py"), "composer_text": os.path.join(lc, "compose_text_v1.py"),
         "readout_json": os.path.join(lr, "ladders_readout_v1.json"), "readout_json_md5": md5(jp),
         "readout_npz": os.path.join(lr, "ladders_readout_v1.npz"), "readout_npz_md5": md5(npz), "box_dir": a.readout_dir,
         "engine": a.engine, "engine_md5": md5(a.engine)}
    for k in ("ladders_catalog_v1.py", "ladders_compose_v1.py", "compose_text_v1.py", "ladders_figs_v1.py", "ladders_sources_v1.py"):
        S["md5_" + k] = md5(os.path.join(a.code_dir, k))

    def fo(nm):
        return M[nm]["formation"]

    def lastc(nm):
        return M[nm]["convergence"]["last_converged"]

    def none_to(nm):
        return M[nm]["convergence"]["last"]
    mo = []
    if not fo("M3a_s4"):
        mo.append("The (0.1, 64) member M3a_s4 holds no trapping zero in its records to %.3f (its requested end), nor do its sigma/a 2 control "
                  "(capped from its first record, first above 1e-3 at %s) and its tau 4 control (to %.3f): the mass ladder's formation readouts at "
                  "r_s/a 64 are 2 members." % (none_to("M3a_s4"), M["M3a_s2"]["convergence"]["first_above_1e-3"], none_to("M3a_s4_tau4")))
    c = M["M3c_s4_nk"]["convergence"]
    mo.append("The (0.1, 96) member M3c_s4_nk (Newton, v36j) is read to its last converged record %.4f; the record 22.0000 is capped (converged "
              "False, kept residual %.2e) and excluded; its end record %.4f is kept at %.2e (above 1e-3) and excluded; the member line \"to 22.167\" "
              "of the busi item names that end record, which does not count under the rule. Its formation %.4f lies before both." % (
                  c["last_converged"], next((x["residual"] for x in c["excluded"] if abs(x["T"] - 22.0) < 1e-6), float("nan")), c["last"],
                  c["residual_max"], fo("M3c_s4_nk")["T"]))
    mo.append("The (0.05, 326) member M3d_s4 has not reached formation: its stored records end at %.4f (21 records; capped records named in 6.2; "
              "its last converged record %.4f); its own catalog is not composed (no t0 line in its status file, and its preparation at 3912 sites "
              "was not rerun on the box CPU). The member read is the kappa_sea 12 run (engine v36e, mirrored from R1); no kappa_sea 14 run of this member "
              "is among the mirrored run files." % (none_to("M3d_s4"), lastc("M3d_s4")))
    mo.append("No step control exists for the (0.4, 64) and (0.1, 96) members, and no averaging-scale control for the (0.1, 96) member; the "
              "residuals of the fixed-exponent test are printed beside the spreads that are stored.")
    mo.append("The tau 2 control of the (0.2, 64) member ends at %.3f and is uncertified from %.4f (after its formation); the tau 4 control holds "
              "1 capped record (%s) after its formation." % (none_to("P64_s4_tau2"), M["P64_s4_tau2"]["convergence"]["first_above_1e-3"],
                                                             ", ".join("%.4f" % t for t in M["P64_s4_tau4"]["convergence"]["noncounting_inside_r303"])))
    so = []
    if not fo("S4_16_s4"):
        so.append("The (0.2, 16) member S4_16_s4 holds no trapping zero to %.3f; neither do its sigma/a 2 control (to %.3f) and its tau 1 control "
                  "(to %.3f)." % (none_to("S4_16_s4"), none_to("S4_16_s2"), none_to("S4_16_s4_tau1")))
    f128 = fo("S4_128_s4")
    so.append("The (0.2, 128) sigma/a 4 member S4_128_s4 (plain) is read to its last converged record %.4f: capped from %.4f, first above 1e-3 "
              "at %.4f; its records continue (resumed by the bounce policy) to %.4f%s, all excluded from the fits. Its Newton runs stopped "
              "early: v36m at %.4f (last converged %.4f), v36q at %.4f (last converged %.4f); no Newton run of this member has landed." % (
                  lastc("S4_128_s4"), M["S4_128_s4"]["convergence"]["first_noncounting"], M["S4_128_s4"]["convergence"]["first_above_1e-3"],
                  none_to("S4_128_s4"), (" with a first zero at %.4f (R_t %.3f a), uncertified" % (f128["T"], f128["R_t"])) if f128 else "",
                  none_to("S4_128_s4_nk_v36m"), lastc("S4_128_s4_nk_v36m"), none_to("S4_128_s4_nk_v36q"), lastc("S4_128_s4_nk_v36q")))
    so.append("The free exponent on the 4 size members is therefore not evaluable; the size direction prints as the dependence of 5.1 and "
              "the fixed-exponent rows of 5.2.")
    so.append("S4_128_s8 is running on its card: read from the mirrored records to %.4f (a snapshot copied to the box readout directory, md5 in "
              "SOURCES); its formation %.4f lies inside them." % (none_to("S4_128_s8"), fo("S4_128_s8")["T"] if fo("S4_128_s8") else float("nan")))
    so.append("The sigma/a 2 point at r_s/a 64 is the composite history (Newton across its seams); the plain rung 2 json "
              "(md5 a405b4923eab8ff22ddccab78e59af6d in COLLAPSE_ladder_v2.1) is no longer on the box (the pod3_final directory was moved; its "
              "md5 is listed in <internal>/MANIFEST_pod3_final.md5) and is not used.")
    so.append("The r_s/a 32 member runs on engine v31 (its script field) with tau 2 and 1024 sites; its step control (tau 1) is engine v25 and "
              "its sigma/a 2 control is a Newton run: the controls carry their engine versions.")
    rs = M["RS32_s4"]["catalog"]
    hc = rs.get("head_comparison") or {}
    rs32 = []
    if hc:
        rs32.append("The r_s/a 32 member's preparation rule: the engine_collapse_v36d preparation (the well's filling within kappa_sea 10, "
                    "prepare_dust with fill_lowest) run on the box CPU with that member's stored arguments reproduces its stored head: N_B "
                    "relative difference %.1e, E0 %.1e, f_kin %.1e; it is the same rule as every other member's." % (
                        hc["N_B"]["rel_diff"], hc["E0"]["rel_diff"], hc["f_kin"]["rel_diff"]))
    figs_m = ["figure: %s/MASS_ladder_v1_fig.png (and .pdf), drawn by %s/ladders_figs_v1.py from the arrays below" % (lr, lc),
              "arrays: %s (per member per record: T/r_s, E/E0, R_t, R_t/r_s, R_t/(2gE), min(N - |v|), R_c, residual, confirming residual, "
              "converged, counts, zeros, tangencies, pockets, inversions, slope at R_t; keys <member>__<column>), md5 in SOURCES" % S["readout_npz"],
              "numbers: %s (formation, episodes, matched clocks, fits with coefficients and residuals, control spreads, catalogs)" % S["readout_json"]]
    figs_s = [x.replace("MASS_ladder_v1_fig", "SCALING_ladder_v1_fig") for x in figs_m]
    S.update({"mass_open": mo, "scaling_open": so, "rs32_prep": rs32, "mass_figs": figs_m, "scaling_figs": figs_s})
    json.dump(S, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
