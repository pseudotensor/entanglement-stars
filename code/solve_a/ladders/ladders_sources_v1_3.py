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
    ap.add_argument("--certext", required=True, help="v1.3: certext_production_v1.json")
    ap.add_argument("--followup", required=True, help="v1.3: certext_followup_v1.json")
    a = ap.parse_args()
    D = json.load(open(os.path.join(a.readout_dir, "ladders_readout_v1_3.json")))
    M = D["members"]
    jp = os.path.join(a.readout_dir, "ladders_readout_v1_3.json"); npz = os.path.join(a.readout_dir, "ladders_readout_v1_3.npz")
    lc, lr = a.local_code, a.local_results
    S = {"catalogs": [(c, md5(c)) for c in a.catalog], "catalog_script": os.path.join(lc, "ladders_catalog_v1.py"),
         "producer": os.path.join(lc, "ladders_readout_v1_3.py"), "producer_md5": md5(os.path.join(a.code_dir, "ladders_readout_v1_3.py")),
         "lc_md5": md5(os.path.join(a.code_dir, "engine_ladder_compare_v5.py")),
         "composer": os.path.join(lc, "ladders_compose_v1_3.py"), "composer_text": os.path.join(lc, "compose_text_v1_3.py"),
         "readout_json": os.path.join(lr, "ladders_readout_v1_3.json"), "readout_json_md5": md5(jp),
         "readout_npz": os.path.join(lr, "ladders_readout_v1_3.npz"), "readout_npz_md5": md5(npz), "box_dir": a.readout_dir,
         "engine": a.engine, "engine_md5": md5(a.engine),
         "certext": a.certext, "certext_md5": md5(a.certext), "followup": a.followup, "followup_md5": md5(a.followup),
         "followup_script": os.path.join(lc, "certext_followup_v1.py")}
    for k in ("ladders_catalog_v1.py", "ladders_compose_v1_3.py", "compose_text_v1_3.py", "ladders_figs_v1_3.py", "ladders_sources_v1_3.py",
              "certext_followup_v1.py", "ladders_diff_v1_3.py"):
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
                  "(its first record does not count; first above 1e-3 at %s) and its tau 4 control (to %.3f): the mass ladder's formation readouts at "
                  "r_s/a 64 are 2 members." % (none_to("M3a_s4"), M["M3a_s2"]["convergence"]["first_above_1e-3"], none_to("M3a_s4_tau4")))
    c = M["M3c_s4_nk"]["convergence"]
    b22 = next((x for x in c["banded_records"] if abs(x["T"] - 22.0) < 1e-6), {})
    mo.append("The (0.1, 96) member M3c_s4_nk (Newton, v36j) is certified to %.4f: the record 22.0000 counts banded (kept %.2e, confirming %.2e); "
              "its end record %.4f is kept at %.2e and does not count. The busi member line's \"to 22.167\" names that end record. Its formation %.4f "
              "lies inside the certified extent." % (c["last_converged"], b22.get("kept", float("nan")), b22.get("confirming", float("nan")), c["last"],
                                                     c["residual_max"], fo("M3c_s4_nk")["T"]))
    cd = M["M3d_s4"]["convergence"]; ud = cd.get("unconverged_plain") or []
    mo.append("The (0.05, 326) member M3d_s4 has not reached formation: records to %.4f, certified to %.4f (its %d records kept unconverged have kept "
              "residuals at most %.2e and following entries at most %.2e); it is the kappa_sea 12 plain run (v36e), not the isotropic set the "
              "requirement names; its catalog is not composed (no t0 line; a 3912-site preparation was not rerun)." % (
                  none_to("M3d_s4"), lastc("M3d_s4"), len(ud), max([x["kept"] for x in ud] or [float("nan")]),
                  max([x["following"] for x in ud if x["following"] is not None] or [float("nan")])))
    mo.append("No step control exists for the (0.4, 64) and (0.1, 96) members, and no averaging-scale control for the (0.1, 96) member.")
    def nc_text(nm):
        v = M[nm]["convergence"]
        if v["first_noncounting"] is None:
            return "through its last record"
        x = next((y for y in v["excluded"] if abs(y["T"] - v["first_noncounting"]) < 1e-9), {})
        return "to %s (the record %s: %s; kept %.2e, confirming %s)" % (lastc(nm), v["first_noncounting"], x.get("gate"), x.get("residual") or float("nan"),
                                                                       ("%.2e" % x["confirming"]) if x.get("confirming") is not None else "-")
    bt4 = [y for y in M["P64_s4_tau4"]["convergence"]["banded_records"] if y["certified"]]
    mo.append("The tau 2 control of the (0.2, 64) member is certified %s, the tau 4 control %s%s; the tau 2 control's later closings, second roots "
              "and radii are uncertified." % (nc_text("P64_s4_tau2"), nc_text("P64_s4_tau4"),
                                              "".join(" (its record %s kept unconverged counts banded: kept %.2e, following entry %.2e)" % (
                                                  y["T"], y["kept"], y["confirming"]) for y in bt4)))
    mo.append("Not composed (the reading's routing): the birth located between records (busi B7), the T = 0 valley census for the per-valley "
              "onsets, the (32, 4, 8) step member, the core's compression radius and bare energy per particle, nu, chi^(1/3), epsilon_c and c_b.")
    so = []
    if not fo("S4_16_s4"):
        so.append("The (0.2, 16) member S4_16_s4 holds no trapping zero to %.3f; neither do its sigma/a 2 control (to %.3f) and its tau 1 control "
                  "(to %.3f)." % (none_to("S4_16_s4"), none_to("S4_16_s2"), none_to("S4_16_s4_tau1")))
    f128 = fo("S4_128_s4")
    so.append("The (0.2, 128) sigma/a 4 member S4_128_s4 (plain) is certified %s; its first kept residual above 1e-3 is at %.4f; its records "
              "continue to %.4f%s. Its Newton runs stopped early: v36m at %.4f, certified %s; v36q at %.4f, certified %s; no Newton run of this member "
              "has landed." % (
                  nc_text("S4_128_s4"), M["S4_128_s4"]["convergence"]["first_above_1e-3"],
                  none_to("S4_128_s4"), (" with a first zero at %.4f (R_t %.3f a), uncertified" % (f128["T"], f128["R_t"])) if f128 else "",
                  none_to("S4_128_s4_nk_v36m"), nc_text("S4_128_s4_nk_v36m"), none_to("S4_128_s4_nk_v36q"), nc_text("S4_128_s4_nk_v36q")))
    so.append("The sigma/a 2 point at r_s/a 64 is the composite history (Newton across its seams), certified %s; the plain rung 2 json (md5 "
              "a405b4923eab8ff22ddccab78e59af6d) is not on the box." % nc_text("P64_s2"))
    b0 = next((y for y in M["RS32_s2"]["convergence"]["banded_records"] if y["T"] == 0.0), {})
    so.append("RS32_s2 is certified %s, its T = 0 record banded (kept %.2e, confirming %.2e); the r_s/a 32 averaging-scale comparison is certified on "
              "both sides (6.3)." % (nc_text("RS32_s2"), b0.get("kept", float("nan")), b0.get("confirming", float("nan"))))
    so.append("The (0.2, 64) release member at 2 r_s, S4_R2_m0p2, is certified %s; its formation at %s lies inside." % (
        nc_text("S4_R2_m0p2"), fo("S4_R2_m0p2")["T"] if fo("S4_R2_m0p2") else "-"))
    so.append("S4_128_s8 finished on R1 (queue one_S4_128_s8: finished 2026-10-19T23:14:48Z, resumed from its T/r_s 30 slice at 21:22:33Z); read from its final run json mirrored to the box (records to %.4f)." % none_to("S4_128_s8"))
    so.append("Not composed (the reading's routing): the birth located between records (busi B7), the (32, 4, 8) step member, the T = 0 valley "
              "census, chi^(1/3) on the shed energy share, epsilon_c and c_b.")
    rs = M["RS32_s4"]["catalog"]
    hc = rs.get("head_comparison") or {}
    rs32 = []
    if hc:
        rs32.append("The r_s/a 32 member's preparation rule: the engine_collapse_v36d preparation (the well's filling within kappa_sea 10, "
                    "prepare_dust with fill_lowest) run on the box CPU with that member's stored arguments reproduces its stored head: N_B "
                    "relative difference %.1e, E0 %.1e, f_kin %.1e; it is the same rule as every other member's." % (
                        hc["N_B"]["rel_diff"], hc["E0"]["rel_diff"], hc["f_kin"]["rel_diff"]))
    figs_m = ["figure: %s/MASS_ladder_v1.3_fig.png (and .pdf), drawn by %s/ladders_figs_v1_3.py from the arrays below" % (lr, lc),
              "arrays: %s (per member per record: T/r_s, E/E0, R_t, R_t/r_s, R_t/(2gE), min(N - |v|), R_c, residual, confirming residual, "
              "converged as stored, counts, certified, conf_fail, capped, above, the number of zeros, tangencies, pockets, inversions, slope at R_t; keys <member>__<column>), md5 in SOURCES" % S["readout_npz"],
              "numbers: %s (formation, episodes, matched clocks, fits with coefficients and residuals, control spreads, catalogs)" % S["readout_json"]]
    figs_s = [x.replace("MASS_ladder_v1.3_fig", "SCALING_ladder_v1.3_fig") for x in figs_m]
    S.update({"mass_open": mo, "scaling_open": so, "rs32_prep": rs32, "mass_figs": figs_m, "scaling_figs": figs_s})
    json.dump(S, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
