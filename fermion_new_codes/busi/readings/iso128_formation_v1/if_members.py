#!/usr/bin/env python
"""if_members.py (busi reading of ISO128_formation_readout_v1): the comparison members and the matched-clock comparison, with my own
reader (if_common): per member the md5, the production rule's extent and bands (and the kept-only rule for reference), the first
record with a trapping zero and its row, the rows at the delivery's clocks, E/E0 across formation (the loss over the 7 records
before formation and over read 3's 15-record window where the records exist, the loss per record and per r_s at formation), and the
H2 run against the plain run at every common clock (E/E0, min(N - |v|), R_c). Writes OUT/members.json."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if_common as C  # noqa: E402

CLOCKS = [2.75, 2.875, 3.0, 3.5625, 11.0, 11.0625, 11.1875, 11.25, 11.5, 12.375]


def ee0_series(name):
    """(T, E/E0) of a member's stored records; for the H2 member the json to 11.25 and the log's E/E0 after it."""
    if name == "ISO_128_s4_H2":
        J = C.load(C.H2_JSON)
        R = C.records(J)
        s = [(float(r["T_over_rs"]), float(r["E_over_E0"]), float(r["min_N_minus_absv"])) for r in R]
        NL, EL, _ = C.parse_log(C.H2_LOG)
        for k in sorted(EL, key=float):
            if float(k) > R[-1]["T_over_rs"] + 1e-6:
                s.append((float(k), EL[k]["EE0"], EL[k]["minNv"]))
        return s
    J = C.load(C.MEMBERS[name])
    return [(float(r["T_over_rs"]), float(r["E_over_E0"]), float(r["min_N_minus_absv"])) for r in C.records(J)]


def windows(series, tau_rs):
    """Formation = the first record with min(N - |v|) < 0; the losses of E0 over [f-7, f], [f-7, f+7], [f-4, f] (0.5 r_s at
    tau/r_s 0.125, 0.25 r_s at 0.0625), and the per-record loss at f."""
    T = [x[0] for x in series]
    E = [x[1] for x in series]
    m = [x[2] for x in series]
    f = next((i for i, v in enumerate(m) if v < 0), None)
    if f is None:
        return {"formation_T": None}
    out = {"formation_T": T[f], "EE0_at_f": E[f], "EE0_at_f_minus_1": E[f - 1], "loss_per_record_at_f": E[f - 1] - E[f],
           "loss_per_rs_at_f": (E[f - 1] - E[f]) / tau_rs, "tau_rs": tau_rs}
    for a, b, lab in ((7, 0, "f-7..f"), (7, 7, "f-7..f+7"), (4, 0, "f-4..f"), (8, 0, "f-8..f"), (0, 7, "f..f+7")):
        i, j = f - a, f + b
        if i >= 0 and j < len(E):
            out["loss_" + lab] = E[i] - E[j]
            out["T_" + lab] = (T[i], T[j])
            out["EE0_" + lab] = (E[i], E[j])
    # the loss over the same clock span before formation for every member: 0.4375 r_s (ISO_128's f-7..f), by linear interpolation
    span = 0.4375
    t0 = T[f] - span
    if t0 >= T[0]:
        e0 = float(np.interp(t0, T, E))
        out["loss_span_0p4375_rs_before_f"] = e0 - E[f]
    return out


def main():
    os.makedirs(C.OUT, exist_ok=True)
    out = {"members": {}}
    for name, path in C.MEMBERS.items():
        J = C.load(path)
        R = C.records(J)
        newton = C.is_newton(J)
        A, H = J["args"], J["head"]
        ext, rows = C.certify(R, newton, t0_as_plain=True)
        extA, _ = C.certify(R, newton, t0_as_plain=False)
        ko = C.certify_kept_only(R, newton)
        fz = next((i for i, r in enumerate(R) if C.zeros_of(r)), None)
        first_zero = None
        if fz is not None:
            first_zero = C.row_of(R[fz], J)
            first_zero["certified"] = rows[fz]["certified"]
            first_zero["counts"] = rows[fz]["counts"]
            first_zero["kept"] = rows[fz]["kept"]
            first_zero["conf"] = rows[fz]["conf"]
        clocks = {}
        for t in CLOCKS:
            i = C.at_clock(R, t)
            if i is None:
                continue
            row = C.row_of(R[i], J)
            row.update({"counts": rows[i]["counts"], "certified": rows[i]["certified"], "band": rows[i]["band"], "kept": rows[i]["kept"], "conf": rows[i]["conf"]})
            clocks[str(t)] = row
        cert_rows = [x for x in rows if x["certified"]]
        nonc = [(x["T"], x["kept"], x["conf"], x["why"]) for x in rows if not x["counts"]]
        out["members"][name] = {
            "path": path, "md5": C.md5(path), "bytes": os.path.getsize(path), "script": J.get("script"), "newton": newton, "records": len(R),
            "T_first": R[0]["T_over_rs"], "T_last": R[-1]["T_over_rs"], "E0": H.get("E0"), "rs": A.get("rs"), "g": A.get("g"), "tau": A.get("tau"),
            "tau_rs": float(A["tau"]) / float(A["rs"]), "nsites": A.get("nsites"), "ell": A.get("ell"), "kappa_sea": A.get("kappa_sea"),
            "production_rule": {k: ext[k] for k in ("first_noncounting_T", "first_noncounting_why", "certified_through_T", "n_certified", "band_T", "band_T_certified", "noncounting_T", "max_kept", "max_kept_T", "max_conf", "max_conf_T")},
            "production_rule_record0_as_coded": {k: extA[k] for k in ("first_noncounting_T", "certified_through_T")},
            "kept_only_rule": ko,
            "max_kept_certified": max((x["kept"] for x in cert_rows if x["kept"] is not None), default=None),
            "max_conf_certified": max((x["conf"] for x in cert_rows if x["conf"] is not None), default=None),
            "noncounting": nonc[:30], "n_noncounting": len(nonc),
            "unconverged_T": [x["T"] for x in rows if x["converged"] is not True][:40], "n_unconverged": sum(1 for x in rows if x["converged"] is not True),
            "first_zero": first_zero, "clocks": clocks,
            "windows": windows([(float(r["T_over_rs"]), float(r["E_over_E0"]), float(r["min_N_minus_absv"])) for r in R], float(A["tau"]) / float(A["rs"])),
            "record0": {k: rows[0][k] for k in ("kept", "conf", "counts", "how")},
        }
    # the H2 member's windows (json + log) and its matched-clock rows against the plain run
    h2 = ee0_series("ISO_128_s4_H2")
    out["ISO_128_s4_H2_windows"] = windows(h2, 0.0625)
    P = C.records(C.load(C.MEMBERS["ISO_128_s4_plain"]))
    JH = C.load(C.H2_JSON)
    RH = C.records(JH)
    common = []
    for r in RH:
        i = C.at_clock(P, float(r["T_over_rs"]))
        if i is None:
            continue
        p = P[i]
        common.append({"T": float(r["T_over_rs"]), "dEE0": float(r["E_over_E0"]) - float(p["E_over_E0"]), "dminNv": float(r["min_N_minus_absv"]) - float(p["min_N_minus_absv"]),
                       "dRc": float(r["R_c"]) - float(p["R_c"]), "plain_converged": p.get("converged"), "plain_kept": C.residuals(p, False)["kept"]})
    out["H2_vs_plain_common_clocks"] = {"n": len(common), "max_abs_dEE0": max(abs(c["dEE0"]) for c in common), "max_abs_dEE0_T": max(common, key=lambda c: abs(c["dEE0"]))["T"],
                                        "max_abs_dminNv": max(abs(c["dminNv"]) for c in common), "max_abs_dminNv_T": max(common, key=lambda c: abs(c["dminNv"]))["T"],
                                        "max_abs_dRc": max(abs(c["dRc"]) for c in common),
                                        "stretch_0_to_3p375": {"max_abs_dEE0": max(abs(c["dEE0"]) for c in common if c["T"] <= 3.375 + 1e-9), "max_abs_dminNv": max(abs(c["dminNv"]) for c in common if c["T"] <= 3.375 + 1e-9),
                                                               "max_abs_dRc": max(abs(c["dRc"]) for c in common if c["T"] <= 3.375 + 1e-9)},
                                        "stretch_0_to_1": {"max_abs_dEE0": max(abs(c["dEE0"]) for c in common if c["T"] <= 1.0 + 1e-9)},
                                        "rows_every_16": [c for c in common if abs((c["T"] * 16) % 16) < 1e-9 or c["T"] in (0.0625, 0.125, 0.25, 0.5, 2.75, 3.0, 3.375, 3.4375, 5.0, 11.0625, 11.1875, 11.25)]}
    # the log's 11.5 against the plain's 11.5 record
    NL, EL, _ = C.parse_log(C.H2_LOG)
    e = EL["11.500"]
    ip = C.at_clock(P, 11.5)
    prow = C.row_of(P[ip], C.load(C.MEMBERS["ISO_128_s4_plain"]))
    out["H2_11p5_log_vs_plain_11p5"] = {"H2": {"Rt": e["Rt"], "Rt_rs": e["Rt"] / 128.0, "Rt_2gE": e["Rt"] / e["twogE"], "rin": e["rin"], "rin_rs": e["rin"] / 128.0, "EE0": e["EE0"], "minNv": e["minNv"]},
                                        "plain": prow, "dRt_rel_pct": 100 * (e["Rt"] / 128.0 - prow["Rt_rs"]) / prow["Rt_rs"], "dEE0": e["EE0"] - prow["EE0"], "dminNv": e["minNv"] - prow["minNv"],
                                        "drin_rel_pct": 100 * (e["rin"] / 128.0 - prow["rin_rs"]) / prow["rin_rs"]}
    # the first-zero table's comparisons, ISO_128 (log) against each member
    cmp = {}
    for name, m in out["members"].items():
        fz = m["first_zero"]
        if not fz:
            cmp[name] = None
            continue
        cmp[name] = {"T_f": fz["T"], "dT_f": 11.5 - fz["T"], "Rt_rs": fz["Rt_rs"], "dRt_rel_pct": 100 * (e["Rt"] / 128.0 - fz["Rt_rs"]) / fz["Rt_rs"], "certified": fz["certified"]}
    out["first_zero_comparison"] = cmp
    # the matched-clock 11.5 comparisons
    m115 = {}
    for name, m in out["members"].items():
        row = m["clocks"].get("11.5")
        if row is None:
            continue
        m115[name] = {"Rt_rs": row["Rt_rs"], "dRt_rel_pct": (100 * (e["Rt"] / 128.0 - row["Rt_rs"]) / row["Rt_rs"]) if row["Rt_rs"] else None, "EE0": row["EE0"], "minNv": row["minNv"],
                      "Rt_2gE": row["Rt_2gE"], "rin_rs": row["rin_rs"], "counts": row["counts"], "certified": row["certified"]}
    out["matched_11p5"] = m115
    # E0 comparisons
    E0s = {n: m["E0"] for n, m in out["members"].items()}
    out["E0"] = {"values": E0s, "H2": JH["head"]["E0"], "H2_equals_plain": JH["head"]["E0"] == E0s["ISO_128_s4_plain"],
                 "S4_128_vs_ISO_128_rel": (E0s["ISO_128_s4_plain"] - E0s["S4_128_s4"]) / E0s["S4_128_s4"], "S4_128_members_equal": len(set(E0s[n] for n in ("S4_128_s4", "S4_128_s4_nk_v36m", "S4_128_s4_nk_v36q", "S4_128_s8", "S4_128_s16"))) == 1}
    C.jdump(out, C.OUT + "/members.json")
    for name, m in out["members"].items():
        print("== %s md5 %s records %d (%s to %s) newton %s E0 %s tau/rs %.4f" % (name, m["md5"], m["records"], m["T_first"], m["T_last"], m["newton"], m["E0"], m["tau_rs"]))
        print("   rule:", m["production_rule"], "| as coded:", m["production_rule_record0_as_coded"], "| kept-only:", m["kept_only_rule"])
        print("   max kept/conf certified:", m["max_kept_certified"], m["max_conf_certified"], "| noncounting:", m["noncounting"][:6], "| unconverged n", m["n_unconverged"])
        print("   first zero:", m["first_zero"])
        for t, row in m["clocks"].items():
            print("   clock %s: EE0 %.7f minNv %.5f Rt_rs %s Rt_2gE %s (json %s) rin_rs %s counts %s certified %s kept %s conf %s" % (
                t, row["EE0"], row["minNv"], row["Rt_rs"], row["Rt_2gE"], row["Rt_2gE_json"], row["rin_rs"], row["counts"], row["certified"], row["kept"], row["conf"]))
        print("   windows:", m["windows"])
    print("== H2 windows:", out["ISO_128_s4_H2_windows"])
    print("== H2 vs plain:", {k: v for k, v in out["H2_vs_plain_common_clocks"].items() if k != "rows_every_16"})
    for c in out["H2_vs_plain_common_clocks"]["rows_every_16"]:
        print("   ", c)
    print("== H2 11.5 vs plain:", out["H2_11p5_log_vs_plain_11p5"])
    print("== first zero comparison:", cmp)
    print("== matched 11.5:", m115)
    print("== E0:", out["E0"])


if __name__ == "__main__":
    main()
