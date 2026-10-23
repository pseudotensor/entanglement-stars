#!/usr/bin/env python
"""if_h2.py (busi reading of ISO128_formation_readout_v1): the ISO_128_s4 H2 member from its stored files only. Own reader
(if_common); writes OUT/h2.json and prints a summary. Groups: the inputs' md5s and the pull record; the json's composition by
provenance; records 0 to 3.0 against the T0005.00 mirror; the production rule on the 181 stored records (record 0 read as the
producer codes it and as the plain T = 0 row it is; the kept-only rule for reference); the per-record maxima and the unconverged
records; the log against the json on 3.0625 to 11.25 and the four log-only records to 11.5; the 11.5 record's arithmetic; the
min(N - |v|) trend into 11.5; the restart and stop files; the write times."""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if_common as C  # noqa: E402


def utc(path):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(path)))


def main():
    os.makedirs(C.OUT, exist_ok=True)
    out = {}
    inputs = {k: p for k, p in (("h2_json", C.H2_JSON), ("h2_fields", C.H2_FIELDS), ("h2_log", C.H2_LOG), ("slice_11p5", C.SLICE_11p5),
                                ("slice_5", C.SLICE_5), ("mirror_5_json", C.MIRROR_5_JSON), ("restart_log", C.RESTART_LOG), ("stop_file", C.STOP_FILE))}
    out["inputs"] = {k: {"path": p, "md5": C.md5(p), "bytes": os.path.getsize(p), "mtime_utc": utc(p)} for k, p in inputs.items()}
    # the pull record: MD5_box against MD5_card, line by line
    box = sorted(open(C.MD5_BOX).read().split("\n"))
    card = sorted(open(C.MD5_CARD).read().split("\n"))
    out["pull"] = {"MD5_box_equals_MD5_card": box == card, "n_lines": len([x for x in box if x.strip()]), "pull_log": open(C.PULL_LOG).read(),
                   "MD5_box": open(C.MD5_BOX).read()}
    # my own md5s against MD5_box's lines
    mine = {}
    for line in open(C.MD5_BOX):
        if line.strip():
            h, rel = line.split()
            p = C.B + "/iso128_h2/" + rel
            mine[rel] = {"listed": h, "mine": C.md5(p), "equal": C.md5(p) == h}
    out["md5_box_vs_mine"] = mine

    J = C.load(C.H2_JSON)
    R = C.records(J)
    A, H = J["args"], J["head"]
    newton = C.is_newton(J)
    out["json"] = {"records": len(R), "T_first": R[0]["T_over_rs"], "T_last": R[-1]["T_over_rs"], "newton": newton, "E0": H["E0"],
                   "g": A["g"], "rs": A["rs"], "tau": A["tau"], "end_above": A.get("end_above"), "stop_fields": A.get("stop_fields"),
                   "records_dropped_without_E": len(J["records"]) - len(R),
                   "resume_slice_v34": {k: H["resume_slice_v34"].get(k) for k in ("slice", "run", "T_over_rs", "record_index", "slice_fields_vs_stored_max")},
                   "resume_channels_eig_vs_stored_max": {k: v.get("eig_vs_stored_max") for k, v in (H["resume_slice_v34"].get("channels") or {}).items()},
                   "cadence_check": {"steps_equal_1_16": bool(np.allclose(np.diff([r["T_over_rs"] for r in R]), 0.0625)),
                                     "n_expected_0_to_11p25": int(round(11.25 / 0.0625)) + 1}}
    # composition by each record's provenance string
    prov = []
    for r in R:
        s = r.get("engine_v34") or ""
        if "plain_v36n_T0000.00.npz" in s:
            tag = "original nkq run (from the plain T = 0 slice)"
        elif "nkq_m0p2" in s and "T0003.00.npz" in s:
            tag = "H2 restart (from the nkq T0003.00 slice)"
        elif not s:
            tag = "no engine_v34 string (the T = 0 row)"
        else:
            tag = "other: " + s[:80]
        prov.append((float(r["T_over_rs"]), tag))
    runs = []
    for t, tag in prov:
        if runs and runs[-1]["tag"] == tag:
            runs[-1]["last"], runs[-1]["n"] = t, runs[-1]["n"] + 1
        else:
            runs.append({"tag": tag, "first": t, "last": t, "n": 1})
    out["provenance"] = runs
    out["engine_strings"] = sorted(set(str(r.get("engine")) for r in R))

    # records 0 to 3.0 against the mirror json
    M = C.records(C.load(C.MIRROR_5_JSON))
    i3 = max(i for i, r in enumerate(R) if r["T_over_rs"] <= 3.0 + 1e-12)
    out["records_0_to_3"] = {"n": i3 + 1, "canon_md5_final": C.canon_md5(R[:i3 + 1]), "canon_md5_mirror": C.canon_md5(M[:i3 + 1]),
                             "mirror_records": len(M), "mirror_T_last": M[-1]["T_over_rs"],
                             "mirror_equals_final_prefix_canon": C.canon_md5(M) == C.canon_md5(R[:len(M)]),
                             "mirror_equals_final_prefix_direct": all(C.canon_md5(a) == C.canon_md5(b) for a, b in zip(M, R[:len(M)]))}
    # record 0 against the plain run's record 0 (the resume keeps the row)
    P = C.records(C.load(C.MEMBERS["ISO_128_s4_plain"]))
    out["record0_vs_plain_record0"] = {"canon_equal": C.canon_md5(R[0]) == C.canon_md5(P[0]), "h2_history": R[0].get("iteration_history"),
                                       "plain_history": P[0].get("iteration_history"), "h2_has_solve_v32": "solve_v32" in R[0],
                                       "h2_converged": R[0].get("converged"), "E0_equal": J["head"]["E0"] == C.load(C.MEMBERS["ISO_128_s4_plain"])["head"]["E0"],
                                       "differing_keys": sorted(k for k in set(R[0]) | set(P[0]) if C.canon_md5(R[0].get(k)) != C.canon_md5(P[0].get(k)))}

    # the production rule on the stored records
    extA, rowsA = C.certify(R, newton, t0_as_plain=False)
    extB, rowsB = C.certify(R, newton, t0_as_plain=True)
    out["rule_record0_as_coded"] = extA
    out["rule_record0_as_plain_row"] = extB
    out["rule_kept_only"] = C.certify_kept_only(R, newton)
    out["record0_reading"] = {k: rowsA[0][k] for k in ("kept", "conf", "counts", "why", "how")}
    out["rows"] = [{k: x[k] for k in ("T", "kept", "conf", "counts", "band", "why", "converged", "certified")} for x in rowsB]
    k03 = [x for x in rowsB if x["T"] <= 3.0 + 1e-12 and x["T"] > 0]
    k311 = [x for x in rowsB if x["T"] > 3.0 + 1e-12]
    out["maxima"] = {"kept_0p0625_to_3": max(x["kept"] for x in k03), "kept_0p0625_to_3_T": max(k03, key=lambda x: x["kept"])["T"],
                     "conf_0p0625_to_3": max(x["conf"] for x in k03), "conf_0p0625_to_3_T": max(k03, key=lambda x: x["conf"])["T"],
                     "kept_3p0625_to_11p25": max(x["kept"] for x in k311), "kept_3p0625_to_11p25_T": max(k311, key=lambda x: x["kept"])["T"],
                     "conf_3p0625_to_11p25": max(x["conf"] for x in k311), "conf_3p0625_to_11p25_T": max(k311, key=lambda x: x["conf"])["T"],
                     "records_above_1e-4_either": [(x["T"], x["kept"], x["conf"]) for x in rowsB if x["kept"] > C.BAND or x["conf"] > C.BAND],
                     "records_above_1e-3_either": [(x["T"], x["kept"], x["conf"]) for x in rowsB if x["kept"] > C.CAP or x["conf"] > C.CAP],
                     "second_largest_kept": sorted((x["kept"], x["T"]) for x in rowsB)[-2], "second_largest_conf": sorted((x["conf"], x["T"]) for x in rowsB)[-2]}
    unc = []
    for r in R:
        if r.get("converged") is not True:
            sv = r.get("solve_v32") or {}
            st = ""
            try:
                st = str(sv.get("solves"))[:400]
            except Exception:  # noqa: BLE001
                pass
            unc.append({"T": r["T_over_rs"], "converged": r.get("converged"), "unconverged": r.get("unconverged"), "kept": sv.get("kept_residual"),
                        "conf": sv.get("confirming_residual"), "evaluations": sv.get("evaluations"), "meets_stop_rule": sv.get("meets_stop_rule"),
                        "ends_member": sv.get("ends_member"), "seconds": r.get("seconds"), "solves": st})
    out["unconverged_records"] = unc
    out["totals"] = {"pockets_sum": sum(C.pockets_of(r) for r in R), "pockets_max": max(C.pockets_of(r) for r in R),
                     "inversions_sum": sum(int(r.get("inversions_total") or 0) for r in R), "records_with_zero": sum(1 for r in R if C.zeros_of(r)),
                     "ends_member_true": sum(1 for r in R if str((r.get("solve_v32") or {}).get("ends_member")) == "True"),
                     "meets_stop_rule_false": [r["T_over_rs"] for r in R if str((r.get("solve_v32") or {}).get("meets_stop_rule")) == "False"],
                     "converged_false": [r["T_over_rs"] for r in R if r.get("converged") is False]}

    # the log against the json
    NL, EL, other = C.parse_log(C.H2_LOG)
    out["log"] = {"newton_lines": len(NL), "ee0_lines": len(EL), "unparsed_T_lines": other[:5], "n_unparsed": len(other),
                  "first_T": min(NL, key=float), "last_T": max(NL, key=float)}
    cmp = {"E/E0 %.7f": 0, "kept %.2e": 0, "conf %.2e": 0, "evaluations": 0, "min(N-|v|) %.4f": 0, "R_c %.2f": 0, "dust %.4f": 0, "pairs %.3e": 0,
           "pockets": 0, "inversions": 0, "converged": 0, "records compared": 0}
    diffs = []
    for r in R:
        t = float(r["T_over_rs"])
        if t <= 3.0 + 1e-12:
            continue
        key = C.log_key(t)
        if key not in NL or key not in EL:
            diffs.append((t, "missing in log"))
            continue
        n, e = NL[key], EL[key]
        sv = r["solve_v32"]
        cmp["records compared"] += 1
        tests = [("E/E0 %.7f", "%.7f" % r["E_over_E0"] == "%.7f" % e["EE0"]), ("kept %.2e", "%.2e" % float(sv["kept_residual"]) == "%.2e" % n["kept"]),
                 ("conf %.2e", "%.2e" % float(sv["confirming_residual"]) == "%.2e" % n["conf"]), ("evaluations", int(sv["evaluations"]) == n["evaluations"]),
                 ("min(N-|v|) %.4f", "%.4f" % r["min_N_minus_absv"] == "%.4f" % e["minNv"]), ("R_c %.2f", "%.2f" % r["R_c"] == "%.2f" % e["Rc"]),
                 ("dust %.4f", "%.4f" % r["E_dust_shells"] == "%.4f" % e["dust"]), ("pairs %.3e", "%.3e" % r["E_pairs_shells"] == "%.3e" % e["pairs"]),
                 ("pockets", C.pockets_of(r) == e["pockets"]), ("inversions", int(r.get("inversions_total") or 0) == e["inversions"]),
                 ("converged", (r.get("converged") is True) == n["converged"])]
        for name, ok in tests:
            cmp[name] += int(ok)
            if not ok:
                diffs.append((t, name))
    out["log_vs_json"] = {"counts": cmp, "differences": diffs[:40]}
    after = {}
    for key in sorted(NL, key=float):
        if float(key) > R[-1]["T_over_rs"] + 1e-6:
            n, e = NL[key], EL.get(key, {})
            after[key] = {"evaluations": n["evaluations"], "seconds": n["seconds"], "kept": n["kept"], "kept_evaluation": n["kept_evaluation"], "conf": n["conf"],
                          "dE": n["dE"], "converged": n["converged"], "status": n["status"], "line_search": n["line_search"],
                          "EE0": e.get("EE0"), "minNv": e.get("minNv"), "Rt": e.get("Rt"), "rin": e.get("rin"), "twogE": e.get("twogE"), "Rc": e.get("Rc"),
                          "pockets": e.get("pockets"), "inversions": e.get("inversions"), "dust": e.get("dust"), "pairs": e.get("pairs"),
                          "counts_under_rule": (n["kept"] <= C.CAP and n["conf"] <= C.CAP), "band": (n["kept"] > C.BAND or n["conf"] > C.BAND)}
    out["log_after_json"] = after
    # the 11.5 record's arithmetic
    e, n = EL["11.500"], NL["11.500"]
    g, E0, rs = float(A["g"]), float(H["E0"]), float(A["rs"])
    out["record_11p5"] = {"Rt": e["Rt"], "Rt_rs": e["Rt"] / rs, "twogE_log": e["twogE"], "twogE_from_EE0": 2 * g * E0 * e["EE0"],
                          "Rt_over_2gE_log_values": e["Rt"] / e["twogE"], "Rt_over_2gE_from_EE0": e["Rt"] / (2 * g * E0 * e["EE0"]),
                          "rin": e["rin"], "rin_rs": e["rin"] / rs, "minNv": e["minNv"], "EE0": e["EE0"], "Rc": e["Rc"], "Rsh": e["Rsh"],
                          "pockets": e["pockets"], "inversions": e["inversions"], "kept": n["kept"], "conf": n["conf"], "dE": n["dE"],
                          "evaluations": n["evaluations"], "seconds": n["seconds"], "status": n["status"], "line_search": n["line_search"],
                          "d4_of_N_plus_v": e["d4"], "line": e["line"][:420]}
    # the min(N - |v|) trend into 11.5 (json to 11.25, log after): first and second differences, the quadratic extrapolation
    seq = [(float(r["T_over_rs"]), float(r["min_N_minus_absv"])) for r in R if r["T_over_rs"] >= 11.0 - 1e-9]
    seq += [(float(k), EL[k]["minNv"]) for k in sorted(EL, key=float) if float(k) > R[-1]["T_over_rs"] + 1e-6 and float(k) < 11.5 - 1e-6]
    vals = [v for _, v in seq]
    d1 = np.diff(vals)
    d2 = np.diff(d1)
    out["minNv_trend"] = {"series": seq, "first_differences": d1.tolist(), "second_differences": d2.tolist(),
                          "quadratic_extrapolation_11p5": float(vals[-1] + d1[-1] + d2[-1]), "linear_extrapolation_11p5": float(vals[-1] + d1[-1]),
                          "log_11p5": e["minNv"]}
    # the plain run's same trend (its own records)
    pseq = [(float(r["T_over_rs"]), float(r["min_N_minus_absv"])) for r in P if 11.0 - 1e-9 <= r["T_over_rs"] <= 11.5 + 1e-9]
    pv = [v for _, v in pseq]
    out["minNv_trend_plain"] = {"series": pseq, "quadratic_extrapolation_11p5_from_11p4375": float(pv[-2] + (pv[-2] - pv[-3]) + ((pv[-2] - pv[-3]) - (pv[-3] - pv[-4]))),
                                "stored_11p5": pv[-1]}
    # the restart log's gate lines and the stop file
    rl = open(C.RESTART_LOG).read()
    out["restart_log"] = {"text": rl, "gate_lines": re.findall(r"^GATE.*$", rl, re.M), "has_differs": "differs" in rl,
                          "gate_pass_time": re.findall(r"^\[(.*?)\] GATE PASS", rl, re.M), "exit_line": re.findall(r"^\[(.*?)\] engine .* exit (\d+)", rl, re.M),
                          "slice_line": re.findall(r"slice ([0-9a-f]+) (\d+) B", rl), "reference_json": re.findall(r"reference json copied \(([0-9a-f]+)\)", rl)}
    st = open(C.STOP_FILE).read()
    out["stop_file"] = {"text": st, "times": re.findall(r"^(\S+Z) ", st, re.M), "has_11p5_newton_line": NL["11.500"]["line"] in st,
                        "has_11p5_ee0_line_prefix": EL["11.500"]["line"][:300] in st}
    C.jdump(out, C.OUT + "/h2.json")
    # summary
    print("H2 json: %d records %s to %s; newton %s; E0 %.8f" % (len(R), R[0]["T_over_rs"], R[-1]["T_over_rs"], newton, H["E0"]))
    print("provenance:", runs)
    print("records 0..3.0:", out["records_0_to_3"])
    print("record0 vs plain record0:", {k: v for k, v in out["record0_vs_plain_record0"].items() if k != "plain_history"})
    print("rule, record 0 as coded:", {k: extA[k] for k in ("first_noncounting_T", "first_noncounting_why", "certified_through_T", "n_certified", "band_T")})
    print("rule, record 0 as the plain row:", {k: extB[k] for k in ("first_noncounting_T", "certified_through_T", "n_certified", "band_T", "max_kept", "max_kept_T", "max_conf", "max_conf_T")})
    print("kept-only rule:", out["rule_kept_only"])
    print("maxima:", out["maxima"])
    print("unconverged:", [(u["T"], u["kept"], u["conf"], u["evaluations"], u["seconds"]) for u in unc])
    print("totals:", out["totals"])
    print("log:", out["log"])
    print("log vs json:", cmp, diffs[:10])
    print("log after json:", {k: (v["evaluations"], v["kept"], v["conf"], v["EE0"], v["minNv"], v["Rt"], v["rin"]) for k, v in after.items()})
    print("11.5:", {k: v for k, v in out["record_11p5"].items() if k != "line"})
    print("minNv trend:", out["minNv_trend"])
    print("minNv trend plain:", out["minNv_trend_plain"])
    print("restart:", {k: v for k, v in out["restart_log"].items() if k != "text"})
    print("stop:", {k: v for k, v in out["stop_file"].items() if k != "text"})
    print("mtimes:", {k: v["mtime_utc"] for k, v in out["inputs"].items()})
    print("pull:", out["pull"]["MD5_box_equals_MD5_card"], out["pull"]["n_lines"], all(v["equal"] for v in mine.values()))


if __name__ == "__main__":
    main()
