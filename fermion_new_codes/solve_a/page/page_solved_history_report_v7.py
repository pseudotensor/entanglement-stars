#!/usr/bin/env python
"""page_solved_history_report_v7.py (the lane's analyst builder; engine spec 41.4 (2), 41.7 item 3; inbox #62 item 1): the composer of
PAGE_solved_history version 3. It runs page_solved_history_report_v6.py unchanged (its readers, tables, figure block and wording net) and adds:
(1) the header and script line of version 3;
(2) sec. 8, A to T/r_s 100 (page_history_v6i from T = 0, pod 2, on the box's B records): its rows as they stand (every row kept in the json),
    its final flag and end, every row from T/r_s 58 labelled 'after the chain end's first return' (D's return marker, 41.2 (1)), and A to 100's
    values at the clocks FILE 3 reads (14, 36 to 40, 40) beside version 2's run (v6bA_s4) with their differences; the 40-to-100 sentence is
    turn 50's and is not written here;
(3) sec. 9, the variant attribution of S_col at T/r_s 40 (#62 item 1): each variant's selected-archive entropy at 40 by name and code, the
    spread over eps_loc 1e-3 and the kappa 3 set printed as the digits' error, the far collector labelled 'not yet arrived' (its matched arrival
    clock is #62 item 3's and is printed only when its file exists), the sigma/a 8 history a different object beside;
(4) sec. 10, the amended drain rule on c15 and c30 (results/page/DRAIN/drain_rule_read_c15_c30_v1.json, by drain_rule_read_v1.py) with the
    composer's own clean-window flux ranges printed beside #56's pre-marker ranges (5.36975 to 6.26798 and -0.0171821 to 0.0756697) and their
    agreement asserted where the rows coincide.
Usage: python page_solved_history_report_v7.py RESULTS_PAGE OUT_MD OUT_JSON [--draft-label TEXT]"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_solved_history_report_v6 as V6     # noqa: E402

A100 = "v6bA_runs/v6iA_s4to100/v6iA_s4to100_summary.json"
DRAIN = "DRAIN/drain_rule_read_c15_c30_v1.json"
ARRIVAL = "S_col_attribution/far_collector_arrival_v1.json"     # #62 item 3's matched arrival clock, when it exists
MARKER_RS = 58.0
R56 = {"c15": (5.36975, 6.26798), "c30": (-0.0171821, 0.0756697)}
VARIANTS = [("s4", "v6bA_runs/v6bA_s4/v6bA_s4_summary.json", "the baseline (page_history_v6b)"),
            ("s4loc3", "v6bA_runs/v6dA_s4loc3/v6dA_s4loc3_summary.json", "eps_loc 1e-3 (page_history_v6d)"),
            ("s4k3", "v6bA_runs/v6fA_s4k3/v6fA_s4k3_summary.json", "the kappa 3 channel set (page_history_v6f)"),
            ("s4far", "v6bA_runs/v6dA_s4far/v6dA_s4far_summary.json", "the far collector (page_history_v6d)"),
            ("s8", "v6bA_runs/v6fA_s8/v6fA_s8_summary.json", "the sigma/a 8 history (page_history_v6f), a different object")]
FILE3_CLOCKS = (14.0, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0)


def at(rows, t):
    for r in rows:
        if abs(r["t_over_rs"] - t) < 1e-9:
            return r
    return None


def main():
    args = sys.argv[1:]
    base, p_md, p_json = args[:3]
    extra = args[3:]
    sys.argv = [sys.argv[0], base, p_md, p_json] + extra
    V6.main()
    md = open(p_md).read()
    J = json.load(open(p_json))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = md.replace("PAGE_solved_history version 2 (S1 revised, inbox #36; with #41; engine spec 38.1)",
                    "PAGE_solved_history version 3 (S1 revised, inbox #36; with #41; engine spec 38.1, 41.4 (2); inbox #62 item 1)", 1)
    md = md.replace("by code/page/page_solved_history_report_v6.py", "by code/page/page_solved_history_report_v7.py (on page_solved_history_report_v6.py)", 1)
    J["script"] = "code/page/page_solved_history_report_v7.py (on page_solved_history_report_v6.py and v1's readers)"
    J["version"] = 3
    L = []
    # sec. 8: A to 100
    pa = os.path.join(base, A100)
    a = V6.V1.load(pa)
    s4rows = (J.get("A_s4") or {}).get("rows") or []
    L.append("\n## 8. A to T/r_s 100 (page_history_v6i from T = 0 on the box's B records; pod 2)\n")
    if a:
        rows = a["rows"]
        for r in rows:
            r["after_first_return"] = bool(r["t_over_rs"] >= MARKER_RS - 1e-9)
        last = rows[-1]
        J["A_to100"] = {"path": "results/page/" + A100, "md5": V6.V1.md5(pa), "final": a.get("final"), "end_reason": a.get("end_reason"), "label": a.get("label"),
                        "label_v6i": a.get("label_v6i"), "marker_rs": MARKER_RS, "rows": rows}
        L.append("- %s (md5 %s): final %s; %d rows to T/r_s %.3f; the end: %s; every row from T/r_s %.0f is labelled 'after the chain end's first return' "
                 "(D's return marker, engine spec 41.2 (1)): %d rows.\n" % (J["A_to100"]["path"], J["A_to100"]["md5"], a.get("final"), len(rows), last["t_over_rs"],
                                                                             a.get("end_reason"), MARKER_RS, sum(r["after_first_return"] for r in rows)))
        L.append("| T/r_s | E | E_inv/E (A to 100) | E_inv/E (version 2's run) | S_col (A to 100) | S_col (version 2's run) | L/L_H(kappa_T) (A to 100) | (version 2's run) |")
        L.append("|---|---|---|---|---|---|---|---|")
        cmp_ = []
        for t in FILE3_CLOCKS:
            x, y = at(rows, t), at(s4rows, t)
            if x is None and y is None:
                continue

            def q(r, k, fmt="%.6g"):
                return (fmt % r[k]) if (r is not None and r.get(k) is not None) else "-"
            ex = (x["E_inv_weighted"] / x["E"]) if x else None; ey = (y["E_inv_weighted"] / y["E"]) if y else None
            lx = x.get("L_over_LH_kappaT") if x else None     # the interval flux over the comparator, the row's own (v6b to v6i), as version 2 reads it
            ly = y.get("L_over_LH_kappaT") if y else None
            cmp_.append({"t_over_rs": t, "Einv_over_E": [ex, ey], "S_col": [x["S_esc_weighted"] if x else None, y["S_esc_weighted"] if y else None], "L_over_LH": [lx, ly]})
            L.append("| %.3f | %s | %s | %s | %s | %s | %s | %s |" % (t, q(x or y, "E", "%.6f"), "%.6g" % ex if ex is not None else "-", "%.6g" % ey if ey is not None else "-",
                                                                q(x, "S_esc_weighted"), q(y, "S_esc_weighted"), "%.4g" % lx if lx is not None else "-", "%.4g" % ly if ly is not None else "-"))
        J["A_to100"]["file3_clocks_against_version2"] = cmp_
        L.append("\nFILE 3 is filled from version 2's run (v6bA_s4, the run its printed defaults come from); A to 100's values at the same clocks are printed above "
                 "beside it (another code version, page_history_v6i); the 40-to-100 sentence is turn 50's.\n")
    else:
        J["A_to100"] = None
        L.append("- not landed (%s).\n" % A100)
    # sec. 9: variants
    L.append("\n## 9. The variant attribution of S_col at T/r_s 40 (inbox #62 item 1)\n")
    var = {}
    for n, p, what in VARIANTS:
        d = V6.V1.load(os.path.join(base, p))
        r = at(d["rows"], 40.0) if d else None
        var[n] = {"what": what, "path": "results/page/" + p, "S_col_40": r["S_esc_weighted"] if r else None, "E_40": r["E"] if r else None, "final": (d or {}).get("final")}
        L.append("- %s, %s: S_col(40) %s nats (E(40) %s)" % (n, what, "%.6g" % var[n]["S_col_40"] if var[n]["S_col_40"] is not None else "-", "%.6f" % var[n]["E_40"] if var[n]["E_40"] else "-"))
    spread = [var[k]["S_col_40"] for k in ("s4loc3", "s4k3") if var[k]["S_col_40"] is not None]
    arr = V6.V1.load(os.path.join(base, ARRIVAL))
    J["variants_S_col_40"] = {"variants": var, "digits_error": [min(spread), max(spread)] if spread else None,
                              "far_collector": {"S_col_40": var["s4far"]["S_col_40"], "label": "not yet arrived", "matched_arrival_clock": arr}}
    if spread:
        L.append("\nThe selected archive's entropy at 40 is a collector-defined quantity: %.6g nats on the baseline, its digits' error %.6g to %.6g nats over eps_loc 1e-3 "
                 "and the kappa 3 set; the far collector's %.6g nats is labelled 'not yet arrived'%s; the sigma/a 8 history's %.6g nats is a different object, beside.\n" % (
                     var["s4"]["S_col_40"], min(spread), max(spread), var["s4far"]["S_col_40"],
                     (", its matched arrival clock T/r_s %s (#62 item 3)" % arr.get("arrival_clock_over_rs")) if arr else " (its matched arrival clock is #62 item 3's, not yet in a file)",
                     var["s8"]["S_col_40"]))
    # sec. 10: drain rule
    L.append("\n## 10. The amended drain rule on the fixed-field controls (requirements_new.txt sec. 5 P1 as amended; inbox #62 item 1)\n")
    dr = V6.V1.load(os.path.join(base, DRAIN))
    J["drain_rule"] = dr
    agree = {}
    for n in ("c15", "c30"):
        hr = ((J.get("C") or {}).get(n) or {}).get("hold_readouts") or {}
        lo, hi = hr.get("L_clean_min"), hr.get("L_clean_max")
        ok = (lo is not None and abs(lo - R56[n][0]) <= 5e-6 * max(1.0, abs(R56[n][0])) and abs(hi - R56[n][1]) <= 5e-6 * max(1.0, abs(R56[n][1])))
        agree[n] = ok
        L.append("- %s: the composer's clean-window L/L_H(kappa_T) from T/r_s %s: %s to %s; #56's pre-marker range %s to %s; agreement where the rows coincide: %s" % (
            n, hr.get("clean_window_from"), V6.g(lo, "%.6g"), V6.g(hi, "%.6g"), R56[n][0], R56[n][1], "yes" if ok else "NO"))
    J["drain_rule_ranges_against_56"] = agree
    if dr:
        for o in dr:
            L.append("- %s: clause 1 (excitation conserved) largest relative change %.1e; clause 2 (energy identity) largest residual %.1e; clause 3 (no return): %s; "
                     "clause 4: medians of L/L_H by quarter %s, falling: %s; inside the collar %.4f of its start at the last row before the reflection clock, the linear "
                     "emptying clock T/r_s %s; clause 5: %s." % (o["tag"], o["c1_Einv_max_rel_change"], o["c2_energy_identity_max_abs"],
                                                               "readable" if o["c3_readable"] else "not readable from the stored rows (no signed split at the collar)",
                                                               ", ".join("%.4g" % x for x in o["c4_quarter_medians"]), "yes" if o["c4_envelope_falling"] else "no",
                                                               o["c4_inside_fraction_left"], "%.0f" % o["c4_emptying_clock_rs"] if o["c4_emptying_clock_rs"] else "never", o["c5_reading"]))
    else:
        L.append("- the drain-rule read file is not landed (%s)." % DRAIN)
    cnt = {}
    add = V6.replace_all("\n".join(L) + "\n", cnt)
    md = md.rstrip("\n") + "\n" + add
    if cnt:
        md = md.replace("(the wording net", "(sections 8 to 10's replacements %s; the wording net" % "; ".join("\"%s\" %d" % (k, v) for k, v in cnt.items()), 1)
    cj = {}
    for k in ("A_to100", "variants_S_col_40", "drain_rule"):
        J[k] = V6.walk(J[k], cj)
    J["wording_replacements_v7"] = {"md": cnt, "json": cj}
    J["written_v7"] = now
    open(p_md, "w").write(md)
    with open(p_json, "w") as f:
        json.dump(J, f, indent=1, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    print("version 3 composed: A to 100 %s; variants %s; drain %s; #56 ranges %s" % ("landed" if a else "-", {k: v["S_col_40"] for k, v in var.items()}, "landed" if dr else "-", agree))


if __name__ == "__main__":
    main()
