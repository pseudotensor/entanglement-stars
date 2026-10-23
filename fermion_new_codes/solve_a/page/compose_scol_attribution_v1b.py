#!/usr/bin/env python
"""compose_scol_attribution_v1b.py (the lane's analyst builder; engine spec 44.2 (1) to (3)): SCOL_attribution_v1b, a compose (no run).
compose_scol_attribution_v1.py unchanged, then: (1) #62 item 3's "6.55 of about 106" traced in the stored v6fA_s4k3 summary (every checkpoint
row from the first to T/r_s 40 scanned, per channel weighted 16 kappa and unweighted, the set's total beside): printed with the row's clock and
convention when a row matches, otherwise stated untraced; version 1's parenthetical replaced by that statement; (2) the kappa 3 channel's
weighted E_inv, its share of the set's weighted E_inv, its own signed collar flow (weighted) and its own S_esc (weighted) at every checkpoint
row through 40; (3) the two v6j variant rows (v6jA_s4gen, v6jA_s4cub) from their summaries when landed in results/page/v6bA_runs/, else
named as running or queued; the reading sentence and the digits' error unchanged.
Usage: python compose_scol_attribution_v1b.py RESULTS_PAGE LABELS_MD OUT_MD OUT_JSON"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compose_scol_attribution_v1 as V1      # noqa: E402
import page_solved_history_report_v7b as V7B  # noqa: E402

K3 = "v6bA_runs/v6fA_s4k3/v6fA_s4k3_summary.json"
BASE = "v6bA_runs/v6bA_s4/v6bA_s4_summary.json"
VAR = [("s4gen", "v6bA_runs/v6jA_s4gen/v6jA_s4gen_summary.json", "running on pod 5 (queue v4's start, 11:22 UTC)"),
       ("s4cub", "v6bA_runs/v6jA_s4cub/v6jA_s4cub_summary.json", "queued on pod 2 after A to 100 ends (engine spec 44.3 (3))")]
OLD = "(its E_inv change 6.55 of about 106 (#62 item 3; another clock or reading, not recomputed here))"


def main():
    base, p_lab, p_md, p_json = sys.argv[1:5]
    sys.argv = [sys.argv[0], base, p_lab, p_md, p_json]
    V1.main()
    md = open(p_md, encoding="utf-8").read(); J = json.load(open(p_json))
    k3 = json.load(open(os.path.join(base, K3))); b = json.load(open(os.path.join(base, BASE)))
    bt = {round(r["t_over_rs"], 6): r for r in b["rows"]}
    rows = []
    for r in k3["rows"]:
        if r["t_over_rs"] > 40.0 + 1e-9:
            continue
        p3 = r["per_channel"]["3"]
        e3w = 48.0 * p3["E_inv"]; tot = r["E_inv_weighted"]; br = bt.get(round(r["t_over_rs"], 6))
        rows.append({"t_over_rs": r["t_over_rs"], "E": r["E"], "E_inv_weighted_set": tot, "E_inv_weighted_production": br["E_inv_weighted"] if br else None,
                     "kappa3_E_inv": p3["E_inv"], "kappa3_E_inv_weighted": e3w, "share": (e3w / tot) if tot else None,
                     "kappa3_collar_flow_weighted": 48.0 * (p3.get("collar_flow_acc") or 0.0), "set_collar_flow_weighted": r.get("collar_flow_acc_weighted"),
                     "kappa3_S_esc_weighted": 48.0 * (p3.get("S_esc") or 0.0)})
    # (1) the trace: a row whose kappa 3 weighted E_inv rounds to 6.55, against a total of about 106 in some stored convention
    tr = None
    for x in rows:
        if abs(x["kappa3_E_inv_weighted"] - 6.55) < 0.005:
            tr = dict(x); tr["change_against_production"] = (x["E_inv_weighted_set"] - x["E_inv_weighted_production"]) if x["E_inv_weighted_production"] is not None else None
            break
    if tr:
        txt = ("(#62 item 3's \"6.55 of about 106\" traced: the stored row at T/r_s %s, the kappa 3 channel's weighted E_inv %s (weights 16 kappa), equal to the set's "
               "weighted E_inv change against the production there (%s against %s, change %s), set against the charge E (%s at that row; 106.2 at 40), not against "
               "the excitation; the guard's share of the weighted excitation %s at T/r_s %s, %s at 40)" % (
                   V1.p4(tr["t_over_rs"]), V1.p4(tr["kappa3_E_inv_weighted"]), V1.p4(tr["E_inv_weighted_set"]), V1.p4(tr["E_inv_weighted_production"]),
                   V1.p4(tr["change_against_production"]), V1.p4(tr["E"]), V1.p4(tr["share"]), V1.p4(tr["t_over_rs"]), V1.p4(rows[-1]["share"])))
    else:
        txt = "(#62 item 3's \"6.55 of about 106\" is untraced in the stored rows and prints as no value)"
    n_old = md.count(OLD)
    md = md.replace(OLD, txt)
    J["kappa3_62_trace"] = {"found": bool(tr), "row": tr, "statement": txt, "replaced_v1_parenthetical": n_old}
    # (2) the guard at every checkpoint
    L = ["", "## The kappa 3 channel at every checkpoint row through T/r_s 40 (v6fA_s4k3; weights 16 kappa)", "",
         "| T/r_s | E | the set's weighted E_inv | the production's | kappa 3 weighted E_inv | share | kappa 3 signed collar flow (weighted) | the set's | kappa 3 S_esc (weighted, nats) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % tuple(V1.p4(x[k]) for k in (
            "t_over_rs", "E", "E_inv_weighted_set", "E_inv_weighted_production", "kappa3_E_inv_weighted", "share", "kappa3_collar_flow_weighted",
            "set_collar_flow_weighted", "kappa3_S_esc_weighted")))
    J["kappa3_rows"] = rows
    # (3) the variant rows
    L += ["", "## The two v6j variant rows (version 1b)", ""]
    J["variants_v6j"] = {}
    for n, p, state in VAR:
        pa = os.path.join(base, p)
        if not os.path.exists(pa):
            L.append("- %s: not landed (%s)." % (n, state)); J["variants_v6j"][n] = {"landed": False, "state": state}; continue
        d = json.load(open(pa)); r = V1.at40(d)
        if r is None:
            L.append("- %s: landed without a row at T/r_s 40 (last %s)." % (n, V1.p4(d["rows"][-1]["t_over_rs"]))); continue
        J["variants_v6j"][n] = {"landed": True, "S_col_40": r["S_esc_weighted"], "E_remove_40_weighted": r.get("E_remove_weighted"), "collar_flow_40_weighted": r.get("collar_flow_acc_weighted")}
        L.append("- %s: S_col(40) %s nats; E_remove weighted %s; the signed collar flow weighted %s." % (n, V1.p4(r["S_esc_weighted"]), V1.p4(r.get("E_remove_weighted")), V1.p4(r.get("collar_flow_acc_weighted"))))
    i = md.index("\nGuard (")
    md = md[:i] + "\n".join(L) + "\n" + md[i:]
    md = md.replace("# SCOL_attribution_v1:", "# SCOL_attribution_v1b:", 1).replace("by code/page/compose_scol_attribution_v1.py", "by code/page/compose_scol_attribution_v1b.py (on compose_scol_attribution_v1.py; engine spec 44.2)", 1)
    md = md[:md.index("\nGuard (")] + "\n"
    J.pop("guard", None)
    cm, cj = V7B.counts(md), V7B.counts(json.dumps(J))
    md = md + ( "\nGuard (requirements_new.txt sec. 4's and D15's words, counted, not quoted): md %d and %d, json %d and %d.\n" % (cm["sec4"], cm["D15"], cj["sec4"], cj["D15"]))
    J["script"] = "code/page/compose_scol_attribution_v1b.py"; J["guard"] = {"md": cm, "json": cj}
    open(p_md, "w", encoding="utf-8").write(md)
    with open(p_json, "w") as f:
        json.dump(J, f, indent=1)
    print("SCOL v1b: trace %s; rows %d; variants %s; guard %s %s" % ("found at %s" % tr["t_over_rs"] if tr else "untraced", len(rows), {k: v.get("landed") for k, v in J["variants_v6j"].items()}, cm, cj))
    if cm["sec4"] or cm["D15"] or cj["sec4"] or cj["D15"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
