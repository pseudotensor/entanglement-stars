#!/usr/bin/env python
"""compose_scol_attribution_v1.py (the lane's analyst builder; engine spec 42.3 (3), 43.5 (4); inbox #62 item 3): SCOL_attribution_v1, the
variant attribution of the selected archive's entropy at T/r_s 40 on the rows present, composed from the stored run summaries only (no run).
Rows: the production (v6bA_s4); eps_loc 1e-3 (v6dA_s4loc3, the threshold); the kappa 3 set (v6fA_s4k3, the channel set; the kappa 3 channel's
weighted E_inv at 40 and its share printed as the truncation weight); the far collector (v6dA_s4far, R_cut 48 r_s, collar 32 r_s) labelled
'not yet arrived' at 40, its matched arrival clock about T/r_s 74 (engine spec 42.3 (1), the lead's arithmetic by F(48) - F(24)); the sigma/a 8
history (v6fA_s8) beside as a different object. The normalized selector (v6jA_s4gen) and the cubic window (v6jA_s4cub) rows enter version 1b as
they land. Columns at T/r_s 40: S_col (S_esc_weighted); the collected-energy ledger per channel (E_remove per channel, and weighted 16 kappa);
the signed collar flow through 40 (collar_flow_acc, per channel and weighted); the Killing spectrum by bin (the collected blocks' excess
occupation per Killing bin, bins_occupation, summed over every collection through 40 and weighted 16 kappa). The spread of S_col over eps_loc
1e-3 and the kappa 3 set is printed as the digits' error; each other readout is printed at 4 significant digits and stated as left at the
printed digits by a variant when its printed value equals the production's, or not, as found. The label line is FILE 1 item 1 of
ITEM7_light_channel_labels_v1.md with its brackets filled as page_solved_history_report_v7b fills them; sec. 4's and D15's words counted on
the outputs and printed as counts (exit 4 if nonzero).
Usage: python compose_scol_attribution_v1.py RESULTS_PAGE LABELS_MD OUT_MD OUT_JSON"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_solved_history_report_v7b as V7B     # noqa: E402  (FILE 1's reader and counts)

ROWS = [("s4", "v6bA_runs/v6bA_s4/v6bA_s4_summary.json", "the production (page_history_v6b; eps_loc 1e-2, channels 1 and 2, collector at 24 r_s)", "row"),
        ("s4loc3", "v6bA_runs/v6dA_s4loc3/v6dA_s4loc3_summary.json", "eps_loc 1e-3: the threshold (page_history_v6d)", "spread"),
        ("s4k3", "v6bA_runs/v6fA_s4k3/v6fA_s4k3_summary.json", "the kappa 3 channel set: the channel set (page_history_v6f)", "spread"),
        ("s4far", "v6bA_runs/v6dA_s4far/v6dA_s4far_summary.json", "the far collector (R_cut 48 r_s, collar 32 r_s; page_history_v6d): not yet arrived at 40", "labelled"),
        ("s8", "v6bA_runs/v6fA_s8/v6fA_s8_summary.json", "the sigma/a 8 history (page_history_v6f): a different object, beside", "beside")]
LATER = [("s4gen", "the normalized selector (page_history_v6j --collector-select gen, A's settings to 40)"),
         ("s4cub", "the cubic window of 8 r_s over 36 to 40 (page_history_v6j --cubic-window 32,40 from T = 0, labelled a variant)")]
FAR_CLOCK = "about T/r_s 74 (engine spec 42.3 (1), the lead's arithmetic: F(48) - F(24) about 29 r_s after the near collector's 45)"


def p4(x):
    return "-" if x is None else ("%.4g" % x)


def at40(d):
    for r in d["rows"]:
        if abs(r["t_over_rs"] - 40.0) < 1e-9:
            return r
    return None


def spectrum(d, chans=("1", "2")):
    """the collected blocks' excess per (sign, bin) summed over every collection through 40, weighted 16 kappa, on the common channels 1 and 2
    (the kappa 3 channel's own part is printed apart as the truncation's)."""
    sp = {}
    for r in d["rows"]:
        if r["t_over_rs"] > 40.0 + 1e-9:
            continue
        for k, pc in r["per_channel"].items():
            if k not in chans:
                continue
            for b in pc.get("bins_occupation") or []:
                key = (int(b["sign"]), int(b["bin"]))
                e = sp.setdefault(key, {"lo_TH": b["lo_TH"], "hi_TH": b["hi_TH"], "excess": 0.0})
                e["excess"] += 16.0 * int(k) * float(b["excess"])
    return sp


def main():
    base, p_lab, p_md, p_json = sys.argv[1:5]
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s4 = json.load(open(os.path.join(base, ROWS[0][1])))
    items = V7B.file1_items(p_lab)
    lab, vals, sel = V7B.fill_label({"A_s4": {"rows": s4["rows"]},
                                     "figure": {"marks": {"v2": {"collar_flow_total": at40(s4)["collar_flow_acc_weighted"]}}}}, items[1])
    out = {"script": "code/page/compose_scol_attribution_v1.py", "written": now, "label": lab, "label_brackets": vals, "rows": {}, "later_rows": dict(LATER)}
    srcs = []
    for n, p, what, role in ROWS:
        pa = os.path.join(base, p)
        if not os.path.exists(pa):
            out["rows"][n] = {"what": what, "role": role, "landed": False}; continue
        raw = open(pa, "rb").read(); d = json.loads(raw)
        srcs.append("results/page/%s (md5 %s)" % (p, hashlib.md5(raw).hexdigest()))
        r = at40(d)
        ch = {k: {"E_remove": pc.get("E_remove"), "collar_flow_acc": pc.get("collar_flow_acc"), "E_inv": pc.get("E_inv"), "S_esc": pc.get("S_esc")}
              for k, pc in r["per_channel"].items()}
        sp = spectrum(d)
        sp3 = spectrum(d, ("3",)) if "3" in r["per_channel"] else {}
        out["rows"][n] = {"what": what, "role": role, "landed": True, "code": d.get("code"), "channels": d["settings"].get("channels"),
                          "R_cut_rs": d["settings"].get("R_cut_rs"), "R_collar_rs": d["settings"].get("R_collar_rs"), "eps_loc": d["settings"].get("eps_loc"),
                          "E_40": r["E"], "S_col_40": r["S_esc_weighted"], "E_inv_40_weighted": r["E_inv_weighted"], "E_remove_40_weighted": r.get("E_remove_weighted"),
                          "collar_flow_40_weighted": r.get("collar_flow_acc_weighted"), "per_channel": ch,
                          "collar_flow_40_weighted_k12": sum(16.0 * int(k) * (ch[k]["collar_flow_acc"] or 0.0) for k in ("1", "2")),
                          "spectrum_kappa3_total": sum(v["excess"] for v in sp3.values()) if sp3 else None,
                          "spectrum": [{"sign": k[0], "bin": k[1], "lo_TH": v["lo_TH"], "hi_TH": v["hi_TH"], "excess_weighted": v["excess"]} for k, v in sorted(sp.items())]}
    R = out["rows"]
    k3 = R["s4k3"]["per_channel"].get("3") if R["s4k3"].get("landed") else None
    trunc = None
    if k3:
        e3 = 48.0 * k3["E_inv"]
        trunc = {"kappa3_E_inv_40_weighted": e3, "E_inv_40_weighted_with_kappa3": R["s4k3"]["E_inv_40_weighted"], "share": e3 / R["s4k3"]["E_inv_40_weighted"],
                 "kappa3_S_esc_40_weighted": 48.0 * k3["S_esc"], "E_inv_change_against_production": R["s4k3"]["E_inv_40_weighted"] - R["s4"]["E_inv_40_weighted"],
                 "inbox_62_quote": "its E_inv change 6.55 of about 106 (#62 item 3; another clock or reading, not recomputed here)"}
    out["truncation_kappa3"] = trunc
    spr = [R[k]["S_col_40"] for k in ("s4loc3", "s4k3") if R[k].get("landed")]
    out["digits_error_S_col_40"] = [min(spr), max(spr)] if spr else None
    # left at the printed digits, per readout, against the production
    base_r = R["s4"]
    left = {}
    for n in ("s4loc3", "s4k3", "s4far"):
        if not R[n].get("landed"):
            continue
        L_ = {}
        for c in ("1", "2"):
            L_["E_remove_k%s" % c] = p4(R[n]["per_channel"][c]["E_remove"]) == p4(base_r["per_channel"][c]["E_remove"])
            L_["collar_flow_k%s" % c] = p4(R[n]["per_channel"][c]["collar_flow_acc"]) == p4(base_r["per_channel"][c]["collar_flow_acc"])
        L_["collar_flow_weighted"] = p4(R[n]["collar_flow_40_weighted_k12"]) == p4(base_r["collar_flow_40_weighted_k12"])
        bs = {(b["sign"], b["bin"]): p4(b["excess_weighted"]) for b in base_r["spectrum"]}
        vs = {(b["sign"], b["bin"]): p4(b["excess_weighted"]) for b in R[n]["spectrum"]}
        L_["spectrum_bins_equal"] = sum(1 for k in bs if vs.get(k) == bs[k]); L_["spectrum_bins"] = len(set(bs) | set(vs))
        L_["S_col"] = p4(R[n]["S_col_40"]) == p4(base_r["S_col_40"])
        left[n] = L_
    out["left_at_printed_digits"] = left
    # the md
    M = ["# SCOL_attribution_v1: the variant attribution of the selected archive's entropy at T/r_s 40 (engine spec 42.3 (3), 43.5 (4); inbox #62 item 3)", "",
         "Composed %s by code/page/compose_scol_attribution_v1.py from the stored summaries (no run): %s. Every number is copied from those files or "
         "summed from them as stated; printed at 4 significant digits." % (now, "; ".join(srcs)), "",
         "Label on every number below: **%s**" % lab, "",
         "## Rows and columns at T/r_s 40", "",
         "| row | what | channels | collector R_cut, collar (r_s) | S_col (nats) | E_remove weighted | E_remove kappa 1; 2 | signed collar flow weighted | flow kappa 1; 2 | E_inv weighted |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, _, what, role in ROWS:
        r = R[n]
        if not r.get("landed"):
            M.append("| %s | %s | not landed | | | | | | | |" % (n, what)); continue
        pc = r["per_channel"]
        M.append("| %s | %s | %s | %s, %s | %s | %s | %s; %s | %s | %s; %s | %s |" % (
            n, what, r["channels"], p4(r["R_cut_rs"]), p4(r["R_collar_rs"]), p4(r["S_col_40"]), p4(r["E_remove_40_weighted"]), p4(pc["1"]["E_remove"]), p4(pc["2"]["E_remove"]),
            p4(r["collar_flow_40_weighted"]), p4(pc["1"]["collar_flow_acc"]), p4(pc["2"]["collar_flow_acc"]), p4(r["E_inv_40_weighted"])))
    for n, what in LATER:
        M.append("| %s | %s | version 1b, as it lands | | | | | | | |" % (n, what))
    M += ["", "The far collector's row is labelled not yet arrived: its matched arrival clock is %s; A to 100 reads it only in rows labelled 'after the chain end's "
          "first return' (from 58), and A2 on the doubled chain reads it clean, in version 1b or 3b." % FAR_CLOCK, ""]
    if trunc:
        M.append("The kappa 3 set's truncation weight at 40: the kappa 3 channel's weighted E_inv %s of %s with it (share %s); its weighted S_esc %s nats; the set's E_inv change "
                 "against the production %s (%s)." % (p4(trunc["kappa3_E_inv_40_weighted"]), p4(trunc["E_inv_40_weighted_with_kappa3"]), p4(trunc["share"]),
                                                      p4(trunc["kappa3_S_esc_40_weighted"]), p4(trunc["E_inv_change_against_production"]), trunc["inbox_62_quote"]))
    if spr:
        M.append("\nThe digits' error: S_col at 40 is %s nats on the production; over eps_loc 1e-3 and the kappa 3 set it spans %s to %s nats." % (
            p4(R["s4"]["S_col_40"]), p4(min(spr)), p4(max(spr))))
    M += ["", "## The Killing spectrum by bin (the collected blocks' excess occupation per Killing bin, summed over every collection through 40, weighted 16 kappa, channels 1 and 2)", "",
          "The kappa 3 channel's own collected excess summed over its bins: %s (the truncation's part, apart from the table)." % p4(R["s4k3"].get("spectrum_kappa3_total")), "",
          "| sign | bin (T_H) | " + " | ".join(n for n, _, _, _ in ROWS if R[n].get("landed")) + " |", "|---|---|" + "---|" * sum(1 for n, _, _, _ in ROWS if R[n].get("landed"))]
    keys = sorted({(b["sign"], b["bin"], b["lo_TH"], b["hi_TH"]) for n, _, _, _ in ROWS if R[n].get("landed") for b in R[n]["spectrum"]})
    for sgn, b, lo, hi in keys:
        cells = []
        for n, _, _, _ in ROWS:
            if not R[n].get("landed"):
                continue
            v = [x["excess_weighted"] for x in R[n]["spectrum"] if x["sign"] == sgn and x["bin"] == b]
            cells.append(p4(v[0]) if v else "-")
        M.append("| %s | %s to %s | %s |" % ("+" if sgn > 0 else "-", p4(lo), p4(hi), " | ".join(cells)))
    M += ["", "## Left at the printed digits (against the production, 4 significant digits)", ""]
    for n, L_ in left.items():
        M.append("- %s: S_col %s; E_remove kappa 1 %s, kappa 2 %s; the signed collar flow kappa 1 %s, kappa 2 %s, weighted over channels 1 and 2 %s; the spectrum's bins (channels 1 and 2) equal %d of %d." % (
            n, *["left" if L_[k] else "not left" for k in ("S_col", "E_remove_k1", "E_remove_k2", "collar_flow_k1", "collar_flow_k2", "collar_flow_weighted")],
            L_["spectrum_bins_equal"], L_["spectrum_bins"]))
    M += ["", "The reading (#62 item 3's form, stated as found above): the selected archive's entropy is a collector-defined quantity whose digits carry the spread "
          "printed as their error, while the collected-energy ledger per channel, the signed collar flow and the Killing spectrum by bin are left at the printed digits "
          "by a variant or not as the list above states.", ""]
    md = "\n".join(M) + "\n"
    cm, cj = V7B.counts(md), V7B.counts(json.dumps(out))
    out["guard"] = {"md": cm, "json": cj}
    md += "\nGuard (requirements_new.txt sec. 4's and D15's words, counted, not quoted): md %d and %d, json %d and %d.\n" % (cm["sec4"], cm["D15"], cj["sec4"], cj["D15"])
    open(p_md, "w", encoding="utf-8").write(md)
    with open(p_json, "w") as f:
        json.dump(out, f, indent=1)
    print("SCOL_attribution_v1: rows %s; digits' error %s; left %s; guard md %s json %s" % (
        [n for n in R if R[n].get("landed")], out["digits_error_S_col_40"], {n: (L_["S_col"], L_["collar_flow_weighted"], L_["spectrum_bins_equal"]) for n, L_ in left.items()}, cm, cj))
    if cm["sec4"] or cm["D15"] or cj["sec4"] or cj["D15"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
