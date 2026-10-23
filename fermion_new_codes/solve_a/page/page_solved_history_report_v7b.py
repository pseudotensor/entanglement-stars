#!/usr/bin/env python
"""page_solved_history_report_v7b.py (the lane's analyst builder; engine spec 43.3 (b), (c); turn 51's FILE 1
ITEM7_light_channel_labels_v1.md): the composer of PAGE_solved_history version 3 with turn 51's labels. It runs
page_solved_history_report_v7.py unchanged and then, on its outputs only (no stored run json changes):
(1) the label line: FILE 1 item 1, copied verbatim from the delivered file, its brackets filled from the composer's own json (version 2's run
    v6bA_s4, the digits' range): [value] = the net signed collar flow through T/r_s 40 over E(40) (the figure block's collar_flow_total, as
    the paragraph's fill reads it); [T_1] to [T_2] = the run's first and last checkpoint clocks through 40; [range] = the collected modes'
    outside weight per block (outside_weight_min over the blocks with a collection), in place of the inherited label and its owner-decision
    clause, in the md and in the json (label, the figure block's label, A to 100's label);
(2) deleted (FILE 1 item 4): the inherited line on the quantity owner decision D15 removed, wherever it occurs in the md (the header's filter
    clause, the label's second sentence, the A.12 line's closing clause) and every label_d15 key of the json; the retained-channel definition that shared that line
    (S(M) - S(M') as minus the conditional entropy of each collected batch) is kept as its own sentence;
(3) the guard: the words of requirements_new.txt sec. 4 (the list the lane's composers use) and of D15 counted on the md and the json and
    printed as counts, never quoted; exit 4 if any count is nonzero.
Usage: python page_solved_history_report_v7b.py RESULTS_PAGE OUT_MD OUT_JSON LABELS_MD [--draft-label TEXT]"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_solved_history_report_v7 as V7     # noqa: E402

SEC4 = ("frozen", "freeze", "prediction")          # the sec. 4 words the lane's compose scripts guard (compose_page_v3_final_v1.sh)
D15W = ("mutual information", "mutual-information", "D15")
OLD_D15_HEADER = ", fetched through the no-mutual-information filter, #41 D15)"
OLD_D15_A12 = ", another builder's; no mutual information is printed here."
DEF_KEEP = "S(M) - S(M') = S_old - S_new is minus the conditional entropy S(B|M) of each collected batch given the archive"


def file1_items(p):
    """the quoted text of each numbered item of the delivered labels file, verbatim."""
    t = open(p, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'^(\d+)\. (.*)$', t, re.M):
        line = m.group(2)
        i, j = line.find('"'), line.rfind('"')
        out[int(m.group(1))] = line[i + 1:j] if (i >= 0 and j > i) else line
    return out


def counts(text):
    lo = text.lower()
    return {"sec4": sum(lo.count(w) for w in SEC4), "D15": sum(lo.count(w.lower()) for w in D15W)}


def drop_key(x, key):
    if isinstance(x, dict):
        return {k: drop_key(v, key) for k, v in x.items() if k != key}
    if isinstance(x, list):
        return [drop_key(v, key) for v in x]
    return x


def fill_label(J, item1):
    rows = J["A_s4"]["rows"]
    upto = [r for r in rows if r["t_over_rs"] <= 40.0 + 1e-9]
    r40 = [r for r in upto if abs(r["t_over_rs"] - 40.0) < 1e-9][0]
    flow = J["figure"]["marks"]["v2"]["collar_flow_total"]
    ow = [p["outside_weight_min"] for r in upto for p in r["per_channel"].values() if p.get("outside_weight_min") is not None and p.get("collected")]
    vals = {"[value]": "%.4e" % (flow / r40["E"]), "[T_1]": "%g" % upto[0]["t_over_rs"], "[T_2]": "%g" % r40["t_over_rs"],
            "[range]": "%.4f to %.4f" % (min(ow), max(ow))}
    lab = item1
    for k, v in vals.items():
        if k not in lab:
            sys.exit("bracket %s not in FILE 1 item 1" % k)
        lab = lab.replace(k, v, 1)
    sel = {"[value]": "the net signed collar flow through T/r_s 40, %.9g, over E(40) %.9g (version 2's run v6bA_s4)" % (flow, r40["E"]),
           "[T_1]": "the run's first checkpoint clock", "[T_2]": "its checkpoint at T/r_s 40 (the digits' range)",
           "[range]": "outside_weight_min over the %d blocks with a collection through 40" % len(ow)}
    return lab, vals, sel


def main():
    args = sys.argv[1:]
    base, p_md, p_json, p_lab = args[:4]
    extra = args[4:]
    sys.argv = [sys.argv[0], base, p_md, p_json] + extra
    V7.main()
    md = open(p_md, encoding="utf-8").read()
    J = json.load(open(p_json))
    items = file1_items(p_lab)
    lab, vals, sel = fill_label(J, items[1])
    md = md.replace("by code/page/page_solved_history_report_v7.py (on page_solved_history_report_v6.py)",
                    "by code/page/page_solved_history_report_v7b.py (on page_solved_history_report_v7.py and v6.py; turn 51's labels, ITEM7_light_channel_labels_v1)", 1)
    n_hdr = md.count(OLD_D15_HEADER)
    md = md.replace(OLD_D15_HEADER, ")")
    n_a12 = md.count(OLD_D15_A12)
    md = md.replace(OLD_D15_A12, ", another builder's.")
    lines = md.split("\n")
    n_lab = 0
    for i, ln in enumerate(lines):
        if ln.startswith("Label on every number below: **"):
            keep = (" " + DEF_KEEP + ".") if DEF_KEEP in ln else ""
            lines[i] = "Label on every number below: **%s**%s" % (lab, keep)
            n_lab += 1
    md = "\n".join(lines)
    J["label"] = lab
    J["figure"]["label"] = lab
    if J.get("A_to100"):
        J["A_to100"]["label"] = lab
        J["A_to100"].pop("label_v6i", None)
    J = drop_key(J, "label_d15")
    J["retained_channel_definition"] = DEF_KEEP
    J["script"] = "code/page/page_solved_history_report_v7b.py (on page_solved_history_report_v7.py, v6.py and v1's readers)"
    J["labels_v7b"] = {"file": os.path.basename(p_lab), "item1_filled": lab, "brackets": vals, "selection": sel,
                       "paragraph_extension_slot": items.get(3), "figure_caption": items.get(2)}
    cm, cj = counts(md), counts(json.dumps(J))
    J["guard_v7b"] = {"md": cm, "json": cj, "replaced": {"label_lines": n_lab, "header_filter_clause": n_hdr, "A12_clause": n_a12}}
    open(p_md, "w", encoding="utf-8").write(md)
    with open(p_json, "w") as f:
        json.dump(J, f, indent=1, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    print("v7b: label lines %d, header clause %d; guard counts md sec4 %d D15 %d, json sec4 %d D15 %d" % (n_lab, n_hdr, cm["sec4"], cm["D15"], cj["sec4"], cj["D15"]))
    if cm["sec4"] or cm["D15"] or cj["sec4"] or cj["D15"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
