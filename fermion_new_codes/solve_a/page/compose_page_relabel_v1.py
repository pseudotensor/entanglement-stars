#!/usr/bin/env python
"""compose_page_relabel_v1.py (builder analyst, 2026-10-16; engine spec 38.1 (2) and 38.0): the first draft PAGE_solved_history_v1 (md,
summary json) and its figure's SOURCES file under the lead's added label, verbatim, "A to T/r_s 40, before most of the formation's emission has
reached the collar; C, the ledger run and the variants pending", with the wording of requirements_new.txt sec. 4 applied by literal replacement
(the replacement counts printed in the md): "frozen" prints only as "held fixed" or "fixed-field" ("the freeze" as "the hold's start"),
"window" in the run's sense ("over the window") as "over the run" (a window observable, the 13 r_s window of readout 9, is kept), and
"prediction" never. The numbers are v1's, unchanged (no recomputation); the json's string values are replaced the same way, its keys and
numbers unchanged. v1's files stay as they are (committed); this writes new names; the figure and its SOURCES file are re-rendered from
the new json by item7_fig_page_draft_v2.py. The counts print by the replacing words (the replaced ones are not printed). Usage:
python compose_page_relabel_v1.py IN_MD IN_JSON OUT_MD OUT_JSON"""
import hashlib
import json
import sys

LABEL = "A to T/r_s 40, before most of the formation's emission has reached the collar; C, the ledger run and the variants pending"
REPL = [("its frozen-record controls", "its fixed-field controls"), ("The frozen-record controls", "The fixed-field controls"),
        ("frozen-record", "fixed-field"), ("the frozen h's eigenbasis", "the held-fixed h's eigenbasis"), ("frozen h", "held-fixed h"),
        ("Frozen", "Fixed-field"), ("frozen", "fixed-field"), ("the freeze", "the hold's start"), ("over the window", "over the run"),
        ("predictions", "statements"), ("prediction", "statement")]


def replace_all(s, counts):
    for a, b in REPL:
        n = s.count(a)
        if n:
            counts[b] = counts.get(b, 0) + n                                  # counted by the replacing words
            s = s.replace(a, b)
    return s


def walk(o, counts):
    if isinstance(o, dict):
        return {k: walk(v, counts) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(x, counts) for x in o]
    if isinstance(o, str):
        return replace_all(o, counts)
    return o


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    in_md, in_json, out_md, out_json = sys.argv[1:5]
    c_md = {}; c_js = {}
    md = replace_all(open(in_md, encoding="utf-8").read(), c_md)
    lines = md.split("\n")
    head = ["", "**%s.** (Engine spec 38.1 (2): the first draft stands under this label.)" % LABEL, "",
            "Relabelled by code/page/compose_page_relabel_v1.py from %s (md5 %s); the numbers are v1's, unchanged. Wording (requirements_new.txt sec. 4, applied by literal replacement; counted by the replacing words): %s." % (
                in_md, md5(in_md), "; ".join("\"%s\" %d" % (k, v) for k, v in c_md.items()) or "no replacement")]
    lines = lines[:1] + head + lines[1:]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    J = walk(json.load(open(in_json)), c_js)
    J["draft_label"] = LABEL
    J["relabelled_from"] = {"md": [in_md, md5(in_md)], "json": [in_json, md5(in_json)], "replacements_json": c_js, "replacements_md": c_md}
    if isinstance(J.get("figure"), dict):
        J["figure"]["draft_label"] = LABEL
    with open(out_json, "w") as f:
        json.dump(J, f, indent=1)
    print("wrote %s and %s; replacements (by the replacing words) md %s, json %s" % (out_md, out_json, c_md, c_js))


if __name__ == "__main__":
    main()
