#!/usr/bin/env python
"""compose_ell8_records_label_v1.py (builder analyst; engine spec 28.6 and 28.10: "The ell 8 control's records are delivered as
COLLAPSE_records_v1 in the set members' form ... with FILE 1's section E, whose blanks BUILDER fills: the record and pocket counts, the ops
consistency and straddling-block counts, the largest |v|/N over the run against 1.2198"): reads section E of the lead's FILE 1
(COLLAPSE_records_v1.1_labels_turn36.md, the paragraph after its "## E." heading, verbatim) and fills its three [BUILDER...] blanks, each
from a file, never retyped: (1) the record count = the run json's records with an E, with the count of records whose n_pocket sum is
positive (if it is not zero the fill says so instead of leaving "none" unqualified); (2) the window-consistency and partly-occupied
(straddling) block counts from the ops check's md table row (engine_ops_check_records_v1), with its (record, channel) pair count; (3) the
largest max |v|/N of the dense fields from the records summary's line (engine_records_summary_v5). Prints the filled paragraph to --out
(the label text for engine_collapse_records_v1 --label) and a builder's note line beside it. Args: --labels --run-json --ops-md --summary-md
--note --out."""
import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--run-json", required=True); ap.add_argument("--ops-md", required=True)
    ap.add_argument("--summary-md", required=True); ap.add_argument("--note", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = open(a.labels).read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("## E.")][0]
    j = i + 1
    while not lines[j].strip():
        j += 1
    para = lines[j]
    assert para.count("[BUILDER") == 3, "section E holds %d blanks" % para.count("[BUILDER")
    J = json.load(open(a.run_json))
    recs = [r for r in J["records"] if r.get("E") is not None]

    def npk(r):
        q = r.get("n_pocket")
        return sum(q.values()) if isinstance(q, dict) else int(q or 0)
    n_pk = sum(1 for r in recs if npk(r) > 0)
    fill1 = "%d" % len(recs) if n_pk == 0 else "%d (the builder's count: %d of them with a positive pocket count, so not none)" % (len(recs), n_pk)
    row = [l for l in open(a.ops_md).read().split("\n") if l.startswith("| c") or l.startswith("| s")]
    assert len(row) == 1, "the ops table has %d run rows" % len(row)
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    head = [l for l in open(a.ops_md).read().split("\n") if l.startswith("| run |")][0]
    hcells = [c.strip() for c in head.strip("|").split("|")]
    col = dict(zip(hcells, cells))
    fill2 = "window consistency below 1 - 1e-12 in %s and a partly occupied block at the cut (straddling) in %s of %s (record, channel) pairs; engine_ops_check_records_v1" % (
        col["window consistency < 1"], col["partly occupied block"], col["records, pairs"].split(",")[1].strip())
    m = [l for l in open(a.summary_md).read().split("\n") if l.startswith("- the largest max |v|/N of the dense fields")]
    assert len(m) == 1
    mm = re.search(r": ([0-9.]+) at T/rs ([0-9.]+)\.", m[0])
    fill3 = "%s (the dense fields, at T/rs %s; engine_records_summary_v5)" % (mm.group(1), mm.group(2))
    out = para.replace("[BUILDER: the record count]", fill1, 1).replace("[BUILDER: the consistency and straddling-block counts]", fill2, 1).replace("[BUILDER]", fill3, 1)
    assert "[BUILDER" not in out
    if a.note:
        out = out + " " + a.note
    open(a.out, "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
