#!/usr/bin/env python
"""compose_ell8_records_label_v2.py (builder analyst; engine spec 29.11 and FILE 1 of turn 37: "E1 appended to E"): version 1's filled section
E (its three blanks filled from the run's json, the ops check and the records summary; version 1's output, read verbatim, without
version 1's own builder's note), then the lead's E1 (COLLAPSE_records_v1.1_labels_turn37.md, the paragraph after "## E1", verbatim), then a
builder's note naming the replay whose exactness table the record now carries (--replay-note). Args: --e-filled (version 1's output) --t37
--replay-note --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-filled", required=True); ap.add_argument("--t37", required=True); ap.add_argument("--replay-note", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = open(a.e_filled).read().strip()
    note1 = re.search(r" \(Builder: [^)]*\)\.?\s*$", e)
    if note1:
        e = e[:note1.start()].rstrip()
    m = re.search(r"^## E1 [^\n]*\n(.*?)(?=^## |\Z)", open(a.t37).read(), re.S | re.M)
    assert m, "E1 not found"
    out = e + " " + m.group(1).strip() + " (Builder: " + a.replay_note.strip() + ")"
    open(a.out, "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
