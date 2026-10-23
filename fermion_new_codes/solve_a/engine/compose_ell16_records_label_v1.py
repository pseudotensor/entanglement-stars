#!/usr/bin/env python
"""compose_ell16_records_label_v1.py (builder analyst; engine spec 29.1 and 29.11: "COLLAPSE_records_v1 for the ell 16 record with E2 and the
indicator explained ... its record line's 'indicator fail from T/rs 16.25' (the brief) is explained in the records' head (which indicator, its
definition, where it fails)"): reads E2 from the lead's FILE 1 (COLLAPSE_records_v1.1_labels_turn37.md, the paragraph after "## E2.",
verbatim) and fills its two [BUILDER...] blanks from the run's own json, never retyped: (1) the indicator: engine_collapse_v25's indicator
(engine 18.1 for the initial-value solve: B_estimate = delta_rtol, the dense fields' largest change on a re-solve at rtol 1e-10, atol 1e-14
against the run's 1e-12, 1e-16, plus the source splines' error; the record passes when B_estimate is below 1e-9), with the count of records
that fail, the first, the largest B_estimate and where, and the count whose collar screen is flagged; (2) the record count, with the count
of records whose n_pocket sum is positive. Writes the filled paragraph (the label text for engine_collapse_records_v1 --label). Args:
--labels --run-json --out."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--run-json", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = open(a.labels).read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("## E2.")][0]
    j = i + 1
    while not lines[j].strip():
        j += 1
    para = lines[j]
    assert para.count("[BUILDER") == 2, para.count("[BUILDER")
    J = json.load(open(a.run_json)); rs = float(J["args"]["rs"])
    recs = [r for r in J["records"] if r.get("E") is not None]
    ind = [(float(r["T"]) / rs, r.get("indicator") or {}) for r in recs]
    fail = [(t, x) for t, x in ind if x.get("indicator_pass") is False]
    worst = max(fail, key=lambda tx: tx[1].get("B_estimate") or 0.0) if fail else None
    flagged = sum(1 for t, x in ind if (x.get("collar_label") == "flagged"))

    def npk(r):
        q = r.get("n_pocket")
        return sum(q.values()) if isinstance(q, dict) else int(q or 0)
    n_pk = sum(1 for r in recs if npk(r) > 0)
    fill1 = ("the field solve's tolerance indicator of engine 18.1, B_estimate = delta_rtol (the dense fields' largest change on a re-solve at rtol 1e-10, atol 1e-14 against the run's 1e-12, 1e-16) "
             "plus the source splines' error, the record passing below 1e-9: it fails on %d of %d records, the first at T/rs %.3f, the largest B_estimate %.3e at T/rs %.3f; the collar screen is flagged on %d records "
             "(a numerical tolerance flag of the field solve at the 1e-9 level, not a physical statement)") % (
        len(fail), len(recs), fail[0][0] if fail else float("nan"), worst[1].get("B_estimate") if worst else float("nan"), worst[0] if worst else float("nan"), flagged)
    fill2 = "%d" % len(recs) if n_pk == 0 else "%d (the builder's count: %d with a positive pocket count)" % (len(recs), n_pk)
    out = para.replace("[BUILDER: the indicator's definition and what fails]", fill1, 1).replace("[BUILDER: the record count]", fill2, 1)
    assert "[BUILDER" not in out
    open(a.out, "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
