#!/usr/bin/env python
"""rd4_walks_summary_v1.py (busi lane, PERF RD-4 self-test (b)): the per-walk replay's json (rd4_walks_v1.py) summarized: per walk the ladder's
pieces, failure probes (the crossings' s), crossing windows (piece [s_lo, s_hi]) and swaps, the tabulated arms' identity, the verdict tiers and the
Cholesky identity (the ladder's Cholesky count = the tabulated verdicts + the reopening tests, the latter being the tabulated route's Cholesky
count minus its tier-3 calls); the totals; the ladder against the stored record (information).
Usage. python rd4_walks_summary_v1.py rd4_walks_v1.json"""
import json
import sys


def main():
    J = json.load(open(sys.argv[1])); rows = J["rows"]
    tot = {"walks": 0, "identical": 0, "chol_identity": 0, "lad_chol": 0, "tab_chol": 0, "aud_dis": 0, "table": 0, "count": 0, "chol3": 0, "stored_pieces_eq": 0, "stored_n": 0,
           "lad_s": 0.0, "tab_s": 0.0, "table_s": 0.0, "crossings": 0, "swaps": 0, "end_fail": 0, "count_fail": 0}
    for r in rows:
        st = r["TAB"]["stats"]; lad = r["LAD"]
        ver = int(st.get("verdict_table", 0) + st.get("verdict_count", 0) + st.get("verdict_cholesky", 0))
        reopen = int(r["TAB"]["cholesky"] or 0) - int(st.get("verdict_cholesky", 0))
        ident = (ver + reopen == int(lad["cholesky"] or 0))
        same = r["TAB"]["n_differences"] == 0 and r["AUD"]["n_differences"] == 0
        dis = int(sum(v for k, v in r["AUD"]["stats"].items() if k.startswith("audit_disagree")))
        tot["walks"] += 1; tot["identical"] += int(same); tot["chol_identity"] += int(ident); tot["lad_chol"] += int(lad["cholesky"] or 0)
        tot["tab_chol"] += int(r["TAB"]["cholesky"] or 0); tot["aud_dis"] += dis; tot["table"] += int(st.get("verdict_table", 0)); tot["count"] += int(st.get("verdict_count", 0))
        tot["chol3"] += int(st.get("verdict_cholesky", 0)); tot["lad_s"] += lad["wall"]; tot["tab_s"] += r["TAB"]["wall"]; tot["table_s"] += float(st.get("table_s", 0.0))
        tot["end_fail"] += int(st.get("table_end_check_fail", 0)); tot["count_fail"] += int(st.get("segment_count_check_fail", 0))
        cr = lad["crossings"]; tot["crossings"] += len(cr); tot["swaps"] += sum(1 for c in cr if c["swap"])
        if r["stored"]["pieces"] is not None:
            tot["stored_n"] += 1; tot["stored_pieces_eq"] += int(r["stored"]["pieces"] == lad["pieces"])
        if cr or not same or not ident or dis:
            print("T %.5f k%d: pieces %d, cholesky %d; crossings %s; TAB/AUD %s; verdicts table %d count %d cholesky %d + reopen %d = %d (ladder %d: %s); audit disagreements %d" % (
                r["T"], r["kappa"], lad["pieces"], lad["cholesky"],
                ["s %.6f window [%.6f, %.6f] swap %s" % (c["s"], c["piece"][0], c["piece"][1], c["swap"]) for c in cr], "identical" if same else "DIFFER",
                st.get("verdict_table", 0), st.get("verdict_count", 0), st.get("verdict_cholesky", 0), reopen, ver + reopen, lad["cholesky"], "equal" if ident else "UNEQUAL", dis))
    print("TOTAL %d walks: %d identical (TAB and AUD against LAD, every key but seconds and Cholesky counts); Cholesky identity %d of %d; audit disagreements %d; "
          "crossings %d (swaps %d); ladder Cholesky %d -> tabulated %d (tier 1 %d, tier 2 %d, tier 3 %d verdicts); end-check fails %d, count-check fails %d; "
          "walls ladder %.1f s, tabulated %.1f s (of which tables %.1f s); stored pieces equal on %d of %d" % (
              tot["walks"], tot["identical"], tot["chol_identity"], tot["walks"], tot["aud_dis"], tot["crossings"], tot["swaps"], tot["lad_chol"], tot["tab_chol"],
              tot["table"], tot["count"], tot["chol3"], tot["end_fail"], tot["count_fail"], tot["lad_s"], tot["tab_s"], tot["table_s"], tot["stored_pieces_eq"], tot["stored_n"]))
    ok = tot["identical"] == tot["walks"] and tot["chol_identity"] == tot["walks"] and tot["aud_dis"] == 0
    print("%s (b) per walk" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
