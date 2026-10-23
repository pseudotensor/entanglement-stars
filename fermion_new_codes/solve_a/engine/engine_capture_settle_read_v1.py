#!/usr/bin/env python
"""engine_capture_settle_read_v1.py (builder analyst; engine spec 34.4 (2): "a stop counts as settled only if the inward-branch weight on
[R_t + 2, R_max) is below 10 percent of sigma at the stop; otherwise the row reads 'stopped by the rule at T r_s with [x] of sigma in
transit: not settled'", and (3): "the omega/m 2 row prints 'truncated at the channel cap of 200 (A at the cap [x] at the reading time)', a
lower bound where A at the cap exceeds 1e-3"): the post-hoc reading of C-capture version 7 and 8 jsons, no new propagation. Per background
and probe: the stop (the version 7 rule's), the in-transit inward-branch weight at the stop over sigma there (both summed over the channels
as sigma is), the verdict as 34.4 (2) words it; where the channel list reaches the cap, the largest A of the last channel (both signs) at
the stop and whether it exceeds 1e-3. Every number printed by the script. Args: --json (repeated) --cap (200) --share (0.10) --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--share", type=float, default=0.10)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## C-capture's post-hoc settle reading (engine_capture_settle_read_v1.py; engine spec 34.4 (2), (3))", "",
         "- the in-transit share: the inward-branch weight on [R_t + 2, R_max) summed over the channels as sigma is, over sigma, at the version 7 rule's stop; a stop counts as settled only with that share below %g (34.4 (2)); where the channel list reaches the cap of %d, the last channel's A (the larger sign) at the stop, the row a lower bound where it exceeds 1e-3 (34.4 (3))." % (a.share, a.cap), "",
         "| run | background | probe | stop (the version 7 rule) | sigma/(pi b_c(omega, m)^2) at the stop | in transit over sigma | the reading (34.4 (2)) | channels | at the cap: A of the last channel |",
         "|---|---|---|---|---|---|---|---|---|"]
    for jp in a.json:
        d = json.load(open(jp))
        run = os.path.basename(jp)
        for r in d["results"]:
            st = r["settle"]
            T = r["T_list"][-1]
            key = "%.4f" % T
            sig = r["sigma"][key]
            tr = r["inward_transit_summed"][key]
            share = tr / sig if sig > 0 else float("nan")
            ratio = sig / r["norm_ratio"]
            Tr = st["T_over_rs"] if st["settled"] else st["at_T_over_rs"]
            if st["settled"]:
                verdict = "settled (share below %g)" % a.share if share < a.share else "stopped by the rule at T %g r_s with %.3f of sigma in transit: not settled" % (Tr, share)
            else:
                verdict = "not settled within %g r_s (the rule)" % Tr
            nch = len(r["channels"])
            capcell = "-"
            if nch >= a.cap:
                last = r["channels"][-1]
                Avals = [x.get("A", {}).get(key) for x in (last["plus"], last["minus"])]
                Avals = [v for v in Avals if v is not None]
                Am = max(Avals) if Avals else None
                capcell = ("truncated at the channel cap of %d (A at the cap %.3e at the reading time)%s" % (a.cap, Am, ", a lower bound" if Am > 1e-3 else "")) if Am is not None else "the last channel excluded at launch"
            L.append("| %s | %s | %s | %s | %.4f | %.4f | %s | %d | %s |" % (run, r["background"], r["probe"], ("settled at T %g r_s" % Tr) if st["settled"] else ("none to %g r_s" % Tr),
                                                                       ratio, share, verdict, nch, capcell))
    L += ["", "- every number above is printed by the script from the runs' jsons: %s." % "; ".join(a.json)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
