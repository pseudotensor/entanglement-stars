#!/usr/bin/env python
"""engine_v19_check_v3.py (builder analyst; version 3 = version 2 with the bound's fallback: a record whose iteration history yields no
fixed-point estimate in either run (record 0, converged in its first trial) is bounded by the stop rule's field tolerance (the runs'
--stop-fields), labelled. Version 2's docstring: engine spec 22.2: v19's gate on the corrected criterion): the first --records records of a v19
run from T = 0 against the stored run at the same T: |dE/E| below 1e-11 (4 times the stop rule's measured effect, 2.4e-12), the applied
fields' largest difference within 3 fixed-point estimates of either run (19.1's estimate c q/(1 - q) from each record's iteration history,
engine_relabel_v2.estimates, q_obs and q_ref both printed), the occupation masks equal to the negative-energy set in every channel, and the
continuation unsubdivided in both (v17's nsub 1, v19's nsub 1); beside them each run's E0 and record 0's E (whether the record carries E0
or the first re-solve's E). Exit 0 when every criterion holds. Args: --new --ref --records --md."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_relabel_v2 as RL  # noqa: E402


def load(run):
    d, tag = os.path.split(run)
    return tag, json.load(open(os.path.join(d, "collapse_%s.json" % tag))), np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    tn, Jn, Zn = load(a.new); tr, Jr, Zr = load(a.ref)
    rn = [r for r in Jn["records"] if r.get("E") is not None][:a.records]
    rr = {round(r["T"], 6): r for r in Jr["records"] if r.get("E") is not None}
    Tn = np.asarray(Zn["T"], float); Tr = np.asarray(Zr["T"], float)
    E0n, E0r = float(Jn["head"]["E0"]), float(Jr["head"]["E0"])
    L = ["## engine_collapse_v19's gate on the corrected criterion (engine_v19_check_v3.py; engine spec 22.2)", "",
         "New: %s (%s), E0 %.15f, record 0's E %.15f (record 0 minus E0: %+.2e relative); reference: %s (%s), E0 %.15f, record 0's E %.15f (%+.2e relative)." % (
             tn, Jn.get("script"), E0n, rn[0]["E"], (rn[0]["E"] - E0n) / E0n, tr, Jr.get("script"), E0r, rr[round(rn[0]["T"], 6)]["E"],
             (rr[round(rn[0]["T"], 6)]["E"] - E0r) / E0r), "",
         "| T/rs | dE/E | max abs dN_applied | max abs dv_applied | estimate new (q_obs, q_ref) | estimate ref (q_obs, q_ref) | fields within 3 estimates | masks = negative set | nsub new, ref | pass |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for r in rn:
        q = rr.get(round(r["T"], 6))
        if q is None:
            continue
        dE = abs(r["E"] - q["E"]) / abs(q["E"])
        kn = int(np.argmin(np.abs(Tn - r["T"]))); kr = int(np.argmin(np.abs(Tr - r["T"])))
        dNa = float(np.max(np.abs(np.asarray(Zn["N_applied"][kn]) - np.asarray(Zr["N_applied"][kr]))))
        dva = float(np.max(np.abs(np.asarray(Zn["v_applied"][kn]) - np.asarray(Zr["v_applied"][kr]))))
        en = RL.estimates(r) or {}; er = RL.estimates(q) or {}
        ests = [x for x in (en.get("estimate"), en.get("estimate_ref"), er.get("estimate"), er.get("estimate_ref")) if x]
        bound = 3.0 * max(ests) if ests else float(Jn["args"].get("stop_fields", 1e-6))
        bound_lab = "3 estimates" if ests else "the stop rule's %.0e (no estimate)" % bound
        within = bool(bound is not None and max(dNa, dva) <= bound)
        me = bool(r.get("mask_equals_negative_set"))
        ns_new = r.get("nsub_max", 1); ns_ref = (q.get("weyl_summary") or {}).get("nsub_max", 1)
        p = bool(dE < 1e-11 and within and me and ns_new == 1 and ns_ref == 1)
        ok_all = ok_all and p
        f_ = lambda e: ("%.2e (%s, %s)" % (e.get("estimate_ref") or e.get("estimate") or float("nan"), ("%.3f" % e["q_obs"]) if e.get("q_obs") is not None else "-",
                                         ("%.3f" % e["q_ref"]) if e.get("q_ref") is not None else "-")) if e else "-"
        L.append("| %.3f | %.2e | %.2e | %.2e | %s | %s | %s (%s) | %s | %s, %s | %s |" % (r["T"] / float(Jn["args"]["rs"]), dE, dNa, dva, f_(en), f_(er), within, bound_lab, me, ns_new, ns_ref, p))
    L += ["", "- verdict: %s (|dE/E| below 1e-11, the applied fields within 3 times the larger fixed-point estimate of the two runs, the masks bitwise, nsub 1 in both)." % ("pass" if ok_all else "fail"),
          "- every number above is read from the two runs' json and fields npz by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
