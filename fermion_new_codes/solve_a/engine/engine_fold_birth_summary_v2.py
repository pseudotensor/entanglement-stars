#!/usr/bin/env python
"""engine_fold_birth_summary_v2.py (builder analyst; engine spec 37.3 (1)): version 1 of the summary read on engine_fold_birth_v2's json
(its "runs" with their "records"; the tau 4 and tau 2 histories included), printed so that the report quotes script output only; nothing
else changes. Version 1's description: Per run and root (R_t, r_in), over the records holding the growth identity's both sides (the records'
centred dR/dT and the right side with the rows' D_M) inside each episode (between a birth and the next closing): the largest and the median
|dR/dT - right side|/|dR/dT| and the same for the kinematic form, over the records at least --skip r_s after the episode's birth record (the
centred record difference straddles the fold's square root before that) and over all; the records whose matter term has the sign opposite
to dR/dT (the growth carried by the defect term), listed; the largest |D_M rows - D_M direct|/|D_M rows|; and the largest |M/(r/2G) - 1| at
both roots. Nothing is read or ruled. Args: --json --skip --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--skip", type=float, default=0.25)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    S = json.load(open(a.json))["runs"]
    L = ["## Summary of fold_birth_v2 (engine_fold_birth_summary_v2.py; engine spec 37.3 (1))", "",
         "- per run and root: the growth identity's two sides over the records inside an episode (birth to the next closing) holding both; \"after skip\" the records at least %g r_s after the episode's first record with zeros." % a.skip, ""]
    for lab, d in S.items():
        rows = d.get("records") or []
        ev = d.get("events") or []
        births = [e for e in ev if e["kind"] == "birth"]
        closes = [e for e in ev if e["kind"] == "closing"]
        L.append("### %s" % lab)
        L.append("- events: %s." % "; ".join("%s at T/rs %.4f (linear; quadratic %s), R_* %s" % (e["kind"], e["T_lin"], ("%.4f" % e["T_quad"]) if e.get("T_quad") is not None else "-",
                                                                                       ("%.3f" % e["R_lin"]) if e.get("R_lin") is not None else "-") for e in ev))
        mR = [abs(r["M_Rt_dev"]) for r in rows if r.get("M_Rt_dev") is not None]
        mi = [abs(r["M_rin_dev"]) for r in rows if r.get("M_rin_dev") is not None]
        ap_ = []
        L.append("- M = r w^2/(2G) at the roots on the re-solved dense fields: largest |M/(r/2G) - 1| %s at R_t (%d records), %s at r_in (%d records); M(R_t)/E from %s to %s." % (
            ("%.1e" % max(mR)) if mR else "-", len(mR), ("%.1e" % max(mi)) if mi else "-", len(mi), ("%.6f" % min(r["M_Rt_over_E"] for r in rows if r.get("M_Rt_over_E") is not None)) if mR else "-", ("%.6f" % max(r["M_Rt_over_E"] for r in rows if r.get("M_Rt_over_E") is not None)) if mR else "-"))
        for nm, title in (("Rt", "R_t"), ("rin", "r_in")):
            for bi, b in enumerate(births):
                t0 = b["records"][1]
                c = next((e for e in closes if e["T_lin"] > b["T_lin"]), None)
                t1 = c["records"][0] if c else max(r["T_over_rs"] for r in rows)
                ep = [r for r in rows if t0 - 1e-9 <= r["T_over_rs"] <= t1 + 1e-9 and r.get("growth_" + nm) and r.get("dRdT_" + nm) is not None]
                if not ep:
                    continue
                q = lambda r, key: abs(r["dRdT_" + nm] - r["growth_" + nm][key]) / max(abs(r["dRdT_" + nm]), 1e-300)
                late = [r for r in ep if r["T_over_rs"] >= t0 + a.skip - 1e-9]
                opp = [r for r in ep if np.sign(r["growth_" + nm]["matter"]) != np.sign(r["dRdT_" + nm])]
                dm = [abs(r["growth_" + nm]["DM_rows"] - r["growth_" + nm]["DM_direct"]) / max(abs(r["growth_" + nm]["DM_rows"]), 1e-300) for r in ep]
                share = [abs(r["growth_" + nm]["defect_rows"]) / max(abs(r["growth_" + nm]["matter"]), 1e-300) for r in ep]
                L.append("- %s, episode %d (records %.3f to %.3f, %d with both sides): |dR/dT - right side|/|dR/dT| largest %.3e, median %.3e (after skip: largest %s over %d records); kinematic form largest %.3e, median %.3e; matter term opposite in sign to dR/dT on %d records%s; |defect term|/|matter term| from %.3e to %.3e (median %.3e); |D_M rows - D_M direct|/|D_M rows| largest %.3e." % (
                    title, bi + 1, t0, t1, len(ep), max(q(r, "rhs_rows") for r in ep), float(np.median([q(r, "rhs_rows") for r in ep])),
                    ("%.3e" % max(q(r, "rhs_rows") for r in late)) if late else "-", len(late),
                    max(q(r, "kinematic") for r in ep), float(np.median([q(r, "kinematic") for r in ep])),
                    len(opp), (" (T/rs " + ", ".join("%.3f" % r["T_over_rs"] for r in opp[:40]) + (", ..." if len(opp) > 40 else "") + ")") if opp else "",
                    min(share), max(share), float(np.median(share)), max(dm)))
        L.append("")
    L.append("- every number above is printed by the script from fold_birth_v2.json.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
