#!/usr/bin/env python
"""item4_w2_energy.py (busi readouts #235, item 4): the collar charge's drift on the discrete-gradient replay W2
(lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f): per record |E(t) - E(0)|/E(0) with E the record key 'E' (the collar
charge) and E(0) the T = 0 record's E (= head E0), the record-to-record step (E_k - E_{k-1})/E(0), the record's engine/step label and,
where the record carries it, the discrete-gradient block's own energy defect.
Usage: python item4_w2_energy.py <run.json> <out_prefix>
"""
import json
import sys

import numpy as np


def first_float(d, keys):
    for k in keys:
        if isinstance(d, dict) and isinstance(d.get(k), (int, float)):
            return k, float(d[k])
    return None, None


def main():
    j = json.load(open(sys.argv[1])); outp = sys.argv[2]
    R = j["records"]; E0 = R[0]["E"]
    print("head E0 %.15g, record 0 E %.15g" % (j["head"]["E0"], E0))
    dgk = R[-1].get("dg_v36f")
    print("dg_v36f keys (last record):", list(dgk.keys()) if isinstance(dgk, dict) else type(dgk).__name__)
    print("step (last record):", json.dumps(R[-1].get("step"))[:600])
    rows = []
    prev = None
    for r in R:
        E = r["E"]
        dg = r.get("dg_v36f") if isinstance(r.get("dg_v36f"), dict) else {}
        k_def, v_def = first_float(dg, ["energy_defect_final"])
        extra = {k: dg.get(k) for k in ("r_chain_over_E_rec", "E_rec", "E_pred", "E_end", "level", "nsub", "iterations", "chart", "E_minus_E0_over_E0", "converged")}
        st = r.get("step")
        rows.append({"T_over_rs": r["T_over_rs"], "E": E, "rel": abs(E - E0) / E0, "signed": (E - E0) / E0,
                     "step_rel": None if prev is None else (E - prev) / E0, "converged": r.get("converged"),
                     "engine": r.get("engine"), "step": st if isinstance(st, (str, int, float)) else (st.get("kind") or st.get("name") or st.get("mode") if isinstance(st, dict) else None),
                     "dg_defect_key": k_def, "dg_defect": v_def, "dg": extra})
        prev = E
    json.dump(rows, open(outp + ".json", "w"), indent=0, default=str)
    L = ["T/rs      E                    |E-E0|/E0    (E-E0)/E0     step (E_k-E_k-1)/E0   converged  step-label  dg-defect"]
    for x in rows:
        L.append("%7.3f  %.12f  %.3e   %+.3e   %s   %s  %s  %s" % (x["T_over_rs"], x["E"], x["rel"], x["signed"],
                 ("%+.3e" % x["step_rel"]) if x["step_rel"] is not None else "   -      ", x["converged"], x["step"],
                 ("%s=%.3e" % (x["dg_defect_key"], x["dg_defect"])) if x["dg_defect"] is not None else "-"))
    L.append("")
    L.append("discrete-gradient block per record from 11.875 (energy_defect_final, r_chain_over_E_rec, E_end/E_rec - 1, level, nsub, chart, E_minus_E0_over_E0):")
    for x in rows:
        if x["T_over_rs"] >= 11.875 - 1e-9:
            g = x["dg"]
            ee = (g["E_end"] / g["E_rec"] - 1) if isinstance(g.get("E_end"), (int, float)) and isinstance(g.get("E_rec"), (int, float)) else None
            L.append("  %7.3f  defect %s  r_chain %s  E_end/E_rec-1 %s  level %s nsub %s chart %s  E-E0/E0 %s" % (x["T_over_rs"], g and x["dg_defect"], g.get("r_chain_over_E_rec"), ee, g.get("level"), g.get("nsub"), g.get("chart"), g.get("E_minus_E0_over_E0")))
    rel = np.array([x["rel"] for x in rows]); T = np.array([x["T_over_rs"] for x in rows])
    pre = T <= 12.25 + 1e-9
    L.append("")
    L.append("SUMMARY: %d records 0..%.3f; max |E-E0|/E0 through 12.25: %.3e (at %.3f); at 12.375, 12.5, 12.625: %s; record steps there: %s" % (
        len(rows), T[-1], rel[pre].max(), T[pre][np.argmax(rel[pre])],
        ", ".join("%.3e" % x["rel"] for x in rows if x["T_over_rs"] >= 12.375 - 1e-9),
        ", ".join("%+.3e" % x["step_rel"] for x in rows if x["T_over_rs"] >= 12.375 - 1e-9)))
    open(outp + ".txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
