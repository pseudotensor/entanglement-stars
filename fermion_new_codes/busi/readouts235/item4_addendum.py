#!/usr/bin/env python
"""item4_addendum.py (busi readouts #235, item 4, v2): the records the second reader asked to be read before W2's last three records print.
Per record: the record solve (the engine's plain self-consistency loop at T + tau, the record's E): iterations, converged, unconverged,
last |dE/E| and last field change against stop_E/stop_fields; n_pocket summed over channels; mask_equals_negative_set; nsub_max.
Per step (the record's dg_v36f block): converged, iterations, cap, mix, chart, level, nsub, attempts and why each failed, tracked_set_changes,
chart_disagreement_total, energy_defect_final, E_rec (the previous record's E), E_pred = E_rec + defect, E_end (v36k: the chart's charge at the
step's end from the quasi-static end passes), r_chain = E_end/E_rec - 1, and the record's E against E_pred and E_end; the step's node
pockets (continuation_nodes_final of v36f, track_end of v36k).
Whole history: per record (E_k - E_pred,k)/E0, the record solve's discrepancy against the step's conserved prediction, through 12.25,
against the record solve's last |dE/E|.
Usage: python item4_addendum.py <run.json>
"""
import json
import sys

import numpy as np


def tot(d):
    return sum(int(v) for v in (d or {}).values()) if isinstance(d, dict) else None


def main():
    j = json.load(open(sys.argv[1])); a = j["args"]; R = j["records"]; E0 = R[0]["E"]
    print("stop_E %s, stop_fields %s, maxit %s, record mix %s; dg args: %s" % (a.get("stop_E"), a.get("stop_fields"), a.get("maxit"), a.get("mix"),
          {k: v for k, v in a.items() if k.startswith("dg")}))
    disc = []
    for r in R:
        dg = r.get("dg_v36f") if isinstance(r.get("dg_v36f"), dict) else None
        if dg and dg.get("E_pred") is not None:
            h = r.get("iteration_history") or []
            disc.append((r["T_over_rs"], (r["E"] - dg["E_pred"]) / E0, h[-1][0] if h else None, max(h[-1][1], h[-1][2]) if h else None))
    pre = [x for x in disc if x[0] <= 12.25 + 1e-9]
    print("records with a dg step through 12.25: %d; max |E - E_pred|/E0 %.3e (at %.3f); record solve's last |dE/E| max %.3e, last field change max %.3e" % (
        len(pre), max(abs(x[1]) for x in pre), max(pre, key=lambda x: abs(x[1]))[0], max(x[2] for x in pre if x[2] is not None), max(x[3] for x in pre if x[3] is not None)))
    print("records through 12.25 with |E - E_pred|/E0 > stop_E: %d" % sum(1 for x in pre if abs(x[1]) > a.get("stop_E")))
    print()
    for r in R:
        T = r["T_over_rs"]
        if T < 12.25 - 1e-9:
            continue
        h = r.get("iteration_history") or []
        dg = r.get("dg_v36f") or {}
        print("== T/rs %.3f  record E %.12f  (E-E0)/E0 %+.4e" % (T, r["E"], (r["E"] - E0) / E0))
        print("  record solve: iterations %s, converged %s, unconverged %s, last |dE/E| %s, last field change %s; n_pocket %s (total %s); mask_equals_negative_set %s; nsub_max %s; engine label %s; step %s" % (
            r.get("iterations"), r.get("converged"), r.get("unconverged"), h[-1][0] if h else None, max(h[-1][1], h[-1][2]) if h else None,
            r.get("n_pocket"), tot(r.get("n_pocket")), r.get("mask_equals_negative_set"), r.get("nsub_max"), r.get("engine"), r.get("step")))
        print("  dg step %.3f -> %.3f: converged %s, iterations %s, cap %s, mix %s, chart %s, level %s, nsub %s, attempts %s, tracked_set_changes %s, chart_disagreement_total %s, defect %.3e" % (
            dg.get("from_T_over_rs", float("nan")), dg.get("to_T_over_rs", float("nan")), dg.get("converged"), dg.get("iterations"), dg.get("cap"), dg.get("mix"),
            dg.get("chart", "(v36f: continued)"), dg.get("level"), dg.get("nsub"), dg.get("attempts"), dg.get("tracked_set_changes"), dg.get("chart_disagreement_total"),
            dg.get("energy_defect_final") or float("nan")))
        Erec, Epred, Eend = dg.get("E_rec"), dg.get("E_pred"), dg.get("E_end")
        print("  E_rec(prev record) %.12f  E_pred %.12f  E_end %s;  (E - E_pred)/E0 %+.4e;  (E_end/E_rec - 1) %s;  (E - E_end)/E0 %s" % (
            Erec, Epred, ("%.12f" % Eend) if Eend is not None else "-", (r["E"] - Epred) / E0,
            ("%+.4e" % (Eend / Erec - 1)) if Eend is not None else "-", ("%+.4e" % ((r["E"] - Eend) / E0)) if Eend is not None else "-"))
        for at in dg.get("attempt_list") or []:
            subs = at.get("substeps") or []
            print("    attempt level %s nsub %s: converged %s, iterations %s, why %s; sub-step r_chain %s; sum %+.4e" % (
                at.get("level"), at.get("nsub"), at.get("converged"), at.get("iterations"), at.get("why"),
                " ".join("%+.2e" % s["r_chain_over_E_rec"] for s in subs if s.get("r_chain_over_E_rec") is not None),
                sum((s["E_end"] - s["E0"]) for s in subs if s.get("E_end") is not None) / E0))
        nodes = dg.get("continuation_nodes_final")
        if nodes:
            print("  step nodes' n_pocket (v36f continued reference, nodes 1..3): %s" % [sum(int(c.get("n_pocket") or 0) for c in nd.values()) for nd in nodes])
        te = dg.get("track_end")
        if te:
            print("  step end (v36k tracked set): convention_n_pocket total %s, convention_differs total %s, inversions total %s" % (
                sum(int(c.get("convention_n_pocket") or 0) for c in te.values()), sum(int(c.get("convention_differs") or 0) for c in te.values()),
                sum(int(c.get("inversions") or 0) for c in te.values())))
        an = dg.get("anchor")
        if an:
            print("  anchor (record's accepted set at the step's start): n_pocket total %s" % sum(int(c.get("n_pocket") or 0) for c in an.values()))


if __name__ == "__main__":
    main()
