#!/usr/bin/env python
"""s22_analyze_v1.py (busi, inbox #420, NOTCOMP item 2.2): the interior's supply against the formed excitation's drain, from the lean runs of
s22_lean_queue_v3.sh (B2's history, 8192 sites, the production line page_history_v6s 611d190f / 15e08e28 unchanged).

For each channel kappa in {1, 2} and reference clock T_ref in {60, 80}: from the common state at T_ref the actual history (actK) and the history
with the fields held at T_ref (hTs_kK) propagate the same state. The collar flux L(t) (16 r_s, per chain, at each record's start, every tau = 8 a in
both runs) gives the fluences F_act(T_ref, T) and F_hold(T_ref, T) (left sums); the supply is
    Delta F = F_act - F_hold    (the energy the geometry's change after T_ref delivers through the collar by T; signed),
and the share Delta F / F_act. Weighted per channel by its 16 kappa chains, and summed over kappa 1 and 2 (the paper's L_col channels).
Checks: (i) before T_ref the hold run's records equal the actual run's (same code, same state): max |L_hold - L_act| on t < T_ref;
(ii) the hold conserves E_inv (rows every 10 r_s); (iii) second route (reviewer A): the geometry's work on the deviation after T_ref,
W = E_inv,act(T) - E_inv,act(T_ref) (E_inv = Tr[h(C - P_-)], whose rate is Tr[h-dot (C - P_-)]), beside Delta F;
(iv) causality: Delta F per window from T_ref (the change at the root reaches the collar only after the outgoing travel time).
Also: L in units of the comparator L_H(T_H) of the production summary when present (per chain).
Usage: python s22_analyze_v1.py RUNDIR OUT.json
"""
import json
import os
import sys

import numpy as np

RS = 64.0


def load(path):
    d = json.load(open(path))
    rec = d["records"]
    t = np.array([r["t"] for r in rec], float)
    L = np.array([r["L_at_t"] for r in rec], float)
    held = np.array([bool(r.get("held_step", False)) for r in rec])
    o = np.argsort(t)
    rows = d["rows"]
    key = list(rows[0]["per_channel"].keys())[0]
    rt = np.array([r["t_over_rs"] for r in rows], float)
    Einv = np.array([r["per_channel"][key]["E_inv"] for r in rows], float)
    exc = [r["per_channel"][key].get("interior", r["per_channel"][key]) for r in rows]
    return {"t": t[o], "L": L[o], "held": held[o], "rows_t": rt, "E_inv": Einv, "final": d.get("final"), "kappa": int(key), "rows": rows}


def fluence(R, Ta, Tb):
    """left sum of L dt over the records with Ta <= t < Tb (t in a), dt the spacing to the next record (last: tau)."""
    t, L = R["t"], R["L"]
    dt = np.diff(np.append(t, t[-1] + (t[-1] - t[-2])))
    sel = (t >= Ta - 1e-9) & (t < Tb - 1e-9)
    return float(np.sum(L[sel] * dt[sel]))


PROD = "<internal>/v6o_B2_k%d_summary.json"


def lh_fluence(K, a, b):
    """the comparator L_H(T_H) of channel K (the production summary's L_H_TH, already weighted by its 16 K chains), trapezoid over [a, b] r_s"""
    p = PROD % K
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = np.array([r["t_over_rs"] for r in d["rows"]], float); y = np.array([r.get("L_H_TH") or np.nan for r in d["rows"]], float)
    sel = (t >= a - 1e-9) & (t <= b + 1e-9) & np.isfinite(y)
    if sel.sum() < 2:
        return None
    return float(np.trapezoid(y[sel], t[sel] * RS)) if hasattr(np, "trapezoid") else float(np.trapz(y[sel], t[sel] * RS))


def main():
    rd, outp = sys.argv[1], sys.argv[2]
    runs = {}
    for tag in ("act1", "act2", "h60s_k1", "h80s_k1", "h60s_k2", "h80s_k2"):
        K = tag[-1]
        p = os.path.join(rd, "runs", tag, "%s_k%s_summary.json" % (tag, K))
        if os.path.exists(p):
            runs[tag] = load(p)
            print("%s: %d records to T/r_s %.3f, rows to %.1f, final %s" % (tag, len(runs[tag]["t"]), runs[tag]["t"][-1] / RS, runs[tag]["rows_t"][-1], runs[tag]["final"]))
    out = {"runs": {k: {"n_records": int(len(v["t"])), "t_last_rs": float(v["t"][-1] / RS), "final": v["final"]} for k, v in runs.items()}, "cases": []}
    tot = {}
    for K in (1, 2):
        A = runs.get("act%d" % K)
        if A is None:
            continue
        for Tref in (60, 80):
            Hh = runs.get("h%ds_k%d" % (Tref, K))
            if Hh is None:
                continue
            w = 16.0 * K
            Ta = Tref * RS
            # (i) the common past
            pre = Hh["t"] < Ta - 1e-9
            tA = {round(x, 6): i for i, x in enumerate(A["t"])}
            idx = [tA.get(round(x, 6)) for x in Hh["t"][pre]]
            ok = [i is not None for i in idx]
            dpre = float(np.max(np.abs(Hh["L"][pre][ok] - A["L"][[i for i in idx if i is not None]]))) if any(ok) else None
            Lscale = float(np.max(np.abs(A["L"][A["t"] < Ta]))) if np.any(A["t"] < Ta) else None
            Tend = min(A["t"][-1] + 8.0, Hh["t"][-1] + 8.0, 100.0 * RS + 1e-6)
            case = {"kappa": K, "T_ref": Tref, "weight": w, "pre_max_abs_dL": dpre, "pre_L_scale": Lscale, "windows": []}
            # (ii) the hold's invariant energy
            sel = Hh["rows_t"] >= Tref - 1e-9
            case["hold_E_inv_rows"] = [float(x) for x in Hh["E_inv"][sel]]
            case["hold_E_inv_spread"] = float(np.ptp(Hh["E_inv"][sel])) if sel.any() else None
            # windows of 10 r_s and the whole
            edges = list(np.arange(Tref, 100.0 + 1e-9, 10.0))
            if edges[-1] < 100.0:
                edges.append(100.0)
            for a, b in list(zip(edges[:-1], edges[1:])) + [(Tref, 100.0)]:
                Fa = fluence(A, a * RS, b * RS); Fh = fluence(Hh, a * RS, b * RS)
                row = {"from": a, "to": b, "F_act": Fa, "F_hold": Fh, "dF": Fa - Fh, "share": (Fa - Fh) / Fa if Fa != 0 else None,
                       "F_act_w": w * Fa, "F_hold_w": w * Fh, "LH_fluence": lh_fluence(K, a, b)}
                if row["LH_fluence"]:
                    row["dF_w_over_LH"] = w * (Fa - Fh) / row["LH_fluence"]; row["F_act_w_over_LH"] = w * Fa / row["LH_fluence"]; row["F_hold_w_over_LH"] = w * Fh / row["LH_fluence"]
                ia = int(np.argmin(np.abs(A["rows_t"] - a))); ib = int(np.argmin(np.abs(A["rows_t"] - b)))
                if abs(A["rows_t"][ia] - a) < 1e-6 and abs(A["rows_t"][ib] - b) < 1e-6:
                    row["W_act"] = float(A["E_inv"][ib] - A["E_inv"][ia])
                case["windows"].append(row)
            out["cases"].append(case)
            whole = case["windows"][-1]
            tot.setdefault(Tref, {"F_act_w": 0.0, "F_hold_w": 0.0, "kappas": []})
            tot[Tref]["F_act_w"] += whole["F_act_w"]; tot[Tref]["F_hold_w"] += whole["F_hold_w"]; tot[Tref]["kappas"].append(K)
            print("kappa %d T_ref %d: common past max|dL| %s (L scale %s); hold E_inv spread %s" % (K, Tref, dpre, Lscale, case["hold_E_inv_spread"]))
            for r in case["windows"]:
                print("   %5.1f to %5.1f: F_act %+.4e  F_hold %+.4e  dF %+.4e  share %s  W_act %s  [L_H units: act %s hold %s dF %s]" % (r["from"], r["to"], r["F_act"], r["F_hold"], r["dF"],
                      ("%+.4f" % r["share"]) if r["share"] is not None else "-", ("%+.4e" % r["W_act"]) if "W_act" in r else "-",
                      *[("%+.3f" % r[k]) if r.get(k) is not None else "-" for k in ("F_act_w_over_LH", "F_hold_w_over_LH", "dF_w_over_LH")]))
    for Tref, v in tot.items():
        v["share"] = (v["F_act_w"] - v["F_hold_w"]) / v["F_act_w"] if v["F_act_w"] else None
        print("kappa %s summed (16 kappa chains each), T_ref %d to 100: F_act %.4e F_hold %.4e supply share %s" % (v["kappas"], Tref, v["F_act_w"], v["F_hold_w"], v["share"]))
    out["summed"] = {str(k): v for k, v in tot.items()}
    json.dump(out, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
