#!/usr/bin/env python
"""w1_postpass_v1d.py (reviewer A, the busi lane, W1; v1d of w1_postpass_v1c.py, W1_fix_v1d.md): the join of the production (page_history_v6o, --interp
linear) with the control (page_history_v6n's complete channels and the lean v6o --interp none pass) at every checkpoint and cut, per channel and weighted:
  the raw exterior-cut entropy, the held vacuum's entropy of the same fields at the same cut (the production's, computed at every checkpoint;
  the control's where it has one, the difference printed as the vacuum-route check), the excess of both, the error = production - control;
  E_kappa (= E_inv) of both and the quench excess E_ctrl - E_prod per channel (the quench tail of the piecewise-constant propagation);
  the collar flow of both; the convergence rule (page_w1_sector_v1d.convergence_rule) on the production and on the control, the two verdicts
  side by side and the error beside every partial sum; the regression of the lean control against v6n where both hold a channel; the
  propagation cost (the records' summed Chebyshev degrees, the wall time); the relabel text (defect 12, paper prohibition Y13);
  the excess's propagation floors (low: the measured impurity's entropy; high: the a priori bound) and the resolution beside every excess, weighted
  and per channel, and the state's impurity per channel against the a priori bound (section 5b); v1d: the first checkpoint's impurity T_init against the
  construction allowance beside it (the a priori bound carries the construction term).
v1e (W1_compose_ruling_v1.md sections 1 and 4; every changed line marked "v1e"): the readout at the clocks from --from-rs is the all-channel weighted
  excess sum with two floors summed over every channel, F_meas = 2 m h2((T_trunc + T_meas)/m) + 2 m h2(8 eps) (the 1 percent statement's) and  (v1e)
  F_apriori (the a priori high floor, beside); the resolution against both; the excess tail by its own geometric majorant on |16 kappa x_kappa| over the  (v1e)
  last three channels (premises: the three resolved against F_meas and falling); the verdict converged / not converged / within the floor; the  (v1e)
  beta x energy-tail form, beta and p, and the collar column print nowhere (the collar waits for the ruling's section 2).  (v1e)
The report is HELD (a banner, exit code 2) unless every --selftest json says pass_all: nothing of it prints before that. Only complete channels
(final) enter; running ones are listed.
Usage: python w1_postpass_v1d.py --prod DIR TAG --ctrl DIR TAG [--ctrl DIR TAG ...] --out PREFIX --selftest JSON [--selftest JSON ...]
       [--tol 0.01 --from-rs 40 --kappa-max 64]
Reads <DIR>/<TAG>_k<kappa>_summary.json and _d13.json only (no arrays). Writes PREFIX.md, PREFIX.json and PREFIX_relabel.txt."""
import argparse
import json
import os
import sys

import numpy as np

for _p in (os.environ.get("W1_CODE_V6"), os.environ.get("W1_CODE_D13")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import page_w1_sector_v1d as W1S  # noqa: E402  (v1d)

HEADLINE = (12.0, 14.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0)


def _f(x, fmt):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def _v(x):
    return "-" if x is None else ("yes" if x else "NO")


def _sum(vals):
    vals = list(vals)
    return float(sum(vals)) if vals and all(v is not None for v in vals) else None


def _wsum(pairs):
    """sum of w x over (w, x) pairs; None where any x is None or there is none."""
    pairs = list(pairs)
    return float(sum(w * x for w, x in pairs)) if pairs and all(x is not None for _, x in pairs) else None


def load_channel(out_dir, tag, kappa):
    """one channel's rows (<tag>_k<kappa>_summary.json) and D13 clocks (_d13.json), keyed by the clock rounded to 1e-4 r_s; None without the summary.
    v1c: the rows' W1 block (the impurity, the a priori bound) per clock under 'floors'; v1d: T_init and the construction allowance beside."""
    sj = os.path.join(out_dir, "%s_k%d_summary.json" % (tag, kappa)); dj = os.path.join(out_dir, "%s_k%d_d13.json" % (tag, kappa))
    if not os.path.exists(sj):
        return None
    try:
        S = json.load(open(sj))
    except ValueError:
        return None                                                            # a summary caught half-written by the mirror
    D = {"clocks": {}}
    if os.path.exists(dj):
        try:
            D = json.load(open(dj))
        except ValueError:
            pass
    rows = {round(float(r["t_over_rs"]), 4): r for r in S.get("rows", [])}
    clocks = {round(float(c["t_over_rs"]), 4): ((c.get("per_channel") or {}).get(str(kappa)) or {}) for c in (D.get("clocks") or {}).values()}
    recs = S.get("records", [])
    Ksum = 0; n_summed = 0
    for q in recs:
        pc = (q.get("per_channel") or {}).get(str(kappa)) or {}
        if pc.get("cheb_K_sum") is not None:
            Ksum += int(pc["cheb_K_sum"]); n_summed += 1
        else:
            Ksum += int(pc.get("cheb_K") or 0)
    w1 = S.get("w1") or {}
    floors = {}                                                                                                                 # v1c
    for t, r in rows.items():                                                                                                   # v1c
        wr = ((r.get("per_channel") or {}).get(str(kappa)) or {}).get("w1") or {}                                                # v1c
        fl = wr.get("propagation_floor") or {}                                                                                  # v1c
        if wr:                                                                                                                  # v1c
            floors[t] = {"T_meas": wr.get("sea_unitarity_trace"), "T_cheb": fl.get("T_cheb"), "T_trunc": fl.get("T_cheb_truncation"), "T_round": fl.get("T_cheb_rounding_allowance"),
                         "n_substeps": wr.get("n_substeps"), "cheb_K_max": fl.get("cheb_K_max"), "premise_ok": fl.get("premise_ok"),
                         "T_init": wr.get("T_init", fl.get("T_init")), "T_construction": fl.get("T_cheb_construction"), "construction_ok": fl.get("construction_ok")}   # v1d
    return {"rows": rows, "clocks": clocks, "w": 16.0 * kappa, "final": bool(S.get("final")), "label": S.get("label"), "w1": w1, "code": S.get("code"), "wall_s": S.get("wall_s"),
            "n_records": len(recs), "cheb_K_sum": Ksum, "cheb_K_sum_exact": bool(recs) and n_summed == len(recs),
            "propagation": w1.get("propagation") or ("v6n's piecewise-constant fields over each record" if S.get("code") == "page_history_v6n.py" else "unknown"),
            "cheb_tol": w1.get("cheb_tol") or ((S.get("settings") or {}).get("cheb_tol")),                                      # v1c
            "vacuum_clocks": sorted(t for t, cl in clocks.items() if any(c.get("S_X_vac") is not None for c in (cl.get("cuts") or {}).values())), "floors": floors}   # v1c


def regression(a_, b_):
    """two controls of one channel (the lean v6o pass against v6n, both piecewise-constant): the largest differences at the common clocks and cuts."""
    ta, ca = a_; tb, cb = b_
    out = {"a": ta, "b": tb, "max_abs_dS_X": 0.0, "max_abs_dS_X_vac": 0.0, "max_abs_dE_inv": 0.0, "max_abs_dflow": 0.0, "n_clocks": 0, "n_cuts": 0}
    for t in sorted(set(ca["rows"]) & set(cb["rows"])):
        ra, rb = ca["rows"][t], cb["rows"][t]
        if ra.get("E_inv_weighted") is not None and rb.get("E_inv_weighted") is not None:
            out["max_abs_dE_inv"] = max(out["max_abs_dE_inv"], abs(ra["E_inv_weighted"] - rb["E_inv_weighted"]))
        if ra.get("collar_flow_acc_weighted") is not None and rb.get("collar_flow_acc_weighted") is not None:
            out["max_abs_dflow"] = max(out["max_abs_dflow"], abs(ra["collar_flow_acc_weighted"] - rb["collar_flow_acc_weighted"]))
        out["n_clocks"] += 1
        for lab, cut in ((ca["clocks"].get(t) or {}).get("cuts") or {}).items():
            o = ((cb["clocks"].get(t) or {}).get("cuts") or {}).get(lab)
            if not o:
                continue
            if cut.get("S_X") is not None and o.get("S_X") is not None:
                out["max_abs_dS_X"] = max(out["max_abs_dS_X"], abs(cut["S_X"] - o["S_X"])); out["n_cuts"] += 1
            if cut.get("S_X_vac") is not None and o.get("S_X_vac") is not None:
                out["max_abs_dS_X_vac"] = max(out["max_abs_dS_X_vac"], abs(cut["S_X_vac"] - o["S_X_vac"]))
    return out


def cut_entry(xp, xc):
    """one (clock, channel, cut): the production's raw entropy, held vacuum and excess (v1c: with its floors and resolution); the control's raw
    entropy, its own held vacuum where it has one, and its excess completed with the production's held vacuum (the same fields, the same cut and
    record); the error production - control."""
    Svac_p = xp.get("S_X_vac"); Svac_c = xc.get("S_X_vac")
    Svac_ref = Svac_p if Svac_p is not None else Svac_c
    exc_p = xp.get("excess")
    exc_c = (xc["S_X"] - Svac_ref) if (xc.get("S_X") is not None and Svac_ref is not None) else None
    return {"S_raw_prod": xp.get("S_X"), "S_vac_prod": Svac_p, "excess_prod": exc_p, "t_X_prod": xp.get("t_X"),
            "floor_low_prod": xp.get("excess_floor_low"), "floor_high_prod": xp.get("excess_floor_high"), "resolution_prod": xp.get("excess_resolution"),   # v1c
            "S_raw_ctrl": xc.get("S_X"), "S_vac_ctrl": Svac_c, "excess_ctrl": exc_c,
            "excess_ctrl_vacuum_source": "production" if Svac_p is not None else ("control" if Svac_c is not None else None),
            "error": (exc_p - exc_c) if (exc_p is not None and exc_c is not None) else None,
            "raw_error": (xp["S_X"] - xc["S_X"]) if (xp.get("S_X") is not None and xc.get("S_X") is not None) else None}


def shown(clocks, from_rs):
    return [t for t in clocks if t >= from_rs - 1e-9 or abs(t - round(t / 10.0) * 10.0) < 1e-6 or any(abs(t - z) < 1e-6 for z in (12.0, 14.0, 20.0, 30.0))]


TABLE_A_CLOCKS = (12.0, 14.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)                                       # v1e: the channel table's 12 clocks (table A)


def f_meas(ch, t, lab):                                                                                                          # v1e
    """v1e (ruling sec. 1(b)): the measured floor of one channel's excess at (clock, cut), F_meas = f m h2((T_trunc + T_meas)/m) + f m h2(8 eps),    # v1e
    T_trunc the truncation's theorem bound, T_meas the measured impurity, m the cut's paired eigenvalues, f = 2 on the sector route; None if absent."""  # v1e
    cut = (((ch["clocks"].get(t) or {}).get("cuts") or {}).get(lab)) or {}                                                       # v1e
    fl = ch["floors"].get(t) or {}                                                                                               # v1e
    m = cut.get("m_pairs"); Tt = fl.get("T_trunc"); Tm = fl.get("T_meas"); rnd = cut.get("excess_floor_round")                   # v1e
    if m is None or Tt is None or Tm is None or rnd is None:                                                                     # v1e
        return None                                                                                                              # v1e
    fac = 2.0 if (ch.get("w1") or {}).get("sector_used", True) else 1.0                                                          # v1e: S_X = 2 S_Xc(C_+) on the sector route
    return fac * W1S.entropy_modulus(int(m), float(Tt) + float(Tm)) + float(rnd)                                                 # v1e


def readout_v1e(prod, kappas, labs, clocks, rule_p, weighted, tol, from_rs, n_fit=3):                                           # v1e
    """v1e (ruling secs. 1 and 4): per readout clock (t >= from_rs) and cut, the readout table's row."""                          # v1e
    rows = []                                                                                                                    # v1e
    last = kappas[-n_fit:]                                                                                                       # v1e
    for t in [t for t in clocks if t >= from_rs - 1e-9]:                                                                         # v1e
        en = (rule_p["per_clock"].get(t) or {}).get("energy") or {}                                                              # v1e
        for lab in labs:                                                                                                         # v1e
            cuts = {k: (((prod[k]["clocks"].get(t) or {}).get("cuts") or {}).get(lab)) or {} for k in kappas}                     # v1e
            exc = {k: cuts[k].get("excess") for k in kappas}                                                                     # v1e
            if any(x is None for x in exc.values()):                                                                             # v1e
                continue                                                                                                         # v1e
            w = {k: prod[k]["w"] for k in kappas}                                                                                # v1e
            fm = {k: f_meas(prod[k], t, lab) for k in kappas}                                                                    # v1e
            fa = {k: cuts[k].get("excess_floor_high") for k in kappas}                                                           # v1e
            S = float(sum(w[k] * exc[k] for k in kappas))                                                                        # v1e: every channel's term enters (sec. 1(a))
            Fm = _wsum((w[k], fm[k]) for k in kappas); Fa = _wsum((w[k], fa[k]) for k in kappas)                                 # v1e
            res_m = sum(1 for k in kappas if fm[k] is not None and abs(exc[k]) > fm[k])                                          # v1e
            res_a = sum(1 for k in kappas if fa[k] is not None and abs(exc[k]) > fa[k])                                          # v1e
            xt = [w[k] * exc[k] for k in last]                                                                                   # v1e
            gx = W1S.geometric_tail([abs(x) for x in xt], n_fit, 0.0)                                                            # v1e: the excess's own geometric majorant (sec. 1(c))
            resolved_last = all(fm[k] is not None and abs(exc[k]) > fm[k] for k in last)                                         # v1e
            falls = bool(gx["ok"] and gx.get("q") is not None and gx["q"] < 1.0)                                                 # v1e
            one = tol * abs(S)                                                                                                   # v1e
            if Fm is not None and one < Fm:                                                                                      # v1e
                verdict = "within the floor"                                                                                     # v1e
            elif falls and resolved_last and gx["tail"] is not None and gx["tail"] <= one:                                       # v1e
                verdict = "converged"                                                                                            # v1e
            else:                                                                                                                # v1e
                verdict = "not converged"                                                                                        # v1e
            why = []                                                                                                             # v1e
            if not falls:                                                                                                        # v1e
                why.append("the last %d terms do not fall" % n_fit)                                                              # v1e
            if not resolved_last:                                                                                                # v1e
                why.append("kappa %s not resolved against F_meas" % ",".join(str(k) for k in last if not (fm[k] is not None and abs(exc[k]) > fm[k])))   # v1e
            if falls and gx["tail"] is not None and gx["tail"] > one:                                                            # v1e
                why.append("tail above 1 percent")                                                                               # v1e
            wc = ((weighted.get(t) or {}).get("cuts") or {}).get(lab) or {}                                                      # v1e
            Et = (en.get("terms") or [None] * len(kappas))                                                                       # v1e
            raw = _wsum((w[k], cuts[k].get("S_X")) for k in kappas); vac = _wsum((w[k], cuts[k].get("S_X_vac")) for k in kappas)  # v1e
            rows.append({"t": t, "cut": lab, "S": S, "F_meas": Fm, "F_apriori": Fa, "resolved_meas": res_m, "resolved_apriori": res_a, "n": len(kappas),   # v1e
                         "E_last": Et[-n_fit:], "q_E": (en.get("premise") or {}).get("q"), "E_tail": en.get("tail_bound"), "E_partial": en.get("partial"),   # v1e
                         "x_last": xt, "q_x": gx.get("q"), "x_tail": gx.get("tail") if falls else None, "one_pct": one, "verdict": verdict, "why": "; ".join(why),   # v1e
                         "F_meas_over_S": (Fm / abs(S)) if (Fm is not None and S) else None, "F_apriori_over_S": (Fa / abs(S)) if (Fa is not None and S) else None,   # v1e
                         "x_tail_over_S": (gx["tail"] / abs(S)) if (falls and gx.get("tail") is not None and S) else None,                # v1e
                         "ctrl_error": wc.get("error"), "ctrl_channels": wc.get("channels"), "S_raw": raw, "S_vac": vac,                   # v1e
                         "fm_terms": [w[k] * fm[k] if fm[k] is not None else None for k in kappas]})                                    # v1e
    return rows                                                                                                                  # v1e


RULING_S = {                                                                                                                     # v1e: W1_compose_ruling_v1.md sec. 4, quoted as written
    "S1": "The exterior-cut excess S_X(C) - S_X(P_-(h)) of the B2 history (page_history_v6o: the fields interpolated linearly in 8 sub-steps, Chebyshev 1e-16, the massless-sector halving, 8192 sites), summed over kappa = 1 to 18 with weight 16 kappa, prints at the readout clocks 40 to 100 r_s and the cuts 2, 16, 24, 30 and 48 r_s in the table, as a number at 1 percent where the pair's verdict is converged: the formed-energy majorant 16 kappa E_kappa falls over the last three channels at every clock from 14 r_s (q 0.88 to 0.90 from 40 r_s, the energy tail at most 1.1e-3 of the partial sum), the excess tail by its own majorant is below 1 percent of the partial sum, and the propagation floor (the measured impurity's entropy with the truncation's theorem bound) is below 1e-3 of it; the a priori allowance, 0.34 to 0.40 nats at 100 r_s, prints beside as the bound the impurity was checked against, and against it the sum is within 4 to 7 percent.",   # v1e
    "S2": "Where the verdict is not converged (the 16 r_s cut at 40 to 50 r_s, the front at the cut, and the pairs the table marks) the sum prints with its tail bound and no percentage; where 1 percent of the sum is below the floor (the 48 r_s cut before 80 r_s) the excess has not formed at that cut and prints within the floor.",   # v1e
    "S3": "The accumulated collar flow prints no number at K = 18: its terms at kappa 12 to 18 are negative, grow in magnitude (q 1.17 to 1.36) and exceed the channel's formed energy (-1.52e-4 against 7.04e-5 weighted at 100 r_s, kappa 18); whether they are the held vacuum's own energy current through the collar, which the rule's collar term then subtracts, or the sub-step work term's attribution, is decided by the check of W1_compose_ruling_v1.md section 2; until then the collar column is absent.",   # v1e
    "S4": "The piecewise-constant control (page_history_v6n, kappa 1 to 8) rises from kappa 4 (16 kappa E_kappa 2.3e-2 to 7.1e-2 at 100 r_s against the production's 3.9e-3 to 1.7e-4), the quench floor of the record-wise propagation and the reason the production interpolates; its excess at 100 r_s exceeds the production's by 0.31 to 0.57 nats at the cuts 16 to 48 r_s, its extra quench entropy, and its kappa sum has no majorant at K = 8; the control for kappa 9 to 18 is pending.",   # v1e
    "S5": "The flat held vacuum's kappa 1 cut entropy on the 8192-site chain agrees with icloud X3's at 2, 16, 24 and 48 r_s to 4.6e-6 nats (sector route; the dense route within 1.8e-9 of it)"}   # v1e
X3_FLAT = {2.0: 2.05436, 16.0: 2.74643, 24.0: 2.88147, 48.0: 3.11088}                                                            # v1e


def sentences_b2(readout, rule_p, clocks, kappas, prod, ctrl, kc, weighted, flat, relabel, from_rs):                            # v1e
    """v1e (ruling sec. 4): the sentences S1 to S5 as the ruling quotes them, each followed by the numbers this post-pass computes for it."""   # v1e
    E = {t: (rule_p["per_clock"].get(t) or {}).get("energy") or {} for t in clocks}                                              # v1e
    qE = [(E[t].get("premise") or {}).get("q") for t in clocks if t >= from_rs - 1e-9 and E[t]]                                  # v1e
    fr = lambda t0: [E[t]["tail_bound"] / abs(E[t]["partial"]) for t in clocks if t >= t0 - 1e-9 and E[t] and E[t].get("tail_bound") is not None and E[t].get("partial")]   # v1e
    allfall = all((E[t].get("premise") or {}).get("ok") for t in clocks if t >= 14.0 - 1e-9 and E[t])                            # v1e
    conv = [r for r in readout if r["verdict"] == "converged"]                                                                   # v1e
    tlast = max(r["t"] for r in readout) if readout else None                                                                    # v1e
    late = [r for r in readout if tlast is not None and abs(r["t"] - tlast) < 1e-6]                                              # v1e
    L = ["## 2c. The PAGE_solved_history_v4 sentences (W1_compose_ruling_v1.md sec. 4, quoted) with this post-pass's numbers (v1e)", ""]   # v1e
    L += ["S1. \"%s\"" % RULING_S["S1"], "",                                                                                     # v1e
          "  computed: E majorant falls at every clock from 14 r_s: %s; q_E from %g r_s %s; energy tail / partial at most %s from 14 r_s and %s from %g r_s; converged pairs %d of %d; "   # v1e
          "among them excess tail / sum %s and F_meas / sum %s; at T/r_s %s F_apriori %s nats and F_apriori / sum %s." % (   # v1e
              "yes" if allfall else "NO", from_rs, _rng(qE, "%.3f"), _f(max(fr(14.0), default=None), "%.2e"), _f(max(fr(from_rs), default=None), "%.2e"), from_rs, len(conv), len(readout),   # v1e
              _rng([r["x_tail_over_S"] for r in conv]), _rng([r["F_meas_over_S"] for r in conv]), _f(tlast, "%g"), _rng([r["F_apriori"] for r in late], "%.3f"),   # v1e
              _rng([r["F_apriori_over_S"] for r in late], "%.3f")), ""]                                                          # v1e
    nc = [r for r in readout if r["verdict"] == "not converged"]; wf = [r for r in readout if r["verdict"] == "within the floor"]   # v1e
    L += ["S2. \"%s\"" % RULING_S["S2"], "",                                                                                     # v1e
          "  computed: not converged (%d): %s; within the floor (%d): %s." % (   # v1e
              len(nc), ", ".join("%g/%s" % (r["t"], r["cut"]) for r in nc) or "none", len(wf), ", ".join("%g/%s" % (r["t"], r["cut"]) for r in wf) or "none"), ""]   # v1e
    L += ["S3. \"%s\"" % RULING_S["S3"], "", "  computed: the collar column is absent from every table of this report (the ruling's sec. 2 check is reported beside the post-pass).", ""]   # v1e
    t100 = clocks[-1]                                                                                                            # v1e
    c48 = [(ctrl[k]["rows"].get(t100) or {}).get("E_inv_weighted") for k in kc if k >= 4]                                       # v1e
    p48 = [(prod[k]["rows"].get(t100) or {}).get("E_inv_weighted") for k in kc if k >= 4]                                        # v1e
    exd = [-((weighted.get(t100) or {}).get("cuts") or {}).get(lab, {}).get("error") for lab in ("16", "24", "30", "48")   # v1e
           if (((weighted.get(t100) or {}).get("cuts") or {}).get(lab) or {}).get("error") is not None]                         # v1e
    L += ["S4. \"%s\"" % RULING_S["S4"], "",                                                                                     # v1e
          "  computed at T/r_s %g: control 16 kappa E_kappa, kappa 4 to %s: %s; production's: %s (kappa 4 first: %s, kappa %s: %s); control excess minus production's (weighted, kappa %s) at the cuts 16 to 48 r_s: %s." % (   # v1e
              t100, max(kc) if kc else "-", _rng(c48), _rng(p48), _f(p48[0] if p48 else None, "%.2e"), max(kc) if kc else "-", _f(p48[-1] if p48 else None, "%.2e"),   # v1e
              ("1 to %d" % max(kc)) if kc else "-", _rng(exd, "%.3f")), ""]                                                      # v1e
    vals = {float(k_): float(v_) for k_, v_ in ((flat or {}).get("values") or {}).items()}                                       # v1e
    dmax = max([abs(vals[c] - x) for c, x in X3_FLAT.items() if c in vals], default=None)                                        # v1e
    L += ["S5. \"%s\", followed by the RELABEL text as printed:" % RULING_S["S5"], "", "    " + relabel, "",                    # v1e
          "  computed: max |flat - X3| over 2, 16, 24, 48 r_s %s; the 30 r_s value %s (reads 2.95571 to 5 decimals)." % (_f(dmax, "%.1e"), _f(vals.get(30.0), "%.7f")), ""]   # v1e
    return L                                                                                                                     # v1e


def _rng(vals, fmt="%.2e"):                                                                                                      # v1e
    vals = [v for v in vals if v is not None]                                                                                    # v1e
    return ("%s to %s" % (fmt % min(vals), fmt % max(vals))) if vals else "-"                                                    # v1e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prod", nargs=2, metavar=("DIR", "TAG"), required=True)
    ap.add_argument("--ctrl", nargs=2, metavar=("DIR", "TAG"), action="append", default=[], help="a control run (repeatable): the first that holds a channel is its control, a second its regression")
    ap.add_argument("--out", required=True, help="the output prefix (PREFIX.md, PREFIX.json, PREFIX_relabel.txt)")
    ap.add_argument("--tol", type=float, default=0.01); ap.add_argument("--from-rs", dest="from_rs", type=float, default=40.0)
    ap.add_argument("--kappa-max", dest="kappa_max", type=int, default=64)
    ap.add_argument("--sentences-b2", dest="sentences_b2", action="store_true", help="v1e: print the ruling's PAGE v4 sentences S1 to S5 (the B2 history) with the computed checks")   # v1e
    ap.add_argument("--selftest", action="append", default=[], help="w1_selftest_v1.json (repeatable: every one must say pass_all; the relabel text comes from the one with the flat stage)")
    a = ap.parse_args()
    # the gate on printing
    st_all = []; gates_ok = bool(a.selftest)
    for p in a.selftest:
        st = json.load(open(p)) if os.path.exists(p) else None
        st_all.append((p, st)); gates_ok = gates_ok and bool(st and st.get("pass_all"))
    gate_lines = ["%s: %s" % (p, ("pass_all %s" % st.get("pass_all")) if st else "MISSING") for p, st in st_all] or ["no --selftest given"]
    banner = ("ALL GATES PASS (%s)" % "; ".join(gate_lines)) if gates_ok else ("HELD: the gates have not all passed (%s): a working file, nothing of it prints" % "; ".join(gate_lines))
    # the channels (complete ones only)
    prod = {}; running = []; ctrl = {}; ctrl_src = {}; regress = {}
    for k in range(1, a.kappa_max + 1):
        p = load_channel(a.prod[0], a.prod[1], k)
        if p is None:
            continue
        if not p["final"]:
            running.append(k); continue
        prod[k] = p
        found = []
        for (d_, t_) in a.ctrl:
            c = load_channel(d_, t_, k)
            if c is not None and c["final"]:
                found.append((t_, c))
        if found:
            ctrl[k] = found[0][1]; ctrl_src[k] = found[0][0]
            if len(found) > 1:
                regress[k] = regression(found[0], found[1])
    if not prod:
        raise SystemExit("no complete production channel under %s/%s" % tuple(a.prod))
    kappas = sorted(prod); kc = [k for k in kappas if k in ctrl]
    labs = []
    for k in kappas:
        for cl in prod[k]["clocks"].values():
            if cl.get("cuts"):
                labs = list(cl["cuts"].keys()); break
        if labs:
            break
    clocks = sorted(set.intersection(*[set(prod[k]["rows"]) for k in kappas]))
    # the production's rule
    per_p = {k: {"rows": prod[k]["rows"], "w": prod[k]["w"]} for k in kappas}
    rule_p = W1S.convergence_rule(per_p, kappas, clocks, labs, a.tol, lambda k, t: prod[k]["clocks"].get(t) or {}, a.from_rs)
    # the join: the control's excess completed with the production's held-vacuum entropy at every checkpoint
    join = {}
    for t in clocks:
        for k in kc:
            cp = prod[k]["clocks"].get(t) or {}; cc = ctrl[k]["clocks"].get(t) or {}
            rp = prod[k]["rows"].get(t) or {}; rc = ctrl[k]["rows"].get(t) or {}
            e = {"E_prod": rp.get("E_inv_weighted"), "E_ctrl": rc.get("E_inv_weighted"), "flow_prod": rp.get("collar_flow_acc_weighted"), "flow_ctrl": rc.get("collar_flow_acc_weighted"),
                 "ctrl_has_row": bool(rc), "cuts": {lab: cut_entry((cp.get("cuts") or {}).get(lab) or {}, (cc.get("cuts") or {}).get(lab) or {}) for lab in labs}}
            join.setdefault(t, {})[k] = e
    # the control's rule on the channels with a control (its excess the completed one)
    rule_c = None
    if kc:
        clocks_c = [t for t in clocks if all(t in ctrl[k]["rows"] for k in kc)]
        per_c = {k: {"rows": ctrl[k]["rows"], "w": ctrl[k]["w"]} for k in kc}

        def d13_of_ctrl(k, t):
            e = (join.get(t) or {}).get(k)
            return {"cuts": {lab: {"S_X": c["S_raw_ctrl"], "excess": c["excess_ctrl"]} for lab, c in e["cuts"].items()}} if e else {}
        if clocks_c:
            rule_c = W1S.convergence_rule(per_c, kc, clocks_c, labs, a.tol, d13_of_ctrl, a.from_rs)
    # the weighted sums over the channels with a control, and the error (v1c: the floors)
    weighted = {}
    for t in clocks:
        ks = [k for k in kc if join[t][k]["ctrl_has_row"]]
        wt = {"channels": ks, "cuts": {}, "E_prod": _sum(join[t][k]["E_prod"] for k in ks), "E_ctrl": _sum(join[t][k]["E_ctrl"] for k in ks),
              "flow_prod": _sum(join[t][k]["flow_prod"] for k in ks), "flow_ctrl": _sum(join[t][k]["flow_ctrl"] for k in ks)}
        wt["E_error"] = (wt["E_prod"] - wt["E_ctrl"]) if (wt["E_prod"] is not None and wt["E_ctrl"] is not None) else None
        wt["E_quench"] = (-wt["E_error"]) if wt["E_error"] is not None else None
        wt["flow_error"] = (wt["flow_prod"] - wt["flow_ctrl"]) if (wt["flow_prod"] is not None and wt["flow_ctrl"] is not None) else None
        for lab in labs:
            both = [(k, join[t][k]["cuts"][lab]) for k in ks if join[t][k]["cuts"][lab]["error"] is not None]
            if not both:
                continue
            c = {"channels": [k for k, _ in both], "S_raw_prod": _wsum((prod[k]["w"], x["S_raw_prod"]) for k, x in both), "S_vac_prod": _wsum((prod[k]["w"], x["S_vac_prod"]) for k, x in both),
                 "excess_prod": _wsum((prod[k]["w"], x["excess_prod"]) for k, x in both), "S_raw_ctrl": _wsum((prod[k]["w"], x["S_raw_ctrl"]) for k, x in both),
                 "excess_ctrl": _wsum((prod[k]["w"], x["excess_ctrl"]) for k, x in both),
                 "floor_low_prod": _wsum((prod[k]["w"], x["floor_low_prod"]) for k, x in both), "floor_high_prod": _wsum((prod[k]["w"], x["floor_high_prod"]) for k, x in both)}   # v1c
            c["error"] = c["excess_prod"] - c["excess_ctrl"]
            c["raw_error"] = (c["S_raw_prod"] - c["S_raw_ctrl"]) if (c["S_raw_prod"] is not None and c["S_raw_ctrl"] is not None) else None
            wt["cuts"][lab] = c
        weighted[t] = wt
    readout = readout_v1e(prod, kappas, labs, clocks, rule_p, weighted, a.tol, a.from_rs)                                       # v1e
    tally = {}                                                                                                                   # v1e
    for r in readout:                                                                                                            # v1e
        tally.setdefault(r["cut"], {"converged": 0, "not converged": 0, "within the floor": 0})[r["verdict"]] += 1                # v1e
    # the quench excess per channel at the headline clocks, the vacuum-route check, the cost
    heads = [t for t in clocks if any(abs(t - z) < 1e-6 for z in HEADLINE)]
    quench = {k: {t: ((join[t][k]["E_ctrl"] - join[t][k]["E_prod"]) if (join[t][k]["E_ctrl"] is not None and join[t][k]["E_prod"] is not None) else None) for t in heads} for k in kc}
    vac_check = {}
    for k in kc:
        d = {"max_abs_diff": 0.0, "n": 0, "worst_at": None, "control_vacuum_clocks": ctrl[k]["vacuum_clocks"], "production_vacuum_clocks": len(prod[k]["vacuum_clocks"])}
        for t in clocks:
            for lab in labs:
                c = join[t][k]["cuts"][lab]
                if c["S_vac_prod"] is not None and c["S_vac_ctrl"] is not None:
                    diff = abs(c["S_vac_prod"] - c["S_vac_ctrl"]); d["n"] += 1
                    if diff > d["max_abs_diff"]:
                        d["max_abs_diff"] = diff; d["worst_at"] = {"t_over_rs": t, "cut": lab}
        vac_check[k] = d
    cost = {k: {"propagation_prod": prod[k]["propagation"], "cheb_tol_prod": prod[k].get("cheb_tol"), "cheb_K_sum_prod": prod[k]["cheb_K_sum"], "cheb_K_sum_prod_exact": prod[k]["cheb_K_sum_exact"], "wall_s_prod": prod[k]["wall_s"],
                "ctrl_source": ctrl_src.get(k), "propagation_ctrl": ctrl[k]["propagation"] if k in ctrl else None, "cheb_K_sum_ctrl": ctrl[k]["cheb_K_sum"] if k in ctrl else None,
                "wall_s_ctrl": ctrl[k]["wall_s"] if k in ctrl else None, "sector_used_prod": (prod[k]["w1"] or {}).get("sector_used"), "audit_prod": (prod[k]["w1"] or {}).get("audit")} for k in kappas}
    resolution = {}                                                                                                             # v1c: the resolution counts per channel
    for k in kappas:
        cnt = {"resolved": 0, "marginal": 0, "unresolved": 0, "none": 0}
        for t, cl in prod[k]["clocks"].items():
            for cut in (cl.get("cuts") or {}).values():
                r_ = cut.get("excess_resolution")
                cnt[r_ if r_ in cnt else "none"] += 1
        resolution[k] = cnt
    # the relabel text (defect 12)
    relabel = None; flat = None
    for p, st in st_all:
        f_ = ((st or {}).get("stages") or {}).get("flat")
        if f_:
            flat = f_; relabel = f_.get("relabel_text")
            if relabel is None and f_.get("values"):
                relabel = W1S.relabel_text({float(k_): float(v_) for k_, v_ in f_["values"].items()})
    if relabel is None:
        relabel = ("RELABEL (defect 12, Y13), pending the flat gate (w1_selftest_v1d.py --stages flat): the smoke test's S_X values at 16, 24, 30 and 48 r_s are the flat held vacuum's "
                   "raw kappa 1 exterior-cut entropy on the 8192-site chain, not an escaping entropy; the readout is the excess S_X(C) - S_X(P_-(h))")
    res = {"banner": banner, "gates_ok": gates_ok, "selftest": [{"path": p, "pass_all": (st or {}).get("pass_all"), "stages": sorted(((st or {}).get("stages") or {}).keys())} for p, st in st_all],
           "production": a.prod, "controls": a.ctrl, "kappas": kappas, "running": running, "kappas_with_control": kc, "control_source": {str(k): v for k, v in ctrl_src.items()},
           "cut_labels": labs, "clocks": clocks, "rule_production": rule_p, "rule_control": rule_c, "join": {str(t): {str(k): v for k, v in d.items()} for t, d in join.items()},
           "weighted": {str(t): v for t, v in weighted.items()}, "quench_excess_E_ctrl_minus_E_prod": {str(k): {str(t): v for t, v in d.items()} for k, d in quench.items()},
           "vacuum_check": {str(k): v for k, v in vac_check.items()}, "regression_lean_vs_v6n": {str(k): v for k, v in regress.items()}, "cost": {str(k): v for k, v in cost.items()},
           "floors": {str(k): {str(t): v for t, v in prod[k]["floors"].items()} for k in kappas}, "resolution_counts": {str(k): v for k, v in resolution.items()},   # v1c
           "relabel_text": relabel, "flat_gate": flat,   # v1e
           "readout_v1e": readout, "tally_v1e": tally,                                                                           # v1e
           "note_v1e": "rule_production and rule_control are v1d's convergence_rule kept for the record (their excess tails are the retired beta x energy-tail form and the collar's); they print nowhere in v1e"}   # v1e
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=1)
    with open(a.out + "_relabel.txt", "w") as f:
        f.write(relabel + "\n")
    # the markdown
    L = ["# W1 on B2 to T/r_s 100: the production (page_history_v6o, interpolated propagation) against the control (piecewise-constant), per channel and weighted", "",
         "**%s**" % banner, "",
         "Production: %s/%s, %s (Chebyshev tolerance %s); complete channels %s%s. Controls: %s; channels with a control %s (%s). Every S_X below is the raw exterior-cut entropy (regulated, not the escaping "
         "entropy); the readout is the excess over the held vacuum of the same fields at the same cut and record (the production's held vacuum, computed at every checkpoint, completes "
         "the control's excess where the control has none); the error of every production digit is production minus control, the record-interpolation spread; every excess carries its "
         "propagation floor (v1c, W1_fix_v1c.md: low = the measured impurity's entropy 2 m h2(T/m) + the rounding term, high = the a priori bound's with the construction and rounding allowances, "
         "W1_fix_v1d.md; the resolution resolved/marginal/unresolved)." % (
             a.prod[0], a.prod[1], prod[kappas[0]]["propagation"], _f(prod[kappas[0]].get("cheb_tol"), "%g"), kappas, (", running (not entered): %s" % running) if running else "",
             ", ".join("%s/%s" % tuple(c) for c in a.ctrl) or "none", kc, ", ".join("kappa %d from %s" % (k, ctrl_src[k]) for k in kc) or "-"), ""]   # v1d
    show = shown(clocks, a.from_rs)
    K = kappas[-1]                                                                                                               # v1e
    L += ["## 1. The verdicts (v1e, W1_compose_ruling_v1.md secs. 1 and 4)", "",                                                 # v1e
          "Production K = %d, readout clocks from %g r_s, tolerance %g of the excess sum: %s (per cut: %s). The excess sum is the %d-channel weighted partial sum, every term entered; its error is "   # v1e
          "the floors' sum over all channels, F_meas = 2 m h2((T_trunc + T_meas)/m) + 2 m h2(8 eps) (the 1 percent statement's; the statement's one indicative premise: the rounding's share "   # v1e
          "that rotates the range without a norm drift is of the measured impurity's order) and F_apriori (the a priori high floor, the bound the premise T_meas <= T_cheb was checked against). "   # v1e
          "The excess tail is its own geometric majorant on |16 kappa x_kappa| over kappa %s (q the larger ratio, tail = |w_K x_K| q/(1 - q); premises: the three terms resolved against "   # v1e
          "F_meas and falling). Verdict: within the floor where 1 percent of the sum is below F_meas; converged where the premises hold and the tail is at most 1 percent; else not converged. "   # v1e
          "The collar column prints nowhere (the ruling's sec. 2 decides it)." % (   # v1e
              K, a.from_rs, a.tol, ", ".join("%d %s" % (sum(d[v_] for d in tally.values()), v_) for v_ in ("converged", "not converged", "within the floor")),   # v1e
              "; ".join("%s r_s: %d / %d / %d" % (lab, d["converged"], d["not converged"], d["within the floor"]) for lab, d in tally.items()), len(kappas),   # v1e
              ", ".join(str(k) for k in kappas[-3:])), ""]                                                                          # v1e
    if kc:                                                                                                                       # v1e
        ce = [((rule_c or {}).get("per_clock") or {}).get(t, {}).get("energy") for t in clocks if t >= a.from_rs - 1e-9]         # v1e
        nf = sum(1 for e in ce if e and not (e.get("premise") or {}).get("ok"))                                                  # v1e
        L += ["Control (K = %d, the channels with a control %s): the formed-energy majorant does not fall at %d of %d readout clocks; its kappa sum has no majorant at K = %d." % (   # v1e
            max(kc), kc, nf, len(ce), max(kc)), ""]                                                                              # v1e
    L += ["## 2. The readout table (v1e)", "",                                                                                   # v1e
          "| T/r_s | cut r_s | excess sum (%d ch, weighted, nats) | F_meas | F_apriori | resolved of %d vs F_meas / F_apriori | 16k E_%s | q_E | energy tail | 16k x_%s | q_x | excess tail | 1 percent | verdict | prod - ctrl (kappa %s share) | raw S_X (regulated) | S_X^vac (regulated) |" % (   # v1e
              len(kappas), len(kappas), ", ".join(str(k) for k in kappas[-3:]), ", ".join(str(k) for k in kappas[-3:]), ("%d to %d" % (min(kc), max(kc))) if kc else "-"),   # v1e
          "|" + "---|" * 17]                                                                                                     # v1e
    for r in readout:                                                                                                            # v1e
        L.append("| %.1f | %s | %s | %s | %s | %d / %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (                # v1e
            r["t"], r["cut"], _f(r["S"], "%.6e"), _f(r["F_meas"], "%.2e"), _f(r["F_apriori"], "%.2e"), r["resolved_meas"], r["resolved_apriori"],   # v1e
            ", ".join(_f(x, "%.3e") for x in r["E_last"]), _f(r["q_E"], "%.4f"), _f(r["E_tail"], "%.2e"), ", ".join(_f(x, "%.3e") for x in r["x_last"]), _f(r["q_x"], "%.4f"),   # v1e
            _f(r["x_tail"], "%.2e"), _f(r["one_pct"], "%.2e"), r["verdict"] + ((" (%s)" % r["why"]) if (r["verdict"] == "not converged" and r["why"]) else ""),   # v1e
            _f(r["ctrl_error"], "%.3e"), _f(r["S_raw"], "%.6f"), _f(r["S_vac"], "%.6f")))                                        # v1e
    L += ["", "## 2b. The channel table (v1e): 16 kappa E_kappa at the table-A clocks, the ratio to the channel before, the control beside (no collar column)", ""]   # v1e
    tA = [t for t in clocks if any(abs(t - z) < 1e-6 for z in TABLE_A_CLOCKS)]                                                   # v1e
    L += ["| kappa | " + " | ".join("T/r_s %g" % t for t in tA) + " | ratio to kappa - 1 at T/r_s %g | 16k E^ctrl at T/r_s %g |" % (tA[-1], tA[-1]), "|" + "---|" * (len(tA) + 3)]   # v1e
    for i, k in enumerate(kappas):                                                                                               # v1e
        Ek = [(prod[k]["rows"].get(t) or {}).get("E_inv_weighted") for t in tA]                                                  # v1e
        Ep = (prod[kappas[i - 1]]["rows"].get(tA[-1]) or {}).get("E_inv_weighted") if i > 0 else None                            # v1e
        rat = (Ek[-1] / Ep) if (Ek[-1] is not None and Ep) else None                                                             # v1e
        Ec = ((ctrl[k]["rows"].get(tA[-1]) or {}).get("E_inv_weighted")) if k in ctrl else None                                  # v1e
        L.append("| %d | %s | %s | %s |" % (k, " | ".join(_f(x, "%.4e") for x in Ek), _f(rat, "%.4f"), _f(Ec, "%.4e")))         # v1e
    L += [""]                                                                                                                    # v1e
    if a.sentences_b2:                                                                                                           # v1e
        L += sentences_b2(readout, rule_p, clocks, kappas, prod, ctrl, kc, weighted, flat, relabel, a.from_rs)                   # v1e
    L += ["", "## 3. The weighted sums over the channels with a control: the raw entropy, the held vacuum's, the excess with its floors, production against control (v1e: no collar column)", "",   # v1e
          "| T/r_s | cut r_s | channels | S_X raw (prod) | S_X held vacuum (prod) | excess (prod) | excess floor low (prod) | excess floor high (prod) | S_X raw (ctrl) | excess (ctrl) | error prod - ctrl | raw error | 16 kappa E (prod) | 16 kappa E (ctrl) | quench E_ctrl - E_prod |",   # v1e
          "|" + "---|" * 15]                                                                                                     # v1e
    for t in show:
        wt = weighted.get(t) or {}
        for lab in labs:
            c = (wt.get("cuts") or {}).get(lab)
            if not c:
                continue
            L.append("| %.3f | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (                          # v1e
                t, lab, len(c["channels"]), _f(c["S_raw_prod"], "%.6f"), _f(c["S_vac_prod"], "%.6f"), _f(c["excess_prod"], "%.6f"), _f(c.get("floor_low_prod"), "%.2e"), _f(c.get("floor_high_prod"), "%.2e"),
                _f(c["S_raw_ctrl"], "%.6f"), _f(c["excess_ctrl"], "%.6f"), _f(c["error"], "%.3e"), _f(c["raw_error"], "%.3e"), _f(wt.get("E_prod"), "%.6e"), _f(wt.get("E_ctrl"), "%.6e"), _f(wt.get("E_quench"), "%.3e")))   # v1e
    L += ["", "## 4. Per channel at the headline clocks (v1e: the production's excess prints as a number only where resolved against F_meas, else its label)", "",   # v1e
          "| T/r_s | kappa | cut r_s | S_X raw (prod) | S_X held vacuum (prod) | excess (prod) | F_meas | F_apriori | resolution vs F_meas / F_apriori | t_X (prod) | S_X raw (ctrl) | S_X held vacuum (ctrl, where the control has one) | excess (ctrl, completed with the production's vacuum) | error | 16 kappa E_kappa (prod) | 16 kappa E_kappa (ctrl) |",   # v1e
          "|" + "---|" * 16]
    for t in heads:
        for k in kappas:
            e = (join.get(t) or {}).get(k)
            rp = prod[k]["rows"].get(t) or {}; cp = prod[k]["clocks"].get(t) or {}
            for lab in labs:
                x = e["cuts"][lab] if e else cut_entry((cp.get("cuts") or {}).get(lab) or {}, {})
                fm_ = f_meas(prod[k], t, lab); fa_ = x.get("floor_high_prod"); xp_ = x["excess_prod"]                            # v1e
                rm_ = None if (xp_ is None or fm_ is None) else ("resolved" if abs(xp_) > fm_ else "marginal")                  # v1e
                ra_ = None if (xp_ is None or fa_ is None) else ("resolved" if abs(xp_) > fa_ else "marginal")                  # v1e
                L.append("| %.3f | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    t, k, lab, _f(x["S_raw_prod"], "%.6f"), _f(x["S_vac_prod"], "%.6f"), (_f(xp_, "%.6f") if rm_ == "resolved" else (rm_ or "-")), _f(fm_, "%.2e"), _f(fa_, "%.2e"),   # v1e
                    "%s / %s" % (rm_ or "-", ra_ or "-"),                                                                        # v1e
                    _f(x["t_X_prod"], "%.4e"), _f(x["S_raw_ctrl"], "%.6f"), _f(x["S_vac_ctrl"], "%.6f"), _f(x["excess_ctrl"], "%.6f"), _f(x["error"], "%.3e"), _f(rp.get("E_inv_weighted"), "%.6e"), _f(e["E_ctrl"] if e else None, "%.6e")))
    L += ["", "## 5. The quench excess E_ctrl - E_prod per channel (the piecewise-constant propagation's excitation at the record boundaries; E_kappa = E_inv, weighted 16 kappa)", "",
          "| kappa | " + " | ".join("T/r_s %.0f: E prod / E ctrl / ctrl - prod" % t for t in heads) + " |", "|" + "---|" * (1 + len(heads))]
    for k in kc:
        L.append("| %d | %s |" % (k, " | ".join("%s / %s / %s" % (_f(join[t][k]["E_prod"], "%.4e"), _f(join[t][k]["E_ctrl"], "%.4e"), _f(quench[k][t], "%.2e")) for t in heads)))
    L += ["", "## 5b. The state's impurity per channel at the headline clocks (v1c, v1d): ||Phi^dagger Phi - 1||_1 against the propagation's a priori bound (construction + truncation + rounding allowance), the sub-steps, the first checkpoint's impurity against the construction allowance, the resolution counts over all clocks and cuts", "",
          "| kappa | " + " | ".join("T/r_s %.0f: impurity / bound / within" % t for t in heads) + " | sub-steps at the end | T_init / construction allowance / within | resolved / marginal / unresolved / no floor |", "|" + "---|" * (4 + len(heads))]   # v1d
    for k in kappas:
        fl = prod[k]["floors"]; cnt = resolution[k]
        last = fl.get(clocks[-1]) or {}
        first = fl.get(clocks[0]) or {}                                                                                           # v1d
        L.append("| %d | %s | %s | %s / %s / %s | %d / %d / %d / %d |" % (k, " | ".join("%s / %s / %s" % (_f((fl.get(t) or {}).get("T_meas"), "%.2e"), _f((fl.get(t) or {}).get("T_cheb"), "%.2e"), _v((fl.get(t) or {}).get("premise_ok"))) for t in heads),
                                                               _f(last.get("n_substeps"), "%d"), _f(first.get("T_init"), "%.2e"), _f(first.get("T_construction"), "%.2e"), _v(first.get("construction_ok")),   # v1d
                                                               cnt["resolved"], cnt["marginal"], cnt["unresolved"], cnt["none"]))
    L += ["", "## 6. The vacuum-route check: the control's own held-vacuum entropy (v6n's dense eigendecomposition at its baseline clocks; the lean pass's sector route at every checkpoint) against the production's at the same clock and cut", "",
          "| kappa | control source | clocks with a control vacuum | production vacuum clocks | compared (clock, cut) pairs | max |dS_X^vac| | worst at |", "|" + "---|" * 7]
    for k in kc:
        d = vac_check[k]
        L.append("| %d | %s | %d | %d | %d | %.3e | %s |" % (k, ctrl_src[k], len(d["control_vacuum_clocks"]), d["production_vacuum_clocks"], d["n"], d["max_abs_diff"],
                                                      ("T/r_s %.3f, cut %s" % (d["worst_at"]["t_over_rs"], d["worst_at"]["cut"])) if d["worst_at"] else "-"))
    L += ["", "## 7. The regression of the lean v6o control against v6n (both piecewise-constant; the sector propagation against the full one)", "",
          "| kappa | a | b | clocks | (clock, cut) pairs | max |dS_X| | max |dS_X^vac| | max |dE_inv| (weighted) |", "|" + "---|" * 8]   # v1e: no collar column
    for k in sorted(regress):
        r = regress[k]
        L.append("| %d | %s | %s | %d | %d | %.3e | %.3e | %.3e |" % (k, r["a"], r["b"], r["n_clocks"], r["n_cuts"], r["max_abs_dS_X"], r["max_abs_dS_X_vac"], r["max_abs_dE_inv"]))   # v1e
    if not regress:
        L.append("(no channel held by two controls)")
    L += ["", "## 8. The propagation cost", "",
          "| kappa | production | Chebyshev tolerance | Chebyshev degree summed over the records (prod) | exact sum | wall h (prod) | sector halving used | audit records failed | floor premise failures | control source | control propagation | degree sum (ctrl: cheb_K_sum, else cheb_K per record) | wall h (ctrl) | degree ratio prod/ctrl |",
          "|" + "---|" * 14]
    for k in kappas:
        c = cost[k]
        ratio = (c["cheb_K_sum_prod"] / c["cheb_K_sum_ctrl"]) if (c["cheb_K_sum_ctrl"]) else None
        L.append("| %d | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k, c["propagation_prod"], _f(c.get("cheb_tol_prod"), "%g"), c["cheb_K_sum_prod"], "yes" if c["cheb_K_sum_prod_exact"] else "no", _f((c["wall_s_prod"] or 0.0) / 3600.0 if c["wall_s_prod"] is not None else None, "%.2f"),
            c["sector_used_prod"], (c["audit_prod"] or {}).get("records_failed", "-"), (c["audit_prod"] or {}).get("floor_premise_failures", "-"), c["ctrl_source"] or "-", c["propagation_ctrl"] or "-", _f(c["cheb_K_sum_ctrl"], "%d"),
            _f((c["wall_s_ctrl"] or 0.0) / 3600.0 if c["wall_s_ctrl"] is not None else None, "%.2f"), _f(ratio, "%.2f")))
    L += ["", "## 9. The gates and the relabel (defect 12, Y13)", ""]
    for p, st in st_all:
        stg = (st or {}).get("stages") or {}
        L.append("- %s: %s; stages %s" % (p, ("pass_all %s" % st.get("pass_all")) if st else "MISSING", ", ".join("%s %s" % (s_, "pass" if v_.get("pass") else "FAIL") for s_, v_ in stg.items()) or "-"))
    if flat and flat.get("values"):
        L.append("- the flat held vacuum's kappa 1 cut entropy (8192 sites): " + ", ".join("%s r_s %.6f" % (k_, float(v_)) for k_, v_ in sorted(flat["values"].items(), key=lambda z: float(z[0]))) + "; X3's gates %s at %s" % (flat.get("gates"), flat.get("tol", "-")))   # v1d
    L += ["", relabel, ""]
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(banner)                                                                                                                # v1e
    print("v1e readout at K = %d from %g r_s: %s" % (kappas[-1], a.from_rs, "; ".join("%s r_s: %d converged / %d not converged / %d within the floor" % (lab, d["converged"], d["not converged"], d["within the floor"]) for lab, d in tally.items())))   # v1e
    print("written: %s.md, %s.json, %s_relabel.txt" % (a.out, a.out, a.out))
    sys.exit(0 if gates_ok else 2)


if __name__ == "__main__":
    main()
