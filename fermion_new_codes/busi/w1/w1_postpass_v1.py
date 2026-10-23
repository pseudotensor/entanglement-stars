#!/usr/bin/env python
"""w1_postpass_v1.py (reviewer A, the busi lane, W1): the join of the production (page_history_v6o, --interp linear) with the control
(page_history_v6n's complete channels and the lean v6o --interp none pass) at every checkpoint and cut, per channel and weighted:
  the raw exterior-cut entropy, the held vacuum's entropy of the same fields at the same cut (the production's, computed at every checkpoint;
  the control's where it has one, the difference printed as the vacuum-route check), the excess of both, the error = production - control;
  E_kappa (= E_inv) of both and the quench excess E_ctrl - E_prod per channel (the quench tail of the piecewise-constant propagation);
  the collar flow of both; the convergence rule (page_w1_sector_v1.convergence_rule) on the production and on the control, the two verdicts
  side by side and the error beside every partial sum; the regression of the lean control against v6n where both hold a channel; the
  propagation cost (the records' summed Chebyshev degrees, the wall time); the relabel text (defect 12, paper prohibition Y13).
The report is HELD (a banner, exit code 2) unless every --selftest json says pass_all: nothing of it prints before that. Only complete channels
(final) enter; running ones are listed.
Usage: python w1_postpass_v1.py --prod DIR TAG --ctrl DIR TAG [--ctrl DIR TAG ...] --out PREFIX --selftest JSON [--selftest JSON ...]
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
import page_w1_sector_v1 as W1S  # noqa: E402

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
    """one channel's rows (<tag>_k<kappa>_summary.json) and D13 clocks (_d13.json), keyed by the clock rounded to 1e-4 r_s; None without the summary."""
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
    return {"rows": rows, "clocks": clocks, "w": 16.0 * kappa, "final": bool(S.get("final")), "label": S.get("label"), "w1": w1, "code": S.get("code"), "wall_s": S.get("wall_s"),
            "n_records": len(recs), "cheb_K_sum": Ksum, "cheb_K_sum_exact": bool(recs) and n_summed == len(recs),
            "propagation": w1.get("propagation") or ("v6n's piecewise-constant fields over each record" if S.get("code") == "page_history_v6n.py" else "unknown"),
            "vacuum_clocks": sorted(t for t, cl in clocks.items() if any(c.get("S_X_vac") is not None for c in (cl.get("cuts") or {}).values()))}


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
    """one (clock, channel, cut): the production's raw entropy, held vacuum and excess; the control's raw entropy, its own held vacuum where it has one,
    and its excess completed with the production's held vacuum (the same fields, the same cut and record); the error production - control."""
    Svac_p = xp.get("S_X_vac"); Svac_c = xc.get("S_X_vac")
    Svac_ref = Svac_p if Svac_p is not None else Svac_c
    exc_p = xp.get("excess")
    exc_c = (xc["S_X"] - Svac_ref) if (xc.get("S_X") is not None and Svac_ref is not None) else None
    return {"S_raw_prod": xp.get("S_X"), "S_vac_prod": Svac_p, "excess_prod": exc_p, "t_X_prod": xp.get("t_X"),
            "S_raw_ctrl": xc.get("S_X"), "S_vac_ctrl": Svac_c, "excess_ctrl": exc_c,
            "excess_ctrl_vacuum_source": "production" if Svac_p is not None else ("control" if Svac_c is not None else None),
            "error": (exc_p - exc_c) if (exc_p is not None and exc_c is not None) else None,
            "raw_error": (xp["S_X"] - xc["S_X"]) if (xp.get("S_X") is not None and xc.get("S_X") is not None) else None}


def shown(clocks, from_rs):
    return [t for t in clocks if t >= from_rs - 1e-9 or abs(t - round(t / 10.0) * 10.0) < 1e-6 or any(abs(t - z) < 1e-6 for z in (12.0, 14.0, 20.0, 30.0))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prod", nargs=2, metavar=("DIR", "TAG"), required=True)
    ap.add_argument("--ctrl", nargs=2, metavar=("DIR", "TAG"), action="append", default=[], help="a control run (repeatable): the first that holds a channel is its control, a second its regression")
    ap.add_argument("--out", required=True, help="the output prefix (PREFIX.md, PREFIX.json, PREFIX_relabel.txt)")
    ap.add_argument("--tol", type=float, default=0.01); ap.add_argument("--from-rs", dest="from_rs", type=float, default=40.0)
    ap.add_argument("--kappa-max", dest="kappa_max", type=int, default=64)
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
    # the weighted sums over the channels with a control, and the error
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
                 "excess_ctrl": _wsum((prod[k]["w"], x["excess_ctrl"]) for k, x in both)}
            c["error"] = c["excess_prod"] - c["excess_ctrl"]
            c["raw_error"] = (c["S_raw_prod"] - c["S_raw_ctrl"]) if (c["S_raw_prod"] is not None and c["S_raw_ctrl"] is not None) else None
            wt["cuts"][lab] = c
        weighted[t] = wt
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
    cost = {k: {"propagation_prod": prod[k]["propagation"], "cheb_K_sum_prod": prod[k]["cheb_K_sum"], "cheb_K_sum_prod_exact": prod[k]["cheb_K_sum_exact"], "wall_s_prod": prod[k]["wall_s"],
                "ctrl_source": ctrl_src.get(k), "propagation_ctrl": ctrl[k]["propagation"] if k in ctrl else None, "cheb_K_sum_ctrl": ctrl[k]["cheb_K_sum"] if k in ctrl else None,
                "wall_s_ctrl": ctrl[k]["wall_s"] if k in ctrl else None, "sector_used_prod": (prod[k]["w1"] or {}).get("sector_used"), "audit_prod": (prod[k]["w1"] or {}).get("audit")} for k in kappas}
    # the relabel text (defect 12)
    relabel = None; flat = None
    for p, st in st_all:
        f_ = ((st or {}).get("stages") or {}).get("flat")
        if f_:
            flat = f_; relabel = f_.get("relabel_text")
            if relabel is None and f_.get("values"):
                relabel = W1S.relabel_text({float(k_): float(v_) for k_, v_ in f_["values"].items()})
    if relabel is None:
        relabel = ("RELABEL (defect 12, Y13), pending the flat gate (w1_selftest_v1.py --stages flat): the smoke test's S_X values at 16, 24, 30 and 48 r_s are the flat held vacuum's "
                   "raw kappa 1 exterior-cut entropy on the 8192-site chain, not an escaping entropy; the readout is the excess S_X(C) - S_X(P_-(h))")
    res = {"banner": banner, "gates_ok": gates_ok, "selftest": [{"path": p, "pass_all": (st or {}).get("pass_all"), "stages": sorted(((st or {}).get("stages") or {}).keys())} for p, st in st_all],
           "production": a.prod, "controls": a.ctrl, "kappas": kappas, "running": running, "kappas_with_control": kc, "control_source": {str(k): v for k, v in ctrl_src.items()},
           "cut_labels": labs, "clocks": clocks, "rule_production": rule_p, "rule_control": rule_c, "join": {str(t): {str(k): v for k, v in d.items()} for t, d in join.items()},
           "weighted": {str(t): v for t, v in weighted.items()}, "quench_excess_E_ctrl_minus_E_prod": {str(k): {str(t): v for t, v in d.items()} for k, d in quench.items()},
           "vacuum_check": {str(k): v for k, v in vac_check.items()}, "regression_lean_vs_v6n": {str(k): v for k, v in regress.items()}, "cost": {str(k): v for k, v in cost.items()},
           "relabel_text": relabel, "flat_gate": flat}
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=1)
    with open(a.out + "_relabel.txt", "w") as f:
        f.write(relabel + "\n")
    # the markdown
    L = ["# W1 on B2 to T/r_s 100: the production (page_history_v6o, interpolated propagation) against the control (piecewise-constant), per channel and weighted", "",
         "**%s**" % banner, "",
         "Production: %s/%s, %s; complete channels %s%s. Controls: %s; channels with a control %s (%s). Every S_X below is the raw exterior-cut entropy (regulated, not the escaping "
         "entropy); the readout is the excess over the held vacuum of the same fields at the same cut and record (the production's held vacuum, computed at every checkpoint, completes "
         "the control's excess where the control has none); the error of every production digit is production minus control, the record-interpolation spread." % (
             a.prod[0], a.prod[1], prod[kappas[0]]["propagation"], kappas, (", running (not entered): %s" % running) if running else "",
             ", ".join("%s/%s" % tuple(c) for c in a.ctrl) or "none", kc, ", ".join("kappa %d from %s" % (k, ctrl_src[k]) for k in kc) or "-"), ""]
    L += ["## 1. The verdicts of the tail-majorant rule (defect 10)", "", "Production (K = %d): %s" % (rule_p["K"], rule_p["verdict_line"]), ""]
    if rule_c:
        L += ["Control (the channels with a control, K = %d): %s" % (rule_c["K"], rule_c["verdict_line"]), ""]
    L += [rule_p["rule"], ""]
    show = shown(clocks, a.from_rs)
    L += ["## 2. The convergence table with the error beside every partial sum", "",
          "| T/r_s | quantity | production, all %d channels | production, the channels with a control | control | error prod - ctrl | tail bound (production) | tolerance | converged (production) | converged (control) |" % len(kappas),
          "|" + "---|" * 10]
    for t in show:
        ep = rule_p["per_clock"].get(t) or {}; ec = (rule_c["per_clock"].get(t) or {}) if rule_c else {}; wt = weighted.get(t) or {}
        en = ep.get("energy") or {}; enc = ec.get("energy") or {}
        if en:
            L.append("| %.3f | 16 kappa E_kappa (the majorant's terms) | %s | %s | %s | %s | %s | - | %s | %s |" % (
                t, _f(en.get("partial"), "%.6e"), _f(wt.get("E_prod"), "%.6e"), _f(wt.get("E_ctrl"), "%.6e"), _f(wt.get("E_error"), "%.3e"), _f(en.get("tail_bound"), "%.3e"),
                "falls" if (en.get("premise") or {}).get("ok") else "does not fall", ("falls" if (enc.get("premise") or {}).get("ok") else "does not fall") if enc else "-"))
        for lab in labs:
            cp = (ep.get("cuts") or {}).get(lab) or {}; cc = (ec.get("cuts") or {}).get(lab) or {}; wc = (wt.get("cuts") or {}).get(lab) or {}
            if cp.get("excess_partial") is None and not wc:
                continue
            L.append("| %.3f | excess at %s r_s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t, lab, _f(cp.get("excess_partial"), "%.6e"), _f(wc.get("excess_prod"), "%.6e"), _f(wc.get("excess_ctrl"), "%.6e"), _f(wc.get("error"), "%.3e"),
                _f(cp.get("tail_bound"), "%.3e"), _f(cp.get("tol_abs"), "%.3e"), _v(cp.get("converged")), _v(cc.get("converged"))))
        co = ep.get("collar") or {}; coc = ec.get("collar") or {}
        if co:
            L.append("| %.3f | the accumulated collar flow | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t, _f(co.get("partial"), "%.6e"), _f(wt.get("flow_prod"), "%.6e"), _f(wt.get("flow_ctrl"), "%.6e"), _f(wt.get("flow_error"), "%.3e"), _f(co.get("tail_bound"), "%.3e"),
                _f(co.get("tol_abs"), "%.3e"), _v(co.get("converged")), _v(coc.get("converged"))))
    L += ["", "### 2b. The production's majorant tables (page_w1_sector_v1.convergence_markdown)"]
    L += W1S.convergence_markdown(rule_p)
    L += ["", "## 3. The weighted sums over the channels with a control: the raw entropy, the held vacuum's, the excess, production against control", "",
          "| T/r_s | cut r_s | channels | S_X raw (prod) | S_X held vacuum (prod) | excess (prod) | S_X raw (ctrl) | excess (ctrl) | error prod - ctrl | raw error | 16 kappa E (prod) | 16 kappa E (ctrl) | quench E_ctrl - E_prod | collar flow acc (prod) | collar flow acc (ctrl) |",
          "|" + "---|" * 15]
    for t in show:
        wt = weighted.get(t) or {}
        for lab in labs:
            c = (wt.get("cuts") or {}).get(lab)
            if not c:
                continue
            L.append("| %.3f | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t, lab, len(c["channels"]), _f(c["S_raw_prod"], "%.6f"), _f(c["S_vac_prod"], "%.6f"), _f(c["excess_prod"], "%.6f"), _f(c["S_raw_ctrl"], "%.6f"), _f(c["excess_ctrl"], "%.6f"),
                _f(c["error"], "%.3e"), _f(c["raw_error"], "%.3e"), _f(wt.get("E_prod"), "%.6e"), _f(wt.get("E_ctrl"), "%.6e"), _f(wt.get("E_quench"), "%.3e"),
                _f(wt.get("flow_prod"), "%.6e"), _f(wt.get("flow_ctrl"), "%.6e")))
    L += ["", "## 4. Per channel at the headline clocks", "",
          "| T/r_s | kappa | cut r_s | S_X raw (prod) | S_X held vacuum (prod) | excess (prod) | t_X (prod) | S_X raw (ctrl) | S_X held vacuum (ctrl, where the control has one) | excess (ctrl, completed with the production's vacuum) | error | 16 kappa E_kappa (prod) | 16 kappa E_kappa (ctrl) |",
          "|" + "---|" * 13]
    for t in heads:
        for k in kappas:
            e = (join.get(t) or {}).get(k)
            rp = prod[k]["rows"].get(t) or {}; cp = prod[k]["clocks"].get(t) or {}
            for lab in labs:
                x = e["cuts"][lab] if e else cut_entry((cp.get("cuts") or {}).get(lab) or {}, {})
                L.append("| %.3f | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    t, k, lab, _f(x["S_raw_prod"], "%.6f"), _f(x["S_vac_prod"], "%.6f"), _f(x["excess_prod"], "%.6f"), _f(x["t_X_prod"], "%.4e"), _f(x["S_raw_ctrl"], "%.6f"),
                    _f(x["S_vac_ctrl"], "%.6f"), _f(x["excess_ctrl"], "%.6f"), _f(x["error"], "%.3e"), _f(rp.get("E_inv_weighted"), "%.6e"), _f(e["E_ctrl"] if e else None, "%.6e")))
    L += ["", "## 5. The quench excess E_ctrl - E_prod per channel (the piecewise-constant propagation's excitation at the record boundaries; E_kappa = E_inv, weighted 16 kappa)", "",
          "| kappa | " + " | ".join("T/r_s %.0f: E prod / E ctrl / ctrl - prod" % t for t in heads) + " |", "|" + "---|" * (1 + len(heads))]
    for k in kc:
        L.append("| %d | %s |" % (k, " | ".join("%s / %s / %s" % (_f(join[t][k]["E_prod"], "%.4e"), _f(join[t][k]["E_ctrl"], "%.4e"), _f(quench[k][t], "%.2e")) for t in heads)))
    L += ["", "## 6. The vacuum-route check: the control's own held-vacuum entropy (v6n's dense eigendecomposition at its baseline clocks; the lean pass's sector route at every checkpoint) against the production's at the same clock and cut", "",
          "| kappa | control source | clocks with a control vacuum | production vacuum clocks | compared (clock, cut) pairs | max |dS_X^vac| | worst at |", "|" + "---|" * 7]
    for k in kc:
        d = vac_check[k]
        L.append("| %d | %s | %d | %d | %d | %.3e | %s |" % (k, ctrl_src[k], len(d["control_vacuum_clocks"]), d["production_vacuum_clocks"], d["n"], d["max_abs_diff"],
                                                      ("T/r_s %.3f, cut %s" % (d["worst_at"]["t_over_rs"], d["worst_at"]["cut"])) if d["worst_at"] else "-"))
    L += ["", "## 7. The regression of the lean v6o control against v6n (both piecewise-constant; the sector propagation against the full one)", "",
          "| kappa | a | b | clocks | (clock, cut) pairs | max |dS_X| | max |dS_X^vac| | max |dE_inv| (weighted) | max |d collar flow| (weighted) |", "|" + "---|" * 9]
    for k in sorted(regress):
        r = regress[k]
        L.append("| %d | %s | %s | %d | %d | %.3e | %.3e | %.3e | %.3e |" % (k, r["a"], r["b"], r["n_clocks"], r["n_cuts"], r["max_abs_dS_X"], r["max_abs_dS_X_vac"], r["max_abs_dE_inv"], r["max_abs_dflow"]))
    if not regress:
        L.append("(no channel held by two controls)")
    L += ["", "## 8. The propagation cost", "",
          "| kappa | production | Chebyshev degree summed over the records (prod) | exact sum | wall h (prod) | sector halving used | audit records failed | control source | control propagation | degree sum (ctrl: cheb_K_sum, else cheb_K per record) | wall h (ctrl) | degree ratio prod/ctrl |",
          "|" + "---|" * 12]
    for k in kappas:
        c = cost[k]
        ratio = (c["cheb_K_sum_prod"] / c["cheb_K_sum_ctrl"]) if (c["cheb_K_sum_ctrl"]) else None
        L.append("| %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k, c["propagation_prod"], c["cheb_K_sum_prod"], "yes" if c["cheb_K_sum_prod_exact"] else "no", _f((c["wall_s_prod"] or 0.0) / 3600.0 if c["wall_s_prod"] is not None else None, "%.2f"),
            c["sector_used_prod"], (c["audit_prod"] or {}).get("records_failed", "-"), c["ctrl_source"] or "-", c["propagation_ctrl"] or "-", _f(c["cheb_K_sum_ctrl"], "%d"),
            _f((c["wall_s_ctrl"] or 0.0) / 3600.0 if c["wall_s_ctrl"] is not None else None, "%.2f"), _f(ratio, "%.2f")))
    L += ["", "## 9. The gates and the relabel (defect 12, Y13)", ""]
    for p, st in st_all:
        stg = (st or {}).get("stages") or {}
        L.append("- %s: %s; stages %s" % (p, ("pass_all %s" % st.get("pass_all")) if st else "MISSING", ", ".join("%s %s" % (s_, "pass" if v_.get("pass") else "FAIL") for s_, v_ in stg.items()) or "-"))
    if flat and flat.get("values"):
        L.append("- the flat held vacuum's kappa 1 cut entropy (8192 sites): " + ", ".join("%s r_s %.6f" % (k_, float(v_)) for k_, v_ in sorted(flat["values"].items(), key=lambda z: float(z[0]))) + "; gates %s" % flat.get("gates"))
    L += ["", relabel, ""]
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(banner); print(rule_p["verdict_line"])
    if rule_c:
        print(rule_c["verdict_line"])
    print("written: %s.md, %s.json, %s_relabel.txt" % (a.out, a.out, a.out))
    sys.exit(0 if gates_ok else 2)


if __name__ == "__main__":
    main()
