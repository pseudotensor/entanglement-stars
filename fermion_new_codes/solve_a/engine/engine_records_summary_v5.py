#!/usr/bin/env python
"""engine_records_summary_v5.py (builder analyst; engine spec 25.2 BUILDER: "the maximum of |v|/N over the smooth fields per record, printed
for the set and the production"): version 4 took the maximum over the record's site fields, which for kernel-averaged runs are smoother
than the dense solution on which the record's min(N - |v|) is taken; version 5 re-solves every record's stored shell totals with the run's own
field module (engine_fields_v12 for engine_collapse_v17, v14 for v19, v16 for v24 and v25; collar policy and support as the run) and takes
max |v|/N of the dense functions on the grid of the record's trapping radius (spacing ell/64 to the last coarse node), with its radius; the
site-field value beside. Version 4's docstring: version 3 with the column max |v|/N over the sites of the record's solve (the fields npz N and v) and the
radius where it is reached, and per run the largest value with its time. Version 3's docstring: engine spec 24.3 BUILDER: version 2 with the trapping episodes: each contiguous run of
records with a trapping zero, its formation and closing times (the first record with R_t and the first after it without), and between two
episodes the untrapped interval with its largest min(N - |v|)). Version 2: engine spec 23.1 BUILDER: version 1 (a collapse run's records from T/rs --from on, one
line each, from its json) with the columns E - M(R_t), M(R_t) = R_t/(2g) (the mass function at the trapping zero), the dust's and the
pairs' shell energy inside R_t (from the fields npz: eps_tot - eps_pairs and eps_pairs summed over the shells below R_t), and per run the
formation and closing times (the first and the last record with R_t) and the lifetime. Args: --run dir/tag --from --md."""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    Tz = np.asarray(Z["T"], float); Rz = np.asarray(Z["R"], float)
    rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"]); g = float(J["args"]["g"])
    L = ["## %s from T/rs %g (engine_records_summary_v5.py; %s)" % (tag, a.t0, J.get("script")), "",
         "| T/rs | E/E0 | dE/dT | R_t | 2gE | R_t/2gE | E - M(R_t) | dust inside R_t | pairs inside R_t | r_in | min(N - abs v) | max abs v/N, dense (at r) | max abs v/N, sites | R_c | R_sh | pairs | pockets | inversions | mask = neg | continuation | unconverged | gauge |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    vmax_run = (None, None)
    import importlib
    scr = J.get("script") or ""
    fmod = "engine_fields_v12" if "v17" in scr else ("engine_fields_v14" if "v19" in scr else "engine_fields_v16")
    FM = importlib.import_module(fmod)
    ra_ = J["args"]; ell_ = float(ra_["ell"])
    f2 = lambda x, f: (f % x) if x is not None else "-"
    recs = [r for r in J["records"] if r.get("E") is not None]
    trap_all = [r["T"] / rs for r in recs if r.get("R_t")]
    for r in recs:
        if r["T"] / rs < a.t0 - 1e-9:
            continue
        Rt = r.get("R_t")
        k = int(np.argmin(np.abs(Tz - r["T"])))
        dust_in = pairs_in = None
        if Rt and abs(Tz[k] - r["T"]) < 1e-9:
            ep = np.asarray(Z["eps_pairs"][k], float); et = np.asarray(Z["eps_tot"][k], float)
            sel = Rz < Rt
            dust_in = float((et - ep)[sel].sum()); pairs_in = float(ep[sel].sum())
        gauge = r.get("gauge")
        vtxt = "-"; stxt = "-"
        if abs(Tz[k] - r["T"]) < 1e-9:
            Nk = np.asarray(Z["N"][k], float); vk = np.asarray(Z["v"][k], float)
            q = np.abs(vk) / np.where(Nk > 0, Nk, np.nan)
            stxt = "%.4f" % float(np.nanmax(q))
            _, _, _, inf_ = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, Rz, ell_,
                                                     collar_policy=ra_.get("collar_policy", "flag"), support_rel=ra_.get("support_rel", 1e-14))
            if not inf_.get("failed"):
                rr_ = np.arange(0.0, float(inf_["r_nodes"][-1]) + 1e-9, ell_ / 64.0)[1:]
                Nd = np.asarray(inf_["N_dense"](rr_), float); vd = np.asarray(inf_["v_dense"](rr_), float)
                qd = np.abs(vd) / np.where(Nd > 0, Nd, np.nan)
                iq = int(np.nanargmax(qd))
                vtxt = "%.4f (%.2f)" % (float(qd[iq]), float(rr_[iq]))
                if vmax_run[0] is None or float(qd[iq]) > vmax_run[0]:
                    vmax_run = (float(qd[iq]), r["T"] / rs)
        L.append("| %.3f | %.7f | %s | %s | %.3f | %s | %s | %s | %s | %s | %.4f | %s | %s | %s | %s | %.3e | %s | %s | %s | %s | %s | %s |" % (
            r["T"] / rs, r["E"] / E0, f2(r.get("dE_dT"), "%.3e"), f2(Rt, "%.3f"), 2 * g * r["E"], f2(r.get("R_t_over_2gE"), "%.4f"),
            f2((r["E"] - Rt / (2 * g)) if Rt else None, "%.4f"), f2(dust_in, "%.4f"), f2(pairs_in, "%.4e"), f2(r.get("r_in"), "%.3f"),
            r["min_N_minus_absv"], vtxt, stxt, f2(r.get("R_c"), "%.2f"), f2(r.get("R_sh"), "%.1f"), r["E_pairs_shells"], sum((r.get("n_pocket") or {}).values()),
            r.get("inversions_total", "-"), r.get("mask_equals_negative_set", "-"), (r.get("continuation_label") or (r.get("intersample_label") or "-"))[:70],
            ("%.1e" % r["unconverged"]["residual"]) if r.get("unconverged") else "-", gauge.get("label", "-") if isinstance(gauge, dict) else "-"))
    last_T = recs[-1]["T"] / rs if recs else None
    eps_ = []
    ep_lines = []
    cur = None
    gap_max = None
    for r in recs:
        t = r["T"] / rs
        if r.get("R_t"):
            if cur is None:
                cur = {"formed": t, "last": t, "untrapped_before_max_min": gap_max}
                gap_max = None
            cur["last"] = t
        else:
            if cur is not None:
                cur["closed"] = t; eps_.append(cur); cur = None
            gap_max = r["min_N_minus_absv"] if gap_max is None else max(gap_max, r["min_N_minus_absv"])
    if cur is not None:
        cur["closed"] = None; eps_.append(cur)
    for k, e_ in enumerate(eps_):
        ep_lines.append("- trapping episode %d: formed at T/rs %.3f, last recorded at %.3f, %s%s." % (
            k + 1, e_["formed"], e_["last"], ("closed by %.3f (the first record without a trapping zero)" % e_["closed"]) if e_.get("closed") else "open at the last record",
            ("; the untrapped interval before it: largest min(N - |v|) %+.4f" % e_["untrapped_before_max_min"]) if (k > 0 and e_.get("untrapped_before_max_min") is not None) else ""))
    if trap_all:
        closed = trap_all[-1] < last_T - 1e-9
        tline = "formed at T/rs %.3f, last recorded at %.3f (%d records); %s" % (trap_all[0], trap_all[-1], len(trap_all),
                                                                             ("closed after it: no surface at the later records to T/rs %.3f (lifetime at least %.3f r_s)" % (last_T, trap_all[-1] - trap_all[0])) if closed else "still open at the last record")
    else:
        tline = "none in these records"
    L += ["", "- the trapping surface: %s." % tline] + ep_lines + [
        "- the largest max |v|/N of the dense fields over the records from T/rs %g: %s." % (a.t0, ("%.4f at T/rs %.3f" % vmax_run) if vmax_run[0] is not None else "-"),
        "- every number above is read from the run's json and fields npz by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-10:]))


if __name__ == "__main__":
    main()
