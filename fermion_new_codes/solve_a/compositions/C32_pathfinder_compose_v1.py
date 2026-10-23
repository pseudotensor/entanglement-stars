#!/usr/bin/env python
"""C32_pathfinder_compose_v1.py (solve, composition 1, Reorientation 6): data step.

Reads the C_readout_v3b ledger (busi's C_readout_v3c.py run, tag C32_pathfinder_v1: npz keyed run__field__cut__quantity and its json)
of the r_s/a 32 pathfinder s4c80 (light coupling on) against its control s4offc80 (coupling off), both 40 to 80, and writes
C32_pathfinder_compose_v1.npz + .json: at every matched clock the member's column, the control's, d = member - control and a
numerical-resolution error bar sigma_d per quantity (formulas below; these are this composition's, the readout itself states no tolerance):

  H_in, H_out   sigma = sqrt(sum_runs (64 eps (|bare_in_C| + |bare_in_P|))^2)   (the trace resolution the run file uses for E_inv)
  I_C           sigma = sqrt(sum_runs [(I_C - I_X)^2 + (rel_adapter I_C)^2])       (the two current routes' spread + the adapter's worst
                                                                                   column-route/engine relative agreement, argv)
  int I_C       sigma = sqrt(sum_runs ((T_h - T_2h)/3)^2)                          (Richardson estimate of the slice-clock trapezoid,
                                                                                   at clocks reachable by 2h from 40; NaN elsewhere)
  int I_C       sigcorr = |T_h - T_2h|/3 of the difference series d I_C (the quadrature error common to both runs cancels in d)
  closure_cutlocal = closure(cut) - closure(chain end) per run (the cut-independent bookkeeping piece cancels); its d printed beside.
  int I_C vs evolution: route spread  |d(int I_C) - d(cum_evol)| = |d(closure)| printed beside, not folded into sigma.
  records (every 1/16 r_s): d of R_t, E/E0, H_in, M1_I_E, min(N-|v|) at 40, 44, ..., 80 with the resume gate's pass levels beside.

No compute beyond array arithmetic; nothing is fitted.
"""
import sys, json, hashlib
import numpy as np

EPS = np.finfo(float).eps
FIELDS = ["heavy", "light_sourcing", "light_all", "L1", "L2", "L3"] + ["H%d" % k for k in range(1, 11)]
CUTS = ["r16", "r24", "end"]
QTY = ["H_in", "H_out", "I_C", "I_X", "cum_int_I_C", "cum_evol", "cum_work", "cum_ref", "cum_memb", "cum_closure_C", "bare_in_C", "bare_in_P", "n_in"]
GATE_PASS = {"R_t": 3.4e-06, "E_over_E0": 3e-07, "min_N_minus_absv": 4.6e-08}   # the resume gate's pass levels (C_readout_v3b md 2.1)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def trap(T, y):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(T))])


def main():
    npz_p, json_p, out_p, rel_adapter = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    Z = np.load(npz_p)
    J = json.load(open(json_p))
    rs = None
    out, meta = {}, {"inputs": {npz_p: md5(npz_p), json_p: md5(json_p)}, "rel_adapter": rel_adapter, "fields": {}, "notes": []}
    for F in FIELDS:
        for c in CUTS:
            key = lambda run, q: "%s__%s__%s__%s" % (run, F, c, q)
            if key("member", "T") not in Z.files or key("control", "T") not in Z.files:
                meta["notes"].append("absent %s %s" % (F, c)); continue
            TC, TK = Z[key("member", "T")], Z[key("control", "T")]
            common = np.intersect1d(np.round(TC, 6), np.round(TK, 6))
            iC = np.array([np.argmin(abs(TC - t)) for t in common]); iK = np.array([np.argmin(abs(TK - t)) for t in common])
            pre = "%s__%s__" % (F, c)
            out[pre + "T"] = common
            col = {}
            for q in QTY:
                if key("member", q) in Z.files and key("control", q) in Z.files:
                    a, b = Z[key("member", q)][iC].astype(float), Z[key("control", q)][iK].astype(float)
                    out[pre + "C__" + q], out[pre + "K__" + q], out[pre + "d__" + q] = a, b, a - b
                    col[q] = (a, b)
            # sigma for H_in and H_out: the trace resolution
            if "bare_in_C" in col and "bare_in_P" in col:
                r = [64 * EPS * (np.abs(col["bare_in_C"][i]) + np.abs(col["bare_in_P"][i])) for i in (0, 1)]
                s = np.sqrt(r[0] ** 2 + r[1] ** 2)
                out[pre + "sig__H_in"] = s; out[pre + "sig__H_out"] = s
            if "I_C" in col and "I_X" in col:
                s2 = sum((col["I_C"][i] - col["I_X"][i]) ** 2 + (rel_adapter * col["I_C"][i]) ** 2 for i in (0, 1))
                out[pre + "sig__I_C"] = np.sqrt(s2)
            # Richardson on the trapezoid of the currents from 40 (the run's own full clock list, then sampled at the matched clocks)
            sig_int = np.zeros(len(common)); ok = np.ones(len(common), bool)
            for run, T_all in (("member", TC), ("control", TK)):
                I_all = Z[key(run, "I_C")].astype(float)
                m40 = T_all >= 40 - 1e-9
                T_, I_ = T_all[m40], I_all[m40]
                Th = trap(T_, I_)
                # 2h: every other clock
                sel = np.arange(0, len(T_), 2)
                T2h = trap(T_[sel], I_[sel])
                rich = np.full(len(T_), np.nan); rich[sel] = (Th[sel] - T2h) / 3.0
                # check the readout's own cum_int_I_C (from 40) against this trapezoid
                ci = Z[key(run, "cum_int_I_C")].astype(float)[m40]
                dev = float(np.nanmax(np.abs(ci - Th))) if len(ci) == len(Th) else float("nan")
                meta["fields"].setdefault(pre, {})["trap_vs_readout_%s" % run] = dev
                rr = np.array([rich[np.argmin(abs(T_ - t))] for t in common])
                ok &= np.isfinite(rr); sig_int += np.where(np.isfinite(rr), rr, 0.0) ** 2
            sig_int = np.sqrt(sig_int); sig_int[~ok] = np.nan
            out[pre + "sig__cum_int_I_C"] = sig_int
            # correlated route: Richardson on the difference series d I_C itself (the quadrature error common to both runs cancels in d)
            dI = Z[key("member", "I_C")][iC].astype(float) - Z[key("control", "I_C")][iK].astype(float)
            m40 = common >= 40 * 32.0 - 1e-6 if common.max() > 200 else common >= 40 - 1e-9
            Tc, dIc = common[m40], dI[m40]
            Th = trap(Tc, dIc); sel = np.arange(0, len(Tc), 2); T2h = trap(Tc[sel], dIc[sel])
            rc = np.full(len(common), np.nan); idx = np.where(m40)[0]; rc[idx[sel]] = np.abs(Th[sel] - T2h) / 3.0
            out[pre + "sigcorr__cum_int_I_C"] = rc
            # the cut-local closure: closure at this cut minus the closure at the chain end (the cut-independent part cancels), per run, and its d
    for F in FIELDS:
        for c in ("r16", "r24"):
            p, pe = "%s__%s__" % (F, c), "%s__end__" % F
            if p + "T" in out and pe + "T" in out and p + "C__cum_closure_C" in out:
                je = np.array([np.argmin(abs(out[pe + "T"] - t)) for t in out[p + "T"]])
                for r in ("C", "K"):
                    out[p + r + "__closure_cutlocal"] = out[p + r + "__cum_closure_C"] - out[pe + r + "__cum_closure_C"][je]
                out[p + "d__closure_cutlocal"] = out[p + "C__closure_cutlocal"] - out[p + "K__closure_cutlocal"]
    # the per-record series (records_continued)
    S = J["records_continued"]["series"]
    Tr = np.asarray(S.get("T_over_rs") or S.get("C.T_over_rs") or [], float)
    for k, v in S.items():
        try:
            out["rec__" + k] = np.asarray([np.nan if x is None else x for x in v], float)
        except Exception:
            pass
    P1 = J["preconditions"]["P1_join"]["member"]
    n = len(S["d.R_t"]); sp = float(P1["record_spacing"])
    out["rec__T"] = np.arange(n) * sp + float(P1["continued_first_T_over_rs"])
    meta["records_T_rule"] = "T/r_s = %g + %g i, i < %d (member joined_records %d, last %g)" % (P1["continued_first_T_over_rs"], sp, n, P1["joined_records"], P1["joined_last_T_over_rs"])
    meta["records_series_keys"] = sorted(S.keys())
    meta["gate_pass"] = GATE_PASS
    meta["preconditions_summary"] = J.get("summary")
    meta["charges"] = J["ledger"].get("charges")
    np.savez_compressed(out_p + ".npz", **out)
    json.dump(meta, open(out_p + ".json", "w"), indent=1, default=str)
    print("written", out_p + ".npz", len(out), "arrays")


if __name__ == "__main__":
    main()
