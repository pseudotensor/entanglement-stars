#!/usr/bin/env python
"""cvconfirm_cmp_v1.py (busi, analyst; CVCONFIRM, ReviewA_cvcause_v1 sec. 3): the confirmation's comparator. Reads four engine runs
(dir/tag each; json and fields npz only, read only) and prints and writes the ruling's criterion:
    X  = CV2048's stored run (2048 sites, 16 r_s);   P = job (b)'s stored run (4096 sites, 32 r_s);
    A1 = arm 1, CV2048 resumed from its slice on engine_collapse_v36nq_rd13mfi.py with the option absent;
    A2 = arm 2, the same with --inject-fields (P's stored row at T_inject) --inject-T-over-rs T_inject.
PASS (the ruling): |N_A2(0) - N_P(0)| <= tol (1e-5) at every record clock on [T_inject, T_end] (3.3125 to 3.5), N(0) the stored nodes' first
value (r_nodes[0] = 0, checked), as CVREAD_cause_v1 read X - P. Beside it, not gating: X - P and A1 - P at r = 0 (the term, about 1e-3);
A2 - A1 at r = 0; the site-field profile max |N - N_P| on r <= --profile-r sites; the applied fields A2 - P on X's sites; arm 1 against X
(every fields-npz row array_equal: the control that the pod reproduces X; a difference is reported with its size); the injected row
against P's (bitwise at the mapped sites, the engine's own claim); the record's field_inject entry. Exit 0 on PASS, 1 on FAIL, 2 when a
clock is missing.
    python cvconfirm_cmp_v1.py --X D/T --P D/T --A1 D/T --A2 D/T --rs 128 --t-inject 3.3125 --t-end 3.5 --after 3.0 [--tol 1e-5] [--out J]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROWS = ("N", "v", "N_applied", "v_applied", "eps_tot", "J_tot", "eps_pairs", "J_pairs")


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Run(object):
    def __init__(self, spec, rs):
        d, t = os.path.split(spec)
        self.pj = os.path.join(d, "collapse_%s.json" % t); self.pf = os.path.join(d, "collapse_%s_fields.npz" % t)
        self.J = json.load(open(self.pj)); self.Z = np.load(self.pf, allow_pickle=True); self.rs = rs
        self.T = np.round(np.asarray(self.Z["T"], float) / rs, 6)
        self.recs = {round(r["T"] / rs, 6): r for r in self.J["records"] if r.get("T") is not None}
        self.n = len(self.Z["R"]); self.spec = spec

    def i(self, t):
        w = np.where(self.T == round(t, 6))[0]
        return int(w[0]) if len(w) == 1 else None

    def N0(self, t):
        i = self.i(t)
        r0 = float(np.asarray(self.Z["r_nodes"][i], float)[0])
        if r0 != 0.0:
            raise SystemExit("%s: r_nodes[0] = %g at T/rs %g, not 0" % (self.spec, r0, t))
        return float(np.asarray(self.Z["N_nodes"][i], float)[0])

    def row(self, k, t, n=None):
        x = np.asarray(self.Z[k][self.i(t)], float)
        return x if n is None else x[:n]


def main():
    ap = argparse.ArgumentParser()
    for k in ("X", "P", "A1", "A2"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--rs", type=float, required=True); ap.add_argument("--t-inject", type=float, required=True); ap.add_argument("--t-end", type=float, required=True)
    ap.add_argument("--after", type=float, required=True, help="the slice's clock: records after it are the arms' own")
    ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--profile-r", type=float, default=64.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {k: Run(getattr(a, k), a.rs) for k in ("X", "P", "A1", "A2")}
    n = R["X"].n
    for k in ("A1", "A2"):
        if R[k].n != n or not np.array_equal(np.asarray(R[k].Z["R"]), np.asarray(R["X"].Z["R"])):
            raise SystemExit("%s's chain differs from X's" % k)
    if not np.array_equal(np.asarray(R["P"].Z["R"])[:n], np.asarray(R["X"].Z["R"])):
        raise SystemExit("P's first %d radii differ from X's" % n)
    Rx = np.asarray(R["X"].Z["R"], float); pr = Rx <= a.profile_r
    ts = sorted(t for t in R["A2"].recs if t > a.after + 1e-9 and t <= a.t_end + 1e-9)
    miss = [t for t in ts if any(R[k].i(t) is None for k in R)]
    win = [t for t in ts if t >= a.t_inject - 1e-9]
    if miss or not win or abs(win[0] - a.t_inject) > 1e-9 or abs(ts[-1] - a.t_end) > 1e-9:
        print("cvconfirm_cmp_v1: clocks missing (%s) or the window [%g, %g] not covered by arm 2's records %s" % (miss, a.t_inject, a.t_end, ts))
        sys.exit(2)
    rows = []
    for t in ts:
        N0 = {k: R[k].N0(t) for k in R}
        q = {"T_over_rs": t, "N0": N0, "A2_minus_P": N0["A2"] - N0["P"], "X_minus_P": N0["X"] - N0["P"], "A1_minus_P": N0["A1"] - N0["P"],
             "A1_minus_X": N0["A1"] - N0["X"], "A2_minus_A1": N0["A2"] - N0["A1"],
             "profile_A2_minus_P": float(np.max(np.abs(R["A2"].row("N", t) - R["P"].row("N", t, n))[pr])),
             "profile_X_minus_P": float(np.max(np.abs(R["X"].row("N", t) - R["P"].row("N", t, n))[pr])),
             "applied_A2_minus_P_N": float(np.max(np.abs(R["A2"].row("N_applied", t) - R["P"].row("N_applied", t, n)))),
             "applied_A2_minus_P_v": float(np.max(np.abs(R["A2"].row("v_applied", t) - R["P"].row("v_applied", t, n)))),
             "in_window": t >= a.t_inject - 1e-9}
        ctl = {}
        for k in ROWS:
            x, y = R["A1"].row(k, t), R["X"].row(k, t)
            ctl[k] = 0.0 if np.array_equal(x, y) else float(np.max(np.abs(x - y)))
        x, y = np.asarray(R["A1"].Z["N_nodes"][R["A1"].i(t)], float), np.asarray(R["X"].Z["N_nodes"][R["X"].i(t)], float)
        ctl["N_nodes"] = 0.0 if np.array_equal(x, y) else (float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf"))
        q["A1_vs_X_rows"] = ctl; q["A1_bitwise_X"] = all(v == 0.0 for v in ctl.values())
        rec2 = R["A2"].recs[t]
        q["A2_converged"] = rec2.get("converged"); q["A2_kept_residual"] = (rec2.get("solve_v32") or {}).get("kept_residual")
        q["P_converged"] = R["P"].recs[t].get("converged"); q["P_kept_residual"] = (R["P"].recs[t].get("solve_v32") or {}).get("kept_residual")
        rows.append(q)
    ti = win[0]
    inj = R["A2"].recs[ti].get("field_inject")
    eq_src = {k: bool(np.array_equal(R["A2"].row(k, ti), R["P"].row(k, ti, n))) for k in ("N_applied", "v_applied")}
    dw = [abs(q["A2_minus_P"]) for q in rows if q["in_window"]]
    ok = max(dw) <= a.tol
    a1_bit = all(q["A1_bitwise_X"] for q in rows)
    out = {"criterion": "ReviewA_cvcause_v1 sec. 3: arm 2 - P <= %g at r = 0 on T/rs %g to %g" % (a.tol, a.t_inject, a.t_end), "PASS": ok,
           "max_abs_A2_minus_P_window": max(dw), "max_abs_X_minus_P_window": max(abs(q["X_minus_P"]) for q in rows if q["in_window"]),
           "injected_row_equals_source_at_mapped_sites": eq_src, "field_inject_entry": inj, "arm1_bitwise_with_X_all_rows": a1_bit,
           "rows": rows, "inputs": {k: {"spec": R[k].spec, "json_md5": md5f(R[k].pj), "fields_md5": md5f(R[k].pf), "sites": R[k].n} for k in R},
           "tol": a.tol, "profile_r": a.profile_r, "script": "cvconfirm_cmp_v1.py"}
    print("cvconfirm_cmp_v1: X %s (%d sites), P %s (%d sites); arms A1 %s, A2 %s" % (R["X"].spec, n, R["P"].spec, R["P"].n, R["A1"].spec, R["A2"].spec))
    print("%-9s %14s %14s %14s %14s %14s %12s %12s %10s  %s" % ("T/rs", "A2-P (r=0)", "X-P (r=0)", "A1-P (r=0)", "A2-A1 (r=0)", "A1-X (r=0)", "prof A2-P", "prof X-P", "Napp A2-P", "A1 rows vs X"))
    for q in rows:
        bad = {k: v for k, v in q["A1_vs_X_rows"].items() if v != 0.0}
        print("%-9.4f %+14.3e %+14.3e %+14.3e %+14.3e %+14.3e %12.3e %12.3e %10.2e  %s%s" % (q["T_over_rs"], q["A2_minus_P"], q["X_minus_P"], q["A1_minus_P"], q["A2_minus_A1"], q["A1_minus_X"],
              q["profile_A2_minus_P"], q["profile_X_minus_P"], q["applied_A2_minus_P_N"], "BITWISE" if not bad else "differ " + ", ".join("%s %.1e" % kv for kv in bad.items()),
              "  [window]" if q["in_window"] else ""))
    print("the injected record T/rs %g: arm 2's applied fields equal P's stored row at the %d mapped sites bitwise: %s" % (ti, n, eq_src))
    if inj:
        print("  field_inject: injected minus own N_applied %.3e v_applied %.3e; residual at the injected fields N %.3e v %.3e (own %.3e %.3e); solve minus P's stored N %s v %s; N(0) %.9f (own %.9f)" % (
            inj["injected_minus_own_N_applied"], inj["injected_minus_own_v_applied"], inj["injected_residual_N"], inj["injected_residual_v"], inj["own_residual_N"], inj["own_residual_v"],
            inj["solve_minus_source_N"], inj["solve_minus_source_v"], inj["N0_injected"] or float("nan"), inj["N0_own"] or float("nan")))
    else:
        print("  arm 2's record at T/rs %g has NO field_inject entry" % ti)
    print("arm 1 (option absent) against X on every fields row of the %d records: %s" % (len(rows), "BITWISE" if a1_bit else "NOT bitwise (sizes above)"))
    print("VERDICT: %s: max |A2 - P| at r = 0 on [%g, %g] = %.3e against tol %g (X - P there up to %.3e)" % (
        "PASS" if ok else "FAIL", a.t_inject, a.t_end, max(dw), a.tol, out["max_abs_X_minus_P_window"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
