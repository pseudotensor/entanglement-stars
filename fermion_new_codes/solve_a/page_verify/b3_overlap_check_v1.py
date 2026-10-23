#!/usr/bin/env python3
"""b3_overlap_check_v1.py (solve, page_verify B3 kit): the overlap reading of a 2048-site v19 history against the 4096-site B2 record, and
the continuation's own bookkeeping. Reads stored jsons and fields npz only.
 (a) overlap: at every common record clock in [--from, --to] (T/r_s): E/E0, E, R_t, R_t/2gE, kappa_T, min(N - |v|), r_in, n_pocket of each run
     and their differences; the applied fields on the common sites (R <= 2048 a, the same R grid on both chains): max |dN|, max |dv|,
     and the same restricted to R <= --inner-rs r_s (the interior, away from the 2048-site chain's end at 32 r_s);
 (b) with --cont: the continuation's records up to its resume clock against the run it resumed from (they must be the stored ones, bitwise),
     and its first --first-n new records' wall (json 'seconds'), median and max, and the projected wall to its T end.
Usage: b3_overlap_check_v1.py --a DIR/TAG (2048) --b DIR/TAG (4096 B2) [--cont DIR/TAG --cont-ref DIR/TAG --first-n 50] --from 90 --to 100
       --out OUT.json --md OUT.md"""
import argparse, json, os
import numpy as np


def load(dt):
    d, tag = os.path.split(dt)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return J, Z


def recs(J):
    return {round(r["T_over_rs"], 6): r for r in J["records"] if r.get("E") is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--cont", default=None); ap.add_argument("--cont-ref", default=None); ap.add_argument("--first-n", type=int, default=50)
    ap.add_argument("--from", dest="t0", type=float, default=90.0); ap.add_argument("--to", dest="t1", type=float, default=100.0)
    ap.add_argument("--inner-rs", type=float, default=16.0)
    ap.add_argument("--out", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    JA, ZA = load(a.a); JB, ZB = load(a.b)
    rs = float(JA["args"]["rs"]); assert rs == float(JB["args"]["rs"])
    RA = np.asarray(ZA["R"], float); RB = np.asarray(ZB["R"], float)
    n = min(len(RA), len(RB)); assert np.array_equal(RA[:n], RB[:n]), "the chains' R grids differ on the common sites"
    TA = np.asarray(ZA["T"], float) / rs; TB = np.asarray(ZB["T"], float) / rs
    ra, rb = recs(JA), recs(JB)
    rows = []
    inner = RA[:n] <= a.inner_rs * rs
    for t in sorted(set(ra) & set(rb)):
        if not (a.t0 - 1e-9 <= t <= a.t1 + 1e-9):
            continue
        x, y = ra[t], rb[t]
        ka = int(np.argmin(np.abs(TA - t))); kb = int(np.argmin(np.abs(TB - t)))
        dN = np.abs(np.asarray(ZA["N_applied"][ka], float)[:n] - np.asarray(ZB["N_applied"][kb], float)[:n])
        dv = np.abs(np.asarray(ZA["v_applied"][ka], float)[:n] - np.asarray(ZB["v_applied"][kb], float)[:n])
        row = {"t": t}
        for k in ("E_over_E0", "E", "R_t", "R_t_over_2gE", "kappa_T", "min_N_minus_absv", "r_in", "n_pocket"):
            va, vb = x.get(k), y.get(k)
            row[k] = [va, vb, (va - vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None]
        row["fields_max_dN"] = float(dN.max()); row["fields_max_dv"] = float(dv.max())
        row["fields_max_dN_inner"] = float(dN[inner].max()); row["fields_max_dv_inner"] = float(dv[inner].max())
        rows.append(row)
    res = {"a": a.a, "b": a.b, "nsites": [JA["args"]["nsites"], JB["args"]["nsites"]], "common_sites": n, "inner_rs": a.inner_rs, "rows": rows}
    if rows:
        def col(k, i=2):
            return np.array([r[k][i] for r in rows if r[k][i] is not None], float)
        res["summary"] = {k: {"max_abs_diff": float(np.max(np.abs(col(k)))) if len(col(k)) else None,
                              "a_range": [float(np.min(col(k, 0))), float(np.max(col(k, 0)))] if len(col(k, 0)) else None,
                              "b_range": [float(np.min(col(k, 1))), float(np.max(col(k, 1)))] if len(col(k, 1)) else None}
                          for k in ("E_over_E0", "R_t", "R_t_over_2gE", "kappa_T", "min_N_minus_absv", "r_in")}
        res["summary"]["fields"] = {k: float(max(r[k] for r in rows)) for k in ("fields_max_dN", "fields_max_dv", "fields_max_dN_inner", "fields_max_dv_inner")}
        res["summary"]["n_records"] = len(rows)
    if a.cont:
        JC, ZC = load(a.cont); rc = recs(JC)
        res_T = JC.get("runner", {}).get("resume", {}).get("T_over_rs") or (JC["args"].get("resume_T_over_rs"))
        bit = None
        if a.cont_ref:
            JR, _ = load(a.cont_ref); rr = recs(JR)
            common = [t for t in rc if t <= res_T + 1e-9 and t in rr]
            bit = {"records": len(common), "identical": sum(1 for t in common if json.dumps(rc[t], sort_keys=True, default=str) == json.dumps(rr[t], sort_keys=True, default=str))}
        new = [rc[t] for t in sorted(rc) if t > res_T + 1e-9]
        w = [r.get("seconds") for r in new[:a.first_n] if isinstance(r.get("seconds"), (int, float))]
        Tend = float(JC["args"]["T_over_rs"]); tau = float(JC["args"]["tau"])
        nleft = int(round((Tend - (new[-1]["T_over_rs"] if new else res_T)) * rs / tau))
        res["continuation"] = {"resume_T_over_rs": res_T, "kept_records_vs_ref": bit, "new_records": len(new),
                               "last_T_over_rs": new[-1]["T_over_rs"] if new else None,
                               "first_n_wall_s": {"n": len(w), "median": float(np.median(w)) if w else None, "max": float(np.max(w)) if w else None,
                                                  "sum": float(np.sum(w)) if w else None},
                               "projected_remaining_h": (float(np.median(w)) * nleft / 3600.0) if w else None,
                               "slices_saved": [r.get("slice_saved") for r in new if r.get("slice_saved")],
                               "slices_skipped": JC.get("runner", {}).get("skipped_slices")}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    L = ["# B3 overlap check: %s (nsites %d) against %s (nsites %d), T/r_s %g to %g" % (a.a, JA["args"]["nsites"], a.b, JB["args"]["nsites"], a.t0, a.t1), ""]
    if rows:
        s = res["summary"]
        L.append("%d common records; common sites R <= %d a (%.0f r_s). Max |a - b|: E/E0 %.3e; R_t %.3e a; R_t/2gE %.3e; kappa_T %.3e; min(N - |v|) %.3e; r_in %.3e. Applied fields on the common sites: max |dN| %.3e, |dv| %.3e; within %g r_s: %.3e, %.3e." % (
            s["n_records"], n, n / rs, s["E_over_E0"]["max_abs_diff"], s["R_t"]["max_abs_diff"] or float("nan"), s["R_t_over_2gE"]["max_abs_diff"] or float("nan"),
            s["kappa_T"]["max_abs_diff"] or float("nan"), s["min_N_minus_absv"]["max_abs_diff"], s["r_in"]["max_abs_diff"] or float("nan"),
            s["fields"]["fields_max_dN"], s["fields"]["fields_max_dv"], a.inner_rs, s["fields"]["fields_max_dN_inner"], s["fields"]["fields_max_dv_inner"]))
        L += ["", "| T/r_s | E/E0 a | E/E0 b | diff | R_t a | R_t b | min(N-abs v) a | b | max dN (all, inner) | max dv (all, inner) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows[::max(1, len(rows) // 21)]:
            L.append("| %.3f | %.7f | %.7f | %+.2e | %s | %s | %+.4f | %+.4f | %.2e, %.2e | %.2e, %.2e |" % (
                r["t"], r["E_over_E0"][0], r["E_over_E0"][1], r["E_over_E0"][2], ("%.3f" % r["R_t"][0]) if r["R_t"][0] else "-", ("%.3f" % r["R_t"][1]) if r["R_t"][1] else "-",
                r["min_N_minus_absv"][0], r["min_N_minus_absv"][1], r["fields_max_dN"], r["fields_max_dN_inner"], r["fields_max_dv"], r["fields_max_dv_inner"]))
    if a.cont:
        L += ["", "Continuation: " + json.dumps(res["continuation"], default=float)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
