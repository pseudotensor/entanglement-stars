#!/usr/bin/env python
"""fast_fields_proto_v1.py (busi lane, reviewer A redesign prototype R2): the collocation field solve against the production
engine_fields_v16.solve_constraints_ivp on stored sources (eps_tot, J_tot of mirrored records), accuracy against a production run at a
tighter tolerance and the measured factor on the box CPU.
usage: python fast_fields_proto_v1.py OUT_DIR CODE_DIR FIELDS_NPZ NREC [--ell ELL] [--p 10] [--ppe 8] [--threads 8] [--tag T]
  CODE_DIR: a production code tree with engine_fields_v4/v12/v14/v16 and engine_fields_kcache_v1 (installed as --field-kernel-cache does).
  FIELDS_NPZ: a run's fields npz (R, g, ell, eps_tot[i], J_tot[i]); the last NREC records are solved three ways:
  P12 = production at rtol 1e-12 / atol 1e-16 (the engines' setting), P14 = production at rtol 1e-14 / atol 1e-18 (the reference),
  FAST = fast_fields_v1 (p nodes, ppe panels per ell); each followed by site_fields_kernel_average (the engines' site_fields averaged).
Outputs OUT_DIR/fast_fields_proto_<tag>.json and .txt."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def diffs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir"); ap.add_argument("fields"); ap.add_argument("nrec", type=int)
    ap.add_argument("--ell", type=float, default=None); ap.add_argument("--p", type=int, default=8); ap.add_argument("--ppe", type=int, default=2)
    ap.add_argument("--tag", default="v1"); ap.add_argument("--no-ref", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_fields_v16 as FLD
    import engine_fields_v4 as F4
    import engine_fields_v12 as F12
    import engine_fields_v14 as F14
    import engine_fields_kcache_v1 as KC
    import fast_fields_v1 as FF
    KC.install([F4, F12, F14, FLD])
    os.makedirs(a.out_dir, exist_ok=True)
    z = np.load(a.fields, allow_pickle=False)
    R = np.asarray(z["R"], float); g = float(z["g"]); ell = float(z["ell"]) if a.ell is None else a.ell
    nrec_total = len(z["T"]); idx = list(range(nrec_total - a.nrec, nrec_total))
    lines = ["fast_fields_proto %s: %s, %d sites, ell %g, g %g, records %d..%d (T %.4g..%.4g); p %d, panels per ell %d; threads %s" % (
        a.tag, os.path.basename(a.fields), len(R), ell, g, idx[0], idx[-1], float(z["T"][idx[0]]), float(z["T"][idx[-1]]), a.p, a.ppe, os.environ.get("OMP_NUM_THREADS"))]
    out = []
    # warm-up: the kernel cache and the JIT-free numpy paths
    e0, j0 = np.asarray(z["eps_tot"][idx[0]], float), np.asarray(z["J_tot"][idx[0]], float)
    FLD.solve_constraints_ivp(e0, j0, g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    FF.solve_constraints_collocation(e0, j0, g, R, ell, F12, FLD, p=a.p, panels_per_ell=a.ppe)
    tP, tF, tR = [], [], []
    for i in idx:
        eps = np.asarray(z["eps_tot"][i], float); J = np.asarray(z["J_tot"][i], float)
        rec = {"i": i, "T": float(z["T"][i])}
        t0 = time.time()
        NP, vP, WP, iP = FLD.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
        okP = not iP.get("failed")
        if okP:
            NaP, vaP, _ = FLD.site_fields_kernel_average(iP, R, ell)
        t1 = time.time(); tP.append(t1 - t0)
        rec["P12_s"] = t1 - t0; rec["P12_nfev"] = (iP.get("ivp") or {}).get("nfev"); rec["P12_failed"] = iP.get("failed"); rec["P12_gauge"] = (iP.get("gauge") or {}).get("label")
        t0 = time.time()
        NF, vF, WF, iF = FF.solve_constraints_collocation(eps, J, g, R, ell, F12, FLD, p=a.p, panels_per_ell=a.ppe)
        okF = not iF.get("failed")
        if okF:
            NaF, vaF, _ = FLD.site_fields_kernel_average(iF, R, ell)
        t1 = time.time(); tF.append(t1 - t0)
        rec["FAST_s"] = t1 - t0; rec["FAST_failed"] = iF.get("failed"); rec["FAST_newton"] = iF.get("newton")
        if okF:
            rec["FAST_U_min_over_max"] = iF["U_min_over_max"]
        if not a.no_ref:
            t0 = time.time()
            NR, vR, WR, iR = FLD.solve_constraints_ivp(eps, J, g, R, ell, N0=None, rtol=1e-14, atol=1e-18, collar_policy="flag", support_rel=1e-14)
            okR = not iR.get("failed")
            if okR:
                NaR, vaR, _ = FLD.site_fields_kernel_average(iR, R, ell)
            t1 = time.time(); tR.append(t1 - t0)
            rec["P14_s"] = t1 - t0; rec["P14_nfev"] = (iR.get("ivp") or {}).get("nfev"); rec["P14_failed"] = iR.get("failed")
        else:
            okR = False
        line = "rec %d T %.4f: P12 %.3f s (nfev %s, gauge %s) FAST %.3f s (newton %s it%s)%s" % (
            i, rec["T"], rec["P12_s"], rec["P12_nfev"], rec["P12_gauge"], rec["FAST_s"], (iF.get("newton") or {}).get("iterations"), "" if okF else "; FAST failed: %s" % iF.get("failed"),
            (" P14 %.3f s (nfev %s)" % (rec["P14_s"], rec["P14_nfev"])) if not a.no_ref else "")
        if okP and okF:
            rfine = iP["rfine"]
            for lab, xP, xF in (("N_sites", NP, NF), ("v_sites", vP, vF), ("N_avg", NaP, NaF), ("v_avg", vaP, vaF), ("W_nodes", WP, WF), ("N_nodes", iP["N_nodes"], iF["N_nodes"]),
                                ("N_dense", iP["N_dense"](rfine), iF["N_dense"](rfine)), ("W_dense", iP["W_field"](rfine), iF["W_field"](rfine)), ("U_dense", iP["U_dense"](rfine), iF["U_dense"](rfine))):
                rec["FAST_vs_P12_" + lab] = diffs(xF, xP)
            rec["FAST_vs_P12_E_rel"] = abs(iF["E_collar"] - iP["E_collar"]) / abs(iP["E_collar"])
            rec["FAST_vs_P12_Nmin"] = abs(iF["N_min"] - iP["N_min"])
            rec["P12_residuals"] = [iP["residual_W"], iP["residual_invN"]]; rec["FAST_residuals"] = [iF["residual_W"], iF["residual_invN"]]
            if "W_tail" in iF and iP.get("W_tail"):
                rec["FAST_vs_P12_Etail_rel"] = abs(iF["W_tail"]["E_tail_end"] - iP["W_tail"]["E_tail_end"]) / abs(iP["W_tail"]["E_tail_end"])
            line += " | FAST-P12: N %.1e v %.1e Navg %.1e vavg %.1e Wnodes rel %.1e E rel %.1e Nmin %.1e" % (
                rec["FAST_vs_P12_N_sites"][0], rec["FAST_vs_P12_v_sites"][0], rec["FAST_vs_P12_N_avg"][0], rec["FAST_vs_P12_v_avg"][0], rec["FAST_vs_P12_W_nodes"][1], rec["FAST_vs_P12_E_rel"], rec["FAST_vs_P12_Nmin"])
        if okP and okR:
            for lab, xP, xR in (("N_sites", NP, NR), ("v_sites", vP, vR), ("N_avg", NaP, NaR), ("v_avg", vaP, vaR), ("W_nodes", WP, WR)):
                rec["P12_vs_P14_" + lab] = diffs(xP, xR)
            rec["P12_vs_P14_E_rel"] = abs(iP["E_collar"] - iR["E_collar"]) / abs(iR["E_collar"])
            line += " | P12-P14: N %.1e v %.1e Navg %.1e vavg %.1e E rel %.1e" % (rec["P12_vs_P14_N_sites"][0], rec["P12_vs_P14_v_sites"][0], rec["P12_vs_P14_N_avg"][0], rec["P12_vs_P14_v_avg"][0], rec["P12_vs_P14_E_rel"])
        if okF and okR:
            for lab, xF, xR in (("N_sites", NF, NR), ("v_sites", vF, vR), ("N_avg", NaF, NaR), ("v_avg", vaF, vaR), ("W_nodes", WF, WR)):
                rec["FAST_vs_P14_" + lab] = diffs(xF, xR)
            rec["FAST_vs_P14_E_rel"] = abs(iF["E_collar"] - iR["E_collar"]) / abs(iR["E_collar"])
            line += " | FAST-P14: N %.1e v %.1e Navg %.1e vavg %.1e E rel %.1e" % (rec["FAST_vs_P14_N_sites"][0], rec["FAST_vs_P14_v_sites"][0], rec["FAST_vs_P14_N_avg"][0], rec["FAST_vs_P14_v_avg"][0], rec["FAST_vs_P14_E_rel"])
        lines.append(line); print(line, flush=True)
        out.append(rec)
    okF_recs = [r for r in out if not r["FAST_failed"] and not r["P12_failed"]]
    summ = {"n_records": len(out), "n_fast_ok": len(okF_recs), "P12_mean_s": float(np.mean(tP)), "FAST_mean_s": float(np.mean(tF)), "factor_mean": float(np.mean(tP) / np.mean(tF)),
            "FAST_median_s": float(np.median(tF)), "P12_median_s": float(np.median(tP))}
    if tR:
        summ["P14_mean_s"] = float(np.mean(tR))
    for key in ("FAST_vs_P12_N_sites", "FAST_vs_P12_v_sites", "FAST_vs_P12_N_avg", "FAST_vs_P12_v_avg", "FAST_vs_P12_E_rel", "FAST_vs_P14_N_sites", "FAST_vs_P14_v_sites", "FAST_vs_P14_N_avg", "FAST_vs_P14_v_avg", "FAST_vs_P14_E_rel",
                "P12_vs_P14_N_sites", "P12_vs_P14_v_sites", "P12_vs_P14_N_avg", "P12_vs_P14_v_avg", "P12_vs_P14_E_rel"):
        vals = [r[key] for r in okF_recs if key in r]
        if vals:
            summ[key + "_max"] = float(np.max([v[0] if isinstance(v, (list, tuple)) else v for v in vals]))
    lines.append("SUMMARY: %s" % json.dumps(summ))
    print(lines[-1])
    with open(os.path.join(a.out_dir, "fast_fields_proto_%s.txt" % a.tag), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "fast_fields_proto_%s.json" % a.tag), "w") as f:
        json.dump({"summary": summ, "records": out, "args": vars(a)}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
