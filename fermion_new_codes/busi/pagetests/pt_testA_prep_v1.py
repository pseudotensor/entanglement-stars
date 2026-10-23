#!/usr/bin/env python
"""pt_testA_prep_v1.py (busi, inbox #428, Test A): the comparator L_H at the record's charge and the re-solved field table for the
prescribed recession (pt_fieldsA_v1). Box CPU. Writes <out>/testA_prep_<label>_T<T>.json and _table.npz.

L_H: the paper's greybody flux of the channels k = 1 and 2 (page_history_v6q.continuum_comparator: sum_k (64 k/2 pi) int_0^1.5 dx x
Gamma_k(x) f(x/(r_s T)) / r_s^2 on 48 Gauss-Legendre nodes, Gamma_k from page_scatter_v5.gamma_continuum) at T_H = 1/(4 pi r_s),
r_s = 2 g E_rec; beside it the full massless luminosity sum over |k| <= 8, against the paper's L r_s^2 = 0.005237 (a check of Gamma).
Checks printed: the lambda = 1 re-solve against the record's stored site rows N, v and its applied rows; the spline in lambda against
direct re-solves at 4 midpoints; the dense roots along the table."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "code_v6"))
import pt_fieldsA_v1 as FA  # noqa: E402
import pt_lib_v1 as L  # noqa: E402


def comparator(kappas, rs, T):
    import page_scatter_v5 as SC
    xs, ws = np.polynomial.legendre.leggauss(48)
    x = 0.75 * (xs + 1.0); w = 0.75 * ws
    tot = {}
    for k in kappas:
        G = []
        for xi in x:
            try:
                G.append(SC.gamma_continuum(float(xi), float(k))[0])
            except RuntimeError:
                G.append(0.0)
        G = np.array(G)
        tot[k] = 64.0 * k / (2.0 * math.pi) * float(np.sum(w * x * G / (np.exp(np.clip(x / (rs * T), -700, 700)) + 1.0))) / (rs * rs)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="P64_s4_B2n4096")
    ap.add_argument("--T-rec-rs", type=float, default=100.0)
    ap.add_argument("--Leff-max", type=float, default=30.0, help="largest L_eff in units of L_H")
    ap.add_argument("--span-a", type=float, default=150000.0, help="the longest post-record clock span (lattice units)")
    ap.add_argument("--n-lam", type=int, default=33)
    ap.add_argument("--lam-lo", type=float, default=None, help="the table's lowest lambda (default 1 - 1.4 dE/E; E falls slower than lambda)")
    ap.add_argument("--suffix", default="", help="write to the stem with this suffix (then mv into place)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    H = FA.load_record(a.label, a.T_rec_rs)
    g, E = H["g"], H["E_rec"]
    rs = 2.0 * g * E
    TH = 1.0 / (4.0 * math.pi * rs)
    print("record %s T/r_s %.4f (T %.1f a): E %.8f, 2 g E %.6f, T_H %.6e, kappa_T %s (4 g E kappa_T %s), R_t %s, zeros %s" % (
        a.label, a.T_rec_rs, H["T_rec"], E, rs, TH, H["rec"]["kappa_T"], H["rec"]["four_gE_kappa_T"], H["rec"]["R_t"], H["rec"]["trapping_zeros"]), flush=True)
    c12 = comparator((1, 2), rs, TH)
    call = comparator(tuple(range(1, 9)), rs, TH)
    LH = c12[1] + c12[2]
    Ltot = sum(call.values())
    print("L_H (k = 1, 2, weights 64 k/2 pi) %.6e: k=1 %.6e, k=2 %.6e; the full |k| <= 8 luminosity %.6e, L r_s^2 = %.6f (paper 0.005237); L_H/L_tot %.4f" % (
        LH, c12[1], c12[2], Ltot, Ltot * rs * rs, LH / Ltot), flush=True)
    dE = a.Leff_max * LH * a.span_a
    lam_lo = a.lam_lo if a.lam_lo is not None else 1.0 - 1.4 * dE / E
    lam = np.linspace(lam_lo, 1.0, a.n_lam)
    print("largest charge loss %.4f (%.3f percent of E): lambda grid [%.5f, 1] with %d nodes" % (dE, 100 * dE / E, lam_lo, a.n_lam), flush=True)
    tab = FA.resolve_table(H, lam)
    i1 = int(np.argmin(np.abs(lam - 1.0)))
    chk = {"N_resolved_minus_site_row": float(np.max(np.abs(tab["N"][i1] - H["N_site"]))), "v_resolved_minus_site_row": float(np.max(np.abs(tab["v"][i1] - H["v_site"]))),
           "N_resolved_minus_applied": float(np.max(np.abs(tab["N"][i1] - H["N_app"][H["k_rec"]]))),
           "v_resolved_minus_applied": float(np.max(np.abs(tab["v"][i1] - H["v_app"][H["k_rec"]]))),
           "E_resolved_minus_record": float(tab["E"][i1] - E)}
    print("lambda = 1 re-solve: %s" % json.dumps(chk), flush=True)
    # the spline against direct re-solves at midpoints
    from scipy.interpolate import CubicSpline
    Ns = CubicSpline(lam, tab["N"], axis=0); vs = CubicSpline(lam, tab["v"], axis=0); Es = CubicSpline(lam, tab["E"])
    mids = [0.5 * (lam[j] + lam[j + 1]) for j in (0, a.n_lam // 3, 2 * a.n_lam // 3, a.n_lam - 2)]
    direct = FA.resolve_table(H, mids)
    sp = []
    for j, lm in enumerate(mids):
        sp.append({"lambda": lm, "max_dN": float(np.max(np.abs(Ns(lm) - direct["N"][j]))), "max_dv": float(np.max(np.abs(vs(lm) - direct["v"][j]))),
                   "dE": float(Es(lm) - direct["E"][j])})
    print("spline against direct re-solves: %s" % json.dumps(sp), flush=True)
    stem = os.path.join(a.out, "testA_prep_%s_T%g%s" % (a.label, a.T_rec_rs, a.suffix))
    if tab["E"].min() > E - dE:
        print("WARNING: the table's lowest charge %.6f is above the largest run's %.6f" % (tab["E"].min(), E - dE), flush=True)
    np.savez(stem + "_table.npz", lam=tab["lam"], N=tab["N"], v=tab["v"], E=tab["E"], zeros=np.array([json.dumps(z) for z in tab["zeros"]]),
             mins=np.array([m if m is not None else np.nan for m in tab["min"]], float))
    J = {"script": "pt_testA_prep_v1.py", "args": vars(a), "record": {k: H[k] for k in ("label", "npz", "json", "npz_md5", "json_md5", "rs_nom", "T_rec", "E_rec", "g", "ell", "tau", "k_rec")},
         "record_json": H["rec"], "rs_E": rs, "T_H": TH, "L_H": LH, "L_H_k": c12, "L_total_k": call, "L_total": Ltot, "L_total_rs2": Ltot * rs * rs,
         "lambda_grid": list(map(float, lam)), "E_grid": list(map(float, tab["E"])), "zeros_grid": tab["zeros"], "min_grid": tab["min"],
         "lambda1_check": chk, "spline_check": sp, "engine_md5s": L.engine_md5s(), "field_module_md5": RH_md5(), "seconds": time.time() - t0}
    json.dump(J, open(stem + ".json", "w"), indent=1, default=float)
    print("wrote %s.json and _table.npz (%.0f s)" % (stem, time.time() - t0), flush=True)


def RH_md5():
    import read_history_v1 as RH
    return RH.md5(os.path.join(RH.FM_DIR, "engine_fields_v16.py"))


if __name__ == "__main__":
    main()
