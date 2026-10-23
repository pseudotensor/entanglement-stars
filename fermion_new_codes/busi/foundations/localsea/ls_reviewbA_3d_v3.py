#!/usr/bin/env python
"""ls_reviewbA_3d_v3.py (busi, inbox #455; ReviewB_found7_v1.md sec. 3 items 1 and 5, sec. 2 'A sign inconsistency'): reviewer B A
redone with the source sign fixed. ls_srcsign_v1.py shows the record's pair sources are +(sea - R2) to 1e-12 on all seven records,
so replacing the production reference R2 by P_0 + b changes the source by

    Delta s = +(eps_R2 - eps_0) - b        (v1.1 sec. 6.1's text),

whereas ls_reviewbA_3d_v2.py added b - (eps_R2 - eps_0) (the reversed sign). This script solves both signs (the reversed one only to
reproduce the v1.1 sec. 6.4 table as a check of the solve), for the variants: record; LS2_3d (b = the 3D cubic second-order sea
everywhere); LS2_3d_outside_4a; flat_P0 (b = 0). For each solve it prints the before-and-after source totals passed to
engine_fields_v16 (the module smooths them itself), the roots, min(N - |v|), the outer root's slope d(N - |v|)/dr, the collar charge
E = r w^2/(2 g), and the field changes |Delta N|, |Delta v| (max over sites, and at r = 8, 16, 32, R_t, 2 R_t). For LS2_3d with the
correct sign it then recomputes b at the returned site fields (b depends on u) and re-solves, up to --iters times, reporting the
residual max|b[u_k] - b[u_{k-1}]| relative and the change in E per step (reviewer B: the counterterm's field derivative). The record's
numerical floors (interp_floor_N, interp_floor_v, iteration_history) are copied from the record json as the scale to compare with.
Post-processing only (field solves, no eigensolves), one thread. Writes <out>/ls_reviewbA_3d_v3.json.

Inputs (read only): audit npz of ls_chain_audit_v1 mode slice; sigma/a = 4 json and fields from the resume-mirror copy whose md5s
equal those recorded in v1.1 (json d7d69adc..., fields 01444a39...); sigma/a = 8 from pod_sync."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse   # noqa: E402
import json   # noqa: E402
import sys   # noqa: E402

import numpy as np   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ls_audit_read_v1 import field_module, record_of   # noqa: E402
if "--threads" not in sys.argv:
    sys.argv += ["--threads", "1"]
from ls_chain_audit_v1 import local_sea_3d   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="<internal>/audit")
ap.add_argument("--json4", required=True); ap.add_argument("--fields4", required=True)
ap.add_argument("--json8", required=True); ap.add_argument("--fields8", required=True)
ap.add_argument("--coef", default="0.003048063693947,-0.01110837131864,-0.001773874623294")
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--out", default="<internal>/localsea")
ap.add_argument("--tags", default="s4_T07,s4_T13,s4_T14,s4_T15,s4_T16,s8_T14,s8_T16")
ap.add_argument("--threads", default="1", help="read by ls_chain_audit_v1 at import (its torch thread count)")
a = ap.parse_args()
co = dict(zip(("s1", "s2", "s3"), [float(x) for x in a.coef.split(",")]))
FM = field_module()
src = {"s4": (a.json4, a.fields4), "s8": (a.json8, a.fields8)}


def solve(eps, J, g, Rs, ell):
    _, _, _, info = FM.solve_constraints_ivp(eps, J, g, Rs, ell, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        return None
    tz = FM.trapping_radius(None, None, Rs, info)
    rc = float(info["r_collar"])
    Nd, vd = info["N_dense"], info["v_dense"]
    Ns = np.asarray(Nd(Rs), float); vs = np.asarray(vd(Rs), float)
    zeros = [float(z) for z in tz["zeros"]]
    slope = None
    if zeros:
        z = zeros[-1]; h = 1e-3
        f = lambda r: float(Nd(r)) - abs(float(vd(r)))   # noqa: E731
        slope = (f(z + h) - f(z - h)) / (2 * h)
    Nc, vc = float(Nd(rc)), float(vd(rc))
    return {"zeros": zeros, "min": tz.get("min"), "r_min": tz.get("r_min"), "E_collar": rc * (vc / Nc) ** 2 / (2 * g),
            "r_collar": rc, "outer_root_slope": slope, "_N": Ns, "_v": vs}


def pub(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


res = {"sign_note": "record pair sources = +(sea - R2) (ls_srcsign_v1); correct Delta s = +(eps_R2 - eps_0) - b"}
for tag, hist, T in (("s4_T07", "s4", 7.0), ("s4_T13", "s4", 13.0), ("s4_T14", "s4", 14.0), ("s4_T15", "s4", 15.0),
                     ("s4_T16", "s4", 16.0), ("s8_T14", "s8", 14.0), ("s8_T16", "s8", 16.0)):
    p = os.path.join(a.dir, "ls_chain_audit_v1_%s.npz" % tag)
    if tag not in a.tags.split(",") or not os.path.exists(p):
        continue
    Z = np.load(p)
    rec, _ = record_of(src[hist][0], T)
    F = np.load(src[hist][1], allow_pickle=True)
    k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - T) < 1e-9)[0])
    g, ell = float(F["g"]), float(F["ell"]); Rs = np.asarray(F["R"], float)
    eps0, J0 = np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float)
    R = Z["R"]
    pol_e = Z["eps_R2"] - Z["eps_0"]; pol_J = Z["J_R2"] - Z["J_0"]
    b_e, b_J = Z["rho3d"], Z["J3d"]
    out = {"T_over_rs": T, "record_zeros": rec.get("trapping_zeros"), "record_min": rec.get("min_N_minus_absv"), "record_E": rec.get("E"),
           "record_kappa_T": rec.get("kappa_T"), "record_interp_floor_N": rec.get("interp_floor_N"),
           "record_interp_floor_v": rec.get("interp_floor_v"), "record_iteration_history_last": (rec.get("iteration_history") or [None])[-1],
           "sources_eps_total_before": float(eps0.sum()), "sources_J_total_before": float(J0.sum()), "solves": {}}
    base = solve(eps0, J0, g, Rs, ell)
    if base is None:
        print(tag, "record solve failed", flush=True)
        res[tag] = out
        continue
    Rt = base["zeros"][-1] if base["zeros"] else None
    probe = [8.0, 16.0, 32.0] + ([Rt, 2 * Rt] if Rt else [64.0, 128.0])

    def dfields(s):
        dN = s["_N"] - base["_N"]; dv = s["_v"] - base["_v"]
        at = {("%.1f" % r): [float(np.interp(r, Rs, dN)), float(np.interp(r, Rs, dv))] for r in probe}
        return {"max_abs_dN": float(np.max(np.abs(dN))), "r_max_dN": float(Rs[np.argmax(np.abs(dN))]),
                "max_abs_dv": float(np.max(np.abs(dv))), "r_max_dv": float(Rs[np.argmax(np.abs(dv))]), "dN_dv_at": at}

    out["solves"]["record"] = pub(base)
    print(tag, "record", json.dumps(pub(base), default=float), flush=True)
    variants = []
    for sgn, slab in ((+1.0, "correct"), (-1.0, "reversed_v2")):
        de_all, dj_all = sgn * (pol_e - b_e), sgn * (pol_J - b_J)
        variants += [("LS2_3d_" + slab, de_all, dj_all),
                     ("LS2_3d_outside_4a_" + slab, np.where(R >= 4, de_all, 0.0), np.where(R >= 4, dj_all, 0.0)),
                     ("flat_P0_" + slab, sgn * pol_e, sgn * pol_J)]
    for lab, de, dj in variants:
        s = solve(eps0 + de, J0 + dj, g, Rs, ell)
        if s is None:
            out["solves"][lab] = {"failed": True, "d_eps_total": float(de.sum()), "d_J_total": float(dj.sum())}
            print(tag, lab, "solve failed", flush=True)
            continue
        o = pub(s); o.update({"d_eps_total": float(de.sum()), "d_J_total": float(dj.sum()),
                              "sources_eps_total_after": float((eps0 + de).sum()), "sources_J_total_after": float((J0 + dj).sum()),
                              "dE_rel": s["E_collar"] / base["E_collar"] - 1.0})
        o.update(dfields(s))
        out["solves"][lab] = o
        print(tag, lab, json.dumps(o, default=float), flush=True)
        if lab == "LS2_3d_correct":
            # the counterterm recomputed at the returned fields: b[u] - b[u_record], added to b on the applied fields
            bN_e, bN_J = local_sea_3d(Rs, base["_N"], base["_v"], co)
            hist_ = []; cur = s; prev_be = b_e
            for it in range(a.iters):
                ce, cJ = local_sea_3d(Rs, cur["_N"], cur["_v"], co)
                be_k, bJ_k = b_e + (ce - bN_e), b_J + (cJ - bN_J)
                resid = float(np.max(np.abs(be_k - prev_be))) / max(float(np.max(np.abs(b_e))), 1e-300)
                nxt = solve(eps0 + pol_e - be_k, J0 + pol_J - bJ_k, g, Rs, ell)
                if nxt is None:
                    hist_.append({"it": it + 1, "b_residual_rel": resid, "failed": True})
                    print(tag, "iter", it + 1, "failed", flush=True)
                    break
                hist_.append({"it": it + 1, "b_residual_rel": resid, "b_eps_total": float(be_k.sum()), "b_J_total": float(bJ_k.sum()),
                              "E_collar": nxt["E_collar"], "zeros": nxt["zeros"], "min": nxt["min"],
                              "dE_rel_vs_record": nxt["E_collar"] / base["E_collar"] - 1.0,
                              "dE_rel_vs_prev": nxt["E_collar"] / cur["E_collar"] - 1.0})
                print(tag, "iter", json.dumps(hist_[-1], default=float), flush=True)
                prev_be = be_k; cur = nxt
                if abs(hist_[-1]["dE_rel_vs_prev"]) < 1e-8:
                    break
            out["LS2_3d_correct_selfconsistent"] = hist_
            if hist_ and not hist_[-1].get("failed"):
                out["LS2_3d_correct_selfconsistent_fields"] = dfields(cur)
    res[tag] = out
    with open(os.path.join(a.out, "ls_reviewbA_3d_v3.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
print("DONE")
