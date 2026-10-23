#!/usr/bin/env python
"""ls_scale_read_v3.py (busi, inbox #455; ReviewA_found7_v1.md sec. 3, ReviewB_found7_v1.md sec. 3 items 2 to 5): the reader of
the lambda scaling at s4_T15 (ls_audit_read_v2 part scale, with the lambda = 3 output under its _fromnpz name, the lambda = 1
recovery check, an extrapolation uncertainty, and the source sign of ls_srcsign_v1).

Shell totals per site of the chain's negative-set polarization at r = lambda R_n on the lambda-scaled configuration, for
lambda = 1, 2, 3, are fitted as e2 + e4/lambda^2 + e6/lambda^4 (exact on three points; e2 = the chain's own second-order local sea,
e2 + e4 = through fourth order). Uncertainty of e2 (reviewer B item 3): the spread over three other readings, (i) the two-point
e2 + e4/lambda^2 on lambda = 2, 3; (ii) e2 + e1/lambda + e4/lambda^2 (odd orders allowed); (iii) the two-point on lambda = 1, 2. Each
is printed window by window, energy and momentum; the uncertainty quoted is max |e2(reading) - e2(main)|.

reviewer B A with the chain's own candidate (sign: record pairs = +(sea - R2), so Delta s = +(pol - b)): field solves with b = e2 (LS2),
b = e2 + e4 (LS4), LS2 outside r = 4a, and b = 0 (flat_P0), each reporting roots, min(N - |v|), the outer root's slope, E and the
field changes. The sigma/a = 4 json and fields are the resume-mirror copy whose md5s equal v1.1's recorded ones. One thread.
Writes <out>/ls_scale_read_v3.json."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse   # noqa: E402
import json   # noqa: E402
import sys   # noqa: E402

import numpy as np   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ls_audit_read_v1 import field_module, record_of, windows, wsum   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="<internal>/audit")
ap.add_argument("--json4", required=True); ap.add_argument("--fields4", required=True)
ap.add_argument("--out", default="<internal>/localsea")
a = ap.parse_args()
names = {1: "ls_chain_audit_v1_s4_T15_lam1.npz", 2: "ls_chain_audit_v1_s4_T15_lam2.npz", 3: "ls_chain_audit_v1_s4_T15_lam3_fromnpz.npz"}
L = {lam: np.load(os.path.join(a.dir, p)) for lam, p in names.items()}
chk = json.load(open(os.path.join(a.dir, "ls_chain_audit_v1_s4_T15_lam1_fromnpz.json"))).get("check_vs_stored")
R = L[1]["R"]
idx = {lam: (lam * R).astype(int) - 1 for lam in L}
E = np.stack([L[lam]["eps_pol"][idx[lam]] for lam in (1, 2, 3)]); Jp = np.stack([L[lam]["J_pol"][idx[lam]] for lam in (1, 2, 3)])
lam = np.array([1.0, 2.0, 3.0])


def fit(rows_, cols, Y):
    A = np.array([[c(l_) for c in cols] for l_ in lam[rows_]])
    return np.linalg.solve(A, Y[rows_])


one = lambda l_: 1.0   # noqa: E731
inv2 = lambda l_: l_ ** -2   # noqa: E731
inv4 = lambda l_: l_ ** -4   # noqa: E731
inv1 = lambda l_: l_ ** -1   # noqa: E731
readings = {}
for Y, lab in ((E, "eps"), (Jp, "J")):
    main = fit([0, 1, 2], [one, inv2, inv4], Y)
    readings[lab] = {"main": main,
                     "two_23": fit([1, 2], [one, inv2], Y), "odd": fit([0, 1, 2], [one, inv1, inv2], Y),
                     "two_12": fit([0, 1], [one, inv2], Y)}
e2, e4, e6 = readings["eps"]["main"]; j2, j4, j6 = readings["J"]["main"]
rec, _ = record_of(a.json4, 15.0)
Rt = rec["R_t"]
rows = []
for (lo, hi) in windows(Rt, int(R[-1])):
    r_ = {"r": [lo, hi]}
    for lab, Y, rd in (("eps", E, readings["eps"]), ("J", Jp, readings["J"])):
        m2 = wsum(R, rd["main"][0], lo, hi)
        alt = {k: wsum(R, rd[k][0], lo, hi) for k in ("two_23", "odd", "two_12")}
        r_.update({lab + "_prod": wsum(R, Y[0], lo, hi), lab + "_lam2": wsum(R, Y[1], lo, hi), lab + "_lam3": wsum(R, Y[2], lo, hi),
                   lab + "_2": m2, lab + "_4": wsum(R, rd["main"][1], lo, hi), lab + "_6plus": wsum(R, rd["main"][2], lo, hi),
                   lab + "_2_alt": alt, lab + "_2_unc": max(abs(v - m2) for v in alt.values())})
        r_[lab + "_prod_minus_LS2"] = r_[lab + "_prod"] - m2
        r_[lab + "_prod_minus_LS4"] = r_[lab + "_prod"] - m2 - r_[lab + "_4"]
    rows.append(r_)
    print("r %7.1f to %7.1f | eps prod %.4e lam2 %.4e lam3 %.4e | e2 %.4e +- %.1e e4 %.4e e6+ %.2e | prod-LS2 %.3e prod-LS4 %.3e" % (
        lo, hi, r_["eps_prod"], r_["eps_lam2"], r_["eps_lam3"], r_["eps_2"], r_["eps_2_unc"], r_["eps_4"], r_["eps_6plus"],
        r_["eps_prod_minus_LS2"], r_["eps_prod_minus_LS4"]), flush=True)
    print("                       | J   prod %.4e lam2 %.4e lam3 %.4e | j2 %.4e +- %.1e j4 %.4e j6+ %.2e | prod-LS2 %.3e prod-LS4 %.3e" % (
        r_["J_prod"], r_["J_lam2"], r_["J_lam3"], r_["J_2"], r_["J_2_unc"], r_["J_4"], r_["J_6plus"], r_["J_prod_minus_LS2"],
        r_["J_prod_minus_LS4"]), flush=True)

FM = field_module()
F = np.load(a.fields4, allow_pickle=True)
k = int(np.flatnonzero(np.abs(np.asarray(F["T"], float) / 64.0 - 15.0) < 1e-9)[0])
g, ell = float(F["g"]), float(F["ell"]); Rs = np.asarray(F["R"], float)
eps0, J0 = np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float)


def solve(eps, J):
    _, _, _, info = FM.solve_constraints_ivp(eps, J, g, Rs, ell, collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        return None
    tz = FM.trapping_radius(None, None, Rs, info)
    rc = float(info["r_collar"]); Nd, vd = info["N_dense"], info["v_dense"]
    zeros = [float(z) for z in tz["zeros"]]; slope = None
    if zeros:
        z = zeros[-1]; h = 1e-3
        f = lambda r: float(Nd(r)) - abs(float(vd(r)))   # noqa: E731
        slope = (f(z + h) - f(z - h)) / (2 * h)
    return {"zeros": zeros, "min": tz.get("min"), "r_min": tz.get("r_min"), "E_collar": rc * (float(vd(rc)) / float(Nd(rc))) ** 2 / (2 * g),
            "outer_root_slope": slope, "_N": np.asarray(Nd(Rs), float), "_v": np.asarray(vd(Rs), float)}


pol_e, pol_J = E[0], Jp[0]
base = solve(eps0, J0)
solves = {"record": {k_: v for k_, v in base.items() if not k_.startswith("_")}}
print("record", json.dumps(solves["record"], default=float), flush=True)
for lab, de, dj in (("LS2", pol_e - e2, pol_J - j2), ("LS4", pol_e - e2 - e4, pol_J - j2 - j4),
                    ("LS2_outside_4a", np.where(R >= 4, pol_e - e2, 0.0), np.where(R >= 4, pol_J - j2, 0.0)),
                    ("flat_P0", pol_e, pol_J)):
    s = solve(eps0 + de, J0 + dj)
    if s is None:
        solves[lab] = {"failed": True}
        print(lab, "failed", flush=True)
        continue
    dN = s["_N"] - base["_N"]; dv = s["_v"] - base["_v"]
    o = {k_: v for k_, v in s.items() if not k_.startswith("_")}
    o.update({"d_eps_total": float(de.sum()), "d_J_total": float(dj.sum()), "dE_rel": s["E_collar"] / base["E_collar"] - 1.0,
              "max_abs_dN": float(np.max(np.abs(dN))), "r_max_dN": float(Rs[np.argmax(np.abs(dN))]),
              "max_abs_dv": float(np.max(np.abs(dv))), "r_max_dv": float(Rs[np.argmax(np.abs(dv))])})
    solves[lab] = o
    print(lab, json.dumps(o, default=float), flush=True)
res = {"T_over_rs": 15.0, "R_t": Rt, "lam1_recovery_check": chk, "windows": rows, "solves": solves, "record_E": rec.get("E"),
       "record_zeros": rec.get("trapping_zeros"), "record_min": rec.get("min_N_minus_absv"), "record_interp_floor_N": rec.get("interp_floor_N"),
       "record_interp_floor_v": rec.get("interp_floor_v"), "inputs": {"json4": a.json4, "fields4": a.fields4, "npz": names}}
with open(os.path.join(a.out, "ls_scale_read_v3.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print("DONE")
