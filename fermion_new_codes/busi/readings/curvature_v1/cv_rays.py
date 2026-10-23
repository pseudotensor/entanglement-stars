#!/usr/bin/env python
"""cv_rays.py (busi lane, correctness reading of CURVATURE_readout_v1 sec. 5): own tracer for the radial null rays
  dr/dT = v + s N (s = +1 out, -1 in),  d ln eps/dT = -(v' + s N'),  d lambda/dT = N/eps  (eps = 1 at launch),
on the unbarred quintic-parity-spline fields rebuilt from the stored nodes, tabulated on a parity-extended radial grid of spacing --table
(0.25a here; the delivery used 0.5a), linear in r on that table and linear in T between the records kept at cadence c (records kform,
kform + c, ...). RK4 with dT = tau * --dt-frac (the delivery: tau/8). Launched at the formation record (the first record holding a trapping
zero) from every node out to 3 r_out plus the labelled radii (0.25a, r_in/2, r_in, mid, r_out, 1.5, 2, 4 r_out), both directions.
Reflection at the centre is the continuation through r < 0 on the parity-extended fields (exact: the odd v and even N make the ingoing
equation at r < 0 the outgoing one at |r|). Rays leaving the table (|r| > r_max - 2a) are frozen as escaped.
Usage: python cv_rays.py --member B2 --out DIR [--cadence 1] [--dt-frac 0.125] [--table 0.25] [--tag x]"""
import argparse
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_common as CC  # noqa: E402

LABELS = ["centre", "r_in/2", "r_in", "mid", "r_out", "1.5 r_out", "2 r_out", "4 r_out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cadence", type=int, default=1); ap.add_argument("--dt-frac", type=float, default=0.125); ap.add_argument("--table", type=float, default=0.25)
    ap.add_argument("--tag", default=""); ap.add_argument("--rmax-table", type=float, default=None)
    a = ap.parse_args()
    t0 = time.time()
    M = CC.load(a.member, with_handoff=False)
    T, Trs, rs, tau, rn = M["T"], M["Trs"], M["rs"], M["tau"], M["rn"]
    nrec = len(T)
    kform = CC.first_zero_record(M)
    assert kform is not None
    c = a.cadence
    keep = np.arange(kform, nrec, c); Tk = T[keep]; nk = len(keep)
    rmax = float(rn[-1]) if a.rmax_table is None else min(float(rn[-1]), a.rmax_table)
    h = a.table
    rr = np.arange(-rmax, rmax + 1e-9, h)
    G = {key: np.zeros((nk, len(rr))) for key in ("N", "v", "Nr", "vr")}
    for i, k in enumerate(keep):
        sN = CC.parity_spline(rn, M["Nn"][k], +1); sv = CC.parity_spline(rn, M["vn"][k], -1)
        G["N"][i] = sN(rr); G["v"][i] = sv(rr); G["Nr"][i] = sN(rr, 1); G["vr"][i] = sv(rr, 1)
    z0 = M["zeros"][kform]; r_in, r_out = float(min(z0)), float(max(z0))
    base = {"centre": 0.25, "r_in/2": 0.5 * r_in, "r_in": r_in, "mid": 0.5 * (r_in + r_out), "r_out": r_out, "1.5 r_out": 1.5 * r_out, "2 r_out": 2.0 * r_out, "4 r_out": 4.0 * r_out}
    nodes = [float(x) for x in rn if 0 < x <= 3.0 * r_out]
    r0s = np.array(sorted(set([round(x, 6) for x in list(base.values()) + nodes])))
    labels = []
    for x in r0s:
        lab = ""
        for nm, val in base.items():
            if abs(x - round(val, 6)) < 1e-9:
                lab = nm
        labels.append(lab)
    n0 = len(r0s)
    r = np.concatenate([r0s, r0s]).astype(float); s = np.concatenate([np.ones(n0), -np.ones(n0)])
    le = np.zeros(2 * n0); lam = np.zeros(2 * n0); alive = np.ones(2 * n0, bool)
    t_c = np.full(2 * n0, np.nan); lam_c = np.full(2 * n0, np.nan); le_c = np.full(2 * n0, np.nan)
    rmin = np.abs(r.copy()); le_max = np.zeros(2 * n0)
    dt = tau * a.dt_frac
    rlim = rr[-1] - 2.0

    def fields(t, x):
        j = np.clip(np.searchsorted(Tk, t, side="right") - 1, 0, nk - 2)
        w = (t - Tk[j]) / (Tk[j + 1] - Tk[j])
        xi = (x - rr[0]) / h
        i = np.clip(np.floor(xi).astype(int), 0, len(rr) - 2); u = xi - i
        out = []
        for key in ("N", "v", "Nr", "vr"):
            A = G[key]
            out.append((A[j, i] * (1 - u) + A[j, i + 1] * u) * (1 - w) + (A[j + 1, i] * (1 - u) + A[j + 1, i + 1] * u) * w)
        return out

    def rhs(t, x, ss, l_):
        N, v, Nr, vr = fields(t, x)
        return v + ss * N, -(vr + ss * Nr), N * np.exp(-l_)

    t = float(T[kform]); tend = float(Tk[-1])
    nstep_per_rec = int(round(c * tau / dt))
    assert abs(nstep_per_rec * dt - c * tau) < 1e-9
    traj_t = [t]; traj_r = [r.copy()]; traj_lam = [lam.copy()]; traj_le = [le.copy()]
    nstep = 0
    while t < tend - 1e-9:
        hh = min(dt, tend - t)
        a_ = alive; idx = np.where(a_)[0]
        x0, l0, m0, ss = r[a_], le[a_], lam[a_], s[a_]
        k1 = rhs(t, x0, ss, l0)
        k2 = rhs(t + hh / 2, x0 + hh / 2 * k1[0], ss, l0 + hh / 2 * k1[1])
        k3 = rhs(t + hh / 2, x0 + hh / 2 * k2[0], ss, l0 + hh / 2 * k2[1])
        k4 = rhs(t + hh, x0 + hh * k3[0], ss, l0 + hh * k3[1])
        x1 = x0 + hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        l1 = l0 + hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        m1 = m0 + hh / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        cross = ((x0 > 0) & (x1 <= 0)) | ((x0 < 0) & (x1 >= 0))
        for ii in np.where(cross)[0]:
            gidx = idx[ii]
            if np.isnan(t_c[gidx]):
                f = x0[ii] / (x0[ii] - x1[ii])
                t_c[gidx] = t + f * hh; lam_c[gidx] = m0[ii] + f * (m1[ii] - m0[ii]); le_c[gidx] = l0[ii] + f * (l1[ii] - l0[ii])
        r[a_] = x1; le[a_] = l1; lam[a_] = m1
        rmin[a_] = np.minimum(rmin[a_], np.abs(x1)); le_max[a_] = np.maximum(le_max[a_], l1)
        alive = alive & ~(np.abs(r) > rlim)
        t += hh; nstep += 1
        if nstep % nstep_per_rec == 0 or t >= tend - 1e-9:
            traj_t.append(t); traj_r.append(r.copy()); traj_lam.append(lam.copy()); traj_le.append(le.copy())
    traj_t = np.array(traj_t); traj_r = np.array(traj_r); traj_lam = np.array(traj_lam); traj_le = np.array(traj_le)
    # end state
    zl = M["zeros"][keep[-1]]
    Ne, ve, _, _ = fields(tend - 1e-12, np.where(alive, r, 0.0))
    sgn = np.sign(np.where(r == 0, 1.0, r))
    drdT_end = np.where(alive, (ve + s * Ne) * sgn, np.nan)
    reff = np.abs(r); seff = s * sgn
    Kt2_end = np.where(alive, (ve / (np.maximum(reff, 1e-9) * Ne)) ** 2, np.nan)
    status = []
    for i in range(2 * n0):
        if not alive[i]:
            status.append("escaped (left the table)"); continue
        x = reff[i]
        if len(zl) >= 2:
            lo, hi = min(zl), max(zl)
            status.append(("outside the outer root, outgoing: escaping" if seff[i] > 0 else "outside the outer root, ingoing") if x > hi else
                          ("between the roots (trapped shell)" if x >= lo else "inside the inner root (untrapped core)"))
        elif len(zl) == 1:
            status.append("beyond the single zero" if x > zl[0] else "inside the single zero")
        else:
            status.append("no trapping zero at the window's end")

    def last5(arr_t, arr, i):
        j5 = int(np.searchsorted(arr_t, arr_t[-1] - 5.0 * rs)); j5 = min(max(j5, 0), len(arr_t) - 1)
        return float(arr[-1, i] - arr[j5, i]), float(arr_t[-1] - arr_t[j5])
    rows = []
    for i in range(2 * n0):
        dl5, dT5 = last5(traj_t, traj_lam, i); de5, _ = last5(traj_t, traj_le, i)
        rows.append({"label": labels[i % n0], "r0": float(r0s[i % n0]), "dir": "out" if s[i] > 0 else "in", "alive": bool(alive[i]),
                     "T_centre_over_rs": (None if np.isnan(t_c[i]) else float(t_c[i] / rs)), "lam_centre": (None if np.isnan(lam_c[i]) else float(lam_c[i])),
                     "eps_centre": (None if np.isnan(le_c[i]) else float(np.exp(le_c[i]))), "r_min": float(rmin[i]),
                     "r_end": float(reff[i]), "eps_end": float(np.exp(le[i])), "log10_eps_end": float(le[i] / np.log(10.0)), "eps_max": float(np.exp(le_max[i])),
                     "lam_end": float(lam[i]), "dlam_last_5rs": dl5, "dlneps_dT_last_5rs": (de5 / dT5 if dT5 > 0 else None),
                     "drdT_end": (None if np.isnan(drdT_end[i]) else float(drdT_end[i])), "Kt2_end": (None if np.isnan(Kt2_end[i]) else float(Kt2_end[i])), "status": status[i]})
    # samples at round T/r_s marks (for the window-cut comparison, e.g. 40 on B2)
    marks = {}
    for mk in (20.0, 30.0, 40.0, 50.0, 75.0, 100.0):
        jj = np.where(np.abs(traj_t / rs - mk) < 1e-9)[0]
        if len(jj):
            jj = int(jj[0]); j5 = int(np.searchsorted(traj_t, traj_t[jj] - 5.0 * rs))
            marks[str(mk)] = {"r": [float(abs(x)) for x in traj_r[jj]], "log10_eps": [float(x / np.log(10.0)) for x in traj_le[jj]], "lam": [float(x) for x in traj_lam[jj]],
                              "dlam_last_5rs": [float(x) for x in (traj_lam[jj] - traj_lam[j5])], "dlneps_dT_last_5rs": [float(x) for x in (traj_le[jj] - traj_le[j5]) / (traj_t[jj] - traj_t[j5])],
                              "zeros": M["zeros"][int(np.where(np.abs(Trs - mk) < 1e-9)[0][0])] if (np.abs(Trs - mk) < 1e-9).any() else None}
    out = {"member": a.member, "cadence": c, "dt": dt, "table": h, "rmax_table": rmax, "nsteps": nstep, "T_start_over_rs": float(T[kform] / rs), "T_end_over_rs": float(tend / rs),
           "formation_k": int(kform), "zeros_formation": z0, "zeros_end": zl, "n_launch_radii": n0, "rows": rows, "marks": marks,
           "fields_md5": CC.md5(M["files"]["fields"]), "json_md5": CC.md5(M["files"]["json"]), "seconds": time.time() - t0}
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or ("c%d_dt%g_h%g" % (c, a.dt_frac, h))
    json.dump(out, open(os.path.join(a.out, "cv_rays_%s_%s.json" % (a.member, tag)), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(a.out, "cv_rays_%s_%s.npz" % (a.member, tag)), traj_t=traj_t, traj_r=traj_r, traj_lam=traj_lam, traj_le=traj_le,
                        r0=np.concatenate([r0s, r0s]), s=s, labels=np.array(labels + labels))
    nin = sum(1 for x in rows if x["dir"] == "in"); reach = sum(1 for x in rows if x["dir"] == "in" and x["T_centre_over_rs"] is not None)
    esc = sum(1 for x in rows if "escap" in x["status"]); trap = sum(1 for x in rows if x["status"].startswith("between")); core = sum(1 for x in rows if "untrapped core" in x["status"])
    print("%s c%d dt %.3g h %.3g: %d launches, %d steps, %.0f s; in reaching r=0 %d/%d; escaping %d, shell %d, core %d; zeros end %s" % (
        a.member, c, dt, h, 2 * n0, nstep, out["seconds"], reach, nin, esc, trap, core, zl), flush=True)


if __name__ == "__main__":
    main()
