#!/usr/bin/env python
"""b7_birth_v1.py (busi lane, inbox #301, item B7 of deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md).

Locates the birth between records on each listed ladder member: the clock T_birth in (T_f - step, T_f] where
m(T) = min over sites of (N - |v|) crosses 0, T_f the member's first saved record holding a trapping zero (the readout's
definition: the record's trapping_zeros list is non-empty) and step = T_f minus the previous record's clock.

The stored dense fields (the run's <stem>_fields.npz: R, T, N, v per record) are interpolated in time, and m is evaluated
on the interpolated fields (min over the 'R' sites):
  - linear:  N, v linear in T between the last unformed record (i-1) and the first formed one (i);
  - cubic:   N, v by the Lagrange cubic in T through the 4 records i-2, i-1, i, i+1 (when all exist in the fields file);
the crossing is found by bisection (brentq) on m(T) inside the bracket. The interpolation error printed is
|T_cubic - T_linear|. Also printed, as a check, the same two interpolations applied to the stored series
m_k = min_sites(N_k - |v_k|) itself (min of interpolant vs interpolant of min), and the json's own min_N_minus_absv.
Nothing is fitted; no physics is ruled. Read-only on all inputs.

Args: --members FILE (lines: name class json_path [fields_path]), --out JSON. Fields path defaults to <json stem>_fields.npz;
an optional 4th column overrides it.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def lagrange_weights(ts, t):
    w = np.ones(len(ts))
    for a in range(len(ts)):
        for b in range(len(ts)):
            if a != b:
                w[a] *= (t - ts[b]) / (ts[a] - ts[b])
    return w


def mfun(N, v):
    return float(np.min(N - np.abs(v)))


def argm(N, v):
    return int(np.argmin(N - np.abs(v)))


def root(f, a, b):
    fa, fb = f(a), f(b)
    if not (fa > 0 > fb):
        return None, fa, fb
    return brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=200), fa, fb


def one(name, klass, jpath, fpath):
    out = {"member": name, "class": klass, "json": jpath, "fields": fpath}
    if not os.path.exists(jpath):
        out["error"] = "json missing"
        return out
    if not os.path.exists(fpath):
        out["error"] = "fields npz missing"
        return out
    out["json_bytes"] = os.path.getsize(jpath)
    out["json_md5"] = md5(jpath)
    out["fields_bytes"] = os.path.getsize(fpath)
    out["fields_md5"] = md5(fpath)
    J = json.load(open(jpath))
    A = J.get("args", {}) or {}
    rs = float(A["rs"])
    out["args"] = {k: A.get(k) for k in ("m", "g", "rs", "R0_over_rs", "nsites", "ell", "prep_ell", "tau", "kappa_sea", "mix",
                                          "T_over_rs")}
    out["script"] = J.get("script")
    recs = [r for r in J["records"] if r.get("E") is not None]  # the readout's load rule (records without E dropped)
    out["json_records"] = len(recs)
    Tj = np.array([float(r["T_over_rs"]) for r in recs])
    nz = np.array([len(r.get("trapping_zeros") or []) for r in recs])
    mj = np.array([np.nan if r.get("min_N_minus_absv") is None else float(r["min_N_minus_absv"]) for r in recs])
    zi = np.where(nz > 0)[0]
    if len(zi) == 0:
        out["error"] = "no record holds a trapping zero"
        return out
    i = int(zi[0])
    out["T_f"] = float(Tj[i])
    out["T_prev"] = float(Tj[i - 1]) if i > 0 else None
    out["step"] = float(Tj[i] - Tj[i - 1]) if i > 0 else None
    out["json_minNv_prev"] = float(mj[i - 1]) if i > 0 else None
    out["json_minNv_f"] = float(mj[i])
    out["zeros_f"] = recs[i].get("trapping_zeros")
    out["R_t_f"] = recs[i].get("R_t")
    out["converged_window"] = {str(round(float(Tj[k]), 6)): recs[k].get("converged") for k in range(max(0, i - 2), min(len(recs), i + 2))}

    z = np.load(fpath, allow_pickle=False)
    R = np.asarray(z["R"], dtype=float)
    Tf_all = np.asarray(z["T"], dtype=float) / rs
    out["fields_rows"] = int(len(Tf_all))
    out["fields_sites"] = int(len(R))

    def row(t):
        k = np.where(np.abs(Tf_all - t) < 1e-7)[0]
        return int(k[-1]) if len(k) else None  # last match (a composite's later replay of a clock, if any)

    want = [i - 2, i - 1, i, i + 1]
    rows = {}
    for k in want:
        if 0 <= k < len(recs):
            rows[k] = row(Tj[k])
    out["fields_rows_used"] = {str(round(float(Tj[k]), 6)): rows[k] for k in rows}
    if rows.get(i) is None or rows.get(i - 1) is None:
        out["error"] = "the bracket's records are not in the fields file"
        return out
    NV = {}
    for k, rk in rows.items():
        if rk is not None:
            NV[k] = (np.asarray(z["N"][rk], dtype=float), np.asarray(z["v"][rk], dtype=float))
    # the stored series and the check against the json's own minimum
    ms = {k: mfun(*NV[k]) for k in NV}
    out["fields_minNv"] = {str(round(float(Tj[k]), 6)): ms[k] for k in sorted(ms)}
    out["fields_minNv_site_r"] = {str(round(float(Tj[k]), 6)): float(R[argm(*NV[k])]) for k in sorted(ms)}
    out["fields_vs_json_minNv"] = {str(round(float(Tj[k]), 6)): (ms[k] - mj[k]) if np.isfinite(mj[k]) else None for k in sorted(ms)}

    t0, t1 = float(Tj[i - 1]), float(Tj[i])
    # (a) linear interpolation of the dense fields
    N0, v0 = NV[i - 1]
    N1, v1 = NV[i]

    def m_lin(t):
        s = (t - t0) / (t1 - t0)
        return mfun((1 - s) * N0 + s * N1, (1 - s) * v0 + s * v1)

    tl, fa, fb = root(m_lin, t0, t1)
    out["bracket_m"] = [fa, fb]
    out["T_birth_lin"] = tl
    if tl is not None:
        s = (tl - t0) / (t1 - t0)
        out["r_birth_lin"] = float(R[argm((1 - s) * N0 + s * N1, (1 - s) * v0 + s * v1)])
    # (b) cubic (Lagrange) through 4 records
    c4 = [k for k in (i - 2, i - 1, i, i + 1) if k in NV]
    out["cubic_records"] = [float(Tj[k]) for k in c4]
    if len(c4) == 4:
        ts = np.array([float(Tj[k]) for k in c4])
        Ns = np.array([NV[k][0] for k in c4])
        Vs = np.array([NV[k][1] for k in c4])

        def m_cub(t):
            w = lagrange_weights(ts, t)
            return mfun(w @ Ns, w @ Vs)

        tc, ca, cb = root(m_cub, t0, t1)
        out["T_birth_cubic"] = tc
        if tc is not None:
            w = lagrange_weights(ts, tc)
            out["r_birth_cubic"] = float(R[argm(w @ Ns, w @ Vs)])
        # (c) the same on the stored min series (check)
        mser = np.array([ms[k] for k in c4])
        out["T_series_cubic"] = root(lambda t: float(lagrange_weights(ts, t) @ mser), t0, t1)[0]
    else:
        out["T_birth_cubic"] = None
        out["cubic_note"] = "fewer than 4 of the records i-2..i+1 are in the fields file"
    a0, a1 = ms[i - 1], ms[i]
    out["T_series_lin"] = t0 + (t1 - t0) * a0 / (a0 - a1) if a0 > 0 > a1 else None
    if out.get("T_birth_lin") is not None and out.get("T_birth_cubic") is not None:
        out["interp_error"] = abs(out["T_birth_cubic"] - out["T_birth_lin"])
        out["shift_cubic"] = out["T_birth_cubic"] - out["T_f"]
    if out.get("T_birth_lin") is not None:
        out["shift_lin"] = out["T_birth_lin"] - out["T_f"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    res = []
    for line in open(a.members):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        name, klass, jpath = p[0], p[1], p[2]
        if only and name not in only:
            continue
        fpath = p[3] if len(p) > 3 else jpath[:-5] + "_fields.npz"
        try:
            o = one(name, klass, jpath, fpath)
        except Exception as e:  # report and continue
            o = {"member": name, "class": klass, "json": jpath, "fields": fpath, "error": "%s: %s" % (type(e).__name__, e)}
        res.append(o)
        print(json.dumps({k: o.get(k) for k in ("member", "T_f", "step", "T_birth_lin", "T_birth_cubic", "interp_error",
                                                 "T_series_lin", "T_series_cubic", "error")}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
