#!/usr/bin/env python
"""b7_birth_v2.py (busi lane, inbox #301, item B7 of deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md).

Locates the birth between records on each listed ladder member: the clock T_birth in (T_f - step, T_f] where
m(T) = min over r of (N - |v|) crosses 0, with T_f the first saved record holding a trapping zero (the readout's definition) and
step = T_f minus the previous record's clock.

Why v2: the run's <stem>_fields.npz stores N, v as SITE AVERAGES (site_fields 'averaged': the kernel average of the dense functions
onto the sites, engine_fields_v12.site_fields_kernel_average), while the readout's trapping zeros and min_N_minus_absv are taken on
the DENSE functions (engine_fields_v12.trapping_radius: grid ell/64, brentq, bounded minimization). v1 (b7_birth_v1.py) interpolated
the site averages; their minimum sits 0.004 to 0.10 above the json's on every record, and on 9 members it does not change sign in the
bracket. v2 rebuilds the dense fields of each record from the same npz's stored sources (eps_tot, J_tot) with the field module the
record names (engine_fields_v14 or v16 solve_constraints_ivp, the run's rtol/atol/collar_policy/support_rel), checks the rebuild
against the json (min_N_minus_absv and the trapping zeros of each record used: a known-answer control), then interpolates the dense
fields in time on a common grid r = 0 .. r_collar, spacing ell/64 (the engine's grid), the minimum refined by a parabola through the
3 grid points around the grid minimum:
  - linear: N(r), v(r) linear in T between the last unformed record (i-1) and the first formed one (i);
  - cubic:  N(r), v(r) by the Lagrange cubic in T through the 4 records i-2, i-1, i, i+1;
crossing by brentq inside the bracket. Interpolation error = |T_cubic - T_linear|. Also printed (checks): the linear and cubic zero of
the json's own min_N_minus_absv series through the same records, and the r of the minimum at the crossing (the birth radius of the
interpolant, not a solve). Nothing is fitted; nothing is ruled. Read-only on all inputs.

Args: --members FILE (name class json [fields]), --out JSON, --engine-dir DIR (the engine_fields modules), --only a,b.
"""
import argparse
import hashlib
import importlib
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


def gridmin(f, rr):
    k = int(np.argmin(f))
    if 0 < k < len(rr) - 1:
        y0, y1, y2 = f[k - 1], f[k], f[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            d = 0.5 * (y0 - y2) / den
            h = rr[1] - rr[0]
            return float(y1 - 0.25 * (y0 - y2) * d), float(rr[k] + d * h)
    return float(f[k]), float(rr[k])


def root(f, a, b):
    fa, fb = f(a), f(b)
    if not (fa > 0 > fb):
        return None, fa, fb
    return brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=200), fa, fb


def one(name, klass, jpath, fpath, edir):
    out = {"member": name, "class": klass, "json": jpath, "fields": fpath}
    for p, k in ((jpath, "json"), (fpath, "fields")):
        if not os.path.exists(p):
            out["error"] = "%s missing" % k
            return out
        out[k + "_bytes"] = os.path.getsize(p)
        out[k + "_md5"] = md5(p)
    J = json.load(open(jpath))
    A = J.get("args", {}) or {}
    rs = float(A["rs"]); g = float(A["g"]); ell = A["ell"]
    rtol = A.get("field_rtol") or 1e-12; atol = A.get("field_atol") or 1e-16
    out["args"] = {k: A.get(k) for k in ("m", "g", "rs", "R0_over_rs", "nsites", "ell", "prep_ell", "tau", "kappa_sea", "mix", "T_over_rs",
                                          "collar_policy", "support_rel", "field_rtol", "field_atol")}
    out["script"] = J.get("script")
    recs = [r for r in J["records"] if r.get("E") is not None]  # the readout's load rule
    out["json_records"] = len(recs)
    Tj = np.array([float(r["T_over_rs"]) for r in recs])
    nz = np.array([len(r.get("trapping_zeros") or []) for r in recs])
    mj = np.array([np.nan if r.get("min_N_minus_absv") is None else float(r["min_N_minus_absv"]) for r in recs])
    zi = np.where(nz > 0)[0]
    if len(zi) == 0:
        out["error"] = "no record holds a trapping zero"
        return out
    i = int(zi[0])
    out["T_f"] = float(Tj[i]); out["T_prev"] = float(Tj[i - 1]); out["step"] = float(Tj[i] - Tj[i - 1])
    out["R_t_f"] = recs[i].get("R_t"); out["zeros_f"] = recs[i].get("trapping_zeros")

    z = np.load(fpath, allow_pickle=False)
    R = np.asarray(z["R"], dtype=float)
    Tf_all = np.asarray(z["T"], dtype=float) / rs
    out["fields_rows"] = int(len(Tf_all))
    use = [k for k in (i - 2, i - 1, i, i + 1) if 0 <= k < len(recs)]
    rows = {}
    for k in use:
        m_ = np.where(np.abs(Tf_all - Tj[k]) < 1e-7)[0]
        rows[k] = int(m_[-1]) if len(m_) else None
    out["fields_rows_used"] = {"%.6f" % Tj[k]: rows[k] for k in use}
    if rows.get(i) is None or rows.get(i - 1) is None:
        out["error"] = "the bracket's records are not in the fields file"
        return out
    EPS = z["eps_tot"]; JJ = z["J_tot"]
    sys.path.insert(0, edir)
    FLD12 = importlib.import_module("engine_fields_v12")
    dense = {}
    checks = {}
    for k in use:
        if rows[k] is None:
            continue
        mod = ((recs[k].get("solver") or {}).get("module") or "engine_fields_v16.solve_constraints_ivp").split(".")[0]
        F = importlib.import_module(mod)
        Ns, vs, Wn, info = F.solve_constraints_ivp(np.asarray(EPS[rows[k]], float), np.asarray(JJ[rows[k]], float), g, R, ell,
                                                    rtol=rtol, atol=atol, collar_policy=A.get("collar_policy") or "flag",
                                                    support_rel=A.get("support_rel") or 1e-14)
        if info.get("failed"):
            checks["%.6f" % Tj[k]] = {"failed": info["failed"]}
            continue
        tz = FLD12.trapping_radius(Ns, vs, R, info)
        zj = recs[k].get("trapping_zeros") or []
        checks["%.6f" % Tj[k]] = {"module": mod, "min_rebuilt": tz["min"], "min_json": float(mj[k]),
                                  "d_min": tz["min"] - float(mj[k]), "zeros_rebuilt": tz["zeros"], "zeros_json": zj,
                                  "max_d_zero": (max(abs(a - b) for a, b in zip(tz["zeros"], zj)) if len(zj) == len(tz["zeros"]) and zj else
                                                 (0.0 if not zj and not tz["zeros"] else "count differs")),
                                  "N_sites_vs_stored": float(np.max(np.abs(Ns - np.asarray(z["N"][rows[k]], float))))}
        dense[k] = (info["N_dense"], info["v_dense"], float(info["r_nodes"][-1]))
    out["rebuild_checks"] = checks
    if i not in dense or (i - 1) not in dense:
        out["error"] = "the bracket's records could not be rebuilt"
        return out
    rc = min(d[2] for d in dense.values())
    rr = np.arange(0.0, rc + 1e-9, ell / 64.0)
    NV = {k: (np.asarray(dense[k][0](rr), float), np.asarray(dense[k][1](rr), float)) for k in dense}
    out["grid"] = {"r_max": rc, "h": ell / 64.0, "points": int(len(rr))}
    t0, t1 = float(Tj[i - 1]), float(Tj[i])

    def mm(N, v):
        return gridmin(N - np.abs(v), rr)

    out["grid_min_at_records"] = {"%.6f" % Tj[k]: mm(*NV[k])[0] for k in sorted(NV)}
    N0, v0 = NV[i - 1]; N1, v1 = NV[i]

    def lin_fields(t):
        s = (t - t0) / (t1 - t0)
        return (1 - s) * N0 + s * N1, (1 - s) * v0 + s * v1

    tl, fa, fb = root(lambda t: mm(*lin_fields(t))[0], t0, t1)
    out["bracket_m"] = [fa, fb]
    out["T_birth_lin"] = tl
    if tl is not None:
        out["r_birth_lin"] = mm(*lin_fields(tl))[1]
    c4 = [k for k in (i - 2, i - 1, i, i + 1) if k in NV]
    out["cubic_records"] = [float(Tj[k]) for k in c4]
    out["T_birth_cubic"] = None
    if len(c4) == 4:
        ts = np.array([float(Tj[k]) for k in c4])
        Ns_ = np.array([NV[k][0] for k in c4]); Vs_ = np.array([NV[k][1] for k in c4])

        def cub_fields(t):
            w = lagrange_weights(ts, t)
            return w @ Ns_, w @ Vs_

        tc, _, _ = root(lambda t: mm(*cub_fields(t))[0], t0, t1)
        out["T_birth_cubic"] = tc
        if tc is not None:
            out["r_birth_cubic"] = mm(*cub_fields(tc))[1]
        ms = np.array([mj[k] for k in c4])
        out["T_jsonseries_cubic"] = root(lambda t: float(lagrange_weights(ts, t) @ ms), t0, t1)[0]
    else:
        out["cubic_note"] = "fewer than 4 of the records i-2..i+1 available"
    a0, a1 = mj[i - 1], mj[i]
    out["T_jsonseries_lin"] = float(t0 + (t1 - t0) * a0 / (a0 - a1)) if a0 > 0 > a1 else None
    if tl is not None and out["T_birth_cubic"] is not None:
        out["interp_error"] = abs(out["T_birth_cubic"] - tl)
        out["shift_cubic"] = out["T_birth_cubic"] - out["T_f"]
    if tl is not None:
        out["shift_lin"] = tl - out["T_f"]
    out["converged_window"] = {"%.6f" % Tj[k]: recs[k].get("converged") for k in use}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-dir", default="<internal>/engine")
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
            o = one(name, klass, jpath, fpath, a.engine_dir)
        except Exception as e:
            import traceback
            o = {"member": name, "class": klass, "json": jpath, "fields": fpath, "error": "%s: %s" % (type(e).__name__, e),
                 "trace": traceback.format_exc()}
        res.append(o)
        print(json.dumps({k: o.get(k) for k in ("member", "T_f", "step", "T_birth_lin", "T_birth_cubic", "interp_error",
                                                 "T_jsonseries_lin", "T_jsonseries_cubic", "error")}), flush=True)
        print("   checks:", json.dumps({t: {kk: c.get(kk) for kk in ("d_min", "max_d_zero", "failed")} for t, c in (o.get("rebuild_checks") or {}).items()}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
