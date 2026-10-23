#!/usr/bin/env python
"""static_rows_v1.py (lead reviewer A; STATIC_rows_v1 per solve/specs/STATIC_rows_v1_ruling.md; the builder runs it as written and returns
failures to the lead).

On a converged member of a static-star record (engine_star_v6 or v7: the site fields N, v and the shell totals eps_tot, J_tot per site,
R = arange(nsites) + 1): the occupied columns are rebuilt at the stored site fields (engine_rows_static_v2.orbitals), the sources from them
(engine_ops_v1.sources_static; checked against the stored shell totals), the Noether stresses per shell (engine_stress_v1.stress_parts and
shell_values: S_r, S_t), the dense fields re-solved from the shell totals by engine_fields_v14.solve_constraints_ivp at each ell of --ells
(the member's own ell and the controls), the averaged sources rho_ell, J_ell (the solve's own kernel sums: the Gaussian of standard
deviation ell on the harmonics l = 0 and 1) and S_r,ell, S_t,ell (engine_stress_v1.smooth_stress: l = 0 and 2). On the grid
r = ell/16 + k ell/32 up to r_collar - 2 ell:
  the continuity defects of the certificate's BI2 with the time derivatives zero (a stationary member: rho-dot = J-dot = 0):
    W_0 = r^-2 [r^2 (v rho + N J)]' + J N' + S_r v' + 2 S_t v/r,
    W_r = r^-2 [r^2 (v J + N S_r)]' - 2 N S_t/r + rho N' + J v',
  each as the sum of 7 terms (the largest term at a radius is the scale); D^t = W_0/N^2 and D^r = W_r/N + v W_0/N^2 beside them;
  the rows from the defects by BI5: R_t = [4 pi g int_0^r s^2 (N W_0 + v W_r) ds]/(N r^2 v) (the regular centre: c = 0; the integral by the
  antiderivative of the quintic spline of the parity-extended integrand, which is even and vanishes at s = 0; the trapezoid beside it),
  R_r = (4 pi g W_0/N - (A + B) R_t)/B with A = v'/N, B = v/(r N);
  the rows directly by engine_rows_v2.brackets with d_t k = 0: R_t = -rhs_tt/N, R_r = -rhs_rr/N (E = d_t k - rhs of the engine's rows;
  R = E/N is the certificate's normalization), normalized at each radius by the brackets' largest term as engine_rows_static_v2 prints them;
  at the member's own ell also the direct rows on the stored point-sampled site fields (a quintic parity spline through the sites) and the
  whole set at h = ell/64 (the derivative control).
Every array is written to one json per member, an md per member (band table, node table, the statement), and a summary md and json over the
members with the ruling's statement (V against E per row; the ruling's section 1(d)).
Derivatives by engine_rows_v2.deriv's Richardson-combined central differences (h = ell/32). Lattice units a = 1; g = G/a^2; v < 0 inward;
J_ell = <P_r> = -J_r of the goal's display (the engine's dictionary line).
Usage: python static_rows_v1.py --record <json> --out <dir> [--NB-list a,b,c] [--select all|trapped|untrapped] [--ells 4,8,16]
       [--include-failed] [--label text] [--code <internal>/code] [--threads 8] [--load-guard 90] [--NBmax 140.3158]
       python static_rows_v1.py --summarize --out <dir>"""
import os
import sys


def _early_threads():
    for i, x in enumerate(sys.argv):
        if x == "--threads" and i + 1 < len(sys.argv):
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(k, sys.argv[i + 1])


_early_threads()

import argparse
import glob
import json
import math
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import make_interp_spline

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:                                                     # older scipy
    from scipy.integrate import cumtrapz as cumulative_trapezoid

BOX_CODE = "<internal>/code"
INTERIOR = "interior (ell_m to R_surf)"
_E = {}


def engine(code=None):
    """the engine modules imported from the code directory on first use."""
    if not _E:
        for p in (code, BOX_CODE):
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        import engine_rows_static_v2 as S2
        import engine_fields_v14 as F14
        import engine_stress_v1 as STR
        import engine_rows_v2 as RW
        from engine_ops_v1 import sources_static
        _E.update(S2=S2, F14=F14, STR=STR, RW=RW, sources_static=sources_static)
    return SimpleNamespace(**_E)


# ----------------------------------------------------------------------------------------------------------------------------- numerics

def deriv_multi(F, r, h):
    """F(r) -> tuple of arrays; returns [(f, f', f''), ...] per field by engine_rows_v2.deriv's formula (central differences at h and
    h/2, Richardson-combined), one evaluation of F per shifted grid for all fields."""
    r = np.asarray(r, float)
    f0 = F(r); fp = F(r + h); fm = F(r - h); fp2 = F(r + 0.5 * h); fm2 = F(r - 0.5 * h)
    out = []
    for k in range(len(f0)):
        a0, ap, am, ap2, am2 = (np.asarray(x[k], float) for x in (f0, fp, fm, fp2, fm2))
        d1h = (ap - am) / (2.0 * h); d1h2 = (ap2 - am2) / h
        d2h = (ap - 2.0 * a0 + am) / (h * h); d2h2 = (ap2 - 2.0 * a0 + am2) / (0.25 * h * h)
        out.append((a0, (4.0 * d1h2 - d1h) / 3.0, (4.0 * d2h2 - d2h) / 3.0))
    return out


W0_TERMS = ["v rho'", "rho v'", "N J'", "2 J N'", "2 (v rho + N J)/r", "S_r v'", "2 S_t v/r"]
WR_TERMS = ["v J'", "2 J v'", "N S_r'", "N' S_r", "2 (v J + N S_r)/r", "-2 N S_t/r", "rho N'"]


def defects_static(fields, g, r, h):
    """BI2 with the time derivatives zero. fields: callables N, v, rho, J (arrays in, arrays out) and S (returns (S_r, S_t))."""
    r = np.asarray(r, float)
    (N, N1, N2), (v, v1, v2) = deriv_multi(lambda x: (fields["N"](x), fields["v"](x)), r, h)
    (rho, rho1, _), (J, J1, _) = deriv_multi(lambda x: (fields["rho"](x), fields["J"](x)), r, h)
    (Sr, Sr1, _), (St, St1, _) = deriv_multi(fields["S"], r, h)
    t0 = [v * rho1, rho * v1, N * J1, 2.0 * J * N1, 2.0 * (v * rho + N * J) / r, Sr * v1, 2.0 * St * v / r]
    tr = [v * J1, 2.0 * J * v1, N * Sr1, N1 * Sr, 2.0 * (v * J + N * Sr) / r, -2.0 * N * St / r, rho * N1]
    t0 = np.array(t0); tr = np.array(tr)
    W0 = t0.sum(axis=0); Wr = tr.sum(axis=0)
    return {"r": r, "N": N, "N1": N1, "N2": N2, "v": v, "v1": v1, "v2": v2, "rho": rho, "rho1": rho1, "J": J, "J1": J1,
            "Sr": Sr, "Sr1": Sr1, "St": St, "St1": St1, "W0": W0, "Wr": Wr, "W0_terms": t0, "Wr_terms": tr,
            "scale0": np.max(np.abs(t0), axis=0), "scaler": np.max(np.abs(tr), axis=0),
            "Dt": W0 / N ** 2, "Dr": Wr / N + v * W0 / N ** 2}


def quadrature_rows(d, g):
    """BI5 with c = 0 (the regular centre): R_t from the integral of 4 pi g s^2 (N W_0 + v W_r) from 0, R_r from W_0 and R_t."""
    r, N, v, v1, W0, Wr = d["r"], d["N"], d["v"], d["v1"], d["W0"], d["Wr"]
    f = 4.0 * math.pi * g * r ** 2 * (N * W0 + v * Wr)
    s = np.concatenate([-r[::-1], [0.0], r]); fs = np.concatenate([f[::-1], [0.0], f])        # even integrand, zero at the centre
    spl = make_interp_spline(s, fs, k=5)
    Fa = spl.antiderivative()
    I = np.asarray(Fa(r) - Fa(0.0), float)
    I_trap = 4.0 * math.pi * g * N[0] * W0[0] * r[0] ** 3 / 3.0 + cumulative_trapezoid(f, r, initial=0.0)
    Rt = I / (N * r ** 2 * v)
    A = v1 / N; B = v / (r * N)
    Rr = (4.0 * math.pi * g * W0 / N - (A + B) * Rt) / B
    return {"integrand": f, "I_cum": I, "I_trap": I_trap, "Rt_quad": Rt, "Rr_quad": Rr, "A": A, "B": B}


def direct_rows(fields, g, r, h):
    """the two rows by engine_rows_v2.brackets with d_t k = 0: R = -rhs/N; the brackets' largest term (over N) is the normalization."""
    E = engine()
    r = np.asarray(r, float)
    rho = np.asarray(fields["rho"](r), float); J = np.asarray(fields["J"](r), float)
    Sr, St = fields["S"](r)
    b = E.RW.brackets(fields["N"], fields["v"], rho, np.asarray(Sr, float), np.asarray(St, float), g, r, h)
    N = b["N"]
    return {"r": r, "Rt_dir": -b["rhs_tt"] / N, "Rr_dir": -b["rhs_rr"] / N, "terms_tt": b["terms_tt"] / N, "terms_rr": b["terms_rr"] / N,
            "rel_t": np.abs(b["rhs_tt"]) / np.maximum(b["terms_tt"], 1e-300), "rel_r": np.abs(b["rhs_rr"]) / np.maximum(b["terms_rr"], 1e-300),
            "k_r": b["k_r"], "k_t": b["k_t"]}


def parity_site_spline(R, y, par):
    """quintic spline through the site values extended through the centre by parity (the stored point-sampled fields as a function)."""
    R = np.asarray(R, float); y = np.asarray(y, float)
    xx = np.concatenate([-R[::-1], R]); yy = np.concatenate([par * y[::-1], y])
    return make_interp_spline(xx, yy, k=5)


# ---------------------------------------------------------------------------------------------------------------------------- the member

def rebuild_member(mem, half_offset, threads=None):
    """engine_rows_static_v2's rebuild: the occupied columns at the stored site fields, the sources and the Noether stresses."""
    E = engine()
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    m = float(mem["m"]); nsites = int(mem["nsites"])
    N = np.asarray(mem["N"], float); v = np.asarray(mem["v"], float)
    R, chans = E.S2.orbitals(N, v, m, nsites, float(mem["N_B"]), half_offset)
    eps = np.zeros(nsites); J = np.zeros(nsites); P = None; syk = np.zeros(nsites); ncol = 0
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = E.sources_static(h, Q, U, f, w)
        eps += e1; J += J1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = E.STR.ChanStress(ch, torch, torch.device("cpu"))
        p = E.STR.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
        syk += w * kappa * np.asarray(p["sy"], float)
        ncol += int(U.shape[1])
    P["sy_k"] = syk
    SV = E.STR.shell_values(P, R, m)
    es = np.asarray(mem["eps_tot"], float); Js = np.asarray(mem["J_tot"], float)
    V = 4.0 * math.pi * R ** 2
    checks = {"eps_rebuilt_vs_stored_rel": float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300),
              "J_rebuilt_vs_stored_rel": float(np.max(np.abs(J - Js))) / max(float(np.max(np.abs(Js))), 1e-300),
              "eps_stress_route_vs_sources_rel": float(np.max(np.abs(V * SV["rho"] - eps))) / max(float(np.max(np.abs(eps))), 1e-300),
              "channels_occupied": len(chans), "columns": ncol, "stress_identity_max_abs": float(SV["identity_max_abs"]),
              "virial_rel": float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / max(float(np.max(np.abs(SV["virial_rhs"]))), 1e-300)}
    return R, eps, J, SV, checks


def resolve_fields(eps, J, g, R, ell, SV):
    """engine_fields_v14's dense re-solve from the shell totals at ell; the stresses smoothed at ell. Returns (fields, info, N_sites, v_sites)."""
    E = engine()
    Ns, vs, Wn, info = E.F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
    if info.get("failed"):
        return None, info, None, None
    Rsh = np.asarray(R, float); Sr_sh = np.asarray(SV["S_r"], float); St_sh = np.asarray(SV["S_t"], float)

    def S(x):
        x = np.atleast_1d(np.asarray(x, float))
        Sr_l, St_l, _ = E.STR.smooth_stress(Sr_sh, St_sh, Rsh, ell, x)
        return np.asarray(Sr_l, float), np.asarray(St_l, float)
    fields = {"N": lambda x: np.asarray(info["N_dense"](np.atleast_1d(np.asarray(x, float))), float),
              "v": lambda x: np.asarray(info["v_dense"](np.atleast_1d(np.asarray(x, float))), float),
              "rho": lambda x: np.asarray(info["rho_ell"](np.atleast_1d(np.asarray(x, float))), float),
              "J": lambda x: np.asarray(info["J_ell"](np.atleast_1d(np.asarray(x, float))), float), "S": S}
    return fields, info, Ns, vs


def info_subset(info):
    keys = ["r_collar", "E_collar", "residual_W", "residual_invN", "residual_ladder", "residual_matching_region", "interp_floor_N", "interp_floor_v",
            "centre_series_check", "source_spline_error", "charge_conservation_quadrature", "collar_margin", "spectrum", "N_min", "ell", "module", "dictionary"]
    out = {k: info.get(k) for k in keys}
    gg = info.get("gauge", {}) or {}
    out["gauge"] = {k: gg.get(k) for k in ("label", "intervals", "min_U_over_U_ref", "delta_L_bound", "lapse_unresolved", "segments")}
    rg = info.get("regime", {}) or {}
    out["regime"] = {k: rg.get(k) for k in ("label", "max_ratio", "r_at_max", "intervals_above_1")}
    iv = info.get("ivp", {}) or {}
    out["ivp"] = {k: iv.get(k) for k in ("r0", "N_centre", "nfev", "segments")}
    return jsonable(out)


def bands_for(zeros, ell, ell_m, rc, Rs):
    out = [(INTERIOR, ell_m, Rs)]
    if len(zeros) >= 2:
        z1, z2 = float(min(zeros)), float(max(zeros))
        out += [("ell to the inner zero", ell, z1), ("between the zeros (N < abs v)", z1, z2), ("outer zero to R_surf", z2, Rs)]
    else:
        out += [("ell to 0.9 R_surf", ell, 0.9 * Rs), ("0.9 to 1.1 R_surf", 0.9 * Rs, 1.1 * Rs)]
    out += [("R_surf to the collar - 2 ell", Rs, rc - 2.0 * ell), ("1.1 R_surf to the collar - 2 ell", 1.1 * Rs, rc - 2.0 * ell)]
    return out


def band_stats(r, d, q, dr, lo, hi):
    m = (r >= lo - 1e-9) & (r <= hi + 1e-9)
    if not m.any():
        return None
    tt = np.maximum(dr["terms_tt"][m], 1e-300); tr = np.maximum(dr["terms_rr"][m], 1e-300)
    Rt = dr["Rt_dir"][m]; Rr = dr["Rr_dir"][m]
    out = {"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
           "max_rel_t": float(np.max(np.abs(Rt) / tt)), "max_rel_r": float(np.max(np.abs(Rr) / tr)),
           "rms_rel_t": float(np.sqrt(np.mean((Rt / tt) ** 2))), "rms_rel_r": float(np.sqrt(np.mean((Rr / tr) ** 2))),
           "max_term_t": float(np.max(tt)), "max_term_r": float(np.max(tr)),
           "band_rel_t": float(np.max(np.abs(Rt)) / float(np.max(tt))), "band_rel_r": float(np.max(np.abs(Rr)) / float(np.max(tr))),
           "max_abs_Rt": float(np.max(np.abs(Rt))), "max_abs_Rr": float(np.max(np.abs(Rr)))}
    if d is not None:
        s0 = np.maximum(d["scale0"][m], 1e-300); s1 = np.maximum(d["scaler"][m], 1e-300)
        out.update({"max_W0_rel": float(np.max(np.abs(d["W0"][m]) / s0)), "max_Wr_rel": float(np.max(np.abs(d["Wr"][m]) / s1)),
                    "band_W0_rel": float(np.max(np.abs(d["W0"][m])) / float(np.max(s0))), "band_Wr_rel": float(np.max(np.abs(d["Wr"][m])) / float(np.max(s1))),
                    "max_abs_W0": float(np.max(np.abs(d["W0"][m]))), "max_abs_Wr": float(np.max(np.abs(d["Wr"][m]))),
                    "max_scale0": float(np.max(s0)), "max_scaler": float(np.max(s1))})
    if q is not None:
        dt_ = np.abs(q["Rt_quad"][m] - Rt); dr_ = np.abs(q["Rr_quad"][m] - Rr)
        out.update({"quad_vs_dir_t_over_term": float(np.max(dt_ / tt)), "quad_vs_dir_r_over_term": float(np.max(dr_ / tr)),
                    "quad_vs_dir_t_rel": float(np.max(dt_)) / max(float(np.max(np.abs(Rt))), 1e-300),
                    "quad_vs_dir_r_rel": float(np.max(dr_)) / max(float(np.max(np.abs(Rr))), 1e-300),
                    "spline_vs_trapezoid_rel": float(np.max(np.abs(q["I_cum"][m] - q["I_trap"][m]))) / max(float(np.max(np.abs(q["I_cum"][m]))), 1e-300)})
    return out


def centre_diag(d, q, dr, g, npts=4):
    v1 = float(d["v1"][0])
    tgt = 4.0 * math.pi * g * d["W0"] / (3.0 * v1) if v1 != 0.0 else np.full_like(d["W0"], np.nan)
    rows = []
    for i in range(min(npts, len(d["r"]))):
        rows.append({"r": float(d["r"][i]), "Rt_quad": float(q["Rt_quad"][i]), "Rr_quad": float(q["Rr_quad"][i]), "Rt_dir": float(dr["Rt_dir"][i]),
                     "Rr_dir": float(dr["Rr_dir"][i]), "target_4pig_W0_over_3v1": float(tgt[i]),
                     "Wr_over_W0": float(d["Wr"][i] / d["W0"][i]) if d["W0"][i] != 0 else None})
    return {"v1_at_first_point": v1, "rows": rows, "note": "the regular-centre limit of BI3: R_r(0) = R_t(0) = 4 pi g W_0(0)/(3 v_1), W_r = O(r)"}


def pack(d, q, dr):
    out = {}
    if d is not None:
        for k in ("r", "N", "N1", "N2", "v", "v1", "v2", "rho", "rho1", "J", "J1", "Sr", "Sr1", "St", "St1", "W0", "Wr", "scale0", "scaler", "Dt", "Dr"):
            out[k] = d[k]
        out["W0_terms"] = {W0_TERMS[i]: d["W0_terms"][i] for i in range(7)}
        out["Wr_terms"] = {WR_TERMS[i]: d["Wr_terms"][i] for i in range(7)}
    if q is not None:
        for k in ("integrand", "I_cum", "I_trap", "Rt_quad", "Rr_quad", "A", "B"):
            out[k] = q[k]
    if dr is not None:
        if "r" not in out:
            out["r"] = dr["r"]
        for k in ("Rt_dir", "Rr_dir", "terms_tt", "terms_rr", "rel_t", "rel_r", "k_r", "k_t"):
            out[k] = dr[k]
    return jsonable(out)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def member_label(Jf, mem, NBmax):
    script = Jf.get("script", "") or ""
    args = Jf.get("args", {}) or {}
    if "engine_star_v7" in script:
        sf = args.get("site_fields", "averaged")
        fd = ("averaged site fields (engine_star_v7, --site-fields averaged; engine_fields_v14.site_fields_kernel_average: the declared definition)"
              if sf == "averaged" else "point-sampled site fields (engine_star_v7, --site-fields dense; engine_fields_v14)")
    else:
        fd = "point-sampled site fields (%s; engine_fields_v3: the control definition)" % (script or "engine_star_v5/v6")
    mn = float(mem["min_N_minus_absv"])
    if mn < 0:
        cls = "3 (beyond N = abs v: a stationary filling with a trapped interior; not a static star)"
    elif float(mem["N_B"]) > NBmax:
        cls = "2 (ell dependent, N > abs v)"
    else:
        cls = "1 (hydrodynamic)"
    return fd, cls


def statement(per, ell_m, Rs):
    base = per.get(ell_m, {})
    if "bands" not in base or base["bands"].get(INTERIOR) is None:
        return {"available": False, "reason": "no interior band at the member's ell", "both": False}
    out = {"available": True, "band": INTERIOR, "ell_member": ell_m, "R_surf": Rs, "rows": {}}
    for k in ("t", "r"):
        V = base["bands"][INTERIOR]["max_rel_" + k]
        vals = {("%g" % e): per[e]["bands"][INTERIOR]["max_rel_" + k] for e in per if "bands" in per[e] and per[e]["bands"].get(INTERIOR) is not None}
        spread = (max(vals.values()) - min(vals.values())) if len(vals) >= 2 else None
        Vst = base["stored_fields_rows"]["bands"][INTERIOR]["max_rel_" + k]
        Vh = base["h_control"]["bands"][INTERIOR]["max_rel_" + k]
        ff = abs(Vst - V); fh = abs(Vh - V)
        comps = [x for x in (spread, ff, fh) if x is not None]
        Eerr = max(comps)
        out["rows"][k] = {"V": V, "values_per_ell": vals, "spread_ell": spread, "V_stored_fields": Vst, "floor_fields": ff, "V_h_ell_over_64": Vh,
                          "floor_h": fh, "E": Eerr, "satisfied": bool(V <= Eerr), "V_over_E": (V / Eerr) if Eerr > 0 else None}
    out["both"] = bool(out["rows"]["t"]["satisfied"] and out["rows"]["r"]["satisfied"])
    return out


def fe(x, p=3):
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "nan"
    return ("%." + str(p) + "e") % xf


def band_row(ell_lab, lab, b):
    if b is None:
        return "| %s | %s | - | - | 0 | - | - | - | - | - | - | - | - |" % (ell_lab, lab)
    return "| %s | %s | %.2f | %.2f | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        ell_lab, lab, b["lo"], b["hi"], b["n"], fe(b["max_rel_t"]), fe(b["max_rel_r"]), fe(b["band_rel_t"]), fe(b["band_rel_r"]),
        fe(b.get("max_W0_rel")), fe(b.get("max_Wr_rel")), fe(b.get("quad_vs_dir_t_over_term")), fe(b.get("quad_vs_dir_r_over_term")))


def write_member_md(path, a, mem, fd, cls, checks, per, stmt, ell_m, Rs, seconds):
    rec = os.path.basename(a.record)
    L = ["## STATIC_rows_v1 (static_rows_v1.py; the ruling solve/specs/STATIC_rows_v1_ruling.md): %s, N_B %.3f, class %s; %s%s" % (
        rec, float(mem["N_B"]), cls, fd, ("; " + a.label) if a.label else ""), ""]
    C = (2.0 * float(mem["g"]) * float(mem["E"]) / Rs) if Rs else None
    L.append("- the member (the record's scalars): E %.6f, R_surf %s, R_999 %s, 2gE/R_surf %s, N_centre %.5f, min(N - abs v) %.5f at r %.3f, trapping zeros %s, "
             "ell %g, %d sites, %s channels (kappa_max occupied %s), E_top %s, E_F %s, interpolation floors N %s v %s, passes %s%s." % (
                 float(mem["E"]), ("%.2f" % Rs) if Rs else "-", mem.get("R_999"), fe(C, 4), float(mem["N_centre"]), float(mem["min_N_minus_absv"]),
                 float(mem.get("r_min_N_minus_absv") or 0.0), mem.get("trapping_zeros"), float(mem["ell"]), int(mem["nsites"]), mem.get("channels"),
                 mem.get("kappa_max_occupied"), fe(mem.get("E_top"), 6), fe(mem.get("E_F"), 6), fe(mem.get("interp_floor_N"), 2), fe(mem.get("interp_floor_v"), 2),
                 mem.get("passes"), "; FAILED self-consistency: %s" % mem.get("flags") if mem.get("failed") else ""))
    L.append("- the rebuild at the stored site fields: eps %s, J %s relative to the stored shell totals (the stress route's eps %s); %d channels occupied, %d columns; "
             "the momentum identity's largest difference %s; the virial residual %s relative (about 0.2 on the class-1 members, inputs 06 and 07: printed, not a criterion)." % (
                 fe(checks["eps_rebuilt_vs_stored_rel"], 2), fe(checks["J_rebuilt_vs_stored_rel"], 2), fe(checks["eps_stress_route_vs_sources_rel"], 2),
                 checks["channels_occupied"], checks["columns"], fe(checks["stress_identity_max_abs"], 2), fe(checks["virial_rel"], 2)))
    for ell in sorted(per):
        e = per[ell]
        if "failed" in e:
            L.append("- ell %g: the field re-solve failed: %s." % (ell, e["failed"]))
            continue
        inf = e["info"]
        L.append("- ell %g: r_collar %.0f, E_collar %.6f, the constraints' residuals W %s and 1/N %s, the gauge %s, the regime %s; zeros of N - abs v on the dense fields %s, "
                 "min(N - abs v) %.5f at r %.3f; grid %d points of spacing ell/32 from ell/16 to the collar - 2 ell." % (
                     ell, inf["r_collar"], inf["E_collar"], fe(inf["residual_W"], 2), fe(inf["residual_invN"], 2), inf["gauge"]["label"], inf["regime"]["label"],
                     ["%.3f" % z for z in e["zeros_dense"]], e["min_N_minus_absv_dense"], e["r_min_dense"], e["n_grid"]))
        if "field_floor" in e:
            ff = e["field_floor"]
            L.append("  the dense re-solve against the stored site fields inside the collar: max abs dN %s, max abs dv %s (the record's interpolation floors %s, %s); "
                     "E re-solve %.6f against the record's E %.6f (relative %s)." % (fe(ff["max_abs_N_dense_minus_stored"], 2), fe(ff["max_abs_v_dense_minus_stored"], 2),
                                                                                    fe(ff["record_interp_floor_N"], 2), fe(ff["record_interp_floor_v"], 2),
                                                                                    ff["E_resolve"], ff["E_record"], fe(ff["E_resolve"] / ff["E_record"] - 1.0, 2)))
    base = per.get(ell_m, {})
    if "grid" in base:
        Nmin = float(np.min(base["grid"]["N"])); vmax = float(np.max(base["grid"]["v"]))
        L.append("- stationary member: rho-dot = J-dot = 0 in BI2 (the time-derivative parts of D^t and D^r are zero); the identity's premises on the grid at ell %g: "
                 "N > 0 (min %.4e), v < 0 for r > 0 (max v %.3e), the regular centre (c = 0); the quadrature from the centre by the quintic spline of the parity-extended "
                 "integrand (the trapezoid beside it in the json: relative difference %s on the interior)." % (
                     ell_m, Nmin, vmax, fe((base["bands"][INTERIOR] or {}).get("spline_vs_trapezoid_rel"), 1)))
    L += ["", "| ell | band | r from | r to | n | max abs(R_t)/term | max abs(R_r)/term | band max abs(R_t)/max term | band max abs(R_r)/max term | max abs(W_0)/scale | max abs(W_r)/scale | quadrature vs direct R_t (/term) | quadrature vs direct R_r (/term) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ell in sorted(per):
        e = per[ell]
        if "bands" not in e:
            continue
        for lab, lo, hi in e["band_edges"]:
            L.append(band_row("%g" % ell, lab, e["bands"].get(lab)))
        if "stored_fields_rows" in e:
            for lab, lo, hi in e["band_edges"]:
                L.append(band_row("%g (stored point-sampled fields)" % ell, lab, e["stored_fields_rows"]["bands"].get(lab)))
        if "h_control" in e:
            for lab, lo, hi in e["band_edges"]:
                L.append(band_row("%g (h = ell/64)" % ell, lab, e["h_control"]["bands"].get(lab)))
    if "grid" in base:
        G = base["grid"]
        r = np.asarray(G["r"], float)
        L += ["", "| r | N | v | N - abs v | rho_ell | J_ell | S_r,ell | S_t,ell | W_0 | W_0/scale | W_r | W_r/scale | R_t direct | R_t quadrature | R_r direct | R_r quadrature | abs(R_t)/term | abs(R_r)/term |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        kmax = int(np.floor(r[-1] / ell_m + 1e-9))
        for k in range(1, kmax + 1):
            i = int(np.argmin(np.abs(r - k * ell_m)))
            if abs(r[i] - k * ell_m) > 1e-6:
                continue
            g_ = lambda key: float(G[key][i])
            L.append("| %.1f | %.5f | %.5f | %+.5f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r[i], g_("N"), g_("v"), g_("N") - abs(g_("v")), fe(g_("rho"), 4), fe(g_("J"), 4), fe(g_("Sr"), 4), fe(g_("St"), 4),
                fe(g_("W0")), fe(g_("W0") / max(g_("scale0"), 1e-300), 2), fe(g_("Wr")), fe(g_("Wr") / max(g_("scaler"), 1e-300), 2),
                fe(g_("Rt_dir")), fe(g_("Rt_quad")), fe(g_("Rr_dir")), fe(g_("Rr_quad")), fe(g_("rel_t"), 2), fe(g_("rel_r"), 2)))
        cd = base.get("centre")
        if cd:
            L += ["", "- the regular-centre diagnostic at ell %g (v_1 %.4e at the first grid point): " % (ell_m, cd["v1_at_first_point"]) + "; ".join(
                "r %.3f: R_t quad %s, R_r quad %s, R_t direct %s, R_r direct %s, 4 pi g W_0/(3 v_1) %s, W_r/W_0 %s" % (
                    x["r"], fe(x["Rt_quad"]), fe(x["Rr_quad"]), fe(x["Rt_dir"]), fe(x["Rr_dir"]), fe(x["target_4pig_W0_over_3v1"]), fe(x["Wr_over_W0"], 2)) for x in cd["rows"]) + "."]
    L.append("")
    if stmt.get("available"):
        for k, nm in (("t", "R_t"), ("r", "R_r")):
            s = stmt["rows"][k]
            L.append("- %s on [%g, R_surf %.2f] at ell %g (the dense re-solve, h = ell/32): V = %s; the controls: ell spread %s (V at ell %s), field floor %s (the stored point-sampled fields give %s), "
                     "derivative control %s (h = ell/64 gives %s); E = the largest of the three = %s: %s." % (
                         nm, ell_m, Rs, ell_m, fe(s["V"]), fe(s["spread_ell"]), ", ".join("%s: %s" % (e_, fe(v_)) for e_, v_ in sorted(s["values_per_ell"].items(), key=lambda t: float(t[0]))),
                         fe(s["floor_fields"]), fe(s["V_stored_fields"]), fe(s["floor_h"]), fe(s["V_h_ell_over_64"]), fe(s["E"]),
                         "satisfied within the averaging error (V <= E)" if s["satisfied"] else "not satisfied: V exceeds E by the factor %s" % fe(s["V_over_E"], 2)))
        ib = base["bands"][INTERIOR]
        L.append("- both rows within the averaging error: %s. The defects on the same band: max abs(W_0)/scale %s, max abs(W_r)/scale %s; the quadrature against the direct rows %s (R_t) and %s (R_r) of the largest term%s." % (
            "yes" if stmt["both"] else "no", fe(ib.get("max_W0_rel")), fe(ib.get("max_Wr_rel")), fe(ib.get("quad_vs_dir_t_over_term"), 1), fe(ib.get("quad_vs_dir_r_over_term"), 1),
            " (above 1e-2: the identity's check fails on this member, flagged)" if max(ib.get("quad_vs_dir_t_over_term", 0), ib.get("quad_vs_dir_r_over_term", 0)) > 1e-2 else ""))
    else:
        L.append("- the statement is unavailable: %s." % stmt.get("reason"))
    L += ["", "- every number above is printed by the script (%.0f s); the derivatives by engine_rows_v2.deriv (h = ell/32; the control at ell/64); the sources averaged with the engine's "
          "Gaussian kernel of standard deviation ell (engine_fields_v4.kernels; engine_stress_v1.kernels_l); the physics reading is the lead's." % seconds]
    open(path, "w").write("\n".join(L) + "\n")
    return L


def run_member(a, Jf, mem, half_offset, out_dir):
    E = engine(a.code)
    t0 = time.time()
    m = float(mem["m"]); g = float(mem["g"]); ell_m = float(mem["ell"]); nsites = int(mem["nsites"])
    NB = float(mem["N_B"])
    tag = os.path.splitext(os.path.basename(a.record))[0]
    name = "static_rows_v1_%s_NB%.3f" % (tag, NB)
    fd, cls = member_label(Jf, mem, a.NBmax)
    print("[%s] N_B %.3f: class %s; %s" % (tag, NB, cls.split(" ")[0], fd), flush=True)
    R, eps, J, SV, checks = rebuild_member(mem, half_offset, a.threads)
    print("  rebuild: eps %.2e J %.2e (relative), %d channels, %d columns, virial %.2e (%.0f s)" % (
        checks["eps_rebuilt_vs_stored_rel"], checks["J_rebuilt_vs_stored_rel"], checks["channels_occupied"], checks["columns"], checks["virial_rel"], time.time() - t0), flush=True)
    N_st = np.asarray(mem["N"], float); v_st = np.asarray(mem["v"], float)
    Rs = float(mem["R_surf"]) if mem.get("R_surf") else float(mem.get("R_999") or 0.0)
    ells = sorted(set([float(x) for x in a.ells.split(",")] + [ell_m]))
    per = {}
    shells = {"R": R, "eps_stored": mem["eps_tot"], "J_stored": mem["J_tot"], "eps_rebuilt": eps, "J_rebuilt": J, "S_r": SV["S_r"], "S_t": SV["S_t"],
              "S_t_continuum": SV.get("S_t_continuum"), "rho_shell": SV["rho"], "N_stored": N_st, "v_stored": v_st}
    for ell in ells:
        t1 = time.time()
        fields, info, Ns, vs = resolve_fields(eps, J, g, R, ell, SV)
        if fields is None:
            per[ell] = {"ell": ell, "failed": str(info.get("failed"))}
            print("  ell %g: the field re-solve failed: %s" % (ell, info.get("failed")), flush=True)
            continue
        rc = float(info["r_collar"]); h = ell / 32.0
        grid = np.arange(ell / 16.0, rc - 2.0 * ell + 1e-9, ell / 32.0)
        tz = E.F14.trapping_radius(Ns, vs, R, info)
        d = defects_static(fields, g, grid, h); q = quadrature_rows(d, g); dr = direct_rows(fields, g, grid, h)
        bands = bands_for(tz["zeros"], ell, ell_m, rc, Rs)
        entry = {"ell": ell, "h": h, "n_grid": int(len(grid)), "info": info_subset(info), "zeros_dense": [float(z) for z in tz["zeros"]],
                 "min_N_minus_absv_dense": float(tz["min"]), "r_min_dense": float(tz["r_min"]),
                 "band_edges": [[lab, float(lo), float(hi)] for lab, lo, hi in bands],
                 "bands": {lab: band_stats(grid, d, q, dr, lo, hi) for lab, lo, hi in bands}, "grid": pack(d, q, dr)}
        if ell == ell_m:
            inner = R < rc
            Nd = fields["N"](R); vd = fields["v"](R)
            shells["N_dense"] = Nd; shells["v_dense"] = vd
            entry["field_floor"] = {"max_abs_N_dense_minus_stored": float(np.max(np.abs(Nd[inner] - N_st[inner]))),
                                    "max_abs_v_dense_minus_stored": float(np.max(np.abs(vd[inner] - v_st[inner]))),
                                    "record_interp_floor_N": mem.get("interp_floor_N"), "record_interp_floor_v": mem.get("interp_floor_v"),
                                    "E_record": float(mem["E"]), "E_resolve": float(info["E_collar"])}
            Nsp = parity_site_spline(R, N_st, +1); vsp = parity_site_spline(R, v_st, -1)
            fst = dict(fields)
            fst["N"] = lambda x, s_=Nsp: np.asarray(s_(np.atleast_1d(np.asarray(x, float))), float)
            fst["v"] = lambda x, s_=vsp: np.asarray(s_(np.atleast_1d(np.asarray(x, float))), float)
            drs = direct_rows(fst, g, grid, h)
            entry["stored_fields_rows"] = {"note": "the direct rows on the stored point-sampled site fields through a quintic parity spline; the constraints hold on them only to the record's residuals, so no quadrature here",
                                           "bands": {lab: band_stats(grid, None, None, drs, lo, hi) for lab, lo, hi in bands}, "grid": pack(None, None, drs)}
            h2 = ell / 64.0
            d2 = defects_static(fields, g, grid, h2); q2 = quadrature_rows(d2, g); dr2 = direct_rows(fields, g, grid, h2)
            entry["h_control"] = {"h": h2, "bands": {lab: band_stats(grid, d2, q2, dr2, lo, hi) for lab, lo, hi in bands}, "grid": pack(d2, q2, dr2)}
            entry["centre"] = centre_diag(d, q, dr, g)
        per[ell] = entry
        ib = entry["bands"].get(INTERIOR) or {}
        print("  ell %g: collar %.0f, E_collar %.6f, zeros %s, interior max abs(R_t)/term %s, abs(R_r)/term %s, W_0 %s, W_r %s, quad vs direct %s, %s (%.0f s)" % (
            ell, rc, info["E_collar"], ["%.2f" % z for z in tz["zeros"]], fe(ib.get("max_rel_t")), fe(ib.get("max_rel_r")), fe(ib.get("max_W0_rel")), fe(ib.get("max_Wr_rel")),
            fe(ib.get("quad_vs_dir_t_over_term"), 1), fe(ib.get("quad_vs_dir_r_over_term"), 1), time.time() - t1), flush=True)
    stmt = statement(per, ell_m, Rs)
    seconds = time.time() - t0
    member_scalars = {k: mem.get(k) for k in ("N_B", "NB_target", "E", "ell", "m", "g", "nsites", "R_surf", "R_999", "compactness_surf", "N_centre", "N_min",
                                                 "min_N_minus_absv", "r_min_N_minus_absv", "trapping_zeros", "E_top", "E_next", "E_F", "E_F_half_gap", "interp_floor_N",
                                                 "interp_floor_v", "channels", "kappa_max_occupied", "passes", "failed", "flags", "residual_H_max", "residual_M_max",
                                                 "r_collar", "dictionary", "invariant_M_g15_ma2", "binding")}
    member_scalars.update({"class": cls, "field_definition": fd, "record": a.record, "record_script": Jf.get("script"), "label": a.label, "half_offset": bool(half_offset)})
    base = per.get(ell_m, {})
    summ = {"record": os.path.basename(a.record), "N_B": NB, "E": float(mem["E"]), "class": cls, "fields": fd, "label": a.label,
            "min_N_minus_absv_record": float(mem["min_N_minus_absv"]), "zeros_record": mem.get("trapping_zeros"), "zeros_dense": base.get("zeros_dense"),
            "R_surf": Rs, "ell_member": ell_m, "E_resolve": (base.get("field_floor") or {}).get("E_resolve"), "rebuild": checks, "statement": stmt,
            "interior": base.get("bands", {}).get(INTERIOR), "ells_solved": [e for e in per if "bands" in per[e]], "ells_failed": {("%g" % e): per[e]["failed"] for e in per if "failed" in per[e]},
            "failed_member": bool(mem.get("failed")), "seconds": seconds}
    out = {"script": "static_rows_v1.py", "ruling": "solve/specs/STATIC_rows_v1_ruling.md", "record": a.record, "member": member_scalars, "rebuild": checks,
           "shells": shells, "ells": {("%g" % e): per[e] for e in per}, "statement": stmt, "summary": summ, "seconds": seconds,
           "method": {"defects": "BI2 of the certificate with the time derivatives zero; W0_terms and Wr_terms named in the grid dict",
                      "quadrature": "BI5 with c = 0; I_cum by the antiderivative of the quintic spline of the parity-extended integrand; I_trap the trapezoid with the series centre piece",
                      "direct": "engine_rows_v2.brackets with d_t k = 0, R = -rhs/N; the normalization per radius by the brackets' largest term over N",
                      "kernel": "the engine's Gaussian of standard deviation ell on the harmonics (engine_fields_v4.kernels, engine_stress_v1.kernels_l)",
                      "units": "lattice units a = 1, g = G/a^2; densities per site volume; defects per site volume per lattice time; R in 1/a^2; v < 0 inward"}}
    jp = os.path.join(out_dir, name + ".json")
    with open(jp, "w") as fh:
        json.dump(jsonable(out), fh)
    write_member_md(os.path.join(out_dir, name + ".md"), a, mem, fd, cls, checks, per, stmt, ell_m, Rs, seconds)
    print("  written %s (%.0f s)" % (jp, seconds), flush=True)
    return summ


def summarize(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "static_rows_v1_*_NB*.json")))
    S = []
    for f in files:
        try:
            Jm = json.load(open(f))
        except Exception as ex:
            print("unreadable %s: %s" % (f, ex)); continue
        if "summary" in Jm:
            S.append(Jm["summary"])
    S.sort(key=lambda s: (s.get("record", ""), s["N_B"]))
    L = ["## STATIC_rows_v1 summary (static_rows_v1.py; the ruling solve/specs/STATIC_rows_v1_ruling.md)", "",
         "V: the largest abs(R)/largest term over [ell_m, R_surf] at the member's own ell on the dense re-solve (engine_fields_v14) with h = ell/32; "
         "E = the largest of (the ell spread of V over the fields re-solved and the stresses smoothed at ell 4, 8, 16 from the same shell totals; "
         "the field floor abs(V(stored point-sampled site fields) - V); the derivative control abs(V(h = ell/64) - V)); a row is satisfied within the averaging error iff V <= E. "
         "The defects W_0, W_r of BI2 with rho-dot = J-dot = 0; the quadrature of BI5 against the direct rows (the identity's check).", "",
         "| record | N_B | class | fields | label | min(N - abs v) record | zeros (dense, ell_m) | E record / E re-solve | V_t | E_t (spread, field, h) | R_t | V_r | E_r (spread, field, h) | R_r | both | max abs(W_0)/scale | max abs(W_r)/scale | quadrature vs direct (t, r) /term | ells solved |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        st = s.get("statement", {}) or {}
        ib = s.get("interior") or {}
        if st.get("available"):
            rt, rr = st["rows"]["t"], st["rows"]["r"]
            L.append("| %s | %.3f | %s | %s | %s | %+.5f | %s | %.6f / %s | %s | %s (%s, %s, %s) | %s | %s | %s (%s, %s, %s) | %s | %s | %s | %s | %s, %s | %s |" % (
                s["record"], s["N_B"], s["class"].split(" ")[0], ("averaged" if "averaged site" in s["fields"] else "point-sampled"), s.get("label") or "-",
                s["min_N_minus_absv_record"], ["%.2f" % z for z in (s.get("zeros_dense") or [])], s["E"], fe(s.get("E_resolve"), 6),
                fe(rt["V"]), fe(rt["E"]), fe(rt["spread_ell"]), fe(rt["floor_fields"]), fe(rt["floor_h"]), "yes" if rt["satisfied"] else "no",
                fe(rr["V"]), fe(rr["E"]), fe(rr["spread_ell"]), fe(rr["floor_fields"]), fe(rr["floor_h"]), "yes" if rr["satisfied"] else "no",
                "yes" if st["both"] else "no", fe(ib.get("max_W0_rel")), fe(ib.get("max_Wr_rel")), fe(ib.get("quad_vs_dir_t_over_term"), 1), fe(ib.get("quad_vs_dir_r_over_term"), 1),
                ", ".join("%g" % float(e) for e in s.get("ells_solved", []))))
        else:
            L.append("| %s | %.3f | %s | %s | %s | %+.5f | - | %.6f / - | - | - | - | - | - | - | no | - | - | - | %s; failed ells %s |" % (
                s["record"], s["N_B"], s["class"].split(" ")[0], ("averaged" if "averaged site" in s["fields"] else "point-sampled"), s.get("label") or "-",
                s["min_N_minus_absv_record"], s["E"], ", ".join("%g" % float(e) for e in s.get("ells_solved", [])), s.get("ells_failed")))
    trapped = [s for s in S if s["min_N_minus_absv_record"] < 0 and not s.get("failed_member") and not s.get("label")]
    sat = [s for s in trapped if (s.get("statement") or {}).get("both")]
    untr = [s for s in S if s["min_N_minus_absv_record"] >= 0 and not s.get("failed_member") and (s.get("statement") or {}).get("available")]
    L.append("")
    L.append("- trapped stationary fillings (the record's min(N - abs v) < 0) with both rows satisfied within their averaging error: %d of %d%s." % (
        len(sat), len(trapped), (": N_B " + ", ".join("%.3f" % s["N_B"] for s in sat)) if sat else " (none)"))
    if untr:
        vt = [s["statement"]["rows"]["t"]["V"] for s in untr]; vr = [s["statement"]["rows"]["r"]["V"] for s in untr]
        L.append("- the untrapped members at the same spacing and ell (the comparator; N_B %s): V_t %s to %s, V_r %s to %s." % (
            ", ".join("%.3f" % s["N_B"] for s in untr), fe(min(vt)), fe(max(vt)), fe(min(vr)), fe(max(vr))))
    flagged = [s for s in S if s.get("interior") and max(s["interior"].get("quad_vs_dir_t_over_term", 0), s["interior"].get("quad_vs_dir_r_over_term", 0)) > 1e-2]
    L.append("- the identity's check (the quadrature against the direct rows above 1e-2 of the largest term on the interior): %s." % (
        ("flagged at N_B " + ", ".join("%.3f" % s["N_B"] for s in flagged)) if flagged else "no member flagged"))
    L.append("- every number above is printed by the script from the per-member json files in %s; the physics reading is the lead's." % out_dir)
    open(os.path.join(out_dir, "static_rows_v1_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump({"script": "static_rows_v1.py", "members": S}, open(os.path.join(out_dir, "static_rows_v1_summary.json"), "w"), indent=1, default=float)
    print("\n".join(L))


def select_members(Jf, a):
    members = [q for q in Jf["members"] if q.get("N") and q.get("eps_tot") and q.get("N_B") is not None]
    if not a.include_failed:
        members = [q for q in members if not q.get("failed")]
    if a.NB_list:
        want = [float(x) for x in a.NB_list.split(",")]
        members = [q for q in members if any(abs(float(q["N_B"]) - w) < 2e-3 * max(1.0, w) for w in want)]
    if a.select == "trapped":
        members = [q for q in members if float(q["min_N_minus_absv"]) < 0]
    elif a.select == "untrapped":
        members = [q for q in members if float(q["min_N_minus_absv"]) >= 0]
    seen = set(); out = []
    for q in sorted(members, key=lambda q: float(q["N_B"])):
        k = round(float(q["N_B"]), 3)
        if k in seen:
            continue
        seen.add(k); out.append(q)
    return out


def wait_load(cap):
    if not cap or cap <= 0:
        return
    while True:
        l1 = os.getloadavg()[0]
        if l1 < cap:
            return
        print("the 1-minute load %.1f is above %.0f: waiting 60 s" % (l1, cap), flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--NB-list", default=None)
    ap.add_argument("--select", default="all", choices=["all", "trapped", "untrapped"])
    ap.add_argument("--ells", default="4,8,16")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--code", default=BOX_CODE)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--load-guard", type=float, default=90.0)
    ap.add_argument("--NBmax", type=float, default=140.3158, help="the dense zone table's N_B,max at (0.2, 1/4) (STAR_family_handoff_v3_final.md), for the class label only")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summarize:
        summarize(a.out)
        return
    if not a.record:
        ap.error("--record is required unless --summarize")
    wait_load(a.load_guard)
    engine(a.code)
    Jf = json.load(open(a.record))
    half_offset = bool((Jf.get("args") or {}).get("half_offset", 0))
    members = select_members(Jf, a)
    print("%s: %d members selected (%s): N_B %s" % (os.path.basename(a.record), len(members), a.select, ", ".join("%.3f" % float(q["N_B"]) for q in members)), flush=True)
    log = open(os.path.join(a.out, "static_rows_v1_run.log"), "a")
    for q in members:
        wait_load(a.load_guard)
        try:
            s = run_member(a, Jf, q, half_offset, a.out)
            print(json.dumps({"record": s["record"], "N_B": s["N_B"], "both": (s.get("statement") or {}).get("both"), "seconds": s["seconds"]}), file=log, flush=True)
        except Exception:
            tb = traceback.format_exc()
            print("member N_B %s FAILED:\n%s" % (q.get("N_B"), tb), flush=True)
            print(json.dumps({"record": os.path.basename(a.record), "N_B": q.get("N_B"), "error": tb}), file=log, flush=True)
            tag = os.path.splitext(os.path.basename(a.record))[0]
            json.dump({"script": "static_rows_v1.py", "record": a.record, "member": {"N_B": q.get("N_B")}, "error": tb,
                       "summary": {"record": os.path.basename(a.record), "N_B": float(q["N_B"]), "E": float(q.get("E") or 0.0), "class": "-", "fields": "-", "label": a.label,
                                   "min_N_minus_absv_record": float(q.get("min_N_minus_absv") or 0.0), "statement": {"available": False, "reason": "exception", "both": False},
                                   "failed_member": True, "ells_solved": [], "ells_failed": {"all": "exception"}}},
                      open(os.path.join(a.out, "static_rows_v1_%s_NB%.3f.json" % (tag, float(q["N_B"]))), "w"))
    summarize(a.out)


if __name__ == "__main__":
    main()
