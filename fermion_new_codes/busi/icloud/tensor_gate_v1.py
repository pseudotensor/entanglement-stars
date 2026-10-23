#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_gate_v1.py: sigma ||K||_op on the stored dense fields of the r_s/a 64, tau 8, sigma 4a history.

An independent check for the busi lane in the icloud lane's role (WEB40G_icloud_tasks.md item 6; sources
P17 claim 1 and P33 claim 7, read through REVIEWA_REVIEW_web40g.md summary 7). Derived here from the declared
object, not from the solve lane's code.

The object (requirements sec. 2 H2; item 5 of the paper at 22bae778e8). For the radial flow v_i = v(r) n_i
the declared K_ij = (d_i v_j + d_j v_i)/(2N) is
    K_ij = [ v'(r) n_i n_j + (v(r)/r) (delta_ij - n_i n_j) ] / N,
with principal values v'/N (once, eigenvector n: radial) and v/(rN) (twice: tangential). The operator norm
of the symmetric tensor is the largest magnitude of its principal values, so
    sigma ||K||_op = sigma max(|v'/N|, |v/(rN)|).
At a root of N - |v| = 0 with v < 0, v/(rN) = -1/R exactly, no derivative entering, so sigma ||K||_op >=
sigma/R there and the criterion sigma ||K||_op <= 0.1 fails wherever R < 10 sigma.

Usage:
  python3 -I tensor_gate_v1.py FIELDS.npz [--records-json RECORDS.json] [--out PREFIX]
          [--clocks "12.125,30,40:46:0.125"] [--all] [--sigma S] [--rs RS] [--gate 0.1] [--force] [--hash]
Outputs: PREFIX.json (the per-record summary; the resumable cache), PREFIX.md (the tables), PREFIX_profiles.npz
(the profiles along r for the listed clocks), PREFIX_all.md (with --all: 1 line per record of the history).
Numpy and the standard library only; python3 -I; seconds. No plot. No entanglement or mutual-information
quantity is computed (D11 revised, D15).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

VERSION = "tensor_gate_v1"
DEFAULT_CLOCKS = "12.125,30,40:46:0.125"
RULE_NAMES = ("centred", "forward", "backward", "stride_sigma")

# Reference values this readout is read against. Inputs, not computed here.
READ_AGAINST = {
    "12.125": {"value": 0.104, "R": 38.56,
               "source": "REVIEWA_REVIEW_web40g.md summary 7 [arith], 4/38.56 at the birth root; NOTES P33 claim 7 "
                         "verification: R_t 38.556, sigma/R_t = 0.1037 on the birth record"},
    "30.000": {"value": 0.107, "R": 37.51,
               "source": "REVIEWA_REVIEW_web40g.md summary 7 [arith], 4/37.51 at the second surface's outer root; "
                         "NOTES P17 claim 1: 4/37.51 = 0.1066"},
}
LATE_ROOT = {"T_lo": 40.0, "T_hi": 46.0, "R_lo": 36.3, "R_hi": 37.1, "ratio_lo": 0.108, "ratio_hi": 0.110,
             "source": "NOTES_P17_P33.md, claim 1 verification [arith]: the late root 36.31 to 37.13a at T/r_s "
                       "40 to 46 (HORIZON v2 part E) gives sigma/R 0.108 to 0.110"}

OBJECT_STATEMENT = (
    "K_ij = (d_i v_j + d_j v_i)/(2N) with v_i = v(r) n_i (requirements sec. 2 H2) is "
    "[v' n_i n_j + (v/r)(delta_ij - n_i n_j)]/N: principal values v'/N (radial, once) and v/(rN) (tangential, "
    "twice); sigma ||K||_op = sigma max(|v'/N|, |v/(rN)|). At a root of N - |v| = 0 with v < 0 the tangential "
    "value is -1/R exactly, so sigma ||K||_op >= sigma/R there (no derivative enters that bound) and the "
    "criterion sigma ||K||_op <= 0.1 fails wherever R < 10 sigma (a sufficient condition; for R >= 10 sigma the "
    "radial value can still exceed the threshold).")

FIELDS_STATEMENT = (
    "The paper's gate reads the dense solved fields N, v, and so does this script's headline. Ground: item 5 "
    "(22bae778e8) defines the trapping surface as 'the outer zero R_t of N-|v| on the dense solved fields (the "
    "geometric root ...; the matter's root on the back-averaged fields is distinct)' and its tensor clause holds "
    "'at a dense root of N=|v|'; the records' trapping_zeros are those dense roots (NOTES P17 claim 1 "
    "verification: 'placement uses the dense root'); the 1-term test's nodes sigma apart are the node "
    "representation of the same solved geometry (FIELDS_FORMAT: r_nodes, N_nodes, v_nodes; requirements F2: the "
    "coarse radial grid of spacing ell interpolated to the sites); requirements sec. 5 I1 makes the gate a "
    "statement about the geometry's gradient length. The applied fields N_applied, v_applied, the same geometry "
    "averaged back with the Gaussian of sigma onto the sites and the one the matter Hamiltonian sees (requirements "
    "sec. 2, the hydrodynamic geometry), print beside under 'applied' with their own roots. This is a reading of "
    "the inputs, not of the solve lane's code.")

DERIVATIVE_STATEMENT = (
    "v'(r_j) = (v_{j+1} - v_{j-1})/(R_{j+1} - R_{j-1}) at interior sites (the centred 2-point rule at the lattice "
    "spacing a), (v_1 - v_0)/(R_1 - R_0) at the first site and (v_{n-1} - v_{n-2})/(R_{n-1} - R_{n-2}) at the last. "
    "Sensitivity printed: 'forward' (v_{j+1} - v_j)/(R_{j+1} - R_j); 'backward' (v_j - v_{j-1})/(R_j - R_{j-1}); "
    "'stride_sigma', the centred rule on nodes sigma apart, (v_{j+s} - v_{j-s})/(R_{j+s} - R_{j-s}) with "
    "s = round(sigma/a) (one-sided over the same stride within s sites of the ends), the node spacing of the "
    "paper's 1-term test. The tangential value v/(rN) uses no derivative; where R = 0 (a regular centre) v/r is "
    "replaced by v'. Where N <= 0 the tensor is undefined: the site is excluded from every maximum and counted.")


# ----------------------------------------------------------------------------------------------------------
# small helpers

def fl(x):
    """float or None (nan and inf become None so the json stays strict)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def fmt(x, d=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return ("%%.%df" % d) % float(x)
    except (TypeError, ValueError):
        return str(x)


def safe_max(x):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    return fl(x[m].max()) if m.any() else None


def sha256_of(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def parse_clocks(spec):
    """'12.125,30,40:46:0.125' -> sorted unique list of T/r_s values (ranges inclusive, default step 0.125)."""
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            parts = tok.split(":")
            lo, hi = float(parts[0]), float(parts[1])
            step = float(parts[2]) if len(parts) > 2 else 0.125
            k = 0
            while lo + k * step <= hi + 1e-9:
                vals.append(round(lo + k * step, 6))
                k += 1
        else:
            vals.append(round(float(tok), 6))
    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def clocks_in_rs(T, rs):
    """The stored record clocks as T/r_s, with the unit detection stated."""
    T = np.asarray(T, dtype=float)
    dT = float(np.median(np.diff(T))) if T.size > 1 else 0.0
    if abs(dT - 0.125 * rs) <= 1e-6 * max(1.0, rs):
        return T / rs, "stored T in lattice units (median step %.6g = 0.125 r_s); T/r_s = T/%.6g" % (dT, rs)
    if abs(dT - 0.125) <= 1e-9:
        return T, "stored T already in units of r_s (median step 0.125)"
    return T / rs, "stored T step %.6g not recognised; lattice units assumed, T/r_s = T/%.6g" % (dT, rs)


# ----------------------------------------------------------------------------------------------------------
# the computation

def fd_derivative(y, R, rule="centred", stride=1):
    """Finite-difference derivative of y on the increasing grid R (see DERIVATIVE_STATEMENT)."""
    y = np.asarray(y, dtype=float)
    R = np.asarray(R, dtype=float)
    n = y.size
    s = max(1, int(stride))
    d = np.full(n, np.nan)
    if n < 2:
        return d
    if rule == "centred":
        if n > 2 * s:
            d[s:n - s] = (y[2 * s:] - y[:n - 2 * s]) / (R[2 * s:] - R[:n - 2 * s])
        for j in range(min(s, n)):
            k = min(j + s, n - 1)
            if k != j:
                d[j] = (y[k] - y[j]) / (R[k] - R[j])
        for j in range(max(n - s, 0), n):
            k = max(j - s, 0)
            if k != j:
                d[j] = (y[j] - y[k]) / (R[j] - R[k])
    elif rule == "forward":
        d[:-1] = (y[1:] - y[:-1]) / (R[1:] - R[:-1])
        d[-1] = d[-2]
    elif rule == "backward":
        d[1:] = (y[1:] - y[:-1]) / (R[1:] - R[:-1])
        d[0] = d[1]
    else:
        raise ValueError("unknown rule %r" % (rule,))
    return d


def principal_values(N, v, R, dv):
    """lam_r = v'/N (radial), lam_t = v/(rN) (tangential); ok = N > 0 (the tensor is undefined elsewhere)."""
    N = np.asarray(N, dtype=float)
    v = np.asarray(v, dtype=float)
    R = np.asarray(R, dtype=float)
    dv = np.asarray(dv, dtype=float)
    ok = N > 0
    lam_r = np.full(N.shape, np.nan)
    lam_t = np.full(N.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_r[ok] = dv[ok] / N[ok]
        pos = ok & (R > 0)
        lam_t[pos] = v[pos] / (R[pos] * N[pos])
    centre = ok & ~(R > 0)      # a regular centre: v/r -> v'
    lam_t[centre] = lam_r[centre]
    return lam_r, lam_t, ok


def gate_profile(lam_r, lam_t, sigma):
    """sigma ||K||_op = sigma max(|lam_r|, |lam_t|); nan where both are undefined."""
    return sigma * np.fmax(np.abs(lam_r), np.abs(lam_t))


def interp(r, R, y):
    """Linear interpolation of y at r on the increasing grid R."""
    return float(np.interp(float(r), np.asarray(R, dtype=float), np.asarray(y, dtype=float)))


def interpolated_roots(F, R):
    """Zeros of F by linear interpolation between consecutive sites, sorted. Direction +1: F rises through 0
    outward (N - |v| from negative to positive: the trapped region's exterior boundary, an 'outer' root);
    -1: F falls through 0 (an 'inner' root)."""
    F = np.asarray(F, dtype=float)
    R = np.asarray(R, dtype=float)
    out = []
    if F.size < 2:
        return out
    f0, f1 = F[:-1], F[1:]
    fin = np.isfinite(f0) & np.isfinite(f1)
    for j in np.flatnonzero(fin & (f0 * f1 < 0)):
        r = R[j] - f0[j] * (R[j + 1] - R[j]) / (f1[j] - f0[j])
        out.append((float(r), 1 if f1[j] > f0[j] else -1))
    for j in np.flatnonzero(fin & (f0 == 0.0) & (f1 != 0.0)):
        out.append((float(R[j]), 1 if f1[j] > 0 else -1))
    out.sort(key=lambda t: t[0])
    return out


def region_max(gate, lam_r, lam_t, N, R, mask, thr):
    """Maximum of the gate over the masked sites, with its radius, the dominating principal value, the count
    of failing sites and of undefined sites (N <= 0) in the region."""
    mask = np.asarray(mask, dtype=bool)
    fin = np.isfinite(gate)
    m = mask & fin
    out = {"n_sites": int(mask.sum()), "n_undefined": int((mask & ~fin).sum())}
    if not m.any():
        out.update({"value": None, "r": None, "n_failing": 0, "part": None, "N_at_max": None})
        return out
    idx = np.flatnonzero(m)
    j = int(idx[int(np.argmax(gate[m]))])
    out.update({"value": fl(gate[j]), "r": fl(R[j]), "n_failing": int((gate[m] > thr).sum()),
                "part": "tangential" if abs(lam_t[j]) >= abs(lam_r[j]) else "radial", "N_at_max": fl(N[j])})
    return out


def analyse_fields(R, N, v, sigma, thr, stride_sigma, rs, regions_extra=None, label="dense"):
    """The principal values, the gate, the roots and the maxima for 1 pair of fields (N, v) on R."""
    R = np.asarray(R, dtype=float)
    N = np.asarray(N, dtype=float)
    v = np.asarray(v, dtype=float)
    dv = {"centred": fd_derivative(v, R, "centred", 1),
          "forward": fd_derivative(v, R, "forward", 1),
          "backward": fd_derivative(v, R, "backward", 1),
          "stride_sigma": fd_derivative(v, R, "centred", stride_sigma)}
    lam_r, gate = {}, {}
    _, lam_t, ok = principal_values(N, v, R, dv["centred"])
    for k in RULE_NAMES:
        lam_r[k] = principal_values(N, v, R, dv[k])[0]
        gate[k] = gate_profile(lam_r[k], lam_t, sigma)
    g0, lr0 = gate["centred"], lam_r["centred"]

    F = N - np.abs(v)
    roots = interpolated_roots(F, R)
    roots_sum = interpolated_roots(N + v, R)
    outer = [r for r, d in roots if d > 0]
    inner = [r for r, d in roots if d < 0]
    R_out = max(outer) if outer else None
    R_in = min(inner) if inner else None

    root_rows = []
    for r, d in roots:
        row = {"r": fl(r), "kind": "outer" if d > 0 else "inner",
               "sigma_over_R_arith": fl(sigma / r),
               "tangential_interp": fl(interp(r, R, lam_t)),
               "tangential_exact_at_root_arith": fl(-1.0 / r),
               "radial_interp": {k: fl(interp(r, R, lam_r[k])) for k in RULE_NAMES},
               "gate_interp": {k: fl(interp(r, R, gate[k])) for k in RULE_NAMES},
               "N_interp": fl(interp(r, R, N)), "v_interp": fl(interp(r, R, v))}
        row["tangential_interp_residual"] = (fl(row["tangential_interp"] + 1.0 / r)
                                             if row["tangential_interp"] is not None else None)
        gi = row["gate_interp"]["centred"]
        row["gate_centred_minus_sigma_over_R"] = fl(gi - sigma / r) if gi is not None else None
        row["gate_centred_ge_sigma_over_R"] = (gi is not None and gi >= sigma / r * (1.0 - 1e-6))
        row["fails_gate"] = bool(gi is not None and gi > thr)
        root_rows.append(row)

    pos = R > 0
    regions = {}
    if R_out is not None:
        regions["r_le_R_out"] = region_max(g0, lr0, lam_t, N, R, pos & (R <= R_out), thr)
    for name, rad in (regions_extra or {}).items():
        if rad is not None:
            regions[name] = region_max(g0, lr0, lam_t, N, R, pos & (R <= rad), thr)
    regions["r_le_rs_nominal"] = region_max(g0, lr0, lam_t, N, R, pos & (R <= rs), thr)
    regions["all"] = region_max(g0, lr0, lam_t, N, R, pos, thr)

    fail = np.isfinite(g0) & (g0 > thr)
    r_fail_max = fl(R[fail].max()) if fail.any() else None
    ext, r_pass_min_outside = {}, None
    if R_out is not None:
        m_ext = (R > R_out) & (R <= R_out + sigma) & np.isfinite(g0)
        if m_ext.any():
            ext = {"min": fl(g0[m_ext].min()), "max": fl(g0[m_ext].max()), "n_sites": int(m_ext.sum()),
                   "n_failing": int((g0[m_ext] > thr).sum())}
        passing_out = np.isfinite(g0) & (R > R_out) & (g0 <= thr)
        r_pass_min_outside = fl(R[passing_out].min()) if passing_out.any() else None

    jm = int(np.argmin(F))
    if R_out is not None:
        core_mask, core_name = pos & (R <= R_out), "0 < r <= R_out (the outer dense root)"
    elif regions_extra and regions_extra.get("r_le_R_c_json") is not None:
        core_mask, core_name = pos & (R <= regions_extra["r_le_R_c_json"]), "0 < r <= R_c (json; no outer root)"
    else:
        core_mask, core_name = pos & (R <= rs), "0 < r <= r_s nominal (no outer root, no R_c)"

    sens = {"centred": {"core_max_gate": safe_max(g0[core_mask])}}
    for k in ("forward", "backward", "stride_sigma"):
        diff = sigma * np.abs(lam_r[k] - lr0)
        sens[k] = {"max_core_sigma_abs_dlam_r": safe_max(diff[core_mask]),
                   "core_max_gate": safe_max(gate[k][core_mask])}
    scalar = {"sigma_abs_dv_centred_core_max": safe_max(sigma * np.abs(dv["centred"][core_mask])),
              "sigma_abs_dv_stride_sigma_core_max": safe_max(sigma * np.abs(dv["stride_sigma"][core_mask]))}

    summary = {
        "label": label,
        "roots": root_rows,
        "R_out": fl(R_out), "R_in": fl(R_in),
        "roots_N_plus_v": [fl(r) for r, d in roots_sum],
        "roots_N_plus_v_equal_roots_N_minus_absv": (len(roots_sum) == len(roots) and all(
            abs(a[0] - b[0]) < 1e-9 for a, b in zip(roots_sum, roots))),
        "regions": regions,
        "r_fail_max": r_fail_max,
        "fail_contained_in_R_out": (None if (R_out is None or r_fail_max is None) else bool(r_fail_max <= R_out)),
        "exterior_side_within_sigma": ext,
        "r_first_passing_site_outside_R_out": r_pass_min_outside,
        "min_N_minus_absv": {"value": fl(F[jm]), "r": fl(R[jm])},
        "core_region_used": core_name,
        "min_N_core": fl(N[core_mask].min()) if core_mask.any() else None,
        "n_sites_N_nonpositive": int((~ok).sum()),
        "scalar_sigma_abs_dv": scalar,
        "sensitivity": sens,
    }
    profiles = {"lam_r_centred": lr0, "lam_t": lam_t, "gate_centred": g0, "gate_stride_sigma": gate["stride_sigma"],
                "dv_centred": dv["centred"], "N": N, "v": v}
    return {"summary": summary, "profiles": profiles}


def analyse_record(idx, tors, T_val, R, N, v, Na, va, cfg, jrec, keep_profile):
    sigma, thr, rs, s = cfg["sigma"], cfg["gate_threshold"], cfg["rs"], cfg["stride_sigma"]
    R_c = fl(jrec.get("R_c")) if jrec else None
    dense = analyse_fields(R, N, v, sigma, thr, s, rs,
                           regions_extra=({"r_le_R_c_json": R_c} if R_c is not None else None), label="dense")
    ds = dense["summary"]
    extra_a = {}
    if ds["R_out"] is not None:
        extra_a["r_le_R_out_dense"] = ds["R_out"]
    if R_c is not None:
        extra_a["r_le_R_c_json"] = R_c
    applied = analyse_fields(R, Na, va, sigma, thr, s, rs, regions_extra=(extra_a or None), label="applied")
    aps = applied["summary"]
    gd, ga = dense["profiles"]["gate_centred"], applied["profiles"]["gate_centred"]
    cross = {
        "applied_gate_centred_at_dense_roots": [{"r": rw["r"], "kind": rw["kind"], "value": fl(interp(rw["r"], R, ga))}
                                                for rw in ds["roots"]],
        "dense_gate_centred_at_applied_roots": [{"r": rw["r"], "kind": rw["kind"], "value": fl(interp(rw["r"], R, gd))}
                                                for rw in aps["roots"]],
    }
    cmp = {}
    if jrec:
        tz = [fl(x) for x in (jrec.get("trapping_zeros") or [])]
        fr = [rw["r"] for rw in ds["roots"]]
        cmp["json_trapping_zeros"] = tz
        cmp["nearest_dense_root_minus_json_zero"] = [(min(fr, key=lambda x: abs(x - zz)) - zz) if fr else None
                                                     for zz in tz]
        cmp["R_t_json"] = fl(jrec.get("R_t"))
        cmp["R_out_dense_minus_R_t_json"] = (fl(ds["R_out"] - cmp["R_t_json"])
                                             if (ds["R_out"] is not None and cmp["R_t_json"] is not None) else None)
        cmp["sigma_over_R_t_json_arith"] = fl(sigma / cmp["R_t_json"]) if cmp["R_t_json"] else None
        cmp["min_N_minus_absv_json"] = fl(jrec.get("min_N_minus_absv"))
        cmp["r_min_N_minus_absv_json"] = fl(jrec.get("r_min_N_minus_absv"))
        mf = ds["min_N_minus_absv"]
        cmp["min_F_dense_minus_json"] = (fl(mf["value"] - cmp["min_N_minus_absv_json"])
                                         if (mf["value"] is not None and cmp["min_N_minus_absv_json"] is not None) else None)
        cmp["r_min_F_dense_minus_json"] = (fl(mf["r"] - cmp["r_min_N_minus_absv_json"])
                                           if (mf["r"] is not None and cmp["r_min_N_minus_absv_json"] is not None) else None)
        cmp["R_c_json"] = R_c
        cmp["E_json"] = fl(jrec.get("E"))
    rec = {"T_over_rs": fl(tors), "record_index": int(idx), "T_stored": fl(T_val),
           "dense": ds, "applied": aps, "cross": cross, "comparison_with_json": cmp}
    if keep_profile:
        rec["profile"] = {"R": [fl(x) for x in R]}
        for name in ("N", "v", "lam_r_centred", "lam_t", "gate_centred", "gate_stride_sigma"):
            rec["profile"]["dense_" + name] = [fl(x) for x in dense["profiles"][name]]
        for name in ("N", "v", "lam_r_centred", "lam_t", "gate_centred"):
            rec["profile"]["applied_" + name] = [fl(x) for x in applied["profiles"][name]]
    prof = {}
    for name in ("lam_r_centred", "lam_t", "gate_centred", "gate_stride_sigma", "dv_centred"):
        prof["dense_" + name] = dense["profiles"][name]
    for name in ("lam_r_centred", "lam_t", "gate_centred"):
        prof["applied_" + name] = applied["profiles"][name]
    return rec, prof


# ----------------------------------------------------------------------------------------------------------
# output

def md_table(headers, rows):
    L = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(L)


def _outer_inner(ds):
    outer = [rw for rw in ds["roots"] if rw["kind"] == "outer"]
    inner = [rw for rw in ds["roots"] if rw["kind"] == "inner"]
    return (outer[-1] if outer else None), (inner[0] if inner else None)


def roots_text(rows):
    return "; ".join("%s %s" % (fmt(rw["r"], 3), rw["kind"][:3]) for rw in rows) or "none"


def acceptance_lines(res):
    cfg, recs = res["config"], res["records"]
    sigma, thr = cfg["sigma"], cfg["gate_threshold"]
    L = []
    for key in sorted(READ_AGAINST):
        ref = READ_AGAINST[key]
        rec = recs.get(key)
        if rec is None:
            L.append("- T/r_s %s: not in this run." % key)
            continue
        ds, cmpj = rec["dense"], rec.get("comparison_with_json", {})
        o, _ = _outer_inner(ds)
        if o is None:
            L.append("- T/r_s %s: no outer root of N - |v| on the dense fields." % key)
            continue
        sr, gi = o["sigma_over_R_arith"], o["gate_interp"]["centred"]
        tan = None if o["tangential_interp"] is None else sigma * abs(o["tangential_interp"])
        rad = None if o["radial_interp"]["centred"] is None else sigma * abs(o["radial_interp"]["centred"])
        core = ds["regions"].get("r_le_R_out") or {}
        ao, _ = _outer_inner(rec["applied"])
        L.append(("- T/r_s %s: dense outer root R_out = %s a (json R_t %s, difference %s); sigma/R_out = %s (arith) "
                  "against %s (%s): difference %s; sigma||K||_op at the root (dense, centred) = %s (tangential part %s, "
                  "radial part %s), %s the threshold %s; core maximum (%s) %s at r = %s (%s part); outermost failing "
                  "site %s a (failure contained inside R_out: %s); applied fields: outer root %s a, sigma/R = %s (arith), "
                  "sigma||K||_app at it %s.") % (
                 key, fmt(o["r"], 4), fmt(cmpj.get("R_t_json"), 4), fmt(cmpj.get("R_out_dense_minus_R_t_json"), 7),
                 fmt(sr, 5), fmt(ref["value"], 3), ref["source"], fmt(None if sr is None else sr - ref["value"], 5),
                 fmt(gi, 5), fmt(tan, 5), fmt(rad, 5), "exceeds" if (gi is not None and gi > thr) else "does not exceed",
                 fmt(thr, 3), ds["core_region_used"], fmt(core.get("value"), 4), fmt(core.get("r"), 3),
                 core.get("part") or "-", fmt(ds.get("r_fail_max"), 3), fmt(ds.get("fail_contained_in_R_out")),
                 fmt(None if ao is None else ao["r"], 4), fmt(None if ao is None else ao["sigma_over_R_arith"], 5),
                 fmt(None if ao is None else ao["gate_interp"]["centred"], 5)))
    keys = [k for k in recs if LATE_ROOT["T_lo"] - 1e-9 <= float(k) <= LATE_ROOT["T_hi"] + 1e-9]
    if keys:
        rows = []
        for k in sorted(keys, key=float):
            o, _ = _outer_inner(recs[k]["dense"])
            if o is not None:
                rows.append((o["r"], o["sigma_over_R_arith"], o["gate_interp"]["centred"]))
        if rows:
            Rv, sv = [r[0] for r in rows], [r[1] for r in rows]
            gv = [r[2] for r in rows if r[2] is not None]
            nfail = sum(1 for r in rows if r[2] is not None and r[2] > thr)
            L.append(("- T/r_s %s to %s (%d records with an outer root of %d in the run): R_out from %s to %s a against "
                      "the band %s to %s a (%s); sigma/R_out from %s to %s (arith) against %s to %s; sigma||K||_op at the "
                      "outer root from %s to %s; %d of %d exceed %s; every root below 10 sigma = %s a (arith) fails by the "
                      "tangential bound alone.") % (
                     fmt(LATE_ROOT["T_lo"], 3), fmt(LATE_ROOT["T_hi"], 3), len(rows), len(keys),
                     fmt(min(Rv), 3), fmt(max(Rv), 3), fmt(LATE_ROOT["R_lo"], 1), fmt(LATE_ROOT["R_hi"], 1),
                     LATE_ROOT["source"], fmt(min(sv), 4), fmt(max(sv), 4), fmt(LATE_ROOT["ratio_lo"], 3),
                     fmt(LATE_ROOT["ratio_hi"], 3), fmt(min(gv) if gv else None, 4), fmt(max(gv) if gv else None, 4),
                     nfail, len(rows), fmt(thr, 3), fmt(10.0 * sigma, 2)))
        else:
            L.append("- T/r_s %s to %s: no outer root on any processed record." % (LATE_ROOT["T_lo"], LATE_ROOT["T_hi"]))
    L.append("- Comparator (arith): the vacuum root at the nominal r_s = %s a would read sigma/r_s = %s, below %s "
             "because r_s >= 10 sigma = %s a." % (fmt(cfg["rs"], 1), fmt(sigma / cfg["rs"], 5), fmt(thr, 3),
                                                   fmt(10.0 * sigma, 1)))
    return L


def write_markdown(path, res, listed_keys):
    cfg, prov, recs = res["config"], res["provenance"], res["records"]
    sigma, thr = cfg["sigma"], cfg["gate_threshold"]
    L = ["# tensor_gate_v1: sigma ||K||_op on the stored dense fields", ""]
    L.append("Object. " + OBJECT_STATEMENT)
    L.append("")
    L.append("Inputs. npz %s (%d bytes%s); records json: %s. Every table number below is computed here from the npz "
             "unless its column says json (the records json) or arith (arithmetic on a number in the same row)." % (
                 prov["npz_path"], prov["npz_size_bytes"],
                 ("; sha256 " + prov["npz_sha256"]) if prov.get("npz_sha256") else "",
                 ", ".join(prov["records_json_paths"]) or "none"))
    L.append("")
    L.append("Constants. a = %s (median site spacing of R, from the npz); ell = %s (npz key ell); sigma = %s (%s); "
             "r_s = %s (%s); gate threshold %s (%s); stride of the sigma-scale rule s = %d sites; %s." % (
                 fmt(cfg["a"], 6), fmt(cfg["ell"], 3), fmt(sigma, 4), cfg["sigma_source"], fmt(cfg["rs"], 3),
                 cfg["rs_source"], fmt(thr, 3), cfg["gate_threshold_source"], cfg["stride_sigma"], cfg["T_units"]))
    L.append("")
    L.append("Fields. " + FIELDS_STATEMENT)
    L.append("")
    L.append("Derivative. " + DERIVATIVE_STATEMENT)
    L.append("")
    L.append("## Acceptance read (values from the json of this run; the reference numbers are inputs, sources named)")
    L.extend(acceptance_lines(res))
    L.append("")
    keys = [k for k in listed_keys if k in recs]
    L.append("## Table A: the gate at the roots and in the core (dense fields; applied beside)")
    hdr = ["T/r_s", "dense roots of N-|v| (fields) [a]", "json trapping_zeros [a]", "sigma/R_out (arith)",
           "at R_out: sigma|v/(rN)| / sigma|v'/N| / sigma||K||", "at R_in: sigma||K||",
           "max in core (@r, part)", "max r<=R_c (json R_c)", "r_fail_max [a] (inside R_out?)",
           "applied: roots [a]; sigma||K||_app at its R_out; max r<=R_out(dense)"]
    rows = []
    for k in keys:
        rec = recs[k]
        ds, aps, cmpj = rec["dense"], rec["applied"], rec.get("comparison_with_json", {})
        o, i = _outer_inner(ds)
        ao, _ = _outer_inner(aps)
        core = ds["regions"].get("r_le_R_out") or ds["regions"].get("r_le_R_c_json") or ds["regions"]["r_le_rs_nominal"]
        rc = ds["regions"].get("r_le_R_c_json")
        jz = cmpj.get("json_trapping_zeros")
        rows.append([
            k, roots_text(ds["roots"]), ("; ".join(fmt(z, 3) for z in jz) if jz else "-"),
            fmt(None if o is None else o["sigma_over_R_arith"], 5),
            "-" if o is None else "%s / %s / %s" % (
                fmt(None if o["tangential_interp"] is None else sigma * abs(o["tangential_interp"]), 5),
                fmt(None if o["radial_interp"]["centred"] is None else sigma * abs(o["radial_interp"]["centred"]), 5),
                fmt(o["gate_interp"]["centred"], 5)),
            fmt(None if i is None else i["gate_interp"]["centred"], 5),
            "%s (@%s, %s)" % (fmt(core.get("value"), 4), fmt(core.get("r"), 2), core.get("part") or "-"),
            "-" if rc is None else "%s (@%s; R_c %s)" % (fmt(rc.get("value"), 4), fmt(rc.get("r"), 2),
                                                        fmt(cmpj.get("R_c_json"), 2)),
            "%s (%s)" % (fmt(ds.get("r_fail_max"), 2), fmt(ds.get("fail_contained_in_R_out"))),
            "%s; %s; %s" % (roots_text(aps["roots"]), fmt(None if ao is None else ao["gate_interp"]["centred"], 5),
                            fmt((aps["regions"].get("r_le_R_out_dense") or {}).get("value"), 4)),
        ])
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("## Table B: the derivative rule's sensitivity and the field-reading cross-checks (dense fields)")
    hdr = ["T/r_s", "sigma||K|| at R_out: centred / forward / backward / sigma-stride",
           "max in core: centred / forward / backward / sigma-stride",
           "max_core sigma|dv'|/N: forward / backward / sigma-stride (vs centred)",
           "scalar sigma|v'| max core: a-rule / sigma-rule", "min N (core)", "n(N<=0)",
           "min(N-|v|) @ r: fields / json", "(R_out, R_out+sigma]: min / max sigma||K|| (n failing / n)"]
    rows = []
    for k in keys:
        rec = recs[k]
        ds, cmpj = rec["dense"], rec.get("comparison_with_json", {})
        o, _ = _outer_inner(ds)
        se, sc, ex, mf = ds["sensitivity"], ds["scalar_sigma_abs_dv"], ds["exterior_side_within_sigma"], ds["min_N_minus_absv"]
        rows.append([
            k,
            "-" if o is None else " / ".join(fmt(o["gate_interp"][r], 5) for r in RULE_NAMES),
            " / ".join(fmt(se[r]["core_max_gate"], 4) for r in RULE_NAMES),
            " / ".join(fmt(se[r]["max_core_sigma_abs_dlam_r"], 5) for r in ("forward", "backward", "stride_sigma")),
            "%s / %s" % (fmt(sc["sigma_abs_dv_centred_core_max"], 4), fmt(sc["sigma_abs_dv_stride_sigma_core_max"], 4)),
            fmt(ds["min_N_core"], 4), ds["n_sites_N_nonpositive"],
            "%s @ %s / %s @ %s" % (fmt(mf["value"], 5), fmt(mf["r"], 3), fmt(cmpj.get("min_N_minus_absv_json"), 5),
                                   fmt(cmpj.get("r_min_N_minus_absv_json"), 3)),
            "-" if not ex else "%s / %s (%d / %d)" % (fmt(ex["min"], 4), fmt(ex["max"], 4), ex["n_failing"], ex["n_sites"]),
        ])
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("Columns. 'core' is %s per record (its name in the json: core_region_used). The tangential part at a root "
             "equals sigma/R_out up to the linear-interpolation residual (json: tangential_interp_residual). The applied "
             "columns are the back-averaged fields, not the gate's object (the matter's root is distinct)." % (
                 "0 < r <= R_out where an outer dense root exists, else 0 < r <= R_c (json), else 0 < r <= r_s"))
    L.append("")
    L.append("Files. json %s; profiles npz %s%s." % (res["files"]["json"], res["files"]["profiles_npz"],
                                                    ("; all-records table " + res["files"]["all_md"]) if res["files"].get("all_md") else ""))
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_all_markdown(path, res):
    recs = res["records"]
    thr = res["config"]["gate_threshold"]
    L = ["# tensor_gate_v1: every record of the history (dense fields; 1 line per record)", "",
         "Columns as in Table A of the main file; sigma ||K||_op at the outer root with the centred rule; "
         "threshold %s." % fmt(thr, 3), ""]
    hdr = ["T/r_s", "dense roots [a]", "sigma/R_out (arith)", "sigma||K|| at R_out", "max in core (@r)",
           "r_fail_max [a]", "inside R_out?", "applied R_out [a]", "sigma||K||_app at applied R_out"]
    rows = []
    for k in sorted(recs, key=float):
        rec = recs[k]
        ds, aps = rec["dense"], rec["applied"]
        o, _ = _outer_inner(ds)
        ao, _ = _outer_inner(aps)
        core = ds["regions"].get("r_le_R_out") or ds["regions"].get("r_le_R_c_json") or ds["regions"]["r_le_rs_nominal"]
        rows.append([k, roots_text(ds["roots"]), fmt(None if o is None else o["sigma_over_R_arith"], 5),
                     fmt(None if o is None else o["gate_interp"]["centred"], 5),
                     "%s (@%s)" % (fmt(core.get("value"), 4), fmt(core.get("r"), 2)),
                     fmt(ds.get("r_fail_max"), 2), fmt(ds.get("fail_contained_in_R_out")),
                     fmt(None if ao is None else ao["r"], 3),
                     fmt(None if ao is None else ao["gate_interp"]["centred"], 5)])
    L.append(md_table(hdr, rows))
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def save_profiles(path, R, new_profiles):
    data = {}
    if os.path.exists(path):
        try:
            old = np.load(path, allow_pickle=False)
            for k in old.files:
                data[k] = old[k]
        except Exception as exc:  # a damaged cache is rewritten
            print("warning: the existing profiles npz could not be read (%s); rewriting it" % (exc,))
            data = {}
    data["R"] = np.asarray(R, dtype=float)
    for key, arrs in new_profiles.items():
        for name, arr in arrs.items():
            data["%s__%s" % (key, name)] = np.asarray(arr, dtype=float)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------
# input

def load_npz(path):
    z = np.load(path, allow_pickle=False)
    try:
        _ = z["R"]
    except ValueError:  # only if the required keys were pickled, which FIELDS_FORMAT does not say
        z = np.load(path, allow_pickle=True)
    return z


def read_scalar(z, key):
    if key not in z.files:
        return None
    try:
        return float(np.asarray(z[key]).ravel()[0])
    except Exception:
        return None


def load_records_json(paths):
    args, recs, used = {}, {}, []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception as exc:
            print("warning: records json %s not read (%s)" % (p, exc))
            continue
        used.append(p)
        args.update(d.get("args", {}) or {})
        for r in d.get("records", []) or []:
            if "T_over_rs" in r:
                recs["%.3f" % float(r["T_over_rs"])] = r
    return args, recs, used


# ----------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="sigma ||K||_op on the stored dense fields (tensor_gate_v1).")
    ap.add_argument("npz", help="the fields npz (FIELDS_FORMAT.md)")
    ap.add_argument("--records-json", default=None,
                    help="a records json (the full json beside the npz is read when present; this one is read as well)")
    ap.add_argument("--out", default="tensor_gate_v1_out/tensor_gate_v1", help="output prefix")
    ap.add_argument("--clocks", default=DEFAULT_CLOCKS,
                    help="T/r_s list; ranges lo:hi[:step] inclusive (default %s)" % DEFAULT_CLOCKS)
    ap.add_argument("--all", action="store_true", help="also every record of the history (summary only)")
    ap.add_argument("--sigma", type=float, default=None, help="override sigma (default ell*a from the npz)")
    ap.add_argument("--rs", type=float, default=None, help="override r_s/a (default: json args rs, else 64)")
    ap.add_argument("--gate", type=float, default=0.1, help="the threshold (default 0.1)")
    ap.add_argument("--force", action="store_true", help="recompute cached records")
    ap.add_argument("--no-json-profiles", action="store_true", help="do not embed the 3 headline profiles in the json")
    ap.add_argument("--hash", action="store_true", help="record the npz's sha256 (reads the whole file once)")
    args = ap.parse_args(argv)
    t_start = time.time()

    z = load_npz(args.npz)
    R = np.asarray(z["R"], dtype=float)
    T = np.asarray(z["T"], dtype=float)
    N = np.asarray(z["N"], dtype=float)
    v = np.asarray(z["v"], dtype=float)
    Na = np.asarray(z["N_applied"], dtype=float)
    va = np.asarray(z["v_applied"], dtype=float)
    ell = read_scalar(z, "ell")

    beside = args.npz[:-len("_fields.npz")] + ".json" if args.npz.endswith("_fields.npz") else None
    jargs, jrecs, jused = load_records_json([beside, args.records_json])

    a = float(np.median(np.diff(R)))
    if args.rs is not None:
        rs, rs_src = float(args.rs), "--rs"
    elif "rs" in jargs:
        rs, rs_src = float(jargs["rs"]), "records json args rs"
    else:
        rs, rs_src = 64.0, "FIELDS_FORMAT.md (T/r_s = T/64; r_s/a 64 history)"
    if args.sigma is not None:
        sigma, sig_src = float(args.sigma), "--sigma"
    elif ell:
        sigma, sig_src = ell * a, "ell * a, ell from the npz key ell (the engine's Gaussian standard deviation, requirements sec. 6)"
    else:
        sigma, sig_src = 4.0 * a, "4a (the brief; npz key ell absent)"
    stride_sigma = max(1, int(round(sigma / a)))
    tors, t_note = clocks_in_rs(T, rs)
    cfg = {"a": a, "ell": ell, "sigma": sigma, "sigma_source": sig_src, "rs": rs, "rs_source": rs_src,
           "gate_threshold": float(args.gate),
           "gate_threshold_source": "requirements sec. 5 I1 (H ell of order 0.1); item 5's illustrative sigma||K||_op <= 0.1",
           "stride_sigma": stride_sigma, "T_units": t_note, "n_records": int(T.size), "n_sites": int(R.size),
           "R_first": fl(R[0]), "R_last": fl(R[-1])}
    prov = {"version": VERSION, "npz_path": os.path.abspath(args.npz), "npz_size_bytes": int(os.path.getsize(args.npz)),
            "npz_keys_used": ["R", "T", "N", "v", "N_applied", "v_applied", "ell"],
            "records_json_paths": [os.path.abspath(p) for p in jused],
            "solver": (str(z["solver"]) if "solver" in z.files else None),
            "label": (str(z["label"]) if "label" in z.files else None)}
    if args.hash:
        prov["npz_sha256"] = sha256_of(args.npz)

    out_prefix = args.out
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path, md_path = out_prefix + ".json", out_prefix + ".md"
    prof_path, all_md_path = out_prefix + "_profiles.npz", out_prefix + "_all.md"

    res = {"version": VERSION, "provenance": prov, "config": cfg,
           "statements": {"object": OBJECT_STATEMENT, "fields": FIELDS_STATEMENT, "derivative": DERIVATIVE_STATEMENT},
           "read_against": {"records": READ_AGAINST, "late_root": LATE_ROOT},
           "files": {"json": os.path.abspath(json_path), "markdown": os.path.abspath(md_path),
                     "profiles_npz": os.path.abspath(prof_path)},
           "records": {}}
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path) as fh:
                old = json.load(fh)
            oc = old.get("config", {})
            same = all(abs(float(oc.get(k, np.nan)) - float(cfg[k])) <= 1e-12 * max(1.0, abs(float(cfg[k])))
                       for k in ("a", "sigma", "rs", "gate_threshold"))
            if not same:
                sys.exit("existing %s was computed with another configuration; use --force or another --out" % json_path)
            res["records"] = old.get("records", {})
            print("resuming: %d records cached in %s" % (len(res["records"]), json_path))
        except (ValueError, OSError) as exc:
            print("warning: the existing json could not be read (%s); recomputing" % (exc,))

    listed = parse_clocks(args.clocks)
    step = float(np.median(np.diff(tors))) if tors.size > 1 else 0.125
    todo = []
    for c in listed:
        idx = int(np.argmin(np.abs(tors - c)))
        if abs(tors[idx] - c) > 0.5 * step:
            print("warning: no record at T/r_s %.3f (nearest %.3f); skipped" % (c, tors[idx]))
            continue
        todo.append((idx, True))
    listed_keys = ["%.3f" % tors[idx] for idx, _ in todo]
    if args.all:
        have = set(idx for idx, _ in todo)
        todo += [(idx, False) for idx in range(tors.size) if idx not in have]

    new_profiles = {}
    n_done = 0
    for count, (idx, keep) in enumerate(todo):
        key = "%.3f" % tors[idx]
        if key in res["records"] and not args.force:
            continue
        jrec = jrecs.get(key)
        rec, prof = analyse_record(idx, tors[idx], T[idx], R, N[idx], v[idx], Na[idx], va[idx], cfg, jrec,
                                   keep_profile=(keep and key in ("12.125", "30.000", "40.000") and not args.no_json_profiles))
        res["records"][key] = rec
        if keep:
            new_profiles[key] = prof
        n_done += 1
        ds = rec["dense"]
        o, _ = _outer_inner(ds)
        core = ds["regions"].get("r_le_R_out") or ds["regions"].get("r_le_R_c_json") or ds["regions"]["r_le_rs_nominal"]
        print("T/r_s %s: dense roots %s; sigma/R_out %s; sigma||K|| at R_out %s; core max %s at r %s; r_fail_max %s; "
              "applied R_out %s" % (key, roots_text(ds["roots"]), fmt(None if o is None else o["sigma_over_R_arith"], 5),
                                    fmt(None if o is None else o["gate_interp"]["centred"], 5),
                                    fmt(core.get("value"), 4), fmt(core.get("r"), 2), fmt(ds.get("r_fail_max"), 2),
                                    fmt(rec["applied"]["R_out"], 3)))
        if keep or (count % 50 == 49):
            write_json(json_path, res)

    if args.all:
        res["files"]["all_md"] = os.path.abspath(all_md_path)
    write_json(json_path, res)
    if new_profiles:
        save_profiles(prof_path, R, new_profiles)
    write_markdown(md_path, res, listed_keys)
    if args.all:
        write_all_markdown(all_md_path, res)
    print("")
    for line in acceptance_lines(res):
        print(line)
    print("")
    print("records computed %d (cached before %d); wrote %s, %s%s; %.1f s" % (
        n_done, len(res["records"]) - n_done, json_path, md_path, (", " + all_md_path) if args.all else "",
        time.time() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
