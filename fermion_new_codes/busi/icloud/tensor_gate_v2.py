#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_gate_v2.py: sigma ||K||_op on the r_s/a 64, tau 8, sigma 4a history: version 1 (the stored site arrays N, v and the
applied arrays N_applied, v_applied) plus the node pass (--node-pass), which re-solves each requested record's dense constraint
solution from the stored sources with the engine's own solver (read-only) and evaluates the gate on it at the record's R_t.

An independent check for the busi lane in the icloud lane's role (WEB40G_icloud_tasks.md item 6; sources P17 claim 1 and
P33 claim 7, read through REVIEWA_REVIEW_web40g.md summary 7; the item 6 reading of the version 1 run found the site arrays
equal to the applied arrays to every printed digit and the deliverable, the gate on the dense fields at R_t, undone).

The object (requirements sec. 2 H2; item 5 of the paper at 22bae778e8). For the radial flow v_i = v(r) n_i the declared
K_ij = (d_i v_j + d_j v_i)/(2N) is
    K_ij = [ v'(r) n_i n_j + (v(r)/r) (delta_ij - n_i n_j) ] / N,
with principal values v'/N (once, eigenvector n: radial) and v/(rN) (twice: tangential). The operator norm of the symmetric
tensor is the largest magnitude of its principal values, so
    sigma ||K||_op = sigma max(|v'/N|, |v/(rN)|).
At a root of N - |v| = 0 with v < 0, v/(rN) = -1/R exactly, no derivative entering, so sigma ||K||_op >= sigma/R there and
the criterion sigma ||K||_op <= 0.1 fails wherever R < 10 sigma.

The node pass. The record's R_t is the last trapping zero of the engine's dense constraint solution (engine_collapse_v19:
FLD.trapping_radius(N, v, R, info) on the info dict of the constraint solve, FLD = engine_fields_v14). The stored npz holds
the sources of that solve (eps_tot, J_tot per record), so the pass calls the engine's solve_constraints_ivp on them at the
run's arguments, takes the zeros through the engine's trapping_radius, and accepts the solution as the record's only when
they agree with the json's trapping_zeros within --zero-tol; the principal values are then read from the dense functions
themselves (v' from the equations the solution integrates, not from a lattice difference). The engine is imported read-only
from --engine-path (sys.path insertion; no bytecode written; nothing copied or edited).

Usage:
  python3 -I -B tensor_gate_v2.py FIELDS.npz [--records-json RECORDS.json] [--out PREFIX]
          [--clocks "12.125,30,40:46:0.125"] [--all] [--sigma S] [--rs RS] [--gate 0.1] [--force] [--hash]
          [--node-pass --engine-path DIR [--zero-tol 1e-3] [--fine 64] [--rtol 1e-12] [--atol 1e-16]
           [--g G] [--collar-policy flag] [--support-rel 1e-14] [--check-applied] [--profile-rmax 96]]
Outputs: PREFIX.json (the per-record summary, the node pass under records[key]["node_pass"]; the resumable cache), PREFIX.md
(the tables), PREFIX_profiles.npz (the profiles along r for the listed clocks; the dense profiles under <key>__node_*),
PREFIX_all.md (with --all: 1 line per record of the history; the node pass runs on the listed clocks only).
Version 1's part: numpy and the standard library, seconds. The node pass: scipy through the engine (solve_ivp, splines,
brentq), the engine's outward integration at rtol 1e-12 per record. python3 -I. No plot. No entanglement or mutual-information
quantity is computed (D11 revised, D15).
Exit status: 0; 2 when a listed record's zeros were not reproduced by the stored sources (that record's dense gate is then
not evaluated and the record says so; the other records are processed).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # the box's cap (FIELDS_FORMAT.md); the RUN
    os.environ.setdefault(_k, "8")                                           # command sets them too

import argparse
import hashlib
import importlib
import json
import sys
import time

import numpy as np

VERSION = "tensor_gate_v2"
DEFAULT_CLOCKS = "12.125,30,40:46:0.125"
RULE_NAMES = ("centred", "forward", "backward", "stride_sigma")
SITE, APPLIED = "site", "applied"

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
    "The paper's gate reads the dense solved fields, which this npz stores as the node representation r_nodes, N_nodes, "
    "v_nodes and whose dense functions the node pass re-solves from the stored sources eps_tot, J_tot; the site arrays N, v "
    "coincide with N_applied, v_applied to every printed digit on every record of the version 1 run and are read as the "
    "back-averaged fields (args site_fields averaged: the fields the matter Hamiltonian sees, requirements sec. 2), so "
    "version 1's tables (A and B here) are the gate on the site fields at the matter's root, printed beside the json's dense "
    "R_t; this script prints max|N - N_applied| and max|v - v_applied| over the whole arrays and warns when both are 0. "
    "Ground for the root: item 5 (22bae778e8) defines the trapping surface as 'the outer zero R_t of N-|v| on the dense "
    "solved fields (the geometric root ...; the matter's root on the back-averaged fields is distinct)' and its tensor clause "
    "holds 'at a dense root of N=|v|'; the records' trapping_zeros are those dense roots (engine_collapse_v19 takes R_t from "
    "FLD.trapping_radius on the constraint solve's info dict; NOTES P17 claim 1 verification: 'placement uses the dense "
    "root'). The node pass (Table C) is the gate on that geometry at that root; the site and applied columns are not the "
    "gate's object.")

DERIVATIVE_STATEMENT = (
    "Version 1's lattice rule, for the site and applied arrays only: v'(r_j) = (v_{j+1} - v_{j-1})/(R_{j+1} - R_{j-1}) at "
    "interior sites (the centred 2-point rule at the lattice spacing a), (v_1 - v_0)/(R_1 - R_0) at the first site and "
    "(v_{n-1} - v_{n-2})/(R_{n-1} - R_{n-2}) at the last. Sensitivity printed: 'forward' (v_{j+1} - v_j)/(R_{j+1} - R_j); "
    "'backward' (v_j - v_{j-1})/(R_j - R_{j-1}); 'stride_sigma', the centred rule on nodes sigma apart, "
    "(v_{j+s} - v_{j-s})/(R_{j+s} - R_{j-s}) with s = round(sigma/a) (one-sided over the same stride within s sites of the "
    "ends), the node spacing of the paper's 1-term test. The tangential value v/(rN) uses no derivative; where R = 0 (a "
    "regular centre) v/r is replaced by v'. Where N <= 0 the tensor is undefined: the site is excluded from every maximum and "
    "counted. The node pass uses no lattice difference (see Node pass).")

NODE_PASS_STATEMENT = (
    "The node pass re-solves each listed record's constraint rows from the stored sources eps_tot and J_tot with the engine's "
    "own solve_constraints_ivp (engine_fields_v14, imported read-only from --engine-path; g and ell from the npz keys g and "
    "ell, the collar policy and support_rel from the records json's args, rtol and atol as given), takes the solution's "
    "trapping zeros through the engine's trapping_radius (the call engine_collapse_v19 makes for R_t) and accepts the "
    "solution as the record's dense geometry only when the zeros' count agrees with the json's trapping_zeros and every zero "
    "agrees within --zero-tol; otherwise the record is reported with the differences and its dense gate is not evaluated. On "
    "an accepted solution both principal values are read from the dense functions themselves: the tangential v/(rN) from "
    "N_dense and v_dense; the radial v'/N from the equations the solution integrates, v = -sqrt(W/r) and "
    "W' = 8 pi g r^2 N^2 rho_ell, so v' = (W' - W/r)/(2 r v) (hand; where W <= 0 the dense v is 0 and so is v'), no lattice "
    "difference entering. Two checks print beside the radial value at R_t: the Richardson-extrapolated centred difference of "
    "v_dense at h = ell/64 (the engine's own residual rule) and the Hamiltonian constraint at a root, "
    "2 lam_t (2 lam_r + lam_t) = 16 pi g rho_ell, which gives lam_r = 1/(2R) - 4 pi g R rho_ell(R) (hand) and makes the "
    "radial value exceed the tangential 1/R only where rho_ell(R) > 3/(8 pi g R^2) (hand). The profiles are on the engine's "
    "fine grid, spacing ell/fine from the centre to the collar; the core is 0 < r <= R_t (json). The stored node arrays "
    "r_nodes, N_nodes, v_nodes and the stored site and applied arrays are compared with the re-solved solution (its nodes, "
    "its dense values at the sites and, with --check-applied, the engine's kernel average back onto the sites).")


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


def fmte(x, d=2):
    if x is None:
        return "-"
    try:
        return ("%%.%de" % d) % float(x)
    except (TypeError, ValueError):
        return str(x)


def safe_max(x):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    return fl(x[m].max()) if m.any() else None


def jsonable(o):
    """numpy scalars, arrays and nested containers to json-strict Python values."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return fl(o)
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()] if o.ndim else jsonable(o.item())
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    return str(o)


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
# version 1's computation (the site and applied arrays; the lattice rule)

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
    """Maximum of the gate over the masked points, with its radius, the dominating principal value, the count
    of failing points and of undefined points (N <= 0) in the region."""
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


def analyse_fields(R, N, v, sigma, thr, stride_sigma, rs, regions_extra=None, label=SITE):
    """The principal values, the gate, the roots and the maxima for 1 pair of lattice arrays (N, v) on R."""
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
        core_mask, core_name = pos & (R <= R_out), "0 < r <= R_out (the outer root of the %s arrays)" % label
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
    site = analyse_fields(R, N, v, sigma, thr, s, rs,
                          regions_extra=({"r_le_R_c_json": R_c} if R_c is not None else None), label=SITE)
    ss = site["summary"]
    extra_a = {}
    if ss["R_out"] is not None:
        extra_a["r_le_R_out_site"] = ss["R_out"]
    if R_c is not None:
        extra_a["r_le_R_c_json"] = R_c
    applied = analyse_fields(R, Na, va, sigma, thr, s, rs, regions_extra=(extra_a or None), label=APPLIED)
    aps = applied["summary"]
    gs, ga = site["profiles"]["gate_centred"], applied["profiles"]["gate_centred"]
    cross = {
        "applied_gate_centred_at_site_roots": [{"r": rw["r"], "kind": rw["kind"], "value": fl(interp(rw["r"], R, ga))}
                                               for rw in ss["roots"]],
        "site_gate_centred_at_applied_roots": [{"r": rw["r"], "kind": rw["kind"], "value": fl(interp(rw["r"], R, gs))}
                                               for rw in aps["roots"]],
    }
    arrays = {"max_abs_N_minus_N_applied": fl(np.max(np.abs(np.asarray(N, float) - np.asarray(Na, float)))),
              "max_abs_v_minus_v_applied": fl(np.max(np.abs(np.asarray(v, float) - np.asarray(va, float))))}
    cmp = {}
    if jrec:
        tz = [fl(x) for x in (jrec.get("trapping_zeros") or [])]
        fr = [rw["r"] for rw in ss["roots"]]
        cmp["json_trapping_zeros"] = tz
        cmp["nearest_site_root_minus_json_zero"] = [(min(fr, key=lambda x: abs(x - zz)) - zz) if fr else None
                                                    for zz in tz]
        cmp["R_t_json"] = fl(jrec.get("R_t"))
        cmp["R_out_site_minus_R_t_json"] = (fl(ss["R_out"] - cmp["R_t_json"])
                                            if (ss["R_out"] is not None and cmp["R_t_json"] is not None) else None)
        cmp["sigma_over_R_t_json_arith"] = fl(sigma / cmp["R_t_json"]) if cmp["R_t_json"] else None
        cmp["min_N_minus_absv_json"] = fl(jrec.get("min_N_minus_absv"))
        cmp["r_min_N_minus_absv_json"] = fl(jrec.get("r_min_N_minus_absv"))
        mf = ss["min_N_minus_absv"]
        cmp["min_F_site_minus_json"] = (fl(mf["value"] - cmp["min_N_minus_absv_json"])
                                        if (mf["value"] is not None and cmp["min_N_minus_absv_json"] is not None) else None)
        cmp["r_min_F_site_minus_json"] = (fl(mf["r"] - cmp["r_min_N_minus_absv_json"])
                                          if (mf["r"] is not None and cmp["r_min_N_minus_absv_json"] is not None) else None)
        cmp["R_c_json"] = R_c
        cmp["E_json"] = fl(jrec.get("E"))
    rec = {"T_over_rs": fl(tors), "record_index": int(idx), "T_stored": fl(T_val),
           SITE: ss, APPLIED: aps, "site_vs_applied_arrays": arrays, "cross": cross, "comparison_with_json": cmp}
    if keep_profile:
        rec["profile"] = {"R": [fl(x) for x in R]}
        for name in ("N", "v", "lam_r_centred", "lam_t", "gate_centred", "gate_stride_sigma"):
            rec["profile"]["site_" + name] = [fl(x) for x in site["profiles"][name]]
        for name in ("N", "v", "lam_r_centred", "lam_t", "gate_centred"):
            rec["profile"]["applied_" + name] = [fl(x) for x in applied["profiles"][name]]
    prof = {}
    for name in ("lam_r_centred", "lam_t", "gate_centred", "gate_stride_sigma", "dv_centred"):
        prof["site_" + name] = site["profiles"][name]
    for name in ("lam_r_centred", "lam_t", "gate_centred"):
        prof["applied_" + name] = applied["profiles"][name]
    return rec, prof


# ----------------------------------------------------------------------------------------------------------
# the node pass: the dense solution and its derivative from the equations it integrates

def _vec(fn, x):
    """fn evaluated at the array x, a constant return broadcast (manufactured functions in the self-test)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(fn(x), dtype=float)
    if y.size == 1 and x.size != 1:
        return np.full(x.shape, float(y.ravel()[0]))
    return y.reshape(x.shape)


def dense_v_prime(rr, fns, g):
    """v'(r) from the dense solution's own equations, no lattice difference: v = -sqrt(W/r) and the W equation
    W' = 8 pi g r^2 N^2 rho_ell give v' = (W' - W/r)/(2 r v) where W > 0 (hand); where W <= 0 the dense v is 0 and so is v'.
    fns = (N_fn, v_fn, W_fn, rho_fn), the engine's info["N_dense"], info["v_dense"], info["W_field"], info["rho_ell"].
    Returns (dv, N, v, W, rho) on rr."""
    N_fn, v_fn, W_fn, rho_fn = fns
    rr = np.asarray(rr, dtype=float)
    N = _vec(N_fn, rr)
    v = _vec(v_fn, rr)
    W = _vec(W_fn, rr)
    rho = _vec(rho_fn, rr)
    Wp = 8.0 * np.pi * g * rr ** 2 * N ** 2 * rho
    dv = np.zeros(rr.shape)
    m = (W > 0.0) & (rr > 0.0) & (v != 0.0)
    dv[m] = (Wp[m] - W[m] / rr[m]) / (2.0 * rr[m] * v[m])
    return dv, N, v, W, rho


def dense_difference_dv(r, v_fn, h):
    """The engine's residual rule on the dense output (residuals_at): the Richardson-extrapolated centred difference
    (4 D(h/2) - D(h))/3 of v_dense. A check of the equation-derived v', not the headline."""
    r = np.asarray(r, dtype=float)

    def cd(hh):
        return (_vec(v_fn, r + hh) - _vec(v_fn, r - hh)) / (2.0 * hh)
    return (4.0 * cd(h / 2.0) - cd(h)) / 3.0


def dense_principal_values(rr, fns, g, sigma):
    """lam_r = v'/N and lam_t = v/(rN) on the dense solution at the radii rr, v' from dense_v_prime. At r = 0 the
    tangential value takes the radial one (a regular centre); where N <= 0 both are nan. Returns a dict of arrays."""
    rr = np.asarray(rr, dtype=float)
    dv, N, v, W, rho = dense_v_prime(rr, fns, g)
    ok = N > 0
    lam_r = np.full(rr.shape, np.nan)
    lam_t = np.full(rr.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_r[ok] = dv[ok] / N[ok]
        pos = ok & (rr > 0)
        lam_t[pos] = v[pos] / (rr[pos] * N[pos])
    centre = ok & ~(rr > 0)
    lam_t[centre] = lam_r[centre]
    gate = sigma * np.fmax(np.abs(lam_r), np.abs(lam_t))
    return {"r": rr, "N": N, "v": v, "W": W, "rho": rho, "dv": dv, "lam_r": lam_r, "lam_t": lam_t, "gate": gate}


def dense_point(r, fns, g, sigma, thr, ell):
    """Both principal values and the gate at 1 radius of the dense solution, with the 2 checks of the radial value."""
    r = float(r)
    p = dense_principal_values(np.array([r]), fns, g, sigma)
    N0, v0, W0, rho0, dv0, lr, lt, gt = (float(p[k][0]) for k in ("N", "v", "W", "rho", "dv", "lam_r", "lam_t", "gate"))
    d = {"r": fl(r), "N": fl(N0), "v": fl(v0), "W": fl(W0), "rho_ell": fl(rho0), "N_minus_absv": fl(N0 - abs(v0)),
         "dv_equations": fl(dv0), "lam_r": fl(lr), "lam_t": fl(lt),
         "sigma_tangential": fl(sigma * abs(lt)), "sigma_radial": fl(sigma * abs(lr)), "gate": fl(gt),
         "part": (("tangential" if abs(lt) >= abs(lr) else "radial") if (np.isfinite(lt) and np.isfinite(lr)) else None),
         "sigma_over_r_arith": fl(sigma / r) if r > 0 else None,
         "tangential_plus_1_over_r": fl(lt + 1.0 / r) if r > 0 else None,
         "fails_gate": bool(np.isfinite(gt) and gt > thr),
         "margin_abs_arith": fl(gt - thr), "margin_percent_arith": (fl(100.0 * (gt / thr - 1.0)) if thr else None)}
    h = ell / 64.0
    if r > h:
        dvd = float(dense_difference_dv(np.array([r]), fns[1], h)[0])
        d["dv_difference_rule_h_ell_over_64"] = fl(dvd)
        d["sigma_radial_difference_rule"] = fl(sigma * abs(dvd / N0)) if N0 > 0 else None
        d["dv_equations_minus_difference_rule"] = fl(dv0 - dvd)
    if r > 0:
        lri = 1.0 / (2.0 * r) - 4.0 * np.pi * g * r * rho0
        d["lam_r_constraint_identity_at_root_arith"] = fl(lri)
        d["sigma_radial_constraint_identity_arith"] = fl(sigma * abs(lri))
        d["lam_r_equations_minus_constraint_identity"] = fl(lr - lri)
        d["rho_threshold_radial_exceeds_tangential_arith"] = fl(3.0 / (8.0 * np.pi * g * r * r))
    return d


def dense_zeros_scan(F, r_lo, r_hi, step, xtol=1e-12):
    """The checker's own scan of the zeros of F (vectorised callable) on [r_lo, r_hi]: sign changes refined by brentq, sampled
    zeros kept; the engine's trapping_radius method, used beside it as a cross-check."""
    from scipy.optimize import brentq
    rr = np.arange(float(r_lo), float(r_hi) + 1e-9, float(step))
    fv = np.asarray(F(rr), dtype=float).ravel()
    zeros = []
    for i in range(len(rr) - 1):
        if fv[i] == 0.0:
            zeros.append(float(rr[i]))
        elif fv[i] * fv[i + 1] < 0.0:
            zeros.append(float(brentq(lambda x: float(np.asarray(F(np.array([x])), dtype=float).ravel()[0]),
                                      rr[i], rr[i + 1], xtol=xtol)))
    return sorted(zeros)


def dense_region(P, mask, thr):
    d = region_max(P["gate"], P["lam_r"], P["lam_t"], P["N"], P["r"], mask, thr)
    d["n_points"] = d.pop("n_sites")
    return d


def node_row(arr, idx):
    """1 record's row of a node array of the npz (FIELDS_FORMAT: r_nodes, N_nodes, v_nodes (776, 513) object) as a float vector."""
    row = arr[idx]
    if isinstance(row, np.ndarray) and row.dtype == object:
        if row.ndim == 0:
            row = row.item()
        else:
            row = np.concatenate([np.asarray(x, dtype=float).ravel() for x in row.ravel()])
    return np.asarray(row, dtype=float).ravel()


def import_engine(path):
    """engine_fields_v14 from the solve lane's code directory, read only: sys.path insertion, no bytecode written, nothing copied."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        sys.exit("--engine-path %s is not a directory" % path)
    sys.dont_write_bytecode = True
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        FLD = importlib.import_module("engine_fields_v14")
    except Exception as exc:
        sys.exit("engine_fields_v14 could not be imported from %s: %s: %s" % (path, type(exc).__name__, exc))
    for name in ("solve_constraints_ivp", "trapping_radius"):
        if not hasattr(FLD, name):
            sys.exit("engine_fields_v14 at %s has no %s" % (path, name))
    files = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and os.path.abspath(f).startswith(path + os.sep):
            files[name] = {"path": os.path.abspath(f), "bytes": int(os.path.getsize(f)), "sha256": sha256_of(f)}
    prov = {"path": path, "module": "engine_fields_v14", "module_file": os.path.abspath(FLD.__file__), "files": files,
            "read_only": "imported by sys.path insertion with sys.dont_write_bytecode set; nothing copied or edited",
            "dictionary": str(getattr(FLD, "DICTIONARY", ""))}
    try:
        import scipy
        prov["scipy_version"] = scipy.__version__
    except Exception:
        prov["scipy_version"] = None
    prov["numpy_version"] = np.__version__
    return FLD, prov


def engine_scalars(info, jrec):
    """The json-able diagnostics of the engine's info dict, with the json's counterparts where the record has them."""
    gauge = info.get("gauge") or {}
    regime = info.get("regime") or {}
    ivp = info.get("ivp") or {}
    ser = ivp.get("series") or {}
    d = {"module": info.get("module"), "failed": info.get("failed"), "it": info.get("it"), "converged": info.get("converged"),
         "r_collar": fl(info.get("r_collar")), "R_last": fl(info.get("R_last")), "collar_margin": jsonable(info.get("collar_margin")),
         "collar_policy": info.get("collar_policy"), "source_spline_error": fl(info.get("source_spline_error")),
         "residual_W": fl(info.get("residual_W")), "residual_invN": fl(info.get("residual_invN")),
         "E_collar": fl(info.get("E_collar")), "N_min_sites": fl(info.get("N_min")),
         "gauge_label": gauge.get("label"), "gauge_lapse_unresolved": jsonable(gauge.get("lapse_unresolved")),
         "regime_label": regime.get("label"), "regime_max_ratio": fl(regime.get("max_ratio")),
         "n_nodes": (int(len(info["r_nodes"])) if "r_nodes" in info else None),
         "interp_floor_N": fl(info.get("interp_floor_N")), "interp_floor_v": fl(info.get("interp_floor_v")),
         "N_centre": fl(ivp.get("N_centre")), "segments": jsonable(ivp.get("segments")), "nfev": jsonable(ivp.get("nfev")),
         "q1_series": fl(ser.get("q1")), "W_negative": jsonable(info.get("W_negative")),
         "W_roundoff_negative_points": jsonable(info.get("W_roundoff_negative_points"))}
    if jrec:
        if d["E_collar"] is not None and jrec.get("E") is not None:
            d["E_json"] = fl(jrec["E"])
            d["E_collar_minus_E_json"] = fl(d["E_collar"] - float(jrec["E"]))
        ind = jrec.get("indicator") or {}
        if ind.get("source_spline_error") is not None:
            d["source_spline_error_json"] = fl(ind["source_spline_error"])
    return d


def node_pass_record(FLD, z, idx, key, R, N_site, v_site, Na, va, cfg, ncfg, jrec, site_summary, keep_profile):
    """The node pass on 1 record: the re-solve, the reproduction check, the dense gate at R_t and over the core.
    Returns (summary, profiles for the npz)."""
    t0 = time.time()
    sigma, thr = cfg["sigma"], cfg["gate_threshold"]
    g, ell, tol = ncfg["g"], ncfg["ell"], ncfg["zero_tol"]
    R = np.asarray(R, dtype=float)
    out = {"reproduced": None, "stopped": None, "tolerance_a": fl(tol)}
    eps = np.asarray(z["eps_tot"][idx], dtype=float)
    J = np.asarray(z["J_tot"][idx], dtype=float)
    nz = (eps != 0.0) | (J != 0.0)
    out["sources"] = {"sum_eps_tot": fl(eps.sum()), "sum_abs_J_tot": fl(np.abs(J).sum()), "n_sites_nonzero": int(nz.sum()),
                      "R_last_nonzero": fl(R[nz].max()) if nz.any() else None}
    try:
        N_s, v_s, W_nodes, info = FLD.solve_constraints_ivp(eps, J, g, R, ell, rtol=ncfg["rtol"], atol=ncfg["atol"],
                                                            collar_policy=ncfg["collar_policy"], support_rel=ncfg["support_rel"])
    except Exception as exc:
        out["stopped"] = "the engine's solve_constraints_ivp raised %s: %s; the dense gate is not evaluated" % (type(exc).__name__, exc)
        out["seconds"] = fl(time.time() - t0)
        return out, {}
    out["engine"] = engine_scalars(info, jrec)
    if N_s is None or info.get("failed"):
        out["stopped"] = "the engine's solve failed: %s; the dense gate is not evaluated" % (info.get("failed"),)
        out["seconds"] = fl(time.time() - t0)
        return out, {}
    try:
        tz = FLD.trapping_radius(N_s, v_s, R, info)
    except Exception as exc:
        out["stopped"] = "the engine's trapping_radius raised %s: %s; the dense gate is not evaluated" % (type(exc).__name__, exc)
        out["seconds"] = fl(time.time() - t0)
        return out, {}
    zeros = [float(x) for x in (tz.get("zeros") or [])]
    out["zeros_engine"] = [fl(x) for x in zeros]
    out["tangencies_engine"] = [fl(x) for x in (tz.get("tangencies") or [])]
    out["min_N_minus_absv_engine"] = {"value": fl(tz.get("min")), "r": fl(tz.get("r_min"))}
    if jrec is None:
        out["stopped"] = ("no records json entry at this clock: the record's trapping_zeros are unknown, the solution cannot be "
                          "checked and the dense gate is not evaluated (engine zeros [%s])" % ", ".join("%.6f" % x for x in zeros))
        out["seconds"] = fl(time.time() - t0)
        return out, {}
    jz = [float(x) for x in (jrec.get("trapping_zeros") or [])]
    out["zeros_json"] = [fl(x) for x in jz]
    out["min_N_minus_absv_json"] = {"value": fl(jrec.get("min_N_minus_absv")), "r": fl(jrec.get("r_min_N_minus_absv"))}
    if tz.get("min") is not None and jrec.get("min_N_minus_absv") is not None:
        out["min_engine_minus_json"] = fl(float(tz["min"]) - float(jrec["min_N_minus_absv"]))
    if tz.get("r_min") is not None and jrec.get("r_min_N_minus_absv") is not None:
        out["r_min_engine_minus_json"] = fl(float(tz["r_min"]) - float(jrec["r_min_N_minus_absv"]))
    ztxt = lambda L_: ", ".join("%.6f" % x for x in L_) or "none"
    if len(zeros) != len(jz):
        out["reproduced"] = False
        out["stopped"] = ("NOT REPRODUCED: the stored sources give %d trapping zero(s) [%s] against the record's %d [%s]; "
                          "the dense gate is not evaluated" % (len(zeros), ztxt(zeros), len(jz), ztxt(jz)))
        out["seconds"] = fl(time.time() - t0)
        return out, {}
    diffs = [a_ - b_ for a_, b_ in zip(zeros, jz)]
    out["zero_differences_engine_minus_json"] = [fl(d_) for d_ in diffs]
    mx = max([abs(d_) for d_ in diffs]) if diffs else 0.0
    out["max_abs_zero_difference"] = fl(mx)
    out["reproduced"] = bool(mx <= tol)
    if not out["reproduced"]:
        out["stopped"] = ("NOT REPRODUCED: the stored sources do not reproduce the record's trapping zeros within %.1e a: engine [%s] "
                          "against json [%s], differences engine - json [%s] (max |difference| %.3e a); the dense gate is not evaluated"
                          % (tol, ztxt(zeros), ztxt(jz), ", ".join("%.3e" % d_ for d_ in diffs), mx))
        out["seconds"] = fl(time.time() - t0)
        return out, {}

    # the accepted dense solution: the principal values from its own functions and equations
    fns = (info["N_dense"], info["v_dense"], info["W_field"], info["rho_ell"])
    r_collar = float(info["r_collar"])
    rr = np.arange(0.0, r_collar + 1e-9, ell / float(ncfg["fine"]))
    P = dense_principal_values(rr, fns, g, sigma)
    R_t = fl(jrec.get("R_t"))
    if R_t is None and zeros:
        R_t = zeros[-1]
    R_t_eng = zeros[-1] if zeros else None
    r_in_json = jz[0] if len(jz) >= 2 else None
    R_c = fl(jrec.get("R_c"))
    out["R_t_json"] = R_t
    out["R_t_engine"] = fl(R_t_eng)
    out["sigma_over_R_t_json_arith"] = fl(sigma / R_t) if R_t else None
    pts = {}
    if R_t is not None:
        pts["R_t_json"] = dense_point(R_t, fns, g, sigma, thr, ell)
    if R_t_eng is not None:
        pts["R_t_engine"] = dense_point(R_t_eng, fns, g, sigma, thr, ell)
    if r_in_json is not None:
        pts["r_in_json"] = dense_point(r_in_json, fns, g, sigma, thr, ell)
    R_out_site = (site_summary or {}).get("R_out")
    if R_out_site is not None:
        pts["R_out_site"] = dense_point(R_out_site, fns, g, sigma, thr, ell)
    out["at"] = pts
    try:
        F_ = lambda x: np.asarray(info["N_dense"](x), float).ravel() - np.abs(np.asarray(info["v_dense"](x), float).ravel())
        own = dense_zeros_scan(F_, 0.0, r_collar, ell / float(ncfg["fine"]))
        out["zeros_own_scan"] = [fl(x) for x in own]
        out["max_abs_own_scan_minus_engine"] = (fl(max(abs(a_ - b_) for a_, b_ in zip(own, zeros))) if len(own) == len(zeros) and own else None)
    except Exception as exc:
        out["zeros_own_scan"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    pos = rr > 0
    regions = {}
    if R_t is not None:
        regions["r_le_R_t_json"] = dense_region(P, pos & (rr <= R_t), thr)
    if R_c is not None:
        regions["r_le_R_c_json"] = dense_region(P, pos & (rr <= R_c), thr)
    regions["r_le_rs_nominal"] = dense_region(P, pos & (rr <= cfg["rs"]), thr)
    regions["all_to_collar"] = dense_region(P, pos, thr)
    out["regions"] = regions
    out["core_region_used"] = "0 < r <= R_t (json) on the fine grid" if R_t is not None else "none (no R_t)"
    g0 = P["gate"]
    fail = np.isfinite(g0) & (g0 > thr)
    out["r_fail_max"] = fl(rr[fail].max()) if fail.any() else None
    out["fail_contained_in_R_t"] = (None if (R_t is None or out["r_fail_max"] is None) else bool(out["r_fail_max"] <= R_t))
    if R_t is not None:
        outside_pass = np.isfinite(g0) & (rr > R_t) & (g0 <= thr)
        out["r_first_passing_outside_R_t"] = fl(rr[outside_pass].min()) if outside_pass.any() else None
        m_ext = (rr > R_t) & (rr <= R_t + sigma) & np.isfinite(g0)
        out["exterior_side_within_sigma"] = ({"min": fl(g0[m_ext].min()), "max": fl(g0[m_ext].max()), "n_points": int(m_ext.sum()),
                                              "n_failing": int((g0[m_ext] > thr).sum())} if m_ext.any() else {})
        core = pos & (rr <= R_t)
        out["min_N_dense_core"] = fl(P["N"][core].min()) if core.any() else None
        jc = int(np.argmin(P["N"][core])) if core.any() else None
        out["r_min_N_dense_core"] = fl(rr[core][jc]) if jc is not None else None
    out["n_points_N_nonpositive"] = int((P["N"] <= 0).sum())
    N0 = float(P["N"][0])
    q1 = out["engine"].get("q1_series")
    out["centre"] = {"N_centre": fl(N0), "q1_series": q1,
                     "sigma_K_centre_arith": (fl(sigma * abs(q1) / N0) if (q1 is not None and N0 > 0) else None)}
    Fd = P["N"] - np.abs(P["v"])
    jm = int(np.argmin(Fd))
    out["min_N_minus_absv_fine_grid"] = {"value": fl(Fd[jm]), "r": fl(rr[jm])}

    # the stored node and site arrays against the re-solved solution
    nodes = {}
    try:
        rn_s, Nn_s, vn_s = node_row(z["r_nodes"], idx), node_row(z["N_nodes"], idx), node_row(z["v_nodes"], idx)
        rn_e = np.asarray(info["r_nodes"], dtype=float)
        Nn_e = np.asarray(info["N_nodes"], dtype=float)
        vn_e = np.asarray(info["v_nodes"], dtype=float)
        nodes = {"n_stored": int(rn_s.size), "n_engine": int(rn_e.size)}
        if rn_s.size == rn_e.size:
            nodes.update({"max_abs_dr": fl(np.max(np.abs(rn_s - rn_e))), "max_abs_dN": fl(np.max(np.abs(Nn_s - Nn_e))),
                          "max_abs_dv": fl(np.max(np.abs(vn_s - vn_e))), "min_N_nodes_stored": fl(Nn_s.min()),
                          "stored_node_spacing_median": fl(np.median(np.diff(rn_s))) if rn_s.size > 1 else None})
    except Exception as exc:
        nodes = {"error": "%s: %s" % (type(exc).__name__, exc)}
    out["nodes_stored_vs_engine"] = nodes
    N_site, v_site, Na, va = (np.asarray(x_, dtype=float) for x_ in (N_site, v_site, Na, va))
    sites = {"max_abs_N_site_minus_N_applied": fl(np.max(np.abs(N_site - Na))),
             "max_abs_v_site_minus_v_applied": fl(np.max(np.abs(v_site - va))),
             "max_abs_N_site_minus_N_dense_at_sites": fl(np.max(np.abs(N_site - np.asarray(N_s, float)))),
             "max_abs_v_site_minus_v_dense_at_sites": fl(np.max(np.abs(v_site - np.asarray(v_s, float)))),
             "max_abs_N_applied_minus_N_dense_at_sites": fl(np.max(np.abs(Na - np.asarray(N_s, float)))),
             "max_abs_v_applied_minus_v_dense_at_sites": fl(np.max(np.abs(va - np.asarray(v_s, float))))}
    if ncfg.get("check_applied"):
        try:
            Nk, vk, norms = FLD.site_fields_kernel_average(info, R, ell)
            Nk, vk = np.asarray(Nk, float), np.asarray(vk, float)
            sites["kernel_average"] = {"max_abs_N_applied_minus_kernel_average": fl(np.max(np.abs(Na - Nk))),
                                       "max_abs_v_applied_minus_kernel_average": fl(np.max(np.abs(va - vk))),
                                       "max_abs_N_site_minus_kernel_average": fl(np.max(np.abs(N_site - Nk))),
                                       "max_abs_v_site_minus_kernel_average": fl(np.max(np.abs(v_site - vk))),
                                       "norms_min": fl(np.min(norms)), "norms_max": fl(np.max(norms))}
        except Exception as exc:
            sites["kernel_average"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    out["sites_stored_vs_engine"] = sites
    out["fine_grid"] = {"spacing": fl(ell / float(ncfg["fine"])), "n_points": int(rr.size), "r_max": fl(rr[-1])}
    out["seconds"] = fl(time.time() - t0)

    keep = rr <= float(ncfg["profile_rmax"]) + 1e-9
    prof = {"node_r": rr[keep]}
    for name in ("N", "v", "W", "rho", "dv", "lam_r", "lam_t", "gate"):
        prof["node_" + name] = P[name][keep]
    if keep_profile:
        out["profile"] = {k: [fl(x) for x in prof[k]] for k in ("node_r", "node_N", "node_v", "node_lam_r", "node_lam_t", "node_gate")}
    return out, prof


def node_line(key, npr, ss):
    """1 line per record: the node pass beside the site arrays' values."""
    so, _ = _outer_inner(ss)
    score = ss["regions"].get("r_le_R_out") or ss["regions"].get("r_le_R_c_json") or ss["regions"]["r_le_rs_nominal"]
    beside = "beside, site arrays: root %s a, sigma||K||_app %s, core max %s" % (
        fmt(None if so is None else so["r"], 3), fmt(None if so is None else so["gate_interp"]["centred"], 5), fmt(score.get("value"), 4))
    if npr.get("stopped"):
        return "T/r_s %s node pass: %s; %s; %.1f s" % (key, npr["stopped"], beside, npr.get("seconds") or 0.0)
    a = npr["at"].get("R_t_json") or {}
    core = npr["regions"].get("r_le_R_t_json") or {}
    e = npr.get("engine") or {}
    return ("T/r_s %s node pass: zeros [%s] reproduce the json's [%s] (max |diff| %s a <= %s a); E_collar - E_json %s; at R_t %s: "
            "sigma|v/(rN)| %s, sigma|v'/N| %s (equations; difference rule %s; constraint identity %s), sigma||K||_op %s, margin %s "
            "(%s percent) against %s; dense core max %s at r %s (%s), min N_dense core %s; r_fail_max %s, first pass beyond R_t %s; "
            "%s; %.1f s" % (
                key, ", ".join(fmt(x, 4) for x in npr["zeros_engine"]), ", ".join(fmt(x, 4) for x in npr["zeros_json"]),
                fmte(npr["max_abs_zero_difference"]), fmte(npr["tolerance_a"], 1), fmte(e.get("E_collar_minus_E_json")),
                fmt(npr.get("R_t_json"), 4), fmt(a.get("sigma_tangential"), 5), fmt(a.get("sigma_radial"), 5),
                fmt(a.get("sigma_radial_difference_rule"), 5), fmt(a.get("sigma_radial_constraint_identity_arith"), 5),
                fmt(a.get("gate"), 5), fmt(a.get("margin_abs_arith"), 5), fmt(a.get("margin_percent_arith"), 1),
                fmt(0.1 if a.get("gate") is None else a.get("gate") - (a.get("margin_abs_arith") or 0.0), 3),
                fmt(core.get("value"), 4), fmt(core.get("r"), 3), core.get("part") or "-", fmt(npr.get("min_N_dense_core"), 4),
                fmt(npr.get("r_fail_max"), 3), fmt(npr.get("r_first_passing_outside_R_t"), 3), beside, npr.get("seconds") or 0.0))


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
        ss, cmpj = rec[SITE], rec.get("comparison_with_json", {})
        o, _ = _outer_inner(ss)
        if o is None:
            L.append("- T/r_s %s: no outer root of N - |v| on the site arrays." % key)
            continue
        sr, gi = o["sigma_over_R_arith"], o["gate_interp"]["centred"]
        tan = None if o["tangential_interp"] is None else sigma * abs(o["tangential_interp"])
        rad = None if o["radial_interp"]["centred"] is None else sigma * abs(o["radial_interp"]["centred"])
        core = ss["regions"].get("r_le_R_out") or {}
        ao, _ = _outer_inner(rec[APPLIED])
        L.append(("- T/r_s %s (site arrays, version 1's object = the applied arrays to every printed digit): outer root R_out = %s a "
                  "(json dense R_t %s, difference %s); sigma/R_out = %s (arith) against %s (%s): difference %s; sigma||K||_op at the "
                  "root (site, centred) = %s (tangential part %s, radial part %s), %s the threshold %s; core maximum (%s) %s at r = %s "
                  "(%s part); outermost failing site %s a (failure contained inside R_out: %s); applied arrays: outer root %s a, "
                  "sigma/R = %s (arith), sigma||K||_app at it %s.") % (
                 key, fmt(o["r"], 4), fmt(cmpj.get("R_t_json"), 4), fmt(cmpj.get("R_out_site_minus_R_t_json"), 7),
                 fmt(sr, 5), fmt(ref["value"], 3), ref["source"], fmt(None if sr is None else sr - ref["value"], 5),
                 fmt(gi, 5), fmt(tan, 5), fmt(rad, 5), "exceeds" if (gi is not None and gi > thr) else "does not exceed",
                 fmt(thr, 3), ss["core_region_used"], fmt(core.get("value"), 4), fmt(core.get("r"), 3),
                 core.get("part") or "-", fmt(ss.get("r_fail_max"), 3), fmt(ss.get("fail_contained_in_R_out")),
                 fmt(None if ao is None else ao["r"], 4), fmt(None if ao is None else ao["sigma_over_R_arith"], 5),
                 fmt(None if ao is None else ao["gate_interp"]["centred"], 5)))
    keys = [k for k in recs if LATE_ROOT["T_lo"] - 1e-9 <= float(k) <= LATE_ROOT["T_hi"] + 1e-9]
    if keys:
        rows = []
        for k in sorted(keys, key=float):
            o, _ = _outer_inner(recs[k][SITE])
            if o is not None:
                rows.append((o["r"], o["sigma_over_R_arith"], o["gate_interp"]["centred"]))
        if rows:
            Rv, sv = [r[0] for r in rows], [r[1] for r in rows]
            gv = [r[2] for r in rows if r[2] is not None]
            nfail = sum(1 for r in rows if r[2] is not None and r[2] > thr)
            L.append(("- T/r_s %s to %s (site arrays; %d records with an outer root of %d in the run): R_out from %s to %s a against "
                      "the dense band %s to %s a (%s); sigma/R_out from %s to %s (arith) against %s to %s; sigma||K||_op at the "
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


def node_acceptance_lines(res):
    cfg, recs = res["config"], res["records"]
    sigma, thr = cfg["sigma"], cfg["gate_threshold"]
    L = []
    if not res.get("node_pass_config"):
        return L
    for key in sorted(READ_AGAINST):
        ref = READ_AGAINST[key]
        rec = recs.get(key)
        if rec is None or "node_pass" not in rec:
            L.append("- T/r_s %s (node pass): not run." % key)
            continue
        npr = rec["node_pass"]
        if npr.get("stopped"):
            L.append("- T/r_s %s (node pass): %s" % (key, npr["stopped"]))
            continue
        a = npr["at"].get("R_t_json") or {}
        core = npr["regions"].get("r_le_R_t_json") or {}
        ss = rec[SITE]
        so, _ = _outer_inner(ss)
        score = ss["regions"].get("r_le_R_out") or {}
        sr = npr.get("sigma_over_R_t_json_arith")
        L.append(("- T/r_s %s (node pass, the dense solution re-solved from the stored sources): the record's zeros reproduced (max "
                  "|difference| %s a, tolerance %s a; E_collar - E_json %s); at R_t = %s a (json) sigma/R_t = %s (arith) against %s "
                  "(%s): difference %s; on the dense solution sigma||K||_op at R_t = %s (tangential part %s, radial part %s by the "
                  "equations, %s by the difference rule on the dense output, %s by the constraint identity with rho_ell(R_t) = %s "
                  "against the threshold %s (arith) for the radial part to exceed the tangential), %s the threshold %s by %s "
                  "(%s percent); dense core maximum (0 < r <= R_t) %s at r = %s (%s part), min N_dense there %s; outermost failing "
                  "radius %s a, first passing radius beyond R_t %s a; beside, the site arrays at the matter's root %s a: %s, their "
                  "core maximum %s.") % (
                 key, fmte(npr.get("max_abs_zero_difference")), fmte(npr.get("tolerance_a"), 1),
                 fmte((npr.get("engine") or {}).get("E_collar_minus_E_json")), fmt(npr.get("R_t_json"), 4), fmt(sr, 5),
                 fmt(ref["value"], 3), ref["source"], fmt(None if sr is None else sr - ref["value"], 5),
                 fmt(a.get("gate"), 5), fmt(a.get("sigma_tangential"), 5), fmt(a.get("sigma_radial"), 5),
                 fmt(a.get("sigma_radial_difference_rule"), 5), fmt(a.get("sigma_radial_constraint_identity_arith"), 5),
                 fmte(a.get("rho_ell"), 3), fmte(a.get("rho_threshold_radial_exceeds_tangential_arith"), 3),
                 "exceeds" if a.get("fails_gate") else "does not exceed", fmt(thr, 3), fmt(a.get("margin_abs_arith"), 5),
                 fmt(a.get("margin_percent_arith"), 1), fmt(core.get("value"), 4), fmt(core.get("r"), 3), core.get("part") or "-",
                 fmt(npr.get("min_N_dense_core"), 4), fmt(npr.get("r_fail_max"), 3), fmt(npr.get("r_first_passing_outside_R_t"), 3),
                 fmt(None if so is None else so["r"], 4), fmt(None if so is None else so["gate_interp"]["centred"], 5),
                 fmt(score.get("value"), 4)))
    keys = [k for k in recs if LATE_ROOT["T_lo"] - 1e-9 <= float(k) <= LATE_ROOT["T_hi"] + 1e-9 and "node_pass" in recs[k]]
    if keys:
        good, bad = [], []
        for k in sorted(keys, key=float):
            npr = recs[k]["node_pass"]
            if npr.get("stopped"):
                bad.append(k)
                continue
            a = npr["at"].get("R_t_json") or {}
            core = npr["regions"].get("r_le_R_t_json") or {}
            good.append((npr.get("R_t_json"), a.get("gate"), a.get("sigma_tangential"), a.get("sigma_radial"), a.get("margin_percent_arith"),
                         core.get("value"), npr.get("max_abs_zero_difference")))
        if good:
            Rt = [x[0] for x in good if x[0] is not None]
            gv = [x[1] for x in good if x[1] is not None]
            tv = [x[2] for x in good if x[2] is not None]
            rv = [x[3] for x in good if x[3] is not None]
            mv = [x[4] for x in good if x[4] is not None]
            cv = [x[5] for x in good if x[5] is not None]
            dz = [x[6] for x in good if x[6] is not None]
            nfail = sum(1 for x in good if x[1] is not None and x[1] > thr)
            L.append(("- T/r_s %s to %s (node pass; %d of %d records reproduced, max |zero difference| %s a): R_t (json) from %s to %s a "
                      "against the band %s to %s a (%s); on the dense solution sigma||K||_op at R_t from %s to %s (tangential part %s to "
                      "%s, radial part %s to %s); %d of %d exceed %s, margin %s to %s percent; dense core maximum from %s to %s.%s") % (
                     fmt(LATE_ROOT["T_lo"], 3), fmt(LATE_ROOT["T_hi"], 3), len(good), len(keys), fmte(max(dz) if dz else None),
                     fmt(min(Rt) if Rt else None, 3), fmt(max(Rt) if Rt else None, 3), fmt(LATE_ROOT["R_lo"], 1), fmt(LATE_ROOT["R_hi"], 1),
                     LATE_ROOT["source"], fmt(min(gv) if gv else None, 5), fmt(max(gv) if gv else None, 5),
                     fmt(min(tv) if tv else None, 5), fmt(max(tv) if tv else None, 5), fmt(min(rv) if rv else None, 5),
                     fmt(max(rv) if rv else None, 5), nfail, len(good), fmt(thr, 3), fmt(min(mv) if mv else None, 1),
                     fmt(max(mv) if mv else None, 1), fmt(min(cv) if cv else None, 4), fmt(max(cv) if cv else None, 4),
                     (" Not reproduced: %s." % ", ".join(bad)) if bad else ""))
        else:
            L.append("- T/r_s %s to %s (node pass): no record reproduced; not reproduced: %s." % (
                fmt(LATE_ROOT["T_lo"], 3), fmt(LATE_ROOT["T_hi"], 3), ", ".join(bad)))
    return L


def node_cfg_text(res):
    n = res["node_pass_config"]
    e = res["provenance"].get("engine") or {}
    files = "; ".join("%s %s (sha256 %s)" % (k, v["path"], v["sha256"][:16] + "...") for k, v in sorted((e.get("files") or {}).items()))
    return ("engine path %s (module %s; %s; numpy %s, scipy %s); g = %s (%s); ell = %s (%s); rtol %s, atol %s; collar policy %s (%s); "
            "support_rel %s (%s); zero tolerance %s a (every zero's |engine - json| and the counts); fine grid spacing ell/%d = %s a from the "
            "centre to the collar; dense profiles stored to r <= %s a; kernel-average check %s. The engine is read only: %s." % (
                e.get("path"), e.get("module"), files or "no files recorded", e.get("numpy_version"), e.get("scipy_version"),
                fmt(n["g"], 6), n["g_source"], fmt(n["ell"], 3), n["ell_source"], fmte(n["rtol"], 0), fmte(n["atol"], 0),
                n["collar_policy"], n["collar_policy_source"], fmte(n["support_rel"], 0), n["support_rel_source"], fmte(n["zero_tol"], 1),
                n["fine"], fmt(n["ell"] / float(n["fine"]), 5), fmt(n["profile_rmax"], 1), "on" if n["check_applied"] else "off",
                e.get("read_only")))


def write_markdown(path, res, listed_keys):
    cfg, prov, recs = res["config"], res["provenance"], res["records"]
    sigma, thr = cfg["sigma"], cfg["gate_threshold"]
    L = ["# tensor_gate_v2: sigma ||K||_op on the stored site and applied arrays (version 1) and on the re-solved dense solution "
         "(the node pass)", ""]
    L.append("Object. " + OBJECT_STATEMENT)
    L.append("")
    L.append("Inputs. npz %s (%d bytes%s); records json: %s. Every table number below is computed here from the npz "
             "unless its column says json (the records json) or arith (arithmetic on a number in the same row); the node pass columns "
             "are computed here from the npz's sources through the engine's solver. max|N - N_applied| over all records %s, "
             "max|v - v_applied| %s%s." % (
                 prov["npz_path"], prov["npz_size_bytes"],
                 ("; sha256 " + prov["npz_sha256"]) if prov.get("npz_sha256") else "",
                 ", ".join(prov["records_json_paths"]) or "none",
                 fmte(prov.get("max_abs_N_minus_N_applied")), fmte(prov.get("max_abs_v_minus_v_applied")),
                 (" (both 0: the site arrays equal the applied arrays; version 1's headline is not on the dense fields)"
                  if prov.get("site_arrays_equal_applied") else "")))
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
    if res.get("node_pass_config"):
        L.append("Node pass. " + NODE_PASS_STATEMENT)
        L.append("")
        L.append("Node pass configuration. " + node_cfg_text(res))
        L.append("")
    L.append("## Acceptance read (values from the json of this run; the reference numbers are inputs, sources named)")
    L.extend(acceptance_lines(res))
    L.extend(node_acceptance_lines(res))
    L.append("")
    keys = [k for k in listed_keys if k in recs]
    L.append("## Table A: the gate at the roots and in the core (site arrays N, v; applied beside; version 1's object)")
    hdr = ["T/r_s", "site roots of N-|v| [a]", "json trapping_zeros (dense) [a]", "sigma/R_out (arith)",
           "at R_out: sigma|v/(rN)| / sigma|v'/N| / sigma||K||", "at R_in: sigma||K||",
           "max in core (@r, part)", "max r<=R_c (json R_c)", "r_fail_max [a] (inside R_out?)",
           "applied: roots [a]; sigma||K||_app at its R_out; max r<=R_out(site)"]
    rows = []
    for k in keys:
        rec = recs[k]
        ss, aps, cmpj = rec[SITE], rec[APPLIED], rec.get("comparison_with_json", {})
        o, i = _outer_inner(ss)
        ao, _ = _outer_inner(aps)
        core = ss["regions"].get("r_le_R_out") or ss["regions"].get("r_le_R_c_json") or ss["regions"]["r_le_rs_nominal"]
        rc = ss["regions"].get("r_le_R_c_json")
        jz = cmpj.get("json_trapping_zeros")
        rows.append([
            k, roots_text(ss["roots"]), ("; ".join(fmt(z, 3) for z in jz) if jz else "-"),
            fmt(None if o is None else o["sigma_over_R_arith"], 5),
            "-" if o is None else "%s / %s / %s" % (
                fmt(None if o["tangential_interp"] is None else sigma * abs(o["tangential_interp"]), 5),
                fmt(None if o["radial_interp"]["centred"] is None else sigma * abs(o["radial_interp"]["centred"]), 5),
                fmt(o["gate_interp"]["centred"], 5)),
            fmt(None if i is None else i["gate_interp"]["centred"], 5),
            "%s (@%s, %s)" % (fmt(core.get("value"), 4), fmt(core.get("r"), 2), core.get("part") or "-"),
            "-" if rc is None else "%s (@%s; R_c %s)" % (fmt(rc.get("value"), 4), fmt(rc.get("r"), 2),
                                                        fmt(cmpj.get("R_c_json"), 2)),
            "%s (%s)" % (fmt(ss.get("r_fail_max"), 2), fmt(ss.get("fail_contained_in_R_out"))),
            "%s; %s; %s" % (roots_text(aps["roots"]), fmt(None if ao is None else ao["gate_interp"]["centred"], 5),
                            fmt((aps["regions"].get("r_le_R_out_site") or {}).get("value"), 4)),
        ])
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("## Table B: the derivative rule's sensitivity and the field-reading cross-checks (site arrays)")
    hdr = ["T/r_s", "sigma||K|| at R_out: centred / forward / backward / sigma-stride",
           "max in core: centred / forward / backward / sigma-stride",
           "max_core sigma|dv'|/N: forward / backward / sigma-stride (vs centred)",
           "scalar sigma|v'| max core: a-rule / sigma-rule", "min N (core)", "n(N<=0)",
           "min(N-|v|) @ r: site / json", "(R_out, R_out+sigma]: min / max sigma||K|| (n failing / n)",
           "max|N-N_app| / max|v-v_app| (this record)"]
    rows = []
    for k in keys:
        rec = recs[k]
        ss, cmpj = rec[SITE], rec.get("comparison_with_json", {})
        o, _ = _outer_inner(ss)
        se, sc, ex, mf = ss["sensitivity"], ss["scalar_sigma_abs_dv"], ss["exterior_side_within_sigma"], ss["min_N_minus_absv"]
        ar = rec.get("site_vs_applied_arrays") or {}
        rows.append([
            k,
            "-" if o is None else " / ".join(fmt(o["gate_interp"][r], 5) for r in RULE_NAMES),
            " / ".join(fmt(se[r]["core_max_gate"], 4) for r in RULE_NAMES),
            " / ".join(fmt(se[r]["max_core_sigma_abs_dlam_r"], 5) for r in ("forward", "backward", "stride_sigma")),
            "%s / %s" % (fmt(sc["sigma_abs_dv_centred_core_max"], 4), fmt(sc["sigma_abs_dv_stride_sigma_core_max"], 4)),
            fmt(ss["min_N_core"], 4), ss["n_sites_N_nonpositive"],
            "%s @ %s / %s @ %s" % (fmt(mf["value"], 5), fmt(mf["r"], 3), fmt(cmpj.get("min_N_minus_absv_json"), 5),
                                   fmt(cmpj.get("r_min_N_minus_absv_json"), 3)),
            "-" if not ex else "%s / %s (%d / %d)" % (fmt(ex["min"], 4), fmt(ex["max"], 4), ex["n_failing"], ex["n_sites"]),
            "%s / %s" % (fmte(ar.get("max_abs_N_minus_N_applied")), fmte(ar.get("max_abs_v_minus_v_applied"))),
        ])
    L.append(md_table(hdr, rows))
    L.append("")
    nkeys = [k for k in keys if "node_pass" in recs[k]]
    if nkeys:
        L.append("## Table C: the node pass: the dense solution re-solved from the stored sources, sigma ||K||_op at R_t (the "
                 "record's dense root) and over the core, beside the site arrays' value at the matter's root")
        hdr = ["T/r_s", "reproduced (max|dz| [a]; tol)", "engine zeros [a] (json beside)", "E_collar - E_json",
               "R_t (json) [a]; sigma/R_t (arith)", "at R_t: sigma|v/(rN)| / sigma|v'/N| (equations) / sigma||K||",
               "radial check: difference rule / constraint identity", "margin vs threshold: abs / percent",
               "dense core max (@r, part); min N_dense core", "dense r_fail_max / first pass beyond R_t [a]",
               "site arrays: R_out [a]; sigma||K||_app at it; core max", "rho_ell(R_t) / threshold 3/(8 pi g R_t^2) (arith)"]
        rows = []
        for k in nkeys:
            rec = recs[k]
            npr, ss = rec["node_pass"], rec[SITE]
            so, _ = _outer_inner(ss)
            score = ss["regions"].get("r_le_R_out") or {}
            beside = "%s; %s; %s" % (fmt(None if so is None else so["r"], 3), fmt(None if so is None else so["gate_interp"]["centred"], 5),
                                     fmt(score.get("value"), 4))
            if npr.get("stopped"):
                rows.append([k, "no" if npr.get("reproduced") is False else "stopped", npr["stopped"], "-", "-", "-", "-", "-", "-", "-", beside, "-"])
                continue
            a = npr["at"].get("R_t_json") or {}
            core = npr["regions"].get("r_le_R_t_json") or {}
            e = npr.get("engine") or {}
            rows.append([
                k, "yes (%s; %s)" % (fmte(npr.get("max_abs_zero_difference")), fmte(npr.get("tolerance_a"), 0)),
                "%s (%s)" % ("; ".join(fmt(x, 3) for x in npr["zeros_engine"]), "; ".join(fmt(x, 3) for x in npr["zeros_json"])),
                fmte(e.get("E_collar_minus_E_json")),
                "%s; %s" % (fmt(npr.get("R_t_json"), 4), fmt(npr.get("sigma_over_R_t_json_arith"), 5)),
                "%s / %s / %s" % (fmt(a.get("sigma_tangential"), 5), fmt(a.get("sigma_radial"), 5), fmt(a.get("gate"), 5)),
                "%s / %s" % (fmt(a.get("sigma_radial_difference_rule"), 5), fmt(a.get("sigma_radial_constraint_identity_arith"), 5)),
                "%s / %s" % (fmt(a.get("margin_abs_arith"), 5), fmt(a.get("margin_percent_arith"), 1)),
                "%s (@%s, %s); %s" % (fmt(core.get("value"), 4), fmt(core.get("r"), 3), core.get("part") or "-", fmt(npr.get("min_N_dense_core"), 4)),
                "%s / %s" % (fmt(npr.get("r_fail_max"), 3), fmt(npr.get("r_first_passing_outside_R_t"), 3)),
                beside,
                "%s / %s" % (fmte(a.get("rho_ell"), 3), fmte(a.get("rho_threshold_radial_exceeds_tangential_arith"), 3)),
            ])
        L.append(md_table(hdr, rows))
        L.append("")
        L.append("## Table D: the node pass's reproduction diagnostics (the stored record against the re-solved solution)")
        hdr = ["T/r_s", "zeros engine - json [a]", "min(N-|v|): engine - json; r_min diff [a]",
               "nodes: n stored / engine; max|dr| / max|dN| / max|dv|",
               "sites: max|N_site - N_dense@sites| / max|v_site - v_dense@sites|; max|N_site - N_app| / max|v_site - v_app|",
               "kernel average: max|N_app - avg| / max|v_app - avg| (norms)", "gauge; regime", "residual_W / residual_invN; spline error engine / json",
               "r_collar; collar deficit [ell]; own scan - engine zeros; s"]
        rows = []
        for k in nkeys:
            npr = recs[k]["node_pass"]
            e = npr.get("engine") or {}
            nd = npr.get("nodes_stored_vs_engine") or {}
            st = npr.get("sites_stored_vs_engine") or {}
            ka = st.get("kernel_average") or {}
            cm = e.get("collar_margin") or {}
            rows.append([
                k, "; ".join(fmte(d_) for d_ in (npr.get("zero_differences_engine_minus_json") or [])) or (npr.get("stopped") or "-"),
                "%s; %s" % (fmte(npr.get("min_engine_minus_json")), fmte(npr.get("r_min_engine_minus_json"))),
                ("%s / %s; %s / %s / %s" % (nd.get("n_stored", "-"), nd.get("n_engine", "-"), fmte(nd.get("max_abs_dr")), fmte(nd.get("max_abs_dN")),
                                            fmte(nd.get("max_abs_dv")))) if nd and "error" not in nd else (nd.get("error") or "-"),
                "%s / %s; %s / %s" % (fmte(st.get("max_abs_N_site_minus_N_dense_at_sites")), fmte(st.get("max_abs_v_site_minus_v_dense_at_sites")),
                                      fmte(st.get("max_abs_N_site_minus_N_applied")), fmte(st.get("max_abs_v_site_minus_v_applied"))),
                ("%s / %s (%s to %s)" % (fmte(ka.get("max_abs_N_applied_minus_kernel_average")), fmte(ka.get("max_abs_v_applied_minus_kernel_average")),
                                         fmt(ka.get("norms_min"), 6), fmt(ka.get("norms_max"), 6))) if ka and "error" not in ka else (ka.get("error") if ka else "off"),
                "%s; %s" % (e.get("gauge_label") or "-", e.get("regime_label") or "-"),
                "%s / %s; %s / %s" % (fmte(e.get("residual_W")), fmte(e.get("residual_invN")), fmte(e.get("source_spline_error")),
                                      fmte(e.get("source_spline_error_json"))),
                "%s; %s; %s; %s" % (fmt(e.get("r_collar"), 1), cm.get("deficit_ell", "-"), fmte(npr.get("max_abs_own_scan_minus_engine")),
                                    fmt(npr.get("seconds"), 1)),
            ])
        L.append(md_table(hdr, rows))
        L.append("")
    L.append("Columns. 'core' in Tables A and B is %s per record (its name in the json: core_region_used); in Table C it is 0 < r <= R_t "
             "(json) on the engine's fine grid. The tangential part at a root equals sigma/R up to the root's residual (Tables A, B: the "
             "linear-interpolation residual, json tangential_interp_residual; Table C: tangential_plus_1_over_r). The applied columns "
             "coincide with the site columns to every printed digit on the version 1 run; neither is the gate's object on the dense "
             "fields, which the node pass holds (records[key].node_pass in the json)." % (
                 "0 < r <= R_out where an outer root of the site arrays exists, else 0 < r <= R_c (json), else 0 < r <= r_s"))
    L.append("")
    L.append("Files. json %s; profiles npz %s%s." % (res["files"]["json"], res["files"]["profiles_npz"],
                                                    ("; all-records table " + res["files"]["all_md"]) if res["files"].get("all_md") else ""))
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_all_markdown(path, res):
    recs = res["records"]
    thr = res["config"]["gate_threshold"]
    L = ["# tensor_gate_v2: every record of the history (site arrays; 1 line per record; the node pass where run)", "",
         "Columns as in Table A of the main file; sigma ||K||_op at the outer root with the centred rule; "
         "threshold %s." % fmt(thr, 3), ""]
    hdr = ["T/r_s", "site roots [a]", "sigma/R_out (arith)", "sigma||K|| at R_out", "max in core (@r)",
           "r_fail_max [a]", "inside R_out?", "applied R_out [a]", "sigma||K||_app at applied R_out",
           "node pass: reproduced (max|dz|)", "dense sigma||K|| at R_t (json)", "margin percent"]
    rows = []
    for k in sorted(recs, key=float):
        rec = recs[k]
        ss, aps = rec[SITE], rec[APPLIED]
        o, _ = _outer_inner(ss)
        ao, _ = _outer_inner(aps)
        core = ss["regions"].get("r_le_R_out") or ss["regions"].get("r_le_R_c_json") or ss["regions"]["r_le_rs_nominal"]
        npr = rec.get("node_pass")
        if npr is None:
            ncols = ["-", "-", "-"]
        elif npr.get("stopped"):
            ncols = ["no" if npr.get("reproduced") is False else "stopped", "-", "-"]
        else:
            a = npr["at"].get("R_t_json") or {}
            ncols = ["yes (%s)" % fmte(npr.get("max_abs_zero_difference")), fmt(a.get("gate"), 5), fmt(a.get("margin_percent_arith"), 1)]
        rows.append([k, roots_text(ss["roots"]), fmt(None if o is None else o["sigma_over_R_arith"], 5),
                     fmt(None if o is None else o["gate_interp"]["centred"], 5),
                     "%s (@%s)" % (fmt(core.get("value"), 4), fmt(core.get("r"), 2)),
                     fmt(ss.get("r_fail_max"), 2), fmt(ss.get("fail_contained_in_R_out")),
                     fmt(None if ao is None else ao["r"], 3),
                     fmt(None if ao is None else ao["gate_interp"]["centred"], 5)] + ncols)
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

def load_npz(path, allow_pickle=False):
    """allow_pickle only when the node pass needs the object node arrays (FIELDS_FORMAT); the lattice arrays never need it."""
    z = np.load(path, allow_pickle=allow_pickle)
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
    ap = argparse.ArgumentParser(description="sigma ||K||_op on the stored site and applied arrays (version 1) and, with --node-pass, "
                                             "on the dense solution re-solved from the stored sources with the engine (tensor_gate_v2).")
    ap.add_argument("npz", help="the fields npz (FIELDS_FORMAT.md)")
    ap.add_argument("--records-json", default=None,
                    help="a records json (the full json beside the npz is read when present; this one is read as well)")
    ap.add_argument("--out", default="tensor_gate_v2_out/tensor_gate_v2", help="output prefix")
    ap.add_argument("--clocks", default=DEFAULT_CLOCKS,
                    help="T/r_s list; ranges lo:hi[:step] inclusive (default %s)" % DEFAULT_CLOCKS)
    ap.add_argument("--all", action="store_true", help="also every record of the history (version 1's summary only)")
    ap.add_argument("--sigma", type=float, default=None, help="override sigma (default ell*a from the npz)")
    ap.add_argument("--rs", type=float, default=None, help="override r_s/a (default: json args rs, else 64)")
    ap.add_argument("--gate", type=float, default=0.1, help="the threshold (default 0.1)")
    ap.add_argument("--force", action="store_true", help="recompute cached records")
    ap.add_argument("--no-json-profiles", action="store_true", help="do not embed the 3 headline profiles in the json")
    ap.add_argument("--hash", action="store_true", help="record the npz's sha256 (reads the whole file once)")
    ap.add_argument("--node-pass", action="store_true",
                    help="re-solve the listed records' dense solution from eps_tot, J_tot with the engine and evaluate the gate on it at R_t")
    ap.add_argument("--engine-path", default=None, help="the solve lane's code directory (engine_fields_v14 and its chain), imported read-only")
    ap.add_argument("--zero-tol", type=float, default=1e-3,
                    help="the tolerance in a on every trapping zero, engine against json, for the solution to count as the record's (default 1e-3)")
    ap.add_argument("--fine", type=int, default=64, help="the fine grid spacing ell/fine (default 64, the engine's trapping_radius default)")
    ap.add_argument("--rtol", type=float, default=1e-12, help="the engine's solve_ivp rtol (default 1e-12, the run's)")
    ap.add_argument("--atol", type=float, default=1e-16, help="the engine's solve_ivp atol (default 1e-16, the run's)")
    ap.add_argument("--g", type=float, default=None, help="override g (default: npz key g, else json args g)")
    ap.add_argument("--collar-policy", default=None, help="override the collar policy (default: json args collar_policy, else flag)")
    ap.add_argument("--support-rel", type=float, default=None, help="override support_rel (default: json args support_rel, else 1e-14)")
    ap.add_argument("--check-applied", action="store_true",
                    help="also run the engine's site_fields_kernel_average on the re-solved solution and compare with the stored applied arrays")
    ap.add_argument("--profile-rmax", type=float, default=96.0,
                    help="store the dense profiles to this radius in a (default 96 = 1.5 r_s nominal, arith; the summaries use the whole grid)")
    args = ap.parse_args(argv)
    t_start = time.time()

    if args.node_pass and not args.engine_path:
        sys.exit("--node-pass needs --engine-path DIR (the engine is imported from the argument, never from the environment)")

    z = load_npz(args.npz, allow_pickle=bool(args.node_pass))
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
    dN_all = float(np.max(np.abs(N - Na)))
    dv_all = float(np.max(np.abs(v - va)))
    prov = {"version": VERSION, "npz_path": os.path.abspath(args.npz), "npz_size_bytes": int(os.path.getsize(args.npz)),
            "npz_keys_used": ["R", "T", "N", "v", "N_applied", "v_applied", "ell"] + (
                ["g", "eps_tot", "J_tot", "r_nodes", "N_nodes", "v_nodes"] if args.node_pass else []),
            "records_json_paths": [os.path.abspath(p) for p in jused],
            "solver": (str(z["solver"]) if "solver" in z.files else None),
            "label": (str(z["label"]) if "label" in z.files else None),
            "max_abs_N_minus_N_applied": fl(dN_all), "max_abs_v_minus_v_applied": fl(dv_all),
            "site_arrays_equal_applied": bool(dN_all == 0.0 and dv_all == 0.0),
            "fields_read": "site arrays N, v (= N_applied, v_applied where the maxima above are 0); the dense fields by the node pass"}
    if args.hash:
        prov["npz_sha256"] = sha256_of(args.npz)
    print("inputs: %d records, %d sites; max|N - N_applied| = %.3e, max|v - v_applied| = %.3e%s" % (
        T.size, R.size, dN_all, dv_all,
        " WARNING: the site arrays equal the applied arrays: version 1's headline is not on the dense fields (the node pass is)"
        if prov["site_arrays_equal_applied"] else ""))

    ncfg, FLD = None, None
    if args.node_pass:
        FLD, eng_prov = import_engine(args.engine_path)
        prov["engine"] = eng_prov
        g_npz = read_scalar(z, "g")
        if args.g is not None:
            g_val, g_src = float(args.g), "--g"
        elif g_npz is not None:
            g_val, g_src = g_npz, "npz key g"
            if "g" in jargs and abs(float(jargs["g"]) - g_npz) > 1e-12:
                print("warning: npz g %r differs from the json args g %r; the npz value is used" % (g_npz, jargs["g"]))
        elif "g" in jargs:
            g_val, g_src = float(jargs["g"]), "records json args g"
        else:
            sys.exit("g is neither in the npz nor in the json args; pass --g")
        if ell is not None:
            ell_val, ell_src = ell, "npz key ell"
        elif "ell" in jargs:
            ell_val, ell_src = float(jargs["ell"]), "records json args ell"
        else:
            sys.exit("ell is neither in the npz nor in the json args")
        ell_val = int(round(ell_val)) if abs(ell_val - round(ell_val)) < 1e-12 else ell_val
        if args.collar_policy is not None:
            cp, cp_src = args.collar_policy, "--collar-policy"
        elif "collar_policy" in jargs:
            cp, cp_src = str(jargs["collar_policy"]), "records json args collar_policy"
        else:
            cp, cp_src = "flag", "default (the run's label: collar policy flag)"
        if args.support_rel is not None:
            sr_, sr_src = float(args.support_rel), "--support-rel"
        elif "support_rel" in jargs:
            sr_, sr_src = float(jargs["support_rel"]), "records json args support_rel"
        else:
            sr_, sr_src = 1e-14, "the engine's default"
        ncfg = {"g": g_val, "g_source": g_src, "ell": ell_val, "ell_source": ell_src, "rtol": float(args.rtol), "atol": float(args.atol),
                "collar_policy": cp, "collar_policy_source": cp_src, "support_rel": sr_, "support_rel_source": sr_src,
                "zero_tol": float(args.zero_tol), "fine": int(args.fine), "check_applied": bool(args.check_applied),
                "profile_rmax": float(args.profile_rmax),
                "sigma_equals_ell_a": bool(abs(sigma - ell_val * a) < 1e-12)}
        print("node pass: " + node_cfg_text({"node_pass_config": ncfg, "provenance": prov}))

    out_prefix = args.out
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path, md_path = out_prefix + ".json", out_prefix + ".md"
    prof_path, all_md_path = out_prefix + "_profiles.npz", out_prefix + "_all.md"

    res = {"version": VERSION, "provenance": prov, "config": cfg, "node_pass_config": ncfg,
           "statements": {"object": OBJECT_STATEMENT, "fields": FIELDS_STATEMENT, "derivative": DERIVATIVE_STATEMENT,
                          "node_pass": NODE_PASS_STATEMENT},
           "read_against": {"records": READ_AGAINST, "late_root": LATE_ROOT},
           "files": {"json": os.path.abspath(json_path), "markdown": os.path.abspath(md_path),
                     "profiles_npz": os.path.abspath(prof_path)},
           "records": {}}
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path) as fh:
                old = json.load(fh)
            if old.get("version") != VERSION:
                sys.exit("existing %s is from %s, not %s; use --force or another --out" % (json_path, old.get("version"), VERSION))
            oc = old.get("config", {})
            same = all(abs(float(oc.get(k, np.nan)) - float(cfg[k])) <= 1e-12 * max(1.0, abs(float(cfg[k])))
                       for k in ("a", "sigma", "rs", "gate_threshold"))
            if not same:
                sys.exit("existing %s was computed with another configuration; use --force or another --out" % json_path)
            onc = old.get("node_pass_config")
            if ncfg is not None and onc:
                same_n = all(onc.get(k) == ncfg.get(k) for k in ("collar_policy", "fine")) and all(
                    abs(float(onc.get(k, np.nan)) - float(ncfg[k])) <= 1e-15 * max(1.0, abs(float(ncfg[k])))
                    for k in ("g", "ell", "rtol", "atol", "support_rel", "zero_tol"))
                if not same_n:
                    sys.exit("existing %s has a node pass with another configuration; use --force or another --out" % json_path)
            elif ncfg is None and onc:
                res["node_pass_config"] = onc
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
    n_done, n_node, n_not_reproduced = 0, 0, 0
    for count, (idx, keep) in enumerate(todo):
        key = "%.3f" % tors[idx]
        jrec = jrecs.get(key)
        headline = keep and key in ("12.125", "30.000", "40.000") and not args.no_json_profiles
        rec = res["records"].get(key)
        if rec is None or args.force:
            rec, prof = analyse_record(idx, tors[idx], T[idx], R, N[idx], v[idx], Na[idx], va[idx], cfg, jrec, keep_profile=headline)
            res["records"][key] = rec
            if keep:
                new_profiles[key] = prof
            n_done += 1
            ss = rec[SITE]
            o, _ = _outer_inner(ss)
            core = ss["regions"].get("r_le_R_out") or ss["regions"].get("r_le_R_c_json") or ss["regions"]["r_le_rs_nominal"]
            print("T/r_s %s: site roots %s; sigma/R_out %s; sigma||K|| at R_out %s; core max %s at r %s; r_fail_max %s; "
                  "applied R_out %s; json R_t %s" % (
                      key, roots_text(ss["roots"]), fmt(None if o is None else o["sigma_over_R_arith"], 5),
                      fmt(None if o is None else o["gate_interp"]["centred"], 5),
                      fmt(core.get("value"), 4), fmt(core.get("r"), 2), fmt(ss.get("r_fail_max"), 2),
                      fmt(rec[APPLIED]["R_out"], 3), fmt((rec.get("comparison_with_json") or {}).get("R_t_json"), 4)))
            if keep or (count % 50 == 49):
                write_json(json_path, res)
        if args.node_pass and keep and (args.force or "node_pass" not in rec):
            npr, nprof = node_pass_record(FLD, z, idx, key, R, N[idx], v[idx], Na[idx], va[idx], cfg, ncfg, jrec, rec[SITE],
                                          keep_profile=headline)
            rec["node_pass"] = jsonable(npr)
            if nprof:
                new_profiles.setdefault(key, {}).update(nprof)
            n_node += 1
            if npr.get("stopped"):
                n_not_reproduced += 1
            print(node_line(key, rec["node_pass"], rec[SITE]))
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
    for line in acceptance_lines(res) + node_acceptance_lines(res):
        print(line)
    print("")
    n_stopped_total = sum(1 for k in listed_keys if k in res["records"] and (res["records"][k].get("node_pass") or {}).get("stopped"))
    print("records computed %d (cached before %d); node passes run %d (%d not reproduced or stopped in this run; %d among the listed "
          "records in the json); wrote %s, %s%s; %.1f s" % (
              n_done, len(res["records"]) - n_done, n_node, n_not_reproduced, n_stopped_total, json_path, md_path,
              (", " + all_md_path) if args.all else "", time.time() - t_start))
    return 2 if n_stopped_total else 0


if __name__ == "__main__":
    sys.exit(main())
