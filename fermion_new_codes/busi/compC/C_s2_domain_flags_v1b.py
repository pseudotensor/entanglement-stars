#!/usr/bin/env python
"""C_s2_domain_flags_v1b.py (the busi lane, Computation C): calculation (i) of C_s2_fold_reading_v1.2.md, section 3, the domain flags at the scan's
stored points, through C_s2_soft_scan_v1's capture machinery (imported; nothing of the scan copied or edited) and with none of the engine edited.
v1b: built from C_s2_domain_flags_v1.py by exact replacement (build_C_s2_domain_flags_v1b.py). The one change in the computation is the zero_crossings
helper, which took np.sign of the samples and reported an exactly-zero sample twice (its sign 0 differing from both neighbours'); it now reports such a
sample once, at its own abscissa, and interpolates only between adjacent nonzero samples of opposite sign. The output file names carry v1b; read_flags,
the bisection, the summary and the tables are unchanged.

What it does, per s (0, the uncoupled map; 1, the coupled map), on the scan's state file soft_scan_v1_state.npz and its json soft_scan_v1.json:
(1) The engine in process exactly as the scan reaches it: C_s2_soft_scan_v1.import_engine, .capture_setup (engine_twofield_v6.main() under G3's
    arguments, taken from the scan's json args, with newton_record_v32 replaced for that call by the scan's capture: the record setup at T = 0 and
    no Newton step), .make_context (nk_stage_v32 with nk_loop_v32 replaced for one call by the scan's LoopCapture: the stage's F = G_s(x) - x, no
    step) and .eval_F (1 evaluation of F at x; the field solve's (Nn, vn, Wn, info) read right after F returns). No J.v product is made.
(2) One evaluation at each stored point: x_floor (the gate's kept iterate, the scan's dN0 origin), xc_0 (the floor corrected in the complement of v),
    xc_1 to xc_14 (the corrected grid points, the + side then the - side) and xpol (the polished point). The first, x_floor, must reproduce the
    scan's logged kept residual within --floor-rel-tol, the solve's module must be --fields-module and its info must carry the flag keys, or the
    run stops. Per point: dN0 = N at site 0 minus x_floor's, the residual (full = |r|_inf, soft = (v.r) v[0], perp = |r - (v.r) v|_inf with the
    scan's v), its match to the scan's json (--residual-rel-tol), E, and engine_fields_v16's own flags from info: failed, centre_flag,
    centre_series_check (regular_start), ivp.series (rho0, rho2, J1, J3, a, b, q1, q3, L2, L4), gauge (label, intervals, min_U_over_U_ref,
    delta_L_bound, lapse_unresolved, zeros, segments, stretches, near_tangencies), W_negative, regime; the dense readouts rho_ell(0), N_dense(0),
    J_ell(0), U, W_field, W_geom, N_dense, v_dense, rho_ell and J_ell at the first --first-sites sites, U and W_field on F16's fine grid (ell/16),
    and central_cell_v5 (W/2g at r = 1 and 2, J/v at r = 1) as the scan read it.
(3) The regular domain as the reading states it from F16: rho_0 > 0 (centre_flag None, regular_start), U > 0 on (0, r_collar) (no gauge intervals,
    label "intact", U positive on the fine grid), the lapse resolved, W not negative, the solve not failed; every condition kept apart, so the
    reader sees which fails. A continued point is labelled by what left: "centre" (the first interval starts at r = 0, rho_0 <= 0) or "away" (the
    first interval starts at a downward zero r_* > 0 with rho_0 > 0).
(4) The boundary along each curve: for each flag in --bisection-flags (default the combined regular flag and rho0_positive), each adjacent pair of
    stored points (sorted by dN0) where the flag differs defines a chord x(t) = (1 - t) x_a + t x_b between the two corrected points; the flag's
    first switch from the lower end is bisected on the chord with new evaluations at the midpoint until the bracket is 2^(-bisection_steps) of the
    chord wide or --bisection-steps new evaluations were made (the chord's points are one pool shared by the flags; at most --max-chords chords per
    s, the nearest to the floor first). The chord is a secant of the corrected curve, not the curve (no correction is made: no products); the perp
    residual at each chord point says how far off the curve it is. The best regular residual (over the stored and the chord points) is compared
    with the floor's kept residual and with the corrected floor's.
Outputs in --out: C_s2_domain_flags_v1.json (every scalar), C_s2_domain_flags_v1.npz (x, r, the fine-grid U, W and N and the first-site readouts of
every evaluated point), C_s2_domain_flags_v1.md (the tables and the reading). Resumable per point: the json and the npz are written after every
evaluation; a restart skips what is done (--fresh ignores a previous run). No plotting, no dates.
Usage: cd <internal>/code && python C_s2_domain_flags_v1b.py --engine-dir <internal>/code --scan-dir <internal>/soft_scan_v1
  --out <internal>/domain_flags_v1 --gpu 0"""
import argparse
import importlib
import json
import os
import sys
import time
import types

import numpy as np

sys.dont_write_bytecode = True

SC = None                       # C_s2_soft_scan_v1 (the scan module), imported in main(); SC.E is the engine after SC.import_engine
T_START = time.time()
EVALS = {"n": 0, "seconds": 0.0}

CONDITIONS = ("not_failed", "centre_flag_none", "regular_start", "rho0_positive", "no_intervals", "label_intact", "lapse_resolved",
              "U_positive_on_fine_grid", "W_not_negative")
FLAG_NAMES = ("regular",) + CONDITIONS
EXPECTED_INFO_KEYS = ("failed", "module", "centre_flag", "centre_series_check", "ivp", "gauge", "regime", "rfine", "U_dense", "W_field", "W_geom",
                      "N_dense", "v_dense", "rho_ell", "J_ell", "E_collar", "r_collar", "residual_W", "residual_invN", "N_min")
FLOOR_REQUIRED_KEYS = ("centre_flag", "centre_series_check", "ivp", "gauge", "rfine", "U_dense", "W_field")
FIRST_SITE_ROWS = ("W_over_2g", "W_geom_over_2g", "U", "N_dense", "v_dense", "rho_ell", "J_ell")


def say(msg):
    print("[domain flags %6.0f s] %s" % (time.time() - T_START, msg), flush=True)


def f_(x, fm="%.3e"):
    if x is None:
        return "None"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return fm % x
        except (TypeError, ValueError):
            return str(x)
    return str(x)


def _num(x):
    """python scalars and lists out of numpy's (json's sake); strings and None unchanged; anything else its str."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_num(q) for q in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_num(q) for q in x]
    if isinstance(x, dict):
        return {str(k): _num(v) for k, v in x.items()}
    if x is None or isinstance(x, str):
        return x
    return str(x)


def _jd(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def to_numpy(x):
    """a host float array out of whatever the setup holds (numpy, or a device array with detach/cpu/numpy/get)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    if hasattr(x, "get") and not isinstance(x, (np.ndarray, dict)):
        x = x.get()
    return np.asarray(x, float)


def _call1(fn, r):
    """a dense function of the field solve at the radii r (F16's closures take arrays; the source splines take scalars or arrays): a 1-d float array,
    nan where the function cannot be read."""
    r = np.atleast_1d(np.asarray(r, float))
    if fn is None:
        return np.full(r.shape, np.nan)
    try:
        out = np.asarray(fn(r), float).reshape(-1)
        if out.shape[0] == r.shape[0]:
            return out
    except Exception:
        pass
    vals = []
    for q in r:
        try:
            vals.append(float(np.asarray(fn(float(q)), float).reshape(-1)[0]))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, float)


# ------------------------------------------------------------------------------------------------------------ F16's flags (engine-free)
def classify_left_first(rec):
    """what left the regular domain at a continued point: 'centre' (the first interval from r = 0, rho_0 <= 0), 'away' (a downward zero at r_* > 0),
    'failed' (the integration failed), 'other' (U nonpositive on the fine grid or W negative without an interval); None at a regular point."""
    if rec.get("regular"):
        return None
    if rec.get("failed") is not None:
        return "failed"
    gg = rec.get("gauge") or {}
    if rec.get("centre_flag") is not None:
        return "centre"
    st = gg.get("first_interval_start")
    if st is not None and st > 0.0:
        return "away"
    if st is not None and st == 0.0:
        return "centre"
    return "other"


def read_flags(info, g, R_first=None, n_first=4):
    """engine_fields_v16's flags and dense readouts from the field solve's info (the keys as the module writes them; nothing of the engine called):
    returns (rec, arrays). R_first: the radii of the first sites (default: the solve's own chain, info['src'].R, its first n_first sites)."""
    rec = {"info_present": info is not None, "info_keys_missing": [], "failed": None, "module": None, "centre_flag": None, "centre_series_check": None,
           "series": None, "ivp": None, "gauge": None, "first_zero": None, "W_negative": None, "regime": None, "r_collar": None, "E_collar": None,
           "residual_W": None, "residual_invN": None, "N_min": None, "rho_ell_0": None, "N_dense_0": None, "J_ell_0": None, "U_fine": None, "W_fine": None,
           "first_sites": None, "regular_conditions": {k: None for k in CONDITIONS}, "regular": False, "left_first": None}
    arrays = {}
    if info is None or not hasattr(info, "get"):
        rec["info_keys_missing"] = list(EXPECTED_INFO_KEYS)
        rec["regular_conditions"]["not_failed"] = False
        rec["left_first"] = "no solve"
        return rec, arrays
    rec["info_keys_missing"] = [k for k in EXPECTED_INFO_KEYS if k not in info]
    failed = info.get("failed")
    rec["failed"] = (str(failed) if failed else None)
    rec["module"] = (str(info["module"]) if info.get("module") is not None else None)
    cf = info.get("centre_flag")
    rec["centre_flag"] = (str(cf) if cf is not None else None)
    csc = info.get("centre_series_check") or {}
    rec["centre_series_check"] = {k: _num(csc.get(k)) for k in ("rho_spline_minus_series_rel", "J_spline_minus_series_rel", "regular_start")} if csc else None
    ivp = info.get("ivp") or {}
    ser = ivp.get("series") or {}
    rec["series"] = {k: _num(ser.get(k)) for k in ("rho0", "rho2", "J1", "J3", "a", "b", "q1", "q3", "L2", "L4", "regular")} if ser else None
    rec["ivp"] = {k: _num(ivp.get(k)) for k in ("r0", "U0", "lnN0", "lnN_collar", "N_centre", "nfev", "segments")} if ivp else None
    gz = info.get("gauge")
    if isinstance(gz, dict):
        ivs = [[float(p[0]), float(p[1])] for p in (gz.get("intervals") or [])]
        zs = [{k_: _num(z.get(k_)) for k_ in ("r", "kind", "U_prime")} for z in (gz.get("zeros") or []) if isinstance(z, dict)]
        rec["gauge"] = {"label": (str(gz["label"]) if gz.get("label") is not None else None), "intervals": ivs, "n_intervals": len(ivs),
                        "first_interval_start": (ivs[0][0] if ivs else None), "first_interval_end": (ivs[0][1] if ivs else None),
                        "min_U_over_U_ref": _num(gz.get("min_U_over_U_ref")), "delta_L_bound": _num(gz.get("delta_L_bound")),
                        "delta_L_bound_two_resolutions": _num(gz.get("delta_L_bound_two_resolutions")), "lapse_unresolved": _num(gz.get("lapse_unresolved")),
                        "zeros": zs, "segments": _num(gz.get("segments")), "stretches": _num(gz.get("stretches")), "near_tangencies": _num(gz.get("near_tangencies")),
                        "U_scale": _num(gz.get("U_scale")), "U_switch": _num(gz.get("U_switch"))}
        rec["first_zero"] = (zs[0] if zs else None)
    wn = info.get("W_negative")
    rec["W_negative"] = (_num(wn) if isinstance(wn, dict) else None)
    rg = info.get("regime") or {}
    rec["regime"] = {k: _num(rg.get(k)) for k in ("label", "max_ratio", "r_at_max", "intervals_above_1")} if rg else None
    for k in ("r_collar", "E_collar", "residual_W", "residual_invN", "N_min"):
        rec[k] = _num(info.get(k))
    # the dense readouts
    rf = info.get("rfine"); Ud = info.get("U_dense"); Wf = info.get("W_field"); Wg = info.get("W_geom"); Nd = info.get("N_dense"); vd = info.get("v_dense")
    rho = info.get("rho_ell"); J = info.get("J_ell")
    if rho is not None:
        rec["rho_ell_0"] = float(_call1(rho, [0.0])[0])
    if J is not None:
        rec["J_ell_0"] = float(_call1(J, [0.0])[0])
    if Nd is not None:
        rec["N_dense_0"] = float(_call1(Nd, [0.0])[0])
    if rf is not None and Ud is not None:
        rf = np.asarray(rf, float).reshape(-1)
        U = _call1(Ud, rf)
        arrays["rfine"] = rf; arrays["U_fine"] = U
        pos = rf > 0.0
        if pos.any() and np.all(np.isfinite(U[pos])):
            Up = U[pos]; rp = rf[pos]; k = int(np.argmin(Up)); Uref = float(np.max(U))
            rec["U_fine"] = {"min_on_0_r_collar": float(Up[k]), "r_at_min": float(rp[k]), "U_ref": Uref, "min_over_U_ref": (float(Up[k] / Uref) if Uref > 0.0 else None),
                             "positive": bool(Up[k] > 0.0), "n_points": int(pos.sum()), "n_nonpositive": int(np.sum(Up <= 0.0)),
                             "spacing": (float(rf[1] - rf[0]) if len(rf) > 1 else None)}
        if Wf is not None:
            W = _call1(Wf, rf); arrays["W_fine"] = W
            if np.all(np.isfinite(W)):
                k = int(np.argmin(W)); Wm = max(float(np.max(np.abs(W))), 1e-300)
                rec["W_fine"] = {"min": float(W[k]), "r_at_min": float(rf[k]), "max_abs": Wm, "min_over_max_abs": float(W[k] / Wm)}
        if Nd is not None:
            arrays["N_fine"] = _call1(Nd, rf)
    if R_first is None:
        src = info.get("src")
        if src is not None and hasattr(src, "R"):
            try:
                R_first = np.asarray(src.R, float)[:int(n_first)]
            except Exception:
                R_first = None
    if R_first is not None and (Wf is not None or Ud is not None):
        R_first = np.atleast_1d(np.asarray(R_first, float))
        rows = [_call1(Wf, R_first) / (2.0 * g), _call1(Wg, R_first) / (2.0 * g), _call1(Ud, R_first), _call1(Nd, R_first), _call1(vd, R_first),
                _call1(rho, R_first), _call1(J, R_first)]
        arrays["first_sites"] = np.vstack(rows)
        rec["first_sites"] = {"R": R_first.tolist()}
        for i, nm in enumerate(FIRST_SITE_ROWS):
            rec["first_sites"][nm] = rows[i].tolist()
    # the regular domain's conditions, kept apart
    c = rec["regular_conditions"]
    c["not_failed"] = (rec["failed"] is None)
    c["centre_flag_none"] = (rec["centre_flag"] is None) if ("centre_flag" in info) else None
    rs_ = rec["centre_series_check"].get("regular_start") if rec["centre_series_check"] else None
    c["regular_start"] = (bool(rs_) if rs_ is not None else None)
    rho0 = rec["series"].get("rho0") if rec["series"] else None
    c["rho0_positive"] = (bool(rho0 > 0.0) if rho0 is not None else c["regular_start"])     # a = 8 pi g rho0/3 > 0 <=> rho0 > 0 (regular_start when the series is not exported)
    gg = rec["gauge"]
    c["no_intervals"] = (gg["n_intervals"] == 0) if gg else None
    c["label_intact"] = (gg["label"] == "intact") if gg else None
    c["lapse_resolved"] = (not gg["lapse_unresolved"]) if (gg and gg["lapse_unresolved"] is not None) else None
    c["U_positive_on_fine_grid"] = rec["U_fine"]["positive"] if rec["U_fine"] else None
    c["W_not_negative"] = (rec["W_negative"] is None) if ("W_field" in info) else None
    rec["regular"] = bool(all(v is True for v in c.values()))
    rec["left_first"] = classify_left_first(rec)
    return rec, arrays


def flag_value(p, fname):
    v = p.get("regular") if fname == "regular" else (p.get("regular_conditions") or {}).get(fname)
    return (bool(v) if v is not None else None)


def switching_pairs(pts_sorted, fname):
    """adjacent points (sorted by dN0, the flag readable at both) where the flag differs: [(key_lower_dN0, key_higher_dN0)]."""
    usable = [p for p in pts_sorted if flag_value(p, fname) is not None]
    return [(p["key"], q["key"]) for p, q in zip(usable, usable[1:]) if flag_value(p, fname) != flag_value(q, fname)]


def single_switch(vals):
    v = [q for q in vals if q is not None]
    return bool(sum(1 for p, q in zip(v, v[1:]) if p != q) <= 1)


def zero_crossings(pairs):
    """the zeros of a readout along dN0 ((dN0, value) pairs; None and non-finite entries skipped): a sample that is exactly 0 is a zero at its own
    abscissa, reported once with "exact" True whatever its neighbours' signs (v1 took np.sign and reported it twice, its sign 0 differing from both
    neighbours'); between adjacent samples that are both nonzero and of opposite sign the zero by linear interpolation ("exact" False); sorted by dN0."""
    pts = []
    for d, v in pairs:
        if d is None or v is None:
            continue
        try:
            d = float(d); v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(d) and np.isfinite(v):
            pts.append((d, v))
    pts.sort(key=lambda q: q[0])
    out = []
    for d, v in pts:
        if v == 0.0:
            out.append({"dN0": float(d), "between": [d, d], "values": [v, v], "exact": True})
    for (da, va), (db, vb) in zip(pts, pts[1:]):
        if va != 0.0 and vb != 0.0 and (va < 0.0) != (vb < 0.0):
            out.append({"dN0": float(da + (db - da) * va / (va - vb)), "between": [da, db], "values": [va, vb], "exact": False})
    out.sort(key=lambda z: z["dN0"])
    return out


def index_to_line(i, grid):
    """the scan's index -> (side, dN0_line): 0 the floor; 1..len(grid) the + side; len(grid)+1..2 len(grid) the - side."""
    n = len(grid)
    if i == 0:
        return "0", 0.0
    if 1 <= i <= n:
        return "+", float(grid[i - 1])
    if n + 1 <= i <= 2 * n:
        return "-", -float(grid[i - 1 - n])
    return None, None


# ------------------------------------------------------------------------------------------------------------ the chord bisection (engine-free)
def bracket_from_pool(pool, flag_lo, flag_hi, flag_of):
    """the flag's first switch from the lower end along the chord: the ends t = 0 (flag_lo) and t = 1 (flag_hi) implicit, the pool's points (each with
    't') sorted, points whose flag is neither skipped; returns (lo, hi, lo_rec, hi_rec), the records None at an end."""
    pts = []
    for p in pool:
        try:
            t = float(p.get("t"))
        except (TypeError, ValueError):
            continue
        if 0.0 < t < 1.0 and flag_of(p) in (flag_lo, flag_hi):
            pts.append((t, p))
    pts.sort(key=lambda q: q[0])
    lo, hi, lo_rec, hi_rec = 0.0, 1.0, None, None
    for t, p in pts:
        if flag_of(p) == flag_lo:
            lo, lo_rec = t, p
        else:
            hi, hi_rec = t, p
            break
    return lo, hi, lo_rec, hi_rec


def bisect_chord(pool, flag_lo, flag_hi, flag_of, target_width, max_new, evaluate, after=None):
    """new chord evaluations at the bracket's midpoint until the bracket is at most target_width wide or max_new evaluations were made; evaluate(t)
    returns the point's record (appended to the pool, 't' set); after() is called once per new point (the saving); returns (lo, hi, n_new, note)."""
    n_new = 0; note = None
    lo, hi = bracket_from_pool(pool, flag_lo, flag_hi, flag_of)[:2]
    while (hi - lo) > target_width and n_new < int(max_new):
        t = 0.5 * (lo + hi)
        rec = evaluate(t)
        rec["t"] = float(t)
        pool.append(rec); n_new += 1
        if after is not None:
            after()
        lo2, hi2 = bracket_from_pool(pool, flag_lo, flag_hi, flag_of)[:2]
        if (lo2, hi2) == (lo, hi):
            note = "the chord point at t = %.6g did not narrow the bracket (its flag %r is neither end's): the bisection stops" % (t, flag_of(rec))
            break
        lo, hi = lo2, hi2
    if note is None and (hi - lo) > target_width:
        note = "the bracket is above the target width after %d new evaluations" % n_new
    return lo, hi, n_new, note


def point_brief(rec):
    if rec is None:
        return None
    gg = rec.get("gauge") or {}; ser = rec.get("series") or {}; cc = rec.get("central_cell") or {}; wn = rec.get("W_negative") or {}; uf = rec.get("U_fine") or {}
    return {"key": rec.get("key"), "t": rec.get("t"), "dN0": rec.get("dN0"), "N_app_0": rec.get("N_app_0"), "regular": rec.get("regular"), "left_first": rec.get("left_first"),
            "failed": rec.get("failed"), "centre_flag": rec.get("centre_flag"), "rho0": ser.get("rho0"), "rho_ell_0": rec.get("rho_ell_0"), "J1": ser.get("J1"),
            "first_interval": ([gg.get("first_interval_start"), gg.get("first_interval_end")] if gg.get("n_intervals") else None), "n_intervals": gg.get("n_intervals"),
            "min_U_over_U_ref": gg.get("min_U_over_U_ref"), "U_fine_min": uf.get("min_on_0_r_collar"), "U_fine_positive": uf.get("positive"),
            "lapse_unresolved": gg.get("lapse_unresolved"), "W_over_2g_at_1": cc.get("W_over_2g_at_1"), "W_over_2g_at_2": cc.get("W_over_2g_at_2"),
            "W_negative_min_over_max": wn.get("min_W_over_max"), "full_inf": rec.get("full_inf"), "soft_N0": rec.get("soft_N0"), "perp_inf": rec.get("perp_inf"),
            "valid_perp": rec.get("valid_perp"), "regular_conditions": rec.get("regular_conditions")}


def bracket_summary(ch, fname, flag_lo, flag_hi, lo, hi, pa, pb, target, n_new, note, steps):
    _lo, _hi, lo_rec, hi_rec = bracket_from_pool(ch["points"], flag_lo, flag_hi, lambda p: flag_value(p, fname))
    lo_rec = lo_rec if lo_rec is not None else pa
    hi_rec = hi_rec if hi_rec is not None else pb
    dlo, dhi = float(lo_rec["dN0"]), float(hi_rec["dN0"])
    return {"flag": fname, "flag_at_t0": flag_lo, "flag_at_t1": flag_hi, "bracket_t": [float(lo), float(hi)], "width_t": float(hi - lo), "target_width_t": float(target),
            "bisection_steps": int(steps), "reached_target": bool((hi - lo) <= target), "bracket_dN0": [dlo, dhi], "width_dN0": abs(dhi - dlo),
            "bracket_N_app_0": [lo_rec.get("N_app_0"), hi_rec.get("N_app_0")], "lo_point": point_brief(lo_rec), "hi_point": point_brief(hi_rec),
            "new_evaluations": int(n_new), "pool_size": len(ch["points"]), "note": note, "done": True}


def plan_chords(P, flags, max_chords):
    """the chords and flags to bisect from the stored points' flags: [(flag, key_lower, key_higher)] for every switching pair of every flag, the distinct
    chords capped at max_chords by closeness to dN0 = 0; (todo, skipped)."""
    pts = sorted([p for p in (P.get("points") or {}).values() if p.get("done")], key=lambda q: q["dN0"])
    by = {p["key"]: p for p in pts}
    wanted = []
    for fname in flags:
        for ka, kb in switching_pairs(pts, fname):
            wanted.append((fname, ka, kb))
    distinct = []
    for fname, ka, kb in sorted(wanted, key=lambda w: min(abs(by[w[1]]["dN0"]), abs(by[w[2]]["dN0"]))):
        if (ka, kb) not in distinct:
            distinct.append((ka, kb))
    allowed = distinct[:int(max_chords)]
    todo = [w for w in wanted if (w[1], w[2]) in allowed]
    skipped = [w for w in wanted if (w[1], w[2]) not in allowed]
    return todo, skipped


def chord_entry(P, ka, kb):
    for ch in P.setdefault("chords", []):
        if list(ch["keys"]) == [ka, kb]:
            return ch
    pa, pb = P["points"][ka], P["points"][kb]
    ch = {"index": len(P["chords"]), "keys": [ka, kb], "dN0_ends": [pa["dN0"], pb["dN0"]], "N_app_0_ends": [pa["N_app_0"], pb["N_app_0"]], "points": [], "flags": {}}
    P["chords"].append(ch)
    return ch


def chord_best_regular(ch, P):
    ends = [P["points"].get(k) for k in ch["keys"]]
    pts = [p for p in list(ch.get("points") or []) + ends if p and p.get("regular") and p.get("full_inf") is not None]
    return point_brief(min(pts, key=lambda q: q["full_inf"])) if pts else None


# ------------------------------------------------------------------------------------------------------------ the summary per s (engine-free)
def _overlap(a, b):
    return bool(min(a) <= max(b) and min(b) <= max(a))


def leaves_first(P, B):
    """the reading of the chords: at each boundary of the regular flag (nearest the floor first) what left at the first continued point, and where
    rho_0 turns nonpositive (the rho0_positive flag's bracket) beside it."""
    reg_b = [b for b in B if b.get("flag") == "regular"]
    rho_b = [b for b in B if b.get("flag") == "rho0_positive"]
    lines, verdicts = [], []
    for b in sorted(reg_b, key=lambda q: min(abs(q["bracket_dN0"][0]), abs(q["bracket_dN0"][1]))):
        lo_p, hi_p = b.get("lo_point") or {}, b.get("hi_point") or {}
        if hi_p.get("regular") is False and lo_p.get("regular") is True:
            cont, regp = hi_p, lo_p
        elif lo_p.get("regular") is False and hi_p.get("regular") is True:
            cont, regp = lo_p, hi_p
        else:
            cont, regp = hi_p, lo_p
        lf = cont.get("left_first")
        iv = cont.get("first_interval") or [None, None]
        if lf == "centre":
            what = ("the centre's density leaves first: centre_flag set, rho_0 %s (rho_ell(0) %s), the first interval [%s, %s] from r = 0, min U/U_ref %s, W/2g(1) %s" % (
                f_(cont.get("rho0"), "%+.3e"), f_(cont.get("rho_ell_0"), "%+.3e"), f_(iv[0], "%.4f"), f_(iv[1], "%.4f"), f_(cont.get("min_U_over_U_ref"), "%+.2e"), f_(cont.get("W_over_2g_at_1"), "%+.3e")))
            verdict = "centre"
        elif lf == "away":
            what = ("U leaves first away from the centre: the first interval [%s, %s] (a downward zero at r_* = %s) with rho_0 still positive, %s (rho_ell(0) %s); min U/U_ref %s, W/2g(1) %s" % (
                f_(iv[0], "%.4f"), f_(iv[1], "%.4f"), f_(iv[0], "%.4f"), f_(cont.get("rho0"), "%+.3e"), f_(cont.get("rho_ell_0"), "%+.3e"), f_(cont.get("min_U_over_U_ref"), "%+.2e"), f_(cont.get("W_over_2g_at_1"), "%+.3e")))
            verdict = "away"
        elif lf == "failed":
            what = "the solve failed at the first continued point: %s" % cont.get("failed")
            verdict = "failed"
        else:
            what = "neither the centre flag nor an interval at the first continued point (left_first %r; U positive on the fine grid %s; W_negative %s; conditions %s)" % (
                lf, cont.get("U_fine_positive"), f_(cont.get("W_negative_min_over_max"), "%.2e"), cont.get("regular_conditions"))
            verdict = "other"
        if rho_b:
            rb = min(rho_b, key=lambda q: abs(0.5 * sum(q["bracket_dN0"]) - 0.5 * sum(b["bracket_dN0"])))
            rbr = sorted(rb["bracket_dN0"]); bbr = sorted(b["bracket_dN0"])
            if _overlap(rbr, bbr):
                rho_txt = "rho_0 turns nonpositive in dN0 [%+.5f, %+.5f], the same bracket within the resolution: the centre's density and the gauge leave together there" % tuple(rbr)
                if verdict == "away":
                    verdict = "together within resolution"
            else:
                gap = max(rbr[0] - bbr[1], bbr[0] - rbr[1])
                rho_txt = "rho_0 turns nonpositive in dN0 [%+.5f, %+.5f], %.2e %s the regular boundary in dN0 (%s the floor along the curve)" % (
                    rbr[0], rbr[1], gap, "below" if rbr[1] < bbr[0] else "above",
                    "further from" if abs(0.5 * sum(rbr)) > abs(0.5 * sum(bbr)) else "nearer")
        else:
            rho_txt = "rho_0 (series) switches sign between no two adjacent stored points (no rho0_positive boundary bisected)"
        lines.append("chord %s -> %s (dN0 %s to %s): the regular domain ends in dN0 [%+.5f, %+.5f] (width %.1e; target 2^-%d of the chord %s); the regular point nearest it %s "
                     "(dN0 %s, full %s, perp %s); the first continued point %s (dN0 %s, perp %s): %s; %s" % (
                         b["chord"][0], b["chord"][1], f_((b.get("chord_dN0_ends") or [None, None])[0], "%+.5f"), f_((b.get("chord_dN0_ends") or [None, None])[1], "%+.5f"),
                         min(b["bracket_dN0"]), max(b["bracket_dN0"]), b["width_dN0"], b.get("bisection_steps", 0), "reached" if b.get("reached_target") else "NOT reached",
                         regp.get("key"), f_(regp.get("dN0"), "%+.5f"), f_(regp.get("full_inf"), "%.3e"), f_(regp.get("perp_inf"), "%.2e"), cont.get("key"), f_(cont.get("dN0"), "%+.5f"),
                         f_(cont.get("perp_inf"), "%.2e"), what, rho_txt))
        verdicts.append(verdict)
    if not reg_b:
        lines.append("no boundary of the regular domain was bisected (no adjacent stored points with different regular flags, or the bisection skipped)")
    return {"text": lines, "verdicts": verdicts, "regular_boundaries": [{"chord": b["chord"], "bracket_dN0": b["bracket_dN0"], "width_dN0": b["width_dN0"]} for b in reg_b],
            "rho0_boundaries": [{"chord": b["chord"], "bracket_dN0": b["bracket_dN0"], "width_dN0": b["width_dN0"]} for b in rho_b]}


def summarize_s(P):
    pts_d = P.get("points") or {}
    stored = [p for p in pts_d.values() if p.get("done")]
    chords = P.get("chords") or []
    chord_pts = [p for ch in chords for p in (ch.get("points") or []) if p.get("done")]
    allp = sorted(stored + chord_pts, key=lambda q: q["dN0"])
    reg = [p for p in allp if p.get("regular")]; cont = [p for p in allp if not p.get("regular")]
    st_sorted = sorted(stored, key=lambda q: q["dN0"])
    cand = [p for p in reg if p.get("full_inf") is not None]
    best = min(cand, key=lambda q: q["full_inf"]) if cand else None
    fl = P.get("floor") or {}; kept = fl.get("kept_scan")
    xf = pts_d.get("x_floor") or {}; x0 = pts_d.get("xc_0") or {}
    B = []
    for ch in chords:
        for fname, Fd in (ch.get("flags") or {}).items():
            if Fd.get("done"):
                b = dict(Fd); b["chord"] = list(ch["keys"]); b["chord_index"] = ch["index"]; b["chord_dN0_ends"] = ch.get("dN0_ends"); b["best_regular_on_chord"] = chord_best_regular(ch, P)
                B.append(b)
    out = {"n_evaluated": len(allp), "n_stored": len(stored), "n_chord_points": len(chord_pts), "n_regular": len(reg), "n_continued": len(cont),
           "regular_dN0_range": ([reg[0]["dN0"], reg[-1]["dN0"]] if reg else None), "continued_dN0_range": ([cont[0]["dN0"], cont[-1]["dN0"]] if cont else None),
           "regular_N_app_0_range": ([min(p["N_app_0"] for p in reg), max(p["N_app_0"] for p in reg)] if reg else None),
           "stored_sequence": [[p["key"], p["dN0"], p.get("regular"), p.get("left_first")] for p in st_sorted],
           "stored_single_switch": single_switch([p.get("regular") for p in st_sorted]),
           "rho0_series_zero_crossings_dN0": zero_crossings([(p["dN0"], (p.get("series") or {}).get("rho0")) for p in allp]),
           "rho_ell_0_zero_crossings_dN0": zero_crossings([(p["dN0"], p.get("rho_ell_0")) for p in allp]),
           "W_over_2g_at_1_zero_crossings_dN0": zero_crossings([(p["dN0"], (p.get("central_cell") or {}).get("W_over_2g_at_1")) for p in allp]),
           "first_interval_start_vs_dN0": [[p["dN0"], (p.get("gauge") or {}).get("first_interval_start"), (p.get("gauge") or {}).get("first_interval_end"), p.get("left_first"),
                                            (p.get("series") or {}).get("rho0"), p["key"]] for p in cont],
           "best_regular": point_brief(best), "floor_kept_scan": kept, "floor_full_here": xf.get("full_inf"), "corrected_floor_full": x0.get("full_inf"),
           "best_regular_over_floor_kept": (float(best["full_inf"]) / float(kept) if (best is not None and kept) else None),
           "best_regular_over_corrected_floor": (float(best["full_inf"]) / float(x0["full_inf"]) if (best is not None and x0.get("full_inf")) else None),
           "best_regular_is_floor": ((best["key"] in ("x_floor", "xc_0")) if best is not None else None),
           "points_not_matching_scan": [p["key"] for p in stored if p.get("matches_scan") is False],
           "points_failed": [p["key"] for p in allp if p.get("failed")],
           "points_perp_above_valid": [p["key"] for p in allp if p.get("valid_perp") is False],
           "boundaries": B}
    out["leaves_first"] = leaves_first(P, B)
    return out


# ------------------------------------------------------------------------------------------------------------ the outputs
def _row(p, with_t=False):
    gg = p.get("gauge") or {}; ser = p.get("series") or {}; cc = p.get("central_cell") or {}; wn = p.get("W_negative") or {}
    cols = []
    if with_t:
        cols.append("%8s" % f_(p.get("t"), "%.6f"))
    cols += ["%-8s" % p.get("key"), "%+10.5f" % float(p.get("dN0") or 0.0), "%10.7f" % float(p.get("N_app_0") or 0.0), "%4s" % ("REG" if p.get("regular") else "CONT"),
             "%-6s" % (p.get("left_first") or "-"), "%11s" % f_(ser.get("rho0"), "%+.4e"), "%11s" % f_(p.get("rho_ell_0"), "%+.4e"), "%10s" % f_(ser.get("J1"), "%+.3e"),
             "%8s" % f_(gg.get("first_interval_start"), "%.4f"), "%8s" % f_(gg.get("first_interval_end"), "%.4f"), "%3s" % f_(gg.get("n_intervals"), "%d"),
             "%10s" % f_(gg.get("min_U_over_U_ref"), "%+.3e"), "%5s" % ("unres" if gg.get("lapse_unresolved") else ("ok" if gg.get("lapse_unresolved") is False else "-")),
             "%11s" % f_(cc.get("W_over_2g_at_1"), "%+.4e"), "%11s" % f_(cc.get("W_over_2g_at_2"), "%+.4e"), "%9s" % (f_(wn.get("min_W_over_max"), "%.2e") if wn else "-"),
             "%9s" % f_(p.get("full_inf"), "%.3e"), "%10s" % f_(p.get("soft_N0"), "%+.3e"), "%8s" % f_(p.get("perp_inf"), "%.2e"), "%3s" % ("yes" if p.get("valid_perp") else "no"),
             "%5s" % ("yes" if p.get("matches_scan") else ("NO" if p.get("matches_scan") is False else "-")), "%13s" % f_(p.get("E"), "%.9f"), "%4s" % f_(p.get("seconds"), "%.0f")]
    return "    " + " ".join(cols) + ((" FAILED: %s" % p["failed"]) if p.get("failed") else "")


def _header(with_t=False):
    cols = (["%8s" % "t"] if with_t else []) + ["%-8s" % "key", "%10s" % "dN0", "%10s" % "N_app(0)", "%4s" % "dom", "%-6s" % "left", "%11s" % "rho0(ser)", "%11s" % "rho_ell(0)",
                                                 "%10s" % "J1", "%8s" % "iv1_lo", "%8s" % "iv1_hi", "%3s" % "niv", "%10s" % "minU/Uref", "%5s" % "lapse", "%11s" % "W/2g(1)",
                                                 "%11s" % "W/2g(2)", "%9s" % "Wneg_min", "%9s" % "full", "%10s" % "soft_N0", "%8s" % "perp", "%3s" % "vld", "%5s" % "scan", "%13s" % "E", "%4s" % "s"]
    return "    " + " ".join(cols)


def write_md(RES, path):
    A = RES.get("args") or {}
    L = ["# C_s2_domain_flags_v1b: the domain flags at the scan's stored points (calculation (i) of C_s2_fold_reading_v1.2.md, section 3)", ""]
    L.append("1 evaluation of the engine's F at each stored point of the scan (x_floor, the corrected floor xc_0, the corrected grid points xc_1 to xc_14, the polished point xpol), "
             "through C_s2_soft_scan_v1's capture (import_engine, capture_setup, make_context, eval_F; no J.v product; nothing of the scan or the engine edited); the field solve's "
             "info read for engine_fields_v16's own flags. The regular domain: rho_0 > 0 (centre_flag None, regular_start), U > 0 on (0, r_collar) (no gauge intervals, label intact, "
             "U positive on F16's fine grid, spacing ell/16), the lapse resolved, W not negative, the solve not failed. A continued point is labelled by what left: centre (the first "
             "interval starts at r = 0, rho_0 <= 0) or away (a downward zero of U at r_* > 0 with rho_0 > 0). Boundaries: the first switch of a flag between two adjacent stored "
             "points, bisected on the chord x(t) = (1 - t) x_a + t x_b between the two corrected points (a secant of the corrected curve, not corrected itself: its perp residual "
             "says how far off the curve it is), %s new evaluations at most per flag and chord (target width 2^-%s of the chord), flags %s, at most %s chords per s." % (
                 A.get("bisection_steps"), A.get("bisection_steps"), A.get("bisection_flags"), A.get("max_chords")))
    L.append("")
    sc = RES.get("scan") or {}; st = RES.get("setup") or {}; ct = RES.get("counts") or {}; mem = RES.get("memory") or {}
    L.append("Inputs: scan %s (state %s); engine %s; the capture's G3 arguments reproduce the scan's json list: %s; field module required %s; floor tolerance %s; per-point match tolerance %s; "
             "valid perp %s; first sites %s (R %s)." % (sc.get("json"), sc.get("state"), st.get("engine_file"), sc.get("engine_args_match"), A.get("fields_module"), A.get("floor_rel_tol"),
                                                       A.get("residual_rel_tol"), A.get("perp_ok"), A.get("first_sites"), st.get("R_first_sites")))
    L.append("Status %s; %s evaluations in %s s (total %s s); peak device memory %s GiB." % (RES.get("status"), ct.get("evaluations"), f_(ct.get("seconds_in_evaluations"), "%.0f"),
                                                                                           f_(ct.get("seconds_total"), "%.0f"), f_(mem.get("peak_gib"), "%.2f")))
    for sk, P in sorted((RES.get("per_s") or {}).items(), key=lambda kv: float(kv[0])):
        L += ["", "## s = %s: %s" % (sk, P.get("map"))]
        if P.get("skipped"):
            L.append("skipped: %s" % P["skipped"])
            continue
        fl = P.get("floor") or {}; seg = fl.get("state_equals_gate") or {}
        L.append("floor: kept (the scan's json) %s from %s; reproduced here %s (relative difference %s; %s); regular %s; the state's floor against the gate's fields: %s." % (
            f_(fl.get("kept_scan"), "%.6e"), fl.get("kept_from"), f_(fl.get("residual_reproduced"), "%.6e"), f_(fl.get("relative_difference"), "%.1e"),
            "reproduced" if fl.get("reproduced") else "NOT REPRODUCED", fl.get("regular"),
            ("bitwise" if seg.get("bitwise") else "max |diff| %s" % f_(seg.get("max_abs_diff"), "%.1e")) if seg.get("checked") else "not checked (%s)" % seg.get("why")))
        if fl.get("stopped"):
            L.append("STOPPED at the floor: %s" % "; ".join(fl["stopped"]))
        L += ["", "### the stored points (sorted by dN0 = N at site 0 minus x_floor's; dom REG/CONT the combined regular flag; left what left; rho0(ser) the series' coefficient; iv1 the first gauge interval; "
                  "minU/Uref F16's; lapse F16's lapse_unresolved; W/2g from central_cell_v5; Wneg_min F16's W_negative min_W_over_max; vld perp <= the valid bound; scan the residual's match to the scan's json)", ""]
        L.append(_header())
        for p in sorted([q for q in (P.get("points") or {}).values() if q.get("done")], key=lambda q: q["dN0"]):
            L.append(_row(p))
        for ch in P.get("chords") or []:
            pa, pb = (P.get("points") or {}).get(ch["keys"][0]), (P.get("points") or {}).get(ch["keys"][1])
            L += ["", "### chord %d: %s (dN0 %s) -> %s (dN0 %s), t = 0 at the first; %d chord points" % (ch["index"], ch["keys"][0], f_(ch["dN0_ends"][0], "%+.5f"), ch["keys"][1],
                                                                                                           f_(ch["dN0_ends"][1], "%+.5f"), len(ch.get("points") or [])), ""]
            L.append(_header(with_t=True))
            rows = []
            if pa:
                rows.append(dict(pa, t=0.0))
            rows += [p for p in (ch.get("points") or []) if p.get("done")]
            if pb:
                rows.append(dict(pb, t=1.0))
            for p in sorted(rows, key=lambda q: float(q.get("t") if q.get("t") is not None else 0.0)):
                L.append(_row(p, with_t=True))
            for fname, Fd in (ch.get("flags") or {}).items():
                if not Fd.get("done"):
                    continue
                lp, hp = Fd.get("lo_point") or {}, Fd.get("hi_point") or {}
                L.append("    flag %s: t = 0 %s, t = 1 %s; bracket t [%.6f, %.6f] (width %.2e, target %.2e %s); dN0 [%+.6f, %+.6f] (width %.2e); N at site 0 [%s, %s]; new evaluations %d; "
                         "the lower point %s (dN0 %s, regular %s, rho0 %s, first interval %s, full %s, perp %s); the upper point %s (dN0 %s, regular %s, rho0 %s, first interval %s, full %s, perp %s)%s" % (
                             fname, Fd["flag_at_t0"], Fd["flag_at_t1"], Fd["bracket_t"][0], Fd["bracket_t"][1], Fd["width_t"], Fd["target_width_t"], "reached" if Fd["reached_target"] else "NOT reached",
                             Fd["bracket_dN0"][0], Fd["bracket_dN0"][1], Fd["width_dN0"], f_(Fd["bracket_N_app_0"][0], "%.7f"), f_(Fd["bracket_N_app_0"][1], "%.7f"), Fd["new_evaluations"],
                             lp.get("key"), f_(lp.get("dN0"), "%+.6f"), lp.get("regular"), f_(lp.get("rho0"), "%+.3e"), lp.get("first_interval"), f_(lp.get("full_inf"), "%.3e"), f_(lp.get("perp_inf"), "%.2e"),
                             hp.get("key"), f_(hp.get("dN0"), "%+.6f"), hp.get("regular"), f_(hp.get("rho0"), "%+.3e"), hp.get("first_interval"), f_(hp.get("full_inf"), "%.3e"), f_(hp.get("perp_inf"), "%.2e"),
                             (" [%s]" % Fd["note"]) if Fd.get("note") else ""))
        Sm = P.get("summary") or {}
        if Sm:
            L += ["", "### the reading, s = %s" % sk, ""]
            L.append("- %d points evaluated (%d stored, %d on chords): %d regular, %d continued; the stored points' regular flags along dN0 form a single switch: %s." % (
                Sm["n_evaluated"], Sm["n_stored"], Sm["n_chord_points"], Sm["n_regular"], Sm["n_continued"], Sm["stored_single_switch"]))
            rr = Sm.get("regular_dN0_range"); cr = Sm.get("continued_dN0_range"); nr = Sm.get("regular_N_app_0_range")
            L.append("- regular points over dN0 [%s, %s] (N at site 0 %s to %s); continued points over dN0 [%s, %s]." % (
                f_((rr or [None, None])[0], "%+.5f"), f_((rr or [None, None])[1], "%+.5f"), f_((nr or [None, None])[0], "%.7f"), f_((nr or [None, None])[1], "%.7f"),
                f_((cr or [None, None])[0], "%+.5f"), f_((cr or [None, None])[1], "%+.5f")))
            for ln in (Sm.get("leaves_first") or {}).get("text") or []:
                L.append("- %s" % ln)
            for nm, lab in (("rho0_series_zero_crossings_dN0", "rho_0 (the series' coefficient)"), ("rho_ell_0_zero_crossings_dN0", "rho_ell(0) (the spline)"),
                            ("W_over_2g_at_1_zero_crossings_dN0", "W/2g at r = 1 (the cell)")):
                zc = Sm.get(nm) or []
                L.append("- %s crosses 0 at dN0 %s (linear interpolation between adjacent evaluated points of opposite sign; an exactly-zero sample reported once at its own abscissa%s)." % (
                    lab, ", ".join((("%+.5f (an exactly-zero sample)" % z["dN0"]) if z.get("exact") else ("%+.5f (between %+.5f and %+.5f)" % (z["dN0"], z["between"][0], z["between"][1]))) for z in zc) if zc else "nowhere among the evaluated points",
                    "; the reading's estimate from the table: about -0.003 at s = 0 and -0.008 at s = 1" if nm.startswith("W_over") else ""))
            fis = Sm.get("first_interval_start_vs_dN0") or []
            if fis:
                L.append("- the first interval's start as a function of dN0 over the continued points: %s." % "; ".join(
                    "dN0 %+.5f: [%s, %s] (%s, rho0 %s, %s)" % (q[0], f_(q[1], "%.4f"), f_(q[2], "%.4f"), q[3], f_(q[4], "%+.2e"), q[5]) for q in fis))
            br = Sm.get("best_regular") or {}
            L.append("- the best regular residual on this curve: %s at %s (dN0 %s, perp %s; %s); the floor's kept %s (ratio %s), the corrected floor's full %s (ratio %s)." % (
                f_(br.get("full_inf"), "%.4e"), br.get("key"), f_(br.get("dN0"), "%+.5f"), f_(br.get("perp_inf"), "%.2e"),
                "the floor itself" if Sm.get("best_regular_is_floor") else "not the floor", f_(Sm.get("floor_kept_scan"), "%.4e"), f_(Sm.get("best_regular_over_floor_kept"), "%.3f"),
                f_(Sm.get("corrected_floor_full"), "%.4e"), f_(Sm.get("best_regular_over_corrected_floor"), "%.3f")))
            if Sm.get("points_not_matching_scan"):
                L.append("- points whose residual does not match the scan's json within the tolerance: %s (the correspondence to the scan's table is broken there)." % Sm["points_not_matching_scan"])
            if Sm.get("points_failed"):
                L.append("- points whose solve failed: %s." % Sm["points_failed"])
            if Sm.get("points_perp_above_valid"):
                L.append("- points with perp above the valid bound (off the corrected curve by more than the scan accepted): %s." % Sm["points_perp_above_valid"])
            if P.get("chords_skipped_by_cap"):
                L.append("- chords not bisected for the cap: %s." % P["chords_skipped_by_cap"])
    L += ["", "## caveats", "",
          "- The chord between two corrected points is a secant of the scan's corrected curve; its points are not corrected (no J.v product is made here), so a bracket's dN0 is the chord's N at site 0, "
          "which is the curve's to second order in the pair's separation, and the perp residual at the chord points is the off-curve part; the full residual along a chord includes it.",
          "- The flags are engine_fields_v16's own: the intervals are the integration's exact zeros, the centre flag its series' sign, the fine-grid positivity of U its ell/16 grid; none is an error bound, "
          "and between evaluated points nothing is read beyond the brackets.",
          "- rho0 (series) is the analytic Taylor coefficient of the smoothed sources at the centre (centre_coefficients); rho_ell(0) is the source spline's value at 0; both are read.",
          "- The comparison of residuals is in the max norm of the engine's F, as the scan's; the soft/perp split uses the scan's stored v."]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def save_all(RES, ARR, paths):
    RES["counts"] = {"evaluations": int(EVALS["n"]), "seconds_in_evaluations": float(EVALS["seconds"]), "seconds_total": float(time.time() - T_START)}
    peak = None
    try:
        if SC is not None and getattr(SC, "E", None) is not None:
            peak = float(SC.E.MEM["peak_gib"])
    except Exception:
        peak = None
    RES["memory"] = {"peak_gib": peak}
    tmp = paths["json"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(RES, fh, default=_jd, indent=1)
    os.replace(tmp, paths["json"])
    tmpz = paths["npz"] + ".tmp.npz"
    np.savez(tmpz, **ARR)
    os.replace(tmpz, paths["npz"])
    write_md(RES, paths["md"])


# ------------------------------------------------------------------------------------------------------------ the engine, through the scan
def scan_namespace(SJ, o):
    """the argument namespace the scan's functions read: its own json args for G3's call, this script's engine, out and gpu."""
    A = SJ.get("args") or {}
    need = ("records", "nsites", "rs", "tau", "ramp", "bisect", "gain", "floor", "base_above", "v36_opts")
    missing = [k for k in need if k not in A]
    if missing:
        raise SystemExit("the scan's json args lack %s: the capture's arguments cannot be reproduced" % missing)
    ns = types.SimpleNamespace(engine_dir=o.engine_dir, engine=o.engine, out=o.out, gpu=o.gpu)
    for k in need:
        setattr(ns, k, A[k])
    ns.records = int(ns.records); ns.nsites = int(ns.nsites); ns.rs = float(ns.rs); ns.tau = float(ns.tau); ns.bisect = int(ns.bisect); ns.gain = int(ns.gain)
    ns.floor = float(ns.floor); ns.v36_opts = bool(ns.v36_opts); ns.ramp = str(ns.ramp)
    ns.base_above = (None if ns.base_above is None else float(ns.base_above))
    return ns


def import_scan(o):
    global SC
    cands = [o.scan_code_dir, os.path.dirname(os.path.abspath(__file__)), o.engine_dir]
    for d in cands:
        if d and os.path.exists(os.path.join(d, o.scan_module + ".py")):
            d = os.path.abspath(d)
            if d not in sys.path:
                sys.path.insert(0, d)
            SC = importlib.import_module(o.scan_module)
            break
    if SC is None:
        raise SystemExit("the scan module %s.py was found in none of %s" % (o.scan_module, [c for c in cands if c]))
    for name in ("import_engine", "capture_setup", "make_context", "eval_F", "g3_engine_args", "g3_extra_args", "SETUP"):
        if not hasattr(SC, name):
            raise SystemExit("the scan module %s has no %s: not the scan's capture machinery" % (SC.__file__, name))
    say("the scan's capture machinery imported from %s" % SC.__file__)
    return SC


def map_of(S, s):
    """the map G_s as the scan's scan_one_s builds it from the captured setup: (sources_fn, chl_s, light_on_s)."""
    heavy_only = (float(s) == 0.0)
    chl_s = () if heavy_only else S["chl"]
    light_on_s = False if heavy_only else bool(S["light_on"])
    if heavy_only:
        def sources_fn(c_):
            return S["sources_with_heavy"](S["base"], c_)
    else:
        def sources_fn(c_):
            return S["sources_with"](S["base"], c_)
    return sources_fn, chl_s, light_on_s


def check_state_against_gate(o, SJ, sk, x_floor, n):
    A = SJ.get("args") or {}
    gd = A.get("gate_dir") if o.gate_dir is None else o.gate_dir
    if not gd:
        return {"checked": False, "why": "no gate directory given"}
    tags = [q.strip() for q in str(A.get("gate_tags") or "g2_s2_v6_off_ramp,g3_s2_v6_on_ramp").split(",") if q.strip()]
    if len(tags) < 2:
        return {"checked": False, "why": "no gate tags in the scan's json args"}
    tag = tags[0 if sk == "0" else 1]
    pz = os.path.join(gd, "collapse_%s_fields.npz" % tag)
    if not os.path.exists(pz):
        return {"checked": False, "why": "no %s" % pz, "tag": tag}
    try:
        Z = np.load(pz, allow_pickle=True)
        N = np.asarray(Z["N_applied"][0], float); v = np.asarray(Z["v_applied"][0], float)
    except Exception as ex:
        return {"checked": False, "why": "unreadable: %s" % str(ex)[:120], "npz": pz}
    if N.shape != (n,) or v.shape != (n,):
        return {"checked": False, "why": "shapes %s and %s against %d sites" % (N.shape, v.shape, n), "npz": pz}
    xg = np.concatenate([N, v])
    return {"checked": True, "npz": pz, "tag": tag, "bitwise": bool(np.array_equal(xg, x_floor)), "max_abs_diff": float(np.max(np.abs(xg - x_floor)))}


def stored_points(STATE, sk):
    """[(key, state key)] in the evaluation order: x_floor, xc_0 .. xc_14 (the state's indices), xpol."""
    out = []
    if ("x_floor_s%s" % sk) in STATE:
        out.append(("x_floor", "x_floor_s%s" % sk))
    pre = "xc_s%s_" % sk
    idx = sorted(int(k[len(pre):]) for k in STATE if k.startswith(pre) and k[len(pre):].isdigit())
    out += [("xc_%d" % i, "%s%d" % (pre, i)) for i in idx]
    if ("xpol_s%s" % sk) in STATE:
        out.append(("xpol", "xpol_s%s" % sk))
    return out


def scan_meta(PS, grid, key):
    """the scan's own readouts of the stored point (its json), for the comparison; the index -> side/dN0_line mapping as the scan's scan_one_s."""
    m = {"scan_key": key}
    if key == "x_floor":
        fl = PS.get("floor") or {}
        cc = fl.get("central_cell") if isinstance(fl.get("central_cell"), dict) else {}
        m.update({"scan_tag": "the floor iterate (uncorrected; the gate's kept iterate)", "scan_full_inf": fl.get("residual_reproduced", fl.get("kept")), "scan_kept": fl.get("kept"),
                  "scan_perp_inf": fl.get("perp_inf_uncorrected"), "scan_soft_N0": fl.get("soft_N0_uncorrected"), "scan_dN0_corrected": 0.0, "scan_N_app_0": fl.get("N_app_0"),
                  "scan_W_over_2g_at_1": cc.get("W_over_2g_at_1"), "scan_W_over_2g_at_2": cc.get("W_over_2g_at_2")})
        return m
    if key == "xpol":
        bp = (PS.get("polish") or {}).get("best")
        bp = bp if isinstance(bp, dict) else {}
        cc = bp.get("central_cell") if isinstance(bp.get("central_cell"), dict) else {}
        m.update({"scan_tag": bp.get("tag"), "scan_iteration": bp.get("iteration"), "scan_alpha": bp.get("alpha"), "scan_dN0_line": bp.get("dN0_line"),
                  "scan_dN0_corrected": bp.get("dN0_corrected"), "scan_N_app_0": bp.get("N_app_0"), "scan_full_inf": bp.get("full_inf"), "scan_perp_inf": bp.get("perp_inf"),
                  "scan_soft_N0": bp.get("soft_N0"), "scan_W_over_2g_at_1": cc.get("W_over_2g_at_1"), "scan_W_over_2g_at_2": cc.get("W_over_2g_at_2")})
        return m
    try:
        i = int(key.split("_")[1])
    except (IndexError, ValueError):
        i = None
    pts = {}
    for p in (PS.get("points") or []):
        if isinstance(p, dict) and "index" in p:
            pts[int(p["index"])] = p
    p = pts.get(i)
    if p is not None:
        cc = p.get("central_cell") if isinstance(p.get("central_cell"), dict) else {}
        m.update({"scan_index": i, "scan_side": p.get("side"), "scan_tag": p.get("tag"), "scan_alpha": p.get("alpha"), "scan_dN0_line": p.get("dN0_line"),
                  "scan_dN0_corrected": p.get("dN0_corrected"), "scan_N_app_0": p.get("N_app_0"), "scan_full_inf": p.get("full_inf"), "scan_perp_inf": p.get("perp_inf"),
                  "scan_soft_N0": p.get("soft_N0"), "scan_ok": p.get("ok"), "scan_note": p.get("note"), "scan_W_over_2g_at_1": cc.get("W_over_2g_at_1"), "scan_W_over_2g_at_2": cc.get("W_over_2g_at_2")})
    else:
        side, d = index_to_line(i, grid) if i is not None else (None, None)
        m.update({"scan_index": i, "scan_side": side, "scan_dN0_line": d, "scan_tag": "index %s (not in the scan's json points)" % i})
    return m


def evaluate_point(o, S, ctx, vhat, x_floor, x, key, meta, ARR, sk, R_first):
    """1 evaluation of the engine's F at x through the scan's eval_F; the residual, F16's flags and the dense readouts recorded; the arrays into ARR."""
    E = SC.E; a = S["a"]; n = int(a.nsites); g = float(a.g)
    pk = "s%s_%s" % (sk, key)
    t0 = time.time()
    x = np.asarray(x, float)
    rec = {"key": key, "done": False, "s": float(ctx["s"]), "npz_prefix": pk}
    rec.update(meta or {})
    Nx, vx = x[:n], x[n:]
    rec.update({"N_app_0": float(x[0]), "v_app_0": float(x[n]), "dN0": float(x[0] - x_floor[0]), "min_N_app": float(np.min(Nx)), "min_N_minus_abs_v_app": float(np.min(Nx - np.abs(vx)))})
    rec["gauge_ok_before_evaluation"] = bool(np.all(np.isfinite(x)) and rec["min_N_app"] > 0.0 and rec["min_N_minus_abs_v_app"] > 0.0)
    ARR["x_" + pk] = x.copy()
    r, sol, ok = SC.eval_F(ctx, x)
    EVALS["n"] += 1
    info = (sol[3] if (sol is not None and len(sol) > 3) else None)
    rec["ok"] = bool(ok); rec["F_returned"] = (r is not None)
    if r is not None:
        r = np.asarray(r, float)
        c = float(vhat @ r)
        rec.update({"full_inf": float(np.max(np.abs(r))), "soft_coeff": c, "soft_N0": float(c * vhat[0]), "perp_inf": float(np.max(np.abs(r - c * vhat))),
                    "r_finite": bool(np.all(np.isfinite(r)))})
        rec["valid_perp"] = bool(rec["perp_inf"] <= float(o.perp_ok))
        ARR["r_" + pk] = r.copy()
    else:
        rec.update({"full_inf": None, "soft_coeff": None, "soft_N0": None, "perp_inf": None, "r_finite": False, "valid_perp": False})
    sf = rec.get("scan_full_inf")
    if sf is not None and rec["full_inf"] is not None:
        rec["rel_diff_scan"] = abs(rec["full_inf"] - float(sf)) / max(abs(float(sf)), 1e-300)
        rec["matches_scan"] = bool(rec["rel_diff_scan"] <= float(o.residual_rel_tol))
    flags, arrays = read_flags(info, g, R_first, int(o.first_sites))
    rec.update(flags)
    for k_, arr in arrays.items():
        ARR["%s_%s" % (k_, pk)] = np.asarray(arr, float)
    rec["central_cell"] = None; rec["E"] = None
    if info is not None and flags["failed"] is None:
        try:
            rec["central_cell"] = _num(E.central_cell_v5(a, info))
        except Exception as ex:
            rec["central_cell_error"] = str(ex)[:160]
        try:
            rec["E"] = float(E.FLD.charge(info, g))
        except Exception as ex:
            rec["E_error"] = str(ex)[:160]
    rec["seconds"] = float(time.time() - t0); EVALS["seconds"] += rec["seconds"]
    m6 = E.mem_v6()
    if m6 is not None:
        rec["memory_gib"] = {"allocated": float(m6[0]), "peak": float(m6[1]), "reserved": float(m6[2])}
    rec["done"] = True
    del r, sol, info
    E.free_device_v6()
    return rec


def print_rec(sk, rec):
    gg = rec.get("gauge") or {}; ser = rec.get("series") or {}; cc = rec.get("central_cell") or {}; uf = rec.get("U_fine") or {}; wn = rec.get("W_negative") or {}
    iv = ("[%s, %s]" % (f_(gg.get("first_interval_start"), "%.4f"), f_(gg.get("first_interval_end"), "%.4f"))) if gg.get("n_intervals") else "none"
    say("s = %s %-7s%s dN0 %+9.5f N_app(0) %.7f | %s%s | centre_flag %s; rho0 %s (rho_ell(0) %s); J1 %s | gauge %s; 1st interval %s (%s intervals); min U/U_ref %s, fine-grid min U %s at r %s; "
        "lapse unresolved %s | W_negative %s | W/2g(1) %s W/2g(2) %s J/v(1) %s | full %s soft %s perp %s%s | E %s | %.0f s%s" % (
            sk, rec["key"], (" (t %.6f)" % rec["t"]) if rec.get("t") is not None else "", rec["dN0"], rec["N_app_0"], "REGULAR" if rec.get("regular") else "CONTINUED",
            "" if rec.get("regular") else " (%s)" % rec.get("left_first"), "None" if rec.get("centre_flag") is None else "set", f_(ser.get("rho0"), "%+.4e"), f_(rec.get("rho_ell_0"), "%+.4e"),
            f_(ser.get("J1"), "%+.3e"), "intact" if gg.get("label") == "intact" else ("continued" if gg.get("label") else "None"), iv, gg.get("n_intervals"), f_(gg.get("min_U_over_U_ref"), "%+.3e"),
            f_(uf.get("min_on_0_r_collar"), "%+.3e"), f_(uf.get("r_at_min"), "%.3f"), gg.get("lapse_unresolved"),
            "none" if not wn else "%s at r %s" % (f_(wn.get("min_W_over_max"), "%.2e"), f_(wn.get("r"), "%.3f")), f_(cc.get("W_over_2g_at_1"), "%+.4e"), f_(cc.get("W_over_2g_at_2"), "%+.4e"),
            f_(cc.get("J_over_v_at_1"), "%+.3e"), f_(rec.get("full_inf"), "%.3e"), f_(rec.get("soft_N0"), "%+.3e"), f_(rec.get("perp_inf"), "%.2e"),
            (" (scan %s: %s)" % (f_(rec.get("scan_full_inf"), "%.3e"), "match" if rec.get("matches_scan") else "DIFFERS %s" % f_(rec.get("rel_diff_scan"), "%.1e"))) if rec.get("scan_full_inf") is not None else "",
            f_(rec.get("E"), "%.9f"), rec.get("seconds") or 0.0, (" FAILED: %s" % rec["failed"]) if rec.get("failed") else ""))


def floor_checks(o, P, rec, RES, ARR, paths):
    """x_floor's evaluation against the scan's logged kept residual, the field module and the flag keys; a failure stops the run (the state saved)."""
    F_ = P["floor"]; kept = F_.get("kept_scan"); full = rec.get("full_inf")
    rel = (abs(full - float(kept)) / max(abs(float(kept)), 1e-300)) if (kept is not None and full is not None) else None
    F_.update({"residual_reproduced": full, "relative_difference": rel, "reproduced": bool(rel is not None and rel <= float(o.floor_rel_tol)), "regular": rec.get("regular"),
               "module": rec.get("module"), "info_keys_missing": rec.get("info_keys_missing")})
    say("the floor iterate's residual by the engine's F %s against the scan's logged kept %s (relative difference %s; %s); field module %s; flag keys missing %s" % (
        f_(full, "%.10e"), f_(kept, "%.10e"), f_(rel, "%.1e"), "reproduced" if F_["reproduced"] else "NOT REPRODUCED", rec.get("module"), rec.get("info_keys_missing")))
    problems = []
    if not F_["reproduced"]:
        problems.append("the floor iterate is not reproduced to the logged residual (relative difference %s > %g): the state or the machinery is not the scan's" % (f_(rel, "%.2e"), o.floor_rel_tol))
    if rec.get("module") != o.fields_module:
        problems.append("the solve's module is %r, not %s: the flags read here are %s's" % (rec.get("module"), o.fields_module, o.fields_module))
    missing = [k for k in FLOOR_REQUIRED_KEYS if k in (rec.get("info_keys_missing") or [])]
    if missing:
        problems.append("the solve's info lacks %s: the domain flags cannot be read" % missing)
    if problems:
        F_["stopped"] = problems
        save_all(RES, ARR, paths)
        raise SystemExit("the floor evaluation is not the scan's: " + "; ".join(problems))


def memory_guard(o, RES, ARR, paths):
    if o.memory_stop_gib is None or SC is None or getattr(SC, "E", None) is None:
        return
    try:
        peak = float(SC.E.MEM["peak_gib"])
    except Exception:
        return
    if peak > float(o.memory_stop_gib):
        RES["status"] = "stopped at the memory bound (resumable)"
        save_all(RES, ARR, paths)
        raise SystemExit("peak device memory %.2f GiB above --memory-stop-gib %g: stopped after the last saved point (resumable)" % (peak, o.memory_stop_gib))


def run_one_s(o, ns, SJ, STATE, S, s, RES, ARR, paths, grid, R_first):
    E = SC.E; a = S["a"]; n = int(a.nsites); sk = "%g" % float(s)
    PS = (SJ.get("per_s") or {}).get(sk)
    kf, kv = "x_floor_s%s" % sk, "vhat_s%s" % sk
    if PS is None or kf not in STATE or kv not in STATE:
        say("s = %s: the scan's json or state has no floor and soft vector for it: skipped" % sk)
        RES["per_s"][sk] = {"s": float(s), "skipped": "no per_s entry in the scan's json or no %s/%s in the state" % (kf, kv)}
        return
    P = RES["per_s"].setdefault(sk, {"s": float(s)})
    P["map"] = PS.get("map"); P.setdefault("points", {}); P.setdefault("order", []); P.setdefault("chords", [])
    x_floor = np.asarray(STATE[kf], float); vhat = np.asarray(STATE[kv], float)
    if x_floor.shape != (2 * n,) or vhat.shape != (2 * n,):
        raise SystemExit("s = %s: the state's x_floor %s and v %s are not 2 nsites = %d long" % (sk, x_floor.shape, vhat.shape, 2 * n))
    fl = PS.get("floor") or {}
    F_ = P.setdefault("floor", {})
    F_.update({"kept_scan": fl.get("kept"), "kept_from": fl.get("kept_from"), "source": fl.get("source"), "residual_reproduced_by_scan": fl.get("residual_reproduced"),
               "N_app_0": float(x_floor[0]), "v_app_0": float(x_floor[n]), "vhat_N0": float(vhat[0]), "vhat_norm": float(np.linalg.norm(vhat))})
    if "state_equals_gate" not in F_:
        F_["state_equals_gate"] = check_state_against_gate(o, SJ, sk, x_floor, n)
        seg = F_["state_equals_gate"]
        say("s = %s: the state's floor against the gate's fields: %s" % (sk, ("bitwise" if seg.get("bitwise") else "max |diff| %s" % f_(seg.get("max_abs_diff"), "%.1e")) if seg.get("checked") else "not checked (%s)" % seg.get("why")))
    stored = stored_points(STATE, sk)
    P["stored_keys"] = [k for k, _ in stored]
    flags = [q.strip() for q in o.bisection_flags.split(",") if q.strip()]
    pending = [k for k, _ in stored if not (P["points"].get(k) or {}).get("done")]
    todo_open = []
    if not pending and not o.skip_bisection:
        todo, _skipped = plan_chords(P, flags, o.max_chords)
        for fname, ka, kb in todo:
            ch = next((c for c in P["chords"] if list(c["keys"]) == [ka, kb]), None)
            if ch is None or not (ch.get("flags") or {}).get(fname, {}).get("done"):
                todo_open.append((fname, ka, kb))
    if not pending and not todo_open:
        P["summary"] = summarize_s(P)
        save_all(RES, ARR, paths)
        say("s = %s: every stored point and every planned bisection done already; summary rewritten" % sk)
        for ln in (P["summary"].get("leaves_first") or {}).get("text") or []:
            say("s = %s: %s" % (sk, ln))
        return
    say("s = %s: %d stored points (%d pending: %s); bisection %s on flags %s with at most %d new evaluations per flag and chord (target width 2^-%d of the chord), at most %d chords" % (
        sk, len(stored), len(pending), pending, "skipped" if o.skip_bisection else "on", flags, o.bisection_steps, o.bisection_steps, o.max_chords))
    sources_fn, chl_s, light_on_s = map_of(S, s)
    ctx = SC.make_context(ns, S, float(s), sources_fn, chl_s, light_on_s, x_floor)
    try:
        for key, skey in stored:
            if (P["points"].get(key) or {}).get("done"):
                continue
            meta = scan_meta(PS, grid, key)
            meta.update({"kind": "stored point", "state_key": skey})
            rec = evaluate_point(o, S, ctx, vhat, x_floor, np.asarray(STATE[skey], float), key, meta, ARR, sk, R_first)
            P["points"][key] = rec
            if key not in P["order"]:
                P["order"].append(key)
            print_rec(sk, rec)
            if key == "x_floor":
                floor_checks(o, P, rec, RES, ARR, paths)
            save_all(RES, ARR, paths)
            memory_guard(o, RES, ARR, paths)
        if not o.skip_bisection:
            todo, skipped = plan_chords(P, flags, o.max_chords)
            P["bisection_plan"] = [list(w) for w in todo]
            P["chords_skipped_by_cap"] = [list(w) for w in skipped]
            say("s = %s: the stored points' flags give %d (flag, pair) switches: %s%s" % (sk, len(todo), [list(w) for w in todo], ("; skipped for the cap: %s" % [list(w) for w in skipped]) if skipped else ""))
            target = 2.0 ** (-int(o.bisection_steps))
            state_key = dict(stored)
            for fname, ka, kb in todo:
                ch = chord_entry(P, ka, kb)
                Fd = ch["flags"].setdefault(fname, {})
                if Fd.get("done"):
                    continue
                pa, pb = P["points"][ka], P["points"][kb]
                flag_lo, flag_hi = flag_value(pa, fname), flag_value(pb, fname)
                xa, xb = np.asarray(STATE[state_key[ka]], float), np.asarray(STATE[state_key[kb]], float)
                say("s = %s: chord %d: %s (dN0 %+.5f, %s %s) -> %s (dN0 %+.5f, %s %s): bisecting the flag %s to 2^-%d of the chord (at most %d new evaluations; %d chord points in the pool)" % (
                    sk, ch["index"], ka, pa["dN0"], fname, flag_lo, kb, pb["dN0"], fname, flag_hi, fname, o.bisection_steps, o.bisection_steps, len(ch["points"])))

                def evaluate(t, ch=ch, ka=ka, kb=kb, xa=xa, xb=xb):
                    x_t = (1.0 - t) * xa + t * xb
                    key_t = "c%d_p%d" % (int(ch["index"]), len(ch["points"]))
                    rec_t = evaluate_point(o, S, ctx, vhat, x_floor, x_t, key_t, {"kind": "chord point", "chord": [ka, kb], "chord_index": int(ch["index"]), "t": float(t)}, ARR, sk, R_first)
                    print_rec(sk, rec_t)
                    return rec_t

                def after():
                    save_all(RES, ARR, paths)
                    memory_guard(o, RES, ARR, paths)

                lo, hi, n_new, note = bisect_chord(ch["points"], flag_lo, flag_hi, lambda p, fname=fname: flag_value(p, fname), target, int(o.bisection_steps), evaluate, after=after)
                Fd.update(bracket_summary(ch, fname, flag_lo, flag_hi, lo, hi, pa, pb, target, n_new, note, int(o.bisection_steps)))
                save_all(RES, ARR, paths)
                say("s = %s: chord %d flag %s: bracket t [%.6f, %.6f] (width %.2e, target %.2e %s), dN0 [%+.6f, %+.6f] (width %.2e), %d new evaluations%s" % (
                    sk, ch["index"], fname, lo, hi, hi - lo, target, "reached" if Fd["reached_target"] else "NOT reached", Fd["bracket_dN0"][0], Fd["bracket_dN0"][1], Fd["width_dN0"], n_new,
                    (" [%s]" % note) if note else ""))
        P["summary"] = summarize_s(P)
        save_all(RES, ARR, paths)
        for ln in (P["summary"].get("leaves_first") or {}).get("text") or []:
            say("s = %s: %s" % (sk, ln))
        br = P["summary"].get("best_regular") or {}
        say("s = %s: regular points over dN0 %s, continued over %s; the best regular residual %s at %s (dN0 %s) against the floor's kept %s and the corrected floor's %s" % (
            sk, P["summary"].get("regular_dN0_range"), P["summary"].get("continued_dN0_range"), f_(br.get("full_inf"), "%.4e"), br.get("key"), f_(br.get("dN0"), "%+.5f"),
            f_(P["summary"].get("floor_kept_scan"), "%.4e"), f_(P["summary"].get("corrected_floor_full"), "%.4e")))
    finally:
        ctx.clear()
        del ctx
        E.free_device_v6()


def grid_of(SJ):
    g = SJ.get("grid_dN0")
    if isinstance(g, list) and g and all(isinstance(q, (int, float)) for q in g):
        return [float(q) for q in g]
    s = (SJ.get("args") or {}).get("dN0_grid")
    if s:
        return [float(q) for q in str(s).split(",") if q.strip()]
    return []


def main():
    ap = argparse.ArgumentParser(description="the domain flags at the soft scan's stored points (busi lane, Computation C; the fold reading v1.2's calculation (i))")
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--engine", default="engine_twofield_v6.py")
    ap.add_argument("--scan-dir", required=True, help="the scan's outputs: soft_scan_v1_state.npz and soft_scan_v1.json")
    ap.add_argument("--scan-code-dir", default=None, help="where C_s2_soft_scan_v1.py is (default: this script's directory, then --engine-dir)")
    ap.add_argument("--scan-module", default="C_s2_soft_scan_v1")
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gate-dir", default=None, help="gate v6's outputs, for the bitwise check of the state's floors against the gate's fields (default: the scan json's gate_dir; '' skips)")
    ap.add_argument("--s-values", default="0,1", help="0 (the uncoupled map) and/or 1 (the coupled map)")
    ap.add_argument("--first-sites", type=int, default=4, help="W_field, W_geom, U, N_dense, v_dense, rho_ell and J_ell read at the first this many sites")
    ap.add_argument("--floor-rel-tol", type=float, default=1e-6, help="x_floor's residual must reproduce the scan's logged kept residual to this relative difference")
    ap.add_argument("--residual-rel-tol", type=float, default=1e-6, help="a stored point's residual matches the scan's json within this relative difference")
    ap.add_argument("--perp-ok", type=float, default=None, help="the valid bound on perp (default: the scan's --perp-ok from its json, 1e-5)")
    ap.add_argument("--fields-module", default="engine_fields_v16", help="the field module the solve's info must declare (its flags are read)")
    ap.add_argument("--bisection-steps", type=int, default=6, help="new evaluations at most per flag and chord; the target bracket width is 2^-steps of the chord")
    ap.add_argument("--bisection-flags", default="regular,rho0_positive", help="comma list from %s" % ", ".join(FLAG_NAMES))
    ap.add_argument("--max-chords", type=int, default=4, help="distinct chords bisected per s (the nearest to dN0 = 0 first)")
    ap.add_argument("--skip-bisection", action="store_true")
    ap.add_argument("--memory-stop-gib", type=float, default=None, help="stop (resumable) when the peak device memory passes this")
    ap.add_argument("--fresh", action="store_true", help="ignore a previous json/npz in --out")
    o = ap.parse_args()
    flags = [q.strip() for q in o.bisection_flags.split(",") if q.strip()]
    bad = [q for q in flags if q not in FLAG_NAMES]
    if bad or not flags:
        ap.error("--bisection-flags: %s not among %s" % (bad, FLAG_NAMES))
    if o.bisection_steps < 0 or o.max_chords < 0 or o.first_sites < 1:
        ap.error("--bisection-steps and --max-chords nonnegative, --first-sites positive")
    s_values = []
    for q in o.s_values.split(","):
        if q.strip():
            s_ = float(q)
            if s_ not in (0.0, 1.0):
                ap.error("--s-values: only 0 and 1 have stored points")
            s_values.append(s_)
    os.makedirs(o.out, exist_ok=True)
    paths = {"json": os.path.join(o.out, "C_s2_domain_flags_v1b.json"), "npz": os.path.join(o.out, "C_s2_domain_flags_v1b.npz"), "md": os.path.join(o.out, "C_s2_domain_flags_v1b.md")}
    pj = os.path.join(o.scan_dir, "soft_scan_v1.json"); pz = os.path.join(o.scan_dir, "soft_scan_v1_state.npz")
    if not (os.path.exists(pj) and os.path.exists(pz)):
        raise SystemExit("the scan's json and state are needed: %s, %s" % (pj, pz))
    SJ = json.load(open(pj))
    with np.load(pz) as Z:
        STATE = {k: np.array(Z[k]) for k in Z.files}
    grid = grid_of(SJ)
    if o.perp_ok is None:
        o.perp_ok = float((SJ.get("args") or {}).get("perp_ok", 1e-5))
    RES = None; ARR = {}
    if not o.fresh and os.path.exists(paths["json"]):
        RES = json.load(open(paths["json"]))
        if os.path.exists(paths["npz"]):
            with np.load(paths["npz"]) as Z:
                ARR = {k: np.array(Z[k]) for k in Z.files}
        say("resuming from %s (%s)" % (paths["json"], {k_: len([p for p in (P.get("points") or {}).values() if p.get("done")]) for k_, P in (RES.get("per_s") or {}).items()}))
    if RES is None:
        RES = {"script": "C_s2_domain_flags_v1b.py", "args": vars(o), "per_s": {}, "status": "running"}
    RES["args"] = vars(o)
    RES["status"] = "running"
    RES["scan"] = {"dir": o.scan_dir, "json": pj, "state": pz, "state_keys": sorted(STATE.keys()), "grid_dN0": grid, "scan_args": SJ.get("args"), "perp_ok_used": float(o.perp_ok)}
    say("the domain flags: s %s; stored points per s %s; first sites %d; bisection %s (flags %s, steps %d, at most %d chords); floor tolerance %g; match tolerance %g; valid perp %g" % (
        s_values, {("%g" % s): [k for k, _ in stored_points(STATE, "%g" % s)] for s in s_values}, o.first_sites, "skipped" if o.skip_bisection else "on", flags, o.bisection_steps,
        o.max_chords, o.floor_rel_tol, o.residual_rel_tol, o.perp_ok))
    ns = scan_namespace(SJ, o)
    import_scan(o)
    engine_path = os.path.join(o.engine_dir, o.engine)
    if not os.path.exists(engine_path):
        raise SystemExit("no engine at %s" % engine_path)
    SC.import_engine(ns)
    SC.capture_setup(ns, engine_path)
    S = SC.SETUP; a = S["a"]; E = SC.E
    E.mem_v6(reset=True)
    args_here = SC.g3_engine_args(ns) + SC.g3_extra_args(ns, "on")
    stored_args = SJ.get("engine_args_g3")
    RES["scan"]["engine_args_here"] = args_here
    RES["scan"]["engine_args_match"] = (bool(list(stored_args) == list(args_here)) if isinstance(stored_args, list) else None)
    if RES["scan"]["engine_args_match"] is False:
        say("WARNING: the capture's G3 arguments differ from the scan json's engine_args_g3 (the floor check below decides whether the state is reproduced)")
    try:
        R_all = to_numpy(S["R"])
        R_first = R_all[:int(o.first_sites)]
    except Exception as ex:
        say("the chain's radii could not be read from the setup (%s): the first sites are read from the solve's own chain" % str(ex)[:80])
        R_first = None
    RES["setup"] = {"E_first_solve": S["E0"], "nsites": int(a.nsites), "g": float(a.g), "ell": _num(getattr(a, "ell", None)), "light_channels": [c.kappa for c in S["chl"]],
                    "light_sourcing": [c.sourcing for c in S["chl"]], "light_on": bool(S["light_on"]), "N_start_0": float(S["Nn0"][0]), "stop_fields": _num(a.stop_fields), "stop_E": _num(a.stop_E),
                    "R_first_sites": (R_first.tolist() if R_first is not None else None), "engine_file": getattr(E, "__file__", None)}
    say("setup: %d sites, g %g, light channels %s, E of the first solve %.9f; the first sites' radii %s" % (int(a.nsites), float(a.g), RES["setup"]["light_channels"], S["E0"], RES["setup"]["R_first_sites"]))
    save_all(RES, ARR, paths)
    for s in s_values:
        run_one_s(o, ns, SJ, STATE, S, s, RES, ARR, paths, grid, R_first)
    RES["status"] = "done"
    save_all(RES, ARR, paths)
    m6 = E.mem_v6()
    say("DOMAIN FLAGS DONE: %d evaluations in %.0f s (total %.0f s); peak device memory %.2f GiB%s; %s, %s, %s" % (
        EVALS["n"], EVALS["seconds"], time.time() - T_START, float(E.MEM["peak_gib"]), (" (allocated now %.2f, reserved %.2f)" % (m6[0], m6[2])) if m6 is not None else "",
        paths["json"], paths["npz"], paths["md"]))


if __name__ == "__main__":
    main()
