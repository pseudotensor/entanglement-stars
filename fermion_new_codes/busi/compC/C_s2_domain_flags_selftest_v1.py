#!/usr/bin/env python
"""C_s2_domain_flags_selftest_v1.py: manufactured checks of C_s2_domain_flags_v1 without the engine. (1) read_flags on constructed info dicts shaped as
engine_fields_v16 writes them: a regular solve, a centre-flagged continuation (the first interval from r = 0), a continuation from a downward zero
away from the centre, a failed integration (no gauge, no series), a near-tangency with the label intact, an intact label with a negative fine-grid
U, no info; the regular conditions, the left-first label, the first-site and fine-grid readouts, numpy types and the json round trip. (2) The chord
bisection on synthetic flags: a monotone switch (the bracket contains the switch and reaches 2^-k of the chord in k new evaluations), the reversed
orientation, a resume from a partial pool (the same points and bracket as a fresh run), a non-monotone flag (the first switch from the lower end),
an unreadable midpoint (the bisection stops instead of repeating), a second flag on the shared pool (fewer new evaluations). (3) The helpers
(zero_crossings, single_switch, switching_pairs, index_to_line, plan_chords) and a smoke test of summarize_s and write_md on a synthetic result.
Exit status 0 when every check passes, 1 otherwise. Usage: python C_s2_domain_flags_selftest_v1.py [--bisection-steps 6] [--n-random 25] [--seed 7] [--tol 1e-12]"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C_s2_domain_flags_v1 as DF  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("ok" if ok else "FAIL", name, (": " + str(detail)) if detail else ""), flush=True)


# ------------------------------------------------------------------------------------------------------- manufactured F16-shaped infos
def mk_info(kind, g=0.25, ell=2.0, r_collar=40.0):
    if kind == "none":
        return None
    rf = np.arange(0.0, r_collar + 1e-9, ell / 16.0)
    if kind == "failed":
        return {"failed": "integration failed at r = 3.2, U/U_ref = -1.0e-03, no zero bracketed: step size too small", "module": "engine_fields_v16", "centre_flag": None,
                "centre_series_check": {"rho_spline_minus_series_rel": 1e-9, "J_spline_minus_series_rel": 1e-9, "regular_start": True}, "ell": ell, "collar_policy": "flag",
                "W_negative": None}
    rho0 = {"regular": 3.0e-5, "centre": -3.1e-5, "away": 1.0e-5, "near": 2.0e-5, "inconsistent": 3.0e-5}[kind]
    rho2 = -1.0e-6; J1 = 1.0e-5; J3 = -1.0e-7
    a = 8.0 * np.pi * g * rho0 / 3.0
    regular = a > 0.0
    cc = {"rho0": rho0, "rho2": rho2, "J1": J1, "J3": J3, "a": a, "b": 8.0 * np.pi * g / 5.0 * rho2, "q1": (-np.sqrt(a) if regular else 0.0), "q3": 0.0, "L2": 0.0, "L4": 0.0,
          "regular": np.bool_(regular)}

    def base(r):
        r = np.asarray(r, float)
        return 0.02 * r ** 3 / (1.0 + 0.1 * r ** 2)
    if kind == "centre":
        def U(r):
            r = np.asarray(r, float)
            return 0.02 * r ** 2 * (r - 3.1) / (1.0 + 0.1 * r ** 2)
    elif kind == "away":
        def U(r):
            r = np.asarray(r, float)
            p = (r - 1.7) * (r - 2.9)
            return base(r) * p / (p + 1.0)
    elif kind == "inconsistent":
        rbad = float(rf[37])

        def U(r):
            r = np.asarray(r, float)
            return base(r) - 11.0 * base(r) * (np.abs(r - rbad) < 1e-9)
    else:
        U = base

    def N(r):
        return np.exp(-0.01 * np.asarray(r, float) ** 2)

    def W(r):
        return N(r) ** 2 * U(r)
    ivs = {"centre": [[0.0, 3.1]], "away": [[1.7, 2.9]]}.get(kind, [])

    def Wg(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        w = np.asarray(W(r), float)
        for (a_, b_) in ivs:
            w = np.where((r >= a_) & (r <= b_), 0.0, w)
        return np.maximum(w, 0.0)

    def v(r):
        r = np.atleast_1d(np.asarray(r, float))
        w = W(np.abs(r))
        return np.where(np.abs(r) > 0, -np.sqrt(np.where(w > 0.0, w, 0.0) / np.maximum(np.abs(r), 1e-300)), 0.0)

    def rho(r):
        return rho0 + rho2 * np.asarray(r, float) ** 2

    def J(r):
        return J1 * np.asarray(r, float) + J3 * np.asarray(r, float) ** 3
    Uf = U(rf); Umax = max(float(np.max(Uf)), 1e-300)
    gauge = {"label": "intact", "intervals": [], "min_U_over_U_ref": float(Uf.min() / Umax), "delta_L_bound": 0.0, "lapse_unresolved": np.bool_(False), "segments": 1, "zeros": [],
             "stretches": 0, "near_tangencies": 0, "U_scale": 1.0, "U_switch": 1e-6}
    if kind == "centre":
        gauge.update({"intervals": ivs, "zeros": [{"r": 3.1, "kind": "up", "U_prime": np.float64(2.0e-3)}], "segments": 2, "delta_L_bound": 1e-4, "lapse_unresolved": np.bool_(True),
                      "label": "flat-slice gauge continued on [0.0000, 3.1000]: min U/U_ref = %.3e (auxiliary signed U, the energy deficit's bookkeeping); lapse unresolved" % (Uf.min() / Umax)})
    if kind == "away":
        gauge.update({"intervals": ivs, "zeros": [{"r": 1.7, "kind": "down", "U_prime": -1.0e-3}, {"r": 2.9, "kind": "up", "U_prime": 2.0e-3}], "segments": 4, "stretches": 2,
                      "delta_L_bound": 3e-5, "lapse_unresolved": np.bool_(True),
                      "label": "flat-slice gauge continued on [1.7000, 2.9000]: min U/U_ref = %.3e (auxiliary signed U, the energy deficit's bookkeeping); lapse unresolved" % (Uf.min() / Umax)})
    if kind == "near":
        gauge.update({"near_tangencies": 1, "stretches": 1, "segments": 3})
    info = {"failed": None, "module": "engine_fields_v16", "centre_flag": (None if regular else "centre outside the regular series (rho_0 <= 0)"),
            "centre_series_check": {"rho_spline_minus_series_rel": 1e-9, "J_spline_minus_series_rel": 1e-9, "regular_start": np.bool_(regular)},
            "ivp": {"r0": ell / 64.0, "U0": float(U(ell / 64.0)), "lnN0": 0.0, "lnN_collar": 0.1, "N_centre": 0.9, "nfev": 1000, "segments": gauge["segments"], "series": cc},
            "gauge": gauge, "regime": {"max_ratio": 0.1, "r_at_max": 2.0, "intervals_above_1": [], "label": "in the regime"}, "rfine": rf, "U_dense": U, "W_field": W, "W_geom": Wg,
            "N_dense": N, "v_dense": v, "rho_ell": rho, "J_ell": J, "E_collar": 59.7, "r_collar": r_collar, "residual_W": 1e-10, "residual_invN": 1e-10, "N_min": 0.7, "ell": ell}
    if kind in ("centre", "away"):
        Wf = W(rf); k = int(np.argmin(Wf))
        info["W_negative"] = {"min_W_over_max": float(Wf[k] / np.max(np.abs(Wf))), "r": float(rf[k]), "inside_gauge_intervals": np.bool_(True)}
    return info


def test_read_flags(o):
    g = 0.25
    R_first = np.array([1.0, 2.0, 3.0, 4.0])
    # the regular solve
    info = mk_info("regular")
    rec, arrs = DF.read_flags(info, g, R_first)
    check("regular: regular True", rec["regular"] is True, rec["regular_conditions"])
    check("regular: every condition True", all(v is True for v in rec["regular_conditions"].values()), rec["regular_conditions"])
    check("regular: left_first None", rec["left_first"] is None)
    check("regular: no interval", rec["gauge"]["first_interval_start"] is None and rec["gauge"]["n_intervals"] == 0)
    check("regular: no expected key missing", rec["info_keys_missing"] == [], rec["info_keys_missing"])
    W1 = float(info["W_field"](np.array([1.0]))[0]) / (2.0 * g)
    check("regular: W/2g at the first site", abs(rec["first_sites"]["W_over_2g"][0] - W1) <= o.tol * max(1.0, abs(W1)), "%r vs %r" % (rec["first_sites"]["W_over_2g"][0], W1))
    check("regular: rho_ell(0) equals the series rho0", abs(rec["rho_ell_0"] - rec["series"]["rho0"]) <= o.tol)
    check("regular: J_ell(0) zero and N_dense(0) read", rec["J_ell_0"] == 0.0 and abs(rec["N_dense_0"] - 1.0) <= o.tol)
    check("regular: U positive on the fine grid", rec["U_fine"]["positive"] is True and rec["U_fine"]["n_nonpositive"] == 0, rec["U_fine"])
    check("regular: the arrays", set(arrs) >= {"rfine", "U_fine", "W_fine", "N_fine", "first_sites"} and arrs["first_sites"].shape == (7, 4) and len(arrs["rfine"]) == len(arrs["U_fine"]))
    check("regular: numpy types converted (regular_start, lapse_unresolved, series regular)",
          type(rec["centre_series_check"]["regular_start"]) is bool and type(rec["gauge"]["lapse_unresolved"]) is bool and type(rec["series"]["regular"]) is bool)
    try:
        json.loads(json.dumps(rec, default=DF._jd))
        check("regular: json round trip", True)
    except Exception as ex:
        check("regular: json round trip", False, str(ex))
    # the centre-flagged continuation
    rec, _ = DF.read_flags(mk_info("centre"), g, R_first)
    c = rec["regular_conditions"]
    check("centre: regular False", rec["regular"] is False)
    check("centre: left_first centre", rec["left_first"] == "centre", rec["left_first"])
    check("centre: the first interval from 0 to 3.1", rec["gauge"]["first_interval_start"] == 0.0 and rec["gauge"]["first_interval_end"] == 3.1)
    check("centre: conditions (centre_flag_none, regular_start, rho0_positive, no_intervals, label_intact, lapse_resolved, W_not_negative) False and not_failed True",
          c["centre_flag_none"] is False and c["regular_start"] is False and c["rho0_positive"] is False and c["no_intervals"] is False and c["label_intact"] is False
          and c["lapse_resolved"] is False and c["W_not_negative"] is False and c["not_failed"] is True, c)
    check("centre: W_negative parsed", rec["W_negative"] is not None and rec["W_negative"]["min_W_over_max"] < 0.0 and type(rec["W_negative"]["inside_gauge_intervals"]) is bool, rec["W_negative"])
    check("centre: U_fine not positive", rec["U_fine"]["positive"] is False and rec["U_fine"]["n_nonpositive"] > 0)
    check("centre: first_zero the upward one", rec["first_zero"] is not None and rec["first_zero"]["kind"] == "up" and rec["first_zero"]["r"] == 3.1, rec["first_zero"])
    # the continuation from a downward zero away from the centre
    rec, _ = DF.read_flags(mk_info("away"), g, R_first)
    c = rec["regular_conditions"]
    check("away: regular False", rec["regular"] is False)
    check("away: left_first away", rec["left_first"] == "away", rec["left_first"])
    check("away: the first interval [1.7, 2.9]", rec["gauge"]["first_interval_start"] == 1.7 and rec["gauge"]["first_interval_end"] == 2.9)
    check("away: centre conditions True, interval conditions False", c["centre_flag_none"] is True and c["regular_start"] is True and c["rho0_positive"] is True and c["no_intervals"] is False
          and c["label_intact"] is False, c)
    check("away: W_geom zero inside the interval, W_field negative there", rec["first_sites"]["W_geom_over_2g"][1] == 0.0 and rec["first_sites"]["W_over_2g"][1] < 0.0, rec["first_sites"])
    check("away: v_dense zero inside the interval", rec["first_sites"]["v_dense"][1] == 0.0)
    # the failed integration
    rec, arrs = DF.read_flags(mk_info("failed"), g, R_first)
    check("failed: regular False, not_failed False", rec["regular"] is False and rec["regular_conditions"]["not_failed"] is False)
    check("failed: left_first failed", rec["left_first"] == "failed")
    check("failed: the failure string kept", isinstance(rec["failed"], str) and rec["failed"].startswith("integration failed"))
    check("failed: gauge and ivp missing recorded", "gauge" in rec["info_keys_missing"] and "ivp" in rec["info_keys_missing"], rec["info_keys_missing"])
    check("failed: rho0_positive from regular_start when the series is absent", rec["regular_conditions"]["rho0_positive"] is True and rec["series"] is None)
    check("failed: W_negative None recorded, no arrays", rec["W_negative"] is None and arrs == {} and rec["first_sites"] is None)
    # the near-tangency with the label intact
    rec, _ = DF.read_flags(mk_info("near"), g, R_first)
    check("near: regular True with near_tangencies 1 recorded", rec["regular"] is True and rec["gauge"]["near_tangencies"] == 1 and rec["gauge"]["stretches"] == 1)
    # the intact label with a negative fine-grid U
    rec, _ = DF.read_flags(mk_info("inconsistent"), g, R_first)
    check("inconsistent: regular False through U_positive_on_fine_grid only", rec["regular"] is False and rec["regular_conditions"]["U_positive_on_fine_grid"] is False
          and all(v is True for k, v in rec["regular_conditions"].items() if k != "U_positive_on_fine_grid"), rec["regular_conditions"])
    check("inconsistent: left_first other", rec["left_first"] == "other")
    # no info
    rec, arrs = DF.read_flags(None, g, R_first)
    check("none: info_present False, regular False, left_first 'no solve', every key missing", rec["info_present"] is False and rec["regular"] is False and rec["left_first"] == "no solve"
          and set(rec["info_keys_missing"]) == set(DF.EXPECTED_INFO_KEYS) and arrs == {})
    # the first sites from the solve's own chain when R_first is None
    info = mk_info("regular")
    info["src"] = type("Src", (), {"R": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])})()
    rec, _ = DF.read_flags(info, g, None, n_first=3)
    check("first sites from info['src'].R when none is given", rec["first_sites"] is not None and rec["first_sites"]["R"] == [1.0, 2.0, 3.0])
    # _call1 on a scalar-only callable
    vals = DF._call1(lambda r: float(r) ** 2, np.array([1.0, 2.0, 3.0]))
    check("_call1 falls back to scalar calls", np.allclose(vals, [1.0, 4.0, 9.0]))
    check("_call1 on None gives nan", np.all(np.isnan(DF._call1(None, [1.0, 2.0]))))


# ------------------------------------------------------------------------------------------------------- the chord bisection
def mk_eval(flag_fn, flag2_fn=None, dN0_a=-0.01075, dN0_b=-0.00513, counter=None):
    def evaluate(t):
        if counter is not None:
            counter.append(t)
        f = flag_fn(t)
        f2 = flag2_fn(t) if flag2_fn is not None else f
        return {"key": "p%d" % (len(counter) if counter is not None else 0), "done": True, "kind": "chord point", "t": float(t), "dN0": dN0_a + t * (dN0_b - dN0_a),
                "N_app_0": 0.77 + t * 0.005, "regular": f, "left_first": (None if f else "centre"), "regular_conditions": {"rho0_positive": f2, "no_intervals": f},
                "full_inf": 1e-4 * (1 + t), "perp_inf": 1e-6, "valid_perp": True, "series": {"rho0": (1e-5 if f2 else -1e-5)}, "gauge": {"first_interval_start": (None if f else 0.0),
                "first_interval_end": (None if f else 2.3), "n_intervals": (0 if f else 1)}, "central_cell": {"W_over_2g_at_1": (1e-5 if f else -1e-5)}}
    return evaluate


def test_bisection(o):
    rng = np.random.RandomState(o.seed)
    k = int(o.bisection_steps); target = 2.0 ** (-k)
    flag_of = lambda p: DF.flag_value(p, "regular")
    ok_all = True; detail = ""
    for _ in range(int(o.n_random)):
        ts = float(rng.uniform(0.02, 0.98))
        pool = []; cnt = []
        lo, hi, n_new, note = DF.bisect_chord(pool, True, False, flag_of, target, k, mk_eval(lambda t, ts=ts: t < ts, counter=cnt))
        good = (lo < ts <= hi + 1e-15) and abs((hi - lo) - target) <= 1e-15 and n_new == k and len(pool) == k and note is None and len(cnt) == k
        if not good:
            ok_all = False; detail = "t* %.6f: bracket [%.6f, %.6f], n_new %d, note %r" % (ts, lo, hi, n_new, note)
            break
    check("monotone switch: the bracket contains t*, width 2^-k after k new evaluations (%d random t*)" % o.n_random, ok_all, detail)
    # the reversed orientation: t = 0 continued, t = 1 regular
    ts = 0.37; pool = []
    lo, hi, n_new, note = DF.bisect_chord(pool, False, True, flag_of, target, k, mk_eval(lambda t: t >= ts))
    check("reversed orientation: the bracket contains t*", lo < ts <= hi + 1e-15 and abs((hi - lo) - target) <= 1e-15 and n_new == k, "[%.6f, %.6f]" % (lo, hi))
    # the resume: the first 3 points of a fresh run, then the continuation gives the same points and bracket
    cnt_fresh = []; pool_fresh = []
    lo_f, hi_f, _, _ = DF.bisect_chord(pool_fresh, True, False, flag_of, target, k, mk_eval(lambda t: t < ts, counter=cnt_fresh))
    pool_part = [dict(p) for p in pool_fresh[:3]]; cnt_res = []
    lo_r, hi_r, n_res, _ = DF.bisect_chord(pool_part, True, False, flag_of, target, k, mk_eval(lambda t: t < ts, counter=cnt_res))
    check("resume from a partial pool: the same bracket and the same remaining points", (lo_r, hi_r) == (lo_f, hi_f) and n_res == k - 3 and cnt_res == cnt_fresh[3:],
          "fresh [%.6f, %.6f] resumed [%.6f, %.6f], %d new" % (lo_f, hi_f, lo_r, hi_r, n_res))
    # the non-monotone flag: the first switch from the lower end
    pool = []
    lo, hi, n_new, note = DF.bisect_chord(pool, True, False, flag_of, target, k, mk_eval(lambda t: (t < 0.3) or (0.5 < t < 0.7)))
    check("non-monotone flag: the first switch from the lower end (0.3) is bracketed, the ends' flags differ", lo < 0.3 <= hi + 1e-15 and abs((hi - lo) - target) <= 1e-15, "[%.6f, %.6f]" % (lo, hi))
    # an unreadable midpoint: the bisection stops instead of repeating
    pool = []

    def ev_none(t):
        rec = mk_eval(lambda q: q < 0.37)(t)
        rec["regular"] = None
        return rec
    lo, hi, n_new, note = DF.bisect_chord(pool, True, False, flag_of, target, k, ev_none)
    check("unreadable midpoint: one evaluation, the bracket unchanged, a note", n_new == 1 and (lo, hi) == (0.0, 1.0) and note is not None and "did not narrow" in note, note)
    # a second flag on the shared pool: fewer new evaluations
    pool = []; cnt1 = []; cnt2 = []
    flag2_of = lambda p: DF.flag_value(p, "rho0_positive")
    ev = mk_eval(lambda t: t < 0.37, flag2_fn=lambda t: t < 0.2, counter=cnt1)
    lo1, hi1, n1, _ = DF.bisect_chord(pool, True, False, flag_of, target, k, ev)
    ev2 = mk_eval(lambda t: t < 0.37, flag2_fn=lambda t: t < 0.2, counter=cnt2)
    lo2, hi2, n2, note2 = DF.bisect_chord(pool, True, False, flag2_of, target, k, ev2)
    check("second flag on the shared pool: its switch (0.2) bracketed with fewer new evaluations than k", lo2 < 0.2 <= hi2 + 1e-15 and (hi2 - lo2) <= target + 1e-15 and 0 < n2 < k and len(pool) == n1 + n2,
          "flag 1 [%.6f, %.6f] %d new; flag 2 [%.6f, %.6f] %d new (pool %d)" % (lo1, hi1, n1, lo2, hi2, n2, len(pool)))
    # bracket_from_pool: points with an unreadable flag skipped
    pool = [{"t": 0.5, "regular": None}, {"t": 0.25, "regular": True}, {"t": 0.75, "regular": False}]
    lo, hi, lo_rec, hi_rec = DF.bracket_from_pool(pool, True, False, flag_of)
    check("bracket_from_pool skips unreadable flags and returns the end records", (lo, hi) == (0.25, 0.75) and lo_rec is pool[1] and hi_rec is pool[2])
    # the midpoint exactly at target width: no evaluation when the pool already brackets to the target
    pool = [{"t": 0.5, "regular": True}, {"t": 0.5 + target, "regular": False}]
    lo, hi, n_new, note = DF.bisect_chord(pool, True, False, flag_of, target, k, mk_eval(lambda t: t < 0.5 + target / 2))
    check("a pool already at the target width needs no new evaluation", n_new == 0 and note is None and abs((hi - lo) - target) <= 1e-15)


# ------------------------------------------------------------------------------------------------------- the helpers and the smoke test
def test_helpers(o):
    zc = DF.zero_crossings([(-0.02, -1.0), (0.01, 1.0), (-0.01, -0.5), (0.0, 0.5), (0.02, None)])
    check("zero_crossings: one crossing at -0.005 with None skipped", len(zc) == 1 and abs(zc[0]["dN0"] + 0.005) <= o.tol and zc[0]["between"] == [-0.01, 0.0], zc)
    zc = DF.zero_crossings([(0.0, 1.0), (0.01, 0.0), (0.02, -1.0)])
    check("zero_crossings: a zero value at a point gives the crossing at that point", len(zc) == 1 and abs(zc[0]["dN0"] - 0.01) <= o.tol, zc)
    check("single_switch", DF.single_switch([False, False, True, True, None]) and not DF.single_switch([True, False, True]) and DF.single_switch([]))
    grid = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3]
    check("index_to_line", DF.index_to_line(0, grid) == ("0", 0.0) and DF.index_to_line(1, grid) == ("+", 0.005) and DF.index_to_line(7, grid) == ("+", 0.3)
          and DF.index_to_line(8, grid) == ("-", -0.005) and DF.index_to_line(14, grid) == ("-", -0.3) and DF.index_to_line(15, grid) == (None, None))
    # switching pairs and the plan
    pts = []
    for key, d, reg, rho in (("xc_14", -0.29, False, False), ("xc_9", -0.011, False, False), ("xc_8", -0.005, True, True), ("xc_0", -2e-5, True, True), ("x_floor", 0.0, True, True), ("xc_1", 0.005, True, True),
                             ("xc_7", 0.30, True, True)):
        pts.append({"key": key, "done": True, "kind": "stored point", "dN0": d, "N_app_0": 0.77 + d, "regular": reg, "regular_conditions": {"rho0_positive": rho}})
    P = {"points": {p["key"]: p for p in pts}}
    sp = DF.switching_pairs(sorted(pts, key=lambda q: q["dN0"]), "regular")
    check("switching_pairs: one pair (xc_9, xc_8) in dN0 order", sp == [("xc_9", "xc_8")], sp)
    todo, skipped = DF.plan_chords(P, ["regular", "rho0_positive"], 4)
    check("plan_chords: both flags on the one chord, nothing skipped", todo == [("regular", "xc_9", "xc_8"), ("rho0_positive", "xc_9", "xc_8")] and skipped == [], (todo, skipped))
    P["points"]["xc_7"]["regular"] = False
    todo, skipped = DF.plan_chords(P, ["regular"], 1)
    check("plan_chords: the cap keeps the chord nearest dN0 = 0", todo == [("regular", "xc_9", "xc_8")] and skipped == [("regular", "xc_1", "xc_7")], (todo, skipped))
    # the smoke test of summarize_s and write_md on a synthetic result
    g = 0.25
    R_first = np.array([1.0, 2.0, 3.0, 4.0])

    def mk_rec(kind, key, dN0, full, t=None):
        rec, _ = DF.read_flags(mk_info(kind), g, R_first)
        rec.update({"key": key, "done": True, "kind": ("chord point" if t is not None else "stored point"), "dN0": dN0, "N_app_0": 0.7696 + dN0, "v_app_0": -0.0036, "full_inf": full,
                    "soft_N0": -full, "soft_coeff": -full / 0.6, "perp_inf": 1e-6, "valid_perp": True, "matches_scan": True, "E": 59.747, "seconds": 12.0,
                    "central_cell": {"N_dense_0": 0.766, "W_over_2g_at_1": (1.66e-5 if rec["regular"] else -1.2e-5), "W_over_2g_at_2": 1e-4, "J_over_v_at_1": (None if not rec["regular"] else 1e-3)}})
        if t is not None:
            rec["t"] = t
        return rec
    P = {"s": 0.0, "map": "the uncoupled map (synthetic)", "points": {}, "order": [], "chords": [], "floor": {"kept_scan": 1.88e-4, "kept_from": "direct", "source": "synthetic", "residual_reproduced": 1.88e-4,
         "relative_difference": 0.0, "reproduced": True, "regular": True, "state_equals_gate": {"checked": False, "why": "synthetic"}}}
    for kind, key, d, full in (("regular", "x_floor", 0.0, 1.88e-4), ("regular", "xc_0", -1.9e-5, 1.62e-4), ("regular", "xc_1", 0.0051, 6.4e-4), ("centre", "xc_8", -0.00525, 1.75e-3), ("centre", "xc_9", -0.0105, 2.4e-3),
                              ("centre", "xpol", -0.024, 6.3e-9)):
        P["points"][key] = mk_rec(kind, key, d, full); P["order"].append(key)
    todo, skipped = DF.plan_chords(P, ["regular", "rho0_positive"], 4)
    check("smoke: the plan from read_flags records", todo == [("regular", "xc_8", "xc_0"), ("rho0_positive", "xc_8", "xc_0")], todo)
    ch = DF.chord_entry(P, "xc_8", "xc_0")
    pa, pb = P["points"]["xc_8"], P["points"]["xc_0"]
    target = 2.0 ** (-int(o.bisection_steps))

    def evaluate(t):
        kind = "regular" if t > 0.4 else "centre"
        return mk_rec(kind, "c0_p%d" % len(ch["points"]), pa["dN0"] + t * (pb["dN0"] - pa["dN0"]), 1e-3 * (1.0 - 0.5 * t), t=t)
    for fname in ("regular", "rho0_positive"):
        fl_lo, fl_hi = DF.flag_value(pa, fname), DF.flag_value(pb, fname)
        lo, hi, n_new, note = DF.bisect_chord(ch["points"], fl_lo, fl_hi, lambda p, fname=fname: DF.flag_value(p, fname), target, int(o.bisection_steps), evaluate)
        ch["flags"][fname] = DF.bracket_summary(ch, fname, fl_lo, fl_hi, lo, hi, pa, pb, target, n_new, note, int(o.bisection_steps))
    Fr = ch["flags"]["regular"]
    check("smoke: the regular bracket contains the synthetic switch t = 0.4 and reached the target", Fr["bracket_t"][0] < 0.4 <= Fr["bracket_t"][1] + 1e-15 and Fr["reached_target"], Fr["bracket_t"])
    check("smoke: the rho0 bracket reused the pool (0 new evaluations)", ch["flags"]["rho0_positive"]["new_evaluations"] == 0 and ch["flags"]["rho0_positive"]["bracket_t"] == Fr["bracket_t"])
    P["summary"] = DF.summarize_s(P)
    Sm = P["summary"]
    check("smoke: summarize_s counts", Sm["n_stored"] == 6 and Sm["n_chord_points"] == len(ch["points"]) and Sm["n_regular"] + Sm["n_continued"] == Sm["n_evaluated"], (Sm["n_stored"], Sm["n_chord_points"], Sm["n_evaluated"]))
    check("smoke: the verdict 'centre', together within resolution", Sm["leaves_first"]["verdicts"] == ["centre"] and "leave together" in Sm["leaves_first"]["text"][0], Sm["leaves_first"]["text"])
    check("smoke: best regular is the corrected floor, the ratio to the kept below 1", Sm["best_regular"]["key"] == "xc_0" and Sm["best_regular_is_floor"] and Sm["best_regular_over_floor_kept"] < 1.0)
    check("smoke: the W/2g(1) crossing lies in the regular bracket's dN0", len(Sm["W_over_2g_at_1_zero_crossings_dN0"]) == 1
          and min(Fr["bracket_dN0"]) - 1e-12 <= Sm["W_over_2g_at_1_zero_crossings_dN0"][0]["dN0"] <= max(Fr["bracket_dN0"]) + 1e-12, (Sm["W_over_2g_at_1_zero_crossings_dN0"], Fr["bracket_dN0"]))
    RES = {"script": "selftest", "args": {"bisection_steps": o.bisection_steps, "bisection_flags": "regular,rho0_positive", "max_chords": 4, "fields_module": "engine_fields_v16", "floor_rel_tol": 1e-6,
                                           "residual_rel_tol": 1e-6, "perp_ok": 1e-5, "first_sites": 4}, "scan": {"json": "synthetic", "state": "synthetic", "engine_args_match": None},
           "setup": {"engine_file": "synthetic", "R_first_sites": R_first.tolist()}, "per_s": {"0": P, "1": {"s": 1.0, "skipped": "synthetic"}}, "counts": {"evaluations": 6, "seconds_in_evaluations": 70.0, "seconds_total": 90.0},
           "memory": {"peak_gib": None}, "status": "done"}
    d = tempfile.mkdtemp(prefix="domain_flags_selftest_")
    pmd = os.path.join(d, "C_s2_domain_flags_v1.md"); pj = os.path.join(d, "C_s2_domain_flags_v1.json")
    try:
        DF.write_md(RES, pmd)
        txt = open(pmd).read()
        check("smoke: write_md runs and carries the tables and the reading", "### the stored points" in txt and "### chord 0" in txt and "### the reading, s = 0" in txt and "the centre's density leaves first" in txt
              and "skipped: synthetic" in txt)
        with open(pj, "w") as fh:
            json.dump(RES, fh, default=DF._jd)
        check("smoke: the result dumps to json", os.path.getsize(pj) > 0)
    except Exception as ex:
        check("smoke: write_md/json", False, repr(ex))
    check("no date in the report", all(w not in txt for w in ("2024", "2025", "2026", "202", "written")) if "txt" in dir() else False)


def main():
    ap = argparse.ArgumentParser(description="manufactured checks of C_s2_domain_flags_v1 (no engine)")
    ap.add_argument("--bisection-steps", type=int, default=6); ap.add_argument("--n-random", type=int, default=25); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=1e-12, help="tolerance of the equality checks")
    o = ap.parse_args()
    print("C_s2_domain_flags_selftest_v1: flag parsing", flush=True)
    test_read_flags(o)
    print("C_s2_domain_flags_selftest_v1: the chord bisection", flush=True)
    test_bisection(o)
    print("C_s2_domain_flags_selftest_v1: the helpers and the smoke test", flush=True)
    test_helpers(o)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("%s: %d checks, %d failed" % ("SELFTEST PASS" if n_fail == 0 else "SELFTEST FAIL", len(RESULTS), n_fail), flush=True)
    for name, ok, detail in RESULTS:
        if not ok:
            print("  FAILED: %s: %s" % (name, detail), flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
