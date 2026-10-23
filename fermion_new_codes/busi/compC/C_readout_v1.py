#!/usr/bin/env python
"""C_readout_v1.py (the busi lane, Computation C): the readouts of C_two_field_spec_v1.md sec. 7 that need only the stored records
(2, 3, 4, 5, 7 and the "also printed" list), for the sigma/a 4 member alone; CPU (at most 8 threads), minutes, no figure.

Arguments are the paths: --run-json (the coupling-on run's record json, engine_twofield_v3), --s11-npz (its _s11 npz), --follow-json
(gate F's light-follow on the 1-field production's applied fields), --prod-json (the 1-field production, engine_collapse_v31, the solve
lane's, read in place). --out-dir and --tag name the outputs <out-dir>/<tag>.json (every per-record series the md prints, so a later
figure loads it) and <out-dir>/<tag>.md (tables at T/r_s 0, the formation, 16, 20, 24, 28, 32, 36, 40 and one summary line per readout).
--slice-dir (optional) only lists which slice clocks exist there, for readouts 1, 6 and 8, which are not built here.

The mapping of the spec's names onto the stored keys (the digest C_records_digest_v1.json). Every quantity is read from these keys and
nothing else; a key the spec names that is absent stops the readout with MissingKey naming it (nothing is substituted). The keys the
engine writes from the second record on (dE_dT, dHin_dT, balance_d_E_minus_Hin_dT, M1_balance, light_v36.M1_balance,
light_v36.dE_inv_light_dT) may be absent at T = 0 only.
  2  E, E_over_E0, dE_dT, M1_I_E (= M1_parts.I_E_start, the heavy collar cut current), light_v36.I_E_light (all channels, weighted),
     light_v36.I_E_light_sourcing (the guard's current is their difference, = channels.3.I_E_cut_weighted). The spec's
     "channels.<k>.I_E_cut" is NOT a stored key of the coupling-on json: the per-channel current is stored weighted (I_E_cut_weighted)
     beside w; the readout prints I_E_cut_weighted and, labelled derived, I_E_cut_weighted / w. The spec's M1_parts_light and
     M1_balance_light are the stored light_v36.M1_parts and light_v36.M1_balance. "-Delta W" is read as -Delta E, E being W/2g at the
     collar (solver.W_tail.E_collar == E is checked on every record). Integrals are trapezoids over the record clock T; the integral of
     dE_dT uses the backward rule, the definition of dE_dT being read off the records (hypothesis test, both alternatives printed).
  3  R_t, r_in, trapping_zeros, min_N_minus_absv, r_min_N_minus_absv, T_over_rs, E of both runs; the light-follow's E_rad0_over_E
     (= E_inv_light_sourcing / E there, checked); light_v36.E_inv_light_over_E and E_inv_light_sourcing_over_E of the coupling-on run.
     A = 4 pi R_t^2 (the column spec item 8 asks this script to print) is computed from R_t.
  4  s11.{M_min_outside_cell, r_at_M_min, negative_points_outside_cell, source_min_outside_cell, ball_source_min_outside_cell (the heavy
     dust's), pairs_source_min_outside_cell, light_source_min_outside_cell, light_source_integral, light_fed, light_certified_on_record,
     rho_ell_min_outside_cell}; the s11 npz (T, r_nodes, M, rho_ell) cross-checks the minima over r > ell. "The stop with its case": no
     stop key exists in the digest's records (the run ended at its clock); every s11 or record key containing "stop"/"case" is reported
     per record and otherwise the absence is stated, never filled.
  5  light_v36.channels.<k>.{E_inv, eps_sum, J_sum, nocc_deviation_bins, beyond.<collar>.{number, energy_density_route}, mu, gap,
     certified, fallback, cut_crossing} against the light-follow's channels.<k> (the same unweighted names; its I_E_cut and H_in are
     unweighted and are set beside the coupling-on run's weighted / w); the totals light_v36.E_inv_light (all) and E_inv_light_sourcing
     against the follow's E_inv_light_all and E_inv_light_sourcing; E_inv_light_sourcing_over_E against E_rad0_over_E; I_E_light,
     I_E_light_sourcing against I_E_light_all, I_E_light_sourcing; eps_fed_sum, J_fed_sum against the same names. Computation A's own
     readout is the lead's and is not among the inputs; only the engine-side first order is printed.
  7  M1_parts {X1, X2, Y1, Y2, I_E_end_prev_interval, I_E_start, cut, I_E_mid}, M1_balance {dHin, evolution_exact, evolution_trapezoid,
     field_work, reference_reset, closure, per_unit_time} (heavy, version 35's); light_v36.M1_parts, light_v36.M1_balance (the light's);
     light_v36.E_inv_heavy, light_v36.E_inv_light, H_in, light_v36.H_in_light, light_v36.dE_inv_light_dT.
  also  head.light_v36.subcritical_T0 {min_N_minus_absv_sites, min_N_minus_absv_dense, r_min, zeros}, head.light_v36.first_return (all
     entries), head.light_v36.chain_end_over_rs; light_v36.truncation_guard {E_inv_guard_over_sourcing, I_E_guard_over_sourcing,
     eps_guard_over_sourcing} (recomputed from the channel entries); light_v36.channels.<k>.certified {pieces, cholesky_calls,
     certified, why, d_min_certified, norm_dK_G}, channels.<k>.{fallback, why, cut_crossing, mode}, light_v36.label,
     light_v36.certified_on_record, light_v36.timing {walk_pieces, walk_cholesky, walk_fallbacks, cut_crossings} (the record's totals
     over its evaluations); light_v36.threshold_radius_massless.<k> and channels.<k>.threshold_radius.
Identities among the stored keys are recomputed and their residuals printed: the exact ones at roundoff, the relations read off the
records (dE_dT = (E_k - E_{k-1})/tau, evolution_trapezoid's form, field_work = (Y2 - Y1) - (X2 - X1)) named as hypotheses or relations
with their alternatives' residuals, the cross-checks between files as such.
Not read: the fields npz, the spectra npz, the handoff npz and the light-follow's npz (not needed by readouts 2, 3, 4, 5, 7)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import datetime
import glob
import hashlib
import json
import math
import re
import sys

import numpy as np

VERSION = "C_readout_v1"
LABEL = ("Computation C, the self-consistent 2-field collapse (D16): the heavy field the production's dust at r_s/a 32, the light field the "
         "massless 4-component Dirac field of the same operator with multiplicity 16 kappa, kappa 1 and 2 sourcing, kappa 3 the truncation "
         "guard; the light reference 1(h_m - mu S < 0) continued by the certified walk (fallback marked); the chain of 2048 sites, first "
         "return near T/r_s 115")
TABLE_CLOCKS = (0.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0)
EPS = float(np.finfo(float).eps)
_ABSENT = object()

C_REC_ALL = (("T",), ("T_over_rs",), ("E",), ("E_over_E0",), ("E_dust_shells",), ("E_pairs_shells",), ("R_t",), ("r_in",),
             ("trapping_zeros",), ("min_N_minus_absv",), ("r_min_N_minus_absv",), ("M1_I_E",), ("H_in",))
C_REC_AFTER0 = (("dE_dT",), ("dHin_dT",), ("balance_d_E_minus_Hin_dT",))
C_REC_OPT = (("continuation_label",), ("R_c",), ("N_min",), ("R_sh",), ("slice_saved",), ("r_collar",), ("converged",))
M1_PARTS_KEYS = ("X1", "X2", "Y1", "Y2", "I_E_end_prev_interval", "I_E_start", "cut")
M1_BAL_KEYS = ("dHin", "evolution_exact", "evolution_trapezoid", "field_work", "reference_reset", "closure")
M1_PUT_KEYS = ("dHin", "evolution", "field_work", "reference_reset")
M1_PUT_FROM = ("dHin", "evolution_exact", "field_work", "reference_reset")
S11_REQ = ("T_over_rs", "E", "M_min_outside_cell", "r_at_M_min", "negative_points_outside_cell", "source_min_outside_cell",
           "ball_source_min_outside_cell", "pairs_source_min_outside_cell", "light_source_min_outside_cell", "light_source_integral",
           "pairs_source_integral", "ball_source_integral", "rho_ell_min_outside_cell", "light_fed", "light_certified_on_record")
S11_OPT = ("r_at_source_min", "compactness_max_2gM_over_r", "N_centre", "U_phys_min_outside_cell", "W_over_2g_min_outside_cell",
           "J_ell_max_outside_cell", "positive_branch_v_residual_max", "continuation_label", "light_smoothing_error")
LV_REQ = ("E_inv_light", "E_inv_light_sourcing", "E_inv_light_over_E", "E_inv_light_sourcing_over_E", "E_inv_heavy", "I_E_light",
          "I_E_light_sourcing", "H_in_light", "eps_fed_sum", "J_fed_sum", "eps_light_all_sum", "label", "certified_on_record", "fed",
          "coupling", "transport")
LV_TG = ("E_inv_guard_over_sourcing", "I_E_guard_over_sourcing", "eps_guard_over_sourcing")
TIMING_OPT = ("walk_pieces", "walk_cholesky", "walk_fallbacks", "cut_crossings")
CH_REQ_C = ("sourcing", "w", "E_inv", "E_sea", "E_ref", "E_inv_weighted", "I_E_cut_weighted", "H_in_weighted", "eps_sum", "J_sum",
            "eps_sum_weighted", "mu", "gap", "n_plus", "n_pocket", "pockets_occupied_positive", "holes_empty_negative", "fallback", "why",
            "cut_crossing", "mode", "min_abs_e", "threshold_radius")
CH_REQ_F = ("sourcing", "E_inv", "E_sea", "E_ref", "I_E_cut", "H_in", "eps_sum", "J_sum", "mu", "gap", "n_plus", "n_pocket",
            "pockets_occupied_positive", "holes_empty_negative", "fallback", "why", "cut_crossing", "mode", "min_abs_e")
CH_OPT = ("reanchored", "verified", "inversions_sector", "sigma_impurity_max", "min_occupied_weight", "max_empty_weight", "margin_min",
          "norm_dH_G", "mask_equals_negative_set", "rotated_blocks", "nsub", "route")
CERT_REQ = ("pieces", "cholesky_calls", "certified", "why", "d_min_certified", "norm_dK_G")
CERT_OPT = ("smallest_piece", "d_min_effective", "d_floor", "rounding_F", "norm_K_G", "gap_start", "mu0", "mu1", "seconds")
NOCC_BINS = ("(-inf,-1)", "(-1,0)", "(0,1)", "(1,inf)")
F_REC_ALL = (("T",), ("T_over_rs",), ("E",), ("R_t",), ("min_N_minus_absv",))
F_TOT_REQ = ("E_inv_light_sourcing", "E_inv_light_all", "E_rad0_over_E", "I_E_light_sourcing", "I_E_light_all", "eps_fed_sum", "J_fed_sum")
F_TG = ("E_inv_guard_over_sourcing", "I_E_guard_over_sourcing")
P_REC_ALL = (("T",), ("T_over_rs",), ("E",), ("E_over_E0",), ("R_t",), ("r_in",), ("trapping_zeros",), ("min_N_minus_absv",),
             ("r_min_N_minus_absv",), ("H_in",))
P_REC_AFTER0 = (("dE_dT",),)
P_REC_OPT = (("R_c",), ("N_min",), ("continuation_label",), ("M1_I_E",))
ARGS_SKIP = ("out", "tag", "gpu", "slice_dir", "slices_c128", "slices_c64", "slices_light_c128", "resume_from", "resume_T_over_rs",
             "resume_grandparent", "resume_slice", "resume_run", "benchmark", "streams", "light_follow")


# ----------------------------------------------------------------------------------------------------------------------------- errors
class ReadoutError(Exception):
    """A loud stop of the readout: the data are not what the spec names."""


class MissingKey(ReadoutError):
    """A key the spec names is absent: the readout that needs it stops; nothing is substituted."""


class Inconsistent(ReadoutError):
    """Stored entries contradict each other (shapes, flags, channel sets)."""


# ---------------------------------------------------------------------------------------------------------------------------- helpers
def need(d, path, where):
    cur = d
    for i, p in enumerate(path):
        if not isinstance(cur, dict) or p not in cur:
            raise MissingKey("%s: the key '%s' is absent (at '%s'); the readout that needs it stops, nothing is substituted"
                             % (where, ".".join(str(q) for q in path), ".".join(str(q) for q in path[:i + 1])))
        cur = cur[p]
    return cur


def opt(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def need_after0(d, k, path, where):
    """Keys written from the second record on (no previous interval at T = 0): None at k = 0 when absent, required later."""
    if k == 0 and opt(d, path, _ABSENT) is _ABSENT:
        return None
    return need(d, path, where)


def is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def av(*xs):
    """True when every value is a finite or infinite number (not None, not a bool, not nan)."""
    for x in xs:
        if not is_num(x):
            return False
        if isinstance(x, (float, np.floating)) and math.isnan(float(x)):
            return False
    return True


def ratio(a, b):
    if not av(a, b) or b == 0:
        return None
    return float(a) / float(b)


def rel(a, b):
    return ratio(a - b, b) if av(a, b) else None


def maxabs(lst):
    vals = [abs(float(x)) for x in lst if av(x)]
    return max(vals) if vals else None


def argmaxabs(lst):
    best, kb = None, None
    for k, x in enumerate(lst):
        if av(x) and (best is None or abs(x) > best):
            best, kb = abs(float(x)), k
    return kb


def split_list(s):
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ckey(t):
    return int(round(float(t) * 1e6))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fm(x, p=8):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return ("%%.%dg" % p) % x
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fm(v, min(p, 6)) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join("%s: %s" % (k, fm(v, min(p, 6))) for k, v in x.items()) + "}"
    return str(x)


def mdtab(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Series:
    def __init__(self, n):
        self.n = n
        self.d = {}

    def put(self, name, k, val):
        if name not in self.d:
            self.d[name] = [None] * self.n
        self.d[name][k] = val

    def __getitem__(self, name):
        return self.d[name]

    def __contains__(self, name):
        return name in self.d


class Identities:
    def __init__(self):
        self.d = {}
        self.order = []

    def add(self, name, residuals, kind="exact", note="", floor=None, extra=None):
        vals = []
        for r in residuals:
            if r is None:
                continue
            r = float(r)
            if not math.isnan(r):
                vals.append(r)
        rec = {"kind": kind, "n": len(vals), "max_abs_residual": (max(vals) if vals else 0.0), "note": note,
               "floor": (float(floor) if floor is not None else None)}
        if extra:
            rec.update(extra)
        self.d[name] = rec
        self.order.append(name)


# ------------------------------------------------------------------------------------------------------------------------- extraction
def extract_C(run, where):
    args = need(run, ("args",), where)
    head = need(run, ("head",), where)
    R = need(run, ("records",), where)
    if not isinstance(R, list) or not R:
        raise Inconsistent("%s: 'records' is not a non-empty list" % where)
    tau = float(need(args, ("tau",), where + ".args"))
    rs = float(need(args, ("rs",), where + ".args"))
    ell = float(need(args, ("ell",), where + ".args"))
    src = split_list(need(args, ("light_kappas",), where + ".args"))
    grd = split_list(need(args, ("light_guard_kappas",), where + ".args"))
    chans = src + grd
    if not chans:
        raise Inconsistent("%s: no light channels in the args" % where)
    collars = split_list(need(args, ("s11_collars",), where + ".args"))
    hl = need(head, ("light_v36",), where + ".head")
    for c in chans:
        need(hl, ("channels", c, "w"), where + ".head.light_v36")
        need(hl, ("channels", c, "kappa"), where + ".head.light_v36")
    n = len(R)
    S = Series(n)
    for k, r in enumerate(R):
        w = "%s.records[%d]" % (where, k)
        for p in C_REC_ALL:
            S.put(p[0], k, need(r, p, w))
        for p in C_REC_AFTER0:
            S.put(p[0], k, need_after0(r, k, p, w))
        for p in C_REC_OPT:
            S.put(p[0], k, opt(r, p))
        S.put("solver.W_tail.E_collar", k, opt(r, ("solver", "W_tail", "E_collar")))
        mp = need(r, ("M1_parts",), w)
        for q in M1_PARTS_KEYS:
            S.put("M1_parts." + q, k, need(mp, (q,), w + ".M1_parts"))
        S.put("M1_parts.I_E_mid", k, opt(mp, ("I_E_mid",)))
        mb = need_after0(r, k, ("M1_balance",), w)
        for q in M1_BAL_KEYS:
            S.put("M1_balance." + q, k, None if mb is None else need(mb, (q,), w + ".M1_balance"))
        for q in M1_PUT_KEYS:
            S.put("M1_balance.per_unit_time." + q, k, None if mb is None else need(mb, ("per_unit_time", q), w + ".M1_balance"))
        S.put("M1_balance.cut_changed", k, None if mb is None else opt(mb, ("cut_changed",)))
        s11 = need(r, ("s11",), w)
        for q in S11_REQ:
            S.put("s11." + q, k, need(s11, (q,), w + ".s11"))
        for q in S11_OPT:
            S.put("s11." + q, k, opt(s11, (q,)))
        stop_keys = sorted(set([q for q in s11 if "stop" in q.lower() or q.lower() in ("case", "p2_case", "stop_case")]
                               + ["record:" + q for q in r if "stop" in q.lower()]))
        S.put("s11.stop_keys_present", k, stop_keys)
        S.put("s11.stop_values", k, {q: (r.get(q[7:]) if q.startswith("record:") else s11.get(q)) for q in stop_keys})
        lv = need(r, ("light_v36",), w)
        wl = w + ".light_v36"
        for q in LV_REQ:
            S.put("light." + q, k, need(lv, (q,), wl))
        tg = need(lv, ("truncation_guard",), wl)
        for q in LV_TG:
            S.put("light.truncation_guard." + q, k, need(tg, (q,), wl + ".truncation_guard"))
        trm = need(lv, ("threshold_radius_massless",), wl)
        for c in chans:
            S.put("light.threshold_radius_massless." + c, k, need(trm, (c,), wl + ".threshold_radius_massless"))
        lmp = need(lv, ("M1_parts",), wl)
        for q in M1_PARTS_KEYS:
            S.put("light.M1_parts." + q, k, need(lmp, (q,), wl + ".M1_parts"))
        S.put("light.M1_parts.I_E_mid", k, opt(lmp, ("I_E_mid",)))
        lmb = need_after0(lv, k, ("M1_balance",), wl)
        for q in M1_BAL_KEYS:
            S.put("light.M1_balance." + q, k, None if lmb is None else need(lmb, (q,), wl + ".M1_balance"))
        for q in M1_PUT_KEYS:
            S.put("light.M1_balance.per_unit_time." + q, k, None if lmb is None else need(lmb, ("per_unit_time", q), wl + ".M1_balance"))
        S.put("light.dE_inv_light_dT", k, need_after0(lv, k, ("dE_inv_light_dT",), wl))
        for q in TIMING_OPT:
            S.put("light.timing." + q, k, opt(lv, ("timing", q)))
        S.put("light.light_slice_saved", k, opt(lv, ("light_slice_saved",)))
        chd = need(lv, ("channels",), wl)
        for c in chans:
            ch = need(chd, (c,), wl + ".channels")
            wc = wl + ".channels." + c
            for q in CH_REQ_C:
                S.put("L%s.%s" % (c, q), k, need(ch, (q,), wc))
            for q in CH_OPT:
                S.put("L%s.%s" % (c, q), k, opt(ch, (q,)))
            cert = need(ch, ("certified",), wc)
            for q in CERT_REQ:
                S.put("L%s.certified.%s" % (c, q), k, need(cert, (q,), wc + ".certified"))
            for q in CERT_OPT:
                S.put("L%s.certified.%s" % (c, q), k, opt(cert, (q,)))
            nb = need(ch, ("nocc_deviation_bins",), wc)
            for b in NOCC_BINS:
                S.put("L%s.nocc.%s" % (c, b), k, need(nb, (b,), wc + ".nocc_deviation_bins"))
            by = need(ch, ("beyond",), wc)
            for cl in collars:
                for q in ("number", "energy_density_route"):
                    S.put("L%s.beyond.%s.%s" % (c, cl, q), k, need(by, (cl, q), wc + ".beyond"))
                S.put("L%s.beyond.%s.weighted_energy_density_route" % (c, cl), k, opt(by, (cl, "weighted_energy_density_route")))
            if bool(ch["sourcing"]) != (c in src):
                raise Inconsistent("%s: sourcing = %r disagrees with the args (sourcing %s, guard %s)" % (wc, ch["sourcing"], src, grd))
    meta = {"tau": tau, "rs": rs, "ell": ell, "sourcing": src, "guard": grd, "channels": chans, "collars": collars, "n": n,
            "head_light": hl, "head_E0": opt(head, ("E0",)), "args": args, "script": opt(run, ("script",))}
    return S, meta


def extract_F(run, where):
    args = need(run, ("args",), where)
    head = need(run, ("head",), where)
    R = need(run, ("records",), where)
    if not isinstance(R, list) or not R:
        raise Inconsistent("%s: 'records' is not a non-empty list" % where)
    tau = float(need(args, ("tau",), where + ".args"))
    rs = float(need(args, ("rs",), where + ".args"))
    src = split_list(need(args, ("light_kappas",), where + ".args"))
    grd = split_list(need(args, ("light_guard_kappas",), where + ".args"))
    chans = src + grd
    if not chans:
        raise Inconsistent("%s: no light channels in the args" % where)
    collars = split_list(need(args, ("s11_collars",), where + ".args"))
    hl = need(head, ("light_v36",), where + ".head")
    wts = {c: float(need(hl, (c, "w"), where + ".head.light_v36")) for c in chans}
    kap = {c: need(hl, (c, "kappa"), where + ".head.light_v36") for c in chans}
    n = len(R)
    S = Series(n)
    for k, r in enumerate(R):
        w = "%s.records[%d]" % (where, k)
        for p in F_REC_ALL:
            S.put(p[0], k, need(r, p, w))
        for q in F_TOT_REQ:
            S.put(q, k, need(r, (q,), w))
        tg = need(r, ("truncation_guard",), w)
        for q in F_TG:
            S.put("truncation_guard." + q, k, need(tg, (q,), w + ".truncation_guard"))
        for q in TIMING_OPT:
            S.put("timing." + q, k, opt(r, ("timing", q)))
        chd = need(r, ("channels",), w)
        for c in chans:
            ch = need(chd, (c,), w + ".channels")
            wc = w + ".channels." + c
            for q in CH_REQ_F:
                S.put("L%s.%s" % (c, q), k, need(ch, (q,), wc))
            for q in CH_OPT:
                S.put("L%s.%s" % (c, q), k, opt(ch, (q,)))
            S.put("L%s.w" % c, k, wts[c])
            cert = need(ch, ("certified",), wc)
            for q in CERT_REQ:
                S.put("L%s.certified.%s" % (c, q), k, need(cert, (q,), wc + ".certified"))
            for q in CERT_OPT:
                S.put("L%s.certified.%s" % (c, q), k, opt(cert, (q,)))
            nb = need(ch, ("nocc_deviation_bins",), wc)
            for b in NOCC_BINS:
                S.put("L%s.nocc.%s" % (c, b), k, need(nb, (b,), wc + ".nocc_deviation_bins"))
            by = need(ch, ("beyond",), wc)
            for cl in collars:
                for q in ("number", "energy_density_route"):
                    S.put("L%s.beyond.%s.%s" % (c, cl, q), k, need(by, (cl, q), wc + ".beyond"))
            if bool(ch["sourcing"]) != (c in src):
                raise Inconsistent("%s: sourcing = %r disagrees with the args (sourcing %s, guard %s)" % (wc, ch["sourcing"], src, grd))
    meta = {"tau": tau, "rs": rs, "sourcing": src, "guard": grd, "channels": chans, "collars": collars, "n": n, "weights": wts,
            "kappas": kap, "args": args, "script": opt(run, ("script",)), "followed": opt(run, ("followed",)), "label": opt(run, ("label",)),
            "head_subcritical_sites": opt(head, ("subcritical_T0", "min_N_minus_absv_sites"))}
    return S, meta


def extract_P(run, where):
    args = need(run, ("args",), where)
    head = need(run, ("head",), where)
    R = need(run, ("records",), where)
    if not isinstance(R, list) or not R:
        raise Inconsistent("%s: 'records' is not a non-empty list" % where)
    tau = float(need(args, ("tau",), where + ".args"))
    rs = float(need(args, ("rs",), where + ".args"))
    n = len(R)
    S = Series(n)
    for k, r in enumerate(R):
        w = "%s.records[%d]" % (where, k)
        for p in P_REC_ALL:
            S.put(p[0], k, need(r, p, w))
        for p in P_REC_AFTER0:
            S.put(p[0], k, need_after0(r, k, p, w))
        for p in P_REC_OPT:
            S.put(p[0], k, opt(r, p))
    meta = {"tau": tau, "rs": rs, "n": n, "args": args, "script": opt(run, ("script",)), "head_E0": opt(head, ("E0",))}
    return S, meta


def s11_cross(mC, npz_path, where):
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.files)
    for key in ("T", "r_nodes", "M", "rho_ell"):
        if key not in keys:
            raise MissingKey("%s: the key '%s' is absent (keys present: %s); the cross-check stops, nothing is substituted"
                             % (where, key, ", ".join(keys)))
    Tz = np.asarray(z["T"], dtype=float)
    rn = np.asarray(z["r_nodes"], dtype=float)
    M = np.asarray(z["M"], dtype=float)
    rho = np.asarray(z["rho_ell"], dtype=float)
    n = mC["n"]
    if Tz.shape[0] != n or rn.shape[0] != n or M.shape != rn.shape or rho.shape != rn.shape:
        raise Inconsistent("%s: shapes T %s, r_nodes %s, M %s, rho_ell %s against %d records"
                           % (where, Tz.shape, rn.shape, M.shape, rho.shape, n))
    ell = mC["ell"]
    out = {"T_npz": [], "M_min_npz": [], "r_at_M_min_npz": [], "negative_points_npz": [], "rho_ell_min_npz": [], "n_nodes_outside_npz": []}
    for k in range(n):
        mask = rn[k] > ell
        if not mask.any():
            raise Inconsistent("%s: record %d has no node with r > ell = %g" % (where, k, ell))
        Mk, rk, rhk = M[k][mask], rn[k][mask], rho[k][mask]
        j = int(np.argmin(Mk))
        out["T_npz"].append(float(Tz[k]))
        out["M_min_npz"].append(float(Mk[j]))
        out["r_at_M_min_npz"].append(float(rk[j]))
        out["negative_points_npz"].append(int(np.count_nonzero(Mk < 0)))
        out["rho_ell_min_npz"].append(float(rhk.min()))
        out["n_nodes_outside_npz"].append(int(mask.sum()))
    return out, keys


# -------------------------------------------------------------------------------------------------------------------------- arithmetic
def align(TA, TB):
    ib = {}
    for j, t in enumerate(TB):
        if av(t):
            ib.setdefault(ckey(t), j)
    return [ib.get(ckey(t)) if av(t) else None for t in TA]


def find_clock(Trs, c):
    for k, t in enumerate(Trs):
        if av(t) and ckey(t) == ckey(c):
            return k
    return None


def cumtrapz(T, I):
    out = [0.0]
    for k in range(1, len(T)):
        if out[-1] is None or not av(I[k], I[k - 1], T[k], T[k - 1]):
            out.append(None)
        else:
            out.append(out[-1] + 0.5 * (I[k] + I[k - 1]) * (T[k] - T[k - 1]))
    return out


def cum_backward(T, d):
    out = [0.0]
    for k in range(1, len(T)):
        if out[-1] is None or not av(d[k], T[k], T[k - 1]):
            out.append(None)
        else:
            out.append(out[-1] + d[k] * (T[k] - T[k - 1]))
    return out


def cumsum_after0(x):
    out = [0.0]
    for k in range(1, len(x)):
        out.append(None if (out[-1] is None or not av(x[k])) else out[-1] + x[k])
    return out


def table_rows(Trs, kf):
    rows = []
    for c in TABLE_CLOCKS:
        rows.append(("%g" % c, find_clock(Trs, c), c))
        if c == 0.0:
            if kf is not None:
                rows.append(("formation (%g)" % Trs[kf], kf, Trs[kf]))
            else:
                rows.append(("formation (none)", None, None))
    return rows


def build_table(rows, cols, D):
    headers = ["T/r_s"] + [h for h, _, _ in cols]
    body = []
    for label, k, _ in rows:
        if k is None:
            body.append([label + ": absent"] + ["-"] * len(cols))
        else:
            body.append([label] + [fm(D[key][k] if key in D else None, p) for _, key, p in cols])
    return mdtab(headers, body)


def args_diff(a, b):
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if k in ARGS_SKIP:
            continue
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va != vb:
            out.append((k, va, vb))
    return out


# ----------------------------------------------------------------------------------------------------------------------------- compute
def compute(SC, mC, SF, mF, SP, mP, s11x, s11_keys, inputs, slice_dir):
    n, tau, rs, ell = mC["n"], mC["tau"], mC["rs"], mC["ell"]
    src, grd, chans, collars = mC["sourcing"], mC["guard"], mC["channels"], mC["collars"]
    ids = Identities()
    D = dict(SC.d)
    T, Trs, E = D["T"], D["T_over_rs"], D["E"]

    def mapk(f):
        return [f(k) for k in range(n)]

    def resid(stored, recomputed):
        return [abs(a - b) for a, b in zip(stored, recomputed) if av(a, b)]

    def resid_opt(stored, recomputed):
        out = []
        for a, b in zip(stored, recomputed):
            if a is None and b is None:
                out.append(0.0)
            elif av(a, b):
                out.append(abs(a - b))
            else:
                out.append(1.0)
        return out

    def wsum(key, which, k):
        tot = 0.0
        for c in which:
            v = D["L%s.%s" % (c, key)][k]
            if not av(v):
                return None
            tot += v
        return tot

    def wsum_w(key, which, k):
        tot = 0.0
        for c in which:
            v, w = D["L%s.%s" % (c, key)][k], D["L%s.w" % c][k]
            if not av(v, w):
                return None
            tot += w * v
        return tot

    # ---- the alignment of the follow and the production onto the coupling-on clocks
    iF = align(Trs, SF["T_over_rs"])
    iP = align(Trs, SP["T_over_rs"])
    for name, col in SF.d.items():
        D["F." + name] = [col[j] if j is not None else None for j in iF]
    for name, col in SP.d.items():
        D["prod." + name] = [col[j] if j is not None else None for j in iP]
    nF = sum(1 for j in iF if j is not None)
    nP = sum(1 for j in iP if j is not None)
    iFP = align(SF["T_over_rs"], SP["T_over_rs"])
    pairs_FP = [(j, i) for j, i in enumerate(iFP) if i is not None]

    # ---- identities within the coupling-on json
    ids.add("C: T_over_rs = T / rs", resid(Trs, mapk(lambda k: T[k] / rs if av(T[k]) else None)))
    ids.add("C: T[k] - T[k-1] = tau", [abs((T[k] - T[k - 1]) - tau) for k in range(1, n) if av(T[k], T[k - 1])])
    ids.add("C: E_over_E0 = E / E[0]", resid(D["E_over_E0"], mapk(lambda k: ratio(E[k], E[0]))))
    if av(mC["head_E0"], E[0]):
        ids.add("C: head.E0 = E[0]", [abs(mC["head_E0"] - E[0])])
    ids.add("C: solver.W_tail.E_collar = E (W/2g at the collar is E; -Delta W read as -Delta E)", resid(D["solver.W_tail.E_collar"], E),
            note="an optional key; n counts the records carrying it")
    ids.add("C: M1_I_E = M1_parts.I_E_start", resid(D["M1_I_E"], D["M1_parts.I_E_start"]))
    ids.add("C: light_v36.I_E_light = light_v36.M1_parts.I_E_start", resid(D["light.I_E_light"], D["light.M1_parts.I_E_start"]))
    ids.add("C: balance_d_E_minus_Hin_dT = dE_dT - dHin_dT", resid(D["balance_d_E_minus_Hin_dT"],
            mapk(lambda k: D["dE_dT"][k] - D["dHin_dT"][k] if av(D["dE_dT"][k], D["dHin_dT"][k]) else None)))
    for c in chans:
        wc = D["L%s.w" % c]
        ids.add("C: L%s w = 16 kappa" % c, [abs(x - 16.0 * float(c)) for x in wc if av(x)])
        big = maxabs(D["L%s.E_sea" % c] + D["L%s.E_ref" % c]) or 0.0
        ids.add("C: L%s E_inv = E_sea - E_ref" % c, [abs(a - (s - r)) for a, s, r in zip(D["L%s.E_inv" % c], D["L%s.E_sea" % c], D["L%s.E_ref" % c])
                                                   if av(a, s, r)], floor=8.0 * EPS * big, note="floor 8 eps max|E_sea, E_ref| (traces of size %.4g)" % big)
        ids.add("C: L%s E_inv_weighted = w E_inv" % c, [abs(a - w * b) for a, w, b in zip(D["L%s.E_inv_weighted" % c], wc, D["L%s.E_inv" % c]) if av(a, w, b)])
        ids.add("C: L%s eps_sum_weighted = w eps_sum" % c, [abs(a - w * b) for a, w, b in zip(D["L%s.eps_sum_weighted" % c], wc, D["L%s.eps_sum" % c]) if av(a, w, b)])
    ids.add("C: E_inv_light = sum over all channels of E_inv_weighted", resid(D["light.E_inv_light"], mapk(lambda k: wsum("E_inv_weighted", chans, k))))
    ids.add("C: E_inv_light_sourcing = sum over sourcing channels of E_inv_weighted", resid(D["light.E_inv_light_sourcing"], mapk(lambda k: wsum("E_inv_weighted", src, k))))
    ids.add("C: E_inv_light_over_E = E_inv_light / E", resid(D["light.E_inv_light_over_E"], mapk(lambda k: ratio(D["light.E_inv_light"][k], E[k]))))
    ids.add("C: E_inv_light_sourcing_over_E = E_inv_light_sourcing / E", resid(D["light.E_inv_light_sourcing_over_E"], mapk(lambda k: ratio(D["light.E_inv_light_sourcing"][k], E[k]))))
    ids.add("C: I_E_light = sum over all channels of I_E_cut_weighted", resid(D["light.I_E_light"], mapk(lambda k: wsum("I_E_cut_weighted", chans, k))))
    ids.add("C: I_E_light_sourcing = sum over sourcing channels of I_E_cut_weighted", resid(D["light.I_E_light_sourcing"], mapk(lambda k: wsum("I_E_cut_weighted", src, k))))
    ids.add("C: H_in_light = sum over all channels of H_in_weighted", resid(D["light.H_in_light"], mapk(lambda k: wsum("H_in_weighted", chans, k))))
    ids.add("C: eps_fed_sum = sum over sourcing channels of eps_sum_weighted", resid(D["light.eps_fed_sum"], mapk(lambda k: wsum("eps_sum_weighted", src, k))))
    ids.add("C: eps_light_all_sum = sum over all channels of eps_sum_weighted", resid(D["light.eps_light_all_sum"], mapk(lambda k: wsum("eps_sum_weighted", chans, k))))
    ids.add("C: J_fed_sum = sum over sourcing channels of w J_sum", resid(D["light.J_fed_sum"], mapk(lambda k: wsum_w("J_sum", src, k))))

    def guard_ratio(key, tot_name, k):
        return ratio(wsum(key, grd, k), D[tot_name][k])
    ids.add("C: truncation_guard.E_inv_guard_over_sourcing = guard E_inv_weighted / E_inv_light_sourcing",
            resid_opt(D["light.truncation_guard.E_inv_guard_over_sourcing"], mapk(lambda k: guard_ratio("E_inv_weighted", "light.E_inv_light_sourcing", k))))
    ids.add("C: truncation_guard.I_E_guard_over_sourcing = guard I_E_cut_weighted / I_E_light_sourcing",
            resid_opt(D["light.truncation_guard.I_E_guard_over_sourcing"], mapk(lambda k: guard_ratio("I_E_cut_weighted", "light.I_E_light_sourcing", k))))
    ids.add("C: truncation_guard.eps_guard_over_sourcing = guard eps_sum_weighted / eps_fed_sum (null where eps_fed_sum = 0)",
            resid_opt(D["light.truncation_guard.eps_guard_over_sourcing"], mapk(lambda k: guard_ratio("eps_sum_weighted", "light.eps_fed_sum", k))))

    # ---- the M1 balances per field
    trap_sign = {}
    for fld, pre in (("heavy", ""), ("light", "light.")):
        def B(q, pre=pre):
            return D[pre + "M1_balance." + q]

        def P(q, pre=pre):
            return D[pre + "M1_parts." + q]
        parts_big = maxabs(B("dHin") + B("evolution_exact") + B("field_work") + B("reference_reset")) or 0.0
        ids.add("C %s: M1_balance.dHin = evolution_exact + field_work + reference_reset" % fld,
                resid(B("dHin"), mapk(lambda k: (B("evolution_exact")[k] + B("field_work")[k] + B("reference_reset")[k])
                                      if av(B("evolution_exact")[k], B("field_work")[k], B("reference_reset")[k]) else None)),
                floor=4.0 * EPS * parts_big, note="recorded closure max |.| %.3g" % (maxabs(B("closure")) or 0.0))
        for q, qq in zip(M1_PUT_KEYS, M1_PUT_FROM):
            ids.add("C %s: M1_balance.per_unit_time.%s = %s / tau" % (fld, q, qq),
                    resid(D[pre + "M1_balance.per_unit_time." + q], mapk(lambda k, qq=qq: B(qq)[k] / tau if av(B(qq)[k]) else None)))
        traces_big = maxabs(P("X1") + P("X2") + P("Y1") + P("Y2")) or 0.0
        ids.add("C %s: field_work = (Y2 - Y1) - (X2 - X1) (the relation read from the stored parts)" % fld,
                resid(B("field_work"), mapk(lambda k: ((P("Y2")[k] - P("Y1")[k]) - (P("X2")[k] - P("X1")[k]))
                                            if av(P("X1")[k], P("X2")[k], P("Y1")[k], P("Y2")[k]) else None)),
                kind="relation", floor=4.0 * EPS * traces_big, note="the parts are traces of size %.3g; roundoff floor 4 eps max|X, Y|" % traces_big)
        st, I0, Ie = B("evolution_trapezoid"), P("I_E_start"), P("I_E_end_prev_interval")
        res = {}
        for s in (1.0, -1.0):
            res[s] = resid(st, mapk(lambda k, s=s: s * tau * (I0[k - 1] + Ie[k]) / 2.0 if (k >= 1 and av(I0[k - 1], Ie[k])) else None))
        m = {s: (max(r) if r else 0.0) for s, r in res.items()}
        s_best = 1.0 if m[1.0] <= m[-1.0] else -1.0
        trap_sign[fld] = s_best
        ids.add("C %s: evolution_trapezoid = s tau (I_E_start[k-1] + I_E_end_prev_interval[k]) / 2" % fld, res[s_best], kind="hypothesis",
                note=("s = %+d fits (residuals s=+1 %.3g, s=-1 %.3g); with s = +1 the stored cut current counts energy entering the inside of the "
                      "cut, with s = -1 energy leaving it" % (int(s_best), m[1.0], m[-1.0])), extra={"s_plus": m[1.0], "s_minus": m[-1.0], "s_best": s_best})
        Hk = D["H_in"] if fld == "heavy" else D["light.H_in_light"]
        hbig = maxabs(Hk) or 0.0
        ids.add("C %s: M1_balance.dHin = H_in[k] - H_in[k-1] (the stored inside energies)" % fld,
                resid(B("dHin"), mapk(lambda k: (Hk[k] - Hk[k - 1]) if (k >= 1 and av(Hk[k], Hk[k - 1])) else None)),
                kind="cross", floor=4.0 * EPS * max(traces_big, hbig), note="M1's trace arithmetic against the stored H_in; floor 4 eps max|X, Y, H_in|")

    def hyp_fd(name, stored, base):
        rb = resid(stored, mapk(lambda k: (base[k] - base[k - 1]) / tau if (k >= 1 and av(base[k], base[k - 1])) else None))
        rc = resid(stored, mapk(lambda k: (base[k + 1] - base[k - 1]) / (2.0 * tau) if (1 <= k <= n - 2 and av(base[k + 1], base[k - 1])) else None))
        mb, mc = (max(rb) if rb else 0.0), (max(rc) if rc else 0.0)
        best = "backward (x_k - x_{k-1}) / tau" if mb <= mc else "central (x_{k+1} - x_{k-1}) / (2 tau)"
        ids.add(name, rb if mb <= mc else rc, kind="hypothesis",
                note="read off the records: %s (residuals backward %.3g, central %.3g)" % (best, mb, mc), extra={"backward": mb, "central": mc, "best": best})
        return best
    best_dE = hyp_fd("C: dE_dT is a finite difference of E", D["dE_dT"], E)
    hyp_fd("C: dHin_dT is a finite difference of H_in", D["dHin_dT"], D["H_in"])
    hyp_fd("C: light_v36.dE_inv_light_dT is a finite difference of E_inv_light", D["light.dE_inv_light_dT"], D["light.E_inv_light"])

    # ---- s11 (json) against the record and the light entries
    ids.add("C: s11.E = E", resid(D["s11.E"], E))
    ids.add("C: s11.T_over_rs = T_over_rs", resid(D["s11.T_over_rs"], Trs))
    ids.add("C: s11.light_source_integral = light_v36.eps_fed_sum (the light fed source's total)", resid(D["s11.light_source_integral"], D["light.eps_fed_sum"]))
    ids.add("C: s11.pairs_source_integral = E_pairs_shells", resid(D["s11.pairs_source_integral"], D["E_pairs_shells"]))
    ids.add("C: s11.ball_source_integral = E_dust_shells (the heavy dust in the role of the ball)", resid(D["s11.ball_source_integral"], D["E_dust_shells"]))
    ids.add("C: s11.light_certified_on_record = light_v36.certified_on_record",
            [0.0 if bool(a) == bool(b) else 1.0 for a, b in zip(D["s11.light_certified_on_record"], D["light.certified_on_record"])])

    # ---- the head
    hl = mC["head_light"]
    sub0 = need(hl, ("subcritical_T0",), "run.head.light_v36")
    for q in ("min_N_minus_absv_sites", "min_N_minus_absv_dense", "r_min", "zeros"):
        need(sub0, (q,), "run.head.light_v36.subcritical_T0")
    first_return = need(hl, ("first_return",), "run.head.light_v36")
    chain_end = need(hl, ("chain_end_over_rs",), "run.head.light_v36")
    if av(sub0["min_N_minus_absv_dense"], D["min_N_minus_absv"][0]):
        ids.add("C: head subcritical_T0.min_N_minus_absv_dense = records[0].min_N_minus_absv", [abs(sub0["min_N_minus_absv_dense"] - D["min_N_minus_absv"][0])])
    head_channels = {}
    for c in chans:
        hc = need(hl, ("channels", c), "run.head.light_v36")
        head_channels[c] = {q: hc.get(q) for q in ("kappa", "w", "sourcing", "n_sea", "n_plus", "mu0", "gap_mu0", "min_abs_e",
                                                   "selection_equals_negative_set", "n_sea_equals_nsites", "commutator_h_S_max_abs", "h_m_real_part_max")}
        ids.add("C: head channels.%s.w = 16 kappa" % c, [abs(float(hc["w"]) - 16.0 * float(hc["kappa"]))] if av(hc.get("w"), hc.get("kappa")) else [])

    # ---- identities within the light-follow json
    nf, chF, srcF, grdF, wF = mF["n"], mF["channels"], mF["sourcing"], mF["guard"], mF["weights"]

    def fsum(key, which, j):
        tot = 0.0
        for c in which:
            v = SF["L%s.%s" % (c, key)][j]
            if not av(v):
                return None
            tot += wF[c] * v
        return tot

    def fres(stored_name, f):
        out = []
        for j in range(nf):
            a, b = SF[stored_name][j], f(j)
            if av(a, b):
                out.append(abs(a - b))
        return out
    for c in chF:
        ids.add("F: head light_v36.%s.w = 16 kappa" % c, [abs(wF[c] - 16.0 * float(mF["kappas"][c]))] if av(mF["kappas"][c]) else [])
        bigF = maxabs(SF["L%s.E_sea" % c] + SF["L%s.E_ref" % c]) or 0.0
        ids.add("F: L%s E_inv = E_sea - E_ref" % c, [abs(a - (s - r)) for a, s, r in zip(SF["L%s.E_inv" % c], SF["L%s.E_sea" % c], SF["L%s.E_ref" % c]) if av(a, s, r)],
                floor=8.0 * EPS * bigF, note="floor 8 eps max|E_sea, E_ref|")
    ids.add("F: E_inv_light_sourcing = sum over sourcing channels of w E_inv", fres("E_inv_light_sourcing", lambda j: fsum("E_inv", srcF, j)))
    ids.add("F: E_inv_light_all = sum over all channels of w E_inv", fres("E_inv_light_all", lambda j: fsum("E_inv", chF, j)))
    ids.add("F: E_rad0_over_E = E_inv_light_sourcing / E", fres("E_rad0_over_E", lambda j: ratio(SF["E_inv_light_sourcing"][j], SF["E"][j])))
    ids.add("F: I_E_light_sourcing = sum over sourcing channels of w I_E_cut", fres("I_E_light_sourcing", lambda j: fsum("I_E_cut", srcF, j)))
    ids.add("F: I_E_light_all = sum over all channels of w I_E_cut", fres("I_E_light_all", lambda j: fsum("I_E_cut", chF, j)))
    ids.add("F: eps_fed_sum = sum over sourcing channels of w eps_sum", fres("eps_fed_sum", lambda j: fsum("eps_sum", srcF, j)))
    ids.add("F: J_fed_sum = sum over sourcing channels of w J_sum", fres("J_fed_sum", lambda j: fsum("J_sum", srcF, j)))
    ids.add("F: truncation_guard.E_inv_guard_over_sourcing = guard / sourcing", fres("truncation_guard.E_inv_guard_over_sourcing",
            lambda j: ratio(fsum("E_inv", grdF, j), SF["E_inv_light_sourcing"][j])))
    ids.add("F: truncation_guard.I_E_guard_over_sourcing = guard / sourcing", fres("truncation_guard.I_E_guard_over_sourcing",
            lambda j: ratio(fsum("I_E_cut", grdF, j), SF["I_E_light_sourcing"][j])))

    # ---- the follow against the production (the follow ran on the production's records)
    ids.add("F vs production: E equal at matched clocks", [abs(SF["E"][j] - SP["E"][i]) for j, i in pairs_FP if av(SF["E"][j], SP["E"][i])],
            kind="cross", extra={"matched": len(pairs_FP)}, note="the light-follow's E is the followed run's")
    ids.add("F vs production: R_t equal at matched clocks (null where null)",
            [(abs(SF["R_t"][j] - SP["R_t"][i]) if av(SF["R_t"][j], SP["R_t"][i]) else (0.0 if (SF["R_t"][j] is None and SP["R_t"][i] is None) else 1.0))
             for j, i in pairs_FP], kind="cross")
    ids.add("F vs production: min_N_minus_absv equal at matched clocks",
            [abs(SF["min_N_minus_absv"][j] - SP["min_N_minus_absv"][i]) for j, i in pairs_FP if av(SF["min_N_minus_absv"][j], SP["min_N_minus_absv"][i])], kind="cross")

    # ---- the s11 npz against the json
    for name, col in s11x.items():
        D["s11npz." + name] = col
    ids.add("s11 npz: T = records' T", resid(s11x["T_npz"], T), kind="cross")
    ids.add("s11: M_min_outside_cell (json) = min over r > ell of M (npz)", resid(D["s11.M_min_outside_cell"], s11x["M_min_npz"]), kind="cross",
            note="the central cell read as r <= ell, ell = %g" % ell)
    ids.add("s11: r_at_M_min (json) = the argmin's node (npz)", resid(D["s11.r_at_M_min"], s11x["r_at_M_min_npz"]), kind="cross")
    ids.add("s11: negative_points_outside_cell (json) = #{M < 0, r > ell} (npz)", resid(D["s11.negative_points_outside_cell"], s11x["negative_points_npz"]), kind="cross")
    ids.add("s11: rho_ell_min_outside_cell (json) = min over r > ell of rho_ell (npz)", resid(D["s11.rho_ell_min_outside_cell"], s11x["rho_ell_min_npz"]), kind="cross")
    D["r4.dM_min_npz_minus_json"] = [a - b if av(a, b) else None for a, b in zip(s11x["M_min_npz"], D["s11.M_min_outside_cell"])]
    D["s11.stop_keys_str"] = [", ".join(x) if x else "none" for x in D["s11.stop_keys_present"]]

    # ---- derived series: readout 2
    D["dE"] = [E[k] - E[0] if av(E[k], E[0]) else None for k in range(n)]
    D["minus_dE"] = [-x if av(x) else None for x in D["dE"]]
    D["cum.int_dE_dT"] = cum_backward(T, D["dE_dT"])
    ids.add("C: E(T) - E(0) = integral of dE_dT dT (backward rule)", resid(D["dE"], D["cum.int_dE_dT"]), kind="cross",
            note="exact when dE_dT is the backward difference; dE_dT read off as: %s" % best_dE)
    D["cum.int_M1_I_E"] = cumtrapz(T, D["M1_I_E"])
    D["light.I_E_light_guard"] = [a - b if av(a, b) else None for a, b in zip(D["light.I_E_light"], D["light.I_E_light_sourcing"])]
    D["cum.int_I_E_light"] = cumtrapz(T, D["light.I_E_light"])
    D["cum.int_I_E_light_sourcing"] = cumtrapz(T, D["light.I_E_light_sourcing"])
    D["cum.int_I_E_light_guard"] = cumtrapz(T, D["light.I_E_light_guard"])
    D["r2.remainder"] = [m - (h + l) if av(m, h, l) else None for m, h, l in zip(D["minus_dE"], D["cum.int_M1_I_E"], D["cum.int_I_E_light_sourcing"])]
    for pre in ("", "light."):
        for q in ("dHin", "evolution_exact", "evolution_trapezoid", "field_work", "reference_reset"):
            D["cum." + pre + "M1_balance." + q] = cumsum_after0(D[pre + "M1_balance." + q])
    D["dH_in"] = [h - D["H_in"][0] if av(h, D["H_in"][0]) else None for h in D["H_in"]]
    D["dH_in_light"] = [h - D["light.H_in_light"][0] if av(h, D["light.H_in_light"][0]) else None for h in D["light.H_in_light"]]
    ids.add("C heavy: sum of M1_balance.dHin = H_in(T) - H_in(0)", resid(D["dH_in"], D["cum.M1_balance.dHin"]), kind="cross",
            note="the per-record cross residual accumulates over the records")
    ids.add("C light: sum of light M1_balance.dHin = H_in_light(T) - H_in_light(0)", resid(D["dH_in_light"], D["cum.light.M1_balance.dHin"]), kind="cross",
            note="the per-record cross residual accumulates over the records")

    # ---- derived series: readout 3
    D["A"] = [4.0 * math.pi * r * r if av(r) else None for r in D["R_t"]]
    D["prod.A"] = [4.0 * math.pi * r * r if av(r) else None for r in D["prod.R_t"]]
    D["r3.dR_t"] = [a - b if av(a, b) else None for a, b in zip(D["R_t"], D["prod.R_t"])]
    D["r3.dR_t_rel"] = [rel(a, b) for a, b in zip(D["R_t"], D["prod.R_t"])]
    D["r3.dr_in"] = [a - b if av(a, b) else None for a, b in zip(D["r_in"], D["prod.r_in"])]
    D["r3.dE_rel"] = [rel(a, b) for a, b in zip(E, D["prod.E"])]
    D["r3.dmin_N_minus_absv"] = [a - b if av(a, b) else None for a, b in zip(D["min_N_minus_absv"], D["prod.min_N_minus_absv"])]
    kf = next((k for k in range(n) if D["trapping_zeros"][k]), None)
    kfp = next((j for j in range(mP["n"]) if SP["trapping_zeros"][j]), None)
    k0C = next((k for k in range(n) if av(D["min_N_minus_absv"][k]) and D["min_N_minus_absv"][k] <= 0.0), None)
    k0P = next((j for j in range(mP["n"]) if av(SP["min_N_minus_absv"][j]) and SP["min_N_minus_absv"][j] <= 0.0), None)
    formation = {"C": None, "production": None, "dT_over_rs": None, "record_spacing_T_over_rs": tau / rs}
    if kf is not None:
        formation["C"] = {"k": kf, "T_over_rs": Trs[kf], "R_t": D["R_t"][kf], "r_in": D["r_in"][kf], "trapping_zeros": D["trapping_zeros"][kf],
                          "E": E[kf], "min_N_minus_absv": D["min_N_minus_absv"][kf], "A": D["A"][kf],
                          "production_at_this_clock": {"R_t": D["prod.R_t"][kf], "r_in": D["prod.r_in"][kf], "E": D["prod.E"][kf],
                                                       "trapping_zeros": D["prod.trapping_zeros"][kf], "min_N_minus_absv": D["prod.min_N_minus_absv"][kf]},
                          "dR_t": D["r3.dR_t"][kf], "dR_t_rel": D["r3.dR_t_rel"][kf], "dr_in": D["r3.dr_in"][kf], "dE_rel": D["r3.dE_rel"][kf],
                          "E_rad0_over_E_follow": D["F.E_rad0_over_E"][kf], "E_inv_light_over_E": D["light.E_inv_light_over_E"][kf],
                          "E_inv_light_sourcing_over_E": D["light.E_inv_light_sourcing_over_E"][kf]}
    if kfp is not None:
        formation["production"] = {"k": kfp, "T_over_rs": SP["T_over_rs"][kfp], "R_t": SP["R_t"][kfp], "r_in": SP["r_in"][kfp],
                                   "trapping_zeros": SP["trapping_zeros"][kfp], "E": SP["E"][kfp], "min_N_minus_absv": SP["min_N_minus_absv"][kfp]}
    if kf is not None and kfp is not None:
        formation["dT_over_rs"] = Trs[kf] - SP["T_over_rs"][kfp]
    formation["first_min_N_minus_absv_le_0"] = {"C": Trs[k0C] if k0C is not None else None, "production": SP["T_over_rs"][k0P] if k0P is not None else None}

    # ---- derived series: readout 5
    common = [c for c in chans if c in chF]
    D["r5.rel_E_inv_sourcing_over_E"] = [rel(a, b) for a, b in zip(D["light.E_inv_light_sourcing_over_E"], D["F.E_rad0_over_E"])]
    D["r5.rel_E_inv_all"] = [rel(a, b) for a, b in zip(D["light.E_inv_light"], D["F.E_inv_light_all"])]
    D["r5.rel_E_inv_sourcing"] = [rel(a, b) for a, b in zip(D["light.E_inv_light_sourcing"], D["F.E_inv_light_sourcing"])]
    D["r5.d_I_E_light_sourcing"] = [a - b if av(a, b) else None for a, b in zip(D["light.I_E_light_sourcing"], D["F.I_E_light_sourcing"])]
    D["r5.d_I_E_light_all"] = [a - b if av(a, b) else None for a, b in zip(D["light.I_E_light"], D["F.I_E_light_all"])]
    D["r5.rel_eps_fed_sum"] = [rel(a, b) for a, b in zip(D["light.eps_fed_sum"], D["F.eps_fed_sum"])]
    D["r5.rel_J_fed_sum"] = [rel(a, b) for a, b in zip(D["light.J_fed_sum"], D["F.J_fed_sum"])]
    D["r5.F.H_in_light_weighted"] = mapk(lambda k: (sum(wF[c] * D["F.L%s.H_in" % c][k] for c in common) if common and all(av(D["F.L%s.H_in" % c][k]) for c in common) else None))
    D["r5.rel_H_in_light"] = [rel(a, b) for a, b in zip(D["light.H_in_light"], D["r5.F.H_in_light_weighted"])]
    for c in common:
        D["r5.L%s.rel_E_inv" % c] = [rel(a, b) for a, b in zip(D["L%s.E_inv" % c], D["F.L%s.E_inv" % c])]
        D["r5.L%s.I_E_cut_from_weighted" % c] = [ratio(a, w) for a, w in zip(D["L%s.I_E_cut_weighted" % c], D["L%s.w" % c])]
        D["r5.L%s.d_I_E_cut" % c] = [a - b if av(a, b) else None for a, b in zip(D["r5.L%s.I_E_cut_from_weighted" % c], D["F.L%s.I_E_cut" % c])]
        D["r5.L%s.H_in_from_weighted" % c] = [ratio(a, w) for a, w in zip(D["L%s.H_in_weighted" % c], D["L%s.w" % c])]
        D["r5.L%s.rel_eps_sum" % c] = [rel(a, b) for a, b in zip(D["L%s.eps_sum" % c], D["F.L%s.eps_sum" % c])]
        for b in NOCC_BINS:
            D["r5.L%s.d_nocc.%s" % (c, b)] = [x - y if av(x, y) else None for x, y in zip(D["L%s.nocc.%s" % (c, b)], D["F.L%s.nocc.%s" % (c, b)])]
        for cl in collars:
            if "F.L%s.beyond.%s.energy_density_route" % (c, cl) in D:
                D["r5.L%s.d_beyond.%s.energy_density_route" % (c, cl)] = [x - y if av(x, y) else None for x, y in
                                                                          zip(D["L%s.beyond.%s.energy_density_route" % (c, cl)], D["F.L%s.beyond.%s.energy_density_route" % (c, cl)])]

    # ---- the tables
    rows = table_rows(Trs, kf)
    kend = n - 1
    k20 = find_clock(Trs, 20.0)

    def at(name, k):
        return D[name][k] if (k is not None and name in D) else None

    tab2a = build_table(rows, [("E", "E", 10), ("E/E0", "E_over_E0", 10), ("-dE", "minus_dE", 6), ("int dE_dT", "cum.int_dE_dT", 6),
                               ("M1_I_E", "M1_I_E", 4), ("int M1_I_E", "cum.int_M1_I_E", 4), ("I_E_light_src", "light.I_E_light_sourcing", 4),
                               ("int src", "cum.int_I_E_light_sourcing", 4), ("I_E_light_guard", "light.I_E_light_guard", 4),
                               ("int guard", "cum.int_I_E_light_guard", 4), ("remainder", "r2.remainder", 6)], D)
    tab2b = build_table(rows, [("dH_in_light", "dH_in_light", 6), ("sum dHin", "cum.light.M1_balance.dHin", 6),
                               ("sum evolution_exact", "cum.light.M1_balance.evolution_exact", 6), ("sum field_work", "cum.light.M1_balance.field_work", 6),
                               ("sum reference_reset", "cum.light.M1_balance.reference_reset", 6), ("H_in_light", "light.H_in_light", 6),
                               ("E_inv_light", "light.E_inv_light", 6), ("E_inv_light_sourcing", "light.E_inv_light_sourcing", 6)], D)
    tab3 = build_table(rows, [("R_t C", "R_t", 9), ("R_t prod", "prod.R_t", 9), ("dR_t", "r3.dR_t", 4), ("r_in C", "r_in", 9), ("r_in prod", "prod.r_in", 9),
                              ("min(N-|v|) C", "min_N_minus_absv", 6), ("prod", "prod.min_N_minus_absv", 6), ("E C", "E", 11), ("E prod", "prod.E", 11),
                              ("(E_C-E_p)/E_p", "r3.dE_rel", 4), ("E_rad0/E (follow)", "F.E_rad0_over_E", 5), ("E_inv_src/E (C)", "light.E_inv_light_sourcing_over_E", 5),
                              ("E_inv_all/E (C)", "light.E_inv_light_over_E", 5), ("A = 4 pi R_t^2 (C)", "A", 7)], D)
    tab4 = build_table(rows, [("M_min", "s11.M_min_outside_cell", 8), ("r_at_M_min", "s11.r_at_M_min", 6), ("neg pts", "s11.negative_points_outside_cell", 1),
                              ("M_min npz - json", "r4.dM_min_npz_minus_json", 3), ("neg pts npz", "s11npz.negative_points_npz", 1),
                              ("max 2gM/r", "s11.compactness_max_2gM_over_r", 6), ("source_min", "s11.source_min_outside_cell", 4),
                              ("r_at_source_min", "s11.r_at_source_min", 6), ("dust min", "s11.ball_source_min_outside_cell", 4),
                              ("pairs min", "s11.pairs_source_min_outside_cell", 4), ("light min", "s11.light_source_min_outside_cell", 4),
                              ("light integral", "s11.light_source_integral", 6), ("light fed", "s11.light_fed", 1),
                              ("light certified", "s11.light_certified_on_record", 1), ("stop key", "s11.stop_keys_str", 1)], D)
    tab5a = build_table(rows, [("E_inv_src/E C", "light.E_inv_light_sourcing_over_E", 6), ("E_rad0/E F", "F.E_rad0_over_E", 6), ("rel", "r5.rel_E_inv_sourcing_over_E", 4),
                               ("E_inv_all C", "light.E_inv_light", 6), ("E_inv_all F", "F.E_inv_light_all", 6), ("rel", "r5.rel_E_inv_all", 4),
                               ("I_E_light_src C", "light.I_E_light_sourcing", 4), ("F", "F.I_E_light_sourcing", 4),
                               ("eps_fed C", "light.eps_fed_sum", 6), ("F", "F.eps_fed_sum", 6), ("rel", "r5.rel_eps_fed_sum", 4),
                               ("J_fed C", "light.J_fed_sum", 6), ("F", "F.J_fed_sum", 6), ("rel", "r5.rel_J_fed_sum", 4)], D)
    cols5b = []
    for c in common:
        cols5b += [("L%s E_inv C" % c, "L%s.E_inv" % c, 6), ("F", "F.L%s.E_inv" % c, 6), ("rel", "r5.L%s.rel_E_inv" % c, 4)]
    tab5b = build_table(rows, cols5b, D) if cols5b else "(no common light channel)"
    cols5c = []
    for c in common:
        cols5c += [("L%s (0,1) C" % c, "L%s.nocc.(0,1)" % c, 5), ("F", "F.L%s.nocc.(0,1)" % c, 5), ("L%s (-1,0) C" % c, "L%s.nocc.(-1,0)" % c, 5), ("F", "F.L%s.nocc.(-1,0)" % c, 5)]
    tab5c = build_table(rows, cols5c, D) if cols5c else "(no common light channel)"
    cols5d = []
    last_collar = collars[-1] if collars else None
    for c in common:
        cols5d += [("L%s I_E_cut C (weighted/w)" % c, "r5.L%s.I_E_cut_from_weighted" % c, 4), ("F", "F.L%s.I_E_cut" % c, 4)]
        if last_collar is not None and ("F.L%s.beyond.%s.energy_density_route" % (c, last_collar)) in D:
            cols5d += [("L%s beyond %s C" % (c, last_collar), "L%s.beyond.%s.energy_density_route" % (c, last_collar), 4),
                       ("F", "F.L%s.beyond.%s.energy_density_route" % (c, last_collar), 4)]
    tab5d = build_table(rows, cols5d, D) if cols5d else "(no common light channel)"
    tab7a = build_table(rows, [("dHin", "M1_balance.dHin", 6), ("evolution_exact", "M1_balance.evolution_exact", 4), ("evolution_trapezoid", "M1_balance.evolution_trapezoid", 4),
                               ("field_work", "M1_balance.field_work", 6), ("reference_reset", "M1_balance.reference_reset", 4), ("closure", "M1_balance.closure", 3),
                               ("E_inv_heavy", "light.E_inv_heavy", 9), ("H_in", "H_in", 9), ("dHin_dT", "dHin_dT", 6), ("I_E_mid", "M1_parts.I_E_mid", 4)], D)
    tab7b = build_table(rows, [("dHin", "light.M1_balance.dHin", 6), ("evolution_exact", "light.M1_balance.evolution_exact", 4),
                               ("evolution_trapezoid", "light.M1_balance.evolution_trapezoid", 4), ("field_work", "light.M1_balance.field_work", 6),
                               ("reference_reset", "light.M1_balance.reference_reset", 4), ("closure", "light.M1_balance.closure", 3),
                               ("E_inv_light", "light.E_inv_light", 6), ("E_inv_light_sourcing", "light.E_inv_light_sourcing", 6),
                               ("H_in_light", "light.H_in_light", 6), ("dE_inv_light_dT", "light.dE_inv_light_dT", 4)], D)
    tabT = build_table(rows, [("E_inv guard/src", "light.truncation_guard.E_inv_guard_over_sourcing", 6), ("I_E guard/src", "light.truncation_guard.I_E_guard_over_sourcing", 6),
                              ("eps guard/src", "light.truncation_guard.eps_guard_over_sourcing", 6)]
                       + [("L%s E_inv_w" % c, "L%s.E_inv_weighted" % c, 5) for c in grd] + [("L%s I_E_cut_w" % c, "L%s.I_E_cut_weighted" % c, 4) for c in grd]
                       + [("L%s eps_sum_w" % c, "L%s.eps_sum_weighted" % c, 5) for c in grd], D)
    colsC = []
    for c in chans:
        colsC += [("L%s pieces" % c, "L%s.certified.pieces" % c, 1), ("chol" , "L%s.certified.cholesky_calls" % c, 1), ("fallback", "L%s.fallback" % c, 1), ("cut xing", "L%s.cut_crossing" % c, 1)]
    colsC += [("walk_pieces (record)", "light.timing.walk_pieces", 1), ("walk_cholesky", "light.timing.walk_cholesky", 1), ("walk_fallbacks", "light.timing.walk_fallbacks", 1),
              ("cut_crossings", "light.timing.cut_crossings", 1), ("certified_on_record", "light.certified_on_record", 1)]
    tabC = build_table(rows, colsC, D)
    colsR = []
    for c in chans:
        colsR += [("thr massless L%s" % c, "light.threshold_radius_massless.%s" % c, 6), ("L%s.threshold_radius" % c, "L%s.threshold_radius" % c, 6)]
    tabR = build_table(rows, colsR, D)

    # ---- the summaries
    kabs = argmaxabs(D["r3.dR_t"])
    kabsE = argmaxabs(D["r3.dE_rel"])
    fb_count = sum(1 for c in chans for x in D["L%s.fallback" % c] if x)
    cx_count = sum(1 for c in chans for x in D["L%s.cut_crossing" % c] if x)
    whys = sorted(set(str(x) for c in chans for x in D["L%s.why" % c] if x is not None))
    labels = sorted(set(str(x) for x in D["light.label"]))
    cont_labels = sorted(set(str(x) for x in D["continuation_label"] if x is not None))
    modes = sorted(set(re.sub(r"\(\d+ pieces\)", "(N pieces)", str(x)) for c in chans for x in D["L%s.mode" % c]))
    pieces_max = {c: maxabs(D["L%s.certified.pieces" % c]) for c in chans}
    chol_max = {c: maxabs(D["L%s.certified.cholesky_calls" % c]) for c in chans}
    not_cert = sum(1 for c in chans for x in D["L%s.certified.certified" % c] if not x)
    timing_sums = {q: (sum(x for x in D["light.timing." + q] if av(x)) if any(av(x) for x in D["light.timing." + q]) else None) for q in TIMING_OPT}
    thr_nonnull = {c: sum(1 for x in D["light.threshold_radius_massless." + c] if x is not None) for c in chans}
    thr_ch_nonnull = {c: sum(1 for x in D["L%s.threshold_radius" % c] if x is not None) for c in chans}
    kMmin = min((k for k in range(n) if av(D["s11.M_min_outside_cell"][k])), key=lambda k: D["s11.M_min_outside_cell"][k], default=None)
    neg_records = [Trs[k] for k in range(n) if av(D["s11.negative_points_outside_cell"][k]) and D["s11.negative_points_outside_cell"][k] > 0]
    stop_records = [{"T_over_rs": Trs[k], "keys": D["s11.stop_keys_present"][k], "values": D["s11.stop_values"][k]} for k in range(n) if D["s11.stop_keys_present"][k]]
    not_fed = sum(1 for x in D["s11.light_fed"] if not x)
    not_cert_s11 = sum(1 for x in D["s11.light_certified_on_record"] if not x)
    kLmin = min((k for k in range(n) if av(D["s11.light_source_min_outside_cell"][k])), key=lambda k: D["s11.light_source_min_outside_cell"][k], default=None)
    rel20 = at("r5.rel_E_inv_sourcing_over_E", k20)
    relend = at("r5.rel_E_inv_sourcing_over_E", kend)
    relmax = maxabs(D["r5.rel_E_inv_sourcing_over_E"])
    kfr = find_clock(SF["T_over_rs"], 0.0)
    dargs = args_diff(mC["args"], mP["args"])
    g = fm

    def idres(name):
        return ids.d[name]["max_abs_residual"] if name in ids.d else None

    s2 = ("Readout 2: E(T) - E(0) = int dE_dT dT holds to %s (dE_dT read off as %s). At T/r_s %s: -dE = %s; the heavy collar cut current's integral "
          "int M1_I_E dT = %s; the light field's int I_E_light_sourcing dT = %s (the guard's %s); the remainder -dE - int (M1_I_E + I_E_light_sourcing) dT = %s "
          "(%s of -dE): E's change over the run is the quasi-static re-solve's, not a collar flux (the collar at the chain's end %s r_s, the first return near "
          "T/r_s %s). The light field's d<H_in>_light over the run sums to %s = evolution_exact %s + field_work %s + reference_reset %s (closure residual max %s); "
          "its stored H_in_light changed by %s. The trapezoid's form fits evolution_trapezoid with s = %+d (heavy) and %+d (light); -Delta W is -Delta E "
          "(solver.W_tail.E_collar = E to %s)."
          % (g(idres("C: E(T) - E(0) = integral of dE_dT dT (backward rule)"), 3), best_dE, g(Trs[kend], 6), g(at("minus_dE", kend), 6),
             g(at("cum.int_M1_I_E", kend), 4), g(at("cum.int_I_E_light_sourcing", kend), 4), g(at("cum.int_I_E_light_guard", kend), 4), g(at("r2.remainder", kend), 6),
             g(ratio(at("r2.remainder", kend), at("minus_dE", kend)), 4), g(chain_end, 4), g(first_return.get("from_2_rs_back_to_24_rs"), 4),
             g(at("cum.light.M1_balance.dHin", kend), 4), g(at("cum.light.M1_balance.evolution_exact", kend), 4), g(at("cum.light.M1_balance.field_work", kend), 4),
             g(at("cum.light.M1_balance.reference_reset", kend), 4), g(idres("C light: M1_balance.dHin = evolution_exact + field_work + reference_reset"), 3),
             g(at("dH_in_light", kend), 4), int(trap_sign["heavy"]), int(trap_sign["light"]), g(idres("C: solver.W_tail.E_collar = E (W/2g at the collar is E; -Delta W read as -Delta E)"), 3)))
    fC, fP = formation["C"], formation["production"]
    s3 = ("Readout 3: the formation (first trapping zero): coupling on at T/r_s %s (R_t %s, r_in %s, zeros %s), the 1-field production at T/r_s %s (R_t %s, r_in %s); "
          "dT/r_s = %s (record spacing %s); dR_t = %s (%s relative), dr_in = %s; (E_C - E_p)/E_p at that clock = %s; against it E_rad^(0)/E (light-follow) = %s and "
          "E_inv_light_sourcing/E (coupling on) = %s (all channels %s). Over the matched clocks (%d of %d) max |dR_t| = %s at T/r_s %s and max |(E_C - E_p)/E_p| = %s at T/r_s %s, "
          "while E_rad^(0)/E reaches %s at T/r_s %s. min(N - |v|) first <= 0 at T/r_s %s (C) and %s (production). Caveat: the production's args differ from the "
          "coupling-on run's in %d entries (%s); the difference mixes the light coupling with the chain length and the solver; the solver-matched coupling-off control "
          "(spec sec. 6) is not among the inputs."
          % (g(fC["T_over_rs"], 6) if fC else "-", g(fC["R_t"], 9) if fC else "-", g(fC["r_in"], 9) if fC else "-", g(fC["trapping_zeros"], 6) if fC else "-",
             g(fP["T_over_rs"], 6) if fP else "-", g(fP["R_t"], 9) if fP else "-", g(fP["r_in"], 9) if fP else "-",
             g(formation["dT_over_rs"], 4), g(formation["record_spacing_T_over_rs"], 4), g(fC["dR_t"], 4) if fC else "-", g(fC["dR_t_rel"], 3) if fC else "-",
             g(fC["dr_in"], 4) if fC else "-", g(fC["dE_rel"], 4) if fC else "-", g(fC["E_rad0_over_E_follow"], 4) if fC else "-",
             g(fC["E_inv_light_sourcing_over_E"], 4) if fC else "-", g(fC["E_inv_light_over_E"], 4) if fC else "-", nP, n,
             g(maxabs(D["r3.dR_t"]), 4), g(Trs[kabs], 6) if kabs is not None else "-", g(maxabs(D["r3.dE_rel"]), 4), g(Trs[kabsE], 6) if kabsE is not None else "-",
             g(at("F.E_rad0_over_E", kend), 4), g(Trs[kend], 6), g(formation["first_min_N_minus_absv_le_0"]["C"], 6), g(formation["first_min_N_minus_absv_le_0"]["production"], 6),
             len(dargs), "; ".join("%s: C %s, production %s" % (k, fm(va, 6), fm(vb, 6)) for k, va, vb in dargs[:12]) + (" ..." if len(dargs) > 12 else "")))
    s4 = ("Readout 4: M(r) outside the central cell (r > ell = %s) is smallest over the run at T/r_s %s: M_min_outside_cell = %s at r = %s; records with negative points: %d%s; "
          "the stop with its case: %s. The s11 npz cross-check: max |M_min(npz) - M_min(json)| = %s, argmin nodes differing by at most %s, negative-point counts differing by at most %s, "
          "rho_ell minima by %s. light_fed false on %d records, light_certified_on_record false on %d (case (ii) needs both certificates); the light fed source's smoothed minimum "
          "over the run %s at T/r_s %s; its integral equals eps_fed_sum to %s; pairs and dust integrals equal E_pairs_shells and E_dust_shells to %s and %s."
          % (g(ell, 4), g(Trs[kMmin], 6) if kMmin is not None else "-", g(at("s11.M_min_outside_cell", kMmin), 8), g(at("s11.r_at_M_min", kMmin), 6), len(neg_records),
             (" (at T/r_s %s)" % ", ".join(fm(x, 6) for x in neg_records[:8])) if neg_records else "",
             ("none: no stop key on any record (the run ended at its clock)" if not stop_records else
              "; ".join("T/r_s %s: %s" % (fm(s["T_over_rs"], 6), fm(s["values"], 6)) for s in stop_records)),
             g(idres("s11: M_min_outside_cell (json) = min over r > ell of M (npz)"), 3), g(idres("s11: r_at_M_min (json) = the argmin's node (npz)"), 3),
             g(idres("s11: negative_points_outside_cell (json) = #{M < 0, r > ell} (npz)"), 3), g(idres("s11: rho_ell_min_outside_cell (json) = min over r > ell of rho_ell (npz)"), 3),
             not_fed, not_cert_s11, g(at("s11.light_source_min_outside_cell", kLmin), 4), g(Trs[kLmin], 6) if kLmin is not None else "-",
             g(idres("C: s11.light_source_integral = light_v36.eps_fed_sum (the light fed source's total)"), 3), g(idres("C: s11.pairs_source_integral = E_pairs_shells"), 3),
             g(idres("C: s11.ball_source_integral = E_dust_shells (the heavy dust in the role of the ball)"), 3)))
    s5 = ("Readout 5 (the engine-side first order): on %d matched clocks the coupling-on run's E_inv_light_sourcing/E against the light-follow's E_rad0_over_E has relative "
          "difference (C - F)/F %s at T/r_s 20, %s at T/r_s %s, max |.| %s; E_inv over all channels %s at the end; per channel E_inv (C - F)/F at the end: %s; "
          "I_E_light_sourcing C %s against F %s; eps_fed_sum relative %s, J_fed_sum relative %s; H_in_light relative %s. The follow's records are the production's "
          "(E, R_t, min(N - |v|) equal at %d matched clocks to %s, %s, %s). Against Computation A's own readout on the r_s/a 32 production records: not among the inputs "
          "(the lead's); only this first order is printed."
          % (nF, g(rel20, 4), g(relend, 4), g(Trs[kend], 6), g(relmax, 4), g(at("r5.rel_E_inv_all", kend), 4),
             ", ".join("L%s %s" % (c, fm(at("r5.L%s.rel_E_inv" % c, kend), 4)) for c in common) or "-",
             g(at("light.I_E_light_sourcing", kend), 4), g(at("F.I_E_light_sourcing", kend), 4), g(at("r5.rel_eps_fed_sum", kend), 4), g(at("r5.rel_J_fed_sum", kend), 4),
             g(at("r5.rel_H_in_light", kend), 4), len(pairs_FP), g(idres("F vs production: E equal at matched clocks"), 3),
             g(idres("F vs production: R_t equal at matched clocks (null where null)"), 3), g(idres("F vs production: min_N_minus_absv equal at matched clocks"), 3)))
    s7 = ("Readout 7: heavy field: dHin = evolution_exact + field_work + reference_reset to %s (recorded closure %s), per_unit_time = parts/tau to %s, "
          "field_work = (Y2 - Y1) - (X2 - X1) to %s against the traces' floor %s, M1 dHin against the stored H_in differences to %s; sum of dHin over the run %s against "
          "H_in(T) - H_in(0) = %s; E_inv_heavy from %s to %s. Light field: the same identities to %s, %s, %s (floor %s), %s; sum of dHin %s against dH_in_light %s; "
          "E_inv_light from %s to %s (sourcing %s), dE_inv_light_dT read off as %s."
          % (g(idres("C heavy: M1_balance.dHin = evolution_exact + field_work + reference_reset"), 3), g(maxabs(D["M1_balance.closure"]), 3),
             g(max([idres("C heavy: M1_balance.per_unit_time.%s = %s / tau" % (q, qq)) or 0.0 for q, qq in zip(M1_PUT_KEYS, M1_PUT_FROM)]), 3),
             g(idres("C heavy: field_work = (Y2 - Y1) - (X2 - X1) (the relation read from the stored parts)"), 3),
             g(ids.d["C heavy: field_work = (Y2 - Y1) - (X2 - X1) (the relation read from the stored parts)"]["floor"], 3),
             g(idres("C heavy: M1_balance.dHin = H_in[k] - H_in[k-1] (the stored inside energies)"), 3), g(at("cum.M1_balance.dHin", kend), 6), g(at("dH_in", kend), 6),
             g(D["light.E_inv_heavy"][0], 9), g(at("light.E_inv_heavy", kend), 9),
             g(idres("C light: M1_balance.dHin = evolution_exact + field_work + reference_reset"), 3),
             g(max([idres("C light: M1_balance.per_unit_time.%s = %s / tau" % (q, qq)) or 0.0 for q, qq in zip(M1_PUT_KEYS, M1_PUT_FROM)]), 3),
             g(idres("C light: field_work = (Y2 - Y1) - (X2 - X1) (the relation read from the stored parts)"), 3),
             g(ids.d["C light: field_work = (Y2 - Y1) - (X2 - X1) (the relation read from the stored parts)"]["floor"], 3),
             g(idres("C light: M1_balance.dHin = H_in[k] - H_in[k-1] (the stored inside energies)"), 3), g(at("cum.light.M1_balance.dHin", kend), 6), g(at("dH_in_light", kend), 6),
             g(D["light.E_inv_light"][0], 6), g(at("light.E_inv_light", kend), 6), g(at("light.E_inv_light_sourcing", kend), 6),
             ids.d["C: light_v36.dE_inv_light_dT is a finite difference of E_inv_light"]["best"]))
    sA = ("Also printed: the subcritical check at T = 0: min(N - |v|) on the sites %s, dense %s at r = %s (zeros %s; the light-follow's head on its 1024-site chain %s); "
          "the first return on the T = 0 fields (r_s units): %s; chain end %s r_s. Truncation weight at the end (T/r_s %s): E_inv guard/sourcing %s, I_E %s, eps %s "
          "(recomputed from the channel entries to %s, %s, %s). Certificates: certified_on_record false on %d records, channel certificates not certified %d, fallbacks %d "
          "channel-records (why: %s), cut crossings %d; pieces per accepted evaluation at most %s, Cholesky calls at most %s; the record totals over evaluations sum to "
          "walk_pieces %s, walk_cholesky %s, walk_fallbacks %s, cut_crossings %s; labels: %s; modes: %s; heavy continuation labels: %s. The light threshold radii "
          "(threshold_radius_massless) are non-null on %s records per channel, channels.<k>.threshold_radius on %s."
          % (g(sub0["min_N_minus_absv_sites"], 10), g(sub0["min_N_minus_absv_dense"], 10), g(sub0["r_min"], 8), g(sub0["zeros"], 6), g(mF["head_subcritical_sites"], 10),
             "; ".join("%s %s" % (k, fm(first_return[k], 6)) for k in sorted(first_return) if k != "note"), g(chain_end, 4), g(Trs[kend], 6),
             g(at("light.truncation_guard.E_inv_guard_over_sourcing", kend), 5), g(at("light.truncation_guard.I_E_guard_over_sourcing", kend), 5),
             g(at("light.truncation_guard.eps_guard_over_sourcing", kend), 5),
             g(idres("C: truncation_guard.E_inv_guard_over_sourcing = guard E_inv_weighted / E_inv_light_sourcing"), 3),
             g(idres("C: truncation_guard.I_E_guard_over_sourcing = guard I_E_cut_weighted / I_E_light_sourcing"), 3),
             g(idres("C: truncation_guard.eps_guard_over_sourcing = guard eps_sum_weighted / eps_fed_sum (null where eps_fed_sum = 0)"), 3),
             sum(1 for x in D["light.certified_on_record"] if not x), not_cert, fb_count, ", ".join(whys) or "none", cx_count,
             ", ".join("L%s %s" % (c, fm(pieces_max[c], 1)) for c in chans), ", ".join("L%s %s" % (c, fm(chol_max[c], 1)) for c in chans),
             g(timing_sums["walk_pieces"], 1), g(timing_sums["walk_cholesky"], 1), g(timing_sums["walk_fallbacks"], 1), g(timing_sums["cut_crossings"], 1),
             " / ".join(labels), " / ".join(modes), " / ".join(cont_labels) or "-",
             ", ".join("L%s %d" % (c, thr_nonnull[c]) for c in chans), ", ".join("L%s %d" % (c, thr_ch_nonnull[c]) for c in chans)))

    # ---- readouts 1, 6, 8: what each needs
    args = mC["args"]
    full_clocks = split_list(args.get("slices_c128", ""))
    light_clocks = split_list(args.get("slices_light_c128", ""))
    written_full = [(Trs[k], D["slice_saved"][k]) for k in range(n) if D["slice_saved"][k]]
    written_light = [(Trs[k], D["light.light_slice_saved"][k]) for k in range(n) if D["light.light_slice_saved"][k]]
    found = {}
    if slice_dir:
        for c in sorted(set(full_clocks + light_clocks), key=float):
            found[c] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(slice_dir, "*_T%07.2f*.npz" % float(c))))
    not_built = {
        "1": ("S1 A items 8 to 12 on the self-consistent history (the collar flux against the comparator, E_inv and the creation rate, the collected spectrum "
              "and its soft bins, S_esc, the transfer ledger): the lead's page readout on this run's fields npz (N_applied, v_applied, eps_tot, J_tot per record), "
              "spectra npz (mask_L<k>, e_L<k>, nocc_L<k>) and the light slices (sea_L<k>, R2_L<k>, e_L<k>, mask_L<k>, sigma_L<k>, mu_L<k>, nplus_L<k>) at the slice "
              "clocks; its engine-side check values are the light_v36 entries of readout 5's columns"),
        "6": ("the held-field controls per field at the records of the first episode: the full complex128 slices at T/r_s 13 to 20 (both fields' columns, references, "
              "applied fields) propagated by the lead's held-field code on a chain whose first reflection lies beyond the hold, under the drain rule of P1"),
        "8": ("the escaping sector by the retained-channel recursion with the 4 repairs on the light field's channels: the solve lane's page_core on the light slices, "
              "the dense route on the same slices as its oracle; its peak against eta_2 A/a^2 and A/(4G) at g = %s (a^2 = 4G eta_1, eta_2 = eta_1 for the "
              "4-component light field: the 2 coincide at the printed spacing); A = 4 pi R_t^2 per record is printed here (readout 3's table, series 'A')" % fm(args.get("g"), 6)),
        "slice_clocks_full_c128": full_clocks, "slice_clocks_light_c128": light_clocks, "slice_dir_on_pod": args.get("slice_dir"),
        "records_with_slice_saved": [{"T_over_rs": t, "path": p} for t, p in written_full], "records_with_light_slice_saved": [{"T_over_rs": t, "path": p} for t, p in written_light],
        "slice_dir_checked_on_box": slice_dir, "slice_files_found_on_box": found}

    # ---- the md
    L = ["# Computation C: the readouts on the stored records (sigma/a 4 member), %s" % VERSION, "", LABEL, "",
         "Generated %s by %s. The spec: C_two_field_spec_v1.md sec. 7; readouts 2, 3, 4, 5, 7 and the 'also printed' list built on the stored records alone; "
         "1, 6, 8 not built (what each needs is stated at the end)." % (datetime.datetime.now(datetime.timezone.utc).isoformat(), VERSION), "", "## Inputs"]
    for d in inputs:
        L.append("- %s: %s (md5 %s, %d bytes)" % (d["role"], d["path"], d["md5"], d["bytes"]))
    L.append("- coupling-on run: script %s, tag %s, %d records, T/r_s %s to %s, tau %g, r_s/a %g, ell %g, nsites %s, light channels sourcing %s and guard %s, collars %s r_s"
             % (mC["script"], args.get("tag"), n, fm(Trs[0], 6), fm(Trs[-1], 6), tau, rs, ell, args.get("nsites"), ",".join(src), ",".join(grd), ",".join(collars)))
    L.append("- light-follow: script %s, %d records, nsites %s, followed %s (args.light_follow %s)" % (mF["script"], nf, mF["args"].get("nsites"), mF["followed"], mF["args"].get("light_follow")))
    L.append("- 1-field production: script %s, tag %s, %d records, nsites %s, mix %s" % (mP["script"], mP["args"].get("tag"), mP["n"], mP["args"].get("nsites"), mP["args"].get("mix")))
    L.append("- matched clocks on the coupling-on run's: light-follow %d of %d, production %d of %d; follow against production %d" % (nF, n, nP, n, len(pairs_FP)))
    L.append("- the s11 npz keys: %s" % ", ".join(s11_keys))
    L.append("- not read: the fields npz, the spectra npz, the handoff npz, the light-follow's npz (not needed by readouts 2, 3, 4, 5, 7)")
    L += ["", "## The table clocks", ""]
    for label, k, c in rows:
        L.append("- %s: %s" % (label, ("record %d (T %s)" % (k, fm(T[k], 6))) if k is not None else "absent from the records"))
    L += ["", "## Readout 2: E(t) by the collar identity, -Delta W against the light field's collar flux", "",
          "Keys: E, E_over_E0, dE_dT, M1_I_E (= M1_parts.I_E_start), light_v36.I_E_light, light_v36.I_E_light_sourcing (guard = difference), light_v36.M1_parts, "
          "light_v36.M1_balance (the spec's M1_parts_light, M1_balance_light). -Delta W read as -Delta E (E = W/2g at the collar, checked). Integrals: trapezoids in T "
          "(the backward rule for dE_dT). The per-channel I_E_cut is stored weighted only (I_E_cut_weighted; see readout 5's columns).", "",
          tab2a, "", "The light field's energy budget (cumulative over the records; light_v36.M1_balance):", "", tab2b, "", s2,
          "", "## Readout 3: the formation clock and radius against the 1-field production, the difference against E_rad^(0)/E", "",
          "Keys: R_t, r_in, trapping_zeros, min_N_minus_absv, T_over_rs, E (both runs); the light-follow's E_rad0_over_E; light_v36.E_inv_light_over_E and "
          "E_inv_light_sourcing_over_E. A = 4 pi R_t^2 from R_t (spec item 8's printed column).", "",
          "Args differing between the coupling-on run and the production (out, tag, gpu, slices, resume entries skipped): " +
          ("; ".join("%s: C %s, production %s" % (k, fm(va, 6), fm(vb, 6)) for k, va, vb in dargs) if dargs else "none"), "", tab3, "", s3,
          "", "## Readout 4: the sign of the mass function outside the central cell per record (s11)", "",
          "Keys: s11.{M_min_outside_cell, r_at_M_min, negative_points_outside_cell, compactness_max_2gM_over_r, source_min_outside_cell, r_at_source_min, "
          "ball_source_min_outside_cell (the heavy dust's), pairs_source_min_outside_cell, light_source_min_outside_cell, light_source_integral, light_fed, "
          "light_certified_on_record}; the s11 npz (T, r_nodes, M, rho_ell) for the cross-check over r > ell. The stop with its case: reported where a stop key exists; "
          "the digest's records carry none.", "", tab4, "", s4,
          "", "## Readout 5: the coupling-on run's light entries against the light-follow's on the same clocks (the engine-side first order)", "",
          "Keys: light_v36.{E_inv_light, E_inv_light_sourcing, E_inv_light_sourcing_over_E, I_E_light, I_E_light_sourcing, eps_fed_sum, J_fed_sum, H_in_light, "
          "channels.<k>.*} against the follow's {E_inv_light_all, E_inv_light_sourcing, E_rad0_over_E, I_E_light_all, I_E_light_sourcing, eps_fed_sum, J_fed_sum, "
          "channels.<k>.*}. The follow's I_E_cut and H_in are unweighted; the coupling-on run stores them weighted, divided by w here and labelled.", "",
          "5a, the totals:", "", tab5a, "", "5b, E_inv per channel (unweighted, both runs):", "", tab5b, "",
          "5c, the Killing-bin occupation deviations per channel (units of 1/a):", "", tab5c, "",
          "5d, the per-channel collar cut current (C: I_E_cut_weighted / w, derived) and the energy beyond the outermost collar:", "", tab5d, "", s5,
          "", "## Readout 7: the energy identity per field and E_inv per field", "",
          "Keys: M1_parts, M1_balance (heavy, version 35's), light_v36.M1_parts, light_v36.M1_balance (the light's), light_v36.E_inv_heavy, light_v36.E_inv_light, H_in, "
          "light_v36.H_in_light, light_v36.dE_inv_light_dT. The identities recomputed are in the table at the end.", "",
          "7a, the heavy field:", "", tab7a, "", "7b, the light field:", "", tab7b, "", s7,
          "", "## Also printed", "", "### The subcritical check and the first-return estimate (head.light_v36)", ""]
    L.append("- subcritical_T0: min(N - |v|) sites %s, dense %s, r_min %s, zeros %s" % (fm(sub0["min_N_minus_absv_sites"], 12), fm(sub0["min_N_minus_absv_dense"], 12), fm(sub0["r_min"], 10), fm(sub0["zeros"], 6)))
    for kk in sorted(first_return):
        L.append("- first_return.%s: %s" % (kk, fm(first_return[kk], 10)))
    L.append("- chain_end_over_rs: %s" % fm(chain_end, 6))
    for c in chans:
        L.append("- head channel %s: %s" % (c, fm(head_channels[c], 8)))
    L += ["", "### The truncation weight per record (light_v36.truncation_guard: the guard's E_inv, source total and collar current in ratio to the sourcing channels')", "", tabT,
          "", "### The certificate statistics per record (channels.<k>.certified, fallback, cut_crossing; light_v36.timing totals over the record's evaluations; the label)", "", tabC,
          "", "### The light field's threshold radii (light_v36.threshold_radius_massless, channels.<k>.threshold_radius)", "", tabR, "", sA,
          "", "## The identities recomputed from the stored keys", "",
          "kind: exact = an identity among stored keys (roundoff expected); relation = the relation read from the stored parts; hypothesis = a definition read off the "
          "records, the best alternative shown; cross = a cross-check between files or between the M1 arithmetic and the stored energies.", ""]
    L.append(mdtab(["identity", "kind", "n", "max |residual|", "floor", "note"],
                   [[nm, ids.d[nm]["kind"], str(ids.d[nm]["n"]), fm(ids.d[nm]["max_abs_residual"], 3), fm(ids.d[nm]["floor"], 3), ids.d[nm]["note"]] for nm in ids.order]))
    L += ["", "## Readouts 1, 6 and 8: not built here; what each needs", "",
          "- 1: %s." % not_built["1"], "- 6: %s." % not_built["6"], "- 8: %s." % not_built["8"],
          "- The slice clocks (the owner's dataset): full complex128 slices of both fields at T/r_s %s; light-only complex128 slices at T/r_s %s; written by the engine to %s on the pod "
          "(%d records carry slice_saved, %d carry light_slice_saved)." % (", ".join(full_clocks), ", ".join(light_clocks), args.get("slice_dir"), len(written_full), len(written_light)),
          "- On the box: " + (("--slice-dir %s: " % slice_dir + "; ".join("T/r_s %s: %s" % (c, ", ".join(fs) if fs else "none") for c, fs in found.items())) if slice_dir else
                              "not checked (no --slice-dir given); the slices are not among the inputs named on the box, so readouts 1, 6, 8 wait for their copy or for the lead's "
                              "codes to run where they are.")]
    md = "\n".join(L) + "\n"

    out = {"version": VERSION, "label": LABEL, "inputs": inputs, "n_records": n,
           "meta": {"tau": tau, "rs": rs, "ell": ell, "sourcing": src, "guard": grd, "channels": chans, "collars": collars, "run_script": mC["script"], "run_tag": args.get("tag"),
                    "follow_script": mF["script"], "followed": mF["followed"], "production_script": mP["script"], "production_tag": mP["args"].get("tag"),
                    "matched_clocks": {"follow": nF, "production": nP, "follow_vs_production": len(pairs_FP)}, "args_diff_C_vs_production": [list(x) for x in dargs],
                    "s11_npz_keys": s11_keys, "head_channels": head_channels, "head_subcritical_T0": sub0, "head_first_return": first_return, "head_chain_end_over_rs": chain_end,
                    "follow_head_subcritical_sites": mF["head_subcritical_sites"]},
           "table_rows": [{"label": lb, "k": k, "T_over_rs": c} for lb, k, c in rows], "formation": formation,
           "follow_T_over_rs": SF["T_over_rs"], "production_T_over_rs": SP["T_over_rs"],
           "readouts": {"2": {"summary": s2, "end": {"T_over_rs": Trs[kend], "minus_dE": at("minus_dE", kend), "int_M1_I_E": at("cum.int_M1_I_E", kend),
                                                    "int_I_E_light_sourcing": at("cum.int_I_E_light_sourcing", kend), "int_I_E_light_guard": at("cum.int_I_E_light_guard", kend),
                                                    "remainder": at("r2.remainder", kend), "light_M1_sums": {q: at("cum.light.M1_balance." + q, kend) for q in M1_BAL_KEYS[:5]}},
                              "trapezoid_sign": trap_sign, "dE_dT_definition": best_dE},
                        "3": {"summary": s3, "formation": formation},
                        "4": {"summary": s4, "k_M_min": kMmin, "records_with_negative_points_T_over_rs": neg_records, "stop_records": stop_records,
                              "light_fed_false": not_fed, "light_certified_false": not_cert_s11},
                        "5": {"summary": s5, "rel_E_inv_sourcing_over_E_at_20": rel20, "rel_E_inv_sourcing_over_E_at_end": relend, "max_abs_rel": relmax, "common_channels": common},
                        "7": {"summary": s7}, "also": {"summary": sA, "fallbacks": fb_count, "cut_crossings": cx_count, "whys": whys, "labels": labels, "modes": modes,
                                                       "pieces_max": pieces_max, "cholesky_max": chol_max, "timing_sums": timing_sums,
                                                       "threshold_radius_massless_nonnull": thr_nonnull, "channel_threshold_radius_nonnull": thr_ch_nonnull},
                        "not_built": not_built},
           "identities": ids.d, "identities_order": ids.order, "series": D}
    return out, md


def run_readout(run_json, s11_npz, follow_json, prod_json, out_dir, tag, slice_dir=None, quiet=False):
    t0 = datetime.datetime.now(datetime.timezone.utc)
    inputs = []
    for role, p in (("run_json", run_json), ("s11_npz", s11_npz), ("follow_json", follow_json), ("prod_json", prod_json)):
        if not os.path.isfile(p):
            raise ReadoutError("%s: not a file: %s" % (role, p))
        inputs.append({"role": role, "path": os.path.abspath(p), "md5": md5_of(p), "bytes": os.path.getsize(p)})
    run = load_json(run_json)
    SC, mC = extract_C(run, "run")
    del run
    fol = load_json(follow_json)
    SF, mF = extract_F(fol, "follow")
    del fol
    prod = load_json(prod_json)
    SP, mP = extract_P(prod, "production")
    del prod
    s11x, s11_keys = s11_cross(mC, s11_npz, "s11_npz")
    out, md = compute(SC, mC, SF, mF, SP, mP, s11x, s11_keys, inputs, slice_dir)
    out["generated_utc"] = t0.isoformat()
    out["seconds"] = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    os.makedirs(out_dir, exist_ok=True)
    pj, pm = os.path.join(out_dir, tag + ".json"), os.path.join(out_dir, tag + ".md")
    with open(pj, "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    with open(pm, "w") as f:
        f.write(md)
    out["written"] = {"json": pj, "md": pm}
    if not quiet:
        print(md)
        print("written: %s and %s (%.1f s)" % (pj, pm, out["seconds"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Computation C: the readouts of spec sec. 7 on the stored records alone (sigma/a 4 member).")
    ap.add_argument("--run-json", required=True, help="the coupling-on run's record json (engine_twofield_v3)")
    ap.add_argument("--s11-npz", required=True, help="its _s11 npz")
    ap.add_argument("--follow-json", required=True, help="the light-follow's json (gate F's run on the production's applied fields)")
    ap.add_argument("--prod-json", required=True, help="the 1-field production's json (engine_collapse_v31, read in place)")
    ap.add_argument("--out-dir", default="busi/results/compC")
    ap.add_argument("--tag", default="C_readout_s4_v1")
    ap.add_argument("--slice-dir", default=None, help="optional: list which slice clocks exist there (readouts 1, 6, 8 are not built)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        run_readout(a.run_json, a.s11_npz, a.follow_json, a.prod_json, a.out_dir, a.tag, a.slice_dir, a.quiet)
    except ReadoutError as e:
        print("STOP (%s): %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
