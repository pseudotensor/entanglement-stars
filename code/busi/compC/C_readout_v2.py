#!/usr/bin/env python
"""C_readout_v2.py (the busi lane, Computation C): version 1 repeated by import (C_readout_v1.py beside this file, never edited) and, new,
the sigma/a 4 member read against its coupling-off control on the pathfinder's own chain, solver and kernel (pod_launch_C_v7.sh s4off, tag
C32_s4_ell4_tau2_twofield_v3_off): with --light-coupling off the light channels propagate and are read out and source nothing, so the control
is Computation A on that chain. CPU (at most 8 threads), minutes, no figure.

Inputs: --run-json, --s11-npz (the coupled member, as version 1), --control-json, --control-fields-npz, --control-s11-npz (the control's
record json, fields npz, s11 npz; the same keys as the coupled member's, C_records_digest_v1.json), --follow-json, --prod-json (version 1's,
information only). Outputs <out-dir>/<tag>.json and .md; version 1's own json and md are written beside as <tag>_v1part.*.

(a) C minus control at matched clocks (the same record clock T = k tau; both runs 641 records unless --allow-partial): the formation clock and
    R_t; the outer root and min(N - |v|); E, E_inv per field, the collar currents per field (both runs cut at the same collar, the chain's
    end); the mass function's outside-cell minimum (the json's engine value and the two s11 npz at common nodes); every difference series
    to the json, tables at T/r_s 0 (the preparation check), the formation, 16, 20, 24, 28, 32, 36, 40. This replaces readouts 3 and 5's
    comparison against the 1024-site plain-mixer production and the light-follow on it (version 1's, appended as information).
    Keys (both jsons): T_over_rs, R_t, r_in, trapping_zeros, min_N_minus_absv, r_min_N_minus_absv, E, E_over_E0, kappa_T, H_in, M1_I_E,
    r_collar, light_v36.{E_inv_heavy, E_inv_light, E_inv_light_sourcing, I_E_light, I_E_light_sourcing, H_in_light,
    E_inv_light_sourcing_over_E (the control's is E_rad^(0)/E on the solver-matched history), channels.<k>.{E_inv, E_inv_weighted,
    I_E_cut_weighted, eps_sum, J_sum, w, sourcing}}, s11.{M_min_outside_cell, r_at_M_min, negative_points_outside_cell,
    compactness_max_2gM_over_r, light_fed, light_source_integral, ball_source_integral, pairs_source_integral}; the s11 npz (r_nodes, M).
(b) The bound's comparator (TWOFIELD_feedback_bound_v1.tex, hreduce, CF3 to CF5 and its red team) from the control's stored fields, per
    record with a dense root R (the outer root R_t, the inner r_in beside):
      I_L(R) = 4 pi int_0^R r^2 N_H (rho_L^A + u_H J_L^A) dr,   delta R_L = 2 G I_L(R) / (N_H(R) D_R) = G I_L(R) / (R kappa_R),
      D_R = 1 - 8 pi G R^2 (rho_H - J_H) = 1 - 2 G M_H'(R),   kappa_R = N_H(R) D_R / (2 R) = (N_H + v_H)'(R),   u_H = v_H / N_H,
    G = args.g (the records' R_t = 2 g M at the root; R_t_over_2g_MS = 1 is checked).
    rho_L^A, J_L^A: the control's fields npz eps_L<k>, J_L<k> for k in args.light_kappas (the sourcing channels; kappa 3, the guard, never
      enters), weighted by weights_light where the arrays are stored unweighted. Which: read off the records, the site sums against
      light_v36.channels.<k>.eps_sum and w eps_sum (J against J_sum); eps_light_fed, J_light_fed printed beside (zero when nothing is fed).
    N_H, v_H: the fields npz N, v (the dense fields on the sites R). rho_H, J_H: eps_tot, J_tot less the light part (eps_light_fed), read
      off against E_dust_shells + E_pairs_shells; the per-site arrays are shell quantities (their site sums are the stored totals), so
      4 pi r^2 rho dr -> the shell's content and M'(R) -> the shell's content per site spacing; the alternative (volume densities) is tested
      and rejected or accepted on the same totals. The sign with which u J enters the row is read off the control's own records: the root
      condition 2 g M_H(R)/R = 1 at its dense roots from the site densities, and the s11 npz's stored M against cumtrapz 4 pi r^2
      (rho_ell + s u J_ell) at the nodes; the two must agree.
    R: records[k].trapping_zeros; kappa_R: records[k].kappa_T (the engine's Richardson-refined (N + v)'(R_t)), (N + v)' from the sites
      beside; D_R from the heavy densities beside (the CF4 cross-check). Three evaluations of I_L: the stored site densities (primary);
      the s11 npz's engine-smoothed rho_ell_light_fed, J_ell_light_fed at the nodes where populated; the site densities smoothed by this
      script's 3-D Gaussian of standard deviation ell (an estimate of the smoothing sensitivity, not the engine's kernel). The measured
      R_t(C) - R_t(control) at the same clock beside delta R_L, their ratio; the positive-mass margin per record (CF5: M_H - [-M_L]_+
      outside the central cell r > ell, M_H the control's stored M, M_L = 4 pi int r^2 (rho_L + u J_L) at the control's u; the stronger
      allowance B_M = N^-1 int 4 pi r^2 N (|rho_L| + |u||J_L|) beside; no eps_M is supplied by the bound: --eps-M optional); the same split
      on C's actual fields from its s11 npz components (rho_ell_ball + rho_ell_pairs, rho_ell_light_fed, the same for J).
    Any of these keys absent, or no convention reproducing the stored totals and the root condition: part (b) stops loudly (the md and the
    json carry the stop, the exit code is 3) while (a) and (c) still print. Nothing is substituted.
(c) The reading rule, stated before the numbers (RULE below; also in the json).
Not read: the coupled member's fields npz and spectra npz, the control's spectra and handoff npz (not needed)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import datetime
import json
import math
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import C_readout_v1 as V1  # noqa: E402  version 1, imported and never edited
from C_readout_v1 import (ReadoutError, MissingKey, Inconsistent, need, opt, av, ratio, rel, fm, mdtab, md5_of, load_json,  # noqa: E402
                          _jsonable, LABEL, align, split_list, maxabs, argmaxabs, find_clock, build_table, Identities, Series)

VERSION = "C_readout_v2"
BOUND = "TWOFIELD_feedback_bound_v1.tex (hreduce), CF3 to CF5, with its red team (TF1 to TF5)"
TABLE_CLOCKS = (16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0)
FLOOR_DR_T = 3.4e-6   # lattice units: the regression gate's R_t floor, the solver-matched resolution of R_t(C) - R_t(control)
TOL_EXACT = 1e-6      # a stored total reproduced from the stored densities
TOL_NODE = 0.3        # the node trapezoid against the engine's M (the rule's own error is percents; the two signs differ by tens of percent)
TOL_ROOT = 0.2        # median |2 g M_H(R)/R - 1| at the control's dense roots from the site densities
EXPECTED_ARG_DIFF = ("light_coupling",)

RULE = [
    ("1. What is compared.",
     "Delta R_t := R_t(C) - R_t(control) at the same record clock T = k tau (the asymptotic clock of both solves, N(infinity) = 1) and the same "
     "Eulerian radius is the measured displacement of the dense outer root by the light source at the physical coupling (lambda = 1), on "
     "identical chain, kernel, preparation, engine and solver; its floor is the solver-matched resolution of R_t, 3.4e-6 lattice units (the "
     "regression gate's R_t floor). delta R_L := 2 G I_L(R) / (N_H(R) D_R) = G I_L(R) / (R kappa_R), with I_L(R) = 4 pi int_0^R r^2 N_H "
     "(rho_L^A + u_H J_L^A) dr, is the bound's source-only root coefficient (CF4): the first-order response of the root to the light source "
     "alone, evaluated with the control's (Computation A's) light densities on the control's dense fields; kappa_R = (N_H + v_H)'(R) is the "
     "record's kappa_T, D_R = 1 - 8 pi G R^2 (rho_H - J_H) from the heavy densities is printed as the CF4 cross-check, not as the primary "
     "evaluation; G = g = args.g (R_t = 2 g M at the root). At the outer root D_R > 0 and kappa_R > 0, so delta R_L has the sign of I_L: added "
     "enclosed light energy moves the outer root outward; at the inner root D_R < 0."),
    ("2. What agreement means at first order.",
     "The complete first-order coefficient replaces s_L^A in I_L by s_L^A + S_H Z - X_H psi: the heavy propagation response and the "
     "continued-reference response are of the same order as the source term, are not readable from stored densities, and are bounded by "
     "neither E_rad/E nor the light's total source. So Delta R_t = delta R_L + (those two first-order terms) + (the second-order remainder at "
     "lambda = 1) + (numerical defects: the kept residual, the record spacing, the site integration and the smoothing), and delta R_L is a "
     "consistency check, not an identity. Agreement means: the same sign at the formation and a ratio Delta R_t / delta R_L within an order of "
     "magnitude; that says the source-only term has the size and the sign of the measured displacement and the omitted terms do not reverse "
     "it. It does not say the omitted terms are small (a ratio of 0.3 or of 3 is agreement under this rule), it does not bound the feedback, "
     "and it does not make delta R_L the back-reaction; the measured Delta R_t is the back-reaction on the root at lambda = 1 and nothing "
     "else is."),
    ("3. What counts as a conflict for the lead.",
     "(i) A sign difference at the formation: at the control's formation record (its first record carrying a dense root) sign(Delta R_t) != "
     "sign(delta R_L), both above their floors. If C carries no root at that record, the formation shift stands in for the sign: delta R_L > "
     "0 (the trapped region should appear no later) with C forming later, or delta R_L < 0 with C forming earlier, is the sign conflict; the "
     "clock's resolution is one record, tau/r_s = 0.0625, and an equal formation clock is a null shift. (ii) An order-of-magnitude mismatch "
     "where both exceed the floor: |log10 |Delta R_t / delta R_L|| > 1 at the formation record or, when the formation clocks differ, at the "
     "first record where both runs carry the root. The floors: 3.4e-6 lattice units for Delta R_t; for delta R_L the spread of its own "
     "evaluations (the site densities against the engine-smoothed node densities and against the site densities smoothed by this script) plus "
     "roundoff. Below either floor the test is undecided and is printed as undecided, not as agreement. A conflict is a finding for the lead: "
     "the first-order comparator and the measured displacement disagree; it does not by itself say whether the omitted first-order terms, the "
     "remainder, the comparator's own approximations or a defect in C or in the control carries it. The per-record ratios across the episode, "
     "the inner root and the margins are information beside the verdict, not verdicts."),
    ("4. The margin.",
     "On the control's fields M_H is the solved mass function (the heavy field alone sources it) and M_L(r) = 4 pi int_0^r s^2 (rho_L^A + u_H "
     "J_L^A) ds is the light field's first-order enclosed contribution at the same u = v_H / N_H (CF5's split at a fixed u); M_H - [-M_L]_+ > "
     "eps_M outside the central cell (r > ell) is the bound's sufficient positive margin against an enclosed-mass error eps_M that the bound "
     "leaves unquantified: no eps_M is supplied, the margin is printed against 0 and against --eps-M when given, and the stronger allowance "
     "B_M = N_H^-1 int 4 pi s^2 N_H (|rho_L| + |u_H| |J_L|) ds is printed beside. The same split on C's actual fields comes from its stored "
     "smoothed components. A negative margin is not a stop diagnosis of this readout (the engine's P2 rule read M > 0 on every record of C); it "
     "would say the first-order light contribution is not small against the heavy mass there, and the bound's three diagnoses (a numerical "
     "defect in M, an incorrect reference changing its enclosed source, verified M < 0 on the correctly referenced branch) are the run's "
     "questions, not this readout's."),
]

REC_NEED = ("T", "T_over_rs", "E", "E_over_E0", "R_t", "r_in", "trapping_zeros", "min_N_minus_absv", "r_min_N_minus_absv", "M1_I_E", "H_in",
            "E_dust_shells", "E_pairs_shells")
REC_OPT = ("kappa_T", "N_min", "R_c", "converged", "iterations", "continuation_label", "R_t_over_2g_MS", "misner_sharp_at_Rt", "r_collar")
S11_NEED = ("M_min_outside_cell", "r_at_M_min", "negative_points_outside_cell", "light_fed", "light_source_integral", "ball_source_integral",
            "pairs_source_integral")
S11_OPT = ("compactness_max_2gM_over_r", "source_min_outside_cell", "light_source_min_outside_cell", "light_certified_on_record",
           "rho_ell_min_outside_cell")
LV_NEED = ("E_inv_heavy", "E_inv_light", "E_inv_light_sourcing", "I_E_light", "I_E_light_sourcing", "H_in_light", "coupling", "certified_on_record")
LV_OPT = ("eps_fed_sum", "J_fed_sum", "fed", "label", "E_inv_light_over_E", "E_inv_light_sourcing_over_E")
CH_NEED = ("E_inv", "E_inv_weighted", "I_E_cut_weighted", "eps_sum", "J_sum", "w", "sourcing")
CH_OPT = ("eps_sum_weighted", "fallback", "cut_crossing", "H_in_weighted")
FIELDS_NEED = ("R", "T", "N", "v", "eps_tot", "J_tot", "eps_light_fed", "J_light_fed", "kappas_light", "weights_light", "light_sourcing")
FIELDS_OPT = ("g", "ell", "tau", "light_meta", "eps_pairs", "J_pairs")
S11_NPZ_NEED = ("T", "r_nodes", "rho_ell", "J_ell", "N", "v", "M", "rho_ell_ball", "J_ell_ball", "rho_ell_pairs", "J_ell_pairs",
                "rho_ell_light_fed", "J_ell_light_fed")
S11_NPZ_OPT = ("W", "v_positive_branch")
A_NAMES = ("R_t", "r_in", "min_N_minus_absv", "r_min_N_minus_absv", "E", "E_over_E0", "H_in", "M1_I_E", "kappa_T", "N_min", "R_c",
           "light.E_inv_heavy", "light.E_inv_light", "light.E_inv_light_sourcing", "light.I_E_light", "light.I_E_light_sourcing",
           "light.H_in_light", "light.eps_fed_sum", "light.J_fed_sum", "light.E_inv_light_sourcing_over_E", "light.E_inv_light_over_E",
           "s11.M_min_outside_cell", "s11.r_at_M_min", "s11.negative_points_outside_cell", "s11.compactness_max_2gM_over_r",
           "s11.light_source_integral")
A_REL = ("E", "H_in", "light.E_inv_heavy", "light.E_inv_light", "light.E_inv_light_sourcing", "light.H_in_light", "s11.M_min_outside_cell")


# ------------------------------------------------------------------------------------------------------- arithmetic on the stored grids
def interp1(x, y, x0):
    return float(np.interp(x0, x, y))


def cum0(f):
    """[0, f_0, f_0 + f_1, ...]: the content of the shells 0..j-1."""
    return np.concatenate(([0.0], np.cumsum(np.asarray(f, dtype=float))))


def shell_edges(R):
    """The site spacing and the shell edges R_n -+ a/2 (the shell of site n carries its stored content, uniform within the shell)."""
    R = np.asarray(R, dtype=float)
    a = float(R[1] - R[0])
    return a, np.concatenate(([R[0] - 0.5 * a], R + 0.5 * a))


def enclosed(edges, c0, r):
    """The content inside r: the cumulative at the shell edges, linear within a shell (r scalar or array)."""
    return np.interp(r, edges, c0)


def cumtrapz0(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(x.shape[0])
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out


def rel_resid(a, b):
    """max |a - b| / max |b| over the finite entries (None when nothing is finite; 0 when both are identically zero)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return None, 0.0
    scale = float(np.max(np.abs(b[ok])))
    if scale == 0.0:
        scale = float(np.max(np.abs(a[ok])))
    if scale == 0.0:
        return 0.0, 0.0
    return float(np.max(np.abs(a[ok] - b[ok]))) / scale, scale


def gaussian_shell_kernel(R, sigma):
    """The 3-D isotropic Gaussian of standard deviation sigma on a spherical density, acting on shell contents e_n = 4 pi s_n^2 rho(s_n) a:
    (K e)_m = a sum_n e_n (r_m / s_n) [exp(-(r_m - s_n)^2 / 2 sigma^2) - exp(-(r_m + s_n)^2 / 2 sigma^2)] / (sigma sqrt(2 pi)); the column
    sums are 1 away from the chain's end (the total is preserved). An estimate of the engine's kernel, not the engine's kernel."""
    R = np.asarray(R, dtype=float)
    a = float(R[1] - R[0])
    r = R[:, None]
    s = R[None, :]
    pref = a / (sigma * math.sqrt(2.0 * math.pi))
    with np.errstate(divide="ignore", invalid="ignore"):
        core = np.exp(-(r - s) ** 2 / (2.0 * sigma ** 2)) - np.exp(-(r + s) ** 2 / (2.0 * sigma ** 2))
        K = pref * (r / s) * core
    lim = pref * 2.0 * r ** 2 * np.exp(-r ** 2 / (2.0 * sigma ** 2)) / sigma ** 2
    return np.where(s < 1e-12, lim, K)


def root_list(S, j):
    zs = S.d["trapping_zeros"][j]
    out = sorted(set(float(z) for z in (zs or []) if av(z)))
    if not out and av(S.d["R_t"][j]):
        out = [float(S.d["R_t"][j])]
    return out


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ----------------------------------------------------------------------------------------------------------------------- extraction
def extract_min(run, where):
    args = need(run, ("args",), where)
    head = need(run, ("head",), where)
    R = need(run, ("records",), where)
    if not isinstance(R, list) or not R:
        raise Inconsistent("%s: 'records' is not a non-empty list" % where)
    wa = where + ".args"
    tau = float(need(args, ("tau",), wa))
    rs = float(need(args, ("rs",), wa))
    ell = float(need(args, ("ell",), wa))
    g = float(need(args, ("g",), wa))
    src = split_list(need(args, ("light_kappas",), wa))
    grd = split_list(need(args, ("light_guard_kappas",), wa))
    chans = src + grd
    if not chans:
        raise Inconsistent("%s: no light channels in the args" % where)
    coupling = need(args, ("light_coupling",), wa)
    n = len(R)
    S = Series(n)
    for k, r in enumerate(R):
        w = "%s.records[%d]" % (where, k)
        for q in REC_NEED:
            S.put(q, k, need(r, (q,), w))
        for q in REC_OPT:
            S.put(q, k, opt(r, (q,)))
        S.put("solver.residual_W", k, opt(r, ("solver", "residual_W")))
        S.put("solver.residual_invN", k, opt(r, ("solver", "residual_invN")))
        S.put("solve.kept_residual", k, opt(r, ("solve_v32", "kept_residual")))
        s11 = need(r, ("s11",), w)
        for q in S11_NEED:
            S.put("s11." + q, k, need(s11, (q,), w + ".s11"))
        for q in S11_OPT:
            S.put("s11." + q, k, opt(s11, (q,)))
        lv = need(r, ("light_v36",), w)
        wl = w + ".light_v36"
        for q in LV_NEED:
            S.put("light." + q, k, need(lv, (q,), wl))
        for q in LV_OPT:
            S.put("light." + q, k, opt(lv, (q,)))
        chd = need(lv, ("channels",), wl)
        for c in chans:
            ch = need(chd, (c,), wl + ".channels")
            wc = wl + ".channels." + c
            for q in CH_NEED:
                S.put("L%s.%s" % (c, q), k, need(ch, (q,), wc))
            for q in CH_OPT:
                S.put("L%s.%s" % (c, q), k, opt(ch, (q,)))
    meta = {"tau": tau, "rs": rs, "ell": ell, "g": g, "sourcing": src, "guard": grd, "channels": chans, "n": n, "args": args,
            "coupling": coupling, "script": opt(run, ("script",)), "head_E0": opt(head, ("E0",)), "tag": args.get("tag"), "nsites": args.get("nsites")}
    return S, meta


def mapping_to_dict(z, keys, need_keys, opt_keys, where):
    out = {"_keys": list(keys)}
    for key in need_keys:
        if key not in keys:
            raise MissingKey("%s: the key '%s' is absent (keys present: %s); the part that needs it stops, nothing is substituted"
                             % (where, key, ", ".join(str(q) for q in keys)))
        out[key] = np.asarray(z[key])
    for key in opt_keys:
        out[key] = np.asarray(z[key]) if key in keys else None
    return out


def fields_from_mapping(z, keys, where):
    out = mapping_to_dict(z, keys, FIELDS_NEED, FIELDS_OPT, where)
    for c in [int(x) for x in np.asarray(out["kappas_light"]).ravel()]:
        for pre in ("eps_L", "J_L"):
            key = "%s%d" % (pre, c)
            if key not in keys:
                raise MissingKey("%s: the key '%s' (the light channel kappa %d's %s density) is absent (keys present: %s); part (b) stops, "
                                 "nothing is substituted" % (where, key, c, "energy" if pre == "eps_L" else "momentum", ", ".join(str(q) for q in keys)))
            out[key] = np.asarray(z[key], dtype=float)
    for key in ("N", "v", "eps_tot", "J_tot", "eps_light_fed", "J_light_fed"):
        out[key] = np.asarray(out[key], dtype=float)
    out["light_meta_str"] = str(out["light_meta"]) if out["light_meta"] is not None else None
    return out


def load_fields(path, where):
    z = np.load(path, allow_pickle=False)
    return fields_from_mapping(z, list(z.files), where)


def s11_from_mapping(z, keys, where):
    out = mapping_to_dict(z, keys, S11_NPZ_NEED, S11_NPZ_OPT, where)
    for key in S11_NPZ_NEED:
        out[key] = np.asarray(out[key], dtype=float)
    return out


def load_s11(path, where):
    z = np.load(path, allow_pickle=False)
    return s11_from_mapping(z, list(z.files), where)


# ------------------------------------------------------------------------------------------------------------------------ part (a)
def compute_a(SC, mC, SK, mK, iK, rev, pairs, follow_E_rad0=None):
    nC, nK = mC["n"], mK["n"]
    TrsC, TrsK = SC.d["T_over_rs"], SK.d["T_over_rs"]
    chans = mC["channels"]
    names = list(A_NAMES) + ["L%s.%s" % (c, q) for c in chans for q in ("E_inv", "E_inv_weighted", "I_E_cut_weighted", "eps_sum", "J_sum")]
    D = {}
    for nm in names:
        colC = SC.d.get(nm, [None] * nC)
        colK = SK.d.get(nm, [None] * nK)
        D["C." + nm] = list(colC)
        D["K." + nm] = [colK[j] if j is not None else None for j in iK]
        D["d." + nm] = [x - y if av(x, y) else None for x, y in zip(D["C." + nm], D["K." + nm])]
    for nm in A_REL:
        D["rel." + nm] = [rel(x, y) for x, y in zip(D["C." + nm], D["K." + nm])]
    for pre in ("C.", "K."):
        D[pre + "light.I_E_light_guard"] = [x - y if av(x, y) else None for x, y in zip(D[pre + "light.I_E_light"], D[pre + "light.I_E_light_sourcing"])]
        D[pre + "A"] = [4.0 * math.pi * r * r if av(r) else None for r in D[pre + "R_t"]]
    D["d.light.I_E_light_guard"] = [x - y if av(x, y) else None for x, y in zip(D["C.light.I_E_light_guard"], D["K.light.I_E_light_guard"])]
    for c in chans:
        D["rel.L%s.E_inv" % c] = [rel(x, y) for x, y in zip(D["C.L%s.E_inv" % c], D["K.L%s.E_inv" % c])]
    D["F.E_rad0_over_E"] = list(follow_E_rad0) if follow_E_rad0 is not None else [None] * nC
    D["matched"] = [j is not None for j in iK]
    # ---- the formation and the first episode
    trC = [av(x) for x in SC.d["R_t"]]
    trK = [av(x) for x in SK.d["R_t"]]
    kfC = next((k for k in range(nC) if trC[k]), None)
    jfK = next((j for j in range(nK) if trK[j]), None)

    def episode_end(tr, k0):
        if k0 is None:
            return None
        k = k0
        while k + 1 < len(tr) and tr[k + 1]:
            k += 1
        return k
    keC, jeK = episode_end(trC, kfC), episode_end(trK, jfK)
    kfK_C = rev.get(jfK) if jfK is not None else None
    keK_C = rev.get(jeK) if jeK is not None else None
    k0C = next((k for k in range(nC) if av(SC.d["min_N_minus_absv"][k]) and SC.d["min_N_minus_absv"][k] <= 0.0), None)
    j0K = next((j for j in range(nK) if av(SK.d["min_N_minus_absv"][j]) and SK.d["min_N_minus_absv"][j] <= 0.0), None)
    both = [k for k, j in pairs if trC[k] and trK[j]]
    formation = {
        "C": ({"k": kfC, "T_over_rs": TrsC[kfC], "R_t": SC.d["R_t"][kfC], "r_in": SC.d["r_in"][kfC], "trapping_zeros": SC.d["trapping_zeros"][kfC],
               "kappa_T": SC.d["kappa_T"][kfC], "E": SC.d["E"][kfC], "min_N_minus_absv": SC.d["min_N_minus_absv"][kfC]} if kfC is not None else None),
        "control": ({"j": jfK, "k_C": kfK_C, "T_over_rs": TrsK[jfK], "R_t": SK.d["R_t"][jfK], "r_in": SK.d["r_in"][jfK], "trapping_zeros": SK.d["trapping_zeros"][jfK],
                     "kappa_T": SK.d["kappa_T"][jfK], "E": SK.d["E"][jfK], "min_N_minus_absv": SK.d["min_N_minus_absv"][jfK]} if jfK is not None else None),
        "dT_over_rs_C_minus_control": (TrsC[kfC] - TrsK[jfK]) if (kfC is not None and jfK is not None) else None,
        "dT_records": (int(round((TrsC[kfC] - TrsK[jfK]) / (mC["tau"] / mC["rs"]))) if (kfC is not None and jfK is not None) else None),
        "record_spacing_T_over_rs": mC["tau"] / mC["rs"],
        "episode_C": {"first_k": kfC, "last_k": keC, "records": (keC - kfC + 1) if kfC is not None else 0,
                      "last_T_over_rs": TrsC[keC] if keC is not None else None},
        "episode_control": {"first_j": jfK, "last_j": jeK, "records": (jeK - jfK + 1) if jfK is not None else 0,
                            "last_T_over_rs": TrsK[jeK] if jeK is not None else None},
        "first_min_N_minus_absv_le_0": {"C": TrsC[k0C] if k0C is not None else None, "control": TrsK[j0K] if j0K is not None else None},
        "records_with_root_in_both": len(both), "first_common_root_k": (both[0] if both else None),
        "dR_t_at_control_formation": D["d.R_t"][kfK_C] if kfK_C is not None else None,
        "dR_t_rel_at_control_formation": (rel(D["C.R_t"][kfK_C], D["K.R_t"][kfK_C]) if kfK_C is not None else None),
        "dr_in_at_control_formation": D["d.r_in"][kfK_C] if kfK_C is not None else None,
        "dE_rel_at_control_formation": D["rel.E"][kfK_C] if kfK_C is not None else None,
        "dR_t_at_C_formation": D["d.R_t"][kfC] if kfC is not None else None,
    }
    kabs = argmaxabs(D["d.R_t"])
    kabsE = argmaxabs(D["rel.E"])
    signs = [sgn(x) for x in D["d.R_t"] if av(x) and x != 0.0]
    kt20, kt40, kt0 = find_clock(TrsC, 20.0), find_clock(TrsC, 40.0), find_clock(TrsC, 0.0)

    def at(name, k):
        return D[name][k] if (k is not None and name in D) else None
    collars = (sorted(set(str(x) for x in SC.d["r_collar"] if x is not None)), sorted(set(str(x) for x in SK.d["r_collar"] if x is not None)))
    ctrl_conv = {"converged_true": sum(1 for x in SK.d["converged"] if x), "converged_false": sum(1 for x in SK.d["converged"] if x is False),
                 "converged_absent": sum(1 for x in SK.d["converged"] if x is None), "max_kept_residual": maxabs(SK.d["solve.kept_residual"]),
                 "certified_false": sum(1 for x in SK.d["light.certified_on_record"] if not x),
                 "fallbacks": sum(1 for c in mK["channels"] for x in SK.d.get("L%s.fallback" % c, []) if x),
                 "light_fed_true": sum(1 for x in SK.d["s11.light_fed"] if x), "coupling_json": sorted(set(str(x) for x in SK.d["light.coupling"])),
                 "iterations_max": maxabs(SK.d["iterations"])}
    summary = {"formation": formation, "max_abs_dR_t": maxabs(D["d.R_t"]), "T_at_max_abs_dR_t": TrsC[kabs] if kabs is not None else None,
               "dR_t_sign_counts": {"positive": signs.count(1), "negative": signs.count(-1)},
               "max_abs_dE_rel": maxabs(D["rel.E"]), "T_at_max_abs_dE_rel": TrsC[kabsE] if kabsE is not None else None,
               "prep_T0": {"dE": at("d.E", kt0), "d_min_N_minus_absv": at("d.min_N_minus_absv", kt0), "d_E_inv_heavy": at("d.light.E_inv_heavy", kt0),
                           "d_E_inv_light": at("d.light.E_inv_light", kt0), "d_H_in": at("d.H_in", kt0)},
               "at_20": {"rel_E_inv_light": at("rel.light.E_inv_light", kt20), "rel_E_inv_light_sourcing": at("rel.light.E_inv_light_sourcing", kt20),
                         "rel_E": at("rel.E", kt20), "d_min_N_minus_absv": at("d.min_N_minus_absv", kt20)},
               "at_40": {"rel_E_inv_light": at("rel.light.E_inv_light", kt40), "rel_E_inv_light_sourcing": at("rel.light.E_inv_light_sourcing", kt40),
                         "rel_E": at("rel.E", kt40), "d_min_N_minus_absv": at("d.min_N_minus_absv", kt40), "d_M_min": at("d.s11.M_min_outside_cell", kt40),
                         "M1_I_E_C": at("C.M1_I_E", kt40), "M1_I_E_control": at("K.M1_I_E", kt40), "I_E_light_sourcing_C": at("C.light.I_E_light_sourcing", kt40),
                         "I_E_light_sourcing_control": at("K.light.I_E_light_sourcing", kt40)},
               "collars": {"C": collars[0], "control": collars[1]}, "control_convergence": ctrl_conv,
               "max_abs_d_M_min_json": maxabs(D["d.s11.M_min_outside_cell"]), "negative_points_control": sum(1 for x in D["K.s11.negative_points_outside_cell"] if av(x) and x > 0)}
    return D, formation, summary


def s11_common_nodes(ZC, ZK, pairs, ell, TrsC):
    rC, MC, rK, MK = ZC["r_nodes"], ZC["M"], ZK["r_nodes"], ZK["M"]
    out = {"k_C": [], "T_over_rs": [], "nodes_equal": [], "min_M_C_outside": [], "min_M_K_outside": [], "d_min_M_outside": [],
           "max_abs_dM_outside": [], "r_at_max_abs_dM": [], "min_dM_outside": [], "max_dM_outside": [], "first_node_outside": []}
    for k, j in pairs:
        rc, rk = rC[k], rK[j]
        eq = bool(rc.shape == rk.shape and np.allclose(rc, rk))
        Mk = MK[j] if eq else np.interp(rc, rk, MK[j])
        mask = rc > ell
        if not mask.any():
            raise Inconsistent("s11 npz: record %d has no node with r > ell = %g" % (k, ell))
        d = MC[k][mask] - Mk[mask]
        i = int(np.argmax(np.abs(d)))
        out["k_C"].append(k)
        out["T_over_rs"].append(TrsC[k])
        out["nodes_equal"].append(eq)
        out["min_M_C_outside"].append(float(MC[k][mask].min()))
        out["min_M_K_outside"].append(float(Mk[mask].min()))
        out["d_min_M_outside"].append(float(MC[k][mask].min() - Mk[mask].min()))
        out["max_abs_dM_outside"].append(float(abs(d[i])))
        out["r_at_max_abs_dM"].append(float(rc[mask][i]))
        out["min_dM_outside"].append(float(d.min()))
        out["max_dM_outside"].append(float(d.max()))
        out["first_node_outside"].append(float(rc[mask][0]))
    out["nodes_near_cell_record0"] = [float(x) for x in rC[0][:4]]
    return out


# ------------------------------------------------------------------------------------------------------------------------ part (b)
def node_fit(Z, ell):
    """cumtrapz 4 pi r^2 (rho_ell + s u J_ell) (and the shell alternative without 4 pi r^2) against the stored M over the nodes r > ell."""
    rn, Mz, rho, Jl = Z["r_nodes"], Z["M"], Z["rho_ell"], Z["J_ell"]
    un = Z["v"] / Z["N"]
    res = {}
    for s in (1, -1):
        for norm in ("volume", "shell"):
            per = []
            for k in range(rn.shape[0]):
                f = rho[k] + s * un[k] * Jl[k]
                if norm == "volume":
                    f = 4.0 * math.pi * rn[k] ** 2 * f
                Mrec = cumtrapz0(rn[k], f)
                mask = rn[k] > ell
                sc = float(np.max(np.abs(Mz[k][mask]))) if mask.any() else 0.0
                if sc > 0.0:
                    per.append(float(np.max(np.abs(Mrec[mask] - Mz[k][mask]))) / sc)
            res["s=%+d,%s" % (s, norm)] = {"s": s, "normalization": norm, "median_rel_residual": (float(np.median(per)) if per else None),
                                           "max_rel_residual": (float(np.max(per)) if per else None), "records": len(per)}
    cand = [k for k in res if res[k]["median_rel_residual"] is not None]
    best = min(cand, key=lambda k: res[k]["median_rel_residual"]) if cand else None
    return res, best


def compute_b(SC, mC, SK, mK, iK, rev, pairs, FK, ZK, ZC, o):
    ids = Identities()
    conv = {}
    notes = []
    g, ell = mK["g"], mK["ell"]
    nK, nC = mK["n"], mC["n"]
    TrsC, TrsK = SC.d["T_over_rs"], SK.d["T_over_rs"]
    # ---- the site grid and the dense fields
    R = np.asarray(FK["R"], dtype=float)
    if R.ndim != 1 or R.size < 4 or not np.all(np.diff(R) > 0):
        raise Inconsistent("control fields npz: R is not an increasing 1-d grid (shape %s)" % (R.shape,))
    a, edges = shell_edges(R)
    nsites = R.size
    for key in ("N", "v", "eps_tot", "J_tot", "eps_light_fed", "J_light_fed"):
        if FK[key].shape != (nK, nsites):
            raise Inconsistent("control fields npz: %s has shape %s against (%d records, %d sites)" % (key, FK[key].shape, nK, nsites))
    TK = np.array([float(x) for x in SK.d["T"]])
    for nm, Z in (("fields", FK), ("s11", ZK)):
        Tz = np.asarray(Z["T"], dtype=float).ravel()
        if Tz.shape != TK.shape or not np.allclose(Tz, TK):
            raise Inconsistent("control %s npz: T (%d entries) is not the records' T (%d entries)" % (nm, Tz.size, TK.size))
    N, v = FK["N"], FK["v"]
    if not np.all(N > 0):
        raise Inconsistent("control fields npz: N is not positive on every site (min %.4g)" % float(N.min()))
    u = v / N
    kap = [int(x) for x in np.asarray(FK["kappas_light"]).ravel()]
    wts = {c: float(w) for c, w in zip(kap, np.asarray(FK["weights_light"]).ravel())}
    flags = {c: int(x) for c, x in zip(kap, np.asarray(FK["light_sourcing"]).ravel())}
    src = [int(c) for c in mK["sourcing"]]
    grd = [int(c) for c in mK["guard"]]
    for c in src:
        if c not in kap:
            raise MissingKey("control fields npz: the sourcing channel %d (args.light_kappas) is not among kappas_light %s; part (b) stops" % (c, kap))
        ids.add("control fields npz: weights_light of kappa %d = 16 kappa" % c, [abs(wts[c] - 16.0 * c)])
    conv["sourcing_channels"] = {"used_from_args_light_kappas": src, "guard_excluded": grd, "npz_light_sourcing_flags": flags,
                                 "json_sourcing_flags_record0": {c: SK.d["L%d.sourcing" % c][0] for c in src + grd if "L%d.sourcing" % c in SK.d},
                                 "weights_used": {c: wts[c] for c in src}, "light_meta": FK.get("light_meta_str")}
    if any(flags.get(c, 0) != 1 for c in src) or any(flags.get(c, 0) != 0 for c in grd):
        notes.append("the fields npz's light_sourcing flags %s differ from the args' roles (sourcing %s, guard %s): the args' roles are used" % (flags, src, grd))
    # ---- H3: the heavy site densities, their normalization, the light part of eps_tot
    target = np.array([float(SK.d["E_dust_shells"][k]) + float(SK.d["E_pairs_shells"][k]) for k in range(nK)])
    fourpi = 4.0 * math.pi * R ** 2 * a
    cands = [("eps_tot - eps_light_fed", FK["eps_tot"] - FK["eps_light_fed"], FK["J_tot"] - FK["J_light_fed"]), ("eps_tot", FK["eps_tot"], FK["J_tot"])]
    fits = []
    for name, eH, jH in cands:
        for norm in ("shell", "density"):
            sums = eH.sum(axis=1) if norm == "shell" else (eH * fourpi).sum(axis=1)
            res, _ = rel_resid(sums, target)
            fits.append({"residual": (math.inf if res is None else res), "candidate": name, "normalization": norm})
    fits.sort(key=lambda d: d["residual"])
    bestH = fits[0]
    conv["heavy_site_densities"] = {"fits": fits, "chosen": bestH["candidate"], "normalization": bestH["normalization"], "residual": bestH["residual"],
                                    "target": "E_dust_shells + E_pairs_shells (the stored heavy totals per record)"}
    if bestH["residual"] > 1e-2:
        raise Inconsistent("control: no candidate for the heavy site densities sums to E_dust_shells + E_pairs_shells (best: %s as %s quantities, relative "
                           "residual %.3g); the normalization of eps_tot is not one the comparator can use; part (b) stops"
                           % (bestH["candidate"], bestH["normalization"], bestH["residual"]))
    if bestH["residual"] > TOL_EXACT:
        notes.append("the heavy totals are reproduced only to %.3g relative (not at roundoff): the heavy densities are used as %s quantities, flagged"
                     % (bestH["residual"], bestH["normalization"]))
    fac = 1.0 if bestH["normalization"] == "shell" else fourpi
    epsH = JH = None
    for name, eH, jH in cands:
        if name == bestH["candidate"]:
            epsH, JH = eH * fac, jH * fac
    ids.add("control: sum over sites of the heavy densities = E_dust_shells + E_pairs_shells (%s, %s)" % (bestH["candidate"], bestH["normalization"]),
            [bestH["residual"]], kind="hypothesis", note="relative; the alternatives: " + "; ".join("%s/%s %.3g" % (f["candidate"], f["normalization"], f["residual"]) for f in fits[1:]))
    # ---- H1: the light channels' arrays, weighted or not
    dec = []
    for c in src:
        e_site = (FK["eps_L%d" % c] * fac).sum(axis=1)
        j_site = (FK["J_L%d" % c] * fac).sum(axis=1)
        e_json = np.array([x if av(x) else np.nan for x in SK.d["L%d.eps_sum" % c]], dtype=float)
        j_json = np.array([x if av(x) else np.nan for x in SK.d["L%d.J_sum" % c]], dtype=float)
        ru_e, sce = rel_resid(e_site, e_json)
        rw_e, _ = rel_resid(e_site, wts[c] * e_json)
        ru_j, scj = rel_resid(j_site, j_json)
        rw_j, _ = rel_resid(j_site, wts[c] * j_json)
        ru_e, rw_e, ru_j, rw_j = [math.inf if x is None else x for x in (ru_e, rw_e, ru_j, rw_j)]
        if sce == 0.0 and scj == 0.0:
            raise Inconsistent("control: the light channel kappa %d's stored densities (eps_L%d, J_L%d) and the json's eps_sum, J_sum are identically zero on "
                               "every record; no light source to compare; part (b) stops" % (c, c, c))
        best = "unweighted" if max(ru_e, ru_j) <= max(rw_e, rw_j) else "weighted"
        rb = max(ru_e, ru_j) if best == "unweighted" else max(rw_e, rw_j)
        conv["L%d" % c] = {"stored_as": best, "factor_applied": (wts[c] if best == "unweighted" else 1.0), "residual_unweighted_eps": ru_e,
                           "residual_unweighted_J": ru_j, "residual_weighted_eps": rw_e, "residual_weighted_J": rw_j, "scale_eps": sce, "scale_J": scj}
        ids.add("control: sum over sites of eps_L%d, J_L%d = channels.%d.{eps_sum, J_sum} (stored %s)" % (c, c, c, best), [rb], kind="hypothesis",
                note="relative; unweighted fit eps %.3g J %.3g, weighted fit eps %.3g J %.3g" % (ru_e, ru_j, rw_e, rw_j))
        if rb > TOL_EXACT:
            raise Inconsistent("control: the light channel kappa %d's site sums reproduce neither eps_sum/J_sum (unweighted: %.3g, %.3g) nor w eps_sum/w J_sum "
                               "(weighted: %.3g, %.3g) at %g relative; the stored light densities are not what the json totals describe; part (b) stops"
                               % (c, ru_e, ru_j, rw_e, rw_j, TOL_EXACT))
        dec.append(best)
    if len(set(dec)) != 1:
        raise Inconsistent("control: the sourcing channels disagree on their weighting (%s); part (b) stops" % dict(zip(src, dec)))
    weighting = dec[0]
    epsLw = fac * sum((wts[c] if weighting == "unweighted" else 1.0) * FK["eps_L%d" % c] for c in src)
    JLw = fac * sum((wts[c] if weighting == "unweighted" else 1.0) * FK["J_L%d" % c] for c in src)
    conv["rho_L_A"] = "sum over kappa in %s of %s eps_L<kappa> (%s), J likewise" % (src, "w" if weighting == "unweighted" else "1", "stored " + weighting)
    lf, jlf = FK["eps_light_fed"] * fac, FK["J_light_fed"] * fac
    mx = float(np.max(np.abs(lf))) if lf.size else 0.0
    if mx == 0.0:
        conv["eps_light_fed"] = "identically zero on every record (nothing fed: the coupling is off)"
    else:
        r1, _ = rel_resid(lf, epsLw)
        r2, _ = rel_resid(jlf, JLw)
        conv["eps_light_fed"] = "non-zero; against sum over sourcing channels of w eps_L<k>: %.3g (eps), %.3g (J) relative" % (r1, r2)
        ids.add("control: eps_light_fed, J_light_fed = the weighted sourcing sums (the fed source)", [r1, r2], kind="cross")
    rC_, _ = rel_resid((FK["eps_tot"] * fac - epsLw).sum(axis=1), target)
    ids.add("control: sum over sites of (eps_tot - sum_src w eps_L<k>) = E_dust_shells + E_pairs_shells", [rC_ if rC_ is not None else math.inf], kind="relation",
            note="information: whether eps_tot carries the light part (it does not when the residual equals the chosen candidate's)")
    fedK = {"populated": bool(np.max(np.abs(ZK["rho_ell_light_fed"])) > 0.0 or np.max(np.abs(ZK["J_ell_light_fed"])) > 0.0)}
    conv["route_K_s11_light_fed"] = ("populated (the engine-smoothed fed light densities at the nodes)" if fedK["populated"]
                                     else "zero on every record (nothing fed: the coupling is off); route K unavailable")
    # ---- the row's sign s: the node route against the stored M (control and C), the root condition on the sites (control)
    nfK, bestK = node_fit(ZK, ell)
    nfC, bestC = node_fit(ZC, ell) if ZC is not None else (None, None)
    s_node = nfK[bestK]["s"] if bestK is not None else None
    trapped_pairs = [(k, j) for k, j in pairs if av(SK.d["R_t"][j])]
    qs = {1: [], -1: []}
    for k, j in trapped_pairs:
        for s_try in (1, -1):
            c0 = cum0(epsH[j] + s_try * u[j] * JH[j])
            for z in root_list(SK, j):
                qs[s_try].append(2.0 * g * float(enclosed(edges, c0, z)) / z)
    rootfit = {s_try: {"median_abs_dev": (float(np.median(np.abs(np.array(qs[s_try]) - 1.0))) if qs[s_try] else None),
                       "max_abs_dev": (float(np.max(np.abs(np.array(qs[s_try]) - 1.0))) if qs[s_try] else None), "roots": len(qs[s_try])} for s_try in (1, -1)}
    if trapped_pairs:
        s_root = 1 if rootfit[1]["median_abs_dev"] <= rootfit[-1]["median_abs_dev"] else -1
        if rootfit[s_root]["median_abs_dev"] > TOL_ROOT:
            raise Inconsistent("control: neither sign of u J reproduces the root condition 2 g M_H(R)/R = 1 at the dense roots from the site densities "
                               "(median |2gM/R - 1|: s=+1 %.3g, s=-1 %.3g over %d roots); the site densities do not carry the mass function in the "
                               "assumed normalization; part (b) stops" % (rootfit[1]["median_abs_dev"], rootfit[-1]["median_abs_dev"], rootfit[1]["roots"]))
        s = s_root
    else:
        s_root = None
        if s_node is None:
            raise Inconsistent("control: no dense root on any matched record and no node fit; the row's sign cannot be read off; part (b) stops")
        s = s_node
        notes.append("the control carries no dense root on the matched records: the row's sign comes from the node route alone")
    if s_node is not None and s_root is not None and s_node != s_root:
        notes.append("FLAG: the node route prefers s=%+d (median residual %.3g) while the root condition prefers s=%+d (median |2gM/R-1| %.3g); the root "
                     "condition on the sites decides, the comparator is flagged uncertain" % (s_node, nfK[bestK]["median_rel_residual"], s_root, rootfit[s_root]["median_abs_dev"]))
    if bestK is not None and nfK[bestK]["median_rel_residual"] is not None and nfK[bestK]["median_rel_residual"] > TOL_NODE:
        notes.append("FLAG: the node trapezoid reproduces the control's stored M only to %.3g (median relative over nodes r > ell)" % nfK[bestK]["median_rel_residual"])
    conv["row_sign_s"] = {"used": s, "from_root_condition_sites": s_root, "root_condition": rootfit, "from_node_route_control": bestK, "node_fit_control": nfK,
                          "node_fit_C": nfC, "node_route_best_C": bestC,
                          "meaning": "the row M' = 4 pi r^2 (rho + s u J), u = v/N; the light integrand rho_L + s u J_L"}
    if s_root is not None:
        ids.add("control: 2 g M_H(R)/R = 1 at the dense roots (site densities, s=%+d)" % s, [rootfit[s]["max_abs_dev"]], kind="hypothesis",
                note="median %.3g over %d roots; the other sign %.3g" % (rootfit[s]["median_abs_dev"], rootfit[s]["roots"], rootfit[-s]["median_abs_dev"]))
    if bestK is not None:
        ids.add("control s11 npz: M = cumtrapz 4 pi r^2 (rho_ell + s u J_ell) at the nodes (%s)" % bestK, [nfK[bestK]["max_rel_residual"]], kind="hypothesis",
                note="median %.3g; the node trapezoid at the node spacing carries its own error" % nfK[bestK]["median_rel_residual"])
    if bestC is not None:
        ids.add("C s11 npz: M = cumtrapz 4 pi r^2 (rho_ell + s u J_ell) at the nodes (%s)" % bestC, [nfC[bestC]["max_rel_residual"]], kind="hypothesis",
                note="median %.3g" % nfC[bestC]["median_rel_residual"])
    ms = [x for x in SK.d["R_t_over_2g_MS"] if av(x)]
    if ms:
        ids.add("control records: R_t_over_2g_MS = 1 (G = args.g at the root)", [abs(x - 1.0) for x in ms])
    # ---- the smoothing estimate
    sigma = float(o.get("sigma_smooth") or ell)
    K = gaussian_shell_kernel(R, sigma)
    cs = K.sum(axis=0)
    inner = R < R[-1] - 5.0 * sigma
    conv["kernel_estimate"] = {"sigma": sigma, "column_sums_min_inner": float(cs[inner].min()) if inner.any() else None,
                               "column_sums_max_inner": float(cs[inner].max()) if inner.any() else None,
                               "note": "a 3-D Gaussian of standard deviation sigma applied by this script to the shell contents; an estimate of the smoothing "
                                       "sensitivity, not the engine's kernel (which carries the cell term and its own discretization)"}
    # ---- per record with a dense root
    rows = []
    rn, Mz, Nn, vn = ZK["r_nodes"], ZK["M"], ZK["N"], ZK["v"]
    un = vn / Nn
    rlf, jlfn = ZK["rho_ell_light_fed"], ZK["J_ell_light_fed"]
    for k, j in trapped_pairs:
        zeros = root_list(SK, j)
        roots = [("outer", zeros[-1])] + ([("inner", zeros[0])] if len(zeros) > 1 else [])
        Nk, vk, uk = N[j], v[j], u[j]
        fH = epsH[j] + s * uk * JH[j]
        eL, jL = epsLw[j], JLw[j]
        fL = eL + s * uk * jL
        cH, cL, cLN = cum0(fH), cum0(fL), cum0(Nk * fL)
        eLs, jLs, eHs, jHs = K @ eL, K @ jL, K @ epsH[j], K @ JH[j]
        fHs, fLs = eHs + s * uk * jHs, eLs + s * uk * jLs
        cHs, cLNs = cum0(fHs), cum0(Nk * fLs)
        dNv = np.gradient(Nk + vk, R)
        cK = cumtrapz0(rn[j], 4.0 * math.pi * rn[j] ** 2 * Nn[j] * (rlf[j] + s * un[j] * jlfn[j])) if fedK["populated"] else None
        zC = root_list(SC, k)
        for label, Rr in roots:
            N_R = interp1(R, Nk, Rr)
            u_R = interp1(R, vk, Rr) / N_R
            kappa_fd = interp1(R, dNv, Rr)
            kappa_T = SK.d["kappa_T"][j] if (label == "outer" and av(SK.d["kappa_T"][j])) else None
            kappa_use = float(kappa_T) if kappa_T is not None else kappa_fd
            D_kappa = 2.0 * Rr * kappa_use / N_R
            MHp = interp1(R, fH, Rr) / a
            MHp_s = interp1(R, fHs, Rr) / a
            D_rho, D_rho_s = 1.0 - 2.0 * g * MHp, 1.0 - 2.0 * g * MHp_s
            idx = int(np.searchsorted(rn[j], Rr)) - 1
            if 0 <= idx < rn[j].size - 1:
                D_node = 1.0 - 2.0 * g * (Mz[j][idx + 1] - Mz[j][idx]) / (rn[j][idx + 1] - rn[j][idx])
            else:
                D_node = None
            I_S = float(enclosed(edges, cLN, Rr))
            I_Ss = float(enclosed(edges, cLNs, Rr))
            I_K = float(np.interp(Rr, rn[j], cK)) if cK is not None else None
            ML_R = float(enclosed(edges, cL, Rr))
            MH_S = float(enclosed(edges, cH, Rr))
            MH_Ss = float(enclosed(edges, cHs, Rr))
            M_node = float(np.interp(Rr, rn[j], Mz[j]))
            dR_L = g * I_S / (Rr * kappa_use) if kappa_use != 0.0 else None
            dR_L_s = g * I_Ss / (Rr * kappa_use) if kappa_use != 0.0 else None
            dR_L_K = (g * I_K / (Rr * kappa_use)) if (I_K is not None and kappa_use != 0.0) else None
            dR_L_Drho = 2.0 * g * I_S / (N_R * D_rho) if D_rho != 0.0 else None
            dR_L_Drho_s = 2.0 * g * I_Ss / (N_R * D_rho_s) if D_rho_s != 0.0 else None
            alts = [x for x in (dR_L_s, dR_L_K) if x is not None]
            spread = (max(abs(x - dR_L) for x in alts) if (alts and dR_L is not None) else 0.0) + 1e-12 * (abs(dR_L) if dR_L is not None else 1.0) + 1e-15
            if label == "outer":
                R_C = SC.d["R_t"][k]
            else:
                R_C = (zC[0] if len(zC) > 1 else None)
            dRm = (float(R_C) - Rr) if av(R_C) else None
            rat = (dRm / dR_L) if (dRm is not None and dR_L not in (None, 0.0)) else None
            rows.append({"k_C": k, "j_K": j, "T_over_rs": TrsC[k], "root": label, "R": Rr, "N_H_R": N_R, "u_H_R": u_R, "u_plus_1": u_R + 1.0,
                         "kappa_T_stored": kappa_T, "kappa_fd_sites": kappa_fd, "kappa_used": kappa_use, "kappa_T_C": (SC.d["kappa_T"][k] if label == "outer" else None),
                         "D_R_kappa": D_kappa, "D_R_rho_sites": D_rho, "D_R_rho_smoothed": D_rho_s, "D_R_nodes": D_node,
                         "D_R_rho_over_kappa": (D_rho / D_kappa if D_kappa != 0.0 else None), "D_R_rho_smoothed_over_kappa": (D_rho_s / D_kappa if D_kappa != 0.0 else None),
                         "M_H_prime_R_sites": MHp, "I_L_sites": I_S, "I_L_sites_smoothed": I_Ss, "I_L_nodes": I_K, "M_L_R_sites": ML_R,
                         "M_H_R_sites": MH_S, "M_H_R_nodes_stored": M_node, "two_gM_over_R_sites": 2.0 * g * MH_S / Rr, "two_gM_over_R_sites_smoothed": 2.0 * g * MH_Ss / Rr,
                         "two_gM_over_R_stored": 2.0 * g * M_node / Rr, "M_L_over_M_H_at_R": (ML_R / MH_S if MH_S != 0.0 else None),
                         "dR_L": dR_L, "dR_L_sites_smoothed": dR_L_s, "dR_L_nodes": dR_L_K, "dR_L_D_rho": dR_L_Drho, "dR_L_D_rho_smoothed": dR_L_Drho_s,
                         "dR_L_spread": spread, "R_C": (float(R_C) if av(R_C) else None), "dR_measured": dRm, "ratio_measured_over_dR_L": rat,
                         "same_sign": ((sgn(dRm) == sgn(dR_L)) if (dRm is not None and dR_L is not None and dRm != 0.0 and dR_L != 0.0) else None)})
    out_rows = [r for r in rows if r["root"] == "outer"]
    if out_rows:
        ids.add("control: |u_H(R) + 1| at the dense roots (the root is of N + v)", [abs(r["u_plus_1"]) for r in rows], kind="cross")
        ids.add("control: kappa_fd (sites) against kappa_T (stored) at the outer roots, relative", [abs(r["kappa_fd_sites"] - r["kappa_T_stored"]) / abs(r["kappa_T_stored"])
                                                                                                      for r in out_rows if r["kappa_T_stored"] not in (None, 0.0)], kind="cross")
        ids.add("CF4: D_R from the heavy densities / D_R from kappa_T at the outer roots, |ratio - 1|", [abs(r["D_R_rho_over_kappa"] - 1.0) for r in out_rows if r["D_R_rho_over_kappa"] is not None], kind="cross",
                note="site densities unsmoothed; smoothed by the script: " + fm(maxabs([r["D_R_rho_smoothed_over_kappa"] - 1.0 for r in out_rows if r["D_R_rho_smoothed_over_kappa"] is not None]), 3))
    # ---- the margin per record on the control's fields
    marg = {name: [] for name in ("j_K", "k_C", "T_over_rs", "min_M_H_outside", "r_min_M_H", "min_margin", "r_min_margin", "min_margin_B", "r_min_margin_B",
                                  "min_M_L_outside", "max_M_L_outside", "M_L_at_R_t", "M_L_over_M_H_at_R_t", "min_margin_nodes", "margin_minus_eps_M")}
    eps_M = o.get("eps_M")
    for j in range(nK):
        r = rn[j]
        mask = r > ell
        uk = u[j]
        fL = epsLw[j] + s * uk * JLw[j]
        cL = cum0(fL)
        cLa = cum0(N[j] * (np.abs(epsLw[j]) + np.abs(uk) * np.abs(JLw[j])))
        ML = enclosed(edges, cL, r)
        BM = enclosed(edges, cLa, r) / Nn[j]
        MH = Mz[j]
        m1 = MH - np.maximum(-ML, 0.0)
        m2 = MH - BM
        i1, i2, iH = int(np.argmin(m1[mask])), int(np.argmin(m2[mask])), int(np.argmin(MH[mask]))
        marg["j_K"].append(j)
        marg["k_C"].append(rev.get(j))
        marg["T_over_rs"].append(TrsK[j])
        marg["min_M_H_outside"].append(float(MH[mask][iH]))
        marg["r_min_M_H"].append(float(r[mask][iH]))
        marg["min_margin"].append(float(m1[mask][i1]))
        marg["r_min_margin"].append(float(r[mask][i1]))
        marg["min_margin_B"].append(float(m2[mask][i2]))
        marg["r_min_margin_B"].append(float(r[mask][i2]))
        marg["min_M_L_outside"].append(float(ML[mask].min()))
        marg["max_M_L_outside"].append(float(ML[mask].max()))
        if av(SK.d["R_t"][j]):
            Rt = float(SK.d["R_t"][j])
            mlr = float(enclosed(edges, cL, Rt))
            mhr = float(np.interp(Rt, r, MH))
            marg["M_L_at_R_t"].append(mlr)
            marg["M_L_over_M_H_at_R_t"].append(mlr / mhr if mhr != 0.0 else None)
        else:
            marg["M_L_at_R_t"].append(None)
            marg["M_L_over_M_H_at_R_t"].append(None)
        if fedK["populated"]:
            MLK = cumtrapz0(r, 4.0 * math.pi * r ** 2 * (rlf[j] + s * un[j] * jlfn[j]))
            marg["min_margin_nodes"].append(float((MH - np.maximum(-MLK, 0.0))[mask].min()))
        else:
            marg["min_margin_nodes"].append(None)
        marg["margin_minus_eps_M"].append((float(m1[mask][i1]) - eps_M) if eps_M is not None else None)
    # ---- the same split on C's actual fields (its s11 npz components)
    margC = None
    if ZC is not None:
        margC = {name: [] for name in ("k_C", "T_over_rs", "min_M_outside", "min_margin", "r_min_margin", "min_M_L_outside", "max_M_L_outside", "M_L_at_R_t",
                                       "components_vs_M_rel_residual")}
        rc, Mc, Nc, vc = ZC["r_nodes"], ZC["M"], ZC["N"], ZC["v"]
        uc = vc / Nc
        for k in range(ZC["M"].shape[0]):
            r = rc[k]
            mask = r > ell
            MLc = cumtrapz0(r, 4.0 * math.pi * r ** 2 * (ZC["rho_ell_light_fed"][k] + s * uc[k] * ZC["J_ell_light_fed"][k]))
            MHc_rec = cumtrapz0(r, 4.0 * math.pi * r ** 2 * (ZC["rho_ell_ball"][k] + ZC["rho_ell_pairs"][k] + s * uc[k] * (ZC["J_ell_ball"][k] + ZC["J_ell_pairs"][k])))
            MHc = Mc[k] - MLc
            m1 = MHc - np.maximum(-MLc, 0.0)
            i1 = int(np.argmin(m1[mask]))
            sc = float(np.max(np.abs(Mc[k][mask]))) if mask.any() else 0.0
            margC["k_C"].append(k)
            margC["T_over_rs"].append(TrsC[k] if k < nC else None)
            margC["min_M_outside"].append(float(Mc[k][mask].min()))
            margC["min_margin"].append(float(m1[mask][i1]))
            margC["r_min_margin"].append(float(r[mask][i1]))
            margC["min_M_L_outside"].append(float(MLc[mask].min()))
            margC["max_M_L_outside"].append(float(MLc[mask].max()))
            margC["M_L_at_R_t"].append(float(np.interp(float(SC.d["R_t"][k]), r, MLc)) if (k < nC and av(SC.d["R_t"][k])) else None)
            margC["components_vs_M_rel_residual"].append((float(np.max(np.abs((MHc_rec + MLc - Mc[k])[mask]))) / sc) if sc > 0.0 else None)
        ids.add("C s11 npz: cumtrapz of the stored components (ball + pairs + light fed, with s u J) = M at the nodes r > ell", [x for x in margC["components_vs_M_rel_residual"] if x is not None],
                kind="cross", note="the node trapezoid's own error enters")
    # ---- the episode summary
    meas = [r for r in out_rows if r["ratio_measured_over_dR_L"] is not None]
    epis = {"outer_roots": len(out_rows), "with_measured_dR": len(meas), "same_sign": sum(1 for r in meas if r["same_sign"]),
            "opposite_sign": sum(1 for r in meas if r["same_sign"] is False), "median_ratio": (float(np.median([r["ratio_measured_over_dR_L"] for r in meas])) if meas else None),
            "median_abs_log10_ratio": (float(np.median([abs(math.log10(abs(r["ratio_measured_over_dR_L"]))) for r in meas if r["ratio_measured_over_dR_L"] != 0.0])) if meas else None),
            "dR_L_sign_counts": {"positive": sum(1 for r in out_rows if r["dR_L"] is not None and r["dR_L"] > 0), "negative": sum(1 for r in out_rows if r["dR_L"] is not None and r["dR_L"] < 0)},
            "max_abs_dR_L": maxabs([r["dR_L"] for r in out_rows]), "max_abs_dR_measured": maxabs([r["dR_measured"] for r in out_rows])}
    return {"conventions": conv, "notes": notes, "identities": ids.d, "identities_order": ids.order, "sign_s": s, "roots": rows, "margin_control": marg,
            "margin_C_actual": margC, "episode_summary": epis, "route_K_populated": fedK["populated"], "site_spacing": a, "nsites": nsites}


def judge(B, kfC, jfK, rev, TrsC, TrsK, floor_dR, floor_dRL_opt, dT_rec):
    V = {"floor_dR_t_lattice_units": floor_dR, "floor_dR_L": None, "flags": [n for n in B["notes"] if n.startswith("FLAG")], "verdict": None, "basis": None,
         "k_star": None, "dR_L_at_formation": None, "dR_measured_at_formation": None, "ratio_at_formation": None, "formation_shift_records": None}
    outer = {r["k_C"]: r for r in B["roots"] if r["root"] == "outer"}
    if jfK is None:
        V["verdict"] = "undecided: the control carries no dense root on any matched record"
        return V
    kstar = rev.get(jfK)
    if kstar is None or kstar not in outer:
        V["verdict"] = "undecided: the control's formation record has no matched clock in C (partial alignment)"
        return V
    r0 = outer[kstar]
    V["k_star"] = kstar
    dRL = r0["dR_L"]
    fl_L = (r0["dR_L_spread"] if floor_dRL_opt is None else floor_dRL_opt)
    V["floor_dR_L"] = fl_L
    V["dR_L_at_formation"] = dRL
    V["formation_shift_records"] = ((TrsC[kfC] - TrsK[jfK]) / dT_rec) if kfC is not None else None
    if dRL is None:
        V["verdict"] = "undecided: delta R_L could not be evaluated at the control's formation record (kappa = 0)"
        return V
    above_L = abs(dRL) > fl_L

    def magnitude(row, where):
        rat = row["ratio_measured_over_dR_L"]
        above_m = abs(row["dR_measured"]) > floor_dR
        if not (above_L and above_m):
            return "undecided (below floor: |Delta R_t| %s against %s, |delta R_L| %s against %s)" % (fm(abs(row["dR_measured"]), 3), fm(floor_dR, 3), fm(abs(dRL), 3), fm(fl_L, 3)), rat
        if rat is None or rat == 0.0:
            return "undecided (ratio not defined)", rat
        if abs(math.log10(abs(rat))) > 1.0:
            return "CONFLICT (order of magnitude): Delta R_t / delta R_L = %s at %s" % (fm(rat, 4), where), rat
        return "no conflict: Delta R_t / delta R_L = %s at %s" % (fm(rat, 4), where), rat
    if r0["dR_measured"] is not None:
        dRm = r0["dR_measured"]
        V["dR_measured_at_formation"] = dRm
        V["ratio_at_formation"] = r0["ratio_measured_over_dR_L"]
        above_m = abs(dRm) > floor_dR
        V["basis"] = "both runs carry the outer root at the control's formation record (T/r_s %s): the sign and the magnitude at that record" % fm(TrsK[jfK], 6)
        if not (above_L and above_m):
            V["verdict"] = "undecided (below floor at the formation: |Delta R_t| %s against %s, |delta R_L| %s against %s)" % (fm(abs(dRm), 3), fm(floor_dR, 3), fm(abs(dRL), 3), fm(fl_L, 3))
        elif sgn(dRm) != sgn(dRL):
            V["verdict"] = "CONFLICT (sign): at the control's formation Delta R_t = %s and delta R_L = %s" % (fm(dRm, 4), fm(dRL, 4))
        else:
            V["verdict"], _ = magnitude(r0, "the control's formation record")
        return V
    # C carries no root at the control's formation record
    shift = V["formation_shift_records"]
    if kfC is None:
        V["basis"] = "C carries no dense root on any record while the control does: the formation shift stands in for the sign"
        if not above_L:
            V["verdict"] = "undecided (delta R_L %s below its floor %s; C never traps)" % (fm(dRL, 3), fm(fl_L, 3))
        elif dRL > 0:
            V["verdict"] = "CONFLICT (sign): delta R_L = %s > 0 while C carries no dense root on any record" % fm(dRL, 4)
        else:
            V["verdict"] = "no sign conflict (delta R_L = %s < 0, C never traps); the magnitude is undecidable" % fm(dRL, 4)
        return V
    V["basis"] = "C forms %+g records after the control (the control's formation T/r_s %s, C's %s): the formation shift stands in for the sign; the magnitude at the first record where both carry the root" % (shift, fm(TrsK[jfK], 6), fm(TrsC[kfC], 6))
    if not above_L:
        V["verdict"] = "undecided (delta R_L %s below its floor %s)" % (fm(dRL, 3), fm(fl_L, 3))
        return V
    if (shift > 0 and dRL > 0) or (shift < 0 and dRL < 0):
        V["verdict"] = "CONFLICT (sign): delta R_L = %s while C forms %+g records %s than the control" % (fm(dRL, 4), shift, "later" if shift > 0 else "earlier")
        return V
    later = sorted(k for k in outer if outer[k]["dR_measured"] is not None)
    if not later:
        V["verdict"] = "no sign conflict (delta R_L = %s, C forms %+g records %s); no record carries the root in both runs, the magnitude is undecided" % (fm(dRL, 4), shift, "later" if shift > 0 else "earlier")
        return V
    r1 = outer[later[0]]
    V["dR_measured_at_formation"] = None
    V["ratio_at_formation"] = r1["ratio_measured_over_dR_L"]
    verdict, _ = magnitude(r1, "the first record with the root in both runs (T/r_s %s)" % fm(r1["T_over_rs"], 6))
    V["verdict"] = ("no sign conflict (formation shift %+g records against delta R_L = %s); " % (shift, fm(dRL, 4))) + verdict
    return V


# ----------------------------------------------------------------------------------------------------------------------------- md
def rows_for(TrsC, first, clocks):
    rows = list(first)
    for c in clocks:
        rows.append(("%g" % c, find_clock(TrsC, c), c))
    return rows


def build_md(ctx):
    mC, mK = ctx["mC"], ctx["mK"]
    TrsC = ctx["SC"].d["T_over_rs"]
    D, F, A = ctx["A_D"], ctx["A_formation"], ctx["A_summary"]
    B, V, bstop = ctx["B"], ctx["V"], ctx["b_stop"]
    g = fm
    L = ["# Computation C: the sigma/a 4 member against its coupling-off control, %s" % VERSION, "", LABEL, "",
         "Generated %s by %s. The control: Computation A on the pathfinder's own chain, solver and kernel (--light-coupling off, otherwise the "
         "pathfinder's arguments; the light channels propagate and are read out and source nothing). The comparator: %s. Version 1 (readouts 2, 3, "
         "4, 5, 7) is repeated by import and appended as information; its readouts 3 and 5 (against the 1024-site plain-mixer production and the "
         "light-follow on it) are superseded as the coupling's readout by section 2." % (ctx["generated"], VERSION, BOUND), "",
         "## 0. The reading rule (stated before the numbers)", ""]
    for head, text in RULE:
        L.append("**%s** %s" % (head, text))
        L.append("")
    L += ["## 1. The inputs, the matched arguments, the matched clocks", ""]
    for d in ctx["inputs"]:
        L.append("- %s: %s (md5 %s, %d bytes)" % (d["role"], d["path"], d["md5"], d["bytes"]))
    L.append("- C: script %s, tag %s, %d records, T/r_s %s to %s, light_coupling %s; the control: script %s, tag %s, %d records, T/r_s %s to %s, light_coupling %s"
             % (mC["script"], mC["tag"], mC["n"], g(TrsC[0], 6), g(TrsC[-1], 6), mC["coupling"], mK["script"], mK["tag"], mK["n"], g(ctx["SK"].d["T_over_rs"][0], 6),
                g(ctx["SK"].d["T_over_rs"][-1], 6), mK["coupling"]))
    L.append("- matched clocks: %d (C %d records, the control %d); %s" % (len(ctx["pairs"]), mC["n"], mK["n"], "complete" if ctx["complete"] else "PARTIAL (--allow-partial)"))
    L.append("- args differing (out, tag, gpu, slices, resume entries skipped): %s%s" % (
        "; ".join("%s: C %s, control %s" % (k, fm(va, 6), fm(vb, 6)) for k, va, vb in ctx["args_diff"]) or "none",
        ("; SOLVER-MATCHED: only light_coupling differs" if not ctx["unexpected_args"] else "; NOT SOLVER-MATCHED beyond the coupling: %s (allowed by --allow-arg-mismatch)" % ", ".join(ctx["unexpected_args"]))))
    cc = A["control_convergence"]
    L.append("- the control's run: converged true on %d records (false %d, absent %d), largest kept residual %s, certified_on_record false on %d, fallbacks %d channel-records, "
             "light_fed true on %d records, light_v36.coupling %s, iterations at most %s" % (cc["converged_true"], cc["converged_false"], cc["converged_absent"], g(cc["max_kept_residual"], 3),
                                                                                               cc["certified_false"], cc["fallbacks"], cc["light_fed_true"], "/".join(cc["coupling_json"]), g(cc["iterations_max"], 1)))
    L.append("- the collars (r_collar): C %s, the control %s (lattice units; the chain's end): the heavy and light collar cut currents of both runs are at the same cut"
             % ("/".join(A["collars"]["C"]) or "-", "/".join(A["collars"]["control"]) or "-"))
    L.append("- version 1 beside: %s" % ctx["v1_status"])
    # ---- (a)
    kfC = F["C"]["k"] if F["C"] else None
    kfK = F["control"]["k_C"] if F["control"] else None
    first = [("0 (preparation check)", find_clock(TrsC, 0.0), 0.0)]
    if kfK is not None:
        first.append(("formation, control (%g)" % TrsC[kfK], kfK, TrsC[kfK]))
    else:
        first.append(("formation, control (none)", None, None))
    if kfC is not None and kfC != kfK:
        first.append(("formation, C (%g)" % TrsC[kfC], kfC, TrsC[kfC]))
    rows = rows_for(TrsC, first, TABLE_CLOCKS)
    ta1 = build_table(rows, [("R_t C", "C.R_t", 9), ("R_t ctrl", "K.R_t", 9), ("dR_t", "d.R_t", 4), ("r_in C", "C.r_in", 9), ("r_in ctrl", "K.r_in", 9), ("dr_in", "d.r_in", 4),
                             ("min(N-|v|) C", "C.min_N_minus_absv", 6), ("ctrl", "K.min_N_minus_absv", 6), ("d", "d.min_N_minus_absv", 3), ("r_min C", "C.r_min_N_minus_absv", 6),
                             ("ctrl", "K.r_min_N_minus_absv", 6), ("kappa_T C", "C.kappa_T", 5), ("ctrl", "K.kappa_T", 5), ("A = 4 pi R_t^2 C", "C.A", 6)], D)
    ta2 = build_table(rows, [("E C", "C.E", 11), ("E ctrl", "K.E", 11), ("dE", "d.E", 4), ("dE/E", "rel.E", 3), ("E_inv_heavy C", "C.light.E_inv_heavy", 9), ("ctrl", "K.light.E_inv_heavy", 9),
                             ("d", "d.light.E_inv_heavy", 4), ("E_inv_light C", "C.light.E_inv_light", 6), ("ctrl", "K.light.E_inv_light", 6), ("d", "d.light.E_inv_light", 4),
                             ("rel", "rel.light.E_inv_light", 4), ("E_inv_src/E C", "C.light.E_inv_light_sourcing_over_E", 5), ("ctrl (= E_rad0/E, matched)", "K.light.E_inv_light_sourcing_over_E", 5),
                             ("follow (1024, plain; info)", "F.E_rad0_over_E", 5)], D)
    chans = mC["channels"]
    cols3 = [("M1_I_E C", "C.M1_I_E", 4), ("ctrl", "K.M1_I_E", 4), ("d", "d.M1_I_E", 3), ("I_E_light_src C", "C.light.I_E_light_sourcing", 4), ("ctrl", "K.light.I_E_light_sourcing", 4),
             ("d", "d.light.I_E_light_sourcing", 3), ("I_E guard C", "C.light.I_E_light_guard", 4), ("ctrl", "K.light.I_E_light_guard", 4)]
    for c in chans:
        cols3 += [("L%s E_inv C" % c, "C.L%s.E_inv" % c, 6), ("ctrl", "K.L%s.E_inv" % c, 6), ("rel", "rel.L%s.E_inv" % c, 4)]
    ta3 = build_table(rows, cols3, D)
    ta4 = build_table(rows, [("M_min C (json)", "C.s11.M_min_outside_cell", 8), ("ctrl", "K.s11.M_min_outside_cell", 8), ("d", "d.s11.M_min_outside_cell", 4), ("rel", "rel.s11.M_min_outside_cell", 4),
                             ("r_at C", "C.s11.r_at_M_min", 5), ("ctrl", "K.s11.r_at_M_min", 5), ("neg pts C", "C.s11.negative_points_outside_cell", 1), ("ctrl", "K.s11.negative_points_outside_cell", 1),
                             ("max 2gM/r C", "C.s11.compactness_max_2gM_over_r", 6), ("ctrl", "K.s11.compactness_max_2gM_over_r", 6), ("light fed integral C", "C.s11.light_source_integral", 5),
                             ("ctrl", "K.s11.light_source_integral", 5), ("H_in C", "C.H_in", 9), ("ctrl", "K.H_in", 9), ("H_in_light C", "C.light.H_in_light", 6), ("ctrl", "K.light.H_in_light", 6)], D)
    X = ctx["S11X"]
    DX = {"x.max_abs_dM": [None] * mC["n"], "x.r_at": [None] * mC["n"], "x.min_M_C": [None] * mC["n"], "x.min_M_K": [None] * mC["n"], "x.d_min": [None] * mC["n"], "x.eq": [None] * mC["n"]}
    for i, k in enumerate(X["k_C"]):
        DX["x.max_abs_dM"][k] = X["max_abs_dM_outside"][i]
        DX["x.r_at"][k] = X["r_at_max_abs_dM"][i]
        DX["x.min_M_C"][k] = X["min_M_C_outside"][i]
        DX["x.min_M_K"][k] = X["min_M_K_outside"][i]
        DX["x.d_min"][k] = X["d_min_M_outside"][i]
        DX["x.eq"][k] = X["nodes_equal"][i]
    ta5 = build_table(rows, [("min M over nodes r>ell, C", "x.min_M_C", 8), ("ctrl", "x.min_M_K", 8), ("d", "x.d_min", 4), ("max |M_C - M_ctrl| (r>ell)", "x.max_abs_dM", 4), ("at r", "x.r_at", 6), ("nodes equal", "x.eq", 1)], DX)
    fC, fK = F["C"], F["control"]
    s_a = ("(a) C minus control at matched clocks (%d of %d). The preparation (T = 0): dE = %s, d min(N - |v|) = %s, d E_inv_heavy = %s, d E_inv_light = %s (the gate's floors E/E0 3.0e-7, "
           "min(N - |v|) 4.6e-8). The formation (first dense root): C at T/r_s %s (R_t %s, r_in %s), the control at %s (R_t %s, r_in %s): dT = %s r_s = %s records (spacing %s); at the "
           "control's formation dR_t = %s (%s relative), dr_in = %s, dE/E = %s. The first episode: C %d records (through T/r_s %s), the control %d (through %s), both carry the root on %d "
           "records; |dR_t| peaks at %s at T/r_s %s; dR_t is positive on %d and negative on %d trapped records; |dE/E| peaks at %s at %s. The light field (C's against the control's, the "
           "same chain and solver): E_inv_light relative %s at 20 and %s at 40 (sourcing %s, %s); E_inv_light_sourcing/E of the control is E_rad^(0)/E on the solver-matched history "
           "(the follow's on the 1024-site plain history is information). The collar currents at 40 (both at the chain's end): heavy M1_I_E C %s, control %s; light sourcing C %s, "
           "control %s. The mass function's outside-cell minimum (json, the engine's first point past ell): |d| at most %s over the run, %s at 40; at the common nodes r > ell "
           "|M_C - M_ctrl| at most %s; negative points in the control on %d records."
           % (len(ctx["pairs"]), mC["n"], g(A["prep_T0"]["dE"], 3), g(A["prep_T0"]["d_min_N_minus_absv"], 3), g(A["prep_T0"]["d_E_inv_heavy"], 3), g(A["prep_T0"]["d_E_inv_light"], 3),
              g(fC["T_over_rs"], 6) if fC else "-", g(fC["R_t"], 9) if fC else "-", g(fC["r_in"], 9) if fC else "-", g(fK["T_over_rs"], 6) if fK else "-", g(fK["R_t"], 9) if fK else "-",
              g(fK["r_in"], 9) if fK else "-", g(F["dT_over_rs_C_minus_control"], 4), g(F["dT_records"], 1), g(F["record_spacing_T_over_rs"], 4), g(F["dR_t_at_control_formation"], 4),
              g(F["dR_t_rel_at_control_formation"], 3), g(F["dr_in_at_control_formation"], 4), g(F["dE_rel_at_control_formation"], 3), F["episode_C"]["records"], g(F["episode_C"]["last_T_over_rs"], 6),
              F["episode_control"]["records"], g(F["episode_control"]["last_T_over_rs"], 6), F["records_with_root_in_both"], g(A["max_abs_dR_t"], 4), g(A["T_at_max_abs_dR_t"], 6),
              A["dR_t_sign_counts"]["positive"], A["dR_t_sign_counts"]["negative"], g(A["max_abs_dE_rel"], 3), g(A["T_at_max_abs_dE_rel"], 6), g(A["at_20"]["rel_E_inv_light"], 4),
              g(A["at_40"]["rel_E_inv_light"], 4), g(A["at_20"]["rel_E_inv_light_sourcing"], 4), g(A["at_40"]["rel_E_inv_light_sourcing"], 4), g(A["at_40"]["M1_I_E_C"], 4),
              g(A["at_40"]["M1_I_E_control"], 4), g(A["at_40"]["I_E_light_sourcing_C"], 4), g(A["at_40"]["I_E_light_sourcing_control"], 4), g(A["max_abs_d_M_min_json"], 3), g(A["at_40"]["d_M_min"], 3),
              g(maxabs(X["max_abs_dM_outside"]), 3), A["negative_points_control"]))
    L += ["", "## 2. (a) C minus control at matched clocks", "",
          "Keys: R_t, r_in, trapping_zeros, min_N_minus_absv, r_min_N_minus_absv, kappa_T, E, E_over_E0, H_in, M1_I_E, light_v36.{E_inv_heavy, E_inv_light, E_inv_light_sourcing, "
          "I_E_light, I_E_light_sourcing, H_in_light, E_inv_light_sourcing_over_E, channels.<k>.E_inv}, s11.{M_min_outside_cell, r_at_M_min, negative_points_outside_cell, "
          "compactness_max_2gM_over_r, light_source_integral}, both jsons; the two s11 npz (r_nodes, M). d = C - control at the same T; rel = d / control.", "",
          "2a, the roots and the trapping scan:", "", ta1, "", "2b, the energies per field:", "", ta2, "", "2c, the collar currents per field and E_inv per light channel (unweighted):", "", ta3, "",
          "2d, the mass function (s11) and the inside energies:", "", ta4, "",
          "2e, the mass function at the common nodes r > ell (the json's M_min_outside_cell is the engine's value at its first evaluation point past ell, r 4.25; the npz carries the coarse "
          "nodes, the first past ell at r %s; the two are compared only among themselves):" % g(X["first_node_outside"][0] if X["first_node_outside"] else None, 4), "", ta5, "", s_a]
    # ---- (b)
    L += ["", "## 3. (b) The bound's comparator from the control's stored fields", ""]
    if bstop is not None:
        L += ["**STOP: part (b) did not run.** %s" % bstop, "", "The keys the comparator reads: the control's fields npz R, T, N, v, eps_tot, J_tot, eps_light_fed, J_light_fed, "
              "kappas_light, weights_light, light_sourcing, eps_L<k>, J_L<k> for k in args.light_kappas; its s11 npz T, r_nodes, rho_ell, J_ell, N, v, M, rho_ell_ball, J_ell_ball, "
              "rho_ell_pairs, J_ell_pairs, rho_ell_light_fed, J_ell_light_fed; its json's trapping_zeros, R_t, r_in, kappa_T, E_dust_shells, E_pairs_shells, "
              "light_v36.channels.<k>.{eps_sum, J_sum, w}. Nothing was substituted; the verdict of section 0's rule is not issued."]
    else:
        conv = B["conventions"]
        L += ["### 3.1 The keys and the conventions read off the records", ""]
        L.append("- rho_L^A, J_L^A: %s; the sourcing channels from args.light_kappas %s, the guard %s excluded; the fields npz's light_sourcing flags %s; weights %s; light_meta: %s"
                 % (conv["rho_L_A"], conv["sourcing_channels"]["used_from_args_light_kappas"], conv["sourcing_channels"]["guard_excluded"], conv["sourcing_channels"]["npz_light_sourcing_flags"],
                    conv["sourcing_channels"]["weights_used"], conv["sourcing_channels"]["light_meta"]))
        for c in conv["sourcing_channels"]["used_from_args_light_kappas"]:
            d = conv["L%d" % c]
            L.append("- kappa %d: stored %s (site sums against eps_sum/J_sum: %s/%s; against w eps_sum/w J_sum: %s/%s relative); factor applied %s"
                     % (c, d["stored_as"], g(d["residual_unweighted_eps"], 3), g(d["residual_unweighted_J"], 3), g(d["residual_weighted_eps"], 3), g(d["residual_weighted_J"], 3), g(d["factor_applied"], 4)))
        L.append("- eps_light_fed, J_light_fed: %s" % conv["eps_light_fed"])
        h = conv["heavy_site_densities"]
        L.append("- rho_H, J_H: %s as %s quantities (site sums against %s to %s relative; the alternatives %s)"
                 % (h["chosen"], h["normalization"], h["target"], g(h["residual"], 3), "; ".join("%s/%s %s" % (f["candidate"], f["normalization"], g(f["residual"], 3)) for f in h["fits"][1:])))
        rs_ = conv["row_sign_s"]
        rf = rs_["root_condition"]
        L.append("- the row's sign s = %+d (rho + s u J): the root condition 2 g M_H(R)/R = 1 at the control's dense roots from the site densities gives median |2gM/R - 1| %s (s=+1) "
                 "and %s (s=-1) over %s roots; the node trapezoid against the control's stored M prefers %s (median relative residual %s), on C's %s (%s)"
                 % (rs_["used"], g(rf[1]["median_abs_dev"], 3), g(rf[-1]["median_abs_dev"], 3), rf[1]["roots"], rs_["from_node_route_control"],
                    g(rs_["node_fit_control"][rs_["from_node_route_control"]]["median_rel_residual"], 3) if rs_["from_node_route_control"] else "-",
                    rs_["node_route_best_C"], g(rs_["node_fit_C"][rs_["node_route_best_C"]]["median_rel_residual"], 3) if rs_["node_route_best_C"] else "-"))
        L.append("- route K: %s" % conv["route_K_s11_light_fed"])
        ke = conv["kernel_estimate"]
        L.append("- the smoothing estimate: sigma %s, column sums %s to %s away from the chain's end; %s" % (g(ke["sigma"], 4), g(ke["column_sums_min_inner"], 6), g(ke["column_sums_max_inner"], 6), ke["note"]))
        L.append("- R: records[k].trapping_zeros (outer = R_t, inner = r_in); kappa_R: records[k].kappa_T at the outer root (the inner root uses (N + v)' from the sites); G = args.g = %s; "
                 "the site spacing %s, %d sites" % (g(mK["g"], 4), g(B["site_spacing"], 4), B["nsites"]))
        for nt in B["notes"]:
            L.append("- note: %s" % nt)
        # the per-root tables
        Db = {}
        for r in B["roots"]:
            pre = "b." if r["root"] == "outer" else "bi."
            for key, val in r.items():
                Db.setdefault(pre + key, [None] * mC["n"])[r["k_C"]] = val
        epi_first = [("formation, control (%g)" % TrsC[kfK], kfK, TrsC[kfK])] if kfK is not None else []
        if kfC is not None and kfC != kfK:
            epi_first.append(("formation, C (%g)" % TrsC[kfC], kfC, TrsC[kfC]))
        keK_C = ctx["rev"].get(F["episode_control"]["last_j"]) if F["episode_control"]["last_j"] is not None else None
        mid = [c for c in (16.0, 17.0, 18.0, 19.0) if find_clock(TrsC, c) is not None]
        rows_b = rows_for(TrsC, epi_first, mid)
        if keK_C is not None:
            rows_b.append(("episode end, control (%g)" % TrsC[keK_C], keK_C, TrsC[keK_C]))
        rows_b += rows_for(TrsC, [], (20.0, 24.0, 28.0, 32.0, 36.0, 40.0))
        tb1 = build_table(rows_b, [("R (ctrl)", "b.R", 9), ("R_t C", "b.R_C", 9), ("dR_t meas", "b.dR_measured", 4), ("kappa_T", "b.kappa_T_stored", 5), ("kappa_fd", "b.kappa_fd_sites", 5),
                                   ("D_R (kappa)", "b.D_R_kappa", 5), ("D_R (rho sites)", "b.D_R_rho_sites", 5), ("D_R (rho smoothed)", "b.D_R_rho_smoothed", 5), ("D_R (nodes)", "b.D_R_nodes", 5),
                                   ("I_L sites", "b.I_L_sites", 5), ("I_L smoothed", "b.I_L_sites_smoothed", 5), ("I_L nodes", "b.I_L_nodes", 5), ("delta R_L", "b.dR_L", 4), ("spread", "b.dR_L_spread", 3),
                                   ("ratio meas/delta R_L", "b.ratio_measured_over_dR_L", 4), ("same sign", "b.same_sign", 1), ("2gM_H(R)/R sites", "b.two_gM_over_R_sites", 6),
                                   ("stored", "b.two_gM_over_R_stored", 6), ("u(R)+1", "b.u_plus_1", 3), ("M_L(R)/M_H(R)", "b.M_L_over_M_H_at_R", 4)], Db)
        tb2 = build_table(rows_b, [("r_in (ctrl)", "bi.R", 9), ("r_in C", "bi.R_C", 9), ("dr_in meas", "bi.dR_measured", 4), ("kappa_fd", "bi.kappa_fd_sites", 5), ("D_R (kappa_fd)", "bi.D_R_kappa", 5),
                                   ("D_R (rho sites)", "bi.D_R_rho_sites", 5), ("I_L sites", "bi.I_L_sites", 5), ("delta R_L", "bi.dR_L", 4), ("ratio", "bi.ratio_measured_over_dR_L", 4),
                                   ("same sign", "bi.same_sign", 1), ("2gM_H/r sites", "bi.two_gM_over_R_sites", 6)], Db)
        Dm = {}
        M_ = B["margin_control"]
        for i, k in enumerate(M_["k_C"]):
            if k is None:
                continue
            for key in ("min_M_H_outside", "r_min_M_H", "min_margin", "r_min_margin", "min_margin_B", "r_min_margin_B", "min_M_L_outside", "max_M_L_outside", "M_L_at_R_t", "M_L_over_M_H_at_R_t",
                        "min_margin_nodes", "margin_minus_eps_M"):
                Dm.setdefault("m." + key, [None] * mC["n"])[k] = M_[key][i]
        if B["margin_C_actual"] is not None:
            MC_ = B["margin_C_actual"]
            for i, k in enumerate(MC_["k_C"]):
                if k >= mC["n"]:
                    continue
                for key in ("min_M_outside", "min_margin", "r_min_margin", "min_M_L_outside", "max_M_L_outside", "M_L_at_R_t", "components_vs_M_rel_residual"):
                    Dm.setdefault("mc." + key, [None] * mC["n"])[k] = MC_[key][i]
        tb3 = build_table(rows, [("min M_H (r>ell)", "m.min_M_H_outside", 6), ("at r", "m.r_min_M_H", 5), ("min (M_H - [-M_L]_+)", "m.min_margin", 6), ("at r", "m.r_min_margin", 5),
                                 ("min (M_H - B_M)", "m.min_margin_B", 6), ("at r", "m.r_min_margin_B", 5), ("min M_L", "m.min_M_L_outside", 4), ("max M_L", "m.max_M_L_outside", 4),
                                 ("M_L(R_t)", "m.M_L_at_R_t", 4), ("M_L/M_H at R_t", "m.M_L_over_M_H_at_R_t", 4), ("min margin (route K)", "m.min_margin_nodes", 6), ("margin - eps_M", "m.margin_minus_eps_M", 4)], Dm)
        tb4 = build_table(rows, [("min M (stored, r>ell)", "mc.min_M_outside", 6), ("min (M_H^C - [-M_L^C]_+)", "mc.min_margin", 6), ("at r", "mc.r_min_margin", 5), ("min M_L^C", "mc.min_M_L_outside", 4),
                                 ("max M_L^C", "mc.max_M_L_outside", 4), ("M_L^C(R_t)", "mc.M_L_at_R_t", 4), ("components vs M (rel)", "mc.components_vs_M_rel_residual", 3)], Dm)
        E_ = B["episode_summary"]
        L += ["", "### 3.2 The outer root: I_L, D_R, delta R_L against the measured R_t(C) - R_t(control)", "",
              "delta R_L = G I_L(R)/(R kappa_T) (the site densities, the primary evaluation); beside it I_L from the site densities smoothed by the script and from the engine-smoothed node densities "
              "(route K), D_R from the heavy densities (CF4's cross-check: D_R(rho)/D_R(kappa) should be 1 up to the smoothing of the local density), 2 g M_H(R)/R (1 at a root), u(R) + 1 (0 at "
              "a root of N + v). The table clocks 20 to 40 carry no dense root in either run where the row is empty.", "", tb1, "", "### 3.3 The inner root (kappa from the sites; D_R < 0 there)", "",
              tb2, "", "### 3.4 The positive-mass margin per record", "",
              "On the control's fields (M_H the stored M, M_L the first-order light contribution at the control's u; r > ell = %s; eps_M %s):" % (g(mK["ell"], 4), g(ctx["eps_M"], 4) if ctx["eps_M"] is not None else "not supplied by the bound, margins against 0"),
              "", tb3, "", "On C's actual fields (M the stored total, M_L^C from the stored smoothed light fed components, M_H^C = M - M_L^C):", "", tb4, "",
              "The episode: %d outer roots in the control, %d with a measured displacement in C, the same sign on %d and the opposite on %d; the median ratio Delta R_t/delta R_L %s (median |log10| %s); "
              "delta R_L positive on %d and negative on %d; |delta R_L| at most %s against |Delta R_t| at most %s."
              % (E_["outer_roots"], E_["with_measured_dR"], E_["same_sign"], E_["opposite_sign"], g(E_["median_ratio"], 4), g(E_["median_abs_log10_ratio"], 3), E_["dR_L_sign_counts"]["positive"],
                 E_["dR_L_sign_counts"]["negative"], g(E_["max_abs_dR_L"], 4), g(E_["max_abs_dR_measured"], 4)),
              "", "### 3.5 The verdict under the rule of section 0", "",
              "**VERDICT FOR THE LEAD: %s**" % V["verdict"], "",
              "- basis: %s" % V["basis"], "- at the control's formation record: delta R_L = %s (floor %s, its evaluations' spread), Delta R_t = %s (floor %s lattice units), ratio %s; the formation shift %s records"
              % (g(V["dR_L_at_formation"], 5), g(V["floor_dR_L"], 3), g(V["dR_measured_at_formation"], 5), g(V["floor_dR_t_lattice_units"], 3), g(V["ratio_at_formation"], 4), g(V["formation_shift_records"], 3))]
        if V["flags"]:
            for f in V["flags"]:
                L.append("- %s" % f)
        L += ["", "### 3.6 The identities and hypotheses of part (b)", "",
              mdtab(["identity", "kind", "n", "max |residual|", "note"], [[nm, B["identities"][nm]["kind"], str(B["identities"][nm]["n"]), fm(B["identities"][nm]["max_abs_residual"], 3), B["identities"][nm]["note"]] for nm in B["identities_order"]])]
    L += ["", "## 4. The identities of this version (both runs)", "",
          mdtab(["identity", "kind", "n", "max |residual|", "note"], [[nm, ctx["ids"].d[nm]["kind"], str(ctx["ids"].d[nm]["n"]), fm(ctx["ids"].d[nm]["max_abs_residual"], 3), ctx["ids"].d[nm]["note"]] for nm in ctx["ids"].order])]
    L += ["", "## 5. Version 1's readouts (information)", "",
          "Readouts 3 and 5 below compare against the 1024-site plain-mixer production and the light-follow on it: not solver-matched; superseded as the coupling's readout by section 2 "
          "and kept as the comparison against the production. Readouts 2, 4, 7 and the 'also printed' list stand as version 1 prints them.", ""]
    if ctx["v1_md"]:
        L.append(ctx["v1_md"].replace("\n# ", "\n### ").replace("\n## ", "\n### "))
    else:
        L.append("(version 1 did not run: %s)" % ctx["v1_status"])
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------------------ run
def run_v2(o):
    t0 = datetime.datetime.now(datetime.timezone.utc)
    roles = (("run_json", o["run_json"]), ("s11_npz", o["s11_npz"]), ("control_json", o["control_json"]), ("control_fields_npz", o["control_fields_npz"]),
             ("control_s11_npz", o["control_s11_npz"]), ("follow_json", o["follow_json"]), ("prod_json", o["prod_json"]))
    inputs = []
    for role, p in roles:
        if not os.path.isfile(p):
            raise ReadoutError("%s: not a file: %s" % (role, p))
        inputs.append({"role": role, "path": os.path.abspath(p), "md5": md5_of(p), "bytes": os.path.getsize(p)})
    ids = Identities()
    run = load_json(o["run_json"])
    SC, mC = extract_min(run, "run")
    del run
    ctl = load_json(o["control_json"])
    SK, mK = extract_min(ctl, "control")
    del ctl
    if str(mC["coupling"]) != "on":
        raise Inconsistent("run: args.light_coupling is %r, not 'on'" % (mC["coupling"],))
    if str(mK["coupling"]) != "off":
        raise Inconsistent("control: args.light_coupling is %r, not 'off'" % (mK["coupling"],))
    dargs = V1.args_diff(mC["args"], mK["args"])
    unexpected = [k for k, _, _ in dargs if k not in EXPECTED_ARG_DIFF]
    if unexpected and not o["allow_arg_mismatch"]:
        raise Inconsistent("the control is not solver-matched: args differ beyond light_coupling in %s (pass --allow-arg-mismatch to proceed with the warning)"
                           % "; ".join("%s: C %s, control %s" % (k, fm(va, 6), fm(vb, 6)) for k, va, vb in dargs if k in unexpected))
    TrsC, TrsK = SC.d["T_over_rs"], SK.d["T_over_rs"]
    iK = align(TrsC, TrsK)
    pairs = [(k, j) for k, j in enumerate(iK) if j is not None]
    rev = {j: k for k, j in pairs}
    complete = (mC["n"] == 641 and mK["n"] == 641 and len(pairs) == 641)
    if not complete and not o["allow_partial"]:
        raise Inconsistent("C has %d records and the control %d, %d matched clocks (641 expected for both): wait for the control's end or pass --allow-partial" % (mC["n"], mK["n"], len(pairs)))
    if not pairs:
        raise Inconsistent("no matched clocks between C and the control")
    ids.add("C, control: tau, rs, ell, g equal", [abs(mC[q] - mK[q]) for q in ("tau", "rs", "ell", "g")])
    ids.add("C, control: T_over_rs equal at the matched clocks", [abs(TrsC[k] - TrsK[j]) for k, j in pairs])
    k0 = find_clock(TrsC, 0.0)
    if k0 is not None and iK[k0] is not None:
        j0 = iK[k0]
        ids.add("C, control: E at T = 0 (the same preparation)", [abs(SC.d["E"][k0] - SK.d["E"][j0])] if av(SC.d["E"][k0], SK.d["E"][j0]) else [], kind="cross", floor=3.0e-7 * abs(SC.d["E"][k0]),
                note="the gate's E/E0 floor 3.0e-7 times E")
        ids.add("C, control: min(N - |v|) at T = 0", [abs(SC.d["min_N_minus_absv"][k0] - SK.d["min_N_minus_absv"][j0])] if av(SC.d["min_N_minus_absv"][k0], SK.d["min_N_minus_absv"][j0]) else [],
                kind="cross", floor=4.6e-8, note="the gate's floor")
    ids.add("C, control: r_collar equal at the matched clocks", [abs(float(SC.d["r_collar"][k]) - float(SK.d["r_collar"][j])) for k, j in pairs if av(SC.d["r_collar"][k], SK.d["r_collar"][j])], kind="cross",
            note="the collar cut currents of both runs are at the same cut")
    ZC = load_s11(o["s11_npz"], "s11_npz")
    ZK = load_s11(o["control_s11_npz"], "control_s11_npz")
    # ---- version 1, repeated
    v1_out, v1_md, v1_status = None, None, "skipped (--skip-v1)"
    if not o["skip_v1"]:
        try:
            v1_out = V1.run_readout(o["run_json"], o["s11_npz"], o["follow_json"], o["prod_json"], o["out_dir"], o["tag"] + "_v1part", o.get("slice_dir"), quiet=True)
            with open(v1_out["written"]["md"]) as f:
                v1_md = f.read()
            v1_status = "written: %s and %s" % (v1_out["written"]["json"], v1_out["written"]["md"])
        except Exception as e:  # noqa: BLE001  version 1's failure is reported, not fatal for (a), (b), (c)
            v1_status = "FAILED: %s: %s" % (type(e).__name__, e)
    follow = v1_out["series"].get("F.E_rad0_over_E") if (v1_out is not None and "series" in v1_out) else None
    if follow is not None and len(follow) != mC["n"]:
        follow = None
    # ---- (a)
    A_D, A_form, A_sum = compute_a(SC, mC, SK, mK, iK, rev, pairs, follow)
    S11X = s11_common_nodes(ZC, ZK, pairs, mC["ell"], TrsC)
    ids.add("s11 npz: r_nodes equal (C, control) at the matched clocks", [0.0 if e else 1.0 for e in S11X["nodes_equal"]], kind="cross", note="1 where interpolated")
    # ---- (b)
    B, b_stop, b_tb = None, None, None
    try:
        FK = load_fields(o["control_fields_npz"], "control_fields_npz")
        B = compute_b(SC, mC, SK, mK, iK, rev, pairs, FK, ZK, ZC, o)
    except ReadoutError as e:
        b_stop = "%s: %s" % (type(e).__name__, e)
        b_tb = traceback.format_exc()
        print("STOP in part (b) (%s): %s" % (type(e).__name__, e), file=sys.stderr)
    V = None
    if B is not None:
        kfC = A_form["C"]["k"] if A_form["C"] else None
        jfK = A_form["control"]["j"] if A_form["control"] else None
        V = judge(B, kfC, jfK, rev, TrsC, TrsK, o["floor_dR"], o.get("floor_dRL"), mC["tau"] / mC["rs"])
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ctx = {"mC": mC, "mK": mK, "SC": SC, "SK": SK, "pairs": pairs, "rev": rev, "complete": complete, "args_diff": dargs, "unexpected_args": unexpected, "inputs": inputs,
           "A_D": A_D, "A_formation": A_form, "A_summary": A_sum, "S11X": S11X, "B": B, "V": V, "b_stop": b_stop, "ids": ids, "v1_status": v1_status, "v1_md": v1_md,
           "generated": generated, "eps_M": o.get("eps_M")}
    md = build_md(ctx)
    out = {"version": VERSION, "label": LABEL, "bound": BOUND, "generated_utc": generated, "inputs": inputs, "rule": [{"head": h, "text": t} for h, t in RULE],
           "meta": {"C": {"script": mC["script"], "tag": mC["tag"], "n": mC["n"], "coupling": mC["coupling"], "nsites": mC["nsites"]},
                    "control": {"script": mK["script"], "tag": mK["tag"], "n": mK["n"], "coupling": mK["coupling"], "nsites": mK["nsites"]},
                    "tau": mC["tau"], "rs": mC["rs"], "ell": mC["ell"], "g": mC["g"], "channels": mC["channels"], "sourcing": mC["sourcing"], "guard": mC["guard"],
                    "matched_clocks": len(pairs), "complete": complete, "args_diff": [list(x) for x in dargs], "unexpected_args": unexpected, "options": o},
           "a": {"series": A_D, "formation": A_form, "summary": A_sum, "s11_common_nodes": S11X},
           "b": ({"stopped": b_stop, "traceback": b_tb} if B is None else dict(B, stopped=None)),
           "verdict": V, "identities": ids.d, "identities_order": ids.order,
           "v1": {"status": v1_status, "written": (v1_out.get("written") if v1_out else None), "readouts": (v1_out.get("readouts") if v1_out else None),
                  "formation": (v1_out.get("formation") if v1_out else None)},
           "seconds": (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()}
    os.makedirs(o["out_dir"], exist_ok=True)
    pj, pm = os.path.join(o["out_dir"], o["tag"] + ".json"), os.path.join(o["out_dir"], o["tag"] + ".md")
    with open(pj, "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    with open(pm, "w") as f:
        f.write(md)
    if not o.get("quiet"):
        print(md)
        print("written: %s and %s (%.1f s)" % (pj, pm, out["seconds"]))
        if V is not None:
            print("VERDICT FOR THE LEAD: %s" % V["verdict"])
    if B is None:
        print("part (b) STOPPED: %s" % b_stop, file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Computation C: the sigma/a 4 member against its coupling-off control (parts (a), (b), (c)); version 1 repeated by import.")
    ap.add_argument("--run-json", required=True, help="the coupled member's record json")
    ap.add_argument("--s11-npz", required=True, help="the coupled member's _s11 npz")
    ap.add_argument("--control-json", required=True, help="the coupling-off control's record json")
    ap.add_argument("--control-fields-npz", required=True, help="the control's _fields npz")
    ap.add_argument("--control-s11-npz", required=True, help="the control's _s11 npz")
    ap.add_argument("--follow-json", required=True, help="gate F's light-follow json (version 1's input, information)")
    ap.add_argument("--prod-json", required=True, help="the 1-field production's json (version 1's input, information)")
    ap.add_argument("--out-dir", default="busi/results/compC")
    ap.add_argument("--tag", default="C_readout_s4_v2")
    ap.add_argument("--slice-dir", default=None, help="passed to version 1 (lists the slice clocks)")
    ap.add_argument("--floor-dR", type=float, default=FLOOR_DR_T, help="the floor of R_t(C) - R_t(control), lattice units (default the gate's 3.4e-6)")
    ap.add_argument("--floor-dRL", type=float, default=None, help="the floor of delta R_L (default: the spread of its evaluations plus roundoff)")
    ap.add_argument("--eps-M", type=float, default=None, help="an enclosed-mass error allowance for the margin (the bound supplies none)")
    ap.add_argument("--sigma-smooth", type=float, default=None, help="the smoothing estimate's sigma (default ell)")
    ap.add_argument("--allow-partial", action="store_true", help="read a control that has not reached 641 records (common clocks only)")
    ap.add_argument("--allow-arg-mismatch", action="store_true", help="proceed when args differ beyond light_coupling (printed as NOT SOLVER-MATCHED)")
    ap.add_argument("--skip-v1", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    o = {k.replace("-", "_"): v for k, v in vars(a).items()}
    try:
        return run_v2(o)
    except ReadoutError as e:
        print("STOP (%s): %s" % (type(e).__name__, e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
