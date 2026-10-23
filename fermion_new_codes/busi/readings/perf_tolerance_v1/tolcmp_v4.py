#!/usr/bin/env python
"""tolcmp_v4.py (busi lane, 21 Oct): tolcmp_v3 (5ae7977b) rebuilt under ReviewA_s3b_v1 sec. 1 (deliveries/lead_gaps/): the record's mode is
measured, not presumed, and chooses the instrument. v1, v1a, v2 and v3 untouched; v4 writes v4_*.json. No line moves.
Changes against v3 (the pairwise machinery, the table's lines, v3's nocc and growth changes are v3's):
  (1) The mode read, per record, before the field-derived classes are decided. Inputs: the Newton counts (record 'iterations') and kept
      residuals (record solve_v32/kept_residual) of X, D and P (P where it has the record), and D-P's site fields.
        C  a flip: the counts differ at the record; and every later record of a flipped segment that does not measure A (v2 2(c): "at it and at
           every later record").                                                                  -> 2(c)(1)-(4) as v2.
        A  a shared iterate: counts equal, kept residuals equal to four significant figures in all three arms, and D-P's site fields within
           the same-iterate floor measured at the size (4096: 3e-13 across cards; 2048: 2.5e-10 at a resume, the replays being resumes;
           another size: no floor, A not readable), compared at the floor's own stated precision.  -> the table's lines on X-D and on X-P
           (E/E0, E, roots, depth, fields, rows, spectra, nocc, named, decisions); the per-call text reported, not required; no ratio.
        B  same count, kept residuals different (or A not established), some kept residual above erratum v3's 1e-8 trigger.
                                                                                                   -> 2(c)(1)-(4) as v2.
        T  before any flip, counts equal and no arm's kept residual above 1e-8 (or no kept residual: a record without the Newton entry)
           -> the table on X-D, as v2 (b) and v3's pre-flip reading. Erratum v3's trigger "stays as the point above which the mode must be checked".
      The ratio (2) is read in B and C only. E/E0, roots, depth, R_sh, the identical class and the masks keep their (A) lines in every mode.
  (2) Beside each record's verdict: the named class against its own line, max |d| / (1e-6 |x| + 1e-8) and the max relative difference,
      X-D and X-P (gating in T and A, reported in B and C), and the counts, kept residuals and D-P fields that fixed the mode.
  (3) rotated_blocks (ReviewA_s3b_v1 sec. 1, on the v16e (iv) answer that only the log consumes it): a counter, counted as a decision only at a
      record where a set, mask or level label differs with it (any other decision or a spectra mask entry at the same record, same pair).
  (4) --min-records (default 8, sec. 4 (A)'s replay length); a 2-record entry test passes 2. --floor overrides the floor with a measured value.
  (5) The per-call text (2(c)(3)) is required where any record reads B or C (missing: INCONCLUSIVE); v3 required it from a count flip.
  (6) The slice's densities and nodes: by the table (X-D; and X-P in A) when the last record reads T or A, by the three-way test in B or C.
v3 docstring follows.
tolcmp_v3.py (busi lane, 21 Oct): tolcmp_v2 (d9b9ee32) under the RD-1c ruling (deliveries/lead_gaps/ReviewA_rd1c_v1.md sec. 1 and
sec. 2 condition (1)), an alignment of the comparator to the written table 2.3 of REVIEWA_perf_tolerance_ruling_v1; no threshold changes.
v1, v1a and v2 untouched; v2's outputs are kept beside v3's (v3 writes v3_*.json).
Changes against v2 (nothing else):
  (1) nocc on the spectra line. v2 kept the nocc rows at every record with an exact-zero test (`v["nocc"] == 0.0`); that test is withdrawn.
      The nocc_* rows are in table 2.3's spectra class: before the first stop flip they pass at 1e-8 absolute (the spectra line, beside the
      e rows' 1e-8 relative); at and after a flip they are decided with the spectra rows by 2(c) (three_way X-P at most 2 x D-P). The
      discrete content (which levels are occupied) stays gated bitwise through the masks.
  (2) The growth quarters are taken over the records from the first record at which X and D differ in E/E0 (the growth line's quantity,
      nonzero; a resume's first record can carry E/E0 exactly equal in both arms, which made v2's first-quarter median 0). The records from it on
      are split into four quarters (numpy.array_split); the gate is v2's (last-quarter median E/E0 X-D at most 10 x the first quarter's, or
      at most 1e-12); the four quarter medians of X-D, X-P and D-P are printed (X-P and D-P reported, not gated).
v2 docstring follows.
tolcmp_v2.py (busi lane, 21 Oct): the tolerance comparator under ruling v2 (deliveries/lead_gaps/ReviewA_tolv2_v1.md sec. 2 (a) to (c),
which supersedes erratum v1a). v1 and v1a untouched; the pairwise machinery is v1a's, the gate is new.

One call reads a replay three ways: X (change arm) against D (default arm), X against the stored production P, D against P.
  2(a) identical class: /v36c/nk_set, record_set, each solve's kept_round, kept_set, kept_sig and sets, the masks, and every other
       decision of the v1 table (continuation labels, status, meets_stop_rule, ...). Reported: every entry under steps[*] (and its mirror
       line_search[*]) and trials[*], the histories, iteration_history_slice, evaluation_sets (one entry per evaluation), node residuals,
       the counters (kept_evaluation, evaluations, extra_evaluations, memo_dropped, cache hits/misses), a build's own reporting entry
       (/rd13, /fields_rd2, /M9, /perf). previous_record_set and equals_previous_record are compared where both sides carry a value and
       reported at a resume's first record; the solve-level kept_set_equals_previous_record and rounds[*]/flips_vs_previous get the same
       rule (busi's extension, printed as such).
  2(b) lines: the v1 table at every record before the first stop flip, fields 1e-7 for every family.
  2(c) stop flip: a record where the three iteration counts (X, D, P where P has the record) are not all equal is a stop-flip record;
       at it and every later record E/E0 3.0e-8, roots 3.4e-7, depth 4.6e-9, R_sh, the identical class and the masks keep their (A)
       lines; the fields, named, rows, spectra rows and the slice's densities and nodes are decided by (1) X and D each within C of P
       (E/E0 3.0e-7, roots 3.4e-6, depth 4.6e-8, fields 2.5e-5) with the set decisions identical, (2) X's distance from P at most twice
       D's in each of those classes at each record, (3) the per-call evidence (--percall TEXT, a pointer to the stored per-call test at
       the flipped record's inputs; without it (3) is OPEN and the verdict INCONCLUSIVE), (4) the three counts printed. A post-flip
       record P lacks is INCONCLUSIVE for those classes. The slice densities after a flip need P's slice at the end clock (--slice-p);
       without one that line is reported as not readable.
Shared with cmp_rd2_ruling_v2 (written by another busi agent): the flip rule is flip_records() below, the three-way rule three_way();
both are plain functions of per-record numbers, importable.

    tolcmp_v2.py X_DIR D_DIR P_DIR --tag TAG --after T --rs RS [--slice-x P --slice-d P --slice-p P] [--percall TEXT] [--out DIR]
       [--names X,D,P]   (each DIR holds collapse_TAG{,_fields,_spectra,_handoff}.{json,npz})

v1a docstring follows.
tolcmp_v1a.py (busi lane, 21 Oct): tolcmp_v1.py under erratum v1a of the tolerance ruling (ReviewA_rd1gate_v1.md sec. 3 (i)).
Changes against v1 (v1 untouched):
  (1) the v36c set entry: previous_record_set and equals_previous_record (any leaf whose key is one of these two) are compared only
      where both sides carry a value (neither None nor absent); where one side has none the pair is listed under "resume labels"
      (reported, normally the resume's first record), not counted as a decision. record_set and the masks stay decisions.
  (2) --gate A|B prints the ruling's sec. 4 lines per record and overall: (A) the table (E/E0 3.0e-8, E rel 3.0e-8, roots 3.4e-7,
      R_sh equal, depth 4.6e-9, named |d| <= 1e-6 |x| + 1e-8, site fields 1e-7, or 2.5e-6 with --engine newton (erratum v1a: the Newton
      engines' applied fields, C/10 of the fields' C 2.5e-5), rows 1e-6, spectra 1e-8 relative with every mask equal, slice densities 1e-9
      with f identical and masks equal, 0 decisions, no monotone growth of E/E0 (last-quarter median <= 10x the first quarter's));
      (B) C (E/E0 3.0e-7, roots 3.4e-6, depth 4.6e-8, fields 2.5e-5) with 0 decisions. The site-fields class is N, v, N_applied,
      v_applied, N_nodes, v_nodes of the fields npz (and the slice's dg nodes); the rows class is every other fields npz row, the handoff
      and s11 rows.
  (3) --same-class X_vs_prod.json --default-vs-prod Y.json (with --gate B): per record and class, the change arm's distance from the
      production at most twice the default arm's or below the (A) line.
v1 docstring follows.
tolcmp_v1.py (busi lane, reviewer A perf tolerance ruling v1): two engine runs compared by quantity class, independently of the builders'
comparators. Every record at the common clocks after --after, plus the npz rows there (fields, spectra, handoff, s11) and, optionally, one slice.

Classes (the ruling's table):
  energies     E (relative), E_over_E0 (absolute)
  roots        R_t, r_in, R_c, r_min_N_minus_absv, trapping_zeros[*], threshold_radius/* (absolute, lattice units); R_sh exact
  depth        min_N_minus_absv (absolute)
  named        N_min, kappa_T, H_in, E_dust_shells, E_pairs_shells, pairs_inside_Rt, pairs_outside_Rt, misner_sharp_at_Rt, v_at_Rc,
               min_occupied_weight, M1_I_E, R2_minus_R0_F, commutator_h_R2_F, dE_dT, dHin_dT, dRc_dT (relative)
  indicator    indicator/* floats (B_estimate, delta_rtol ...) reported with both values; indicator_pass as a bit
  counters     int leaves whose key is a work counter (nfev, iterations, solves, eigendecompositions, hits, misses, products, pieces,
               probes, eigvals, eigh, calls, evaluations, attempts? no: attempts is a decision) -> reported
  decisions    every other bool / int / str / None leaf, list lengths and key sets -> must be identical
  other floats reported (max abs, max rel per key)
  fields       npz rows: max |d| and max |d| / max |row|
  spectra      masks identical (uint8 rows); e rows max |d| and relative to max |e|; nocc rows
  slice        masks per channel identical; e, densities, dust weights, dg nodes, level

Skipped: walls (seconds, timing/*_s, keys ending _s, slice_saved, start), path strings (label, record_label, engine_v34, engine),
hbar_prop_v36w2, iteration_history (an iterate-level trace whose length is the iteration count; reported through 'iterations').

Usage: tolcmp_v1.py A_DIR B_DIR --tag TAG [--after T] [--rs 64] [--out out.json] [--slice-a P --slice-b P] [--nameA x --nameB y]
       [--prod-json P --prod-npz-dir D] (B taken from a production json + a mirror dir instead of a run dir)
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

SKIP_KEYS = {"seconds", "timing", "slice_saved", "label", "record_label", "engine_v34", "engine", "hbar_prop_v36w2", "start",
             "iteration_history", "script", "path", "out", "slice_dir", "resume_slice", "resume_run"}
COUNTER_RE = re.compile(r"(^|/)(nfev|solve_nfev|iterations|iteration|eigendecompositions|solves|n_solves|hits|misses|products|pieces|"
                        r"probes|probes_in_crossing|eigvals|eigh|calls|evaluations|n_eval|nit|njev|nlu|computed|evicted|hit|"
                        r"rotated_blocks|light_sector_eigh|ladder_sector_pairs|sectorize_exact|sectorize_rotation_kept|cholesky|"
                        r"cholesky_calls|band_plan_hit|band_plan_rebuild|gram_gathers|gram_d2h_bytes_avoided|kernel_cache_hits|"
                        r"n_products|n_jvp|steps|nsteps|nfev_total|it|eigh_count|eigs|n_eigh|eig_computed|eig_hit|eig_evicted|"
                        r"[a-z0-9_]*_hits|[a-z0-9_]*_misses)$")  # v1a: cache hits and misses with a prefix (worker_kcache_hits) are counters
ROOT_KEYS = {"R_t", "r_in", "R_c", "r_min_N_minus_absv"}
NAMED_KEYS = {"N_min", "kappa_T", "H_in", "E_dust_shells", "E_pairs_shells", "pairs_inside_Rt", "pairs_outside_Rt", "misner_sharp_at_Rt",
              "v_at_Rc", "min_occupied_weight", "M1_I_E", "R_t_over_2gE", "W_over_2g_at_Rt", "R_t_over_2g_MS", "four_gE_kappa_T"}
RESIDUAL_KEYS = {"R2_minus_R0_F", "commutator_h_R2_F", "dE_dT", "dHin_dT", "dRc_dT", "balance_d_E_minus_Hin_dT", "intermediate_weights_total"}
TRACE_RE = re.compile(r"(/history|_hist|node_residuals|/iteration_history)")
DIAG_FLOAT = {"sigma_impurity_max"}
# v1a: the Newton solve's per-step and per-trial traces (lists whose length is the iteration or trial count, like iteration_history which v1
# skips) are the table's "iterate traces" (reported); the record's set decisions (v36c record_set, the masks, each solve's status and
# meets_stop_rule) stay decisions. Counted separately as "newton traces"; --strict-traces counts them as decisions (v1's reading).
NEWTON_TRACE_RE = re.compile(r"^/solve_v32/solves\[\d+\]/(line_search|v36c/steps|v36c/evaluation_sets|iteration_history_slice)|"
                             r"^/solve_v32/solves\[\d+\]/v36c/rounds\[\d+\]/(steps|history|best|evaluations)|"
                             r"^/solve_v32/(solves\[\d+\]/)?(kept_evaluation|evaluations)$|^/jvp_v35/extra_evaluations$|^/v36m/memo_dropped$")
STRICT_TRACES = False


def leaves(x, p=""):
    """(path, value) for every scalar leaf; lists give '#len'."""
    if isinstance(x, dict):
        for k, v in x.items():
            ks = str(k)
            if ks in SKIP_KEYS or ks.endswith("_s"):
                continue
            yield from leaves(v, p + "/" + ks)
    elif isinstance(x, (list, tuple)):
        yield p + "#len", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x


def keyname(path):
    return re.sub(r"\[\d+\]", "[]", path)


def classify(path, x, y):
    """-> class name for a float leaf."""
    last = path.rsplit("/", 1)[-1]
    last = re.sub(r"\[\d+\]$", "", last)
    top = path.split("/")[1] if path.count("/") >= 1 else ""
    if path == "/E":
        return "E"
    if path == "/E_over_E0":
        return "E_over_E0"
    if path == "/R_sh":
        return "R_sh"
    if path == "/r_min_N_minus_absv":
        return "argmin"
    if path.lstrip("/") in ROOT_KEYS or top in ("trapping_zeros", "threshold_radius"):
        return "roots"
    if path == "/min_N_minus_absv":
        return "depth"
    if path.lstrip("/") in NAMED_KEYS:
        return "named"
    if path.lstrip("/") in RESIDUAL_KEYS:
        return "residual"
    if TRACE_RE.search(path):
        return "trace"
    if top == "indicator":
        return "indicator"
    if last in DIAG_FLOAT:
        return "diag"
    return "other"


class Acc(object):
    def __init__(self):
        self.cls = {}      # class -> list of (abs, rel, path)
        self.other = {}    # keyname -> (abs, rel, path)
        self.decisions = []
        self.counters = []
        self.bits = []
        self.perkey = {}
        self.resume = []
        self.ntraces = []  # v1a: Newton per-step / per-trial trace leaves that differ (reported)
        self.named_xy = [] # v1a: (abs, rel, path, x, y) of the named class   # v1a: (path, x, y) of previous_record_set / equals_previous_record with a value on one side only

    def add(self, c, ab, rel, path):
        self.cls.setdefault(c, []).append((ab, rel, path))


RESUME_KEYS = {"previous_record_set", "equals_previous_record"}
# the same operand-absent comparison inside each solve's v36c entry (a value against the previous record's set, None at a resume's first
# record); listed under the resume labels with the erratum's two keys, marked "beyond the erratum's two keys"
RESUME_KEYS_X = {"kept_set_equals_previous_record", "flips_vs_previous"}
RESUME_KEYS = RESUME_KEYS | RESUME_KEYS_X


def is_resume_key(k):
    last = re.sub(r"(\[\d+\]|#len)$", "", k.rsplit("/", 1)[-1])
    return last in RESUME_KEYS or any(("/%s/" % q) in (k + "/") or ("/%s[" % q) in k or ("/%s#" % q) in k for q in RESUME_KEYS)


def cmp_record(a, b, acc, t):
    La, Lb = dict(leaves(a)), dict(leaves(b))
    # v1a: a resume key is compared only where both sides carry a value at its top (None or absent on either side -> resume label)
    tops = {}
    for k in set(La) | set(Lb):
        if is_resume_key(k):
            m = re.search(r"^(.*?/(?:%s))" % "|".join(RESUME_KEYS), k)
            if m:
                tops.setdefault(m.group(1), []).append(k)
    for top, ks in tops.items():
        def val(L):
            if top in L:
                return L[top]
            sub = [k for k in ks if k in L]
            return "<value>" if sub else "<absent>"
        x, y = val(La), val(Lb)
        if x in (None, "<absent>") or y in (None, "<absent>"):
            acc.resume.append(("%.5f%s" % (t, top), str(x)[:60], str(y)[:60]) + (("beyond the erratum's two keys",) if top.rsplit("/", 1)[-1] in RESUME_KEYS_X else ()))
            for k in ks:
                La.pop(k, None); Lb.pop(k, None)
    for k in sorted(set(La) | set(Lb)):
        x, y = La.get(k, "<absent>"), Lb.get(k, "<absent>")
        p = "%.5f%s" % (t, k)
        kn = keyname(k)
        if not STRICT_TRACES and NEWTON_TRACE_RE.search(k) and ("<absent>" in (x, y) or not (isinstance(x, float) and isinstance(y, float))):
            if x != y:
                acc.ntraces.append((p, str(x)[:40], str(y)[:40]))
            continue
        if x == "<absent>" or y == "<absent>":
            if kn.startswith("/M9") or kn.startswith("/fields_rd2") or kn.startswith("/perf") or kn.startswith("/rd13") or TRACE_RE.search(kn):
                # v1a: /rd13 is RD-1/RD-3's own reporting entry (calls, banded, fallbacks), present only in the option arm: class (C)
                acc.counters.append((p, "absent", (x if y == "<absent>" else "<absent>", y if x == "<absent>" else "<absent>")))
            else:
                acc.decisions.append((p, "absent", (str(x)[:60], str(y)[:60])))
            continue
        if isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                if k.endswith("indicator_pass"):
                    acc.bits.append((p, x, y))
                else:
                    acc.decisions.append((p, "bool", (x, y)))
            continue
        if isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                acc.decisions.append((p, "label", (str(x)[:80], str(y)[:80])))
            continue
        if isinstance(x, int) and isinstance(y, int):
            if x != y:
                if COUNTER_RE.search(kn) or TRACE_RE.search(kn):
                    acc.counters.append((p, "int", (x, y)))
                else:
                    acc.decisions.append((p, "int", (x, y)))
            continue
        fx, fy = float(x), float(y)
        if (math.isnan(fx) and math.isnan(fy)):
            continue
        if math.isnan(fx) != math.isnan(fy) or math.isinf(fx) != math.isinf(fy):
            acc.decisions.append((p, "nan", (fx, fy))); continue
        d = abs(fx - fy)
        den = max(abs(fx), abs(fy))
        rel = d / den if den > 0 else 0.0
        c = classify(k, fx, fy)
        if c == "other":
            q = acc.other.get(kn)
            if q is None or rel > q[1]:
                acc.other[kn] = (d, rel, p, fx, fy)
        elif c == "indicator":
            acc.add("indicator", d, rel, "%s (%.6e vs %.6e)" % (p, fx, fy))
        elif c in ("trace", "residual"):
            acc.add(c, d, rel, "%s (%.3e vs %.3e)" % (p, fx, fy))
        else:
            acc.add(c, d, rel, "%s (%.12g vs %.12g)" % (p, fx, fy))
        if c == "named":
            acc.named_xy.append((d, rel, p, fx, fy))
        if c in ("roots", "named", "argmin"):
            q = acc.perkey.setdefault(c, {}).get(kn)
            if q is None or d > q[0]:
                acc.perkey[c][kn] = (d, rel, p, fx, fy)


def rows_by_T(Z, rs):
    T = np.asarray(Z["T"], float)
    return {round(float(t) / rs, 6): i for i, t in enumerate(T)}


def cmp_rows(ZA, ZB, ts, rs, name, res):
    ia, ib = rows_by_T(ZA, rs), rows_by_T(ZB, rs)
    keys = [k for k in ZA.files if k in ZB.files and k != "T" and ZA[k].ndim >= 1 and ZA[k].shape[0] == len(ia) and ZB[k].shape[0] == len(ib)]
    for k in keys:
        xa, xb = ZA[k], ZB[k]
        if xa.dtype.kind == "O":
            for t in ts:
                if t in ia and t in ib:
                    ra, rb = np.asarray(xa[ia[t]], float), np.asarray(xb[ib[t]], float)
                    if ra.shape != rb.shape:
                        res.setdefault(name + "/" + k, {})[t] = ("shape", ra.shape, rb.shape); continue
                    d = float(np.max(np.abs(ra - rb))) if ra.size else 0.0
                    sc = float(np.max(np.abs(ra))) if ra.size else 0.0
                    res.setdefault(name + "/" + k, {})[t] = (d, d / sc if sc > 0 else 0.0)
            continue
        if xa.dtype.kind in "SU":
            continue
        for t in ts:
            if t in ia and t in ib:
                ra, rb = xa[ia[t]], xb[ib[t]]
                if xa.dtype.kind in "iub":
                    res.setdefault(name + "/" + k, {})[t] = ("mask", int(np.sum(ra != rb)))
                else:
                    ra = np.asarray(ra, float); rb = np.asarray(rb, float)
                    d = float(np.max(np.abs(ra - rb))) if ra.size else 0.0
                    sc = float(np.max(np.abs(ra))) if ra.size else 0.0
                    res.setdefault(name + "/" + k, {})[t] = (d, d / sc if sc > 0 else 0.0)


def slice_cmp(pa, pb):
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    ks = sorted(int(k.split("_")[1]) for k in A.files if k.startswith("mask_") and k.split("_")[1].isdigit())
    out = {"masks_equal": True, "e": 0.0, "dens_sea": 0.0, "dens_R2": 0.0, "f": 0.0, "nodes": None, "level_equal": None, "channels": len(ks)}
    for k in ks:
        out["masks_equal"] &= bool(np.array_equal(A["mask_%d" % k], B["mask_%d" % k]))
        out["e"] = max(out["e"], float(np.max(np.abs(A["e_%d" % k] - B["e_%d" % k]))))
        for nm, key in (("dens_sea", "sea_%d"), ("dens_R2", "R2_%d")):
            if (key % k) in A.files and (key % k) in B.files:
                da = np.sum(np.abs(A[key % k]) ** 2, axis=1); db = np.sum(np.abs(B[key % k]) ** 2, axis=1)
                out[nm] = max(out[nm], float(np.max(np.abs(da - db))))
        if ("f_%d" % k) in A.files and ("f_%d" % k) in B.files:
            fa, fb = A["f_%d" % k], B["f_%d" % k]
            if fa.shape == fb.shape and fa.size:
                out["f"] = max(out["f"], float(np.max(np.abs(fa - fb))))
            elif fa.shape != fb.shape:
                out["masks_equal"] = False
    if "dg_nodes_N" in A.files and "dg_nodes_N" in B.files:
        out["nodes"] = max(float(np.max(np.abs(A["dg_nodes_N"] - B["dg_nodes_N"]))), float(np.max(np.abs(A["dg_nodes_v"] - B["dg_nodes_v"]))))
    if "dg_level_v36k" in A.files and "dg_level_v36k" in B.files:
        out["level_equal"] = int(A["dg_level_v36k"]) == int(B["dg_level_v36k"])
    # light masks (mask_L*)
    kl = [k for k in A.files if k.startswith("mask_L")]
    for k in kl:
        if k in B.files:
            out["masks_equal"] &= bool(np.array_equal(A[k], B[k]))
    return out


def load_run(d, tag, prod_json=None, prod_dir=None):
    J = json.load(open(prod_json if prod_json else os.path.join(d, "collapse_%s.json" % tag)))
    nd = prod_dir if prod_dir else d
    Z = {}
    for k in ("fields", "spectra", "handoff", "s11"):
        p = os.path.join(nd, "collapse_%s_%s.npz" % (tag, k))
        if os.path.exists(p):
            Z[k] = np.load(p, allow_pickle=True)
    return J, Z


def compare(a):
    """v1a's pairwise read; a has v1a's argument names. -> (out, per, ts, acc)."""
    JA, ZA = load_run(a.A, a.tag)
    JB, ZB = load_run(a.B, a.tag, a.prod_json, a.prod_npz_dir)
    rs = a.rs or float(JA["args"]["rs"])
    args_diff = sorted(k for k in set(JA["args"]) | set(JB["args"]) if JA["args"].get(k) != JB["args"].get(k))
    byT = lambda J: {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None}
    RA, RB = byT(JA), byT(JB)
    ts = sorted(t for t in set(RA) & set(RB) if a.after is None or t > a.after + 1e-9)
    acc = Acc()
    per = {}
    for t in ts:
        a1 = Acc()
        cmp_record(RA[t], RB[t], a1, t)
        per[t] = {c: max([v[0] for v in L] or [0.0]) for c, L in a1.cls.items()}
        per[t]["E_rel"] = max([v[1] for v in a1.cls.get("E", [])] or [0.0])
        per[t]["named_rel"] = max([v[1] for v in a1.cls.get("named", [])] or [0.0])
        per[t]["n_decisions"] = len(a1.decisions); per[t]["n_counters"] = len(a1.counters); per[t]["bits"] = len(a1.bits)
        for c, L in a1.cls.items():
            acc.cls.setdefault(c, []).extend(L)
        for kn, v in a1.other.items():
            q = acc.other.get(kn)
            if q is None or v[1] > q[1]:
                acc.other[kn] = v
        for c, D in a1.perkey.items():
            for kn, v in D.items():
                q = acc.perkey.setdefault(c, {}).get(kn)
                if q is None or v[0] > q[0]:
                    acc.perkey[c][kn] = v
        acc.decisions.extend(a1.decisions); acc.counters.extend(a1.counters); acc.bits.extend(a1.bits)
        acc.resume.extend(a1.resume); per[t]["n_resume_labels"] = len(a1.resume)
        acc.ntraces.extend(a1.ntraces); per[t]["n_newton_traces"] = len(a1.ntraces)
        per[t]["named_excess"] = max([0.0] + [v[0] - (1e-6 * max(abs(v[3]), abs(v[4])) + 1e-8) for v in a1.named_xy])
        # v4: the named class against its own line, as a ratio |d| / (1e-6 |x| + 1e-8) (max over the named leaves), with its leaf
        nr = [(v[0] / (1e-6 * max(abs(v[3]), abs(v[4])) + 1e-8), v[2]) for v in a1.named_xy]
        per[t]["named_ratio"], per[t]["named_ratio_at"] = max(nr) if nr else (0.0, None)
        # v4: rotated_blocks differences at this record (a counter; a decision only where a set, mask or level label differs with it)
        per[t]["rotated_blocks"] = [(p, v) for p, kind, v in a1.counters if "rotated_blocks" in p]
    rows = {}
    for k in ("fields", "spectra", "handoff", "s11"):
        if k in ZA and k in ZB:
            cmp_rows(ZA[k], ZB[k], ts, rs, k, rows)
    out = {"A": a.A, "B": a.B, "nameA": a.nameA, "nameB": a.nameB, "records": len(ts), "clocks": ts, "args_differ": args_diff,
           "per_record": {("%.5f" % t): per[t] for t in ts}, "rows": {}, "decisions": acc.decisions, "counters": acc.counters, "bits": acc.bits,
           "other": {k: {"abs": v[0], "rel": v[1], "at": v[2], "x": v[3], "y": v[4]} for k, v in acc.other.items()}}
    print("== %s against %s: %d common records%s; args differ in: %s" % (a.nameA, a.nameB, len(ts), (" after T/rs %g" % a.after) if a.after else "", args_diff or "nothing"))
    print("-- classes (max over the records; where)")
    summ = {}
    for c in ("E", "E_over_E0", "roots", "argmin", "R_sh", "depth", "named", "residual", "indicator", "trace", "diag"):
        L = acc.cls.get(c, [])
        if not L:
            summ[c] = None; print("   %-11s no differences" % c); continue
        ab = max(L, key=lambda v: v[0]); rl = max(L, key=lambda v: v[1])
        summ[c] = {"max_abs": ab[0], "at_abs": ab[2], "max_rel": rl[1], "at_rel": rl[2], "n": len(L)}
        print("   %-11s max abs %.3e at %s; max rel %.3e at %s (%d leaves differ)" % (c, ab[0], ab[2][:70], rl[1], rl[2][:70], len(L)))
    out["summary"] = summ
    out["perkey"] = {c: {k: {"abs": v[0], "rel": v[1], "at": v[2], "x": v[3], "y": v[4]} for k, v in D.items()} for c, D in acc.perkey.items()}
    for c in ("roots", "argmin", "named"):
        for k, v in sorted(acc.perkey.get(c, {}).items(), key=lambda kv: -kv[1][0]):
            print("     %-8s %-40s max abs %.3e rel %.3e at %s" % (c, k, v[0], v[1], v[2][:60]))
    print("-- per record: E/E0 |d|, E rel, roots |d|, depth |d|, named rel, decisions, counters, bits")
    for t in ts:
        p = per[t]
        print("   %.5f  %.2e  %.2e  %.2e  %.2e  %.2e  %d %d %d" % (t, p.get("E_over_E0", 0.0), p.get("E_rel", 0.0), p.get("roots", 0.0), p.get("depth", 0.0), p.get("named_rel", 0.0), p["n_decisions"], p["n_counters"], p["bits"]))
    print("-- resume labels (v1a: previous_record_set / equals_previous_record with a value on one side only; reported): %d" % len(acc.resume))
    for q in acc.resume:
        print("   %s: %s vs %s%s" % (q[0], q[1], q[2], (" (%s)" % q[3]) if len(q) > 3 else ""))
    out["resume_labels"] = acc.resume
    nt = {}
    for p, x, y in acc.ntraces:
        nt.setdefault(keyname(re.sub(r"^[0-9.]+", "", p)), []).append(p)
    print("-- Newton iterate traces differing (v1a: reported, the table's iterate traces%s): %d leaves at %d records" % (
        "; --strict-traces: none, counted as decisions" if STRICT_TRACES else "", len(acc.ntraces), len(set(p.split("/")[0] for p, _, _ in acc.ntraces))))
    for k, L in sorted(nt.items(), key=lambda kv: -len(kv[1]))[:12]:
        print("   %-70s %d" % (k[:70], len(L)))
    out["newton_traces_by_key"] = {k: len(L) for k, L in nt.items()}
    print("-- decisions differing: %d" % len(acc.decisions))
    byk = {}
    for p, kind, v in acc.decisions:
        byk.setdefault(keyname(p.split(" ", 1)[0][7:]) if False else keyname(re.sub(r"^[0-9.]+", "", p)), []).append((p, kind, v))
    for k, L in sorted(byk.items(), key=lambda kv: -len(kv[1]))[:40]:
        print("   %-70s %d records, e.g. %s %s %s" % (k[:70], len(L), L[0][0][:30], L[0][1], str(L[0][2])[:60]))
    out["decisions_by_key"] = {k: len(L) for k, L in byk.items()}
    print("-- counters differing: %d" % len(acc.counters))
    byk = {}
    for p, kind, v in acc.counters:
        byk.setdefault(keyname(re.sub(r"^[0-9.]+", "", p)), []).append((p, kind, v))
    for k, L in sorted(byk.items(), key=lambda kv: -len(kv[1]))[:30]:
        print("   %-70s %d records, e.g. %s %s" % (k[:70], len(L), L[0][0][:30], str(L[0][2])[:60]))
    out["counters_by_key"] = {k: len(L) for k, L in byk.items()}
    print("-- indicator bits differing: %d" % len(acc.bits))
    for p, x, y in acc.bits[:40]:
        print("   %s %s vs %s" % (p, x, y))
    print("-- other floats (top by relative difference)")
    for k, v in sorted(acc.other.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print("   %-80s rel %.2e abs %.2e (%.6e vs %.6e at %s)" % (k[:80], v[1], v[0], v[3], v[4], v[2][:12]))
    print("-- npz rows (max over records): key: max abs, max rel; masks: rows with any difference")
    for k, D in sorted(rows.items()):
        vals = list(D.values())
        if vals and isinstance(vals[0], tuple) and vals[0][0] == "mask":
            nd = sum(1 for v in vals if v[1] > 0); tot = sum(v[1] for v in vals)
            out["rows"][k] = {"mask_rows_differ": nd, "entries": tot}
            print("   %-28s masks: %d of %d rows differ (%d entries)" % (k, nd, len(vals), tot))
        elif vals and isinstance(vals[0], tuple) and vals[0][0] == "shape":
            out["rows"][k] = {"shape_mismatch": len(vals)}; print("   %-28s shape mismatch" % k)
        else:
            ma = max(v[0] for v in vals); mr = max(v[1] for v in vals); tm = max(D, key=lambda t: D[t][0])
            out["rows"][k] = {"max_abs": ma, "max_rel": mr, "at": tm, "per_record_abs": {("%.5f" % t): D[t][0] for t in D},
                              "per_record_rel": {("%.5f" % t): D[t][1] for t in D}}
            print("   %-28s abs %.3e rel %.3e (at %.5f)" % (k, ma, mr, tm))
    if a.slice_a and a.slice_b and os.path.exists(a.slice_a) and os.path.exists(a.slice_b):
        s = slice_cmp(a.slice_a, a.slice_b)
        out["slice"] = s
        print("-- slice: masks %s (%d channels), level %s; e %.2e, sea dens %.2e, R2 dens %.2e, dust f %.2e, dg nodes %s" % (
            "equal" if s["masks_equal"] else "DIFFER", s["channels"], s["level_equal"], s["e"], s["dens_sea"], s["dens_R2"], s["f"], ("%.2e" % s["nodes"]) if s["nodes"] is not None else "n/a"))
    out["iterations"] = {("%.5f" % t): (RA[t].get("iterations"), RB[t].get("iterations")) for t in ts}
    # v4: the record's mode inputs: each side's kept residual and converged flag, and the spectra mask rows differing per record
    kr = lambda r: (r.get("solve_v32") or {}).get("kept_residual") if isinstance(r.get("solve_v32"), dict) else None
    out["kept_residual"] = {("%.5f" % t): (kr(RA[t]), kr(RB[t])) for t in ts}
    out["converged"] = {("%.5f" % t): (RA[t].get("converged"), RB[t].get("converged")) for t in ts}
    out["nsites"] = (JA["args"].get("nsites"), JB["args"].get("nsites"))
    mpr = {("%.5f" % t): 0 for t in ts}
    for k, D in rows.items():
        for t, v in D.items():
            if isinstance(v, tuple) and v and v[0] == "mask":
                mpr["%.5f" % t] = mpr.get("%.5f" % t, 0) + v[1]
    out["mask_entries_per_record"] = mpr
    out["ntraces"] = len(acc.ntraces)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print("written %s" % a.out)
    return out, per, ts, acc


FIELD_KEYS = {"N", "v", "N_applied", "v_applied", "N_nodes", "v_nodes"}
LINES_A = {"E_over_E0": 3.0e-8, "E_rel": 3.0e-8, "roots": 3.4e-7, "depth": 4.6e-9, "rows": 1e-6, "spectra_rel": 1e-8, "nocc": 1e-8,
           "slice_dens": 1e-9}
LINES_B = {"E_over_E0": 3.0e-7, "roots": 3.4e-6, "depth": 4.6e-8, "fields": 2.5e-5}


def row_class(k):
    grp, key = k.split("/", 1)
    if grp == "fields" and key in FIELD_KEYS:
        return "fields"
    if grp == "spectra":
        if key.startswith("e_"):
            return "spectra_rel"
        if key.startswith("nocc"):
            return "nocc"
        return "spectra_other"
    return "rows"


def per_record_rows(out, ts):
    """-> {t: {class: value}} from out['rows']: max absolute for fields, rows and nocc, max relative for the e rows; masks go to out['_masks']."""
    R = {("%.5f" % t): {} for t in ts}
    for k, v in out["rows"].items():
        c = row_class(k)
        if "mask_rows_differ" in v:
            out.setdefault("_masks", {})[k] = v
            continue
        if "per_record_abs" not in v:
            continue
        src = v["per_record_rel"] if c == "spectra_rel" else v["per_record_abs"]
        for t, val in src.items():
            d = R.setdefault(t, {})
            if val >= d.get(c, 0.0):
                d[c] = val; d["_" + c + "_key"] = k
    return R


LINES_A2 = {"E_over_E0": 3.0e-8, "E_rel": 3.0e-8, "roots": 3.4e-7, "depth": 4.6e-9}          # 2(c): kept at every record
FIELD_LINES = {"fields": 1e-7, "rows": 1e-6, "spectra_rel": 1e-8, "nocc": 1e-8}               # 2(b); named by its own rule; v3: nocc on the spectra line (table 2.3, 1e-8 absolute)
C_LINES = {"E_over_E0": 3.0e-7, "roots": 3.4e-6, "depth": 4.6e-8, "fields": 2.5e-5}
FIELD_CLASSES = ("fields", "named", "rows", "spectra_rel", "nocc")   # v3: nocc with the spectra rows under 2(c)


def flip_records(counts):
    """counts: {t: (n_X, n_D, n_P or None)} -> (first flip clock or None, {t: True if at or after the first flip}).
    A stop-flip record is one where the available Newton iteration counts are not all equal (ruling v2 2(c))."""
    first = None
    for t in sorted(counts):
        v = [c for c in counts[t] if c is not None]
        if len(set(v)) > 1:
            first = t; break
    return first, {t: (first is not None and t >= first) for t in counts}


def three_way(x_vs_p, d_vs_p, line):
    """ruling v2 2(c)(2) for one class at one record: X's distance from P at most twice D's. Returns (ok, ratio). `line` is unused by
    the rule (2(c) has no below-the-line escape) and kept for the caller's printing."""
    if x_vs_p == 0.0:
        return True, 0.0
    if d_vs_p == 0.0:
        return False, float("inf")
    r = x_vs_p / d_vs_p
    return r <= 2.0, r


def recvals(out, per, ts):
    R = per_record_rows(out, ts)
    V = {}
    for t in ts:
        tk = "%.5f" % t; p = per[t]; r = R.get(tk, {})
        V[tk] = {"E_over_E0": p.get("E_over_E0", 0.0), "E_rel": p.get("E_rel", 0.0), "roots": p.get("roots", 0.0), "depth": p.get("depth", 0.0),
                 "R_sh": p.get("R_sh", 0.0), "named": p.get("named", 0.0), "named_excess": p.get("named_excess", 0.0),
                 "fields": r.get("fields", 0.0), "fields_key": r.get("_fields_key"), "rows": r.get("rows", 0.0), "spectra_rel": r.get("spectra_rel", 0.0),
                 "nocc": r.get("nocc", 0.0), "decisions": p["n_decisions"]}
    return V


def first_differ(V):
    """v3: the first record (sorted clock key) at which the two arms differ in the growth line's own quantity, the E/E0 difference
    (nonzero); None if none. (Read as: the quarters start at the first record whose E/E0 difference is not exactly 0, the record that made
    the box smoke's first-quarter median 0; other classes, e.g. rows at 1e-11, can differ there without moving E/E0.)"""
    for tk in sorted(V):
        if V[tk]["E_over_E0"] != 0.0:
            return tk
    return None


def masks_differ(out):
    return sum(v["mask_rows_differ"] for v in out.get("_masks", {}).values())


SIG = 4                     # v4: "kept residuals equal to the reported digits": four significant figures (ReviewA_s3b_v1 sec. 1: "kept residual 1.365e-7 to four figures")
TRIGGER = 1e-8              # erratum v3's trigger (ReviewA_v16e_v1 sec. 3), "the point above which the mode must be checked"
# v4: the same-iterate floor for D-P per size, measured (ReviewA_s3b_v1 sec. 1; floors extend per size by measurement only). The site-fields
# class, max absolute, compared at the floor's own stated precision. A size not listed has no floor: mode A cannot be read there.
FLOORS = {4096: ("3e-13", "4096: 3e-13 fields across cards"), 2048: ("2.5e-10", "2048: 2.5e-10 at a resume (5e-11 across devices)")}


def kfmt(x):
    return "-" if x is None else ("%%.%de" % (SIG - 1)) % x


def within_floor(v, floor_s):
    """v at or below the floor, read at the floor's own number of significant figures (3e-13: one; 2.5e-10: two)."""
    nsf = len(floor_s.split("e")[0].replace(".", "").lstrip("0")) or 1
    return float(("%%.%de" % (nsf - 1)) % v) <= float(floor_s)


def read_mode(cnt, kept, post, dp_fields, floor_s):
    """ReviewA_s3b_v1 sec. 1: the record's mode, measured. cnt, kept: the present arms' Newton counts and kept residuals (X, D, and P
    where P has the record). -> (mode, reason). Modes: C (a flip: counts differ, or a later record of a flipped segment not measured
    shared), A (a shared iterate: counts equal, kept residuals equal to four figures, D-P within the size's floor), B (same count, kept
    residuals different, above the trigger), T (no arm's kept residual above 1e-8 and no earlier flip: the table on X-D, as v2 (b))."""
    if len(set(cnt)) > 1:
        return "C", "counts differ"
    have = all(k is not None for k in kept)
    above = any(k is not None and k > TRIGGER for k in kept)
    same = have and len(set(kfmt(k) for k in kept)) == 1
    p_in = len(cnt) == 3
    fl_ok = (p_in and floor_s is not None and dp_fields is not None and within_floor(dp_fields, floor_s))
    shared = same and p_in and fl_ok
    why_not = ("kept residuals differ" if have and not same else ("no kept residual on an arm" if not have else
               ("no production record" if not p_in else ("no floor at this size" if floor_s is None else "D-P fields above the floor"))))
    if not post:
        if not above:
            return "T", "kept residuals <= 1e-8" if have else ("no kept residual on any arm: counts only" if not any(k is not None for k in kept) else "kept residuals present <= 1e-8 (one arm has none)")
        return ("A", "shared iterate") if shared else ("B", why_not)
    return ("A", "shared iterate after the flip") if shared else ("C", "after the first flip; " + why_not if why_not else "after the first flip")


def dec_eff(V, per, out, tk):
    """v4: decisions with rotated_blocks counted as a decision only where a set, mask or level label differs at the same record."""
    p = per.get(tk); n = V[tk]["decisions"]
    rot = (p or {}).get("rotated_blocks") or []
    if rot and (n > 0 or out.get("mask_entries_per_record", {}).get(tk, 0) > 0):
        return n + len(rot), len(rot)
    return n, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("D"); ap.add_argument("P")
    ap.add_argument("--tag", required=True); ap.add_argument("--after", type=float, default=None); ap.add_argument("--rs", type=float, default=None)
    ap.add_argument("--slice-x", default=None); ap.add_argument("--slice-d", default=None); ap.add_argument("--slice-p", default=None)
    ap.add_argument("--percall", default=None, help="2(c)(3): the stored per-call evidence (required where a record reads B or C; reported in A)")
    ap.add_argument("--names", default="X,D,P"); ap.add_argument("--out", default=None); ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-records", type=int, default=8, help="sec. 4 (A)'s 8 records for a build replay; an entry test passes its own count (2)")
    ap.add_argument("--floor", default=None, help="override the D-P same-iterate floor (a measured value, as a string, e.g. 3e-13)")
    a = ap.parse_args()
    nX, nD, nP = a.names.split(",")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    def ns(A, B, na, nb, sa, sb, o):
        return argparse.Namespace(A=A, B=B, tag=a.tag, after=a.after, rs=a.rs, out=(os.path.join(a.out, o) if a.out else None), slice_a=sa, slice_b=sb,
                                  nameA=na, nameB=nb, prod_json=None, prod_npz_dir=None, top=a.top)
    print("######## %s against %s" % (nX, nD))
    oXD, pXD, tXD, aXD = compare(ns(a.X, a.D, nX, nD, a.slice_x, a.slice_d, "v4_X_vs_D.json"))
    print("######## %s against %s" % (nX, nP))
    oXP, pXP, tXP, aXP = compare(ns(a.X, a.P, nX, nP, a.slice_x, a.slice_p, "v4_X_vs_P.json"))
    print("######## %s against %s" % (nD, nP))
    oDP, pDP, tDP, aDP = compare(ns(a.D, a.P, nD, nP, a.slice_d, a.slice_p, "v4_D_vs_P.json"))
    VXD, VXP, VDP = recvals(oXD, pXD, tXD), recvals(oXP, pXP, tXP), recvals(oDP, pDP, tDP)
    PXD = {("%.5f" % t): pXD[t] for t in tXD}; PXP = {("%.5f" % t): pXP[t] for t in tXP}; PDP = {("%.5f" % t): pDP[t] for t in tDP}
    itXD, itXP = oXD["iterations"], oXP["iterations"]
    kXD, kXP = oXD["kept_residual"], oXP["kept_residual"]
    counts = {tk: (itXD[tk][0], itXD[tk][1], itXP[tk][1] if tk in itXP else None) for tk in itXD}
    kept = {tk: (kXD[tk][0], kXD[tk][1], kXP[tk][1] if tk in kXP else None) for tk in kXD}
    first, post = flip_records(counts)
    ns_ = oXD["nsites"][0]
    if a.floor:
        floor_s, floor_txt = a.floor, "%s (given on the command line)" % a.floor
    else:
        floor_s, floor_txt = FLOORS.get(ns_, (None, "%s sites: no measured floor (mode A not readable)" % ns_))
    rows = []; fails = []; inconcl = []; modes = {}
    for tk in sorted(VXD):
        v = VXD[tk]; res = {}
        pin = tk in VXP and tk in VDP
        cnt = [c for c in counts[tk][:2]] + ([counts[tk][2]] if pin else [])
        kpt = [k for k in kept[tk][:2]] + ([kept[tk][2]] if pin else [])
        dp_f = VDP[tk]["fields"] if pin else None
        mode, why = read_mode(cnt, kpt, post[tk], dp_f, floor_s)
        modes[tk] = (mode, why)
        for c, lim in LINES_A2.items():
            res[c] = (v[c], lim, v[c] <= lim)
        res["R_sh"] = (v["R_sh"], 0.0, v["R_sh"] == 0.0)
        dx, rotx = dec_eff(VXD, PXD, oXD, tk)
        res["decisions"] = (dx, 0, dx == 0)
        if mode in ("T", "A"):
            for c, lim in FIELD_LINES.items():
                res[c] = (v[c], lim, v[c] <= lim)
            res["named"] = (v["named_excess"], 0.0, v["named_excess"] <= 0.0)
            if mode == "A":
                w = VXP[tk]
                for c, lim in LINES_A2.items():
                    res["X-P " + c] = (w[c], lim, w[c] <= lim)
                for c, lim in FIELD_LINES.items():
                    res["X-P " + c] = (w[c], lim, w[c] <= lim)
                res["X-P named"] = (w["named_excess"], 0.0, w["named_excess"] <= 0.0)
                dxp, _ = dec_eff(VXP, PXP, oXP, tk)
                res["X-P decisions"] = (dxp, 0, dxp == 0)
        else:
            if not pin:
                inconcl.append((tk, "no production record at a mode %s record" % mode))
            else:
                for arm, VV, PP, OO in (("X", VXP, PXP, oXP), ("D", VDP, PDP, oDP)):
                    for c, lim in C_LINES.items():
                        res["(1) %s-P %s" % (arm, c)] = (VV[tk][c], lim, VV[tk][c] <= lim)
                    dd, _ = dec_eff(VV, PP, OO, tk)
                    res["(1) %s-P decisions" % arm] = (dd, 0, dd == 0)
                for c in FIELD_CLASSES:
                    ok, r = three_way(VXP[tk][c], VDP[tk][c], FIELD_LINES.get(c))
                    res["(2) %s X-P/D-P" % c] = (r, 2.0, ok)
        for c, (val, lim, ok) in res.items():
            if not ok:
                fails.append((tk, c, val, lim))
        # beside each verdict: the named class against its own line (|d| / (1e-6 |x| + 1e-8); <= 1 is inside), X-D and X-P
        nm = {"X-D": (PXD[tk].get("named_ratio", 0.0), PXD[tk].get("named_rel", 0.0)),
              "X-P": (PXP[tk].get("named_ratio", 0.0), PXP[tk].get("named_rel", 0.0)) if tk in PXP else (None, None)}
        rows.append((tk, counts[tk], kept[tk], mode, why, dp_f, res, nm, rotx))
    mX, mXP, mDP = masks_differ(oXD), masks_differ(oXP), masks_differ(oDP)
    if mX or mXP or mDP:
        fails.append(("all", "masks (spectra rows) differ X-D/X-P/D-P", (mX, mXP, mDP), 0))
    sl = {}
    last = sorted(VXD)[-1]
    if "slice" in oXD:
        s = oXD["slice"]
        sl["X-D masks, f"] = bool(s["masks_equal"] and s["f"] == 0.0 and s["level_equal"] in (None, True))
        if not sl["X-D masks, f"]:
            fails.append(("end", "slice masks/f/level", 1, 0))
        if modes[last][0] in ("T", "A"):
            ok = max(s["dens_sea"], s["dens_R2"]) <= 1e-9 and (s["nodes"] is None or s["nodes"] <= 1e-7)
            sl["X-D densities 1e-9, nodes 1e-7 (mode %s)" % modes[last][0]] = (max(s["dens_sea"], s["dens_R2"]), ok)
            if not ok:
                fails.append(("end", "slice densities/nodes", max(s["dens_sea"], s["dens_R2"]), 1e-9))
            if modes[last][0] == "A" and "slice" in oXP:
                q = oXP["slice"]; ok = max(q["dens_sea"], q["dens_R2"]) <= 1e-9 and (q["nodes"] is None or q["nodes"] <= 1e-7)
                sl["X-P densities 1e-9, nodes 1e-7 (mode A)"] = (max(q["dens_sea"], q["dens_R2"]), ok)
                if not ok:
                    fails.append(("end", "slice densities/nodes X-P", max(q["dens_sea"], q["dens_R2"]), 1e-9))
        elif "slice" in oXP and "slice" in oDP:
            dx = max(oXP["slice"]["dens_sea"], oXP["slice"]["dens_R2"]); dd = max(oDP["slice"]["dens_sea"], oDP["slice"]["dens_R2"])
            ok, r = three_way(dx, dd, 1e-9)
            sl["(2) densities X-P/D-P (mode %s)" % modes[last][0]] = (r, ok)
            if not ok:
                fails.append(("end", "slice densities (2)", r, 2.0))
        else:
            sl["densities (mode %s)" % modes[last][0]] = ("not readable: no production slice at the end clock; X-D %.2e (reported)" % max(s["dens_sea"], s["dens_R2"]), None)
    else:
        sl["slice"] = "not given (reported as not read)"
    gstart = first_differ(VXD)
    gks = [tk for tk in sorted(VXD) if gstart is not None and tk >= gstart]
    QM = {}
    for nm_, VV in (("X-D", VXD), ("X-P", VXP), ("D-P", VDP)):
        ks = [tk for tk in gks if tk in VV]
        QM[nm_] = [float(np.median([VV[tk]["E_over_E0"] for tk in part])) if len(part) else None for part in np.array_split(np.array(ks, dtype=object), 4)] if ks else [None] * 4
    m1, m4 = QM["X-D"][0], QM["X-D"][3]
    growth_ok = True if (m1 is None or m4 is None) else (m4 <= 10 * m1 or m4 <= 1e-12)
    if not growth_ok:
        fails.append(("all", "monotone growth of E/E0", m4, 10 * m1))
    nBC = [tk for tk in modes if modes[tk][0] in ("B", "C")]
    p3 = a.percall
    if nBC and not p3:
        inconcl.append((nBC[0], "2(c)(3) per-call evidence not supplied (--percall) at a mode B/C record"))
    nrec = len(VXD)
    if nrec < a.min_records:
        fails.append(("all", "fewer than %d records" % a.min_records, nrec, a.min_records))
    verdict = "FAIL" if fails else ("INCONCLUSIVE" if inconcl else "PASS")
    mc = {m: sum(1 for tk in modes if modes[tk][0] == m) for m in ("A", "B", "C", "T")}
    nmXD = max([r[7]["X-D"][0] or 0.0 for r in rows] or [0.0]); nmXP = max([r[7]["X-P"][0] or 0.0 for r in rows] or [0.0])
    print("######## GATE (tolcmp_v4: the mode read of ReviewA_s3b_v1 sec. 1 over ruling v2 2(a)-(c), erratum v3, the RD-1c erratum): %s" % verdict)
    print("   modes per record: A %d, B %d, C %d, T %d; first count flip: %s; D-P floor %s" % (mc["A"], mc["B"], mc["C"], mc["T"], first, floor_txt))
    print("   named class against its own line (1e-6 rel + 1e-8; ratio <= 1 inside): max X-D %.3g, X-P %.3g (%s)" % (
        nmXD, nmXP, "inside" if max(nmXD, nmXP) <= 1.0 else "OUTSIDE (gating in T/A records, reported in B/C)"))
    fq = lambda L: ", ".join("n/a" if x is None else "%.2e" % x for x in L)
    print("   records %d (min %d); growth over the %d records from %s: first-quarter median %s, last %s (%s)" % (
        nrec, a.min_records, len(gks), gstart, "n/a" if m1 is None else "%.2e" % m1, "n/a" if m4 is None else "%.2e" % m4, "ok" if growth_ok else "GROWTH"))
    print("   quarter medians E/E0 (q1, q2, q3, q4): X-D %s; X-P %s; D-P %s" % (fq(QM["X-D"]), fq(QM["X-P"]), fq(QM["D-P"])))
    print("   masks rows differing X-D %d, X-P %d, D-P %d; Newton trace leaves X-D %d, X-P %d, D-P %d (reported); resume labels X-D %d, X-P %d, D-P %d" % (
        mX, mXP, mDP, oXD["ntraces"], oXP["ntraces"], oDP["ntraces"], len(aXD.resume), len(aXP.resume), len(aDP.resume)))
    for tk, c, k, mode, why, dp_f, res, nm, rotx in rows:
        print("   %s mode %s (%s); counts %s/%s/%s; kept %s/%s/%s; D-P fields %s; named X-D %s (rel %s), X-P %s (rel %s) [line 1e-6 rel + 1e-8]%s" % (
            tk, mode, why, c[0], c[1], c[2], kfmt(k[0]), kfmt(k[1]), kfmt(k[2]), "-" if dp_f is None else "%.2e" % dp_f,
            "%.2g" % nm["X-D"][0], "%.2e" % nm["X-D"][1], "-" if nm["X-P"][0] is None else "%.2g" % nm["X-P"][0], "-" if nm["X-P"][1] is None else "%.2e" % nm["X-P"][1],
            ("; rotated_blocks counted as %d decisions (a set/mask/level label differs with it)" % rotx) if rotx else ""))
        print("        %s" % "  ".join("%s %s%s" % (q, ("%.2e" % w[0]) if isinstance(w[0], float) else w[0], "" if w[2] else " FAIL") for q, w in res.items()))
    rbs = [(tk, len(PXD[tk].get("rotated_blocks") or [])) for tk in sorted(PXD) if PXD[tk].get("rotated_blocks")]
    print("   rotated_blocks differing X-D (a counter; reported): %s" % (", ".join("%s x%d" % q for q in rbs) or "none"))
    for k_, v_ in sl.items():
        print("   slice %s: %s" % (k_, v_))
    print("   2(c)(3) per-call evidence: %s%s" % (p3 or ("n/a (no mode B/C record)" if not nBC else "OPEN"), " (reported; mode A/T records)" if p3 and not nBC else ""))
    for f in fails:
        print("   FAIL %s" % (f,))
    for f in inconcl:
        print("   INCONCLUSIVE %s" % (f,))
    G = {"verdict": verdict, "comparator": "tolcmp_v4", "modes": {tk: list(v) for tk, v in modes.items()}, "mode_counts": mc, "first_flip": first,
         "floor": [floor_s, floor_txt], "counts": counts, "kept_residual": kept, "records": nrec, "min_records": a.min_records, "fails": fails,
         "inconclusive": inconcl, "percall": p3, "named_max_ratio": {"X-D": nmXD, "X-P": nmXP},
         "per_record": {tk: {"counts": c, "kept": k, "mode": mode, "why": why, "dp_fields": dp_f, "lines": res, "named": nm}
                        for tk, c, k, mode, why, dp_f, res, nm, rotx in rows},
         "slice": sl, "growth": [m1, m4], "growth_start": gstart, "quarter_medians": QM, "masks": [mX, mXP, mDP]}
    if a.out:
        json.dump(G, open(os.path.join(a.out, "v4_gate.json"), "w"), indent=1, default=str)
        print("written %s" % os.path.join(a.out, "v4_gate.json"))


if __name__ == "__main__":
    main()
