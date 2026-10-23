#!/usr/bin/env python
"""C_readout_v3b.py (the busi lane, Computation C; version 3 with the slice adapter's ruling applied, C_readout_v3b_ruling.md: the stored orbitals read
as columns with the engine's one-body operator from the adapter C_readout_v3_hbuilder, the heavy field as its ten channels summed with the weights
16 kappa, the records' own split of work and reference, the engine's cut-index convention at the collars, the end clock T_over_rs an expected
argument difference, the engine's key names, one slice per clock across the directories, the s11 npz's r_nodes rows of a continued run): versions 1
and 2 beside this file (C_readout_v1.py, C_readout_v2.py) are imported and never
edited; new, the readout of C_sigma4_coupling_effect_v1.2 section 6 on the continued runs: the r_s/a 32, sigma/a 4 member (T/r_s 0 to 80,
continued from 40) and its coupling-off control (0 to 50, continued from 40, plan #162), read as C minus control at matched clocks to the
control's last clock and the member alone beyond it. CPU (at most 8 threads), no figure; the slice part costs 3 dense products per field
and slice clock.

Order of operations (nothing later is written when an earlier step stops; the preconditions print before any column):
  P1  the join: the continuation's record at 40 against the delivered run's, every leaf of the record compared (bitwise, or every difference
      printed beside the gate's pass levels R_t 3.4e-6, E/E0 3.0e-7, min(N - |v|) 4.6e-8, which are pass levels for the identical command
      line and not errors), and every other record the two jsons share (the 2-record resume gate recomputed where the continuation json
      carries records before 40); P2 the argument lists (V1.args_diff: delivered against continued, member against control); P3 the engine
      version (the json's script key and any version key); P4 per record over the continuation the iteration count, the kept residual and
      the fallbacks against the delivered maxima (at most 4, 4.07e-09, 0 for the control to 40; 1.22e-08 the member's); P5 the control's
      last clock, the matched window's end; P6 the chain-end closure over 0 to 40 from the delivered member's own records: the per-record
      light budget entries evolution_exact, field_work, reference_reset (found by name at any depth of the record, or named by
      --budget-keys) summed to readout 2's triple (-110196, 0.00753, +110196) and to dH_in_light 0.003977 at the printed digits, the
      chain-end cut currents' integrals against 3.228e-06 and 5.199e-08, E(0) - E(40) against 0.563327; P7 the slices: the listing per run,
      the schema of the first slice, the operators' reading checked at the join clock against the records (H_in and H_in_light at the
      chain-end cut, E_inv per field, the chain-end cut current, the exterior entry), the conventions read off there (the inside rule, the
      light weighting of H_in_light, the current's form and sign, the index layout), nothing substituted, and the sweep of the chain-end
      cut from site 235 to 2038 estimated from the slice at 40 against evolution_exact minus the integrated current.
  L   the collar ledger (section 6 (a) to (g)) at 16 and 24 r_s (R = 512 a and 768 a), the chain-end cut (site n_sites - 10, 2038 of 2048)
      and optionally 8 r_s, per run, slice clock and field (heavy; light kappa 1, 2, 3 with the weights 16 kappa and the sums over 1, 2 and
      1, 2, 3 beside): (a) H_in = sum over the sites inside of Re[h (C - P_ref)]_ii; (b) H_out the same beyond the cut, signed, per channel
      unweighted; (c) I_E the cut current from the same operators, inward-positive (I_E = dH_in/dt at fixed h: the energy entering the
      inside), and its time integral from 40 by the trapezoid over the slice clocks, the step printed as its resolution; (d) field_work
      = the inside sum (at the step's start cut) of Re[(h' - h)(C' - P')]_ii and reference_reset = -(the inside sum of Re[h (P' - P)]_ii), the
      records' own split (engine_twofield_v3.py's M1 balance: field_work = (X1 - X2) - (Y1 - Y2), reference_reset = -(Y2 - X2_prev); v3b), per step
      and cumulative; (e) the cut's site index, the sites inside, the identity of the cut between the runs and the content of the sites
      that change sides (zero where the cut is fixed: at the collars and at the chain end over the continuation); (f) the closure
      H_in(T) - H_in(40) - (the integral of (c) + (d) + (e)), which equals by construction the evolution term (the inside sum of
      Re[h (C' - C)]) minus the current's integral, since evolution + work + reference + membership telescopes exactly to dH_in; (g) the
      collar charge W/2g = M at r_c from the s11 npz, E beside.
      The identification of (c)'s integral with the budget's evolution_exact holds at a fixed cut; at the chain end over 0 to 40 the cut
      moved from site 235 to 2038 and evolution_exact carries the bare content of the swept sites, which reference_reset carries back and
      (e) carries here; over the continuation the chain-end cut is fixed and the records' own evolution_exact, field_work and
      reference_reset over 40 to T are printed beside the ledger's integral, work and reference at the chain end.
  R   the per-record columns continued from version 2's tables at 40, 44, 48, 50 (the matched window's end) and 56, 64, 72, 80.
--selftest runs this version's checks on manufactured operators and records (no real file) and exits 0 when every check passes.
Exit codes: 0 everything ran; 3 the ledger was not written (a precondition failed, or the slices do not reproduce the records at the join:
the md and the json carry the stop and every precondition); 2 the readout did not run (a file missing, the records not joinable)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import copy
import datetime
import glob
import importlib
import json
import math
import re
import sys
import tempfile
import time
import traceback

import numpy as np
import scipy.sparse as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import C_readout_v1 as V1  # noqa: E402  version 1, imported and never edited
import C_readout_v2 as V2  # noqa: E402  version 2, imported and never edited
from C_readout_v1 import (ReadoutError, MissingKey, Inconsistent, fm, mdtab, md5_of, load_json, _jsonable, LABEL, align,  # noqa: E402
                          find_clock, build_table, Identities)

VERSION = "C_readout_v3b"
SPEC = "C_sigma4_coupling_effect_v1.2 section 6 (the continuation's 16 and 24 r_s collar ledger)"
GATE = (("R_t", 3.4e-6), ("E_over_E0", 3.0e-7), ("min_N_minus_absv", 4.6e-8))
BUDGET_NAMES = ("evolution_exact", "field_work", "reference_reset")
READOUT2 = {"evolution_exact": (-110196.0, 0.5), "field_work": (0.00753, 5.0e-6), "reference_reset": (110196.0, 0.5), "dH_in_light": (0.003977, 5.0e-7)}
READOUT2_INTEGRALS = {"M1_I_E": (3.228e-06, 5.0e-10), "light.I_E_light_sourcing": (5.199e-08, 5.0e-12)}
MD_AT_40 = {"C.E": 59.225661763, "K.E": 59.227016525, "C.light.E_inv_heavy": 35.4639183, "K.light.E_inv_heavy": 35.4779288,
            "C.light.E_inv_light": 0.00397695, "K.light.E_inv_light": 0.00398995}
MD_E0, MD_FALL, MD_FORMATION = 59.788989, 0.563327, 15.8125
MD_TRUNC = {"formation": (0.163692, 0.978868, 0.380043), "40": (0.00157312, 0.975971, 0.0164271)}
DELIVERED_SOLVER = {"control": {"iterations_max": 4, "kept_residual_max": 4.07e-09, "fallbacks": 0}, "member": {"kept_residual_max": 1.22e-08}}
COLLARS = (("r16", 16.0), ("r24", 24.0))
COLLAR_8 = ("r8", 8.0)
CHAIN_END = "end"
CUT_LABEL = {"r16": "16 r_s (R = 512 a)", "r24": "24 r_s (R = 768 a)", "r8": "8 r_s (R = 256 a)", "end": "the chain-end cut (site n_sites - 10)"}
TABLE_CLOCKS = (40.0, 44.0, 48.0, 50.0, 56.0, 64.0, 72.0, 80.0)
DEFAULT_KEYS = {"light": {"h": "h_L{k}", "C": "C_L{k}", "P": "P_L{k}", "orbitals": "sea_L{k}", "P_orbitals": "R2_L{k}", "occ": "occ_L{k}", "dust": "dust_L{k}", "dust_occ": "f_L{k}",
                          "eig": "e_L{k}", "mask": "mask_L{k}"},
                "heavy": {"h": "h_{k}", "C": "C_{k}", "P": "P_{k}", "orbitals": "sea_{k}", "P_orbitals": "R2_{k}", "occ": "occ_{k}", "dust": "dust_{k}", "dust_occ": "f_{k}",
                          "eig": "e_{k}", "mask": "mask_{k}"}}  # the engine's slice keys (engine_twofield_v3.py run(): sea_<k>, dust_<k>, f_<k>, R2_<k>, e_<k>, mask_<k>; light_slice_arrays_v36: sea_L<k>, R2_L<k>, ...); a dense h, C, P_ref is read under h_<k>, C_<k>, P_<k> where stored
EPS = float(np.finfo(float).eps)
LAYOUT_RULE_PREFERENCE = (("site_major", "<"), ("identity", "<"), ("component_major", "<"), ("site_major", "<="), ("identity", "<="), ("component_major", "<="))  # the order taken among (layout, inside rule) candidates the record does not tell apart: the engine's own convention first (the inside is the flattened index range [0, 2 cut): engine_collapse_v36.inside_energy_sp with Hr = H[:2 cut, :])
TOL_EXTERIOR = 1e-2  # 5d's exterior entries (the site-density route sum over R > r_c of N eps + v J) against (b) (the operator trace beyond the cut): equal up to the cross-cut term (1/2) Re sum_{i out, j in} [h0_ij (N_j - N_i) + Q_ij (v_j - v_i)] X_ji, so agreement to that term's size and not to roundoff is the expected shape
CLOCK_KEYS = ("T_over_rs", "T")
R_KEYS = ("R", "r", "r_sites")
CLOCK_REGEX = r"(?:T|Trs|clock|slice)[_=-]?(\d+(?:\.\d+)?)(?!\d|\.\d)"  # the number is complete when not followed by a digit or by a further decimal part (the "." of ".npz" is neither)
VERSION_NAMES = {"version", "engine_version", "engine", "git", "git_commit", "commit", "code_version", "engine_file", "script_md5"}
EXPECTED_CONT_ARG = re.compile(r"(^|_)[tT]_?(end|max|final|stop)($|_)|^(until|stop_at|n_?records|nrec|records|t_?total|T_over_rs)$")  # T_over_rs: the engine's --T-over-rs, the run's length (nsteps = round(T_over_rs rs / tau)), expected to differ between a delivered run and its continuation and between the two continuations (the control's 50 is plan #162's limit)
EXT_REGEX = re.compile(r"\.beyond\.24\.(weighted_)?energy_density_route$")  # 5d's exterior entries at 24 r_s per light channel (light_v36.channels.<k>.beyond.24.*; the dust's s11.ball_beyond_24 is the dust alone and is not (b))
TOL_CURRENT = 1e-3
CURRENT_PREFERENCE = (("I_C", 1), ("I_C", -1), ("I_X", 1), ("I_X", -1))  # the order taken when the record does not tell the current's candidates apart (every candidate within TOL_CURRENT reproduces)

READING = [
    ("1. What the ledger is.",
     "Per run, cut r_c (16 and 24 r_s, R = 512 a and 768 a on r_s/a 32; the chain-end cut at site n_sites - 10 for continuity with readout 2; "
     "8 r_s optional), slice clock and field (the heavy field; the light channels kappa 1, 2 and 3, the guard propagated unfed, with the "
     "weights 16 kappa of the md's E_inv identities and the sums over 1, 2 and 1, 2, 3 beside), the columns are the stored operators' own "
     "numbers: (a) H_in, the energy inside the cut with the continued reference, the sum over the sites inside of Re[h (C - P_ref)]; (b) H_out, "
     "the same beyond the cut, signed and unweighted per channel as 5d's exterior entries are; (c) I_E, the cut current, inward-positive "
     "(energy entering the inside), and its time integral from 40 at the slices' resolution; (d) field_work and reference_reset; (e) the cut's "
     "site index, the sites inside, the cut's identity between the two runs and the content of the sites that change sides; (f) the closure; "
     "(g) W/2g at r_c with E. The conventions (the inside rule, the light weighting of H_in_light, the current's form and sign, the index "
     "layout) are read off at the join clock against the records' own H_in, H_in_light, E_inv and chain-end currents and printed with their "
     "residuals; where no convention reproduces a record quantity the ledger is not written and nothing is substituted."),
    ("2. The closure and the identification.",
     "(f) = H_in(T) - H_in(40) - (the integral of (c) + the two terms of (d) + the term of (e)) is a bookkeeping column on the stored history, "
     "not a flux statement: evolution (the inside sum of Re[h (C' - C)] over a step) + work + reference + membership telescopes exactly to "
     "dH_in, so (f) equals the evolution term minus the current's integral and measures how far the trapezoid of the slice-clock currents stands "
     "from the exact evolution at the slices' resolution. The identification of (c)'s integral with the budget's evolution_exact holds at a "
     "fixed cut: at the chain end over 0 to 40 the cut moved from site 235 to 2038 and readout 2's evolution_exact (-110196) differs from the "
     "integrated current (5.199e-08) by the bare content of the swept sites, which reference_reset (+110196) carries back and (e) carries in "
     "this ledger; this readout reproduces the triple from the records before any collar term is written and estimates the swept content from "
     "the slice at 40. At the collars and at the chain end over the continuation the cut is fixed, (e) is zero, and the records' own "
     "evolution_exact, field_work and reference_reset over 40 to T are printed beside the ledger's integral, work and reference at the chain end."),
    ("3. The matched differences.",
     "d = C - control at the same clock on every column to the control's last clock (50 by plan #162), read as section 1 of the note reads the "
     "chain-end quantities: matched-implementation, finite-channel record differences, the gate's tolerances as pass levels for the identical "
     "command line, no tolerance stated for the collar quantities, the kept residuals as the stop rule's scales, nothing converted into an error. "
     "From 50 to 80 the member's ledger is its own, the control's absence the plan's limit and not a finding. By reorientation 4's estimate the "
     "formation's light reaches 16 r_s from about T/r_s 39 and 24 r_s from about 49: the matched window covers the estimated 16 r_s arrival "
     "throughout and the estimated 24 r_s arrival only in its last 1 r_s (16 records); the 24 r_s arrival from 50 to 80 has no control."),
    ("4. What no column is.",
     "No column is a global energy fraction, a radiated fraction, a flux identity, an emitted-energy bound, a feedback bound or an entropy; a term "
     "of (c) written as energy that crossed the collar is the stored current's integral with its resolution and the truncation at kappa 2 (the "
     "guard over the sourcing channels' current, 0.976 at the chain end at 40, printed at the collars beside), and only after (f) has been read "
     "at the chain end over 0 to 40 and at the collars over the continuation. delta R_L against Delta R_t is formed only at records where both "
     "runs carry a dense root; no record from 19.375 to 40 did, and a root's reappearance from 40 on is a fact of the records printed as such."),
]


# ---------------------------------------------------------------------------------------------------------------------- helpers
class _S(object):
    """A Series-shaped holder (version 2's functions read .d)."""

    def __init__(self, d):
        self.d = d


def num(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    return None


def rel_res(a, b, floor=1e-300):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), floor)


def find_paths(obj, names, path=(), out=None):
    """Every (path, value) whose last key is in names, at any depth (dict keys; list indices in the path)."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = path + (str(k),)
            if str(k) in names:
                out.append((p, v))
            find_paths(v, names, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_paths(v, names, path + (i,), out)
    return out


def all_leaves(obj, path=(), out=None):
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            all_leaves(v, path + (str(k),), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            all_leaves(v, path + (i,), out)
    else:
        out.append((path, obj))
    return out


def get_path(obj, path):
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            key = p if p in cur else str(p)
            if key not in cur:
                return None
            cur = cur[key]
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
    return cur


def dotted(path):
    return ".".join(str(p) for p in path)


def undot(s):
    return tuple(s.split("."))


def leaves_differ(a, b, path=(), out=None):
    """The leaves where two nested records differ (numbers compared exactly, nan equal to nan)."""
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((dotted(path + (k,)), a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                leaves_differ(a[k], b[k], path + (k,), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((dotted(path), "list[%d]" % len(a), "list[%d]" % len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                leaves_differ(x, y, path + (i,), out)
    else:
        xa, xb = num(a), num(b)
        if xa is not None and xb is not None:
            if xa != xb:
                out.append((dotted(path), a, b))
        elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            out.append((dotted(path), a, b))
    return out


def parse_jsonish(s):
    if s is None or isinstance(s, (dict, list)):
        return s
    if s.startswith("@"):
        with open(s[1:]) as f:
            return json.load(f)
    return json.loads(s)


def load_builder(spec, engine_dir, args=None, args_control=None):
    """The builder f(npz, field, kappa) -> h (dense, or a scipy sparse matrix: the engine's host CSR from C_readout_v3_hbuilder:h_of); an import failure of the
    adapter (the engine's modules, torch) is a stop, not a traceback; the module's configure(args, args_control) is called with the continued runs' args."""
    if not spec:
        return None
    if engine_dir and engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    mod, _, fn = spec.partition(":")
    if not fn:
        raise ReadoutError("--h-builder expects module:function, got %r" % spec)
    try:
        M = importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        raise ReadoutError("--h-builder: importing %s failed: %s: %s" % (mod, type(e).__name__, e))
    if not hasattr(M, fn):
        raise ReadoutError("--h-builder: module %s has no attribute %s" % (mod, fn))
    if args is not None and hasattr(M, "configure"):
        try:
            M.configure(args, args_control)
        except Exception as e:  # noqa: BLE001
            raise ReadoutError("--h-builder: %s.configure failed: %s: %s" % (mod, type(e).__name__, e))
    return getattr(M, fn)


# ------------------------------------------------------------------------------------------------------------- the records, joined
def load_run(path, where):
    run = load_json(path)
    S, m = V2.extract_min(run, where)
    top = {k: v for k, v in run.items() if k != "records"}
    vers = {dotted(p): v for p, v in find_paths(top, VERSION_NAMES) if not isinstance(v, (dict, list))}
    return {"S": S.d, "m": m, "records": run["records"], "args": run.get("args") or {}, "script": run.get("script"), "head": run.get("head") or {},
            "version_keys": vers, "path": os.path.abspath(path), "where": where}


def join_runs(D, C, label):
    """The delivered run's records followed by the continuation's records after the delivered last clock; the shared records compared leaf by leaf."""
    Sd, Sc = D["S"], C["S"]
    Td, Tc = Sd["T_over_rs"], Sc["T_over_rs"]
    nd, nc = len(Td), len(Tc)
    iK = align(Td, Tc)
    overlap = [(k, j) for k, j in enumerate(iK) if j is not None]
    used = set(j for _, j in overlap)
    T_last = Td[-1]
    appended = [j for j in range(nc) if Tc[j] > T_last + 1e-9]
    before = [Tc[j] for j in range(nc) if Tc[j] <= T_last + 1e-9 and j not in used]
    J, missing = {}, []
    for name, col in Sd.items():
        cc = Sc.get(name)
        if cc is None:
            missing.append(name)
            J[name] = list(col) + [None] * len(appended)
        else:
            J[name] = list(col) + [cc[j] for j in appended]
    recs = list(D["records"]) + [C["records"][j] for j in appended]
    dT = D["m"]["tau"] / D["m"]["rs"]
    TJ = J["T_over_rs"]
    gaps = [(TJ[k], TJ[k + 1]) for k in range(len(TJ) - 1) if abs((TJ[k + 1] - TJ[k]) - dT) > 1e-6 * dT]
    per = []
    for k, j in overlap:
        diffs = leaves_differ(D["records"][k], C["records"][j])
        gate = {}
        for key, tol in GATE:
            xd, xc = num(Sd[key][k]), num(Sc[key][j])
            d = (xc - xd) if (xd is not None and xc is not None) else None
            gate[key] = {"delivered": xd, "continued": xc, "d": d, "times_tolerance": (abs(d) / tol if d is not None else None), "tolerance": tol}
        per.append({"k_delivered": k, "j_continued": j, "T_over_rs": Td[k], "leaves_differing": len(diffs), "bitwise": len(diffs) == 0,
                    "differences": [{"path": p, "delivered": a, "continued": b} for p, a, b in diffs[:80]], "gate": gate})
    join = [p for p in per if abs(p["T_over_rs"] - T_last) < 1e-9]
    rep = {"label": label, "delivered_records": nd, "continued_records": nc, "continued_first_T_over_rs": Tc[0], "continued_last_T_over_rs": Tc[-1],
           "delivered_last_T_over_rs": T_last, "overlap_records": len(overlap), "overlap_bitwise": sum(1 for p in per if p["bitwise"]),
           "join_record": (join[0] if join else None), "resume_gate_records": [p for p in per if p["T_over_rs"] < T_last - 1e-9],
           "appended_records": len(appended), "continued_before_join_unmatched_clocks": before, "gaps": gaps,
           "series_missing_in_continuation": missing, "joined_records": len(TJ), "joined_last_T_over_rs": TJ[-1], "record_spacing": dT}
    m = dict(D["m"])
    m["n"] = len(TJ)
    return {"S": J, "m": m, "records": recs, "report": rep, "n_delivered": nd, "label": label, "args_delivered": D["args"], "args_continued": C["args"],
            "script_delivered": D["script"], "script_continued": C["script"], "version_keys": {"delivered": D["version_keys"], "continued": C["version_keys"]}}


def solver_stats(J, chans, k_from):
    S = J["S"]
    n = len(S["T_over_rs"])
    its = [(k, num(S["iterations"][k])) for k in range(k_from, n) if num(S["iterations"][k]) is not None]
    kept = [(k, num(S["solve.kept_residual"][k])) for k in range(k_from, n) if num(S["solve.kept_residual"][k]) is not None]
    fb = sum(1 for c in chans for k in range(k_from, n) if S.get("L%s.fallback" % c, [None] * n)[k])
    kk = max(kept, key=lambda t: abs(t[1])) if kept else None
    vals = [v for _, v in its]
    return {"records": n - k_from, "first_T_over_rs": (S["T_over_rs"][k_from] if k_from < n else None), "last_T_over_rs": (S["T_over_rs"][-1] if n else None),
            "iterations_max": (max(vals) if vals else None), "iterations_histogram": ({str(int(v)): vals.count(v) for v in sorted(set(vals))} if vals else {}),
            "kept_residual_max": (kk[1] if kk else None), "T_at_kept_residual_max": (S["T_over_rs"][kk[0]] if kk else None), "fallbacks": fb,
            "converged_false": sum(1 for k in range(k_from, n) if S["converged"][k] is False), "converged_absent": sum(1 for k in range(k_from, n) if S["converged"][k] is None),
            "certified_false": sum(1 for k in range(k_from, n) if not S["light.certified_on_record"][k]), "iterations_absent": sum(1 for k in range(k_from, n) if num(S["iterations"][k]) is None)}


# ------------------------------------------------------------------------------------------------------ P6: the budget over 0 to 40
def budget_closure(recs, S, targets, keys_override=None, k0=None, k1=None, label=""):
    """The light budget's three per-record entries, found by name at any depth, summed (per step) or differenced (cumulative) over the window and
    compared with the printed targets; the mode is read off as the one under which every name reproduces its target."""
    n = len(recs)
    k0 = 0 if k0 is None else k0
    k1 = n - 1 if k1 is None else k1
    out = {"label": label, "window_T_over_rs": [S["T_over_rs"][k0], S["T_over_rs"][k1]], "records": k1 - k0 + 1, "paths_found": {}, "candidates": [],
           "chosen": {}, "mode": None, "reproduced": False, "notes": [], "light_v36_keys_first_record": None, "targets": {k: v[0] for k, v in targets.items()}}
    lv = recs[k0].get("light_v36") if isinstance(recs[k0], dict) else None
    out["light_v36_keys_first_record"] = sorted(str(k) for k in lv.keys()) if isinstance(lv, dict) else None
    found = {nm: set() for nm in BUDGET_NAMES}
    if keys_override:
        for nm in BUDGET_NAMES:
            if nm in keys_override:
                found[nm].add(undot(str(keys_override[nm])))
    else:
        for r in (recs[k0], recs[k1], recs[(k0 + k1) // 2]):
            for p, v in find_paths(r, set(BUDGET_NAMES)):
                found[p[-1]].add(p)
    series = {}
    for nm in BUDGET_NAMES:
        out["paths_found"][nm] = sorted(dotted(p) for p in found[nm])
        for p in sorted(found[nm], key=dotted):
            col = [get_path(recs[k], p) for k in range(k0, k1 + 1)]
            if any(isinstance(x, list) for x in col):
                L = max(len(x) for x in col if isinstance(x, list))
                for i in range(L):
                    series[(nm, dotted(p) + ".%d" % i)] = [num(x[i]) if (isinstance(x, list) and i < len(x)) else None for x in col]
            else:
                series[(nm, dotted(p))] = [num(x) for x in col]
    for (nm, pth), col in series.items():
        vals = [x for x in col if x is not None]
        if not vals:
            continue
        tgt, tol = targets.get(nm, (None, None))
        s_sum, s_last, s_lmf = float(sum(vals)), float(vals[-1]), float(vals[-1] - vals[0])
        tries = (("per_step_sum", s_sum), ("cumulative_last", s_last), ("cumulative_last_minus_first", s_lmf))
        out["candidates"].append({"name": nm, "path": pth, "available": len(vals), "first": float(vals[0]), "sum": s_sum, "last": s_last, "last_minus_first": s_lmf,
                                  "target": tgt, "tolerance": tol, "matches": [md for md, val in tries if tgt is not None and abs(val - tgt) <= tol]})
    for mode in ("per_step_sum", "cumulative_last", "cumulative_last_minus_first"):
        pick = {}
        for nm in BUDGET_NAMES:
            c = [r for r in out["candidates"] if r["name"] == nm and mode in r["matches"]]
            if c:
                pick[nm] = c[0]
        if len(pick) == len(BUDGET_NAMES):
            out["mode"] = mode
            out["chosen"] = {nm: {"path": r["path"], "value": {"per_step_sum": r["sum"], "cumulative_last": r["last"], "cumulative_last_minus_first": r["last_minus_first"]}[mode],
                                  "first_entry": r["first"]} for nm, r in pick.items()}
            out["reproduced"] = True
            break
    h0, h1 = num(S["light.H_in_light"][k0]), num(S["light.H_in_light"][k1])
    out["H_in_light_start"], out["H_in_light_end"] = h0, h1
    out["dH_in_light"] = (h1 - h0) if (h0 is not None and h1 is not None) else None
    tgt, tol = targets.get("dH_in_light", (None, None))
    out["dH_in_light_target"] = tgt
    out["dH_in_light_reproduced"] = ((abs(out["dH_in_light"] - tgt) <= tol) if (out["dH_in_light"] is not None and tgt is not None) else None)
    if out["reproduced"] and out["dH_in_light"] is not None:
        tot = sum(v["value"] for v in out["chosen"].values())
        out["triple_sum"] = tot
        out["closure_residual"] = tot - out["dH_in_light"]
    if not out["reproduced"]:
        out["notes"].append("the budget's three terms are not reproduced at the printed digits under one mode: see the candidates and the light_v36 keys of the first "
                            "record; --budget-keys names the entries (a renaming only, checked against the same digits)")
    return out


def budget_window(recs, chosen, mode, k0, k1):
    """The three budget terms over the window k0 to k1 under the read-off mode (per-step entries read as the step ending at the record)."""
    out = {}
    for nm, ch in chosen.items():
        p = undot(ch["path"])
        if mode == "per_step_sum":
            vals = [num(get_path(recs[k], p)) for k in range(k0 + 1, k1 + 1)]
            out[nm] = float(sum(v for v in vals if v is not None)) if any(v is not None for v in vals) else None
        else:
            a, b = num(get_path(recs[k0], p)), num(get_path(recs[k1], p))
            out[nm] = (b - a) if (a is not None and b is not None) else None
    return out


def current_integral(S, key, k0, k1, rs):
    T = np.array([num(x) if num(x) is not None else np.nan for x in S["T"][k0:k1 + 1]], dtype=float)
    I = np.array([num(x) if num(x) is not None else np.nan for x in S[key][k0:k1 + 1]], dtype=float)
    ok = np.isfinite(T) & np.isfinite(I)
    T, I = T[ok], I[ok]
    if T.size < 2:
        return None
    dT = np.diff(T)
    trap = float(np.sum(0.5 * (I[1:] + I[:-1]) * dT))
    left = float(np.sum(I[:-1] * dT))
    right = float(np.sum(I[1:] * dT))
    return {"trapezoid_T": trap, "left_T": left, "right_T": right, "trapezoid_T_over_rs": trap / rs, "left_T_over_rs": left / rs, "right_T_over_rs": right / rs, "records": int(T.size)}


def read_off_integral(S, k0, k1, rs):
    """Which quadrature and time unit reproduce readout 2's integrated chain-end currents (3.228e-06 heavy, 5.199e-08 light sourcing)."""
    out = {"per_key": {}, "chosen": None}
    votes = {}
    for key, (tgt, tol) in READOUT2_INTEGRALS.items():
        r = current_integral(S, key, k0, k1, rs)
        out["per_key"][key] = {"target": tgt, "values": r, "matches": ([q for q, v in r.items() if q != "records" and abs(v - tgt) <= tol] if r else [])}
        for q in out["per_key"][key]["matches"]:
            votes[q] = votes.get(q, 0) + 1
    both = [q for q, c in votes.items() if c == len(READOUT2_INTEGRALS)]
    out["chosen"] = both[0] if both else None
    out["note"] = ("reproduced by %s" % both[0] if both else "no quadrature/unit reproduces both printed integrals at the printed digits; the trapezoid in T is used, flagged")
    return out


def cumulative_integral_series(S, key, quad):
    """The cumulative integral of a per-record current from record 0 under the read-off quadrature ('trapezoid_T', 'left_T', ... ; the unit by the suffix)."""
    n = len(S["T"])
    out = [None] * n
    rs = None
    acc = 0.0
    out[0] = 0.0 if num(S[key][0]) is not None else None
    for k in range(1, n):
        T0, T1, I0, I1 = num(S["T"][k - 1]), num(S["T"][k]), num(S[key][k - 1]), num(S[key][k])
        if None in (T0, T1, I0, I1):
            out[k] = None
            continue
        dT = T1 - T0
        if quad.startswith("trapezoid"):
            acc += 0.5 * (I0 + I1) * dT
        elif quad.startswith("left"):
            acc += I0 * dT
        else:
            acc += I1 * dT
        out[k] = acc
    if quad.endswith("T_over_rs"):
        rs = num(S["T"][1]) / num(S["T_over_rs"][1]) if (num(S["T_over_rs"][1]) not in (None, 0.0)) else None
        if rs:
            out = [x / rs if x is not None else None for x in out]
    return out


def truncation_candidates(S, k, m):
    """The guard channels over the sourcing channels in the stored E_inv, the chain-end cut current and the shell source total; every definition."""
    src, grd = [str(c) for c in m["sourcing"]], [str(c) for c in m["guard"]]

    def g(name):
        return num(S[name][k]) if name in S else None

    def ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0.0)) else None

    def ssum(names):
        vals = [g(nm) for nm in names]
        return None if any(v is None for v in vals) else float(sum(vals))
    w = {c: g("L%s.w" % c) for c in src + grd}
    out = {}
    out["E_inv:w"] = ratio(ssum(["L%s.E_inv_weighted" % c for c in grd]), ssum(["L%s.E_inv_weighted" % c for c in src]))
    out["E_inv:1"] = ratio(ssum(["L%s.E_inv" % c for c in grd]), ssum(["L%s.E_inv" % c for c in src]))
    IL, ILs = g("light.I_E_light"), g("light.I_E_light_sourcing")
    out["I_E:total"] = ratio((IL - ILs) if (IL is not None and ILs is not None) else None, ILs)
    out["I_E:w"] = ratio(ssum(["L%s.I_E_cut_weighted" % c for c in grd]), ssum(["L%s.I_E_cut_weighted" % c for c in src]))
    ew = {}
    for c in src + grd:
        ew[c] = g("L%s.eps_sum_weighted" % c)
        if ew[c] is None and g("L%s.eps_sum" % c) is not None and w[c] is not None:
            ew[c] = w[c] * g("L%s.eps_sum" % c)
    out["S:w"] = ratio(None if any(ew[c] is None for c in grd) else sum(ew[c] for c in grd), None if any(ew[c] is None for c in src) else sum(ew[c] for c in src))
    out["S:1"] = ratio(ssum(["L%s.eps_sum" % c for c in grd]), ssum(["L%s.eps_sum" % c for c in src]))
    return out


def truncation_read_off(S, m, kf, k40):
    """The definitions matching the md's six printed ratios at the formation and at 40."""
    out = {"at_formation": truncation_candidates(S, kf, m) if kf is not None else None, "at_40": truncation_candidates(S, k40, m) if k40 is not None else None, "chosen": {}}
    for i, group in enumerate(("E_inv", "I_E", "S")):
        best = None
        for defn in [d for d in (out["at_40"] or {}) if d.startswith(group + ":")]:
            res = []
            for key, vals in (("formation", out["at_formation"]), ("40", out["at_40"])):
                if vals is None or vals.get(defn) is None:
                    continue
                res.append(rel_res(vals[defn], MD_TRUNC[key][i]))
            sc = max(res) if res else math.inf
            if best is None or sc < best[1]:
                best = (defn, sc)
        out["chosen"][group] = {"definition": best[0] if best else None, "max_rel_residual_vs_md": best[1] if best else None, "reproduced": bool(best and best[1] <= 5e-6)}
    return out


# ---------------------------------------------------------------------------------------------------------------------- slices
def list_slices(dirs, rs, regex, md5=False):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            raise ReadoutError("slice directory not found: %s" % d)
        files += sorted(glob.glob(os.path.join(d, "*.npz")))
    out = []
    for p in files:
        rec = {"path": p, "bytes": os.path.getsize(p), "keys": None, "clock": None, "clock_from": None, "error": None, "md5": (md5_of(p) if md5 else None)}
        try:
            z = np.load(p, allow_pickle=False)
        except Exception as e:  # noqa: BLE001  a file that is not an npz is listed with its error
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            out.append(rec)
            continue
        keys = list(z.files)
        rec["keys"] = keys
        for key in CLOCK_KEYS:
            if key in keys:
                try:
                    val = float(np.asarray(z[key]).ravel()[0])
                    rec["clock"] = val / rs if key == "T" else val
                    rec["clock_from"] = key
                    break
                except Exception:  # noqa: BLE001
                    pass
        if rec["clock"] is None:
            mm = re.search(regex, os.path.basename(p))
            if mm:
                rec["clock"] = float(mm.group(1))
                rec["clock_from"] = "filename"
        z.close()
        out.append(rec)
    return out


def inspect_slice(path, keys):
    """The keys' shapes and dtypes from the npy headers inside the zip (no array is read: an orbitals slice is 3.5 GB); a member whose header cannot be
    parsed is loaded instead."""
    z = np.load(path, allow_pickle=False)
    out = []
    for key in keys:
        try:
            with z.zip.open(key + ".npy") as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _fo, dtype = np.lib.format.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _fo, dtype = np.lib.format.read_array_header_2_0(fh)
                else:
                    raise ValueError("npy format version %s" % (version,))
            out.append({"key": key, "shape": list(shape), "dtype": str(dtype)})
        except Exception:  # noqa: BLE001
            try:
                a = z[key]
                out.append({"key": key, "shape": list(a.shape), "dtype": str(a.dtype)})
            except Exception as e:  # noqa: BLE001
                out.append({"key": key, "error": "%s: %s" % (type(e).__name__, e)})
    z.close()
    return out


def role_keys(o, field, kappa=None):
    base = dict(DEFAULT_KEYS[field])
    user = o.get("slice_keys") or {}
    base.update(user.get(field, {}) if isinstance(user, dict) else {})
    return {role: (key.format(k=kappa) if kappa is not None else key) for role, key in base.items()}


def carries(keys, roles):
    return bool(keys) and (roles["C"] in keys or roles["orbitals"] in keys)


def upcast(A):
    A = np.asarray(A)
    if A.dtype == np.complex64:
        return A.astype(np.complex128)
    if A.dtype.kind in "iu" or A.dtype in (np.float32, np.float16):
        return A.astype(np.float64)
    return A


def herm_dev(A):
    sc = float(np.max(np.abs(A))) if A.size else 0.0
    if sc == 0.0:
        return 0.0
    return float(np.max(np.abs(A - A.conj().T))) / sc


def herm_dev_any(A):
    """herm_dev for a dense array or a scipy sparse matrix (the builder's host CSR)."""
    if not sp.issparse(A):
        return herm_dev(np.asarray(A))
    sc = float(np.max(np.abs(A.data))) if A.nnz else 0.0
    if sc == 0.0:
        return 0.0
    D = (A - A.conj().T).tocsr()
    return (float(np.max(np.abs(D.data))) / sc) if D.nnz else 0.0


def orient_columns(Phi, dim, where):
    """The orbital columns of a 2-d array: the engine stores them along axis 1 (shape (2 n_sites, n_orbitals): engine_twofield_v3.py run(), s_["sea"].cpu().numpy());
    with the dimension known the axis equal to it is the index axis, without it the longer axis (a square array cannot be read off: the slice must store C)."""
    Phi = np.asarray(Phi)
    if Phi.ndim != 2:
        raise Inconsistent("%s: the orbital array is not 2-d (shape %s)" % (where, Phi.shape))
    n0, n1 = Phi.shape
    if dim is not None:
        if n0 == dim:
            return Phi, dim
        if n1 == dim:
            return Phi.T, dim
        raise Inconsistent("%s: the orbital array's shape %s does not carry the operator dimension %d on either axis" % (where, Phi.shape, dim))
    if n0 > n1:
        return Phi, n0
    if n1 > n0:
        return Phi.T, n1
    raise Inconsistent("%s: a square orbital array (shape %s) does not single out the orbital axis; the slice must store C" % (where, Phi.shape))


def orth_defect(cols):
    """max |cols^dagger cols - 1|: zero for orthonormal columns (then cols cols^dagger is a projector)."""
    if cols.shape[1] == 0:
        return 0.0
    G = cols.conj().T @ cols
    return float(np.max(np.abs(G - np.eye(G.shape[0]))))


def diag_hX(h, cols_list):
    """Re diag(h X) for X = sum over the column sets of cols diag(f) cols^dagger (f None: occupation 1), through the products h cols (h a host CSR or a dense
    array); returns (the diagonal over the flattened index, the products h cols in the sets' order)."""
    e, HX = None, []
    for cols, f in cols_list:
        if cols.shape[1] == 0:
            Y = np.zeros_like(cols)
            d = np.zeros(cols.shape[0])
        else:
            Y = np.asarray(h @ cols)
            if f is None:
                d = np.einsum("ij,ij->i", cols.conj(), Y).real
            else:
                d = np.einsum("ij,ij,j->i", cols.conj(), Y, np.asarray(f, dtype=float)).real
        HX.append(Y)
        e = d if e is None else e + d
    return np.asarray(e, dtype=float), HX


def boundary_indices(co, mask):
    """The flattened indices h couples across the mask (co the COO form of h): every pair (i, j) with h_ij nonzero and mask_i != mask_j has both ends in the set."""
    sel = mask[co.row] != mask[co.col]
    return np.unique(np.concatenate([co.row[sel], co.col[sel]]))


def window_Y(HX, cols_list, B):
    """(h X)_BB = sum over the column sets of (h cols)_B diag(f) cols_B^dagger on the index set B."""
    Y = None
    for (cols, f), Hc in zip(cols_list, HX):
        if cols.shape[1] == 0:
            continue
        HB = Hc[B, :]
        if f is not None:
            HB = HB * np.asarray(f, dtype=float)[None, :]
        Yk = HB @ cols[B, :].conj().T
        Y = Yk if Y is None else Y + Yk
    return Y if Y is not None else np.zeros((B.size, B.size), dtype=complex)


def heavy_fields(per):
    """The heavy field's labels in a prepass's per-field dict and their weights: 'heavy' (one field, weight 1: the dense manufactured slices) or 'H<kappa>' per
    channel with the weight 16 kappa (engine_collapse_v5.Chan: self.w = 16.0 * kappa; the record's H_in, M1_I_E and E_inv_heavy are the weighted sums)."""
    out = {}
    for f in per:
        if f == "heavy":
            out[f] = 1.0
        elif len(f) > 1 and f[0] == "H" and f[1:].isdigit():
            out[f] = 16.0 * int(f[1:])
    return out


def heavy_sum(per, lay, rule, getter):
    """The weighted sum over the heavy channels of getter(per[channel][lay][rule]); None when there is no heavy field or a channel lacks the layout."""
    wH = heavy_fields(per)
    if not wH or not all(lay in per[f] and rule in per[f][lay] for f in wH):
        return None
    return float(sum(wH[f] * getter(per[f][lay][rule]) for f in wH))


def heavy_kappa_list(o, JM, JK):
    """The heavy channels: kappa 1 to args.kappa_sea (engine_twofield_v3.py run(): chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]),
    the two runs' equal; --heavy-channels narrows them (a reading decision, printed in the md)."""
    ks = [int(JM["args_continued"].get("kappa_sea") or 0), int(JK["args_continued"].get("kappa_sea") or 0)]
    if ks[0] <= 0 or ks[0] != ks[1]:
        raise Inconsistent("args.kappa_sea: member %s, control %s; the heavy channels cannot be formed" % tuple(ks))
    sel = str(o.get("heavy_channels") or "all")
    if sel == "all":
        return list(range(1, ks[0] + 1))
    out = sorted(set(int(x) for x in sel.split(",") if x.strip()))
    bad = [k for k in out if k < 1 or k > ks[0]]
    if bad or not out:
        raise Inconsistent("--heavy-channels %s: outside 1..%d" % (sel, ks[0]))
    return out


def heavy_weights_check(s, kappas, ids):
    """The heavy channels' kappas and weights stored in the slice at the join against the readout's 1..kappa_sea and 16 kappa (engine_collapse_v5.Chan.w)."""
    out = {"path": s["path"], "kappas": None, "weights": None, "readout_channels": list(kappas)}
    if s["keys"] and "kappas" in s["keys"] and "weights" in s["keys"]:
        z = np.load(s["path"], allow_pickle=False)
        kk = [int(x) for x in np.asarray(z["kappas"]).ravel()]
        ww = [float(x) for x in np.asarray(z["weights"]).ravel()]
        z.close()
        out.update({"kappas": kk, "weights": ww})
        ids.add("slice at the join: the stored heavy weights = 16 kappa (engine_collapse_v5.Chan.w) and the stored channels contain the readout's",
                [abs(w_ - 16.0 * k_) for k_, w_ in zip(kk, ww)] + [0.0 if set(kappas) <= set(kk) else 1.0], kind="cross")
    return out


def dedupe_slices(mine, label, flabel):
    """One slice per clock: where a clock sits in more than one directory (the md5-verified copies of the facts), the entry with the most keys is read and the
    other copies are listed as dropped; copies with the same keys and different sizes are two different files at one clock and stop the ledger."""
    by = {}
    for s in mine:
        by.setdefault(round(float(s["clock"]), 6), []).append(s)
    keep, dropped = [], []
    for cl in sorted(by):
        grp = by[cl]
        nmax = max(len(g["keys"] or []) for g in grp)
        cands = [g for g in grp if len(g["keys"] or []) == nmax]
        others = [g for g in grp if g not in cands]
        if len(set(int(g["bytes"]) for g in cands)) > 1:
            raise Inconsistent("%s %s: %d slices at T/r_s %g with the same keys and different sizes (%s): two different files at one clock, the ledger cannot tell which is the run's"
                               % (label, flabel, len(cands), cl, "; ".join("%s %d bytes" % (g["path"], g["bytes"]) for g in cands)))
        cands = sorted(cands, key=lambda g: mine.index(g))
        keep.append(cands[0])
        dropped += [g["path"] for g in cands[1:] + others]
    return keep, dropped


def gram(Phi, dim, occ, where):
    """C = Phi Phi^dagger (Phi diag(occ) Phi^dagger with occupations) from an orbital array whose orbital axis is read off against the operator dimension."""
    Phi = upcast(Phi)
    if Phi.ndim != 2:
        raise Inconsistent("%s: the orbital array is not 2-d (shape %s)" % (where, Phi.shape))
    n0, n1 = Phi.shape
    if dim is None:
        dim = max(n0, n1)
    if n0 == dim and n1 != dim:
        cols = Phi
    elif n1 == dim and n0 != dim:
        cols = Phi.T
    else:
        raise Inconsistent("%s: the orbital array's shape %s does not single out the orbital axis against the operator dimension %d (a square array's "
                           "orientation cannot be read off); the slice must store C" % (where, Phi.shape, dim))
    if occ is not None:
        occ = np.asarray(occ, dtype=float).ravel()
        if occ.size != cols.shape[1]:
            raise Inconsistent("%s: %d occupations against %d orbitals" % (where, occ.size, cols.shape[1]))
        return (cols * occ[None, :]) @ cols.conj().T
    return cols @ cols.conj().T


def field_ops(z, keys, roles, builder, field, kappa, where):
    """The operator triple of one field from a slice mapping. The dense route where the slice stores the state C (roles 'C') as a square array: (h, C, P_ref)
    dense, P_ref dense or the gram of its orbitals. The column route where it stores the state's orbital columns (roles 'orbitals': the engine's sea_<k>,
    occupation 1 unless roles 'occ' is present; plus roles 'dust' with the occupations roles 'dust_occ': the engine's dust_<k> and f_<k>) and the reference's
    columns (roles 'P_orbitals': the engine's R2_<k>): h (the builder's host CSR, or a stored dense h) with the column sets, nothing formed densely (C = sea
    sea^dagger + dust diag(f) dust^dagger and P_ref = R2 R2^dagger enter through their products with h). None when the slice does not carry the field's state;
    a loud stop when it carries the state but not h or P_ref (nothing substituted)."""
    kC, kP, kh = roles["C"], roles["P"], roles["h"]
    ko, kpo, kd, kf = roles.get("orbitals"), roles.get("P_orbitals"), roles.get("dust"), roles.get("dust_occ")
    if kC in keys:
        route = "dense"
        C, C_from = upcast(z[kC]), kC
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise Inconsistent("%s: %s is not a square matrix (shape %s)" % (where, C_from, C.shape))
        dim = int(C.shape[0])
    elif ko in keys:
        route = "columns"
        sea, dim = orient_columns(upcast(z[ko]), None, where + ": " + ko)
        occ = None
        if roles.get("occ") in keys:
            occ = np.asarray(z[roles["occ"]], dtype=float).ravel()
            if occ.size != sea.shape[1]:
                raise Inconsistent("%s: %d occupations (%s) against %d columns (%s)" % (where, occ.size, roles["occ"], sea.shape[1], ko))
        cols_C = [(sea, occ)]
        C_from = "%s (%d columns%s)" % (ko, sea.shape[1], (", occupations " + roles["occ"]) if occ is not None else ", occupation 1")
        if kd in keys:
            dust, _ = orient_columns(upcast(z[kd]), dim, where + ": " + kd)
            if dust.shape[1]:
                if kf not in keys:
                    raise MissingKey("%s: the key '%s' (the dust's occupations) is absent while '%s' carries %d columns (keys present: %s); the ledger stops, nothing is substituted"
                                     % (where, kf, kd, dust.shape[1], ", ".join(keys)))
                f = np.asarray(z[kf], dtype=float).ravel()
                if f.size != dust.shape[1]:
                    raise Inconsistent("%s: %d occupations (%s) against %d dust columns (%s)" % (where, f.size, kf, dust.shape[1], kd))
                cols_C.append((dust, f))
                C_from += " + %s (%d columns, occupations %s)" % (kd, dust.shape[1], kf)
            else:
                C_from += " (+ %s empty)" % kd
    else:
        return None
    if kh in keys:
        h, h_from = upcast(z[kh]), kh
    elif builder is not None:
        h, h_from = builder(z, field, kappa), "builder"
        if not sp.issparse(h):
            h = upcast(h)
    else:
        raise MissingKey("%s: the key '%s' (the field's one-body operator h) is absent and no --h-builder is given (keys present: %s); the ledger stops, nothing is substituted"
                         % (where, kh, ", ".join(keys)))
    if sp.issparse(h) and route == "dense":
        h = upcast(h.toarray())
    if h.shape != (dim, dim):
        raise Inconsistent("%s: h has shape %s against the state's dimension %d" % (where, h.shape, dim))
    if route == "dense":
        if kP in keys:
            P, P_from = upcast(z[kP]), kP
        elif kpo in keys:
            P, P_from = gram(z[kpo], dim, None, where + ": " + kpo), kpo + " (gram)"
        else:
            raise MissingKey("%s: the key '%s' (the continued reference P_ref) is absent and so is '%s' (its orbitals) (keys present: %s); the ledger stops, nothing is substituted"
                             % (where, kP, kpo, ", ".join(keys)))
        if P.shape != (dim, dim):
            raise Inconsistent("%s: P_ref has shape %s against the state's dimension %d" % (where, P.shape, dim))
        return {"h": h, "C": C, "P": P, "dim": dim, "route": "dense", "from": {"h": h_from, "C": C_from, "P": P_from},
                "herm": {"h": herm_dev(h), "C": herm_dev(C), "P": herm_dev(P)}, "dtype": {"h": str(h.dtype), "C": str(C.dtype), "P": str(P.dtype)}}
    if kP in keys:
        raise Inconsistent("%s: the state is stored as columns (%s) while the reference is stored dense (%s); the two routes do not mix" % (where, ko, kP))
    if kpo not in keys:
        raise MissingKey("%s: the key '%s' (the continued reference's columns) is absent (keys present: %s); the ledger stops, nothing is substituted" % (where, kpo, ", ".join(keys)))
    pc, _ = orient_columns(upcast(z[kpo]), dim, where + ": " + kpo)
    cols_P = [(pc, None)]
    return {"h": h, "cols_C": cols_C, "cols_P": cols_P, "dim": dim, "route": "columns", "from": {"h": h_from, "C": C_from, "P": "%s (%d columns)" % (kpo, pc.shape[1])},
            "herm": {"h": herm_dev_any(h), "C": 0.0, "P": 0.0}, "orth_P": orth_defect(pc), "n_cols": {"C": [int(c.shape[1]) for c, _ in cols_C], "P": int(pc.shape[1])},
            "dtype": {"h": str(h.dtype), "C": str(sea.dtype), "P": str(pc.dtype)}}


def site_index_map(dim, nsites, layout):
    if dim == nsites:
        return np.arange(dim)
    if nsites <= 0 or dim % nsites:
        raise Inconsistent("the operator dimension %d is not a multiple of the chain's %d sites; the inside of a cut cannot be formed" % (dim, nsites))
    m = dim // nsites
    if layout == "site_major":
        return np.arange(dim) // m
    if layout == "component_major":
        return np.arange(dim) % nsites
    raise Inconsistent("index layout %r unknown (site_major or component_major)" % (layout,))


def inside_mask(site, i_c, rule):
    return (site <= i_c) if rule == "<=" else (site < i_c)


def cut_index_from_r_collar(r_collar, nsites):
    """The engine's cut: round(r_collar) - 1 clipped to n_sites - 10 (icloud's reading of the digest: r_collar 236 -> 235, 2048 -> 2038)."""
    if num(r_collar) is None:
        return None
    return int(min(int(round(float(r_collar))) - 1, nsites - 10))


def collar_site_index(R, r_c):
    """The engine's cut index of a collar at r_c: the first site with R > r_c (engine_twofield_v3.py run(), the 'beyond' entries: i0 = int(np.searchsorted(R, Rc_rs * rs,
    side="right")), the sites beyond being R > r_c), so that under the inside rule '<' the inside is R <= r_c, the same convention as the chain-end cut
    (cut_index_from_r_collar: the first site beyond the collar; the inside is the flattened index range [0, 2 cut))."""
    R = np.asarray(R, dtype=float).ravel()
    i0 = int(np.searchsorted(R, float(r_c), side="right"))
    if i0 == 0:
        raise Inconsistent("no site at or inside r = %g" % r_c)
    i_near = int(np.argmin(np.abs(R - r_c)))
    return {"index": i0, "R_at_index": (float(R[i0]) if i0 < R.size else None), "R_last_inside": float(R[i0 - 1]), "nearest_index": i_near, "R_nearest": float(R[i_near]), "r_c": r_c,
            "rule": "the first site with R > r_c (the engine's 'beyond' index searchsorted(R, r_c, side='right')); under the read-off inside rule '<' the inside is R <= r_c"}


def cut_current(h, Y, mask):
    """Im sum_{i,j} (Pi_i - Pi_j) h_ij Y_ji with Y = h X: d/dt Re Tr[Pi h X] under X' = -i [h, X] at fixed h (energy entering the inside, s = +1)."""
    i_in = np.flatnonzero(mask)
    i_out = np.flatnonzero(~mask)
    if i_in.size == 0 or i_out.size == 0:
        return 0.0
    t1 = np.sum(h[np.ix_(i_in, i_out)] * Y[np.ix_(i_out, i_in)].T)
    t2 = np.sum(h[np.ix_(i_out, i_in)] * Y[np.ix_(i_in, i_out)].T)
    return float(np.imag(t1 - t2))


def point_terms(ops, masks):
    """At one clock: Re diag(h C), Re diag(h P_ref), their difference and, per cut mask, the inside and outside sums and the two cut currents. The dense route
    (ops carries C and P) and the column route (ops carries the column sets cols_C, cols_P; h sparse or dense) give the same numbers (the self-test's check 9):
    on the column route the diagonals come from the products h cols and the currents from the boundary index set B (the indices h couples across the mask)
    with (h X)_BB = sum over the column sets of (h cols)_B diag(f) cols_B^dagger, every pair the current sums over lying within B."""
    h = ops["h"]
    if "C" in ops:
        YC = np.asarray(h @ ops["C"])
        YP = np.asarray(h @ ops["P"])
        eC = np.array(np.real(np.diagonal(YC)), dtype=float)
        eP = np.array(np.real(np.diagonal(YP)), dtype=float)
        YX = YC - YP

        def currents(mask):
            return cut_current(h, YX, mask), cut_current(h, YC, mask)
    else:
        eC, HC = diag_hX(h, ops["cols_C"])
        eP, HP = diag_hX(h, ops["cols_P"])
        hs = h if sp.issparse(h) else sp.csr_matrix(h)
        co = hs.tocoo()

        def currents(mask):
            B = boundary_indices(co, mask)
            if B.size == 0:
                return 0.0, 0.0
            hB = hs[B][:, B].toarray()
            YC_B = window_Y(HC, ops["cols_C"], B)
            YP_B = window_Y(HP, ops["cols_P"], B)
            return cut_current(hB, YC_B - YP_B, mask[B]), cut_current(hB, YC_B, mask[B])
    eX = eC - eP
    cuts = {}
    for name, mask in masks.items():
        iX, iC = currents(mask)
        cuts[name] = {"H_in": float(eX[mask].sum()), "H_out": float(eX[~mask].sum()), "bare_in_C": float(eC[mask].sum()), "bare_in_P": float(eP[mask].sum()),
                      "I_X": iX, "I_C": iC, "n_in": int(mask.sum()), "n_out": int((~mask).sum())}
    return {"eC": eC, "eP": eP, "eX": eX, "trace_X": float(eX.sum()), "trace_C": float(eC.sum()), "trace_P": float(eP.sum()), "cuts": cuts}


def step_terms(h_prev, pt_prev, ops_cur, pt_cur, masks_prev, masks_cur, dT):
    """Over one step, at the step's start cut: the evolution term (the inside sum of Re[h (C' - C)], h the start operator), the work (the inside sum of
    Re[(h' - h)(C' - P')]) and the reference reset (-(the inside sum of Re[h (P' - P)])) in the records' own split (engine_twofield_v3.py's M1 balance:
    field_work = (X1 - X2) - (Y1 - Y2), reference_reset = -(Y2 - X2_prev), evolution_exact = Y1 - X1_prev, Y the start operator on the end's columns), the
    membership term of a moved cut (the four telescope exactly to dH_in), the trapezoid of the endpoint currents and the closure dH_in - (integral + work +
    reference + membership) = evolution - integral."""
    if "C" in ops_cur:
        dZ = np.array(np.real(np.diagonal(np.asarray(h_prev @ ops_cur["C"]))), dtype=float)
        dZP = np.array(np.real(np.diagonal(np.asarray(h_prev @ ops_cur["P"]))), dtype=float)
    else:
        dZ, _ = diag_hX(h_prev, ops_cur["cols_C"])
        dZP, _ = diag_hX(h_prev, ops_cur["cols_P"])
    out = {}
    for name in masks_prev:
        mp, mc = masks_prev[name], masks_cur[name]
        evol = float((dZ - pt_prev["eC"])[mp].sum())
        work = float(((pt_cur["eC"] - pt_cur["eP"]) - (dZ - dZP))[mp].sum())
        ref = -float((dZP - pt_prev["eP"])[mp].sum())
        memb = float(pt_cur["eX"][mc & ~mp].sum() - pt_cur["eX"][mp & ~mc].sum())
        dH = pt_cur["cuts"][name]["H_in"] - pt_prev["cuts"][name]["H_in"]
        iX = 0.5 * (pt_prev["cuts"][name]["I_X"] + pt_cur["cuts"][name]["I_X"]) * dT
        iC = 0.5 * (pt_prev["cuts"][name]["I_C"] + pt_cur["cuts"][name]["I_C"]) * dT
        out[name] = {"dH_in": dH, "evol": evol, "work": work, "ref": ref, "memb": memb, "telescoping_residual": dH - (evol + work + ref + memb),
                     "int_I_X": iX, "int_I_C": iC, "closure_X": dH - (iX + work + ref + memb), "closure_C": dH - (iC + work + ref + memb), "dT": dT,
                     "members_changed": int((mc ^ mp).sum())}
    return out


# --------------------------------------------------------------------------------------- P7: the conventions read off at the join clock
def prepass(label, run, s40, fields, builder, collar_idx, o, nsites):
    """At the slice clock of the join: every (layout, inside rule) candidate's cut quantities per field, and the content of the sites swept by the
    chain-end cut from record 0 to this clock (the sweep's estimate from this slice)."""
    S = run["S"]
    k = find_clock(S["T_over_rs"], s40["clock"])
    if k is None:
        raise Inconsistent("%s: no record at the slice clock T/r_s %g" % (label, s40["clock"]))
    ic_end = cut_index_from_r_collar(S["r_collar"][k], nsites)
    ic0 = cut_index_from_r_collar(S["r_collar"][0], nsites)
    if ic_end is None or ic0 is None:
        raise Inconsistent("%s: r_collar absent at record %d or 0; the chain-end cut cannot be formed" % (label, k))
    idx = dict(collar_idx)
    idx[CHAIN_END] = ic_end
    z = np.load(s40["path"], allow_pickle=False)
    per, info = {}, {}
    for flabel, field, kappa, roles in fields:
        ops = field_ops(z, s40["keys"], roles, builder, field, kappa, "%s %s T/r_s %g" % (label, flabel, s40["clock"]))
        if ops is None:
            info[flabel] = None
            continue
        dim = ops["dim"]
        if dim == nsites:
            layouts = ["identity"]
        elif o.get("index_layout") in (None, "auto"):
            layouts = ["site_major", "component_major"]
        else:
            layouts = [o["index_layout"]]
        per[flabel] = {}
        for lay in layouts:
            site = site_index_map(dim, nsites, lay)
            per[flabel][lay] = {}
            for rule in ("<=", "<"):
                masks = {nm: inside_mask(site, i, rule) for nm, i in idx.items()}
                pt = point_terms(ops, masks)
                swept = inside_mask(site, ic_end, rule) & ~inside_mask(site, ic0, rule)
                per[flabel][lay][rule] = {"cuts": pt["cuts"], "trace_X": pt["trace_X"], "trace_C": pt["trace_C"], "trace_P": pt["trace_P"],
                                           "swept_C": float(pt["eC"][swept].sum()), "swept_P": float(pt["eP"][swept].sum()), "swept_X": float(pt["eX"][swept].sum()),
                                           "swept_sites": int(np.unique(site[swept]).size)}
        info[flabel] = {"dim": dim, "from": ops["from"], "herm": ops["herm"], "dtype": ops["dtype"], "route": ops.get("route"), "orth_P": ops.get("orth_P"), "n_cols": ops.get("n_cols")}
        del ops
    z.close()
    return {"label": label, "k": k, "clock": s40["clock"], "file": s40["path"], "cut_index_end": ic_end, "cut_index_record0": ic0, "cut_index": idx, "per": per, "info": info}


def decide_conventions(pre, run, m):
    S, k = run["S"], pre["k"]
    chans, src = [str(c) for c in m["channels"]], [str(c) for c in m["sourcing"]]
    per = pre["per"]
    w = {c: (num(S["L%s.w" % c][k]) if ("L%s.w" % c) in S else None) for c in chans}
    wH = heavy_fields(per)
    have_heavy = bool(wH)
    light_present = [c for c in chans if ("L%s" % c) in per]
    have_light = bool(chans) and len(light_present) == len(chans)
    if not have_heavy and not have_light:
        raise Inconsistent("the slice at T/r_s %g carries neither the heavy nor every light channel's operators under the schema (fields found: %s)" % (pre["clock"], sorted(per)))
    recH, recHL = num(S["H_in"][k]), num(S["light.H_in_light"][k])
    layouts = sorted(set(lay for f in per for lay in per[f]))
    rows = []
    for lay in layouts:
        for rule in ("<=", "<"):
            hv = heavy_sum(per, lay, rule, lambda d: d["cuts"][CHAIN_END]["H_in"]) if have_heavy else None
            best = None
            if have_light and all(lay in per["L%s" % c] for c in chans):
                for wtag in ("w", "1"):
                    if wtag == "w" and any(w[c] is None for c in chans):
                        continue
                    for stag, cs in (("sourcing", src), ("all", chans)):
                        val = sum((w[c] if wtag == "w" else 1.0) * per["L%s" % c][lay][rule]["cuts"][CHAIN_END]["H_in"] for c in cs)
                        rr = rel_res(val, recHL)
                        if best is None or (rr is not None and (best[2] is None or rr < best[2])):
                            best = (wtag, stag, rr, val)
            rows.append({"layout": lay, "rule": rule, "H_in_heavy_slice": hv, "H_in_heavy_record": recH, "res_heavy": rel_res(hv, recH),
                         "weighting": ((best[0], best[1]) if best else None), "H_in_light_slice": (best[3] if best else None), "H_in_light_record": recHL,
                         "res_light": (best[2] if best else None)})

    def score(r):
        xs = [x for x in (r["res_heavy"], r["res_light"]) if x is not None]
        return max(xs) if xs else math.inf
    rows.sort(key=score)
    # the record tells two (layout, inside rule) candidates apart only beyond the tolerance at which it is reproduced (the cut site's own content may lie below
    # it): among the candidates reproducing within TOL_EXACT the engine's own convention is taken (site_major, '<': the inside is the flattened index range
    # [0, 2 cut), engine_collapse_v36.inside_energy_sp with Hr = H[:2 cut, :]), and the tie is a NOTE below
    tied_lr = [r for r in rows if score(r) <= V2.TOL_EXACT]
    if len(tied_lr) > 1:
        pick_lr = min(tied_lr, key=lambda r: (LAYOUT_RULE_PREFERENCE.index((r["layout"], r["rule"])) if (r["layout"], r["rule"]) in LAYOUT_RULE_PREFERENCE else len(LAYOUT_RULE_PREFERENCE)))
        rows.remove(pick_lr)
        rows.insert(0, pick_lr)
    best = rows[0]
    conv = {"layout": best["layout"], "rule": best["rule"], "weighting": best["weighting"], "rows": rows, "score": score(best),
            "H_in_reproduced": score(best) <= V2.TOL_EXACT, "weights": w, "flags": [], "k": k, "clock": pre["clock"], "have_heavy": have_heavy, "have_light": have_light,
            "light_present": light_present, "layout_rule_tie": len(tied_lr) > 1, "layout_rule_tied": [(r["layout"], r["rule"]) for r in tied_lr], "heavy_channels": sorted(wH), "heavy_weights": wH}
    lay, rule = best["layout"], best["rule"]
    recI, recIls = num(S["M1_I_E"][k]), num(S["light.I_E_light_sourcing"][k])
    crow = []
    for form in ("I_X", "I_C"):
        for sign in (1, -1):
            hv = (sign * heavy_sum(per, lay, rule, lambda d: d["cuts"][CHAIN_END][form])) if have_heavy else None
            lv = ((sign * sum(w[c] * per["L%s" % c][lay][rule]["cuts"][CHAIN_END][form] for c in src)) if (have_light and all(w[c] is not None for c in src)) else None)
            crow.append({"form": form, "sign": sign, "heavy_slice": hv, "heavy_record": recI, "res_heavy": rel_res(hv, recI, 1e-20),
                         "light_sourcing_slice": lv, "light_sourcing_record": recIls, "res_light": rel_res(lv, recIls, 1e-20)})

    def cscore(r):
        xs = [x for x in (r["res_light"], r["res_heavy"]) if x is not None]
        return max(xs) if xs else math.inf
    crow.sort(key=cscore)
    # the record tells two candidates apart only beyond the tolerance at which it is reproduced: every candidate reproducing within TOL_CURRENT ties
    # (with none reproducing, only exact equals tie), and the stated preference decides among tied candidates (I_C, the current of C alone, whose
    # fixed-cut integral is the evolution term that the closure (f) and the records' evolution_exact test; inward-positive first); a tie is a NOTE below
    best_c = cscore(crow[0])
    tied = [r for r in crow if cscore(r) <= TOL_CURRENT] or [r for r in crow if cscore(r) == best_c]
    pick = min(tied, key=lambda r: CURRENT_PREFERENCE.index((r["form"], r["sign"])))
    crow.remove(pick)
    crow.insert(0, pick)
    conv.update({"current_form": pick["form"], "current_sign": pick["sign"], "current_rows": crow, "current_score": cscore(pick),
                 "current_reproduced": cscore(pick) <= TOL_CURRENT, "current_tie": len(tied) > 1, "current_tied": [(r["form"], r["sign"]) for r in tied]})
    ev = {}
    if have_heavy:
        tr, rc = heavy_sum(per, lay, rule, lambda d: d["trace_X"]), num(S["light.E_inv_heavy"][k])
        fl = 64.0 * EPS * (abs(heavy_sum(per, lay, rule, lambda d: d["trace_C"]) or 0.0) + abs(heavy_sum(per, lay, rule, lambda d: d["trace_P"]) or 0.0))
        ev["heavy"] = {"trace": tr, "record": rc, "res": rel_res(tr, rc), "floor_abs": fl, "channels": sorted(wH), "weights": wH,
                       "within_floor": ((abs(tr - rc) <= max(V2.TOL_EXACT * abs(rc), fl)) if (tr is not None and rc is not None) else None)}
    for c in light_present:
        d_ = per["L%s" % c][lay][rule]
        tr, rc = d_["trace_X"], num(S["L%s.E_inv" % c][k])
        fl = 64.0 * EPS * (abs(d_["trace_C"]) + abs(d_["trace_P"]))
        ev["L%s" % c] = {"trace": tr, "record": rc, "res": rel_res(tr, rc), "floor_abs": fl, "within_floor": ((abs(tr - rc) <= max(V2.TOL_EXACT * abs(rc), fl)) if (tr is not None and rc is not None) else None)}
    conv["E_inv_checks"] = ev
    if conv["layout_rule_tie"]:
        conv["flags"].append("NOTE: the (layout, inside rule) candidates %s all reproduce the record's H_in and H_in_light within %s relative at T/r_s %s: the record does not tell them apart "
                             "(the cut site's own content lies below the reproduction's resolution), and the engine's own convention is taken, layout %s with the inside rule '%s' (the inside is "
                             "the flattened index range [0, 2 cut): engine_collapse_v36.inside_energy_sp with Hr = H[:2 cut, :])"
                             % (", ".join("%s '%s'" % lr for lr in conv["layout_rule_tied"]), fm(V2.TOL_EXACT, 1), fm(pre["clock"], 6), conv["layout"], conv["rule"]))
    if not conv["H_in_reproduced"]:
        conv["flags"].append("STOP: no (layout, inside rule, light weighting) reproduces the record's H_in and H_in_light at the chain-end cut at T/r_s %s (best: %s, %s, %s with "
                             "residuals heavy %s, light %s); the slices are not what the schema says and the ledger is not written"
                             % (fm(pre["clock"], 6), best["layout"], best["rule"], best["weighting"], fm(best["res_heavy"], 3), fm(best["res_light"], 3)))
    if not conv["current_reproduced"]:
        conv["flags"].append("FLAG: this readout's cut current (Im sum (Pi_i - Pi_j) h_ij (hX)_ji, X = C - P_ref or C, either sign) does not reproduce the record's stored chain-end "
                             "current at T/r_s %s to %s relative (best %s, sign %+d: %s); (c) is this readout's own current, its integral tested only by the closure"
                             % (fm(pre["clock"], 6), fm(TOL_CURRENT, 1), crow[0]["form"], crow[0]["sign"], fm(cscore(crow[0]), 3)))
    if conv["current_tie"]:
        conv["flags"].append("NOTE: the current's candidates %s score equally against the record's stored chain-end current at T/r_s %s (%s): the record does not tell them apart, the form and sign are not "
                             "read off, and %s with sign %+d is taken by the stated preference (I_C, the current of C alone, whose fixed-cut integral is the evolution term that the closure (f) and the "
                             "records' evolution_exact test; inward-positive first); forms tie when the reference P_ref carries no current across the cut, signs only when the stored current is zero"
                             % (", ".join("%s sign %+d" % fs for fs in conv["current_tied"]), fm(pre["clock"], 6),
                                ("every one reproduces it within %s relative" % fm(TOL_CURRENT, 1)) if conv["current_reproduced"] else "none reproduces it; the scores are exactly equal",
                                conv["current_form"], conv["current_sign"]))
    worst_ev = max([x["res"] for x in ev.values() if x["res"] is not None] or [0.0])
    beyond = [f for f, x in ev.items() if x.get("within_floor") is False]
    if beyond:
        conv["flags"].append("FLAG: the chain trace of h(C - P_ref) from the slice differs from the record's E_inv beyond the reproduction's resolution in %s (up to %s relative over all fields; "
                             "the resolution is max(%s relative, 64 eps (|Tr h C| + |Tr h P_ref|)): E_inv is the difference of two traces of order 1e3 per light channel and 1e6 for the heavy sum)"
                             % (", ".join(beyond), fm(worst_ev, 3), fm(V2.TOL_EXACT, 1)))
    return conv


def verify_conventions(pre, run, m, conv):
    """The control's slice at the join under the member's read-off conventions."""
    S, k = run["S"], pre["k"]
    chans, src = [str(c) for c in m["channels"]], [str(c) for c in m["sourcing"]]
    per, lay, rule = pre["per"], conv["layout"], conv["rule"]
    out = {"clock": pre["clock"], "k": k, "layout": lay, "rule": rule}
    w = conv["weights"]
    hin = heavy_sum(per, lay, rule, lambda d: d["cuts"][CHAIN_END]["H_in"])
    if hin is not None:
        out["heavy"] = {"H_in_slice": hin, "H_in_record": num(S["H_in"][k]), "res_H_in": rel_res(hin, num(S["H_in"][k])),
                        "trace": heavy_sum(per, lay, rule, lambda d: d["trace_X"]), "E_inv_record": num(S["light.E_inv_heavy"][k]),
                        "I_slice": conv["current_sign"] * heavy_sum(per, lay, rule, lambda d: d["cuts"][CHAIN_END][conv["current_form"]]), "I_record": num(S["M1_I_E"][k]),
                        "channels": sorted(heavy_fields(per))}
        out["heavy"]["res_I"] = rel_res(out["heavy"]["I_slice"], out["heavy"]["I_record"], 1e-20)
    if all(("L%s" % c) in per and lay in per["L%s" % c] for c in chans) and conv["weighting"]:
        wtag, stag = conv["weighting"]
        cs = src if stag == "sourcing" else chans
        val = sum((w[c] if wtag == "w" else 1.0) * per["L%s" % c][lay][rule]["cuts"][CHAIN_END]["H_in"] for c in cs)
        cur = conv["current_sign"] * sum(w[c] * per["L%s" % c][lay][rule]["cuts"][CHAIN_END][conv["current_form"]] for c in src) if all(w[c] is not None for c in src) else None
        out["light"] = {"H_in_light_slice": val, "H_in_light_record": num(S["light.H_in_light"][k]), "res_H_in": rel_res(val, num(S["light.H_in_light"][k])),
                        "I_sourcing_slice": cur, "I_sourcing_record": num(S["light.I_E_light_sourcing"][k]), "res_I": rel_res(cur, num(S["light.I_E_light_sourcing"][k]), 1e-20),
                        "E_inv_per_channel": {c: {"trace": per["L%s" % c][lay][rule]["trace_X"], "record": num(S["L%s.E_inv" % c][k]),
                                                  "res": rel_res(per["L%s" % c][lay][rule]["trace_X"], num(S["L%s.E_inv" % c][k]))} for c in chans}}
    return out


def sweep_estimate(pre, conv, m, P6, integrals_chosen, S):
    """The content of the sites the chain-end cut swept from record 0 to the slice clock, from the slice: bare C against evolution_exact - the integrated
    current, bare P_ref against -reference_reset, referenced (C - P_ref) as (e) over the window; an estimate (the content at the slice clock, not at
    the clock each site was swept)."""
    chans, src = [str(c) for c in m["channels"]], [str(c) for c in m["sourcing"]]
    per, lay, rule, w = pre["per"], conv["layout"], conv["rule"], conv["weights"]
    out = {"sites_swept": None, "cut_record0": pre["cut_index_record0"], "cut_at_clock": pre["cut_index_end"], "per_field": {}, "light_combinations": {}, "note": None}
    for f in per:
        if lay in per[f]:
            d = per[f][lay][rule]
            out["per_field"][f] = {"bare_C": d["swept_C"], "bare_P": d["swept_P"], "referenced_X": d["swept_X"], "sites": d["swept_sites"]}
            out["sites_swept"] = d["swept_sites"]
    wH = heavy_fields(per)
    if wH and all(f in out["per_field"] for f in wH):
        out["heavy_weighted"] = {q: float(sum(wH[f] * out["per_field"][f][q] for f in wH)) for q in ("bare_C", "bare_P", "referenced_X")}
    if P6.get("reproduced"):
        evo = P6["chosen"]["evolution_exact"]["value"]
        rr = P6["chosen"]["reference_reset"]["value"]
        key = "light.I_E_light_sourcing"
        qi = integrals_chosen or "trapezoid_T"
        intI = (P6.get("integrals", {}).get("per_key", {}).get(key, {}).get("values") or {}).get(qi)
        out["evolution_exact"], out["reference_reset"], out["integrated_current"] = evo, rr, intI
        out["evolution_minus_integral"] = (evo - intI) if intI is not None else None
        for wtag in ("w", "1"):
            if wtag == "w" and any(w[c] is None for c in chans):
                continue
            for stag, cs in (("sourcing", src), ("all", chans)):
                if not all(("L%s" % c) in out["per_field"] for c in cs):
                    continue
                bC = sum((w[c] if wtag == "w" else 1.0) * out["per_field"]["L%s" % c]["bare_C"] for c in cs)
                bP = sum((w[c] if wtag == "w" else 1.0) * out["per_field"]["L%s" % c]["bare_P"] for c in cs)
                bX = sum((w[c] if wtag == "w" else 1.0) * out["per_field"]["L%s" % c]["referenced_X"] for c in cs)
                out["light_combinations"]["%s,%s" % (wtag, stag)] = {"bare_C_swept": bC, "bare_P_swept": bP, "referenced_swept": bX,
                                                                      "res_vs_evolution_minus_integral": rel_res(bC, out["evolution_minus_integral"]),
                                                                      "res_vs_minus_reference_reset": rel_res(-bP, rr), "fixed_cut_reference_remainder": rr + bP,
                                                                      "evolution_remainder": (out["evolution_minus_integral"] - bC) if out["evolution_minus_integral"] is not None else None}
        if out["light_combinations"]:
            best = min(out["light_combinations"].items(), key=lambda kv: (kv[1]["res_vs_evolution_minus_integral"] if kv[1]["res_vs_evolution_minus_integral"] is not None else math.inf))
            out["best_combination"] = best[0]
            out["note"] = ("the swept bare content under %s reproduces evolution_exact - the integrated current to %s relative and -reference_reset to %s relative (an estimate at "
                           "the slice clock)" % (best[0], fm(best[1]["res_vs_evolution_minus_integral"], 3), fm(best[1]["res_vs_minus_reference_reset"], 3)))
    else:
        out["note"] = "the budget triple was not reproduced from the records; the sweep's estimate is printed without the comparison"
    return out


def exterior_check(pre, conv, run, m, keys_override):
    """5d's exterior entries at the join clock (the energy beyond 24 r_s per channel, signed) against (b) at 24 r_s from the slice."""
    S, k = run["S"], pre["k"]
    rec = run["records"][k]
    chans = [str(c) for c in m["channels"]]
    per, lay, rule, w = pre["per"], conv["layout"], conv["rule"], conv["weights"]
    out = {"found": [], "candidates_slice": {}, "best": {}, "note": None}
    if keys_override:
        found = [(undot(str(p)), get_path(rec, undot(str(p)))) for p in (keys_override.values() if isinstance(keys_override, dict) else keys_override)]
    else:
        found = [(p, v) for p, v in all_leaves(rec) if EXT_REGEX.search(dotted(p)) and num(v) is not None]
    if "r24" in pre["cut_index"]:
        for c in chans:
            f = "L%s" % c
            if f in per and lay in per[f]:
                ho = per[f][lay][rule]["cuts"]["r24"]["H_out"]
                out["candidates_slice"][f] = {"H_out_unweighted": ho, "H_out_weighted": (w[c] * ho if w[c] is not None else None)}
        hout = heavy_sum(per, lay, rule, lambda d: d["cuts"]["r24"]["H_out"])
        if hout is not None:
            out["candidates_slice"]["heavy"] = {"H_out_unweighted": None, "H_out_weighted": hout}
    for p, v in found:
        x = num(v)
        if x is None:
            continue
        best = None
        for f, cands in out["candidates_slice"].items():
            for tag, val in cands.items():
                r = rel_res(val, x, 1e-20)
                if r is not None and (best is None or r < best[2]):
                    best = (f, tag, r, val)
        out["found"].append({"path": dotted(p), "record_value": x, "best_match": ({"field": best[0], "form": best[1], "rel_residual": best[2], "slice_value": best[3]} if best else None)})
    if not found:
        out["note"] = ("no exterior entry found in the record at T/r_s %s by name (regex %s) and none named by --exterior-keys: 5d's join is not checked; (b) at 24 r_s stands as this "
                       "readout's own column" % (fm(pre["clock"], 6), EXT_REGEX.pattern))
    else:
        good = [f for f in out["found"] if f["best_match"] and f["best_match"]["rel_residual"] <= V2.TOL_EXACT]
        near = [f for f in out["found"] if f["best_match"] and f["best_match"]["rel_residual"] <= TOL_EXTERIOR]
        out["note"] = ("%d of %d exterior entries reproduced by (b) at 24 r_s to %s relative, %d to %s relative. 5d's energy_density_route is the site-density route, the sum over R > r_c of "
                       "N eps + v J with eps, J the unclocked h0's and Q's site densities of C - P_ref (engine_twofield_v3.py's 'beyond'); (b) is the operator trace Re Tr[Pi_out h (C - P_ref)] "
                       "with the clocked h at the same fields; the two differ by the cross-cut term (1/2) Re sum over i outside, j inside of [h0_ij (N_j - N_i) + Q_ij (v_j - v_i)] X_ji, "
                       "so agreement to that term's size and not to roundoff is the expected shape" % (len(good), len(out["found"]), fm(V2.TOL_EXACT, 1), len(near), fm(TOL_EXTERIOR, 1)))
    return out


def record_collar_entries(run, k):
    """What the record's s11 carries under names containing 'collar' (information for the per-record collar currents, if stored)."""
    rec = run["records"][k]
    s11 = rec.get("s11") if isinstance(rec, dict) else None
    out = []
    if isinstance(s11, dict):
        for p, v in all_leaves(s11):
            if "collar" in dotted(p).lower():
                out.append({"path": "s11." + dotted(p), "value": v})
    return out[:200]


# ------------------------------------------------------------------------------------------------------------------- the ledger
def run_ledger(label, run, slices, fields, builder, conv, collar_idx, o, log):
    """Per field, the slices at or after the window's start that carry it, one per clock (dedupe_slices), the heavy field's clocks narrowed by --heavy-clocks
    when given (a reading decision, printed); per clock the point terms and, from the second clock on, the step terms against the previous clock."""
    S, m = run["S"], run["m"]
    nsites, rs = int(m["nsites"]), float(m["rs"])
    out = {"label": label, "fields": {}, "nsites": nsites}
    for flabel, field, kappa, roles in fields:
        mine = sorted([s for s in slices if s["clock"] is not None and s["clock"] >= o["window_start"] - 1e-9 and carries(s["keys"], roles)], key=lambda s: s["clock"])
        if field == "heavy" and o.get("heavy_clocks"):
            mine = [s for s in mine if any(abs(s["clock"] - c) < 1e-9 for c in o["heavy_clocks"])]
        mine, dropped = dedupe_slices(mine, label, flabel)
        rows, prev = [], None
        for s in mine:
            t_f = time.time()
            z = np.load(s["path"], allow_pickle=False)
            ops = field_ops(z, s["keys"], roles, builder, field, kappa, "%s %s T/r_s %g" % (label, flabel, s["clock"]))
            z.close()
            if ops is None:
                continue
            k = find_clock(S["T_over_rs"], s["clock"])
            if k is None:
                raise Inconsistent("%s: the slice at T/r_s %g has no record at that clock" % (label, s["clock"]))
            idx = dict(collar_idx)
            idx[CHAIN_END] = cut_index_from_r_collar(S["r_collar"][k], nsites)
            if idx[CHAIN_END] is None:
                raise Inconsistent("%s: r_collar absent at record %d (T/r_s %g)" % (label, k, s["clock"]))
            site = site_index_map(ops["dim"], nsites, conv["layout"])
            masks = {nm: inside_mask(site, i, conv["rule"]) for nm, i in idx.items()}
            pt = point_terms(ops, masks)
            row = {"clock": s["clock"], "T": s["clock"] * rs, "k": k, "file": s["path"], "dim": ops["dim"], "route": ops.get("route"), "herm": ops["herm"], "orth_P": ops.get("orth_P"),
                   "n_cols": ops.get("n_cols"), "from": ops["from"], "cut_index": idx, "trace_X": pt["trace_X"], "trace_C": pt["trace_C"], "trace_P": pt["trace_P"], "cuts": pt["cuts"], "step": None}
            if prev is not None:
                row["step"] = step_terms(prev["h"], prev["pt"], ops, pt, prev["masks"], masks, (s["clock"] - prev["clock"]) * rs)
            prev = {"h": ops["h"], "pt": pt, "masks": masks, "clock": s["clock"]}
            del ops
            row["seconds"] = time.time() - t_f
            log("%s %s T/r_s %g: dim %d, %s route, %.1f s" % (label, flabel, s["clock"], row["dim"], row["route"], row["seconds"]))
            rows.append(row)
        clocks = [r["clock"] for r in rows]
        out["fields"][flabel] = {"rows": rows, "clocks": clocks, "n_slices": len(rows), "duplicates_dropped": dropped,
                                 "steps_T_over_rs": sorted(set(round(b - a, 6) for a, b in zip(clocks[:-1], clocks[1:]))) if len(clocks) > 1 else []}
    return out


ACC_POINT = ("H_in", "H_out", "I_X", "I_C", "bare_in_C", "bare_in_P")
ACC_STEP = ("evol", "work", "ref", "memb", "int_I_X", "int_I_C", "closure_X", "closure_C", "telescoping_residual")


def accumulate(rows, cut):
    A = {"clock": [], "T": [], "n_in": [], "cut_index": [], "dH_in_from_start": []}
    for q in ACC_POINT:
        A[q] = []
    for q in ACC_STEP:
        A["step_" + q] = []
        A["cum_" + q] = []
    c = {q: 0.0 for q in ACC_STEP}
    H0 = rows[0]["cuts"][cut]["H_in"] if rows else None
    for r in rows:
        ct = r["cuts"][cut]
        A["clock"].append(r["clock"])
        A["T"].append(r["T"])
        A["n_in"].append(ct["n_in"])
        A["cut_index"].append(r["cut_index"][cut])
        A["dH_in_from_start"].append(ct["H_in"] - H0)
        for q in ACC_POINT:
            A[q].append(ct[q])
        st = r["step"][cut] if r["step"] else None
        for q in ACC_STEP:
            v = st[q] if st else None
            A["step_" + q].append(v)
            if v is not None:
                c[q] += v
            A["cum_" + q].append(c[q])
    return A


LINEAR = tuple(ACC_POINT) + tuple("step_" + q for q in ACC_STEP) + tuple("cum_" + q for q in ACC_STEP) + ("dH_in_from_start",)


def combine(acc_by_channel, weights):
    """The weighted sum over channels of the linear columns at the clocks common to all channels."""
    labels = list(acc_by_channel)
    if not labels or any(acc_by_channel[f] is None for f in labels):
        return None
    common = set(acc_by_channel[labels[0]]["clock"])
    for f in labels[1:]:
        common &= set(acc_by_channel[f]["clock"])
    common = sorted(common)
    first = acc_by_channel[labels[0]]
    pos = {f: {cl: i for i, cl in enumerate(acc_by_channel[f]["clock"])} for f in labels}
    out = {"clock": common, "T": [first["T"][pos[labels[0]][cl]] for cl in common], "n_in": [first["n_in"][pos[labels[0]][cl]] for cl in common],
           "cut_index": [first["cut_index"][pos[labels[0]][cl]] for cl in common]}
    for q in LINEAR:
        col = []
        for cl in common:
            vals = [acc_by_channel[f][q][pos[f][cl]] for f in labels]
            col.append(None if any(v is None for v in vals) else float(sum(weights[f] * v for f, v in zip(labels, vals))))
        out[q] = col
    return out


def difference(Am, Ak):
    if Am is None or Ak is None:
        return None
    pk = {cl: i for i, cl in enumerate(Ak["clock"])}
    out = {"clock": [], "n_in_equal": []}
    for q in LINEAR:
        out[q] = []
    for i, cl in enumerate(Am["clock"]):
        if cl not in pk:
            continue
        j = pk[cl]
        out["clock"].append(cl)
        out["n_in_equal"].append(Am["n_in"][i] == Ak["n_in"][j] and Am["cut_index"][i] == Ak["cut_index"][j])
        for q in LINEAR:
            a, b = Am[q][i], Ak[q][j]
            out[q].append((a - b) if (a is not None and b is not None) else None)
    return out


def collar_charges(Z, S, g, cuts_r):
    """M(r_c) from the s11 npz's nodes per record (the identity E = W/2g applied at r_c); W/2g beside when W is stored."""
    Tz = np.asarray(Z["T"], dtype=float).ravel()
    RN = np.asarray(Z["r_nodes"], dtype=float)
    if RN.ndim == 1:
        RN = RN[None, :]
    rn_dev = float(np.max(np.abs(RN - RN[0:1]))) if RN.shape[0] else 0.0
    n = len(S["T"])
    out = {name: [None] * n for name in cuts_r}
    outW = {name: [None] * n for name in cuts_r} if Z.get("W") is not None else None
    for k in range(n):
        T = num(S["T"][k])
        if T is None:
            continue
        j = np.flatnonzero(np.abs(Tz - T) <= 1e-9 * max(1.0, abs(T)))
        if j.size == 0:
            continue
        j = int(j[0])
        rn = RN[j] if j < RN.shape[0] else RN[0]  # a continued s11 npz carries the continuation's r_nodes rows only (engine_collapse_v36.resume_slice_v34 copies every per-record array but r_nodes); the grid is fixed (run(): r_nodes_fixed = FLD.coarse_grid(R, a.ell)), rn_dev checks it
        M = Z["M"][j]
        for name, r_c in cuts_r.items():
            if r_c is None:
                out[name][k] = float(M[-1])
                if outW is not None:
                    outW[name][k] = float(Z["W"][j][-1]) / (2.0 * g)
            else:
                out[name][k] = float(np.interp(r_c, rn, M))
                if outW is not None:
                    outW[name][k] = float(np.interp(r_c, rn, Z["W"][j])) / (2.0 * g)
    return {"M": out, "W_over_2g": outW, "node_spacing_record0": (float(np.median(np.diff(RN[0]))) if RN.shape[1] > 1 else None), "r_nodes_rows": int(RN.shape[0]), "r_nodes_max_dev": rn_dev}


# ------------------------------------------------------------------------------------------------------------------------- md
def tab_cut(cut, Am, Ak, Dd, form, gM, gK, clocks_rows=None):
    """One ledger table at one cut: the member's columns, the control's at the matched clocks, the differences."""
    pk = {cl: i for i, cl in enumerate(Ak["clock"])} if Ak else {}
    pd = {cl: i for i, cl in enumerate(Dd["clock"])} if Dd else {}
    I, intI, clo = form, "cum_int_" + form, "cum_closure_" + form.split("_")[1]
    rows = []
    for i, cl in enumerate(Am["clock"]):
        if clocks_rows is not None and cl not in clocks_rows:
            continue
        j = pk.get(cl)
        d = pd.get(cl)

        def mk(q, dg):
            a = Am[q][i]
            b = Ak[q][j] if j is not None else None
            dd = Dd[q][d] if d is not None else None
            return [fm(a, dg), fm(b, dg), fm(dd, 3)]
        row = [fm(cl, 6), str(Am["n_in"][i])]
        row += mk("H_in", 9) + mk("H_out", 6) + mk(I, 4) + mk(intI, 5) + mk("cum_work", 5) + mk("cum_ref", 5) + mk("cum_memb", 4) + mk(clo, 4)
        row += [fm(gM.get(cl), 9) if gM else "-", fm(gK.get(cl), 9) if gK else "-"]
        rows.append(row)
    head = ["T/r_s", "n_in", "H_in C", "ctrl", "d", "H_out C", "ctrl", "d", "I_E C", "ctrl", "d", "int I_E from 40 C", "ctrl", "d", "work C", "ctrl", "d", "ref C", "ctrl", "d",
            "memb C", "ctrl", "d", "closure C", "ctrl", "d", "W/2g C", "ctrl"]
    return mdtab(head, rows)


def build_md(ctx):
    g = fm
    P, o = ctx["P"], ctx["o"]
    JM, JK = ctx["JM"], ctx["JK"]
    L = ["# Computation C: the continuation's collar ledger, %s" % VERSION, "", LABEL, "",
         "Generated %s by %s. %s. The member %s (T/r_s %s to %s, %d records), its coupling-off control %s (T/r_s %s to %s, %d records); the matched window ends at the "
         "control's last clock; the reading of section 0 is stated before the numbers; the preconditions of section 2 print before any column of the ledger."
         % (ctx["generated"], VERSION, SPEC, JM["m"]["tag"], g(JM["S"]["T_over_rs"][0], 6), g(JM["S"]["T_over_rs"][-1], 6), JM["m"]["n"], JK["m"]["tag"],
            g(JK["S"]["T_over_rs"][0], 6), g(JK["S"]["T_over_rs"][-1], 6), JK["m"]["n"]), "", "## 0. The reading (stated before the numbers)", ""]
    for head, text in READING:
        L += ["**%s** %s" % (head, text), ""]
    L += ["## 1. The inputs", ""]
    for d in ctx["inputs"]:
        L.append("- %s: %s (md5 %s, %d bytes)" % (d["role"], d["path"], d["md5"], d["bytes"]))
    L.append("- slice directories: member %s; control %s" % (", ".join(o["slices"]), ", ".join(o["control_slices"])))
    L += ["", "## 2. The preconditions", "", "### 2.1 (P1) The join at 40 and the shared records", ""]
    for lab in ("member", "control"):
        rep = P["P1_join"][lab]
        jr = rep["join_record"]
        L.append("- %s: delivered %d records to T/r_s %s; continuation %d records from %s to %s; %d shared records, %d bitwise; %d appended; joined %d records to %s; gaps %d%s"
                 % (lab, rep["delivered_records"], g(rep["delivered_last_T_over_rs"], 6), rep["continued_records"], g(rep["continued_first_T_over_rs"], 6), g(rep["continued_last_T_over_rs"], 6),
                    rep["overlap_records"], rep["overlap_bitwise"], rep["appended_records"], rep["joined_records"], g(rep["joined_last_T_over_rs"], 6), len(rep["gaps"]),
                    ("; FLAG the continuation's clocks before the join not in the delivered run: %s" % rep["continued_before_join_unmatched_clocks"][:10]) if rep["continued_before_join_unmatched_clocks"] else ""))
        if jr is None:
            L.append("  - FLAG: the continuation json carries no record at the delivered last clock; the join is not compared leaf by leaf")
        else:
            L.append("  - at T/r_s %s: %s (%d leaves differ)" % (g(jr["T_over_rs"], 6), "bitwise" if jr["bitwise"] else "NOT bitwise", jr["leaves_differing"]))
            for key, gt in jr["gate"].items():
                L.append("    - %s: delivered %s, continued %s, d %s (%s times the gate's pass level %s, a pass level for the identical command line and not an error)"
                         % (key, g(gt["delivered"], 10), g(gt["continued"], 10), g(gt["d"], 3), g(gt["times_tolerance"], 3), g(gt["tolerance"], 2)))
            for df in jr["differences"][:40]:
                L.append("    - differs: %s: delivered %s, continued %s" % (df["path"], g(df["delivered"], 12), g(df["continued"], 12)))
        rg = rep["resume_gate_records"]
        if rg:
            L.append("  - the resume gate recomputed on the %d shared records before the join (T/r_s %s to %s): %d bitwise; largest |d| over the gate scalars: %s"
                     % (len(rg), g(rg[0]["T_over_rs"], 6), g(rg[-1]["T_over_rs"], 6), sum(1 for p in rg if p["bitwise"]),
                        "; ".join("%s %s (%s times)" % (key, g(max((p["gate"][key]["d"] or 0.0) for p in rg), 3), g(max((p["gate"][key]["times_tolerance"] or 0.0) for p in rg), 3)) for key, _ in GATE)))
        else:
            L.append("  - the 2-record resume gate: not recomputable from these jsons (the continuation json carries no record before the join); the builder's gate result is to be filed with this readout")
    L += ["", "### 2.2 (P2) The arguments", ""]
    for key, val in P["P2_args"].items():
        L.append("- %s: %s%s" % (key, "; ".join("%s: %s against %s" % (k, g(va, 6), g(vb, 6)) for k, va, vb in val["differences"]) or "identical (out, tag, gpu, slices, resume entries skipped by V1.args_diff)",
                                 ("; UNEXPECTED beyond %s: %s" % (val["expected"], ", ".join(val["unexpected"])) if val["unexpected"] else "")))
    L += ["", "### 2.3 (P3) The engine version", ""]
    for lab, J in (("member", JM), ("control", JK)):
        L.append("- %s: script delivered %s; continued %s; version keys delivered %s, continued %s" % (lab, J["script_delivered"], J["script_continued"], J["version_keys"]["delivered"] or "none", J["version_keys"]["continued"] or "none"))
    L += ["", "### 2.4 (P4) The solver over the continuation", ""]
    for lab in ("member", "control"):
        s = P["P4_solver"][lab]
        ref = DELIVERED_SOLVER[lab]
        L.append("- %s (%d records, T/r_s %s to %s): iterations at most %s (histogram %s, absent on %d), kept residual at most %s at T/r_s %s, fallbacks %d channel-records, converged false %d "
                 "(absent %d), certified_on_record false %d; the delivered run's: %s"
                 % (lab, s["records"], g(s["first_T_over_rs"], 6), g(s["last_T_over_rs"], 6), g(s["iterations_max"], 1), s["iterations_histogram"], s["iterations_absent"], g(s["kept_residual_max"], 3),
                    g(s["T_at_kept_residual_max"], 6), s["fallbacks"], s["converged_false"], s["converged_absent"], s["certified_false"],
                    ", ".join("%s %s" % (k, g(v, 3)) for k, v in ref.items()) + "; recomputed to 40 here: " + ", ".join("%s %s" % (k, g(v, 3)) for k, v in P["P4_solver"][lab + "_delivered_recomputed"].items() if k in ("iterations_max", "kept_residual_max", "fallbacks"))))
    w = P["P5_window"]
    L += ["", "### 2.5 (P5) The window", "",
          "- the member's last clock %s, the control's %s (the plan's 'about 45 to 50'): the matched window runs from %s to %s (%d matched records, %d matched slice clocks of the light field, %d of the heavy); "
          "member-only from %s to %s" % (g(w["member_last"], 6), g(w["control_last"], 6), g(w["window_start"], 6), g(w["matched_end"], 6), w["matched_records"], w["matched_slice_clocks_light"], w["matched_slice_clocks_heavy"],
                                         g(w["matched_end"], 6), g(w["member_last"], 6))]
    B = P["P6_closure"]
    L += ["", "### 2.6 (P6) The chain-end closure over 0 to 40 from the delivered member's records", "",
          "- the budget entries found by name: %s" % ("; ".join("%s at %s" % (nm, ", ".join(ps) or "none") for nm, ps in B["paths_found"].items())),
          "- light_v36 keys of the first record: %s" % (", ".join(B["light_v36_keys_first_record"] or []) or "none")]
    for c in B["candidates"]:
        L.append("- candidate %s at %s (%d records): per-step sum %s, cumulative last %s, last - first %s; printed %s; matches %s"
                 % (c["name"], c["path"], c["available"], g(c["sum"], 12), g(c["last"], 12), g(c["last_minus_first"], 12), g(c["target"], 6), c["matches"] or "none"))
    if B["reproduced"]:
        L.append("- **readout 2's triple REPRODUCED** (mode %s): evolution_exact %s, field_work %s, reference_reset %s, sum %s against dH_in_light = H_in_light(40) - H_in_light(0) = %s - %s = %s (printed 0.003977%s); "
                 "the triple's closure residual %s" % (B["mode"], g(B["chosen"]["evolution_exact"]["value"], 12), g(B["chosen"]["field_work"]["value"], 8), g(B["chosen"]["reference_reset"]["value"], 12), g(B["triple_sum"], 8),
                                                        g(B["H_in_light_end"], 8), g(B["H_in_light_start"], 8), g(B["dH_in_light"], 8), ", reproduced" if B["dH_in_light_reproduced"] else ", NOT reproduced", g(B["closure_residual"], 3)))
    else:
        L.append("- **STOP: readout 2's triple is NOT reproduced from the records**: %s" % "; ".join(B["notes"]))
    I2 = B["integrals"]
    for key, d in I2["per_key"].items():
        v = d["values"] or {}
        L.append("- the integrated chain-end current %s over 0 to 40 (printed %s): trapezoid in T %s, left %s, right %s; in T/r_s %s, %s, %s; matches %s"
                 % (key, g(d["target"], 4), g(v.get("trapezoid_T"), 6), g(v.get("left_T"), 6), g(v.get("right_T"), 6), g(v.get("trapezoid_T_over_rs"), 6), g(v.get("left_T_over_rs"), 6), g(v.get("right_T_over_rs"), 6), d["matches"] or "none"))
    L.append("- %s" % I2["note"])
    E_ = B["E"]
    L.append("- E(0) %s (md 59.788989), E(40) %s, E(0) - E(40) %s (md 0.563327), E/E0 at 40 %s (md 0.9906); the md's values at 40: %s"
             % (g(E_["E0"], 9), g(E_["E40"], 9), g(E_["fall"], 7), g(E_["E_over_E0_40"], 5), "; ".join("%s %s against md %s (rel %s)" % (k, g(v["value"], 9), g(v["md"], 9), g(v["rel"], 3)) for k, v in B["md_at_40"].items())))
    tr = B["truncation"]
    L.append("- the truncation table's definitions read off against the md's six ratios: %s; at the formation %s, at 40 %s"
             % ("; ".join("%s: %s (max rel %s%s)" % (grp, d["definition"], g(d["max_rel_residual_vs_md"], 3), "" if d["reproduced"] else ", NOT at the printed digits") for grp, d in tr["chosen"].items()),
                {k: g(v, 6) for k, v in (tr["at_formation"] or {}).items()}, {k: g(v, 6) for k, v in (tr["at_40"] or {}).items()}))
    if B.get("v2_json_check"):
        L.append("- version 2's delivered json at 40: %s" % B["v2_json_check"])
    if B.get("control_triple"):
        Bc = B["control_triple"]
        L.append("- the control's own budget over 0 to 40 (information; no printed target): %s" % (", ".join("%s %s" % (nm, g(v, 12)) for nm, v in Bc.items()) if Bc else "not computed"))
    S7 = P["P7_slices"]
    L += ["", "### 2.7 (P7) The slices, the schema and the conventions read off at the join", ""]
    for lab in ("member", "control"):
        ls = S7["listing"][lab]
        cl = sorted(set(s["clock"] for s in ls if s["clock"] is not None))
        L.append("- %s: %d npz files, %d with a clock (%s to %s; %d at or after %s), %d without a clock, %d unreadable; clocks from %s; sizes %s to %s bytes%s"
                 % (lab, len(ls), len(cl), g(cl[0], 6) if cl else "-", g(cl[-1], 6) if cl else "-", sum(1 for s in ls if s["clock"] is not None and s["clock"] >= o["window_start"] - 1e-9), g(o["window_start"], 4),
                    sum(1 for s in ls if s["clock"] is None and s["error"] is None), sum(1 for s in ls if s["error"]), sorted(set(str(s["clock_from"]) for s in ls if s["clock"] is not None)),
                    min((s["bytes"] for s in ls), default=0), max((s["bytes"] for s in ls), default=0), ("; md5 recorded in the json" if o.get("md5_slices") else "")))
        for sch in S7["schemas"][lab]:
            L.append("  - keys of %s: %s" % (os.path.basename(sch["path"]), ", ".join("%s %s %s" % (d["key"], d.get("shape"), d.get("dtype")) if "shape" in d else "%s (%s)" % (d["key"], d.get("error")) for d in sch["inspect"])))
    ci = S7["collar_indices"]
    L.append("- the cuts: %s; the chain-end cut from r_collar (round(r_collar) - 1 clipped to n_sites - 10): member record 0 %s, at the join %s; control %s, %s; the inside rule read off below; site radii from %s"
             % ("; ".join("%s cut index %d (the first site beyond r_c: the last site inside R %s, the first beyond R %s; the engine's 'beyond' index searchsorted(R, r_c, side='right'))" % (nm, d["index"], g(d["R_last_inside"], 6), g(d["R_at_index"], 6)) for nm, d in ci.items()),
                S7["pre"]["member"]["cut_index_record0"] if S7["pre"].get("member") else "-", S7["pre"]["member"]["cut_index_end"] if S7["pre"].get("member") else "-",
                S7["pre"]["control"]["cut_index_record0"] if S7["pre"].get("control") else "-", S7["pre"]["control"]["cut_index_end"] if S7["pre"].get("control") else "-", S7["R_source"]))
    if S7.get("stop"):
        L.append("- **STOP: %s**" % S7["stop"])
    cv = S7.get("conventions")
    if cv:
        L.append("- the member's operators at T/r_s %s: %s" % (g(cv["clock"], 6), "; ".join("%s dim %d from %s, hermiticity %s" % (f, d["dim"], d["from"], {k: g(v, 2) for k, v in d["herm"].items()}) if d else "%s not carried" % f
                                                                                      for f, d in S7["pre"]["member"]["info"].items())))
        for r in cv["rows"]:
            L.append("  - layout %s, inside %s: H_in heavy slice %s against record %s (rel %s); H_in_light slice %s (%s) against %s (rel %s)"
                     % (r["layout"], r["rule"], g(r["H_in_heavy_slice"], 10), g(r["H_in_heavy_record"], 10), g(r["res_heavy"], 3), g(r["H_in_light_slice"], 8), r["weighting"], g(r["H_in_light_record"], 8), g(r["res_light"], 3)))
        L.append("  - read off: layout %s, inside rule '%s', H_in_light = the sum over the %s channels of %s H_in_kappa; %s" % (cv["layout"], cv["rule"], cv["weighting"][1] if cv["weighting"] else "-",
                                                                                                                                   "16 kappa" if (cv["weighting"] and cv["weighting"][0] == "w") else "1 x", "REPRODUCED" if cv["H_in_reproduced"] else "NOT reproduced"))
        for r in cv["current_rows"]:
            L.append("  - current %s, sign %+d: heavy slice %s against M1_I_E %s (rel %s); light sourcing slice %s against I_E_light_sourcing %s (rel %s)"
                     % (r["form"], r["sign"], g(r["heavy_slice"], 5), g(r["heavy_record"], 5), g(r["res_heavy"], 3), g(r["light_sourcing_slice"], 5), g(r["light_sourcing_record"], 5), g(r["res_light"], 3)))
        L.append("  - read off: the current %s with sign %+d (%s); E_inv traces: %s" % (cv["current_form"], cv["current_sign"], "reproduces the stored chain-end current" if cv["current_reproduced"] else "NOT reproducing the stored current, flagged",
                                                                                           "; ".join("%s %s against %s (rel %s)" % (f, g(d["trace"], 9), g(d["record"], 9), g(d["res"], 3)) for f, d in cv["E_inv_checks"].items())))
        for f in cv["flags"]:
            L.append("  - %s" % f)
        vc = S7.get("control_verify")
        if vc:
            L.append("- the control's slice at T/r_s %s under the member's conventions: %s" % (g(vc["clock"], 6), "; ".join(
                "%s H_in rel %s, current rel %s" % (f, g(vc[f]["res_H_in"], 3), g(vc[f]["res_I"], 3)) for f in ("heavy", "light") if f in vc)))
        sw = S7.get("sweep")
        if sw:
            L.append("- the sweep of the chain-end cut from site %s to %s (%s sites), from the member's slice at %s: %s; per field (bare C, bare P_ref, referenced): %s; light combinations: %s"
                     % (sw["cut_record0"], sw["cut_at_clock"], sw["sites_swept"], g(cv["clock"], 6), sw["note"],
                        "; ".join("%s %s, %s, %s" % (f, g(d["bare_C"], 10), g(d["bare_P"], 10), g(d["referenced_X"], 6)) for f, d in sw["per_field"].items()),
                        "; ".join("%s: bare C %s against evolution_exact - integral %s (rel %s), -bare P %s against reference_reset %s (rel %s), referenced %s, fixed-cut reference remainder %s"
                                  % (k, g(d["bare_C_swept"], 10), g(sw.get("evolution_minus_integral"), 10), g(d["res_vs_evolution_minus_integral"], 3), g(-d["bare_P_swept"], 10), g(sw.get("reference_reset"), 10),
                                     g(d["res_vs_minus_reference_reset"], 3), g(d["referenced_swept"], 6), g(d["fixed_cut_reference_remainder"], 6)) for k, d in sw["light_combinations"].items()) or "none"))
        ex = S7.get("exterior")
        if ex:
            L.append("- 5d's exterior entries at the join: %s%s" % (ex["note"], "".join("; %s = %s, best %s" % (f["path"], g(f["record_value"], 6), f["best_match"]) for f in ex["found"][:12])))
        rc = S7.get("record_collar_entries")
        L.append("- the record's s11 entries named 'collar' at the join (information, for per-record collar currents if stored): %s" % ("; ".join("%s = %s" % (d["path"], d["value"]) for d in rc[:30]) if rc else "none"))
    # ---- the ledger
    L += ["", "## 3. The collar ledger", ""]
    if ctx["ledger_stopped"]:
        L += ["**THE LEDGER IS NOT WRITTEN.** %s" % ctx["ledger_stopped"], ""]
    else:
        LG = ctx["ledger"]
        cv = S7["conventions"]
        form = cv["current_form"]
        L += ["Conventions: layout %s, inside '%s', the light weighting %s; the current column is this readout's %s, inward-positive (the stored current's sign %+d read off at the chain end); the steps: light %s r_s, heavy %s r_s; "
              "the integral of (c) by the trapezoid of the slice-clock currents in T (lattice units), work, ref and memb cumulative from %s; closure = dH_in - (integral + work + ref + memb); W/2g = M(r_c) from the s11 npz (node spacing %s)."
              % (cv["layout"], cv["rule"], cv["weighting"], form, cv["current_sign"], LG["steps"]["light"], LG["steps"]["heavy"], g(o["window_start"], 4), g(LG.get("node_spacing"), 4)),
              "The heavy field is the weighted sum over the channels %s (weights 16 kappa: %s; the per-channel unweighted columns are in the json and the npz under H<kappa>). The work and "
              "reference columns are the records' split (work = the inside sum of Re[(h' - h)(C' - P')], ref = -(the inside sum of Re[h (P' - P)]): engine_twofield_v3's field_work and "
              "reference_reset). The slice clocks read per field (member): %s. Copies of one clock in more than one directory dropped (read once): %d."
              % (", ".join(LG["heavy_channels"]), ", ".join("%s %g" % (k_, v_) for k_, v_ in LG["heavy_weights"].items()),
                 "; ".join("%s %d (%s to %s)" % (f_, d_["n_slices"], g(d_["clocks"][0], 6) if d_["clocks"] else "-", g(d_["clocks"][-1], 6) if d_["clocks"] else "-") for f_, d_ in LG["member"].items()),
                 sum(len(d_.get("duplicates_dropped") or []) for lab_ in ("member", "control") for d_ in LG[lab_].values())), ""]
        for cut in LG["cuts"]:
            L += ["### 3.%d %s (site index: member %s, control %s, cut identical at the matched clocks: %s)" % (list(LG["cuts"]).index(cut) + 1, CUT_LABEL.get(cut, cut), LG["cut_sites"][cut]["member"], LG["cut_sites"][cut]["control"], LG["cut_sites"][cut]["identical"]), ""]
            for grp in ("heavy", "light_sourcing", "light_all"):
                Am = LG["acc"]["member"].get(grp, {}).get(cut)
                if Am is None:
                    L += ["(%s: not carried by the slices)" % grp, ""]
                    continue
                Ak = LG["acc"]["control"].get(grp, {}).get(cut)
                Dd = LG["diff"].get(grp, {}).get(cut)
                gM = LG["charges"]["member"].get(cut, {}) if LG["charges"].get("member") else None
                gK = LG["charges"]["control"].get(cut, {}) if LG["charges"].get("control") else None
                L += ["%s:" % {"heavy": "the heavy field", "light_sourcing": "the light sourcing channels (kappa 1, 2, weighted 16 kappa)", "light_all": "the light channels 1, 2, 3 (weighted)"}[grp], "",
                      tab_cut(cut, Am, Ak, Dd, form, gM, gK), ""]
                tele = max((abs(x) for x in Am["step_telescoping_residual"] if x is not None), default=0.0)
                clo = Am["cum_closure_" + form.split("_")[1]][-1] if Am["cum_closure_" + form.split("_")[1]] else None
                L.append("telescoping residual at most %s; closure at the last clock %s against dH_in %s and the integral %s; members changed over the window %s." %
                         (g(tele, 2), g(clo, 4), g(Am["dH_in_from_start"][-1], 6), g(Am["cum_int_" + form][-1], 5), sum(1 for v in Am["step_memb"] if v not in (None, 0.0))))
                L.append("")
            # per channel unweighted at the table clocks
            rows = []
            for cl in ctx["table_clocks_slices"]:
                row = [g(cl, 6)]
                for ch in LG["channels"]:
                    for lab in ("member", "control"):
                        A = LG["acc"][lab].get(ch, {}).get(cut)
                        i = A["clock"].index(cl) if (A and cl in A["clock"]) else None
                        row += [g(A["H_out"][i], 6) if i is not None else "-", g(A[form][i], 4) if i is not None else "-"]
                rows.append(row)
            head = ["T/r_s"] + ["%s %s %s" % (ch, lab, q) for ch in LG["channels"] for lab in ("C", "ctrl") for q in ("H_out", "I_E")]
            L += ["per channel, unweighted (as 5d's exterior entries are), at the table clocks:", "", mdtab(head, rows), ""]
        if LG.get("chain_end_budget"):
            cb = LG["chain_end_budget"]
            L += ["### 3.%d The chain end over the continuation: the records' own budget beside the ledger" % (len(LG["cuts"]) + 1), "",
                  "The chain-end cut is fixed over the continuation (site %s at every clock), so (e) is zero and the records' evolution_exact over 40 to T is the engine's own fixed-cut term, to be matched by the ledger's integral of (c) "
                  "to the resolution; the records' field_work and reference_reset over the same window beside the ledger's work and ref; the light combination %s matches the records' work and reference best (residual %s)."
                  % (LG["cut_sites"][CHAIN_END]["member"], cb["combination"], g(cb["combination_residual"], 3)), "",
                  mdtab(["T/r_s", "evolution_exact (records)", "integral of I_E (ledger)", "evolution term (ledger)", "field_work (records)", "work (ledger)", "reference_reset (records)", "ref (ledger)", "dH_in_light (records)", "dH_in (ledger)"],
                        [[g(r["clock"], 6), g(r["evo_rec"], 8), g(r["int_led"], 8), g(r["evol_led"], 8), g(r["work_rec"], 8), g(r["work_led"], 8), g(r["ref_rec"], 8), g(r["ref_led"], 8), g(r["dH_rec"], 8), g(r["dH_led"], 8)] for r in cb["rows"]]), ""]
    # ---- the per-record columns
    L += ["## 4. The per-record columns continued", ""]
    R = ctx["R"]
    TrsC = JM["S"]["T_over_rs"]
    rows = V2.rows_for(TrsC, [], TABLE_CLOCKS)
    D = R["D"]
    L += ["4a, the roots, the trapping scan, E:", "", build_table(rows, [("R_t C", "C.R_t", 9), ("ctrl", "K.R_t", 9), ("d", "d.R_t", 4), ("r_in C", "C.r_in", 9), ("ctrl", "K.r_in", 9), ("kappa_T C", "C.kappa_T", 5),
                                                                       ("min(N-|v|) C", "C.min_N_minus_absv", 6), ("ctrl", "K.min_N_minus_absv", 6), ("d", "d.min_N_minus_absv", 3), ("E C", "C.E", 11), ("ctrl", "K.E", 11),
                                                                       ("dE", "d.E", 4), ("dE/E", "rel.E", 3), ("E/E0 C", "C.E_over_E0", 6)], D), "",
          "4b, E_inv per field, the chain-end currents and their integrals continued (the quadrature read off in 2.6), the 5d entries where found:", "",
          build_table(rows, [("E_inv_heavy C", "C.light.E_inv_heavy", 9), ("ctrl", "K.light.E_inv_heavy", 9), ("d", "d.light.E_inv_heavy", 4), ("E_inv_light C", "C.light.E_inv_light", 6), ("ctrl", "K.light.E_inv_light", 6),
                             ("rel", "rel.light.E_inv_light", 4), ("E_inv_src/E C", "C.light.E_inv_light_sourcing_over_E", 5), ("ctrl", "K.light.E_inv_light_sourcing_over_E", 5), ("M1_I_E C", "C.M1_I_E", 4), ("ctrl", "K.M1_I_E", 4),
                             ("int M1_I_E C", "C.int_M1_I_E", 5), ("ctrl", "K.int_M1_I_E", 5), ("I_E_light_src C", "C.light.I_E_light_sourcing", 4), ("ctrl", "K.light.I_E_light_sourcing", 4),
                             ("int I_E_light_src C", "C.int_I_E_light_sourcing", 5), ("ctrl", "K.int_I_E_light_sourcing", 5)] + [("5d %s C" % p, "C.ext.%s" % p, 4) for p in R["ext_paths"]], D), "",
          "4c, the truncation table's ratios (guard over sourcing: E_inv, the cut current, the shell source; the definitions of 2.6), the mass function and the solver:", "",
          build_table(rows, [("E_inv ratio C", "C.tr.E_inv", 6), ("ctrl", "K.tr.E_inv", 6), ("I_E ratio C", "C.tr.I_E", 6), ("ctrl", "K.tr.I_E", 6), ("S ratio C", "C.tr.S", 6), ("ctrl", "K.tr.S", 6),
                             ("M_min C", "C.s11.M_min_outside_cell", 8), ("ctrl", "K.s11.M_min_outside_cell", 8), ("d", "d.s11.M_min_outside_cell", 4), ("neg pts C", "C.s11.negative_points_outside_cell", 1),
                             ("ctrl", "K.s11.negative_points_outside_cell", 1), ("iterations C", "C.iterations", 1), ("ctrl", "K.iterations", 1), ("kept residual C", "C.solve.kept_residual", 3), ("ctrl", "K.solve.kept_residual", 3),
                             ("M(16 r_s) C", "C.M_r16", 9), ("ctrl", "K.M_r16", 9), ("M(24 r_s) C", "C.M_r24", 9), ("ctrl", "K.M_r24", 9), ("M(end) C", "C.M_end", 9), ("ctrl", "K.M_end", 9)], D), ""]
    f = R["summary"]["formation"]
    L.append("Roots from 40 on: member %d records with a dense root, control %d (none from 19.375 to 40 in the delivered histories; a reappearance is a fact of the records, printed as such); the formation record of both "
             "%s; matched records %d." % (R["roots_after_40"]["member"], R["roots_after_40"]["control"], g(f["control"]["T_over_rs"], 6) if f["control"] else "-", R["matched"]))
    L += ["", "## 5. The identities of this version", "",
          mdtab(["identity", "kind", "n", "max |residual|", "note"], [[nm, ctx["ids"].d[nm]["kind"], str(ctx["ids"].d[nm]["n"]), fm(ctx["ids"].d[nm]["max_abs_residual"], 3), ctx["ids"].d[nm]["note"]] for nm in ctx["ids"].order])]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------------- run
def summary_lines(ctx):
    P = ctx["P"]
    L = []
    for lab in ("member", "control"):
        rep = P["P1_join"][lab]
        jr = rep["join_record"]
        if jr is None:
            L.append("P1 join %s: the continuation json carries no record at T/r_s %s (FLAG)" % (lab, fm(rep["delivered_last_T_over_rs"], 6)))
        else:
            L.append("P1 join %s at T/r_s %s: %s (%d leaves differ); shared %d records, %d bitwise; appended %d records to T/r_s %s; gaps %d"
                     % (lab, fm(jr["T_over_rs"], 6), "bitwise" if jr["bitwise"] else "NOT bitwise", jr["leaves_differing"], rep["overlap_records"], rep["overlap_bitwise"], rep["appended_records"], fm(rep["joined_last_T_over_rs"], 6), len(rep["gaps"])))
    L.append("P2 args: %s" % "; ".join("%s %s" % (k, "identical" if not v["differences"] else ("%d differ%s" % (len(v["differences"]), ", UNEXPECTED " + ",".join(v["unexpected"]) if v["unexpected"] else ""))) for k, v in P["P2_args"].items()))
    for lab in ("member", "control"):
        s = P["P4_solver"][lab]
        L.append("P4 %s continuation: iterations at most %s, kept residual at most %s (T/r_s %s), fallbacks %d, converged false %d" % (lab, fm(s["iterations_max"], 1), fm(s["kept_residual_max"], 3), fm(s["T_at_kept_residual_max"], 6), s["fallbacks"], s["converged_false"]))
    w = P["P5_window"]
    L.append("P5 window: matched %s to %s (%d records), member-only to %s" % (fm(w["window_start"], 4), fm(w["matched_end"], 6), w["matched_records"], fm(w["member_last"], 6)))
    B = P["P6_closure"]
    if B["reproduced"]:
        L.append("P6 readout 2's triple REPRODUCED (%s): evolution_exact %s, field_work %s, reference_reset %s, sum %s, dH_in_light %s (%s); integrals %s"
                 % (B["mode"], fm(B["chosen"]["evolution_exact"]["value"], 9), fm(B["chosen"]["field_work"]["value"], 6), fm(B["chosen"]["reference_reset"]["value"], 9), fm(B["triple_sum"], 6), fm(B["dH_in_light"], 6),
                    "reproduced" if B["dH_in_light_reproduced"] else "NOT reproduced", B["integrals"]["note"]))
    else:
        L.append("P6 readout 2's triple NOT REPRODUCED: paths %s" % B["paths_found"])
    L.append("P6 E(0) %s, E(40) %s, fall %s, E/E0 %s" % (fm(B["E"]["E0"], 9), fm(B["E"]["E40"], 9), fm(B["E"]["fall"], 7), fm(B["E"]["E_over_E0_40"], 5)))
    S7 = P["P7_slices"]
    cv = S7.get("conventions")
    if cv:
        L.append("P7 conventions: layout %s, inside '%s', weighting %s, current %s sign %+d; H_in %s, current %s%s"
                 % (cv["layout"], cv["rule"], cv["weighting"], cv["current_form"], cv["current_sign"], "REPRODUCED" if cv["H_in_reproduced"] else "NOT REPRODUCED",
                    "reproduced" if cv["current_reproduced"] else "not reproduced (flag)", "; " + "; ".join(cv["flags"]) if cv["flags"] else ""))
    if S7.get("stop"):
        L.append("P7 STOP: %s" % S7["stop"])
    L.append("LEDGER NOT WRITTEN: %s" % ctx["ledger_stopped"] if ctx["ledger_stopped"] else "LEDGER WRITTEN")
    return L


def run_v3(o):
    t0 = time.time()
    log = (lambda s: None) if o.get("quiet") else (lambda s: print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), s), file=sys.stderr))
    inputs = []
    for role in ("run_json", "run_cont_json", "control_json", "control_cont_json", "run_cont_fields_npz", "control_cont_fields_npz", "run_cont_s11_npz", "control_cont_s11_npz", "v2_json"):
        p = o.get(role)
        if p is None:
            continue
        if not os.path.isfile(p):
            raise ReadoutError("%s: not a file: %s" % (role, p))
        inputs.append({"role": role, "path": os.path.abspath(p), "md5": md5_of(p), "bytes": os.path.getsize(p)})
    ids = Identities()
    mem, memc = load_run(o["run_json"], "run"), load_run(o["run_cont_json"], "run_cont")
    ctl, ctlc = load_run(o["control_json"], "control"), load_run(o["control_cont_json"], "control_cont")
    for r, want in ((mem, "on"), (memc, "on"), (ctl, "off"), (ctlc, "off")):
        if str(r["m"]["coupling"]) != want:
            raise Inconsistent("%s: args.light_coupling is %r, not %r" % (r["where"], r["m"]["coupling"], want))
    JM, JK = join_runs(mem, memc, "member"), join_runs(ctl, ctlc, "control")
    chans = [str(c) for c in JM["m"]["channels"]]
    if [str(c) for c in JK["m"]["channels"]] != chans:
        raise Inconsistent("the light channels differ: member %s, control %s" % (chans, JK["m"]["channels"]))
    P = {"P1_join": {"member": JM["report"], "control": JK["report"]}}
    # ---- P2 (v3b: the end clock T_over_rs, the run's length argument, is expected to differ between a delivered run and its continuation and between the two
    #      continuations, the control's 50 being plan #162's limit; the resume entries are skipped by V1.args_diff)
    P2 = {}
    for key, a, b, expected in (("member_delivered_vs_continued", mem["args"], memc["args"], "the resume and the end clock"), ("control_delivered_vs_continued", ctl["args"], ctlc["args"], "the resume and the end clock"),
                                ("member_vs_control_continued", memc["args"], ctlc["args"], "light_coupling and the end clock"), ("member_vs_control_delivered", mem["args"], ctl["args"], "light_coupling")):
        diffs = [list(x) for x in V1.args_diff(a, b)]
        if expected.startswith("light_coupling"):
            unexpected = [k for k, _, _ in diffs if k not in V2.EXPECTED_ARG_DIFF and not ("end clock" in expected and EXPECTED_CONT_ARG.search(str(k)))]
        else:
            unexpected = [k for k, _, _ in diffs if not EXPECTED_CONT_ARG.search(str(k))]
        P2[key] = {"differences": diffs, "unexpected": unexpected, "expected": expected}
    P["P2_args"] = P2
    bad = [k for k, v in P2.items() if v["unexpected"]]
    if bad and not o.get("allow_arg_mismatch"):
        raise Inconsistent("the argument lists differ beyond the expected entries: %s (pass --allow-arg-mismatch to proceed with the differences printed)"
                           % "; ".join("%s: %s" % (k, ", ".join("%s (%s against %s)" % (kk, fm(va, 6), fm(vb, 6)) for kk, va, vb in P2[k]["differences"] if kk in P2[k]["unexpected"])) for k in bad))
    # ---- P3, P4, P5
    P["P3_engine"] = {lab: {"script_delivered": J["script_delivered"], "script_continued": J["script_continued"], "version_keys": J["version_keys"]} for lab, J in (("member", JM), ("control", JK))}
    P["P4_solver"] = {"member": solver_stats(JM, chans, JM["n_delivered"]), "control": solver_stats(JK, chans, JK["n_delivered"]),
                      "member_delivered_recomputed": solver_stats({"S": {k: v[:JM["n_delivered"]] for k, v in JM["S"].items()}}, chans, 0),
                      "control_delivered_recomputed": solver_stats({"S": {k: v[:JK["n_delivered"]] for k, v in JK["S"].items()}}, chans, 0)}
    TM, TK = JM["S"]["T_over_rs"], JK["S"]["T_over_rs"]
    iK = align(TM, TK)
    pairs = [(k, j) for k, j in enumerate(iK) if j is not None]
    rev = {j: k for k, j in pairs}
    matched_end = TK[-1]
    kw0 = find_clock(TM, o["window_start"])
    if kw0 is None:
        raise Inconsistent("the member has no record at the window's start T/r_s %g" % o["window_start"])
    P["P5_window"] = {"member_last": TM[-1], "control_last": matched_end, "window_start": o["window_start"], "matched_end": matched_end,
                      "matched_records": sum(1 for k, j in pairs if TM[k] >= o["window_start"] - 1e-9), "matched_slice_clocks_light": None, "matched_slice_clocks_heavy": None}
    ids.add("member, control: tau, rs, ell, g, nsites equal", [abs(float(JM["m"][q]) - float(JK["m"][q])) for q in ("tau", "rs", "ell", "g")] + [abs(float(JM["m"]["nsites"] or 0) - float(JK["m"]["nsites"] or 0))])
    ids.add("member, control: T_over_rs equal at the matched clocks", [abs(TM[k] - TK[j]) for k, j in pairs])
    ids.add("member, control: r_collar equal at the matched clocks (the chain-end cut shared, continuation included)", [abs(float(JM["S"]["r_collar"][k]) - float(JK["S"]["r_collar"][j])) for k, j in pairs
                                                                                                                      if num(JM["S"]["r_collar"][k]) is not None and num(JK["S"]["r_collar"][j]) is not None], kind="cross")
    # ---- the per-record part (R), continued from version 2
    D, F, Asum = V2.compute_a(_S(JM["S"]), JM["m"], _S(JK["S"]), JK["m"], iK, rev, pairs, None)
    k0, k40 = find_clock(TM, 0.0), find_clock(TM, 40.0)
    kf = F["control"]["k_C"] if F["control"] else None
    if k0 is None or k40 is None:
        raise Inconsistent("the member has no record at T/r_s 0 or 40")
    integ = read_off_integral(JM["S"], k0, k40, float(JM["m"]["rs"]))
    quad = integ["chosen"] or "trapezoid_T"
    for key, nm in (("M1_I_E", "int_M1_I_E"), ("light.I_E_light_sourcing", "int_I_E_light_sourcing")):
        D["C." + nm] = cumulative_integral_series(JM["S"], key, quad)
        colK = cumulative_integral_series(JK["S"], key, quad)
        D["K." + nm] = [colK[j] if j is not None else None for j in iK]
    for pre, S_ in (("C.", JM["S"]), ("K.", JK["S"])):
        n_ = len(S_["T_over_rs"])
        tr_ = [truncation_candidates(S_, k, JM["m"]) for k in range(n_)]
        trdef = truncation_read_off(JM["S"], JM["m"], kf, k40)["chosen"]
        for grp in ("E_inv", "I_E", "S"):
            dname = trdef[grp]["definition"]
            col = [t.get(dname) if dname else None for t in tr_]
            D[pre + "tr." + grp] = col if pre == "C." else [col[j] if j is not None else None for j in iK]
        for q in ("iterations", "solve.kept_residual"):
            D[pre + q] = list(S_[q]) if pre == "C." else [S_[q][j] if j is not None else None for j in iK]
    trunc = truncation_read_off(JM["S"], JM["m"], kf, k40)
    # 5d entries by name (information in the per-record table)
    ext_paths = []
    kjoin = find_clock(TM, o["window_start"])
    for p, v in all_leaves(JM["records"][kjoin]):
        if EXT_REGEX.search(dotted(p)) and num(v) is not None:
            ext_paths.append(dotted(p))
    if o.get("exterior_keys"):
        ext_paths = [str(x) for x in (o["exterior_keys"].values() if isinstance(o["exterior_keys"], dict) else o["exterior_keys"])]
    ext_paths = ext_paths[:6]
    for p in ext_paths:
        D["C.ext.%s" % p] = [num(get_path(r, undot(p))) for r in JM["records"]]
        colK = [num(get_path(r, undot(p))) for r in JK["records"]]
        D["K.ext.%s" % p] = [colK[j] if j is not None else None for j in iK]
    roots_after = {"member": sum(1 for k in range(k40, len(TM)) if num(JM["S"]["R_t"][k]) is not None), "control": sum(1 for j in range(find_clock(TK, 40.0) or 0, len(TK)) if num(JK["S"]["R_t"][j]) is not None)}
    R = {"D": D, "formation": F, "summary": Asum, "ext_paths": ext_paths, "roots_after_40": roots_after, "matched": len(pairs), "quadrature": quad}
    # ---- P6
    B = budget_closure(mem["records"], {k: v[:JM["n_delivered"]] for k, v in JM["S"].items()}, READOUT2, o.get("budget_keys"), k0, k40, "member 0 to 40")
    B["integrals"] = integ
    E0, E40 = num(JM["S"]["E"][k0]), num(JM["S"]["E"][k40])
    B["E"] = {"E0": E0, "E40": E40, "fall": (E0 - E40) if (E0 is not None and E40 is not None) else None, "E_over_E0_40": num(JM["S"]["E_over_E0"][k40]),
              "fall_vs_md": rel_res((E0 - E40) if (E0 is not None and E40 is not None) else None, MD_FALL), "E0_vs_md": rel_res(E0, MD_E0)}
    B["md_at_40"] = {k: {"value": D[k][k40] if k in D else None, "md": v, "rel": rel_res(D[k][k40] if k in D else None, v)} for k, v in MD_AT_40.items()}
    B["truncation"] = trunc
    try:
        Bc = budget_closure(ctl["records"], {k: v[:JK["n_delivered"]] for k, v in JK["S"].items()}, {}, o.get("budget_keys"), 0, JK["n_delivered"] - 1, "control 0 to 40")
        if B["reproduced"]:
            B["control_triple"] = budget_window(ctl["records"], B["chosen"], B["mode"], 0, JK["n_delivered"] - 1)
        else:
            B["control_triple"] = None
        B["control_paths"] = Bc["paths_found"]
    except Exception as e:  # noqa: BLE001  information only
        B["control_triple"] = None
        B["control_error"] = "%s: %s" % (type(e).__name__, e)
    if o.get("v2_json"):
        try:
            v2 = load_json(o["v2_json"])
            ser = v2.get("a", {}).get("series", {})
            res = []
            for key in ("C.E", "K.E", "C.R_t", "C.light.E_inv_light", "K.light.E_inv_light", "C.light.H_in_light", "C.M1_I_E"):
                if key in ser and key in D and len(ser[key]) > k40:
                    a, b = num(ser[key][k40]), num(D[key][k40])
                    res.append(0.0 if (a == b) else (abs((a or 0.0) - (b or 0.0))))
            ids.add("version 2's delivered json: the series at T/r_s 40 equal the delivered records re-extracted here", res, kind="cross")
            B["v2_json_check"] = "%d series compared at 40, max |difference| %s" % (len(res), fm(max(res) if res else None, 3))
        except Exception as e:  # noqa: BLE001
            B["v2_json_check"] = "not compared: %s: %s" % (type(e).__name__, e)
    P["P6_closure"] = B
    ids.add("readout 2's triple from the records (mode %s)" % B["mode"], [abs(B["chosen"][nm]["value"] - READOUT2[nm][0]) for nm in BUDGET_NAMES] if B["reproduced"] else [math.inf], kind="cross",
            note="against the printed -110196, 0.00753, +110196 (tolerances 0.5, 5e-6, 0.5)")
    if B.get("closure_residual") is not None:
        ids.add("the budget's three terms sum to dH_in_light over 0 to 40", [abs(B["closure_residual"])], kind="relation")
    ids.add("E(0) - E(40) = 0.563327 (readout 2), E(0) = 59.788989", [abs(B["E"]["fall"] - MD_FALL) if B["E"]["fall"] is not None else math.inf, abs(E0 - MD_E0) if E0 is not None else math.inf], kind="cross")
    # ---- P7: the slices
    rs = float(JM["m"]["rs"])
    nsites = int(JM["m"]["nsites"])
    listing = {"member": list_slices(o["slices"], rs, o["slice_clock_regex"], o.get("md5_slices")), "control": list_slices(o["control_slices"], rs, o["slice_clock_regex"], o.get("md5_slices"))}
    schemas = {}
    for lab, ls in listing.items():
        seen, out = set(), []
        for s in ls:
            if s["keys"] is None:
                continue
            sig = tuple(sorted(s["keys"]))
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"path": s["path"], "inspect": inspect_slice(s["path"], s["keys"])})
            if len(out) >= 3:
                break
        schemas[lab] = out
    S7 = {"listing": {lab: [{k: v for k, v in s.items()} for s in ls] for lab, ls in listing.items()}, "schemas": schemas, "collar_indices": {}, "R_source": None, "pre": {}, "stop": None}
    P["P7_slices"] = S7
    if o.get("inspect_slices"):
        print(json.dumps({"listing": S7["listing"], "schemas": schemas}, indent=1, default=_jsonable))
        return 0
    # the site radii
    Rsites, Rsrc = None, None
    for role in ("run_cont_fields_npz", "control_cont_fields_npz"):
        if o.get(role):
            z = np.load(o[role], allow_pickle=False)
            if "R" in z.files:
                Rsites, Rsrc = np.asarray(z["R"], dtype=float).ravel(), role + ":R"
            z.close()
            if Rsites is not None:
                break
    if Rsites is None:
        for s in listing["member"]:
            if s["keys"]:
                for key in R_KEYS:
                    if key in s["keys"]:
                        z = np.load(s["path"], allow_pickle=False)
                        Rsites, Rsrc = np.asarray(z[key], dtype=float).ravel(), os.path.basename(s["path"]) + ":" + key
                        z.close()
                        break
            if Rsites is not None:
                break
    if Rsites is None:
        raise Inconsistent("the site radii are not among the inputs (no fields npz with R given, no slice carrying %s); the collar sites cannot be formed" % (R_KEYS,))
    if Rsites.size != nsites:
        raise Inconsistent("the site radii have %d entries against args.nsites %d" % (Rsites.size, nsites))
    S7["R_source"] = Rsrc
    cuts = list(COLLARS) + ([COLLAR_8] if o.get("collar_8") else [])
    collar_idx = {}
    for name, rsm in cuts:
        d = collar_site_index(Rsites, rsm * rs)
        S7["collar_indices"][name] = d
        collar_idx[name] = d["index"]
    heavy_kappas = heavy_kappa_list(o, JM, JK)
    fields = [("H%d" % k, "heavy", k, role_keys(o, "heavy", k)) for k in heavy_kappas] + [("L%s" % c, "light", c, role_keys(o, "light", c)) for c in chans]
    wH_all = {"H%d" % k: 16.0 * k for k in heavy_kappas}  # engine_collapse_v5.Chan: self.w = 16.0 * kappa
    builder = load_builder(o.get("h_builder"), o.get("engine_dir"), JM["args_continued"], JK["args_continued"])

    def slice_at(ls, clock):
        c = [s for s in ls if s["clock"] is not None and abs(s["clock"] - clock) < 1e-9 and s["keys"]]
        c.sort(key=lambda s: -len(s["keys"]))
        return c[0] if c else None
    s40m, s40k = slice_at(listing["member"], o["window_start"]), slice_at(listing["control"], o["window_start"])
    ledger_stopped = None
    conv = None
    try:
        if s40m is None:
            raise Inconsistent("the member has no slice at the window's start T/r_s %g (clocks present: %s)" % (o["window_start"], sorted(set(s["clock"] for s in listing["member"] if s["clock"] is not None))[:60]))
        pre_m = prepass("member", JM, s40m, fields, builder, collar_idx, o, nsites)
        S7["pre"]["member"] = {k: v for k, v in pre_m.items() if k != "per"}
        S7["heavy_weights_check"] = heavy_weights_check(s40m, heavy_kappas, ids)
        conv = decide_conventions(pre_m, JM, JM["m"])
        S7["conventions"] = conv
        S7["sweep"] = sweep_estimate(pre_m, conv, JM["m"], B, integ["chosen"], JM["S"])
        S7["exterior"] = exterior_check(pre_m, conv, JM, JM["m"], o.get("exterior_keys"))
        S7["record_collar_entries"] = record_collar_entries(JM, pre_m["k"])
        if s40k is None:
            S7["control_verify"] = None
            conv["flags"].append("FLAG: the control has no slice at the window's start; the control's ledger starts at its first slice at or after %g" % o["window_start"])
        else:
            pre_k = prepass("control", JK, s40k, fields, builder, collar_idx, o, nsites)
            S7["pre"]["control"] = {k: v for k, v in pre_k.items() if k != "per"}
            S7["control_verify"] = verify_conventions(pre_k, JK, JK["m"], conv)
        for f, d in conv["E_inv_checks"].items():
            if d["res"] is not None:
                ids.add("slice at the join: the chain trace of h(C - P_ref) = the record's E_inv (%s)" % f, [d["res"]], kind="hypothesis", note="relative")
        ids.add("slice at the join: H_in at the chain-end cut = the record's H_in, H_in_light (layout %s, inside %s, weighting %s)" % (conv["layout"], conv["rule"], conv["weighting"]), [conv["score"]], kind="hypothesis", note="relative")
        ids.add("slice at the join: the cut current (%s, sign %+d) = the record's chain-end current" % (conv["current_form"], conv["current_sign"]), [conv["current_score"]], kind="cross", note="relative; a flag, not a stop")
        if not conv["H_in_reproduced"]:
            S7["stop"] = conv["flags"][0]
    except ReadoutError as e:
        S7["stop"] = "%s: %s" % (type(e).__name__, e)
        S7["traceback"] = traceback.format_exc()
    if not B["reproduced"]:
        ledger_stopped = "P6: readout 2's triple is not reproduced from the delivered records (the collar ledger is written only after the chain-end closure over 0 to 40 reproduces it)"
    elif S7["stop"]:
        ledger_stopped = "P7: " + S7["stop"]
    # the slice clocks of the window for P5
    for lab, ls in listing.items():
        for grp, roles in (("light", role_keys(o, "light", chans[0])), ("heavy", role_keys(o, "heavy", heavy_kappas[0]))):
            cl = sorted(set(s["clock"] for s in ls if s["clock"] is not None and s["clock"] >= o["window_start"] - 1e-9 and carries(s["keys"], roles)))
            P["P5_window"]["%s_slice_clocks_%s" % (lab, grp)] = cl
    P["P5_window"]["matched_slice_clocks_light"] = len(set(P["P5_window"].get("member_slice_clocks_light", [])) & set(P["P5_window"].get("control_slice_clocks_light", [])))
    P["P5_window"]["matched_slice_clocks_heavy"] = len(set(P["P5_window"].get("member_slice_clocks_heavy", [])) & set(P["P5_window"].get("control_slice_clocks_heavy", [])))
    # ---- the ledger
    LG = None
    table_clocks_slices = []
    if ledger_stopped is None:
        LM = run_ledger("member", JM, listing["member"], fields, builder, conv, collar_idx, o, log)
        LK = run_ledger("control", JK, listing["control"], fields, builder, conv, collar_idx, o, log)
        cut_names = list(collar_idx) + [CHAIN_END]
        w = {("L%s" % c): conv["weights"][c] for c in chans}
        w.update(wH_all)
        acc = {"member": {}, "control": {}}
        for lab, LL in (("member", LM), ("control", LK)):
            for flabel in [f[0] for f in fields]:
                rows = LL["fields"][flabel]["rows"]
                acc[lab][flabel] = {cut: accumulate(rows, cut) for cut in cut_names} if rows else None
            src = ["L%s" % c for c in JM["m"]["sourcing"]]
            allc = ["L%s" % c for c in chans]
            for grp, labs in (("heavy", list(wH_all)), ("light_sourcing", src), ("light_all", allc)):
                acc[lab][grp] = {}
                for cut in cut_names:
                    byc = {f: (acc[lab][f][cut] if acc[lab].get(f) else None) for f in labs}
                    acc[lab][grp][cut] = combine(byc, w) if all(v is not None for v in byc.values()) and all(w[f] is not None for f in labs) else None
        diff = {}
        for grp in acc["member"]:
            diff[grp] = {}
            for cut in cut_names:
                Am = acc["member"][grp].get(cut) if acc["member"].get(grp) else None
                Ak = acc["control"][grp].get(cut) if acc["control"].get(grp) else None
                diff[grp][cut] = difference(Am, Ak)
        cut_sites = {}
        for cut in cut_names:
            im = sorted(set(r["cut_index"][cut] for f in LM["fields"].values() for r in f["rows"]))
            ik = sorted(set(r["cut_index"][cut] for f in LK["fields"].values() for r in f["rows"]))
            same = []
            for f in LM["fields"]:
                pk = {r["clock"]: r["cut_index"][cut] for r in LK["fields"][f]["rows"]}
                same += [r["cut_index"][cut] == pk[r["clock"]] for r in LM["fields"][f]["rows"] if r["clock"] in pk]
            cut_sites[cut] = {"member": im, "control": ik, "identical": (all(same) if same else None)}
        ids.add("the cut's site index identical in both runs at the matched slice clocks", [0.0 if cut_sites[c]["identical"] in (True, None) else 1.0 for c in cut_names], kind="cross")
        tele = [abs(x) for grp in [f[0] for f in fields] for cut in cut_names for lab in ("member", "control") if acc[lab].get(grp) for x in acc[lab][grp][cut]["step_telescoping_residual"] if x is not None]
        ids.add("the ledger's telescoping: dH_in = evolution + work + reference + membership per step (absolute)", tele, kind="relation")
        herm = [v for lab, LL in (("member", LM), ("control", LK)) for f in LL["fields"].values() for r in f["rows"] for v in r["herm"].values()]
        ids.add("the slices' operators hermitian (max |A - A^dagger| / max |A|)", herm, kind="cross")
        orth = [r["orth_P"] for LL in (LM, LK) for f in LL["fields"].values() for r in f["rows"] if r.get("orth_P") is not None]
        if orth:
            ids.add("the slices' reference columns orthonormal (max |R2^dagger R2 - 1|; P_ref = R2 R2^dagger a projector)", orth, kind="cross")
        charges, node_spacing = {}, None
        gcoef = float(JM["m"]["g"])
        cuts_r = {name: rsm * rs for name, rsm in cuts}
        cuts_r[CHAIN_END] = None
        for lab, J, role in (("member", JM, "run_cont_s11_npz"), ("control", JK, "control_cont_s11_npz")):
            charges[lab] = None
            if o.get(role):
                try:
                    Z = V2.load_s11(o[role], role)
                    ch = collar_charges(Z, J["S"], gcoef, cuts_r)
                    node_spacing = ch["node_spacing_record0"]
                    ids.add("%s s11 npz: the r_nodes rows identical (the coarse grid is fixed; %d rows against %d records, the restart's copy skipping r_nodes)" % (lab, ch["r_nodes_rows"], len(J["S"]["T"])),
                            [ch["r_nodes_max_dev"]], kind="cross")
                    charges[lab] = {cut: {J["S"]["T_over_rs"][k]: ch["M"][cut][k] for k in range(len(J["S"]["T_over_rs"])) if ch["M"][cut][k] is not None} for cut in cuts_r}
                    for cut in cuts_r:
                        pre = "C." if lab == "member" else "K."
                        col = ch["M"][cut]
                        D[pre + "M_" + cut] = col if lab == "member" else [col[j] if j is not None else None for j in iK]
                    Eser = J["S"]["E"]
                    ids.add("%s s11 npz: M at the chain's last node = E (the identity E = W/2g at the chain end)" % lab, [abs(ch["M"][CHAIN_END][k] - num(Eser[k])) / abs(num(Eser[k])) for k in range(len(Eser))
                                                                                                                       if ch["M"][CHAIN_END][k] is not None and num(Eser[k]) not in (None, 0.0)], kind="cross", note="relative")
                    if ch["W_over_2g"] is not None:
                        ids.add("%s s11 npz: W/2g = M at the collars" % lab, [abs(ch["W_over_2g"][c][k] - ch["M"][c][k]) for c in cuts_r for k in range(len(Eser)) if ch["M"][c][k] is not None and ch["W_over_2g"][c][k] is not None], kind="cross")
                except ReadoutError as e:
                    charges[lab] = None
                    ids.add("%s s11 npz: (g) not computed (%s)" % (lab, e), [], kind="cross")
        # the chain end over the continuation: the records' budget beside the ledger
        cb = None
        if B["reproduced"] and acc["member"].get("light_sourcing") and acc["member"]["light_sourcing"].get(CHAIN_END):
            form = conv["current_form"]
            best = None
            for grp in ("light_sourcing", "light_all"):
                A = acc["member"][grp].get(CHAIN_END) if acc["member"].get(grp) else None
                if A is None:
                    continue
                rows = []
                for i, cl in enumerate(A["clock"]):
                    k1 = find_clock(TM, cl)
                    if k1 is None:
                        continue
                    bw = budget_window(JM["records"], B["chosen"], B["mode"], kw0, k1)
                    rows.append({"clock": cl, "evo_rec": bw["evolution_exact"], "work_rec": bw["field_work"], "ref_rec": bw["reference_reset"], "int_led": A["cum_int_" + form][i], "evol_led": A["cum_evol"][i],
                                 "work_led": A["cum_work"][i], "ref_led": A["cum_ref"][i], "dH_led": A["dH_in_from_start"][i],
                                 "dH_rec": (num(JM["S"]["light.H_in_light"][k1]) - num(JM["S"]["light.H_in_light"][kw0])) if (num(JM["S"]["light.H_in_light"][k1]) is not None and num(JM["S"]["light.H_in_light"][kw0]) is not None) else None})
                last = rows[-1] if rows else None
                resid = (abs((last["work_rec"] or 0.0) - (last["work_led"] or 0.0)) + abs((last["ref_rec"] or 0.0) - (last["ref_led"] or 0.0))) if last else math.inf
                if best is None or resid < best[1]:
                    best = (grp, resid, rows)
            if best:
                cb = {"combination": best[0], "combination_residual": best[1], "rows": best[2]}
                ids.add("the chain end over the continuation: the records' evolution_exact = the ledger's integral of (c) (fixed cut; the slices' resolution)", [abs((r["evo_rec"] or 0.0) - (r["int_led"] or 0.0)) for r in best[2]], kind="relation", note="absolute, %s" % best[0])
                ids.add("the chain end over the continuation: the records' field_work, reference_reset = the ledger's work, ref", [abs((r["work_rec"] or 0.0) - (r["work_led"] or 0.0)) for r in best[2]] + [abs((r["ref_rec"] or 0.0) - (r["ref_led"] or 0.0)) for r in best[2]], kind="relation", note="absolute")
        lclk = LM["fields"]["L%s" % chans[0]]["clocks"] if chans else []
        table_clocks_slices = [c for c in TABLE_CLOCKS if c in lclk] or lclk[::8]
        LG = {"member": {f: {k: v for k, v in d.items() if k != "rows"} for f, d in LM["fields"].items()}, "control": {f: {k: v for k, v in d.items() if k != "rows"} for f, d in LK["fields"].items()},
              "cuts": cut_names, "cut_sites": cut_sites, "acc": acc, "diff": diff, "charges": charges, "node_spacing": node_spacing, "channels": ["L%s" % c for c in chans],
              "steps": {"light": LM["fields"]["L%s" % chans[0]]["steps_T_over_rs"] if chans else [], "heavy": LM["fields"][list(wH_all)[0]]["steps_T_over_rs"]}, "chain_end_budget": cb,
              "heavy_channels": list(wH_all), "heavy_weights": wH_all,
              "rows": {"member": {f: [{k: v for k, v in r.items() if k != "step"} | {"step": r["step"]} for r in d["rows"]] for f, d in LM["fields"].items()},
                       "control": {f: [dict(r) for r in d["rows"]] for f, d in LK["fields"].items()}}}
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ctx = {"P": P, "o": o, "JM": JM, "JK": JK, "inputs": inputs, "generated": generated, "ledger_stopped": ledger_stopped, "ledger": LG, "R": R, "ids": ids, "table_clocks_slices": table_clocks_slices}
    md = build_md(ctx)
    out = {"version": VERSION, "label": LABEL, "spec": SPEC, "generated_utc": generated, "inputs": inputs, "reading": [{"head": h, "text": t} for h, t in READING], "options": o,
           "preconditions": P, "ledger_stopped": ledger_stopped, "ledger": LG, "records_continued": {"series": D, "formation": F, "summary": Asum, "ext_paths": ext_paths, "roots_after_40": roots_after, "quadrature": quad},
           "identities": ids.d, "identities_order": ids.order, "summary": summary_lines(ctx), "seconds": time.time() - t0}
    os.makedirs(o["out_dir"], exist_ok=True)
    pj, pm, pn = (os.path.join(o["out_dir"], o["tag"] + ext) for ext in (".json", ".md", ".npz"))
    with open(pj, "w") as f:
        json.dump(out, f, indent=1, default=_jsonable)
    with open(pm, "w") as f:
        f.write(md)
    if LG is not None:
        arrays = {}
        for lab in ("member", "control"):
            for grp, per_cut in LG["acc"][lab].items():
                if not per_cut:
                    continue
                for cut, A in per_cut.items():
                    if A is None:
                        continue
                    for q, col in A.items():
                        arrays["%s__%s__%s__%s" % (lab, grp, cut, q)] = np.array([np.nan if v is None else v for v in col], dtype=float)
        np.savez_compressed(pn, **arrays)
    if not o.get("quiet"):
        print(md)
    for line in out["summary"]:
        print(line)
    print("written: %s, %s%s (%.1f s)" % (pj, pm, (", " + pn) if LG is not None else "", out["seconds"]))
    return 3 if ledger_stopped else 0


# ---------------------------------------------------------------------------------------------------------------------- self-test
def selftest(engine_dir=None):
    fails = []

    def check(name, ok, detail=""):
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""))
        if not ok:
            fails.append(name)

    def chain(n, V=0.3, hop=1.0):
        j = np.arange(n)
        h = np.zeros((n, n), dtype=complex)
        h[j[:-1], j[1:]] = -hop
        h[j[1:], j[:-1]] = -hop
        h[j, j] = V * np.sin(2.0 * np.pi * j / n)
        return h

    def negproj(h):
        w, U = np.linalg.eigh(h)
        neg = U[:, w < 0]
        return neg @ neg.conj().T

    def packet(n, j0, k, sig):
        j = np.arange(n)
        psi = np.exp(-(j - j0) ** 2 / (2.0 * sig ** 2)) * np.exp(1j * k * j)
        return psi / np.linalg.norm(psi)

    def evolve(C, h, dt):
        w, U = np.linalg.eigh(h)
        Ud = (U * np.exp(-1j * w * dt)) @ U.conj().T
        return Ud @ C @ Ud.conj().T
    # ---- 1. the nested helpers
    rec = {"a": 1, "light_v36": {"budget": {"evolution_exact": -2.0, "field_work": 0.5}, "channels": {"1": {"evolution_exact": -1.0}}}, "list": [{"reference_reset": 3.0}]}
    fp = find_paths(rec, set(BUDGET_NAMES))
    check("find_paths finds the budget names at any depth", sorted(dotted(p) for p, _ in fp) == ["light_v36.budget.evolution_exact", "light_v36.budget.field_work", "light_v36.channels.1.evolution_exact", "list.0.reference_reset"], str(sorted(dotted(p) for p, _ in fp)))
    check("get_path follows dict keys and list indices", get_path(rec, undot("list.0.reference_reset")) == 3.0 and get_path(rec, undot("light_v36.channels.1.evolution_exact")) == -1.0 and get_path(rec, undot("no.such")) is None)
    r2 = copy.deepcopy(rec)
    r2["light_v36"]["budget"]["field_work"] = 0.5000001
    dd = leaves_differ(rec, r2)
    check("leaves_differ names the one differing leaf", len(dd) == 1 and dd[0][0] == "light_v36.budget.field_work", str(dd))
    check("leaves_differ: identical records differ nowhere, nan equals nan", not leaves_differ(rec, copy.deepcopy(rec)) and not leaves_differ({"x": float("nan")}, {"x": float("nan")}))
    # ---- 2. the ledger core at a fixed cut, h constant: the current, its sign, the identification with the evolution term
    n, dt, nsub = 48, 0.02, 50
    h0 = chain(n)
    P0 = negproj(h0)
    psi = packet(n, 36.0, -2.0, 3.0)
    phi = (np.eye(n) - P0) @ psi
    phi /= np.linalg.norm(phi)
    C = P0 + np.outer(phi, phi.conj())
    site = site_index_map(n, n, "identity")
    mask = inside_mask(site, 24, "<=")
    ops = {"h": h0, "C": C, "P": P0}
    H, I_X, I_C, Cs = [], [], [], []
    for s in range(8 * nsub + 1):
        pt = point_terms({"h": h0, "C": C, "P": P0}, {"c": mask})
        H.append(pt["cuts"]["c"]["H_in"])
        I_X.append(pt["cuts"]["c"]["I_X"])
        I_C.append(pt["cuts"]["c"]["I_C"])
        Cs.append(C)
        C = evolve(C, h0, dt)
    H, I_X, I_C = np.array(H), np.array(I_X), np.array(I_C)
    fd = (H[2:] - H[:-2]) / (2.0 * dt)
    imax = int(np.argmax(np.abs(I_X[1:-1])))
    err = float(np.max(np.abs(I_X[1:-1] - fd)) / np.max(np.abs(fd)))
    check("the cut current equals dH_in/dt (central difference, dt 0.02) at a fixed cut and fixed h", err < 2e-3, "max rel %.2e" % err)
    check("inward-positive: a positive-energy packet moving to smaller sites raises H_in and the current is positive at its peak", H[-1] - H[0] > 0.3 and I_X[1:-1][imax] > 0, "dH_in %.4f, peak I %.4f" % (H[-1] - H[0], I_X[1:-1][imax]))
    check("with P_ref a spectral projector of h the reference carries no current: I_X = I_C", float(np.max(np.abs(I_X - I_C))) < 1e-10, "%.2e" % float(np.max(np.abs(I_X - I_C))))
    coarse, fine, tele = [], [], []
    for a in range(0, 8 * nsub, nsub):
        pa = point_terms({"h": h0, "C": Cs[a], "P": P0}, {"c": mask})
        pb = point_terms({"h": h0, "C": Cs[a + nsub], "P": P0}, {"c": mask})
        st = step_terms(h0, pa, {"C": Cs[a + nsub], "P": P0}, pb, {"c": mask}, {"c": mask}, nsub * dt)["c"]
        tele.append(abs(st["telescoping_residual"]))
        coarse.append(abs(st["closure_X"]))
        fine.append(abs(st["evol"] - float(np.sum(0.5 * (I_X[a + 1:a + nsub + 1] + I_X[a:a + nsub]) * dt))))
    check("the telescoping identity dH_in = evolution + work + reference + membership holds to roundoff (h constant: work = reference = membership = 0)", max(tele) < 1e-12, "max %.2e" % max(tele))
    check("the closure (evolution minus the trapezoid of the endpoint currents) shrinks with the resolution: fine (dt 0.02) against coarse (step 1.0)", sum(fine) * 20.0 < sum(coarse) and sum(coarse) > 0, "coarse %.3e, fine %.3e" % (sum(coarse), sum(fine)))
    check("work vanishes exactly when h does not change", all(abs(step_terms(h0, point_terms({"h": h0, "C": Cs[0], "P": P0}, {"c": mask}), {"C": Cs[nsub], "P": P0}, point_terms({"h": h0, "C": Cs[nsub], "P": P0}, {"c": mask}), {"c": mask}, {"c": mask}, 1.0)["c"][q]) < 1e-12 for q in ("work", "ref", "memb")))
    # ---- 3. a changing h, a moving cut: work, reference reset, the bare sweep against the referenced membership
    D = np.diag(0.5 * np.cos(np.pi * np.arange(n) / n))
    nmac, nsub2, dt2 = 8, 20, 0.05
    psi = packet(n, 8.0, 0.0, 2.5)
    phi = (np.eye(n) - P0) @ psi
    phi /= np.linalg.norm(phi)
    C = P0 + np.outer(phi, phi.conj())
    Tt = nmac * nsub2 * dt2
    cuts = [16 + 3 * i for i in range(nmac + 1)]
    sumT, rows = {q: 0.0 for q in ("evol", "work", "ref", "memb", "tele")}, []
    prev = None
    bareC, bareP = 0.0, 0.0
    for i in range(nmac + 1):
        t = i * nsub2 * dt2
        ht = h0 + (0.2 * t / Tt) * D
        Pt = negproj(ht)
        mk = inside_mask(site, cuts[i], "<=")
        pt = point_terms({"h": ht, "C": C, "P": Pt}, {"c": mk})
        if prev is not None:
            st = step_terms(prev["h"], prev["pt"], {"C": C, "P": Pt}, pt, {"c": prev["mk"]}, {"c": mk}, nsub2 * dt2)["c"]
            for q in ("evol", "work", "ref", "memb"):
                sumT[q] += st[q]
            sumT["tele"] = max(sumT["tele"], abs(st["telescoping_residual"]))
            new = mk & ~prev["mk"]
            bareC += float(pt["eC"][new].sum())
            bareP += float(pt["eP"][new].sum())
        rows.append(pt["cuts"]["c"]["H_in"])
        prev = {"h": ht, "pt": pt, "mk": mk}
        if i < nmac:
            for s in range(nsub2):
                ts = t + (s + 0.5) * dt2
                C = evolve(C, h0 + (0.2 * ts / Tt) * D, dt2)
    dH = rows[-1] - rows[0]
    check("moving cut, changing h: the telescoping identity holds to roundoff", sumT["tele"] < 1e-10, "max %.2e" % sumT["tele"])
    check("moving cut: the membership term equals the swept bare C minus the swept bare P_ref", abs(sumT["memb"] - (bareC - bareP)) < 1e-10, "memb %.6e, bare C - bare P %.6e" % (sumT["memb"], bareC - bareP))
    eng_evo, eng_ref = sumT["evol"] + bareC, sumT["ref"] - bareP
    check("the engine-style regrouping (evolution carrying the swept bare C, the reference carrying back the swept bare P_ref) sums to the same dH_in", abs(eng_evo + sumT["work"] + eng_ref - dH) < 1e-10, "%.3e" % abs(eng_evo + sumT["work"] + eng_ref - dH))
    check("the sweep's signature: the swept bare C is large and negative (the sea), the referenced membership small beside it, like -110196 against 0.004", bareC < 0 and abs(bareC) > 20.0 * abs(sumT["memb"]), "bare C %.4f, memb %.4e, work %.4e, ref %.4e" % (bareC, sumT["memb"], sumT["work"], sumT["ref"]))
    check("work is nonzero when h changes", abs(sumT["work"]) > 1e-6, "%.3e" % sumT["work"])
    # ---- 4. the budget's discovery on manufactured records
    nr = 41
    e_k = [0.0] + [-0.1 * k for k in range(1, nr)]
    w_k = [0.0] + [0.001 for _ in range(1, nr)]
    r_k = [0.0] + [0.1 * k + 0.0005 for k in range(1, nr)]
    Hc = list(np.cumsum(np.array(e_k) + np.array(w_k) + np.array(r_k)))
    recs = [{"T": 2.0 * k, "light_v36": {"H_in_light": Hc[k], "budget": {"evolution_exact": e_k[k], "field_work": w_k[k], "reference_reset": r_k[k]}}} for k in range(nr)]
    Sd = {"T_over_rs": [k * 0.0625 for k in range(nr)], "light.H_in_light": Hc}
    tg = {"evolution_exact": (sum(e_k), 1e-9), "field_work": (sum(w_k), 1e-12), "reference_reset": (sum(r_k), 1e-9), "dH_in_light": (Hc[-1] - Hc[0], 1e-12)}
    Bq = budget_closure(recs, Sd, tg, None, 0, nr - 1, "test")
    check("budget: per-step entries found at light_v36.budget.* and summed to the targets, the closure to roundoff", Bq["reproduced"] and Bq["mode"] == "per_step_sum" and abs(Bq["closure_residual"]) < 1e-9 and Bq["chosen"]["evolution_exact"]["path"] == "light_v36.budget.evolution_exact", str(Bq["mode"]))
    recs2 = [{"T": 2.0 * k, "light_v36": {"H_in_light": Hc[k], "budget": {"evolution_exact": float(np.sum(e_k[:k + 1])), "field_work": float(np.sum(w_k[:k + 1])), "reference_reset": float(np.sum(r_k[:k + 1]))}}} for k in range(nr)]
    Bq2 = budget_closure(recs2, Sd, tg, None, 0, nr - 1, "test")
    check("budget: cumulative entries read off as cumulative (the last value)", Bq2["reproduced"] and Bq2["mode"] == "cumulative_last", str(Bq2["mode"]))
    bw = budget_window(recs, Bq["chosen"], Bq["mode"], 10, 20)
    check("budget window 10 to 20 under the per-step mode (steps ending at the record)", abs(bw["evolution_exact"] - sum(e_k[11:21])) < 1e-12)
    recs3 = [{"T": 2.0 * k, "light_v36": {"H_in_light": Hc[k], "other": 1.0}} for k in range(nr)]
    Bq3 = budget_closure(recs3, Sd, tg, None, 0, nr - 1, "test")
    check("budget absent: not reproduced, the light_v36 keys listed, nothing substituted", (not Bq3["reproduced"]) and Bq3["light_v36_keys_first_record"] == ["H_in_light", "other"] and Bq3["notes"])
    recs4 = [{"T": 2.0 * k, "light_v36": {"H_in_light": Hc[k], "ledger": {"dE_evo": e_k[k], "dE_work": w_k[k], "dE_ref": r_k[k]}}} for k in range(nr)]
    Bq4 = budget_closure(recs4, Sd, tg, {"evolution_exact": "light_v36.ledger.dE_evo", "field_work": "light_v36.ledger.dE_work", "reference_reset": "light_v36.ledger.dE_ref"}, 0, nr - 1, "test")
    check("budget keys named by --budget-keys (a renaming, checked against the same targets)", Bq4["reproduced"] and Bq4["mode"] == "per_step_sum")
    # ---- 5. the join on version 2's manufactured bundle
    try:
        import C_readout_selftest_v2 as ST2
        runK, _, _ = ST2.make_bundle("off")
        S_, m_ = V2.extract_min(runK, "control")
        Dd = {"S": S_.d, "m": m_, "records": runK["records"], "args": runK["args"], "script": "manufactured", "version_keys": {}}
        cont = copy.deepcopy(runK)
        cont["records"] = cont["records"][6:]
        for k in range(9, 13):
            r = copy.deepcopy(runK["records"][8])
            r["T"], r["T_over_rs"] = ST2.H.tau * k, ST2.H.tau * k / ST2.H.rs
            cont["records"].append(r)
        Sc, mc = V2.extract_min(cont, "control_cont")
        Cc = {"S": Sc.d, "m": mc, "records": cont["records"], "args": cont["args"], "script": "manufactured", "version_keys": {}}
        J = join_runs(Dd, Cc, "test")
        rep = J["report"]
        check("join: 3 shared records bitwise, 4 appended, 13 joined, no gaps, the join record at the delivered last clock", rep["overlap_records"] == 3 and rep["overlap_bitwise"] == 3 and rep["appended_records"] == 4 and rep["joined_records"] == 13
              and not rep["gaps"] and rep["join_record"]["bitwise"] and len(rep["resume_gate_records"]) == 2, str({k: rep[k] for k in ("overlap_records", "overlap_bitwise", "appended_records", "joined_records")}))
        cont2 = copy.deepcopy(cont)
        cont2["records"][2]["E"] = runK["records"][8]["E"] + 1e-9
        Sc2, mc2 = V2.extract_min(cont2, "control_cont")
        J2 = join_runs(Dd, {"S": Sc2.d, "m": mc2, "records": cont2["records"], "args": cont2["args"], "script": "manufactured", "version_keys": {}}, "test")
        jr = J2["report"]["join_record"]
        check("join: a perturbed E at the join is printed by its path with d and the gate's pass level beside", (not jr["bitwise"]) and jr["differences"][0]["path"] == "E" and jr["leaves_differing"] == 1 and jr["gate"]["E_over_E0"]["times_tolerance"] == 0.0, str(jr["differences"][:1]))
        st = solver_stats(J, ["1", "2", "3"], J["n_delivered"])
        check("solver stats over the appended records", st["records"] == 4 and st["iterations_max"] == 3.0 and st["fallbacks"] == 0, str(st))
    except ImportError as e:
        check("join (needs C_readout_selftest_v2 beside this file)", False, str(e))
    # ---- 6. the conventions read off from a manufactured slice and record (inside '<', the weighting, the current's form and sign)
    with tempfile.TemporaryDirectory() as td:
        ns = n

        def refproj(h_ref, j0, k):
            """A reference that carries a current across the cut: the sea of h_ref plus one occupied complex packet centred at the cut, orthogonalised to
            that sea (a projector), so that I_X = I_C - I_P differs from I_C and the read-off has something to tell apart (real h and P_ref give I_P = 0 exactly)."""
            Q = negproj(h_ref)
            chi = (np.eye(ns) - Q) @ packet(ns, j0, k, 2.0)
            chi /= np.linalg.norm(chi)
            return Q + np.outer(chi, chi.conj())
        hH = chain(ns, 0.2)
        PH = refproj(hH + 0.05 * D, 24.0, 2.4)
        psH = packet(ns, 20.0, -2.4, 2.5)
        phH = (np.eye(ns) - negproj(hH)) @ psH
        phH /= np.linalg.norm(phH)
        CH = negproj(hH) + np.outer(phH, phH.conj())  # a moving packet: the heavy field carries a current and takes part in the read-off of the sign
        light = {}
        for c, (j0, kk) in zip((1, 2, 3), ((30.0, -2.0), (20.0, 1.5), (10.0, -1.0))):
            hl = chain(ns, 0.1 * c)
            Pl = refproj(hl + 0.03 * D, 24.0, -2.4)
            ps = packet(ns, j0, kk, 2.0)
            ph = (np.eye(ns) - negproj(hl)) @ ps
            ph /= np.linalg.norm(ph)
            light[c] = (hl, negproj(hl) + np.outer(ph, ph.conj()), Pl)
        arrays = {"T_over_rs": np.array(40.0), "h_H": hH, "C_H": CH, "P_H": PH}
        for c in (1, 2, 3):
            arrays["h_L%d" % c], arrays["C_L%d" % c], arrays["P_L%d" % c] = light[c]
        p40 = os.path.join(td, "slice_T40.0.npz")
        np.savez(p40, **arrays)
        td_list = os.path.join(td, "listing")  # the listing's extra files sit apart: with the filename clock parsed, light_T40.5.npz would otherwise double the ledger's clock 40.5 below
        os.makedirs(td_list)
        np.savez(os.path.join(td_list, "light_T40.5.npz"), **{k: v for k, v in arrays.items() if k != "T_over_rs" and not k.endswith("_H")})
        np.savez(os.path.join(td_list, "junk.npz"), x=np.zeros(3))
        ls = list_slices([td, td_list], 32.0, CLOCK_REGEX)
        clocks = {os.path.basename(s["path"]): (s["clock"], s["clock_from"]) for s in ls}
        check("slice listing: the clock from the key T_over_rs, from the filename when absent, None when neither", clocks["slice_T40.0.npz"] == (40.0, "T_over_rs") and clocks["light_T40.5.npz"] == (40.5, "filename") and clocks["junk.npz"] == (None, None), str(clocks))
        rx = {nm: (float(mm.group(1)) if mm else None) for nm, mm in ((nm, re.search(CLOCK_REGEX, nm)) for nm in ("slice_T41.npz", "run_Trs40.5_light.npz", "T40.5.heavy.npz", "clock=36.npz", "T40.5.3.npz", "junk.npz"))}
        check("the default filename pattern: the clock before .npz with or without a fraction, before an inner dot or an underscore, after '=', none when the number is ambiguous or absent",
              rx == {"slice_T41.npz": 41.0, "run_Trs40.5_light.npz": 40.5, "T40.5.heavy.npz": 40.5, "clock=36.npz": 36.0, "T40.5.3.npz": None, "junk.npz": None}, str(rx))
        wts = {"1": 16.0, "2": 32.0, "3": 48.0}
        icut = cut_index_from_r_collar(25.0, ns)
        mk_lt = inside_mask(site, icut, "<")
        ptH = point_terms({"h": hH, "C": CH, "P": PH}, {"c": mk_lt})
        ptL = {c: point_terms({"h": light[c][0], "C": light[c][1], "P": light[c][2]}, {"c": mk_lt}) for c in (1, 2, 3)}
        S6 = {"T_over_rs": [40.0, 40.0625], "r_collar": [25.0, 25.0], "H_in": [ptH["cuts"]["c"]["H_in"], None], "light.H_in_light": [sum(wts[str(c)] * ptL[c]["cuts"]["c"]["H_in"] for c in (1, 2)), None],
              "light.E_inv_heavy": [ptH["trace_X"], None], "M1_I_E": [-ptH["cuts"]["c"]["I_C"], None], "light.I_E_light_sourcing": [-sum(wts[str(c)] * ptL[c]["cuts"]["c"]["I_C"] for c in (1, 2)), None]}
        for c in (1, 2, 3):
            S6["L%d.E_inv" % c], S6["L%d.w" % c], S6["L%d.I_E_cut_weighted" % c] = [ptL[c]["trace_X"], None], [wts[str(c)], None], [-wts[str(c)] * ptL[c]["cuts"]["c"]["I_C"], None]
        run6 = {"S": S6, "m": {"nsites": ns, "rs": 32.0, "channels": ["1", "2", "3"], "sourcing": ["1", "2"], "guard": ["3"]}, "records": [{}, {}]}
        SK6 = {"slice_keys": {"heavy": {"h": "h_H", "C": "C_H", "P": "P_H"}}}  # the manufactured dense heavy slice keeps version 3's names; the defaults are the engine's (sea_<k>, dust_<k>, f_<k>, R2_<k>)
        fields6 = [("heavy", "heavy", None, role_keys(SK6, "heavy"))] + [("L%s" % c, "light", c, role_keys({}, "light", c)) for c in ("1", "2", "3")]
        s40 = [s for s in ls if s["clock"] == 40.0][0]
        pre = prepass("test", run6, s40, fields6, None, {"r16": 12}, {"index_layout": "auto"}, ns)
        cv = decide_conventions(pre, run6, run6["m"])
        ix_scores = [max(r["res_heavy"], r["res_light"]) for r in cv["current_rows"] if r["form"] == "I_X"]
        check("read off: inside rule '<', H_in_light weighted over the sourcing channels, the current of C alone with the opposite stored sign (the reference carries a current, so I_X is told apart), H_in reproduced to roundoff",
              cv["layout"] == "identity" and cv["rule"] == "<" and cv["weighting"] == ("w", "sourcing") and cv["current_form"] == "I_C" and cv["current_sign"] == -1 and cv["H_in_reproduced"] and cv["current_reproduced"]
              and not cv["current_tie"] and all(x > TOL_CURRENT for x in ix_scores),
              "layout %s rule %s %s %s %+d score %.1e current %.1e tie %s, the I_X rows' scores %s" % (cv["layout"], cv["rule"], cv["weighting"], cv["current_form"], cv["current_sign"], cv["score"], cv["current_score"],
                                                                                                   cv["current_tie"], ["%.2e" % x for x in ix_scores]))
        check("read off: the chain traces equal the manufactured E_inv", max(d["res"] for d in cv["E_inv_checks"].values()) < 1e-12)
        vc = verify_conventions(pre, run6, run6["m"], cv)
        check("verify under the read-off conventions reproduces the same record", vc["heavy"]["res_H_in"] < 1e-12 and vc["light"]["res_H_in"] < 1e-12 and vc["light"]["res_I"] < 1e-10)
        # the degenerate case the real slices may sit in: references carrying no current across the cut (here real h and P_ref, so h P_ref is real and its cut
        # current vanishes identically; a spectral projector of h does the same), I_X = I_C exactly: the tie is declared, I_C taken by the stated preference,
        # the sign still read off from the data, the NOTE printed; the file sits in a subdirectory so the listings of td are unchanged
        td_tie = os.path.join(td, "tie")
        os.makedirs(td_tie)
        P_tie = {"H": negproj(hH + 0.05 * D)}
        P_tie.update({c: negproj(light[c][0] + 0.03 * D) for c in (1, 2, 3)})
        tie_arr = {"T_over_rs": np.array(40.0), "h_H": hH, "C_H": CH, "P_H": P_tie["H"]}
        for c in (1, 2, 3):
            tie_arr["h_L%d" % c], tie_arr["C_L%d" % c], tie_arr["P_L%d" % c] = light[c][0], light[c][1], P_tie[c]
        np.savez(os.path.join(td_tie, "slice_T40.0.npz"), **tie_arr)
        ptHt = point_terms({"h": hH, "C": CH, "P": P_tie["H"]}, {"c": mk_lt})
        ptLt = {c: point_terms({"h": light[c][0], "C": light[c][1], "P": P_tie[c]}, {"c": mk_lt}) for c in (1, 2, 3)}
        S6t = dict(S6)
        S6t.update({"H_in": [ptHt["cuts"]["c"]["H_in"], None], "light.H_in_light": [sum(wts[str(c)] * ptLt[c]["cuts"]["c"]["H_in"] for c in (1, 2)), None], "light.E_inv_heavy": [ptHt["trace_X"], None],
                    "M1_I_E": [-ptHt["cuts"]["c"]["I_C"], None], "light.I_E_light_sourcing": [-sum(wts[str(c)] * ptLt[c]["cuts"]["c"]["I_C"] for c in (1, 2)), None]})
        for c in (1, 2, 3):
            S6t["L%d.E_inv" % c] = [ptLt[c]["trace_X"], None]
        run6t = dict(run6)
        run6t["S"] = S6t
        s40t = list_slices([td_tie], 32.0, CLOCK_REGEX)[0]
        cvt = decide_conventions(prepass("tie", run6t, s40t, fields6, None, {"r16": 12}, {"index_layout": "auto"}, ns), run6t, run6t["m"])
        check("read off, a reference carrying no current (I_X = I_C exactly): the forms tie, the tie is declared, I_C is taken by the stated preference, the sign is still read off, the NOTE printed",
              cvt["current_tie"] and sorted(cvt["current_tied"]) == [("I_C", -1), ("I_X", -1)] and cvt["current_form"] == "I_C" and cvt["current_sign"] == -1 and cvt["current_reproduced"] and cvt["H_in_reproduced"]
              and any(f.startswith("NOTE: the current's candidates") for f in cvt["flags"]) and abs(ptHt["cuts"]["c"]["I_X"] - ptHt["cuts"]["c"]["I_C"]) < 1e-14,
              "tie %s tied %s -> %s %+d, |I_X - I_C| heavy %.1e" % (cvt["current_tie"], cvt["current_tied"], cvt["current_form"], cvt["current_sign"], abs(ptHt["cuts"]["c"]["I_X"] - ptHt["cuts"]["c"]["I_C"])))
        z = np.load(p40, allow_pickle=False)
        keys = list(z.files)
        try:
            field_ops(z, [k for k in keys if k != "h_L2"], role_keys({}, "light", "2"), None, "light", "2", "test")
            check("a slice carrying C but not h stops with MissingKey naming h", False, "no exception")
        except MissingKey as e:
            check("a slice carrying C but not h stops with MissingKey naming h", "h_L2" in str(e))
        check("a slice not carrying the field returns None (no stop)", field_ops(z, [k for k in keys if not k.endswith("_H")], role_keys(SK6, "heavy"), None, "heavy", None, "test") is None)
        z.close()
        w_, U_ = np.linalg.eigh(hH)
        Phi = U_[:, :10]
        G1, G2 = gram(Phi, ns, None, "t"), gram(Phi.T, ns, None, "t")
        check("gram: the orbital axis read off in either orientation", float(np.max(np.abs(G1 - Phi @ Phi.conj().T))) < 1e-14 and float(np.max(np.abs(G2 - Phi @ Phi.conj().T))) < 1e-14)
        G3 = gram(Phi, ns, np.full(10, 0.5), "t")
        check("gram with occupations", float(np.max(np.abs(G3 - 0.5 * Phi @ Phi.conj().T))) < 1e-14)
        # the ledger over a manufactured pair of runs with slices at 40, 40.5, 41 (light) and 40, 41 (heavy)
        Cl = {c: light[c][1] for c in (1, 2, 3)}
        CHt = CH
        for i, cl in enumerate((40.5, 41.0)):
            arr = {"T_over_rs": np.array(cl)}
            for c in (1, 2, 3):
                Cl[c] = evolve(Cl[c], light[c][0], 1.0)
                arr["h_L%d" % c], arr["C_L%d" % c], arr["P_L%d" % c] = light[c][0], Cl[c], light[c][2]
            if cl == 41.0:
                CHt = evolve(CHt, hH, 2.0)
                arr["h_H"], arr["C_H"], arr["P_H"] = hH, CHt, PH
            np.savez(os.path.join(td, "slice_T%g.npz" % cl), **arr)
        ls2 = list_slices([td], 32.0, CLOCK_REGEX)
        nrec = 17
        S7 = dict((k, [v[0]] * nrec) for k, v in S6.items())
        S7["T_over_rs"] = [40.0 + 0.0625 * k for k in range(nrec)]
        run7 = {"S": S7, "m": run6["m"], "records": [{}] * nrec}
        LL = run_ledger("test", run7, ls2, fields6, None, cv, {"r16": 12}, {"window_start": 40.0}, lambda s: None)
        check("ledger: the light field has 3 slice clocks (0.5 r_s steps), the heavy 2 (1 r_s)", LL["fields"]["L1"]["n_slices"] == 3 and LL["fields"]["heavy"]["n_slices"] == 2 and LL["fields"]["L1"]["steps_T_over_rs"] == [0.5] and LL["fields"]["heavy"]["steps_T_over_rs"] == [1.0], str({f: d["clocks"] for f, d in LL["fields"].items()}))
        A1 = accumulate(LL["fields"]["L1"]["rows"], "r16")
        check("ledger: cumulative columns telescope (dH_in from start = cum evolution + work + ref + memb), closure = evolution - integral", abs(A1["dH_in_from_start"][-1] - (A1["cum_evol"][-1] + A1["cum_work"][-1] + A1["cum_ref"][-1] + A1["cum_memb"][-1])) < 1e-12
              and abs(A1["cum_closure_X"][-1] - (A1["cum_evol"][-1] - A1["cum_int_I_X"][-1])) < 1e-12)
        accs = {"L%d" % c: accumulate(LL["fields"]["L%d" % c]["rows"], "r16") for c in (1, 2, 3)}
        comb = combine({"L1": accs["L1"], "L2": accs["L2"]}, {"L1": 16.0, "L2": 32.0})
        check("combine: the weighted sum over channels at the common clocks", comb is not None and abs(comb["H_in"][0] - (16.0 * accs["L1"]["H_in"][0] + 32.0 * accs["L2"]["H_in"][0])) < 1e-12 and comb["clock"] == [40.0, 40.5, 41.0])
        dif = difference(comb, comb)
        check("difference: a run against itself is zero at every common clock", dif is not None and all(abs(x) < 1e-15 for x in dif["H_in"]) and all(dif["n_in_equal"]))
    # ---- 7. the index layouts and the cut index rule
    sm, cm = site_index_map(96, 48, "site_major"), site_index_map(96, 48, "component_major")
    check("index layouts: site_major pairs consecutive indices, component_major repeats the chain", list(sm[:4]) == [0, 0, 1, 1] and list(cm[46:50]) == [46, 47, 0, 1])
    check("the chain-end cut from r_collar: 236 -> 235, 2048 -> 2038 on 2048 sites", cut_index_from_r_collar(236.0, 2048) == 235 and cut_index_from_r_collar(2048.0, 2048) == 2038)
    ci = collar_site_index(np.arange(1, 2049, dtype=float), 512.0)
    check("the collar's cut index is the first site beyond r_c (the engine's 'beyond' index): 512 on R = 1..2048 at r_c = 512 a (the last site inside R 512, the first beyond 513), 512 sites inside under the rule '<'",
          ci["index"] == 512 and ci["R_last_inside"] == 512.0 and ci["R_at_index"] == 513.0 and int(inside_mask(np.arange(2048), ci["index"], "<").sum()) == 512 and ci["nearest_index"] == 511, str(ci))
    try:
        site_index_map(100, 48, "site_major")
        check("an operator dimension that is not a multiple of the sites stops", False, "no exception")
    except Inconsistent:
        check("an operator dimension that is not a multiple of the sites stops", True)
    # ---- 8. the quadrature read-off and the truncation candidates
    Sq = {"T": [2.0 * k for k in range(641)], "T_over_rs": [0.0625 * k for k in range(641)], "M1_I_E": [1e-9] * 641, "light.I_E_light_sourcing": [2e-11] * 641}
    ci2 = current_integral(Sq, "M1_I_E", 0, 640, 32.0)
    check("current integral: a constant current over 0 to 40 (T 0 to 1280) integrates to I x 1280 in T and I x 40 in T/r_s", abs(ci2["trapezoid_T"] - 1.28e-6) < 1e-18 and abs(ci2["trapezoid_T_over_rs"] - 4e-8) < 1e-20)
    cs = cumulative_integral_series(Sq, "M1_I_E", "trapezoid_T")
    check("cumulative integral series ends at the window's integral", abs(cs[-1] - 1.28e-6) < 1e-18 and cs[0] == 0.0)
    St = {"L1.E_inv_weighted": [1.0], "L2.E_inv_weighted": [1.0], "L3.E_inv_weighted": [0.5], "L1.E_inv": [1.0 / 16], "L2.E_inv": [1.0 / 32], "L3.E_inv": [0.5 / 48], "light.I_E_light": [3.0], "light.I_E_light_sourcing": [2.0],
          "L1.I_E_cut_weighted": [1.0], "L2.I_E_cut_weighted": [1.0], "L3.I_E_cut_weighted": [1.0], "L1.eps_sum": [1.0], "L2.eps_sum": [1.0], "L3.eps_sum": [1.0], "L1.w": [16.0], "L2.w": [32.0], "L3.w": [48.0]}
    tc = truncation_candidates(St, 0, {"sourcing": ["1", "2"], "guard": ["3"]})
    check("truncation candidates: guard over sourcing under each definition", abs(tc["E_inv:w"] - 0.25) < 1e-15 and abs(tc["I_E:total"] - 0.5) < 1e-15 and abs(tc["I_E:w"] - 0.5) < 1e-15 and abs(tc["S:w"] - 1.0) < 1e-15 and abs(tc["S:1"] - 0.5) < 1e-15, str(tc))
    # ---- 9. version 3b: the column route (the state and the reference as orbital columns with occupations, h sparse or dense) against the dense route;
    #         the engine's key names and the heavy channels summed with their weights through a manufactured slice; the slice copies; the end clock in P2
    rng9 = np.random.default_rng(9)

    def unitary9(k):
        Q9, _ = np.linalg.qr(rng9.standard_normal((k, k)) + 1j * rng9.standard_normal((k, k)))
        return Q9

    def column_state9(h, h_ref):
        """the state as columns: the sea's negative eigenvectors rotated by a random unitary within their span (the same projector, no eigenvectors), two
        positive-energy packets with the occupations 1 and 1/2 as the dust; the reference the negative eigenvectors of h_ref."""
        w_, U_ = np.linalg.eigh(h)
        sea_ = U_[:, w_ < 0] @ unitary9(int((w_ < 0).sum()))
        Pp = U_[:, w_ > 0] @ U_[:, w_ > 0].conj().T
        dust_, _ = np.linalg.qr(np.stack([Pp @ packet(n, 14.0, -1.2, 2.5), Pp @ packet(n, 30.0, 0.8, 2.5)], axis=1))
        wr, Ur = np.linalg.eigh(h_ref)
        return sea_, dust_, np.array([1.0, 0.5]), Ur[:, wr < 0], w_

    def evolve_cols9(cols, h, dt):
        w_, U_ = np.linalg.eigh(h)
        return (U_ * np.exp(-1j * w_ * dt)) @ (U_.conj().T @ cols)

    def dense_ops9(h, sea_, dust_, f_, ref_):
        return {"h": h, "C": sea_ @ sea_.conj().T + (dust_ * f_[None, :]) @ dust_.conj().T, "P": ref_ @ ref_.conj().T}

    def cols_ops9(h, sea_, dust_, f_, ref_):
        return {"h": h, "cols_C": [(sea_, None), (dust_, f_)], "cols_P": [(ref_, None)], "route": "columns"}
    hA, hB = chain(n, 0.2), chain(n, 0.35, hop=0.9)
    seaA, dustA, fA, refA, eA = column_state9(hA, hA + 0.05 * D)
    seaB, dustB, fB, refB, eB = column_state9(hB, hB + 0.04 * D)
    masks9 = {"a": inside_mask(site, 20, "<"), "b": inside_mask(site, 30, "<=")}
    ptd = point_terms(dense_ops9(hA, seaA, dustA, fA, refA), masks9)
    ptc = point_terms(cols_ops9(sp.csr_matrix(hA), seaA, dustA, fA, refA), masks9)
    ptc2 = point_terms(cols_ops9(hA, seaA, dustA, fA, refA), masks9)
    dvec = max(float(np.max(np.abs(ptd[q] - p_[q]))) for p_ in (ptc, ptc2) for q in ("eC", "eP", "eX"))
    dcut = max(abs(ptd["cuts"][c_][q] - p_["cuts"][c_][q]) for p_ in (ptc, ptc2) for c_ in masks9 for q in ("H_in", "H_out", "I_X", "I_C", "bare_in_C", "bare_in_P"))
    check("the column route (sparse or dense h on the state's and the reference's columns with occupations) gives the dense route's diagonals and cut currents to roundoff",
          dvec < 1e-12 and dcut < 1e-12 and all(ptd["cuts"][c_]["n_in"] == ptc["cuts"][c_]["n_in"] for c_ in masks9), "max |d diag| %.1e, max |d cut| %.1e" % (dvec, dcut))
    hA2 = hA + 0.1 * D
    seaA2, dustA2 = evolve_cols9(seaA, hA, 1.0), evolve_cols9(dustA, hA, 1.0)
    wr2, Ur2 = np.linalg.eigh(hA2 + 0.05 * D)
    refA2 = Ur2[:, wr2 < 0]
    pt2d = point_terms(dense_ops9(hA2, seaA2, dustA2, fA, refA2), masks9)
    pt2c = point_terms(cols_ops9(sp.csr_matrix(hA2), seaA2, dustA2, fA, refA2), masks9)
    std = step_terms(hA, ptd, dense_ops9(hA2, seaA2, dustA2, fA, refA2), pt2d, masks9, masks9, 1.0)
    stc = step_terms(sp.csr_matrix(hA), ptc, cols_ops9(sp.csr_matrix(hA2), seaA2, dustA2, fA, refA2), pt2c, masks9, masks9, 1.0)
    dstep = max(abs(std[c_][q] - stc[c_][q]) for c_ in masks9 for q in ("evol", "work", "ref", "memb", "telescoping_residual", "closure_X", "closure_C"))
    check("the step terms in the records' split (work = the inside sum of Re[(h' - h)(C' - P')], ref = -(the inside sum of Re[h (P' - P)])) agree between the routes and telescope",
          dstep < 1e-12 and all(abs(std[c_]["telescoping_residual"]) < 1e-12 for c_ in masks9), "max |d step| %.1e" % dstep)
    st_fixedP = step_terms(hA, ptd, dense_ops9(hA2, seaA2, dustA2, fA, refA), point_terms(dense_ops9(hA2, seaA2, dustA2, fA, refA), masks9), masks9, masks9, 1.0)["a"]
    st_fixedh = step_terms(hA, ptd, dense_ops9(hA, seaA2, dustA2, fA, refA2), point_terms(dense_ops9(hA, seaA2, dustA2, fA, refA2), masks9), masks9, masks9, 1.0)["a"]
    check("the records' split: the reference term vanishes when P_ref does not change (h changing), the work when h does not change (P_ref changing), the work nonzero when h changes",
          abs(st_fixedP["ref"]) < 1e-12 and abs(st_fixedP["work"]) > 1e-6 and abs(st_fixedh["work"]) < 1e-12, "ref %.1e work %.3e | work %.1e" % (st_fixedP["ref"], st_fixedP["work"], st_fixedh["work"]))
    with tempfile.TemporaryDirectory() as td9:
        arr9 = {"T_over_rs": np.array(40.0), "N_applied": np.ones(n), "v_applied": np.zeros(n), "R": np.arange(1, n + 1, dtype=float), "kappas": np.array([1, 2]), "weights": np.array([16.0, 32.0]),
                "sea_1": seaA, "dust_1": dustA, "f_1": fA, "R2_1": refA, "e_1": eA, "mask_1": np.packbits(eA < 0),
                "sea_2": seaB, "dust_2": dustB, "f_2": fB, "R2_2": refB, "e_2": eB, "mask_2": np.packbits(eB < 0)}
        for c in (1, 2, 3):
            arr9["h_L%d" % c], arr9["C_L%d" % c], arr9["P_L%d" % c] = light[c]
        p9 = os.path.join(td9, "orbitals_T0040.00.npz")
        np.savez(p9, **arr9)
        hmap9 = {1: hA, 2: hB}

        def tb9(z_, field_, kappa_):
            return hmap9[int(kappa_)]
        z9 = np.load(p9, allow_pickle=False)
        keys9 = list(z9.files)
        ops9 = field_ops(z9, keys9, role_keys({}, "heavy", 1), tb9, "heavy", 1, "t9")
        nsea9 = int((eA < 0).sum())
        check("field_ops under the engine's key names: the heavy channel's state from sea_<k> (occupation 1) plus dust_<k> with f_<k>, the reference from R2_<k>, h from the builder (the column route)",
              ops9 is not None and ops9["route"] == "columns" and ops9["dim"] == n and ops9["n_cols"] == {"C": [nsea9, 2], "P": nsea9} and "f_1" in ops9["from"]["C"] and ops9["from"]["P"].startswith("R2_1")
              and ops9["from"]["h"] == "builder" and ops9["orth_P"] < 1e-12, str({k_: ops9[k_] for k_ in ("route", "dim", "n_cols", "from")}))
        try:
            field_ops(z9, [k_ for k_ in keys9 if k_ != "f_1"], role_keys({}, "heavy", 1), tb9, "heavy", 1, "t9")
            check("a dust array without its occupations f_<k> stops with MissingKey naming the key", False, "no exception")
        except MissingKey as e:
            check("a dust array without its occupations f_<k> stops with MissingKey naming the key", "f_1" in str(e))
        check("a light-only key set does not carry a heavy channel (None, no stop)", field_ops(z9, [k_ for k_ in keys9 if "_L" in k_], role_keys({}, "heavy", 1), tb9, "heavy", 1, "t9") is None)
        sh9 = {d_["key"]: (d_.get("shape"), d_.get("dtype")) for d_ in inspect_slice(p9, ["sea_1", "f_1", "T_over_rs", "mask_1"])}
        check("inspect_slice reads the shapes and dtypes from the npz headers without loading the arrays",
              sh9 == {"sea_1": ([n, nsea9], "complex128"), "f_1": ([2], "float64"), "T_over_rs": ([], "float64"), "mask_1": ([int(math.ceil(n / 8.0))], "uint8")}, str(sh9))
        z9.close()
        pt9 = {1: point_terms(dense_ops9(hA, seaA, dustA, fA, refA), {"c": mk_lt}), 2: point_terms(dense_ops9(hB, seaB, dustB, fB, refB), {"c": mk_lt})}
        S9 = {"T_over_rs": [40.0, 40.0625], "r_collar": [25.0, 25.0], "H_in": [16.0 * pt9[1]["cuts"]["c"]["H_in"] + 32.0 * pt9[2]["cuts"]["c"]["H_in"], None],
              "light.H_in_light": [sum(wts[str(c)] * ptL[c]["cuts"]["c"]["H_in"] for c in (1, 2)), None], "light.E_inv_heavy": [16.0 * pt9[1]["trace_X"] + 32.0 * pt9[2]["trace_X"], None],
              "M1_I_E": [16.0 * pt9[1]["cuts"]["c"]["I_C"] + 32.0 * pt9[2]["cuts"]["c"]["I_C"], None], "light.I_E_light_sourcing": [sum(wts[str(c)] * ptL[c]["cuts"]["c"]["I_C"] for c in (1, 2)), None]}
        for c in (1, 2, 3):
            S9["L%d.E_inv" % c], S9["L%d.w" % c], S9["L%d.I_E_cut_weighted" % c] = [ptL[c]["trace_X"], None], [wts[str(c)], None], [wts[str(c)] * ptL[c]["cuts"]["c"]["I_C"], None]
        run9 = {"S": S9, "m": run6["m"], "records": [{}, {}]}
        fields9 = [("H1", "heavy", 1, role_keys({}, "heavy", 1)), ("H2", "heavy", 2, role_keys({}, "heavy", 2))] + [("L%s" % c, "light", c, role_keys({}, "light", c)) for c in ("1", "2", "3")]
        s9 = [s_ for s_ in list_slices([td9], 32.0, CLOCK_REGEX) if s_["clock"] == 40.0][0]
        pre9 = prepass("t9", run9, s9, fields9, tb9, {"r16": 12}, {"index_layout": "auto"}, n)
        cv9 = decide_conventions(pre9, run9, run9["m"])
        check("the heavy channels summed with the weights 16 kappa reproduce the record's H_in, E_inv_heavy and M1_I_E (the stored current inward-positive: sign +1 read off, I_C, no tie); the light weighting and the inside rule as before",
              cv9["H_in_reproduced"] and cv9["layout"] == "identity" and cv9["rule"] == "<" and cv9["weighting"] == ("w", "sourcing") and cv9["current_form"] == "I_C" and cv9["current_sign"] == 1 and cv9["current_reproduced"]
              and not cv9["current_tie"] and cv9["E_inv_checks"]["heavy"]["res"] < 1e-10 and cv9["E_inv_checks"]["heavy"]["channels"] == ["H1", "H2"] and heavy_fields(pre9["per"]) == {"H1": 16.0, "H2": 32.0}
              and not cv9["layout_rule_tie"],
              "layout %s rule %s %s %s %+d score %.1e current %.1e heavy E_inv res %s tie %s" % (cv9["layout"], cv9["rule"], cv9["weighting"], cv9["current_form"], cv9["current_sign"], cv9["score"], cv9["current_score"],
                                                                                                fm(cv9["E_inv_checks"]["heavy"]["res"], 2), cv9["layout_rule_tie"]))
        arr41 = dict(arr9)
        arr41.update({"T_over_rs": np.array(41.0), "sea_1": seaA2, "dust_1": dustA2, "R2_1": refA2, "sea_2": evolve_cols9(seaB, hB, 1.0), "dust_2": evolve_cols9(dustB, hB, 1.0)})
        for c in (1, 2, 3):
            arr41["C_L%d" % c] = evolve(light[c][1], light[c][0], 2.0)
        np.savez(os.path.join(td9, "orbitals_T0041.00.npz"), **arr41)
        td9c = os.path.join(td9, "copy")
        os.makedirs(td9c)
        np.savez(os.path.join(td9c, "orbitals_T0041.00.npz"), **arr41)
        nrec9 = 17
        S9b = dict((k_, [v_[0]] * nrec9) for k_, v_ in S9.items())
        S9b["T_over_rs"] = [40.0 + 0.0625 * k_ for k_ in range(nrec9)]
        run9b = {"S": S9b, "m": run6["m"], "records": [{}] * nrec9}
        LL9 = run_ledger("t9", run9b, list_slices([td9, td9c], 32.0, CLOCK_REGEX), fields9, tb9, cv9, {"r16": 12}, {"window_start": 40.0}, lambda s_: None)
        acc9 = {f_: accumulate(LL9["fields"][f_]["rows"], "r16") for f_ in ("H1", "H2")}
        comb9 = combine(acc9, {"H1": 16.0, "H2": 32.0})
        tele9 = abs(comb9["dH_in_from_start"][-1] - (comb9["cum_evol"][-1] + comb9["cum_work"][-1] + comb9["cum_ref"][-1] + comb9["cum_memb"][-1])) if comb9 else None
        check("the ledger: the heavy group is the 16 kappa-weighted sum of its channels at the common clocks and telescopes; the second copy of one clock is read once and listed as dropped",
              comb9 is not None and comb9["clock"] == [40.0, 41.0] and LL9["fields"]["H1"]["n_slices"] == 2 and len(LL9["fields"]["H1"]["duplicates_dropped"]) == 1
              and abs(comb9["H_in"][1] - (16.0 * acc9["H1"]["H_in"][1] + 32.0 * acc9["H2"]["H_in"][1])) < 1e-12 * max(1.0, abs(comb9["H_in"][1])) and tele9 < 1e-10,
              "clocks %s dropped %s tele %s" % (comb9["clock"] if comb9 else None, LL9["fields"]["H1"]["duplicates_dropped"], fm(tele9, 2)))
        td9d = os.path.join(td9, "bad")
        os.makedirs(td9d)
        arr_bad = dict(arr41)
        arr_bad["sea_1"] = seaA2[:, :-1]
        np.savez(os.path.join(td9d, "orbitals_T0041.00.npz"), **arr_bad)
        try:
            run_ledger("t9", run9b, list_slices([td9, td9d], 32.0, CLOCK_REGEX), fields9[:1], tb9, cv9, {"r16": 12}, {"window_start": 40.0}, lambda s_: None)
            check("two slices with the same keys and different sizes at one clock stop the ledger", False, "no exception")
        except Inconsistent as e:
            check("two slices with the same keys and different sizes at one clock stop the ledger", "different sizes" in str(e))
    check("P2: the end clock T_over_rs is an expected difference (a delivered run against its continuation, the two continuations), light_coupling between the member and its control, nothing else",
          bool(EXPECTED_CONT_ARG.search("T_over_rs")) and bool(EXPECTED_CONT_ARG.search("t_end")) and not EXPECTED_CONT_ARG.search("tau") and not EXPECTED_CONT_ARG.search("light_coupling")
          and not EXPECTED_CONT_ARG.search("nsites") and not EXPECTED_CONT_ARG.search("rs"))
    # ---- 10. the adapter: the engine's own operator on a manufactured slice (the engine's modules must be importable: the box has them under --engine-dir;
    #          elsewhere the line reads SKIP, which the run order does not accept on the box)
    try:
        import C_readout_v3_hbuilder as HB
        rep = HB.selftest_manufactured(nsites=48, engine_dir=engine_dir, log=lambda s_: None)
        check("adapter: the engine's h (engine_collapse_v5.Chan / engine_twofield_v3.LightChan with engine_collapse_v36.clocked_host_sparse) on a manufactured slice equals engine_ops_v1's dense clocked chain, is Hermitian, its own eigenpairs pass the eigenpair check, a wrong R and a differing control stop",
              rep["ok"], rep["summary"])
    except Exception as e:  # noqa: BLE001
        print("SKIP  adapter: %s: %s (the engine's modules are not importable here; on the box, with --engine-dir, this line must read PASS)" % (type(e).__name__, e))
    if fails:
        print("SELFTEST FAIL: %d check(s): %s" % (len(fails), "; ".join(fails)))
        return 1
    print("SELFTEST PASS: every check passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Computation C: the continuation's 16 and 24 r_s collar ledger (section 6), the preconditions and the chain-end closure first; versions 1 and 2 imported.")
    ap.add_argument("--run-json", help="the delivered member's record json (T/r_s 0 to 40)")
    ap.add_argument("--run-cont-json", help="the member's continued record json (to 80)")
    ap.add_argument("--control-json", help="the delivered control's record json (0 to 40)")
    ap.add_argument("--control-cont-json", help="the control's continued record json (to 50)")
    ap.add_argument("--slices", default=None, help="the member's slice directories (comma-separated)")
    ap.add_argument("--control-slices", default=None, help="the control's slice directories (comma-separated)")
    ap.add_argument("--run-cont-fields-npz", default=None, help="the member's continued _fields npz (the site radii R)")
    ap.add_argument("--control-cont-fields-npz", default=None)
    ap.add_argument("--run-cont-s11-npz", default=None, help="the member's continued _s11 npz ((g): M at the collars)")
    ap.add_argument("--control-cont-s11-npz", default=None)
    ap.add_argument("--v2-json", default=None, help="version 2's delivered json (its series at 40 compared with the delivered records)")
    ap.add_argument("--budget-keys", default=None, help="JSON: budget name -> dotted path in the record, when the names differ from evolution_exact, field_work, reference_reset")
    ap.add_argument("--exterior-keys", default=None, help="JSON: dotted paths of 5d's exterior entries in the record (found by name otherwise)")
    ap.add_argument("--slice-keys", default=None, help="JSON (or @file): {'light': {'h':..., 'C':..., 'P':..., 'orbitals':..., 'P_orbitals':..., 'occ':...}, 'heavy': {...}} with {k} for kappa")
    ap.add_argument("--slice-clock-regex", default=CLOCK_REGEX, help="the clock from the filename when the slice carries neither T_over_rs nor T")
    ap.add_argument("--index-layout", default="auto", choices=("auto", "site_major", "component_major"), help="how operator indices map to sites when the dimension is a multiple of the sites")
    ap.add_argument("--h-builder", default=None, help="module:function building h from a slice mapping: f(npz, field, kappa) -> the channel's one-body operator (a dense array or a scipy sparse matrix; C_readout_v3_hbuilder:h_of returns the engine's host CSR) when h is not stored; the module's configure(args, args_control) is called with the continued runs' args when it exists")
    ap.add_argument("--engine-dir", default=None, help="prepended to sys.path for --h-builder (the busi copies of the engine, engine_twofield_v3.py; the adapter appends the solve lane's directories); with --selftest, where the adapter's manufactured check imports the engine from")
    ap.add_argument("--heavy-channels", default="all", help="the heavy channels read (a comma list of kappa; default all, 1..args.kappa_sea); a narrowing is a reading decision and is printed in the md")
    ap.add_argument("--heavy-clocks", default="", help="the heavy field's slice clocks read (a comma list of T/r_s; default every clock the slices carry); a subset is a reading decision and is printed in the md")
    ap.add_argument("--collar-8", action="store_true", help="include the 8 r_s collar")
    ap.add_argument("--window-start", type=float, default=40.0)
    ap.add_argument("--md5-slices", action="store_true", help="md5 every slice file (slow; the facts record them as md5-verified on the box)")
    ap.add_argument("--inspect-slices", action="store_true", help="list the slices (clocks, keys, shapes, dtypes) and exit")
    ap.add_argument("--allow-arg-mismatch", action="store_true")
    ap.add_argument("--out-dir", default="busi/results/compC")
    ap.add_argument("--tag", default="C_readout_s4_v3")
    ap.add_argument("--quiet", action="store_true", help="no md on stdout and no per-clock progress on stderr")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print("%s self-test: manufactured operators and records only" % VERSION)
        return selftest(a.engine_dir)
    o = {k.replace("-", "_"): v for k, v in vars(a).items()}
    for key in ("run_json", "run_cont_json", "control_json", "control_cont_json", "slices", "control_slices"):
        if not o.get(key):
            ap.error("--%s is required" % key.replace("_", "-"))
    o["slices"] = [s for s in o["slices"].split(",") if s]
    o["control_slices"] = [s for s in o["control_slices"].split(",") if s]
    o["heavy_clocks"] = [float(x) for x in str(o.get("heavy_clocks") or "").split(",") if x.strip()]
    try:
        for key in ("budget_keys", "exterior_keys", "slice_keys"):
            o[key] = parse_jsonish(o.get(key))
        return run_v3(o)
    except ReadoutError as e:
        print("STOP (%s): %s" % (type(e).__name__, e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
