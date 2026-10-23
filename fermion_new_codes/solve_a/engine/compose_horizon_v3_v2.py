#!/usr/bin/env python
"""compose_horizon_v3_v2.py (the solve lane; designed by reviewer A for the lane's builder; reorientation 4 item 7 and C11, engine spec
49.1 (1) and 52.2 (1); insert I8). Version 1's composer (compose_horizon_v3_v1.py, never edited) with its binning constants (the 3 shares,
the 4 edges) and its closure identities imported from horizon3_bins_v1.py, the one module behind the export, the dust script and this
composer, and that module's RULE paragraph (the binning rule; what horizon3_selftest_v1's four failures were) printed after the definitions
and kept in the json; otherwise version 1's text and tables.
Composes HORIZON_single_object_v3.md, .json and _arrays.npz from B version 3's six reruns (B3_{hist,fixed,dust}_{prod,s8}.json and .npz;
B3_fixed_s8 from its pod 4 rerun, which supersedes pod 3's) in version 2's form: the label; the definitions with the corrections of
review_HORIZON_single_object_v2 and review_TRANSFER_class_v1; one section per file and launch with the header, three per-column tables (the
launches and launch spectra, the transfer, the end spectra with section 49's threshold columns: each band edge's content within its
half-width neither band nor bound, every bin read at the buffer and at twice it, the difference the bin's threshold term), the signed
readout, its window controls and the signed series; the checks (closure: shares at a buffer + threshold total = raw shares; the error
identity; the comparator identity; the json's end weights against the npz's last row); a comparison section (version 2's cited numbers
searched among the reruns' numeric leaves; B version 2's json files, when given, compared leaf by leaf beyond the added threshold columns,
equal within the named tolerance or the difference printed) and a paragraph for insert I8.
Every number in the outputs is read from the files named on the command line; the script computes nothing beyond sums and differences of
those numbers. Parts A, D and E of version 2 are not rerun and stand as delivered. CPU; no plotting; no dates; refuses to overwrite.
Usage (box): python compose_horizon_v3_v2.py --out-dir OUT [--files TAG=JSON[,NPZ] ...] [--v2-json TAG=JSON ...] [--cited "LABEL=VALUE" ...]
"""
import os
import sys
import json
import math
import hashlib
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import horizon3_bins_v1 as HB   # noqa: E402

BULK = "<internal>/horizon_v3"
DEFAULT_FILES = [
    ("hist_prod", "production rs64_sigma4 (r_s/a 64, sigma/a 4, ma 0.2); the solved history record by record to the last record held",
     BULK + "/pod3_mirror/out/B3_hist_prod.json", BULK + "/pod3_mirror/out/B3_hist_prod.npz", "pod 3's rerun"),
    ("hist_s8", "rs64_sigma8 (r_s/a 64, sigma/a 8, ma 0.2); the solved history record by record to the last record held",
     BULK + "/pod3_mirror/out/B3_hist_s8.json", BULK + "/pod3_mirror/out/B3_hist_s8.npz", "pod 3's rerun"),
    ("fixed_prod", "production rs64_sigma4; the launch record's applied fields held fixed (300 r_s, finite-wall reflections included after the printed clock)",
     BULK + "/pod3_mirror/out/B3_fixed_prod.json", BULK + "/pod3_mirror/out/B3_fixed_prod.npz", "pod 3's rerun"),
    ("fixed_s8", "rs64_sigma8; the launch record's applied fields held fixed (300 r_s, finite-wall reflections included after the printed clock)",
     BULK + "/pod4_s8r/B3_fixed_s8.json", BULK + "/pod4_s8r/B3_fixed_s8.npz", "pod 4's rerun (supersedes pod3_mirror's B3_fixed_s8)"),
    ("dust_prod", "production rs64_sigma4; mode dust (the file's own launches, structure read as found)",
     BULK + "/pod3_mirror/out/B3_dust_prod.json", BULK + "/pod3_mirror/out/B3_dust_prod.npz", "pod 3's rerun"),
    ("dust_s8", "rs64_sigma8; mode dust (the file's own launches, structure read as found)",
     BULK + "/pod3_mirror/out/B3_dust_s8.json", BULK + "/pod3_mirror/out/B3_dust_s8.npz", "pod 3's rerun"),
]
# version 2's printed numbers as the reviews cite them (review_TRANSFER_class_v1 item 2; review_HORIZON_single_object_v2): searched among
# the reruns' numeric leaves; the labels say where version 2 printed them
DEFAULT_CITED = [
    ("review_TRANSFER_class_v1 item 2: launch 15, kappa 1, column 4, the last pre-reflection share beyond 16 r_s (version 2's fixed-field run)", "0.99697"),
    ("review_TRANSFER_class_v1 item 2: launch 15, kappa 1, column 10, the last pre-reflection share beyond 16 r_s (version 2's fixed-field run)", "0.73311"),
    ("review_TRANSFER_class_v1 item 2: launch 15, kappa 1, column 4, the half-share arrival after launch in r_s", "22.48"),
    ("review_TRANSFER_class_v1 item 2: launch 15, kappa 1, column 10, the half-share arrival after launch in r_s", "25.82"),
    ("review_HORIZON_single_object_v2: the dust weight beyond the root at T/r_s 15, production kappa 1 (version 2's part B.5)", "0.17040405"),
    ("review_HORIZON_single_object_v2: the dust weight beyond the root on the stored 40 slice, production kappa 1 (part B.5)", "1.57778813"),
    ("review_HORIZON_single_object_v2: the positive-intrinsic-band outgoing weight on the 40 slice (part D)", "1.0611"),
    ("review_HORIZON_single_object_v2: the positive-intrinsic-band incoming weight on the 40 slice (part D)", "0.4818"),
    ("review_HORIZON_single_object_v2: the instantaneous dust number current through the cut on the 40 slice (part B.5)", "-0.00043035796"),
    ("review_HORIZON_single_object_v2: E's beyond-16 r_s column, production 15 kappa 1, first value (part E, not rerun)", "1.24443e-16"),
    ("review_HORIZON_single_object_v2: E's beyond-16 r_s column, production 15 kappa 1, second value (part E, not rerun)", "-1.39153e-10"),
]
SHARES = HB.SHARES
EDGES = HB.EDGE_NAMES
END_KEYS = [("beyond_outer_root", "beyond R_t"), ("beyond_2GE", "beyond 2GE"), ("beyond_16rs", "beyond 16 r_s"),
            ("collar_16_24", "collar 16-24 r_s"), ("far_32_48", "far collar 32-48 r_s"), ("beyond_Rcut", "beyond R_cut 24 r_s")]
END_TO_NPZ = {"beyond_outer_root": "w_beyond_outer_root", "beyond_2GE": "w_beyond_2GE", "beyond_16rs": "w_beyond_16rs",
              "collar_16_24": "w_collar_16_24", "far_32_48": "w_far_32_48", "beyond_Rcut": "w_beyond_Rcut"}
SIGNED_Q = [("plus_out", "+ out"), ("plus_in", "+ in"), ("minus_out", "- out"), ("minus_in", "- in"),
            ("plus_pi4_pi2", "+ on pi/4 < q < pi/2"), ("minus_negpi2_negpi4", "- on -pi/2 < q < -pi/4"),
            ("unsigned_pi4_pi2", "unsigned pi/4 <= |q| < pi/2"), ("total", "total")]
SQ5 = SIGNED_Q[:5]
LAUNCH_BIG = ("columns", "signed_launch", "signed_launch_window_controls", "signed_series")
KNOWN_COL_KEYS = {"launch", "spectrum_launch", "omega_bar", "soft", "w_above_m", "w_below_minus_m", "w_in_gap",
                  "beyond16_final_before_reflection", "last_pre_reflection_after_launch_rs", "beyond_Rcut_last_pre_reflection",
                  "arrival_after_launch_rs", "arrival_first_step_after_launch_rs", "pred_2rs_log", "pred_kin_log",
                  "reflection_clock_after_launch_rs", "interior_end", "classifier_overlap_end", "classifier_overlap_launch",
                  "max_radius_rs", "end", "spectrum_end", "end_includes_reflection", "comparator_column",
                  "comparator_arrival_after_launch_rs", "delay_minus_comparator_rs"}
HDR1 = ["col", "depth", "comparator", "R_c (a)", "centre < R_inner", "a k_0", "sigma (a)", "N; v at the centre", "omega_bar (1/a)", "soft",
        "launch shares raw: below -m; |omega| < m; above m", "threshold content at launch, buffer 1: -b; -m; m; b",
        "threshold total: buffer 1; buffer 2; term", "shares at buffer 1 (threshold term): below -m; gap; above m"]
HDR2 = ["col", "beyond 16 r_s, last pre-reflection", "its clock after launch (r_s)", "beyond R_cut there", "arrival (half share) after launch (r_s)",
        "arrival, first step", "comparator col; its arrival; delay (r_s)", "2 r_s ln(1/(a omega_bar)); kappa_in^-1 ln(1/(a k_0))",
        "reflection clock after launch (r_s)", "max radius (r_s)", "end: beyond R_t; beyond 2GE; beyond 16 r_s; collar 16-24; far 32-48; beyond R_cut 24 r_s",
        "end includes reflection", "interior at the end", "classifier overlap: launch; end"]
HDR3 = ["col", "end shares raw: below -m; |omega| < m; above m", "mean; spread (1/a)", "threshold content at the end, buffer 1: -b; -m; m; b",
        "buffer 2: -b; -m; m; b", "threshold total: buffer 1; buffer 2; term", "shares at buffer 1 (threshold term): below -m; gap; above m",
        "closure buffer 1; buffer 2; error identity"]

TITLE = ("# HORIZON_single_object_v3: the single object's own cross-surface block and transfer, version 3 (B version 3's six reruns with the "
         "threshold columns; engine spec 49.1 (1), 52.2 (1); reorientation 4 item 7; insert I8)")
LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massive channel ma 0.2 only "
         "(the owner's one-object ruling, D15 and D14: no second preparation, no massless channel run); the threshold columns at each band "
         "edge assigned to neither band nor bound content; every share a finite-cut response of a specified preparation on one object's record, "
         "not an object-independent fraction and not escape.")
COMPOSED = ("Composed by compose_horizon_v3_v2.py (md5 %s; its binning constants and closure identities from horizon3_bins_v1.py, md5 %s) from the "
            "files listed with their md5 at the end; every number below is read from those files by that script (where a check is printed it is a "
            "sum or difference of the files' own numbers). Version 2 (HORIZON_single_object_v2.md) stands as delivered, its parts A (the "
            "correlations across the root), D (the signed readout's controls) and E (the detector density) not rerun; this version's content is B "
            "version 3, the six reruns, one per mode (history, fixed field, dust) and preparation (production rs64_sigma4, rs64_sigma8), each in "
            "its producer's launch order, B3_fixed_s8 from its pod 4 rerun. The raw arrays (the 80-bin histograms at every buffer, the signed "
            "readouts and series, the npz time series) are in %s_arrays.npz under keys named in the json.")
DEFINITIONS = [
    "- (iii) Fixed-field: a record's applied fields held fixed while a column propagates (the launch record's h for 300 r_s; the 300 r_s "
    "endpoints include finite-wall reflections, so every share printed as a result is the last pre-reflection one, its clock beside it); "
    "history: the solved history record by record (piecewise-constant applied fields) to the last record held when each run started "
    "(production B ends at 47.5 r_s).",
    "- (i) The comparator: per launch record and channel a column at R_t + 2 sigma outside the outer root with each k_0 and envelope, beside "
    "the interior columns (the annulus middle and 2 sigma inside R_t). The arrival: the first crossing of half the column's own terminal "
    "pre-reflection weight beyond 16 r_s, interpolated linearly between steps of 0.25 r_s (fixed field) or between records (history). The "
    "delay: the arrival minus that of the comparator with the same carrier and envelope at its different location (not matched Killing "
    "energy), printed beside 2 r_s ln(1/(a omega_bar)) and kappa_in^-1 ln(1/(a k_0)) (absolute frequency and carrier in their guarded "
    "domains, the launch record's 2GE distinct from the nominal r_s): not universal delay laws.",
    "- (viii) The 99 percent classifier (the eigenvectors of the held h whose weight inside R_cut 24 r_s is at least 0.99) is a finite-box "
    "localization classifier, not the infinite-exterior bound-state projector; each column's Killing weight above m is printed beside its "
    "mean energy omega_bar. Weight above m is not escape: neither omega > m, nor a local outward velocity, nor weight beyond a finite cut "
    "certifies escape, and the fixed-charge radial exterior (N = 1, v = -sqrt(r_s/R)) holds infinitely many bound levels above "
    "b_1 = sqrt(m^2 + a^-2) accumulating at b_1 (review_TRANSFER_class_v1); no infinite tower or asymptotic scattering certificate is "
    "asserted for the finite closed chain.",
    "- (xi) The threshold column (engine spec 49.1 (1)): at each band edge the bins use (b, m, -m, -b) the content within b r_s/(2 R_max) of "
    "+-b and within m r_s/(2 R_max) of +-m is 'threshold content, extent comparable to the chain', assigned to neither band nor bound "
    "content (a level that close to the edge extends over the chain and the chain's end displaces it; its side of the edge is a "
    "chain-length statement); b is the sine chain's band top at the record's m, r_s the record's own 2GE, R_max the chain's end; every bin "
    "is read at that buffer and at twice it, the difference printed as the bin's threshold term; the busi lane's 3.3e-3/a is their reading "
    "of the same interval, printed as the comparator.",
    "- (xii) Lapse and both energy signs: the asymptotic gap is |omega| < N_inf m and the essential edges sit at +-N_inf m and +-N_inf b_1 "
    "(cubic: +-N_inf b_3, b_3 = sqrt(m^2 + 3 a^-2)); the files' bins split at +-m and +-b, the producer's edges; the lapse displacement "
    "(1 - N_inf) m is printed per launch against the threshold half-width, which covers it. Nominal bins do not certify exact spectral "
    "content; the 80-bin histograms are saved raw.",
    "- (x) Every readout of weight beyond the outer root is a transfer beyond the moving root within the run; the six end-region entries are, "
    "in order, beyond R_t, beyond 2GE, beyond 16 r_s, the collar 16 to 24 r_s, the far collar 32 to 48 r_s and beyond R_cut = 24 r_s. "
    "Initial exterior tails, coherent cross blocks and moving-root membership are retained: exterior population is not automatically the "
    "transfer of an initially inside projected state.",
    "- (vi) The signed readout: the region's weight split by sine tapers of 32 sites with half overlap, each piece's transform zero-padded to "
    "128, projected at each piece's centre on the local positive intrinsic Dirac branch with the local group-velocity sign (+ out, + in, the "
    "- branch's, and + on the signed interval pi/4 < q < pi/2); positive windowed effects, not global spectral projectors or fluxes, the 16- "
    "and 64-site tapers sensitivity controls; intrinsic positivity does not require positive Killing energy; the unsigned share at "
    "pi/4 <= |ak| < pi/2 is kept separate.",
    "- Launch centres: a column whose centre lies below the launch record's inner root is flagged per column; 'never launches below the inner "
    "root' holds only of the launch-15 subset (the production's launch 30 holds centres below its inner root).",
    "- Not in these files: the applied-field zeros of N + v (DUST_root_history version 2 carries them, the dense roots beside); parts A, D, E "
    "of version 2 (not rerun); any massless channel (owner decision D14, not run); no mutual information anywhere.",
]
NOT_SAYABLE = ("no permanent escape, infinity particle flux, closed charge-loss ledger, physical Page turnover or completed-history no-horizon "
               "result follows from these numbers; the finite-cut shares are responses of specified preparations on one object's record, per "
               "radial channel, interior beyond the hydrodynamic limit; the threshold columns belong to neither band nor bound content; the same "
               "quantity has not been compared across the two descriptions.")


def md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def isnum(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def fmt(x, nd=6):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "True" if x else "False"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return ("%%.%dg" % nd) % x
    return str(x).replace("|", "/")


def lst(xs, nd=6):
    if not isinstance(xs, (list, tuple, np.ndarray)):
        return fmt(xs, nd)
    return "[" + ", ".join(fmt(x, nd) for x in xs) + "]"


def g(d, path, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def arr_at(arr, j):
    if isinstance(arr, list) and 0 <= j < len(arr):
        return arr[j]
    return None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def flatten(obj, prefix="", out=None, skip_keys=(), id_key="key"):
    """numeric/boolean/None leaves by path; a list of dicts carrying id_key is indexed by it (launches[T15p000_k1]), others by position."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            flatten(v, "%s.%s" % (prefix, k) if prefix else str(k), out, skip_keys, id_key)
    elif isinstance(obj, list):
        keyed = bool(obj) and all(isinstance(x, dict) and isinstance(x.get(id_key), str) for x in obj)
        for i, x in enumerate(obj):
            sub = "%s[%s]" % (prefix, x[id_key]) if keyed else "%s[%d]" % (prefix, i)
            flatten(x, sub, out, skip_keys, id_key)
    else:
        out[prefix] = obj
    return out


def compare_flat(a, b, tol_abs, tol_rel, exclude=()):
    """a: version 2's leaves, b: version 3's; equal within tol_abs + tol_rel * max(|x|, |y|); else the difference."""
    common = [p for p in a if p in b and not any(s in p for s in exclude)]
    n_eq, diffs = 0, []
    for p in common:
        x, y = a[p], b[p]
        if isnum(x) and isnum(y):
            d = abs(float(x) - float(y))
            if d <= tol_abs + tol_rel * max(abs(float(x)), abs(float(y))):
                n_eq += 1
            else:
                diffs.append({"path": p, "v2": x, "v3": y, "v3_minus_v2": float(y) - float(x)})
        elif x == y:
            n_eq += 1
        else:
            diffs.append({"path": p, "v2": x, "v3": y, "v3_minus_v2": None})
    only_a = [p for p in a if p not in b]
    only_b = [p for p in b if p not in a]
    return {"n_common": len(common), "n_equal": n_eq, "n_differing": len(diffs), "diffs": diffs, "n_only_v2": len(only_a),
            "n_only_v3": len(only_b), "only_v2_examples": only_a[:20], "only_v3_examples": only_b[:20]}


def parse_cited(items, rel_tol):
    """'LABEL=VALUE' -> (label, value, tolerance = half the unit of the last printed digit + rel_tol * |value|)."""
    out = []
    for it in items:
        label, _, vs = it.rpartition("=")
        vs = vs.strip()
        val = float(vs)
        mant = vs.lower().split("e")[0]
        dec = len(mant.split(".")[1]) if "." in mant else 0
        scale = 10.0 ** int(vs.lower().split("e")[1]) if "e" in vs.lower() else 1.0
        out.append((label.strip(), val, 0.5 * 10.0 ** (-dec) * scale + rel_tol * abs(val)))
    return out


def search_cited(flat_by, val, tol):
    hits, nearest = [], None
    for tag, flat in flat_by.items():
        for p, x in flat.items():
            if not isnum(x):
                continue
            d = abs(float(x) - val)
            if d <= tol:
                hits.append({"file": tag, "path": p, "value": x})
            if nearest is None or d < nearest["abs_diff"]:
                nearest = {"file": tag, "path": p, "value": x, "abs_diff": d}
    return hits, nearest


def is_num_list(v, max_len):
    return isinstance(v, list) and len(v) > max_len and all(isnum(x) or x is None for x in v)


def split_arrays(obj, prefix, store, max_len=24):
    """a copy of obj without numeric lists longer than max_len; those go to store[...] as float arrays (None -> nan), the copy keeps a pointer."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            sub = "%s__%s" % (prefix, k)
            if is_num_list(v, max_len):
                store[sub] = np.array([np.nan if x is None else float(x) for x in v])
                out[k] = {"array_in_npz": sub, "len": len(v)}
            else:
                out[k] = split_arrays(v, sub, store, max_len)
        return out
    if isinstance(obj, list):
        return [split_arrays(x, "%s__%d" % (prefix, i), store, max_len) for i, x in enumerate(obj)]
    return obj


def thresholds_of(sp):
    t = g(sp, "v3_threshold") if isinstance(sp, dict) else None
    return t if isinstance(t, dict) else {}


def closure_checks(sp):
    """closure[mu] = (3 shares at buffer mu) + threshold total(mu) - (3 raw shares); error identity: max |(|b1 - b2|) - printed error|
    (the identities of horizon3_bins_v1; sums and differences of the file's own numbers)."""
    res = {"shares_minus_norm": None, "closure": {}, "error_identity": None}
    if not isinstance(sp, dict):
        return res
    try:
        raw = [float(sp[k]) for k in SHARES]
    except (KeyError, TypeError, ValueError):
        return res
    if isnum(sp.get("norm")):
        res["shares_minus_norm"] = sum(raw) - float(sp["norm"])
    t = thresholds_of(sp)
    B = t.get("buffers") if isinstance(t.get("buffers"), dict) else {}
    for mu, Bm in B.items():
        try:
            res["closure"][mu] = HB.closure_residual(raw, [float(Bm[k]) for k in SHARES], float(Bm["threshold_total"]))
        except (KeyError, TypeError, ValueError):
            res["closure"][mu] = None
    E = t.get("threshold_error")
    if "1" in B and "2" in B and isinstance(E, dict) and isinstance(B["1"], dict) and isinstance(B["2"], dict):
        res["error_identity"] = HB.error_identity_residual(B["1"], B["2"], E)
    return res


def npz_arr(Z, k):
    return np.asarray(Z[k]) if (Z is not None and k in Z.files) else None


def npz_summary(Z, key, pre, arrays):
    nz = {}
    if Z is None:
        return nz
    t = npz_arr(Z, "t_%s" % key)
    if t is not None:
        nz["n_records"] = int(len(t)); nz["t_first"] = float(t[0]); nz["t_last"] = float(t[-1])
        arrays[pre + "__npz__t"] = t
    Rt = npz_arr(Z, "Rt_%s" % key)
    if Rt is not None:
        fin = np.isfinite(Rt)
        nz["records_with_root"] = int(fin.sum()); nz["Rt_last"] = float(Rt[-1]) if fin[-1] else None
        nz["last_record_with_root"] = (float(t[np.nonzero(fin)[0][-1]]) if (fin.any() and t is not None) else None)
        arrays[pre + "__npz__Rt"] = Rt
    g2 = npz_arr(Z, "2gE_%s" % key)
    if g2 is not None:
        nz["gE2_first"] = float(g2[0]); nz["gE2_last"] = float(g2[-1]); arrays[pre + "__npz__2gE"] = g2
    nm = npz_arr(Z, "norm_%s" % key)
    if nm is not None:
        nz["norm_min"] = float(np.nanmin(nm)); nz["norm_max"] = float(np.nanmax(nm)); arrays[pre + "__npz__norm"] = nm
    rf = npz_arr(Z, "refl_%s" % key)
    if rf is not None and rf.ndim == 2:
        first = []
        for j in range(rf.shape[1]):
            idx = np.nonzero(rf[:, j])[0]
            first.append(float(t[idx[0]]) if (len(idx) and t is not None) else None)
        nz["first_reflection_clock_per_column"] = first; nz["columns_flagged_at_last_record"] = int(rf[-1].sum())
        arrays[pre + "__npz__refl"] = rf
    if "X_end_%s" % key in Z.files:
        X = np.asarray(Z["X_end_%s" % key])
        nz["X_end_shape"] = list(X.shape); nz["X_end_dtype"] = str(X.dtype)
        nz["X_end_norms"] = np.sum(np.abs(X) ** 2, axis=0).tolist()
    nz["end_from_npz"] = {}
    for k in END_TO_NPZ.values():
        arr = npz_arr(Z, "%s_%s" % (k, key))
        if arr is not None and arr.ndim == 2:
            nz["end_from_npz"][k] = arr[-1].tolist(); arrays["%s__npz__%s" % (pre, k)] = arr
    return nz


def leaves_table(obj, md, max_rows, title):
    flat = flatten(obj)
    rows = [[p, fmt(v)] for p, v in list(flat.items())[:max_rows]]
    md.append("")
    md.append(title + (" (%d leaves; the first %d, the rest in the json)" % (len(flat), max_rows) if len(flat) > max_rows else " (%d leaves)" % len(flat)))
    md.extend(table(["leaf", "value"], rows))


def render_launch(F, L, a, series_cols, md, arrays):
    tag = F["tag"]; key = str(L.get("key", "?")); Z = F["Z"]; pre = "%s__%s" % (tag, key)
    roots = "; ".join("%s (slope %s)" % (fmt(r.get("R"), 8), fmt(r.get("slope"), 4)) for r in (L.get("roots_dense") or []) if isinstance(r, dict)) or "-"
    md.append("")
    md.append("#### %s / %s: T/r_s %s (record %s), kappa %s, E %s, 2GE %s a, T_H %s; the record's dense roots %s; R_outer %s, R_inner %s, kappa_in %s; "
              "classifier states %s; eigendecomposition %s s; Chebyshev K max %s, tail %s; spectrum conservation max %s" % (
                  tag, key, fmt(L.get("T_launch_over_rs")), fmt(L.get("k0")), fmt(L.get("kappa")), fmt(L.get("E"), 10), fmt(L.get("two_gE"), 10),
                  fmt(L.get("T_H")), roots, fmt(L.get("R_outer"), 8), fmt(L.get("R_inner"), 8), fmt(L.get("kappa_in"), 5), fmt(L.get("classifier_states")),
                  fmt(L.get("eig_seconds"), 3), fmt(L.get("cheb_K_max")), fmt(L.get("cheb_tail_acc"), 3), fmt(L.get("spectrum_conservation_max"), 3)))
    D = {"key": key, "header": split_arrays({k: v for k, v in L.items() if k not in LAUNCH_BIG}, pre + "__header", arrays), "npz": {}, "columns": [], "checks": {}}
    nz = npz_summary(Z, key, pre, arrays)
    D["npz"] = nz
    if nz:
        md.append("The npz: %s records from T/r_s %s to %s; %s records with a root, the last at %s (R_t %s a); 2GE %s to %s a; the norm series %s to %s; "
                  "columns flagged reflected at the last record %s; first reflection clock per column %s; X_end %s with the end norms %s." % (
                      fmt(nz.get("n_records")), fmt(nz.get("t_first")), fmt(nz.get("t_last")), fmt(nz.get("records_with_root")), fmt(nz.get("last_record_with_root")),
                      fmt(nz.get("Rt_last"), 8), fmt(nz.get("gE2_first"), 8), fmt(nz.get("gE2_last"), 8), fmt(nz.get("norm_min"), 10), fmt(nz.get("norm_max"), 10),
                      fmt(nz.get("columns_flagged_at_last_record")), lst(nz.get("first_reflection_clock_per_column"), 5), lst(nz.get("X_end_shape")), lst(nz.get("X_end_norms"), 8)))
    cols = L.get("columns") if isinstance(L.get("columns"), list) else None
    ncol = len(cols) if cols else 0
    tp = {}
    if cols:
        for c in cols:
            if isinstance(c, dict):
                tp = thresholds_of(c.get("spectrum_launch")) or thresholds_of(c.get("spectrum_end"))
                if tp:
                    break
    if tp:
        b_ = tp.get("b"); hw_b, hw_m = tp.get("half_width_b"), tp.get("half_width_m")
        disp_m = (1.0 - a.N_inf) * a.m; disp_b = (1.0 - a.N_inf) * b_ if isnum(b_) else None
        md.append("Threshold columns (49.1 (1)): b = %s/a (b - sqrt(m^2 + 1/a^2) = %s at m = %s/a), r_s = the record's own 2GE = %s a, R_max = %s a: half-width %s/a at +-b and %s/a at +-m "
                  "(the busi lane's reading %s/a, their comparator); label '%s'. The lapse's edge displacement (1 - N_inf) m = %s/a and (1 - N_inf) b = %s/a at N_inf = %s "
                  "(the review's tail value, a named argument), inside the half-widths by the factors %s and %s." % (
                      fmt(b_, 10), fmt(b_ - HB.band_top(a.m), 3) if isnum(b_) else "-", fmt(a.m), fmt(tp.get("r_s"), 10), fmt(tp.get("R_max")), fmt(hw_b, 6), fmt(hw_m, 6),
                      fmt(tp.get("comparator_busi_per_a")), fmt(tp.get("label")), fmt(disp_m, 3), fmt(disp_b, 3), fmt(a.N_inf, 12),
                      fmt(hw_m / disp_m, 3) if (isnum(hw_m) and disp_m > 0) else "-", fmt(hw_b / disp_b, 3) if (isnum(hw_b) and isnum(disp_b) and disp_b > 0) else "-"))
        D["threshold_params"] = {"b": b_, "r_s": tp.get("r_s"), "R_max": tp.get("R_max"), "half_width_b": hw_b, "half_width_m": hw_m,
                                 "comparator_busi_per_a": tp.get("comparator_busi_per_a"), "label": tp.get("label"), "lapse_displacement_m": disp_m,
                                 "lapse_displacement_b": disp_b, "N_inf_used": a.N_inf}
    maxes = {"closure": None, "error_identity": None, "shares_minus_norm": None, "comparator": None, "json_npz_end": None}

    def upd(k, v):
        if isnum(v) and np.isfinite(v):
            maxes[k] = v if maxes[k] is None else max(maxes[k], v)

    if cols and all(isinstance(c, dict) for c in cols):
        R_in = L.get("R_inner"); rows1, rows2, rows3, extras = [], [], [], []
        below_count = 0
        for j, c in enumerate(cols):
            la = c.get("launch") if isinstance(c.get("launch"), dict) else {}
            spL = c.get("spectrum_launch") if isinstance(c.get("spectrum_launch"), dict) else {}
            spE = c.get("spectrum_end") if isinstance(c.get("spectrum_end"), dict) else {}
            tL, tE = thresholds_of(spL), thresholds_of(spE)
            BL = tL.get("buffers") if isinstance(tL.get("buffers"), dict) else {}
            BE = tE.get("buffers") if isinstance(tE.get("buffers"), dict) else {}
            R_c = la.get("R_c"); below = bool(isnum(R_c) and isnum(R_in) and R_c < R_in); below_count += int(below)
            rows1.append([j, fmt(la.get("depth")), fmt(la.get("comparator")), fmt(R_c, 7), fmt(below), fmt(la.get("ak0_name", la.get("k0"))), fmt(la.get("sigma")),
                          "%s; %s" % (fmt(la.get("N_at")), fmt(la.get("v_at"))), fmt(c.get("omega_bar")), fmt(c.get("soft")),
                          "; ".join(fmt(spL.get(k)) for k in SHARES), "; ".join(fmt(g(BL, "1.threshold_weight." + e), 4) for e in EDGES),
                          "%s; %s; %s" % (fmt(g(BL, "1.threshold_total"), 4), fmt(g(BL, "2.threshold_total"), 4), fmt(g(tL, "threshold_error.threshold_total"), 4)),
                          "; ".join("%s (%s)" % (fmt(g(BL, "1." + k)), fmt(g(tL, "threshold_error." + k), 3)) for k in SHARES)])
            e_ = c.get("end") if isinstance(c.get("end"), dict) else {}
            rows2.append([j, fmt(c.get("beyond16_final_before_reflection")), fmt(c.get("last_pre_reflection_after_launch_rs")), fmt(c.get("beyond_Rcut_last_pre_reflection"), 3),
                          fmt(c.get("arrival_after_launch_rs")), fmt(c.get("arrival_first_step_after_launch_rs")),
                          "%s; %s; %s" % (fmt(c.get("comparator_column")), fmt(c.get("comparator_arrival_after_launch_rs")), fmt(c.get("delay_minus_comparator_rs"))),
                          "%s; %s" % (fmt(c.get("pred_2rs_log")), fmt(c.get("pred_kin_log"))), fmt(c.get("reflection_clock_after_launch_rs")), fmt(c.get("max_radius_rs")),
                          "; ".join(fmt(e_.get(k)) for k, _ in END_KEYS), fmt(c.get("end_includes_reflection")), fmt(c.get("interior_end")),
                          "%s; %s" % (fmt(c.get("classifier_overlap_launch")), fmt(c.get("classifier_overlap_end")))])
            chL, chE = closure_checks(spL), closure_checks(spE)
            rows3.append([j, "; ".join(fmt(spE.get(k)) for k in SHARES), "%s; %s" % (fmt(spE.get("mean")), fmt(spE.get("spread"))),
                          "; ".join(fmt(g(BE, "1.threshold_weight." + e), 4) for e in EDGES), "; ".join(fmt(g(BE, "2.threshold_weight." + e), 4) for e in EDGES),
                          "%s; %s; %s" % (fmt(g(BE, "1.threshold_total"), 4), fmt(g(BE, "2.threshold_total"), 4), fmt(g(tE, "threshold_error.threshold_total"), 4)),
                          "; ".join("%s (%s)" % (fmt(g(BE, "1." + k)), fmt(g(tE, "threshold_error." + k), 3)) for k in SHARES),
                          "%s; %s; %s" % (fmt(chE["closure"].get("1"), 2), fmt(chE["closure"].get("2"), 2), fmt(chE["error_identity"], 2))])
            cj = {"closure_launch": chL, "closure_end": chE, "centre_below_inner_root": below}
            for ch in (chL, chE):
                for v in ch["closure"].values():
                    upd("closure", abs(v) if isnum(v) else None)
                upd("error_identity", ch["error_identity"]); upd("shares_minus_norm", abs(ch["shares_minus_norm"]) if isnum(ch["shares_minus_norm"]) else None)
            cc = c.get("comparator_column")
            if isinstance(cc, int) and not isinstance(cc, bool) and 0 <= cc < ncol and isinstance(cols[cc], dict):
                ca = cols[cc].get("arrival_after_launch_rs")
                r1 = abs(ca - c["comparator_arrival_after_launch_rs"]) if (isnum(ca) and isnum(c.get("comparator_arrival_after_launch_rs"))) else None
                r2 = (abs((c["arrival_after_launch_rs"] - ca) - c["delay_minus_comparator_rs"])
                      if (isnum(ca) and isnum(c.get("arrival_after_launch_rs")) and isnum(c.get("delay_minus_comparator_rs"))) else None)
                cj["comparator_residuals"] = [r1, r2]; upd("comparator", r1); upd("comparator", r2)
            if nz.get("end_from_npz"):
                dd = []
                for k, _ in END_KEYS:
                    arrk = nz["end_from_npz"].get(END_TO_NPZ[k])
                    if arrk is not None and j < len(arrk) and isnum(e_.get(k)) and isnum(arrk[j]) and np.isfinite(arrk[j]):
                        dd.append(abs(float(e_[k]) - float(arrk[j])))
                if dd:
                    cj["json_minus_npz_end_max"] = max(dd); upd("json_npz_end", max(dd))
            ex = {k: v for k, v in c.items() if k not in KNOWN_COL_KEYS}
            if ex:
                for p, v in flatten(ex).items():
                    extras.append([j, p, fmt(v)])
            Dc = split_arrays(c, "%s__c%d" % (pre, j), arrays); Dc["checks"] = cj; D["columns"].append(Dc)
        md.append(""); md.append("Table 1, the launches and the launch spectra (shares raw and at buffer 1 with the threshold term in parentheses):")
        md.extend(table(HDR1, rows1))
        md.append(""); md.append("Table 2, the transfer (the arrival the first crossing of half the column's own last pre-reflection weight beyond 16 r_s; the end entries in the ruled order):")
        md.extend(table(HDR2, rows2))
        md.append(""); md.append("Table 3, the end spectra with the threshold columns (buffers 1 and 2; closure = shares at the buffer + threshold total - raw shares; error identity = max |(|buffer 1 - buffer 2|) - printed term|):")
        md.extend(table(HDR3, rows3))
        if extras:
            md.append(""); md.append("Column fields beyond the known layout (%d leaves; the first %d):" % (len(extras), min(len(extras), a.max_leaf_rows)))
            md.extend(table(["col", "leaf", "value"], extras[:a.max_leaf_rows]))
        D["checks"] = {"max_abs_closure": maxes["closure"], "max_error_identity_residual": maxes["error_identity"], "max_abs_shares_minus_norm": maxes["shares_minus_norm"],
                       "max_comparator_residual": maxes["comparator"], "max_json_minus_npz_end": maxes["json_npz_end"],
                       "columns_with_centre_below_inner_root": below_count, "n_columns": ncol}
    else:
        leaves_table({k: v for k, v in L.items() if k not in ("signed_launch", "signed_launch_window_controls", "signed_series")}, md, a.max_leaf_rows,
                     "This launch holds no per-column structure of version 2's layout; its leaves")
        D["checks"] = {"n_columns": 0}
    S = L.get("signed_launch")
    if isinstance(S, dict):
        nsig = max([len(v) for D_ in S.values() if isinstance(D_, dict) for v in D_.values() if isinstance(v, list)] or [0])
        rows, meta = [], []
        for reg in ("whole", "inside_Rt", "beyond_Rt"):
            D_ = S.get(reg)
            if not isinstance(D_, dict):
                continue
            meta.append("%s: window %s, pad %s, region %s, sites %s, windows %s, N <= 0 windows %s, v > 0 windows %s" % (
                reg, fmt(D_.get("window")), fmt(D_.get("pad")), lst(D_.get("region"), 8), lst(D_.get("sites")), fmt(D_.get("windows")),
                fmt(D_.get("N_nonpositive_windows")), fmt(D_.get("v_positive_windows"))))
            for q, _ in SIGNED_Q:
                v = D_.get(q)
                if isinstance(v, list):
                    arrays["%s__signed_launch__%s__%s" % (pre, reg, q)] = np.array([float(x) if isnum(x) else np.nan for x in v])
            for j in range(nsig):
                rows.append([reg, j] + [fmt(arr_at(D_.get(q), j)) for q, _ in SIGNED_Q])
        md.append(""); md.append("Table 4, the signed launch readout (32-site tapers; positive windowed effects, not fluxes; the columns indexed as in the file's arrays): " + "; ".join(meta))
        md.extend(table(["region", "col"] + [lab for _, lab in SIGNED_Q], rows))
        D["signed_launch"] = S
        W = L.get("signed_launch_window_controls")
        if isinstance(W, dict) and isinstance(S.get("whole"), dict):
            crow = []
            for wname, Dw in W.items():
                if not isinstance(Dw, dict):
                    continue
                cells = []
                for q, _ in SIGNED_Q:
                    base, ctrl = S["whole"].get(q), Dw.get(q)
                    if isinstance(ctrl, list):
                        arrays["%s__signed_control_%s__%s" % (pre, wname, q)] = np.array([float(x) if isnum(x) else np.nan for x in ctrl])
                    if isinstance(base, list) and isinstance(ctrl, list):
                        dd = [abs(float(x) - float(y)) for x, y in zip(base, ctrl) if isnum(x) and isnum(y)]
                        cells.append(fmt(max(dd), 3) if dd else "-")
                    else:
                        cells.append("-")
                crow.append([wname, fmt(Dw.get("windows"))] + cells)
            md.append(""); md.append("Window controls on the whole region (sensitivity tests): max over columns of |control - the 32-site readout| per quantity:")
            md.extend(table(["window (sites)", "windows"] + [lab for _, lab in SIGNED_Q], crow))
            D["signed_launch_window_controls"] = W
    SS = L.get("signed_series")
    if isinstance(SS, list) and SS:
        nrow = len(SS)
        nsig2 = max([len(v) for row in SS if isinstance(row, dict) for reg in ("inside_Rt", "beyond_Rt")
                     if isinstance(row.get(reg), dict) for v in row[reg].values() if isinstance(v, list)] or [0])
        tarr, aarr, rarr = np.full(nrow, np.nan), np.full(nrow, np.nan), np.full(nrow, np.nan)
        store = {(reg, q): np.full((nrow, nsig2), np.nan) for reg in ("inside_Rt", "beyond_Rt") for q, _ in SIGNED_Q}
        for i, row in enumerate(SS):
            if not isinstance(row, dict):
                continue
            for arr_, k in ((tarr, "t_over_rs"), (aarr, "after_launch_rs"), (rarr, "R_t")):
                if isnum(row.get(k)):
                    arr_[i] = float(row[k])
            for reg in ("inside_Rt", "beyond_Rt"):
                D_ = row.get(reg)
                if isinstance(D_, dict):
                    for q, _ in SIGNED_Q:
                        v = D_.get(q)
                        if isinstance(v, list):
                            for j, x in enumerate(v[:nsig2]):
                                if isnum(x):
                                    store[(reg, q)][i, j] = float(x)
        arrays[pre + "__series__t_over_rs"] = tarr; arrays[pre + "__series__after_launch_rs"] = aarr; arrays[pre + "__series__R_t"] = rarr
        for (reg, q), arr_ in store.items():
            arrays["%s__series__%s__%s" % (pre, reg, q)] = arr_
        D["series_summary"] = {"n_rows": nrow, "n_columns": nsig2, "t_over_rs": tarr.tolist(), "after_launch_rs": aarr.tolist(), "R_t": rarr.tolist(),
                               "rows_without_root": int(np.isnan(rarr).sum())}
        stride = max(1, int(math.ceil(nrow / float(a.series_max_rows))))
        for j in series_cols:
            if j >= nsig2:
                md.append(""); md.append("(series column %d is not among this launch's %d series columns)" % (j, nsig2)); continue
            rows = []
            for i in range(0, nrow, stride):
                rows.append([fmt(tarr[i]), fmt(aarr[i]), fmt(rarr[i], 7)] + [fmt(store[("inside_Rt", q)][i, j]) for q, _ in SQ5] + [fmt(store[("beyond_Rt", q)][i, j]) for q, _ in SQ5])
            md.append("")
            md.append("Table 5, the signed series of column %d (every %d-th row of %d; the full series in the arrays npz; R_t 'nan' where the record holds no root, the regions then undefined):" % (j, stride, nrow))
            md.extend(table(["T/r_s", "after launch (r_s)", "R_t (a)"] + ["inside R_t: " + lab for _, lab in SQ5] + ["beyond R_t: " + lab for _, lab in SQ5], rows))
    return D


def render_file(F, a, series_cols, md, digest, arrays):
    J = F["J"]; mode = J.get("mode", "-")
    launches = J.get("launches") if isinstance(J.get("launches"), list) else []
    md.append(""); md.append("### %s: mode %s; %s; %s" % (F["tag"], fmt(mode), F["prep"], F["note"]))
    md.append("File %s (md5 %s, %d bytes); npz %s (md5 %s); %d launches: %s." % (
        F["json"], F["json_md5"], F["json_bytes"], fmt(F["npz"]), fmt(F["npz_md5"]), len(launches), ", ".join(fmt(L.get("key")) for L in launches if isinstance(L, dict))))
    top = {k: v for k, v in J.items() if k != "launches"}
    flat_top = flatten(top)
    if flat_top:
        items = list(flat_top.items())
        md.append("Producer fields: " + "; ".join("%s = %s" % (p, fmt(v)) for p, v in items[:a.max_leaf_rows]) +
                  ("" if len(items) <= a.max_leaf_rows else "; ... (%d more in the json)" % (len(items) - a.max_leaf_rows)))
    entry = {"mode": mode, "json": F["json"], "json_md5": F["json_md5"], "npz": F["npz"], "npz_md5": F["npz_md5"], "prep": F["prep"], "note": F["note"],
             "producer_fields": split_arrays(top, F["tag"] + "__producer", arrays), "launches": []}
    for L in launches:
        if isinstance(L, dict):
            entry["launches"].append(render_launch(F, L, a, series_cols, md, arrays))
    digest["per_file"][F["tag"]] = entry


def checks_digest_section(digest, md, a):
    md.append(""); md.append("## Checks digest (sums and differences of the files' own numbers; the tolerance %s a named argument)" % fmt(a.tol_closure))
    rows = []
    for tag, entry in digest["per_file"].items():
        for D in entry["launches"]:
            ch = D.get("checks", {})
            flag = any(isnum(ch.get(k)) and ch.get(k) > a.tol_closure for k in ("max_abs_closure", "max_error_identity_residual", "max_abs_shares_minus_norm", "max_comparator_residual", "max_json_minus_npz_end"))
            rows.append([tag, D["key"], fmt(ch.get("n_columns")), fmt(ch.get("max_abs_closure"), 3), fmt(ch.get("max_error_identity_residual"), 3), fmt(ch.get("max_abs_shares_minus_norm"), 3),
                         fmt(ch.get("max_comparator_residual"), 3), fmt(ch.get("max_json_minus_npz_end"), 3), fmt(ch.get("columns_with_centre_below_inner_root")), "beyond tolerance" if flag else "within"])
            digest["checks_digest"].append({"file": tag, "launch": D["key"], "checks": ch, "beyond_tolerance": flag})
    md.extend(table(["file", "launch", "columns", "max |closure|", "max error-identity residual", "max |shares - norm|", "max comparator residual", "max |json - npz| at the end", "centres below R_inner", "verdict"], rows))


def comparison_section(files, a, md, digest):
    md.append(""); md.append("## Comparison with version 2 (where both exist)")
    flat_by = {F["tag"]: flatten(F["J"], skip_keys=("signed_series",)) for F in files}
    cited = parse_cited(a.cited if a.cited is not None else ["%s=%s" % (lab, val) for lab, val in DEFAULT_CITED], a.cited_rel_tol)
    rows, crec = [], []
    for label, val, tol in cited:
        hits, nearest = search_cited(flat_by, val, tol)
        where = "; ".join("%s: %s = %s" % (h["file"], h["path"], fmt(h["value"], 10)) for h in hits[:8]) + (" (+%d more)" % (len(hits) - 8) if len(hits) > 8 else "")
        near = "%s: %s = %s (|diff| %s)" % (nearest["file"], nearest["path"], fmt(nearest["value"], 10), fmt(nearest["abs_diff"], 3)) if nearest else "-"
        rows.append([label, fmt(val, 12), fmt(tol, 3), len(hits), where if hits else "not found", near])
        crec.append({"label": label, "value": val, "tolerance": tol, "hits": hits, "nearest": nearest})
    md.append("Version 2's numbers as its reviews cite them (a named list; each searched among the six files' numeric leaves outside the signed series, within half the unit of its last printed digit plus %s relative):" % fmt(a.cited_rel_tol))
    md.extend(table(["cited (where version 2 printed it)", "value", "tolerance", "hits", "found at", "nearest leaf"], rows))
    digest["comparison"]["cited"] = crec
    v2 = []
    skip = [s.strip() for s in a.v2_skip.split(",") if s.strip()]
    if a.v2_json:
        md.append("")
        md.append("B version 2's json files leaf by leaf against the reruns (the added threshold columns and the paths containing %s left out; equal within %s + %s relative, else the difference):" % (", ".join(skip), fmt(a.tol_v2_abs), fmt(a.tol_v2_rel)))
        for spec in a.v2_json:
            tag, _, path = spec.partition("=")
            tag = tag.strip(); path = path.strip()
            F = next((x for x in files if x["tag"] == tag), None)
            if F is None or not os.path.exists(path):
                md.append("- %s: version 2 file %s or the rerun's tag missing; not compared." % (tag, path)); v2.append({"tag": tag, "path": path, "error": "missing"}); continue
            with open(path) as fh:
                J2 = json.load(fh)
            res = compare_flat(flatten(J2), flatten(F["J"]), a.tol_v2_abs, a.tol_v2_rel, exclude=tuple(skip))
            res.update({"tag": tag, "path": path, "md5": md5(path)})
            v2.append(res)
            md.append("- %s against %s (md5 %s): %d common leaves, %d equal, %d differing; %d leaves only in version 2, %d only in version 3 (examples: %s / %s)." % (
                tag, path, res["md5"], res["n_common"], res["n_equal"], res["n_differing"], res["n_only_v2"], res["n_only_v3"], ", ".join(res["only_v2_examples"][:5]) or "-", ", ".join(res["only_v3_examples"][:5]) or "-"))
            if res["diffs"]:
                md.extend(table(["leaf", "version 2", "version 3", "v3 - v2"], [[d["path"], fmt(d["v2"], 10), fmt(d["v3"], 10), fmt(d["v3_minus_v2"], 4)] for d in res["diffs"][:a.max_diff_rows]]))
                if len(res["diffs"]) > a.max_diff_rows:
                    md.append("(%d more differing leaves in the json)" % (len(res["diffs"]) - a.max_diff_rows))
    else:
        md.append("")
        md.append("B version 2's json files were not supplied (--v2-json TAG=PATH, the files version 2's md lists at its end); the leaf-by-leaf comparison is not printed; the cited numbers above are the comparison.")
    digest["comparison"]["v2_json"] = v2


def insert_i8(files, a, md, digest):
    md.append(""); md.append("## For insert I8: the numbers this version supplies (every number from the files; the sentences the lead writes)")
    rec = {}
    for F in files:
        for L in (F["J"].get("launches") or []):
            if not isinstance(L, dict) or L.get("key") != a.i8_launch:
                continue
            cols = [c for c in (L.get("columns") or []) if isinstance(c, dict)]
            inter = [c for c in cols if not g(c, "launch.comparator", False)]
            comp = [c for c in cols if g(c, "launch.comparator", False)]
            b16 = [c["beyond16_final_before_reflection"] for c in inter if isnum(c.get("beyond16_final_before_reflection"))]
            clk = [c["last_pre_reflection_after_launch_rs"] for c in inter if isnum(c.get("last_pre_reflection_after_launch_rs"))]
            arr = [c["arrival_after_launch_rs"] for c in inter if isnum(c.get("arrival_after_launch_rs"))]
            tt1 = [g(c, "spectrum_end.v3_threshold.buffers.1.threshold_total") for c in cols]; tt1 = [x for x in tt1 if isnum(x)]
            tt2 = [g(c, "spectrum_end.v3_threshold.buffers.2.threshold_total") for c in cols]; tt2 = [x for x in tt2 if isnum(x)]
            terms = [g(c, "spectrum_end.v3_threshold.threshold_error." + k) for c in cols for k in SHARES]; terms = [x for x in terms if isnum(x)]
            above = [g(c, "spectrum_end.v3_threshold.buffers.1.w_above_m") for c in inter]; above = [x for x in above if isnum(x)]
            r = {"n_interior": len(inter), "n_comparators": len(comp), "beyond16_min": min(b16) if b16 else None, "beyond16_max": max(b16) if b16 else None,
                 "clock_min": min(clk) if clk else None, "clock_max": max(clk) if clk else None, "arrival_min": min(arr) if arr else None, "arrival_max": max(arr) if arr else None,
                 "threshold_total_end_max_b1": max(tt1) if tt1 else None, "threshold_total_end_max_b2": max(tt2) if tt2 else None, "threshold_term_max": max(terms) if terms else None,
                 "above_m_end_b1_min": min(above) if above else None, "above_m_end_b1_max": max(above) if above else None, "kappa": L.get("kappa"), "T": L.get("T_launch_over_rs")}
            rec[F["tag"]] = r
            md.append("- %s, launch %s (T/r_s %s, kappa %s): %d interior columns and %d comparators; last pre-reflection shares beyond 16 r_s from %s to %s (their clocks %s to %s r_s after launch); "
                      "half-share arrivals from %s to %s r_s after launch; at the end records the weight above m at buffer 1 runs from %s to %s, the threshold content is at most %s of a column's weight "
                      "(buffer 1; %s at buffer 2) and the largest threshold term of a band share is %s." % (
                          F["tag"], a.i8_launch, fmt(r["T"]), fmt(r["kappa"]), r["n_interior"], r["n_comparators"], fmt(r["beyond16_min"], 5), fmt(r["beyond16_max"], 5), fmt(r["clock_min"]), fmt(r["clock_max"]),
                          fmt(r["arrival_min"], 4), fmt(r["arrival_max"], 4), fmt(r["above_m_end_b1_min"], 4), fmt(r["above_m_end_b1_max"], 4), fmt(r["threshold_total_end_max_b1"], 3),
                          fmt(r["threshold_total_end_max_b2"], 3), fmt(r["threshold_term_max"], 3)))
    if not rec:
        md.append("- launch %s not found in any file; nothing printed." % a.i8_launch)
    md.append("- The review's cited 0.99697 and 0.73311 (if found above) are last pre-reflection shares beyond 16 r_s of the fixed-field launch-15 kappa-1 columns, not reflected 300 r_s endpoints; the cited 22.48 and 25.82 r_s are their half-share arrivals after launch.")
    md.append("- Sayable: in kind on 2 descriptions (radial lattice-carrier columns centred inside the outer root develop order-one weight beyond 16 r_s on the stored history; the cubic calculation reports its own finite-cut shares), no number here a property of the object. Not sayable: " + NOT_SAYABLE)
    digest["insert_I8"] = rec


def files_section(files, md, digest):
    md.append(""); md.append("## Files (every number above is read from these)")
    rows = []
    for F in files:
        rows.append([F["tag"], F["json"], F["json_md5"], F["json_bytes"], fmt(F["npz"]), fmt(F["npz_md5"]), F["note"]])
        digest["files"].append({"tag": F["tag"], "json": F["json"], "json_md5": F["json_md5"], "json_bytes": F["json_bytes"], "npz": F["npz"], "npz_md5": F["npz_md5"], "note": F["note"]})
    md.extend(table(["tag", "json", "md5", "bytes", "npz", "md5", "note"], rows))


def parse_specs(items):
    if not items:
        return [dict(tag=t, prep=p, json=j, npz=z, note=nt) for (t, p, j, z, nt) in DEFAULT_FILES]
    out = []
    for it in items:
        tag, _, rest = it.partition("=")
        parts = rest.split(",")
        out.append(dict(tag=tag.strip(), prep="(as named on the command line)", json=parts[0].strip(),
                        npz=(parts[1].strip() if len(parts) > 1 and parts[1].strip() else None), note="command line"))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="HORIZON version 3's composer, version 2 (B version 3's six reruns with the threshold columns; the binning rule from horizon3_bins_v1)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="HORIZON_single_object_v3")
    ap.add_argument("--files", action="append", default=None, help="TAG=JSON[,NPZ]; repeatable; default: the six reruns' box paths (B3_fixed_s8 from pod 4)")
    ap.add_argument("--v2-json", action="append", default=[], help="TAG=JSON: B version 2's json of the same tag, compared leaf by leaf beyond the added columns")
    ap.add_argument("--v2-skip", default="v3_threshold,seconds,md5,script,utc,path,file", help="comma list; leaf paths containing any of these are left out of the version 2 comparison")
    ap.add_argument("--tol-v2-abs", type=float, default=1e-9)
    ap.add_argument("--tol-v2-rel", type=float, default=1e-9)
    ap.add_argument("--cited", action="append", default=None, help="'LABEL=VALUE': a number version 2 or its reviews printed, searched among the reruns' leaves; default: the reviews' list")
    ap.add_argument("--cited-rel-tol", type=float, default=1e-6, help="added to half the unit of the cited value's last printed digit")
    ap.add_argument("--tol-closure", type=float, default=1e-9)
    ap.add_argument("--series-columns", default="0,14")
    ap.add_argument("--series-max-rows", type=int, default=64)
    ap.add_argument("--max-leaf-rows", type=int, default=400)
    ap.add_argument("--max-diff-rows", type=int, default=200)
    ap.add_argument("--N-inf", type=float, default=0.9999999590601987, help="the actual tail's lapse (review_TRANSFER_class_v1 item 3); the edge displacement is printed against the half-widths")
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--i8-launch", default="T15p000_k1")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def run(a):
    specs = parse_specs(a.files)
    files = []
    for s in specs:
        if not os.path.exists(s["json"]):
            raise SystemExit("missing %s" % s["json"])
        F = dict(s)
        with open(s["json"]) as fh:
            F["J"] = json.load(fh)
        F["json_md5"] = md5(s["json"]); F["json_bytes"] = os.path.getsize(s["json"])
        if s["npz"] and os.path.exists(s["npz"]):
            F["Z"] = np.load(s["npz"]); F["npz_md5"] = md5(s["npz"])
        else:
            F["Z"] = None; F["npz_md5"] = None; F["npz"] = (s["npz"] + " (missing)") if s["npz"] else None
        files.append(F)
    series_cols = [int(x) for x in a.series_columns.split(",") if x.strip() != ""]
    here = os.path.abspath(__file__)
    bins_path = os.path.abspath(HB.__file__)
    md, arrays = [], {}
    digest = {"script": "compose_horizon_v3_v2.py", "script_md5": md5(here) if os.path.exists(here) else None,
              "bins_module": "horizon3_bins_v1.py", "bins_module_md5": md5(bins_path) if os.path.exists(bins_path) else None, "binning_rule": HB.RULE,
              "args": vars(a), "label": LABEL, "definitions": DEFINITIONS, "not_sayable": NOT_SAYABLE, "files": [], "per_file": {}, "checks_digest": [],
              "comparison": {}, "insert_I8": {}}
    md.append(TITLE); md.append(""); md.append("**Label (every output):** " + LABEL); md.append("")
    md.append(COMPOSED % (digest["script_md5"], digest["bins_module_md5"], a.name)); md.append("")
    md.append("## Definitions (version 2's ruled items with the reviews' corrections)"); md.append("")
    md.extend(DEFINITIONS); md.append(""); md.append("Not sayable from these numbers: " + NOT_SAYABLE); md.append("")
    md.append("## The binning rule (one module, horizon3_bins_v1.py, behind the export, the dust script and this composer's checks)"); md.append("")
    md.append(HB.RULE); md.append("")
    md.append("## B version 3: the transfer, six reruns")
    for F in files:
        render_file(F, a, series_cols, md, digest, arrays)
    checks_digest_section(digest, md, a)
    comparison_section(files, a, md, digest)
    insert_i8(files, a, md, digest)
    files_section(files, md, digest)
    return "\n".join(md) + "\n", digest, arrays


def main():
    a = build_parser().parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {ext: os.path.join(a.out_dir, a.name + ext) for ext in (".md", ".json", "_arrays.npz")}
    clash = [p for p in paths.values() if os.path.exists(p)]
    if clash and not a.overwrite:
        sys.exit("refusing to overwrite %s" % clash)
    md, digest, arrays = run(a)
    with open(paths[".md"], "w") as fh:
        fh.write(md)
    with open(paths[".json"], "w") as fh:
        json.dump(digest, fh, indent=1, default=_jdefault)
    np.savez(paths["_arrays.npz"], **arrays)
    flagged = [c for c in digest["checks_digest"] if c["beyond_tolerance"]]
    print("wrote %s (%d bytes), %s, %s (%d arrays); launches beyond the check tolerance: %d" % (paths[".md"], len(md), paths[".json"], paths["_arrays.npz"], len(arrays), len(flagged)))
    for c in digest["comparison"].get("cited", []):
        print("cited %s = %s: %d hits%s" % (c["label"][:60], c["value"], len(c["hits"]), "" if c["hits"] else " (nearest %s %s = %s)" % (c["nearest"]["file"], c["nearest"]["path"], c["nearest"]["value"])))


if __name__ == "__main__":
    main()
