#!/usr/bin/env python
"""export_horizon_refcol_v1.py (the solve lane; designed by reviewer A for the lane's builder; reorientation 4 item 7, the second_route lane's
task 3 reference). Exports the reference column 14 of X_end_T15p000_k1 of B version 3's history rerun (B3_hist_prod) with: the column
itself (raw, complex128), the site radii, the end record's applied fields N and v (the launch record's beside), the exterior cut and its
block (sites beyond the cut, both spinor components), the block's eigenvalue extremes (Lanczos; dense when --dense covers the block) beside
the full chain's, the band edges (+-m, +-b = +-sqrt(m^2 + 1/a^2); +-N_inf m, +-N_inf b with N_inf the end record's lapse at the chain's last
site; the two-step flow bound b + b r_s/(2 R_cut) at the cut), the 5 raw Killing bins (-inf,-b], (-b,-m], (-m,m], (m,b], (b,inf) of the
column in the end record's clocked radial operator's eigenbasis (a dense eigendecomposition, the producer's spectrum_end recomputed and
compared) and of the column's restriction to the block in the block's eigenbasis, section 49's threshold columns at the buffers 1 and 2
(half-width b r_s/(2 R_max) at +-b, m r_s/(2 R_max) at +-m, r_s the launch record's 2GE as the producer uses, R_max the chain's end) with
the threshold term per bin, the column's stored end weights recomputed from the column (this fixes the column's layout), and the column's
stored time series. Outputs horizon_refcol_T15p000_k1_c14.npz, .json and .md (the md says what each array is, its units and conventions).
CPU; threads capped; no plotting; no dates; refuses to overwrite.
Usage (box): python export_horizon_refcol_v1.py [--b3-json ...] [--b3-npz ...] [--record ... --fields ... --record2 ... --fields2 ...] --out-prefix ...
"""
import os
import sys


def _pre_threads(default="8"):
    t = default
    for i, s in enumerate(sys.argv):
        if s == "--threads" and i + 1 < len(sys.argv):
            t = sys.argv[i + 1]
        elif s.startswith("--threads="):
            t = s.split("=", 1)[1]
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, t)


_pre_threads()

import argparse
import hashlib
import json
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

T0 = time.time()
BULK = "<internal>/horizon_v3"
REC1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
FLD1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz"
REC2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json"
FLD2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz"
W_KEYS = ("w_beyond_outer_root", "w_beyond_2GE", "w_beyond_16rs", "w_collar_16_24", "w_far_32_48", "w_beyond_Rcut")
END_TO_W = {"beyond_outer_root": "w_beyond_outer_root", "beyond_2GE": "w_beyond_2GE", "beyond_16rs": "w_beyond_16rs",
            "collar_16_24": "w_collar_16_24", "far_32_48": "w_far_32_48", "beyond_Rcut": "w_beyond_Rcut"}
EDGES = ("-b", "-m", "m", "b")
BIN_LABELS = ["(-inf, -b]", "(-b, -m]", "(-m, m]", "(m, b]", "(b, inf)"]


def log(msg):
    print("[+%7.0f s] %s" % (time.time() - T0, msg), flush=True)


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


def g(d, path, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def fmt(x, nd=6):
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "True" if x else "False"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "nan" if math.isnan(x) else ("%%.%dg" % nd) % x
    return str(x)


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


def buffer_key(mu):
    return str(int(mu)) if float(mu).is_integer() else str(mu)


# ---------------------------------------------------------------- pure functions (the self-test imports these)
def killing_bins_5(e, w, m, b):
    """the 5 bins (-inf,-b], (-b,-m], (-m,m], (m,b], (b,inf) of the weights w at the levels e (the producers' half-open convention)."""
    e = np.asarray(e, float); w = np.asarray(w, float)
    edges = [-np.inf, -b, -m, m, b, np.inf]
    return np.array([w[(e > edges[i]) & (e <= edges[i + 1])].sum() for i in range(5)])


def threshold_columns(e, w, m, b, r_s, R_max, buffers=(1.0, 2.0)):
    """engine spec 49.1 (1): the content within mu * b r_s/(2 R_max) of +-b and within mu * m r_s/(2 R_max) of +-m is the threshold
    content (neither band nor bound); the 5 bins and the 3 shares are read on the remainder at each buffer mu; the threshold term of each
    bin is |bin(first buffer) - bin(second buffer)|."""
    e = np.asarray(e, float); w = np.asarray(w, float)
    hw = {"b": b * r_s / (2.0 * R_max), "m": m * r_s / (2.0 * R_max)}
    edges = {"-b": -b, "-m": -m, "m": m, "b": b}
    raw5 = killing_bins_5(e, w, m, b)
    tot = float(w.sum())
    out = {"half_width_b": hw["b"], "half_width_m": hw["m"], "r_s": float(r_s), "R_max": float(R_max), "m": float(m), "b": float(b),
           "raw": {"bins5": raw5.tolist(), "w_below_minus_b": float(raw5[0]), "w_below_minus_m": float(raw5[0] + raw5[1]), "w_in_gap": float(raw5[2]),
                   "w_above_m": float(raw5[3] + raw5[4]), "w_above_b": float(raw5[4]), "total": tot, "w_negative": float(w[e < 0].sum()),
                   "mean": float((e * w).sum() / tot) if tot > 0 else None},
           "buffers": {}, "threshold_error": {}}
    for mu in buffers:
        keep = np.ones(len(e), bool); tw = {}
        for name, e0 in edges.items():
            h = hw["b"] if name in ("b", "-b") else hw["m"]
            inb = np.abs(e - e0) <= mu * h
            tw[name] = float(w[inb].sum()); keep &= ~inb
        b5 = killing_bins_5(e[keep], w[keep], m, b)
        out["buffers"][buffer_key(mu)] = {"mu": float(mu), "threshold_weight": tw, "threshold_total": float(sum(tw.values())), "bins5": b5.tolist(),
                                          "w_below_minus_b": float(b5[0]), "w_below_minus_m": float(b5[0] + b5[1]), "w_in_gap": float(b5[2]),
                                          "w_above_m": float(b5[3] + b5[4]), "w_above_b": float(b5[4]), "n_levels_in_threshold": int((~keep).sum())}
    keys = [buffer_key(mu) for mu in buffers]
    if len(keys) >= 2:
        B1, B2 = out["buffers"][keys[0]], out["buffers"][keys[1]]
        for k in ("w_below_minus_b", "w_below_minus_m", "w_in_gap", "w_above_m", "w_above_b", "threshold_total"):
            out["threshold_error"][k] = abs(B1[k] - B2[k])
        out["threshold_error"]["bins5"] = np.abs(np.array(B1["bins5"]) - np.array(B2["bins5"])).tolist()
        out["threshold_error"]["label"] = "the threshold term: |bin(buffer %s) - bin(buffer %s)|" % (keys[0], keys[1])
    return out


def site_weights(X, n, layout):
    X = np.asarray(X)
    if layout == "site_major":
        return np.abs(X[0::2]) ** 2 + np.abs(X[1::2]) ** 2
    if layout == "component_major":
        return np.abs(X[:n]) ** 2 + np.abs(X[n:2 * n]) ** 2
    raise ValueError(layout)


def rows_of(mask, layout):
    """the row indices (into the 2n-vector) of the sites in mask under the layout."""
    mask = np.asarray(mask, bool); n = len(mask)
    if layout == "site_major":
        return np.nonzero(np.repeat(mask, 2))[0]
    return np.nonzero(np.concatenate([mask, mask]))[0]


def region_weights(ws, R, rs, Rt, twoGE, rcut_rs=24.0, strict=True):
    """the producer's six end regions from site weights; strict: beyond c means R > c (else R >= c)."""
    gt = (lambda c: R > c) if strict else (lambda c: R >= c)
    le = (lambda c: R <= c) if strict else (lambda c: R < c)
    out = {"beyond_outer_root": float(ws[gt(Rt)].sum()) if (Rt is not None and np.isfinite(Rt)) else float("nan"),
           "beyond_2GE": float(ws[gt(twoGE)].sum()) if (twoGE is not None and np.isfinite(twoGE)) else float("nan"),
           "beyond_16rs": float(ws[gt(16.0 * rs)].sum()), "collar_16_24": float(ws[gt(16.0 * rs) & le(24.0 * rs)].sum()),
           "far_32_48": float(ws[gt(32.0 * rs) & le(48.0 * rs)].sum()), "beyond_Rcut": float(ws[gt(rcut_rs * rs)].sum())}
    return out


def detect_layout(X, n, R, rs, Rt, twoGE, stored_end, rcut_rs=24.0):
    """which layout (site_major: rows 2j, 2j+1 are site j; component_major: rows j and n+j) and cut convention reproduce the stored end weights."""
    res = {}
    for layout in ("site_major", "component_major"):
        ws = site_weights(X, n, layout)
        for strict in (True, False):
            rw = region_weights(ws, R, rs, Rt, twoGE, rcut_rs, strict)
            devs = [abs(rw[k] - float(stored_end[k])) for k in rw if isnum(stored_end.get(k)) and np.isfinite(rw[k])]
            res["%s/%s" % (layout, "R > c" if strict else "R >= c")] = {"layout": layout, "strict": strict, "max_abs_dev": max(devs) if devs else float("inf"), "recomputed": rw}
    best = min(res, key=lambda k: res[k]["max_abs_dev"])
    return res[best], res


def zeros_of(R, f):
    """every sign change of f on the sites, linear between sites: (R_zero, +1 rising / -1 falling)."""
    f = np.asarray(f, float); R = np.asarray(R, float); out = []
    s0, s1 = f[:-1], f[1:]
    idx = np.nonzero(((s0 < 0) & (s1 >= 0)) | ((s0 >= 0) & (s1 < 0)))[0]
    for i in idx:
        if s1[i] == s0[i]:
            continue
        out.append((float(R[i] - s0[i] * (R[i + 1] - R[i]) / (s1[i] - s0[i])), 1 if s1[i] > s0[i] else -1))
    return out


def applied_outer_inner(R, f):
    """the outer root: the largest rising zero of N + v (version 1b's finder); the inner root: the largest falling zero below it."""
    z = zeros_of(R, f)
    rising = [r for r, s in z if s > 0]
    if not rising:
        return None, None, z
    Ro = max(rising)
    falling = [r for r, s in z if s < 0 and r < Ro]
    return Ro, (max(falling) if falling else None), z


# ---------------------------------------------------------------- the engine (lazy; the producers' modules, never edited)
def load_engine(engine_dir, page_v6_path):
    if page_v6_path:
        os.environ.setdefault("PAGE_V6_PATH", page_v6_path)
        if page_v6_path not in sys.path:
            sys.path.insert(0, page_v6_path)
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    import torch
    import engine_single_object_B_v2 as BV2
    return torch, BV2


def extremes_lanczos(A, tol):
    emax = float(spla.eigsh(A, k=1, which="LA", tol=tol, return_eigenvectors=False)[0])
    emin = float(spla.eigsh(A, k=1, which="SA", tol=tol, return_eigenvectors=False)[0])
    return emin, emax


def dense_eigh(A, driver):
    Ad = A.toarray() if hasattr(A, "toarray") else np.array(A)
    return sla.eigh(Ad, driver=driver, overwrite_a=True, check_finite=False)


def weights_in_basis(U, X):
    """|U^dagger X|^2 per eigenvector without copying U (c = (X^dagger U)^dagger)."""
    c = (X.conj() @ U).conj()
    return np.abs(c) ** 2


def compare_thresholds(thr, stored_sp, keys=("1", "2")):
    rows = []

    def add(name, x, y):
        rows.append({"quantity": name, "recomputed": x, "stored": y, "recomputed_minus_stored": (float(x) - float(y)) if (isnum(x) and isnum(y)) else None})
    add("norm (sum of the weights)", thr["raw"]["total"], stored_sp.get("norm"))
    add("mean Killing energy", thr["raw"]["mean"], stored_sp.get("mean"))
    for k in ("w_below_minus_m", "w_in_gap", "w_above_m", "w_negative"):
        add("raw " + k, thr["raw"][k], stored_sp.get(k))
    st = stored_sp.get("v3_threshold") if isinstance(stored_sp.get("v3_threshold"), dict) else {}
    add("half_width_b", thr["half_width_b"], st.get("half_width_b")); add("half_width_m", thr["half_width_m"], st.get("half_width_m"))
    for mu in keys:
        B = thr["buffers"].get(mu, {}); SB = g(st, "buffers.%s" % mu) or {}
        for e in EDGES:
            add("buffer %s threshold weight at %s" % (mu, e), (B.get("threshold_weight") or {}).get(e), (SB.get("threshold_weight") or {}).get(e))
        add("buffer %s threshold total" % mu, B.get("threshold_total"), SB.get("threshold_total"))
        for k in ("w_below_minus_m", "w_in_gap", "w_above_m"):
            add("buffer %s %s" % (mu, k), B.get(k), SB.get(k))
    for k in ("w_below_minus_m", "w_in_gap", "w_above_m", "threshold_total"):
        add("threshold term " + k, thr["threshold_error"].get(k), (st.get("threshold_error") or {}).get(k))
    return rows


def build_parser():
    ap = argparse.ArgumentParser(description="the HORIZON version 3 reference column 14 of X_end_T15p000_k1 for the second_route lane")
    ap.add_argument("--b3-json", default=BULK + "/pod3_mirror/out/B3_hist_prod.json")
    ap.add_argument("--b3-npz", default=BULK + "/pod3_mirror/out/B3_hist_prod.npz")
    ap.add_argument("--launch", default="T15p000_k1")
    ap.add_argument("--column", type=int, default=14)
    ap.add_argument("--record", default=REC1); ap.add_argument("--fields", default=FLD1)
    ap.add_argument("--record2", default=REC2); ap.add_argument("--fields2", default=FLD2); ap.add_argument("--switch-T-rs", type=float, default=40.0)
    ap.add_argument("--fields-module", default="engine_fields_v14"); ap.add_argument("--nsites", type=int, default=8192)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--engine-dir", default="<internal>/code"); ap.add_argument("--page-v6-path", default="<internal>/v6")
    ap.add_argument("--no-engine", action="store_true", help="export the column, the stored numbers and the layout check only (no operator)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--block-cut", default="outer_root", choices=["outer_root", "r16", "rcut24"], help="the exterior block's cut: the end record's stored outer root, 16 r_s or R_cut")
    ap.add_argument("--rcut-rs", type=float, default=24.0)
    ap.add_argument("--dense", default="both", choices=["none", "chain", "block", "both"], help="which dense eigendecompositions to run (about 16384 x 16384 each)")
    ap.add_argument("--dense-all-blocks", action="store_true", help="dense eigendecompositions of the control blocks too (16 and 24 r_s)")
    ap.add_argument("--eig-driver", default="evr", help="scipy.linalg.eigh driver (evr: the busi lane's; evd: faster, more memory)")
    ap.add_argument("--eigsh-tol", type=float, default=1e-10)
    ap.add_argument("--layout", default="auto", choices=["auto", "site_major", "component_major"])
    ap.add_argument("--tol-layout", type=float, default=1e-8, help="the recomputed end weights must reproduce the stored ones within this for the layout to be accepted")
    ap.add_argument("--tol-stored", type=float, default=1e-9, help="flag recomputed shares and threshold weights differing from the producer's by more")
    ap.add_argument("--buffers", default="1,2")
    ap.add_argument("--R-max", type=float, default=None, help="the chain's end in a for the half-widths; default the json's R_max, else the last site's radius")
    ap.add_argument("--out-prefix", default=BULK + "/refcol/horizon_refcol_T15p000_k1_c14")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    outs = {ext: a.out_prefix + ext for ext in (".npz", ".json", ".md")}
    if any(os.path.exists(p) for p in outs.values()) and not a.overwrite:
        sys.exit("refusing to overwrite %s" % [p for p in outs.values() if os.path.exists(p)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    buffers = [float(x) for x in a.buffers.split(",") if x.strip()]
    bkeys = [buffer_key(mu) for mu in buffers]
    with open(a.b3_json) as fh:
        J = json.load(fh)
    Z = np.load(a.b3_npz)
    L = next((x for x in J.get("launches", []) if isinstance(x, dict) and x.get("key") == a.launch), None)
    if L is None:
        sys.exit("launch %s not in %s" % (a.launch, a.b3_json))
    cols = L.get("columns") or []
    if not (0 <= a.column < len(cols)):
        sys.exit("column %d not among the %d columns of %s" % (a.column, len(cols), a.launch))
    C = cols[a.column]; key = a.launch
    Xall = np.asarray(Z["X_end_%s" % key]); X = Xall[:, a.column].astype(np.complex128)
    nX = X.shape[0]; n = nX // 2
    t = np.asarray(Z["t_%s" % key], float)
    Rt_series = np.asarray(Z["Rt_%s" % key], float) if "Rt_%s" % key in Z.files else None
    gE_series = np.asarray(Z["2gE_%s" % key], float) if "2gE_%s" % key in Z.files else None
    norm_series = np.asarray(Z["norm_%s" % key], float) if "norm_%s" % key in Z.files else None
    refl = np.asarray(Z["refl_%s" % key]) if "refl_%s" % key in Z.files else None
    series = {k: np.asarray(Z["%s_%s" % (k, key)])[:, a.column] for k in W_KEYS if "%s_%s" % (k, key) in Z.files}
    m = a.m; b = math.sqrt(m * m + 1.0)
    tp = g(C, "spectrum_end.v3_threshold") or g(C, "spectrum_launch.v3_threshold") or {}
    r_s_thr = float(tp.get("r_s", L.get("two_gE")))
    Rt_end = float(Rt_series[-1]) if (Rt_series is not None and np.isfinite(Rt_series[-1])) else None
    gE_end = float(gE_series[-1]) if gE_series is not None else None
    log("column %d of %s: %d rows (%d sites); t_end %.4f r_s; stored R_t(end) %s, 2GE(end) %s; b %.12f (json %s)" % (a.column, key, nX, n, t[-1], Rt_end, gE_end, b, tp.get("b")))
    out = {"script": "export_horizon_refcol_v1.py", "args": vars(a), "b3_json": a.b3_json, "b3_json_md5": md5(a.b3_json), "b3_npz": a.b3_npz, "b3_npz_md5": md5(a.b3_npz),
           "launch_header": {k: v for k, v in L.items() if k not in ("columns", "signed_launch", "signed_launch_window_controls", "signed_series")},
           "column_index": a.column, "column_stored": C, "m": m, "b": b, "b_json": tp.get("b"), "b_minus_json": (b - float(tp["b"])) if isnum(tp.get("b")) else None,
           "r_s_for_half_widths": r_s_thr, "t_end_over_rs": float(t[-1]), "n_sites": n, "stored_Rt_end": Rt_end, "stored_2GE_end": gE_end, "engine": {}, "checks": {}}
    arrays = {"X_col": X, "t_over_rs": t}
    for nm, arr in (("Rt_series", Rt_series), ("twoGE_series", gE_series), ("norm_series", norm_series)):
        if arr is not None:
            arrays[nm] = arr
    if refl is not None:
        arrays["refl_col"] = refl[:, a.column]
    for k, arr in series.items():
        arrays["series_" + k] = arr
    for sname in ("spectrum_launch", "spectrum_end"):
        spd = C.get(sname) if isinstance(C.get(sname), dict) else {}
        for hk in ("bins_particles", "bins_holes"):
            if isinstance(spd.get(hk), list):
                arrays["stored_%s_%s_raw" % (sname, hk)] = np.array(spd[hk], float)
            for mu in bkeys:
                v = g(spd, "v3_threshold.buffers.%s.%s" % (mu, hk))
                if isinstance(v, list):
                    arrays["stored_%s_%s_buffer%s" % (sname, hk, mu)] = np.array(v, float)
            v = g(spd, "v3_threshold.threshold_error.%s" % hk)
            if isinstance(v, list):
                arrays["stored_%s_%s_threshold_term" % (sname, hk)] = np.array(v, float)
    # ---------------- the engine: the history, the radii, the fields, the operator
    eng = None; R = None; rs = 64.0; OPS = None
    if not a.no_engine:
        try:
            torch, BV2 = load_engine(a.engine_dir, a.page_v6_path)
            torch.set_num_threads(a.threads)
            OPS = BV2.OPS
            H = BV2.CombinedHistory(a.record, a.fields, a.nsites, a.record2 or None, a.fields2 or None, a.switch_T_rs if a.record2 else None)
            rs = float(H.rs_nom); R = np.asarray(H.R, float); TT = np.asarray(H.T, float) / rs
            kend = int(np.argmin(np.abs(TT - t[-1]))); k0 = int(L["k0"])
            N_end, v_end = [np.asarray(x, float) for x in H.fields(kend)]
            N_l, v_l = [np.asarray(x, float) for x in H.fields(k0)]
            ne = int(getattr(H, "ne", len(R)))
            rec = H.rec[kend]
            Ro_a, Ri_a, za = applied_outer_inner(R[:ne], (N_end + v_end)[:ne])
            eng = {"engine_module": BV2.__file__, "engine_module_md5": md5(BV2.__file__), "ops_module": getattr(OPS, "__file__", None),
                   "ops_module_md5": md5(OPS.__file__) if getattr(OPS, "__file__", None) else None, "files": getattr(H, "files", None), "rs_nominal": rs, "n_sites": int(len(R)),
                   "ne": ne, "k_end": kend, "T_k_end_over_rs": float(TT[kend]), "record_clock_minus_t_end": float(TT[kend] - t[-1]), "k0": k0, "T_k0_over_rs": float(TT[k0]),
                   "record_clock_minus_T_launch": float(TT[k0] - float(L["T_launch_over_rs"])), "N_inf_end": float(N_end[-1]), "v_inf_end": float(v_end[-1]),
                   "N_min_end": float(N_end.min()), "v_min_end": float(v_end.min()), "end_record_zeros": list(rec.get("trapping_zeros") or []), "end_record_R_t": rec.get("R_t"),
                   "end_record_applied_outer": Ro_a, "end_record_applied_inner": Ri_a, "end_record_applied_zeros_all": [{"R": r_, "slope_sign": s_} for r_, s_ in za],
                   "stored_Rt_end_minus_record_zero": (Rt_end - max(rec.get("trapping_zeros"))) if (Rt_end is not None and rec.get("trapping_zeros")) else None,
                   "stored_Rt_end_minus_applied_outer": (Rt_end - Ro_a) if (Rt_end is not None and Ro_a is not None) else None}
            arrays.update({"R": R, "N_end": N_end, "v_end": v_end, "N_launch": N_l, "v_launch": v_l})
            if abs(eng["record_clock_minus_t_end"]) > 1e-6:
                log("WARNING: the record nearest t_end differs from it by %.3e r_s" % eng["record_clock_minus_t_end"])
            if len(R) != n:
                raise RuntimeError("the history's %d sites differ from the column's %d" % (len(R), n))
        except Exception as ex:
            eng = None; out["engine"] = {"error": repr(ex)}
            log("ENGINE UNAVAILABLE (%r): exporting the column, the stored numbers and the layout check with R_j = j a assumed" % (ex,))
    if R is None:
        R = np.arange(1, n + 1, dtype=float); out["R_assumed"] = "R_j = j a (no engine)"
        arrays["R"] = R
    if eng:
        out["engine"] = eng
    # ---------------- the layout from the stored end weights
    stored_end = C.get("end") if isinstance(C.get("end"), dict) else {}
    if a.layout == "auto":
        best, allres = detect_layout(X, n, R, rs, Rt_end, gE_end, stored_end, a.rcut_rs)
        layout, strict = best["layout"], best["strict"]
        out["layout_detection"] = {k: {"max_abs_dev": v["max_abs_dev"], "recomputed": v["recomputed"]} for k, v in allres.items()}
    else:
        layout = a.layout; strict = True
        best = {"max_abs_dev": None}
    ws = site_weights(X, n, layout)
    rw = region_weights(ws, R, rs, Rt_end, gE_end, a.rcut_rs, strict)
    out["layout"] = {"layout": layout, "cut_convention": "R > c" if strict else "R >= c", "max_abs_dev_from_stored_end": best["max_abs_dev"],
                     "accepted": (best["max_abs_dev"] is not None and best["max_abs_dev"] <= a.tol_layout) if a.layout == "auto" else None,
                     "recomputed_end": rw, "stored_end": stored_end, "npz_last_row": {k: float(v[-1]) for k, v in series.items()},
                     "column_norm": float(ws.sum()), "stored_spectrum_end_norm": g(C, "spectrum_end.norm")}
    arrays["site_weight"] = ws
    log("layout %s (%s): max |recomputed - stored end| %s; norm %.12f" % (layout, out["layout"]["cut_convention"], fmt(best["max_abs_dev"], 3), ws.sum()))
    if a.layout == "auto" and not out["layout"]["accepted"]:
        log("WARNING: neither layout reproduces the stored end weights within %.1e; the export continues with the better one, flagged" % a.tol_layout)
    # ---------------- cuts and blocks
    cuts = {}
    if Rt_end is not None:
        cuts["outer_root"] = Rt_end
    cuts["r16"] = 16.0 * rs; cuts["rcut24"] = a.rcut_rs * rs
    primary = a.block_cut if a.block_cut in cuts else "rcut24"
    out["cuts"] = {"radii_a": cuts, "primary": primary, "primary_fallback": a.block_cut not in cuts}
    blocks = {}
    for cname, cR in cuts.items():
        mask = (R > cR) if strict else (R >= cR)
        rows = rows_of(mask, layout)
        wb = float(ws[mask].sum())
        iw = int(np.argmax(ws * mask)) if mask.any() else None
        blocks[cname] = {"cut_a": cR, "cut_over_rs": cR / rs, "n_sites": int(mask.sum()), "first_site_R": float(R[mask][0]) if mask.any() else None, "last_site_R": float(R[-1]),
                         "column_weight_on_block": wb, "column_site_weight_max_on_block": float(ws[iw]) if iw is not None else None, "its_radius": float(R[iw]) if iw is not None else None,
                         "two_step_flow_bound_b_plus_b_rs_over_2Rcut": b * (1.0 + (gE_end if gE_end is not None else r_s_thr) / (2.0 * cR)), "rows": rows, "mask": mask}
        arrays["block_mask_" + cname] = mask
    out["band_edges"] = {"m": m, "b": b, "minus_m": -m, "minus_b": -b, "N_inf_end": eng["N_inf_end"] if eng else None,
                         "N_inf_m": eng["N_inf_end"] * m if eng else None, "N_inf_b": eng["N_inf_end"] * b if eng else None,
                         "lapse_displacement_m": (1.0 - eng["N_inf_end"]) * m if eng else None, "lapse_displacement_b": (1.0 - eng["N_inf_end"]) * b if eng else None,
                         "half_width_b": b * r_s_thr / (2.0 * (a.R_max or float(tp.get("R_max", R[-1])))), "half_width_m": m * r_s_thr / (2.0 * (a.R_max or float(tp.get("R_max", R[-1])))),
                         "R_max": a.R_max or float(tp.get("R_max", R[-1])), "r_s_used": r_s_thr, "note": "b = sqrt(m^2 + 1/a^2) the sine chain's band top; the cubic free top b_3 = sqrt(m^2 + 3/a^2) does not apply to the radial chain"}
    R_max = out["band_edges"]["R_max"]
    # ---------------- the operator at the end record
    spectra = {"chain": None, "blocks": {}}
    if eng and OPS is not None:
        h0, Q = OPS.chain_sparse(R, m, float(L["kappa"]))
        h_end = sp.csr_matrix(OPS.clocked_csr(h0, Q, N_end, v_end))
        herm = float(abs(h_end - h_end.getH()).max()) if h_end.nnz else 0.0
        out["operator"] = {"shape": list(h_end.shape), "nnz": int(h_end.nnz), "hermiticity_defect": herm, "kappa": L.get("kappa"), "definition": "OPS.clocked_csr(OPS.chain_sparse(R, m, kappa), N_end, v_end): the end record's applied fields on the kappa chain (the producers' operator)"}
        log("operator %s, nnz %d, hermiticity %.1e" % (h_end.shape, h_end.nnz, herm))
        emin, emax = extremes_lanczos(h_end, a.eigsh_tol)
        spectra["chain"] = {"e_min_lanczos": emin, "e_max_lanczos": emax}
        log("chain extremes (Lanczos) %.10f %.10f" % (emin, emax))
        for cname, B in blocks.items():
            blk = h_end[B["rows"]][:, B["rows"]]
            bmin, bmax = extremes_lanczos(blk, a.eigsh_tol)
            spectra["blocks"][cname] = {"dim": int(blk.shape[0]), "e_min_lanczos": bmin, "e_max_lanczos": bmax, "e_max_minus_b": bmax - b, "e_max_minus_flow_bound": bmax - B["two_step_flow_bound_b_plus_b_rs_over_2Rcut"]}
            B["block_csr"] = blk
            log("block %s (%d dim): extremes %.10f %.10f; flow bound %.10f" % (cname, blk.shape[0], bmin, bmax, B["two_step_flow_bound_b_plus_b_rs_over_2Rcut"]))
        if a.dense in ("chain", "both"):
            t1 = time.time(); e, U = dense_eigh(h_end, a.eig_driver); w = weights_in_basis(U, X); del U
            thr = threshold_columns(e, w, m, b, r_s_thr, R_max, buffers)
            spectra["chain"].update({"e_min_dense": float(e[0]), "e_max_dense": float(e[-1]), "dense_seconds": time.time() - t1, "n_levels": int(len(e)),
                                     "thresholds": thr, "compare_stored_spectrum_end": compare_thresholds(thr, C.get("spectrum_end") or {}, bkeys)})
            arrays["e_chain"] = e; arrays["w_chain"] = w
            arrays["bins5_chain_raw"] = np.array(thr["raw"]["bins5"])
            for mu in bkeys:
                arrays["bins5_chain_buffer%s" % mu] = np.array(thr["buffers"][mu]["bins5"])
            arrays["bins5_chain_threshold_term"] = np.array(thr["threshold_error"]["bins5"])
            worst = max((abs(r_["recomputed_minus_stored"]) for r_ in spectra["chain"]["compare_stored_spectrum_end"] if r_["recomputed_minus_stored"] is not None), default=None)
            out["checks"]["chain_recomputed_vs_stored_max_abs"] = worst
            log("chain dense: %d levels in %.0f s; bins5 raw %s; max |recomputed - stored| %s" % (len(e), spectra["chain"]["dense_seconds"], np.array2string(np.array(thr["raw"]["bins5"]), precision=8), fmt(worst, 3)))
            del e, w
        if a.dense in ("block", "both"):
            for cname, B in blocks.items():
                if cname != primary and not a.dense_all_blocks:
                    continue
                t1 = time.time(); e, U = dense_eigh(B["block_csr"], a.eig_driver); Xb = X[B["rows"]]; w = weights_in_basis(U, Xb); del U
                thr = threshold_columns(e, w, m, b, r_s_thr, R_max, buffers)
                wb = float(np.abs(Xb) ** 2 .sum()) if False else float((np.abs(Xb) ** 2).sum())
                thr_norm = threshold_columns(e, w / wb if wb > 0 else w, m, b, r_s_thr, R_max, buffers)
                spectra["blocks"][cname].update({"e_min_dense": float(e[0]), "e_max_dense": float(e[-1]), "dense_seconds": time.time() - t1, "n_levels": int(len(e)),
                                                 "block_weight": wb, "thresholds_raw_weights": thr, "thresholds_normalized_to_block_weight": thr_norm})
                arrays["e_block_%s" % cname] = e; arrays["w_block_%s" % cname] = w
                arrays["bins5_block_%s_raw" % cname] = np.array(thr["raw"]["bins5"])
                for mu in bkeys:
                    arrays["bins5_block_%s_buffer%s" % (cname, mu)] = np.array(thr["buffers"][mu]["bins5"])
                arrays["bins5_block_%s_threshold_term" % cname] = np.array(thr["threshold_error"]["bins5"])
                log("block %s dense: %d levels in %.0f s; block weight %.10f; bins5 raw %s" % (cname, len(e), spectra["blocks"][cname]["dense_seconds"], wb, np.array2string(np.array(thr["raw"]["bins5"]), precision=8)))
                del e, w
        for B in blocks.values():
            B.pop("block_csr", None)
    for B in blocks.values():
        B.pop("rows", None); B.pop("mask", None)
    out["blocks"] = blocks; out["spectra"] = spectra
    out["conventions"] = CONVENTIONS(layout, out["layout"]["cut_convention"])
    out["seconds"] = time.time() - T0
    # ---------------- outputs
    np.savez(outs[".npz"], **arrays)
    with open(outs[".json"], "w") as fh:
        json.dump(out, fh, indent=1, default=_jdefault)
    with open(outs[".md"], "w") as fh:
        fh.write(compose_md(out, arrays, outs))
    log("wrote %s (%d arrays), %s, %s" % (outs[".npz"], len(arrays), outs[".json"], outs[".md"]))


def CONVENTIONS(layout, cutconv):
    return {
        "lengths": "radii in units of the lattice spacing a; the sites j = 0..n-1 at R[j] (R saved); 16 r_s means 16 x the nominal r_s = 64 a; 'beyond c' means " + cutconv,
        "energies": "Killing energies omega in units of 1/a (hbar = c = 1); m a = 0.2; b a = sqrt(1 + (m a)^2); the operator is the end record's clocked radial operator of the kappa channel (the producers' OPS.clocked_csr on OPS.chain_sparse), its eigenvalues the Killing energies of the held fields",
        "column": "X_col: the column's state at the end record (the last record held when the rerun started), complex128, 2 n entries, layout %s%s; its norm the sum of |X|^2 (1 at launch up to the Chebyshev tolerance); it was launched at the launch record with the parameters in column_stored.launch and propagated record by record on the piecewise-constant applied fields" % (
            layout, ": entries 2 j and 2 j + 1 are site j's two spinor components" if layout == "site_major" else ": entries j and n + j are site j's two spinor components"),
        "site_weight": "|X|^2 summed over the two components per site",
        "blocks": "block_mask_<cut>: the sites beyond the cut; the exterior block of the operator is its restriction to those sites' rows and columns (both components); e_block_<cut> its eigenvalues (dense), w_block_<cut> the weights of the column's restriction to the block on the block's eigenvectors (sum = the column's weight on the block, not renormalized; a renormalized copy of the bins in the json)",
        "chain": "e_chain: the full operator's eigenvalues; w_chain: the column's weights on them (sum = the column's norm); the producer's spectrum_end is this binning, recomputed and compared in the json",
        "bins": "bins5_*: (-inf,-b], (-b,-m], (-m,m], (m,b], (b,inf) with the half-open convention (e > lo) & (e <= hi); the 3 shares: below -m = bins 1+2, |omega| < m = bin 3, above m = bins 4+5; weight above b has no propagating partner at infinity and is never read as unbound; weight above m is not escape",
        "threshold": "engine spec 49.1 (1): levels within mu x b r_s/(2 R_max) of +-b and within mu x m r_s/(2 R_max) of +-m are threshold content (neither band nor bound); the bins at buffer mu exclude them; the threshold term per bin = |bin(buffer 1) - bin(buffer 2)|; r_s = the launch record's 2GE (the producer's choice, in a), R_max the chain's end in a; the lapse displacement (1 - N_inf) m of the asymptotic edge is printed and lies inside the half-width",
        "series": "t_over_rs, Rt_series (the stored outer root per record, the producer's roots_dense), twoGE_series, norm_series, refl_col (the reflection flag), series_w_*: the column's six region weights per record",
        "dust_roots": "the applied-field zeros of N + v at the end record are in the json (engine.end_record_applied_*), the record's trapping zeros beside; which root the stored R_t series follows is stated in the json by the differences printed there",
    }


def compose_md(out, arrays, outs):
    L = ["# horizon_refcol_T15p000_k1_c14: the HORIZON version 3 reference column for the second_route lane (B3_hist_prod, launch %s, column %d)" % (out["args"]["launch"], out["column_index"]), ""]
    L.append("Exported by export_horizon_refcol_v1.py from %s (md5 %s) and %s (md5 %s); every number here is read or computed from those files and the history's stored fields by that script. "
             "Label: one object's record (the production massive collapse, r_s/a 64, sigma/a 4, ma 0.2), radial channel kappa %s, the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75; the column a specified preparation, its shares finite-cut responses; the threshold columns neither band nor bound content; weight above m is not escape." % (
        out["b3_json"], out["b3_json_md5"], out["b3_npz"], out["b3_npz_md5"], out["launch_header"].get("kappa")))
    L.append("")
    lh = out["launch_header"]; la = out["column_stored"].get("launch", {})
    L.append("## The column")
    L.append("- Launch record T/r_s %s (record %s), E %s, 2GE %s a, T_H %s; the record's dense roots %s; launched at R_c %s a (%s), a k_0 %s (%s), sigma %s a, spinor %s, N %s and v %s at the centre; comparator %s; omega_bar %s/a; soft %s." % (
        lh.get("T_launch_over_rs"), lh.get("k0"), lh.get("E"), lh.get("two_gE"), lh.get("T_H"), lh.get("roots_dense"), la.get("R_c"), la.get("depth"), la.get("k0"), la.get("ak0_name"), la.get("sigma"), la.get("spinor"), la.get("N_at"), la.get("v_at"), la.get("comparator"), out["column_stored"].get("omega_bar"), out["column_stored"].get("soft")))
    L.append("- The end record T/r_s %s; the column's norm there %s (the stored spectrum_end norm %s); layout %s with 'beyond c' = %s, max |recomputed - stored end weights| %s (accepted within %s: %s)." % (
        out["t_end_over_rs"], fmt(out["layout"]["column_norm"], 12), fmt(out["layout"]["stored_spectrum_end_norm"], 12), out["layout"]["layout"], out["layout"]["cut_convention"], fmt(out["layout"]["max_abs_dev_from_stored_end"], 3), out["args"]["tol_layout"], out["layout"]["accepted"]))
    L.append("- The stored end weights (beyond R_t; 2GE; 16 r_s; collar 16-24; far 32-48; beyond R_cut 24 r_s): " + "; ".join("%s = %s (recomputed %s)" % (k, fmt(out["layout"]["stored_end"].get(k), 10), fmt(out["layout"]["recomputed_end"].get(k), 10)) for k in END_TO_W))
    L.append("- Stored transfer numbers of this column: beyond 16 r_s last pre-reflection %s at %s r_s after launch; arrival %s r_s after launch; comparator column %s at %s, delay %s r_s; end includes reflection %s." % (
        out["column_stored"].get("beyond16_final_before_reflection"), out["column_stored"].get("last_pre_reflection_after_launch_rs"), out["column_stored"].get("arrival_after_launch_rs"),
        out["column_stored"].get("comparator_column"), out["column_stored"].get("comparator_arrival_after_launch_rs"), out["column_stored"].get("delay_minus_comparator_rs"), out["column_stored"].get("end_includes_reflection")))
    eng = out.get("engine", {})
    if eng and "error" not in eng:
        L.append("- The end record's fields: N_inf (last site) %s, v there %s; the record's zeros %s; the applied-field zeros of N + v (outer %s, inner %s; all %s); the stored R_t(end) %s minus the record's zero %s, minus the applied outer %s." % (
            fmt(eng["N_inf_end"], 12), fmt(eng["v_inf_end"], 6), eng["end_record_zeros"], fmt(eng["end_record_applied_outer"], 8), fmt(eng["end_record_applied_inner"], 8), eng["end_record_applied_zeros_all"],
            fmt(out["stored_Rt_end"], 8), fmt(eng["stored_Rt_end_minus_record_zero"], 3), fmt(eng["stored_Rt_end_minus_applied_outer"], 3)))
    else:
        L.append("- ENGINE UNAVAILABLE: %s; the operator, its extremes and the dense bins are not in this export (the column, the stored numbers and the layout check are)." % eng.get("error"))
    be = out["band_edges"]
    L.append("")
    L.append("## Band edges, half-widths, extremes")
    L.append("- m = %s/a, b = sqrt(m^2 + 1/a^2) = %s/a (the json's b minus this %s); N_inf m = %s, N_inf b = %s (N_inf = %s); the lapse displacements %s and %s; threshold half-widths %s/a at +-b and %s/a at +-m (r_s = %s a, R_max = %s a)." % (
        fmt(be["m"]), fmt(be["b"], 12), fmt(out["b_minus_json"], 3), fmt(be["N_inf_m"], 12), fmt(be["N_inf_b"], 12), fmt(be["N_inf_end"], 12), fmt(be["lapse_displacement_m"], 3), fmt(be["lapse_displacement_b"], 3), fmt(be["half_width_b"], 8), fmt(be["half_width_m"], 8), fmt(be["r_s_used"], 10), fmt(be["R_max"])))
    sp_ = out["spectra"]
    if sp_.get("chain"):
        c = sp_["chain"]
        L.append("- The full operator (%s): e_min %s, e_max %s (Lanczos); dense %s, %s%s." % (out.get("operator", {}).get("shape"), fmt(c.get("e_min_lanczos"), 12), fmt(c.get("e_max_lanczos"), 12), fmt(c.get("e_min_dense"), 12), fmt(c.get("e_max_dense"), 12),
                                                                                               (" in %.0f s" % c["dense_seconds"]) if c.get("dense_seconds") else ""))
    rows = []
    for cname, B in out["blocks"].items():
        sb = sp_["blocks"].get(cname, {})
        rows.append("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (cname, " (primary)" if cname == out["cuts"]["primary"] else "", fmt(B["cut_a"], 8), fmt(B["cut_over_rs"], 6), B["n_sites"], fmt(B["column_weight_on_block"], 10),
                                                                                  fmt(B["column_site_weight_max_on_block"], 6), fmt(B["its_radius"], 6), fmt(sb.get("e_min_lanczos"), 12), fmt(sb.get("e_max_lanczos"), 12), fmt(sb.get("e_max_dense"), 12), fmt(B["two_step_flow_bound_b_plus_b_rs_over_2Rcut"], 10)))
    L.append("")
    L.append("| block beyond the cut | cut (a) | cut (r_s) | sites | column weight on the block | max site weight | at R (a) | e_min (Lanczos) | e_max (Lanczos) | e_max (dense) | b + b r_s/(2 R_cut) (49.1's two-step bound, r_s the end record's 2GE) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    L.extend(rows)
    L.append("")
    L.append("## The 5 raw Killing bins and the threshold columns (the column's weights; the half-open convention (lo, hi])")
    L.append("| basis | %s | threshold at buffer 1: -b; -m; m; b; total | shares at buffer 1: below -m; gap; above m | threshold at buffer 2: total | shares at buffer 2 | threshold term: below -m; gap; above m; total | mean omega (1/a) |" % " | ".join(BIN_LABELS))
    L.append("|---|---|---|---|---|---|---|---|---|---|")

    def thr_row(name, thr):
        B1 = thr["buffers"].get("1", {}); B2 = thr["buffers"].get("2", {}); E = thr["threshold_error"]
        return "| %s | %s | %s; %s | %s; %s; %s | %s | %s; %s; %s | %s; %s; %s; %s | %s |" % (
            name, " | ".join(fmt(x, 8) for x in thr["raw"]["bins5"]), "; ".join(fmt(B1.get("threshold_weight", {}).get(e_), 4) for e_ in EDGES), fmt(B1.get("threshold_total"), 4),
            fmt(B1.get("w_below_minus_m"), 8), fmt(B1.get("w_in_gap"), 8), fmt(B1.get("w_above_m"), 8), fmt(B2.get("threshold_total"), 4), fmt(B2.get("w_below_minus_m"), 8), fmt(B2.get("w_in_gap"), 8), fmt(B2.get("w_above_m"), 8),
            fmt(E.get("w_below_minus_m"), 3), fmt(E.get("w_in_gap"), 3), fmt(E.get("w_above_m"), 3), fmt(E.get("threshold_total"), 3), fmt(thr["raw"]["mean"], 8))
    if sp_.get("chain") and sp_["chain"].get("thresholds"):
        L.append(thr_row("the full chain's eigenbasis (the producer's spectrum_end)", sp_["chain"]["thresholds"]))
    for cname, sb in sp_["blocks"].items():
        if sb.get("thresholds_raw_weights"):
            L.append(thr_row("the exterior block beyond %s, raw weights (sum = the block weight)" % cname, sb["thresholds_raw_weights"]))
            L.append(thr_row("the exterior block beyond %s, normalized to the block weight" % cname, sb["thresholds_normalized_to_block_weight"]))
    if sp_.get("chain") and sp_["chain"].get("compare_stored_spectrum_end"):
        L.append("")
        L.append("Recomputed against the producer's spectrum_end (max |difference| %s; flagged beyond %s):" % (fmt(out["checks"].get("chain_recomputed_vs_stored_max_abs"), 3), out["args"]["tol_stored"]))
        L.append("| quantity | recomputed | stored | recomputed - stored |")
        L.append("|---|---|---|---|")
        for r_ in sp_["chain"]["compare_stored_spectrum_end"]:
            flag = " (beyond tolerance)" if (r_["recomputed_minus_stored"] is not None and abs(r_["recomputed_minus_stored"]) > out["args"]["tol_stored"]) else ""
            L.append("| %s | %s | %s | %s%s |" % (r_["quantity"], fmt(r_["recomputed"], 10), fmt(r_["stored"], 10), fmt(r_["recomputed_minus_stored"], 3), flag))
    L.append("")
    L.append("## What each array of %s is" % os.path.basename(outs[".npz"]))
    for k, v in out["conventions"].items():
        L.append("- %s: %s" % (k, v))
    L.append("- arrays present: " + ", ".join("%s %s %s" % (k, list(np.shape(v)), getattr(v, "dtype", "")) for k, v in arrays.items()))
    L.append("- stored_spectrum_*_bins_*: the producer's 80-bin histograms of this column (raw, at each buffer, the threshold term); their edges are the producer's, not stated in the json.")
    L.append("")
    L.append("Not sayable from this column: escape, an emission rate, a property of the object; the exterior block's eigenvalue extremes are for the second_route lane's own binning (KPM scaling), their bins to be compared with the 5 raw bins and threshold columns here at the same buffers.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
