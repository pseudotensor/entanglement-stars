#!/usr/bin/env python
"""dust_root_history_v2.py (the solve lane; designed by reviewer A for the lane's builder; reorientation 4 item 7; the lineage of engine
spec 42.3 (4)). Version 1b's computation (engine_dust_root_history_v1b.py, never edited) with the applied roots, both of them, in place of
the dense zeros wherever version 1b used a root, the dense-root columns kept beside, every quantity version 1b printed recomputed and
printed with its difference from version 1b's json and npz.
Applied root = a zero of N + v on the record's applied site fields on the engine's sites, linear between sites: the outer root the largest
rising zero (version 1b's own finder), the inner root the largest falling zero below it (new: version 1b printed the record's inner trapping
zero, the dense inner root, as its applied inner root; the busi lane's dust_readouts_v1 found the field zero elsewhere). Dense (record) roots
= the record's trapping zeros (equal to the dense roots at the launch records, version 1b's check) and, at the launch record, B v2's dense
roots on the record's sources.
Root-dependent, now in two versions (dense, applied): the launch tags X_in and X_out (version 1b carried both cuts already), the weight
beyond R_t(t) per record, the moving-cut reassignment, the fixed-cut currents at the outer and inner roots, the first closing clock, the
outward crossing clocks. Root-independent (recomputed, compared): the weights beyond 16 and 24 r_s per tag, the current at 16 r_s, the
Killing bins at the launch and end records, the mean Killing energy, the jump ledger sampled every --jump-every-rs (recomputed by default;
--jump-ledger copy-v1b copies version 1b's rows, labelled; none skips them). Resumable: a checkpoint per key every --checkpoint-every-rs
(the propagated states and the arrays so far), completed keys kept in the json; --resume continues. CPU by default (--gpu -1), the threads
capped by --threads; the propagator the producer's Chebyshev action (page_ops_v6 through engine_single_object_B_v2) or scipy's
expm_multiply as the control. Outputs dust_root_history_v2.json, .npz (every per-record, per-column array, the dense columns also under
version 1b's names) and .md. No mutual information; no plotting; no dates; refuses to overwrite without --resume.
Usage (box): python dust_root_history_v2.py [--columns-npz ...] [--record ... --fields ... --record2 ... --fields2 ...] --out-prefix ... [--resume]
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
import scipy.linalg as sla
from scipy.sparse.linalg import expm_multiply

T0 = time.time()
REC1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json"
FLD1 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz"
REC2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json"
FLD2 = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz"
NMS = ("all", "in_dense", "out_dense", "in_applied", "out_applied")
CUTS = ("dense", "applied")
CURRENTS = ("outer_dense", "inner_dense", "outer_applied", "inner_applied", "r16")
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


def g(d, path, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


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


# ---------------------------------------------------------------- pure functions (the self-test imports these)
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
    """the outer root: the largest rising zero of N + v (version 1b's finder: the last i with f[i] < 0 <= f[i+1]); the inner root: the largest
    falling zero below it; None where absent. Every zero is returned third so a spurious dip is visible (the count is saved per record)."""
    z = zeros_of(R, f)
    rising = [r for r, s in z if s > 0]
    if not rising:
        return None, None, z
    Ro = max(rising)
    falling = [r for r, s in z if s < 0 and r < Ro]
    return Ro, (max(falling) if falling else None), z


def record_roots(rec):
    """version 1b's applied_roots(rec): the record's trapping zeros (the dense roots), largest and smallest; its R_t / r_in as the fallback."""
    z = rec.get("trapping_zeros") or []
    if z:
        return max(z), (min(z) if len(z) > 1 else None)
    return rec.get("R_t"), rec.get("r_in")


def site_w(X, n):
    """site weights of columns in the site-major layout (rows 2j, 2j+1 = site j): sites x columns."""
    return (np.abs(X) ** 2).reshape(n, 2, -1).sum(axis=1)


def killing_bins(e, U, X, m, b):
    """the 5 bins split at -b, -m, m, b (half-open (lo, hi]) of each column's weights on the eigenvectors U; and the mean Killing energy."""
    c = (X.conj().T @ U).conj().T
    w = np.abs(c) ** 2; ee = np.asarray(e, float)
    edges = [-np.inf, -b, -m, m, b, np.inf]
    bins = np.array([w[(ee > edges[i]) & (ee <= edges[i + 1])].sum(axis=0) for i in range(5)])
    return bins, (ee[:, None] * w).sum(axis=0)


def num_diff(x, y):
    return (float(x) - float(y)) if (isnum(x) and isnum(y)) else None


def arr_diff(x, y):
    """max |x - y| where both finite, the nan-pattern mismatches, the shapes; None where the shapes differ."""
    try:
        x = np.asarray(x, float); y = np.asarray(y, float)
    except (TypeError, ValueError):
        return {"shape_v2": None, "shape_v1b": None, "max_abs_diff": None, "nan_mismatch": None, "both_finite": 0}
    if x.shape != y.shape:
        return {"shape_v2": list(x.shape), "shape_v1b": list(y.shape), "max_abs_diff": None, "nan_mismatch": None, "both_finite": 0}
    fx, fy = np.isfinite(x), np.isfinite(y); both = fx & fy
    return {"shape_v2": list(x.shape), "shape_v1b": list(y.shape), "max_abs_diff": float(np.abs(x[both] - y[both]).max()) if both.any() else None,
            "nan_mismatch": int((fx != fy).sum()), "both_finite": int(both.sum())}


def lists_equal(a, b):
    """None-aware equality of two lists of clocks."""
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and abs(float(x) - float(y)) > 1e-12:
            return False
    return True


# ---------------------------------------------------------------- the engine (lazy)
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


class Backend(object):
    """the producer's operations (OPS.Banded, PH.cheb_apply, OPS.eigh_csr on dev) or the scipy controls, behind one interface on numpy states."""

    def __init__(self, a, torch, BV2, dev):
        self.a = a; self.torch = torch; self.BV2 = BV2; self.dev = dev
        self.OPS, self.PH, self.CD = BV2.OPS, BV2.PH, BV2.CD

    def csr(self, h):
        return h if sp.issparse(h) else sp.csr_matrix(h)

    def propagate(self, h, Snp, tau):
        if self.a.propagator == "ops":
            Bk = self.OPS.Banded(h, self.dev)
            Y, K, _ = self.PH.cheb_apply(Bk, self.torch.as_tensor(Snp, dtype=self.CD, device=self.dev), tau, self.a.cheb_tol)
            return np.asarray(Y.cpu().numpy(), np.complex128), int(K)
        return np.asarray(expm_multiply(-1j * tau * self.csr(h), Snp), np.complex128), -1

    def apply(self, J, Snp):
        if self.a.propagator == "ops":
            Y = self.OPS.Banded(J, self.dev).apply(self.torch.as_tensor(Snp, dtype=self.CD, device=self.dev))
            return np.asarray(Y.cpu().numpy(), np.complex128)
        return np.asarray(self.csr(J) @ Snp, np.complex128)

    def eigh(self, h):
        if self.a.eig_backend == "ops":
            e, U = self.OPS.eigh_csr(h, self.dev)
            return np.asarray(e.cpu().numpy() if hasattr(e, "cpu") else e, float), np.asarray(U.cpu().numpy() if hasattr(U, "cpu") else U, np.complex128)
        A = self.csr(h).toarray()
        return sla.eigh(A, driver=self.a.eig_driver, overwrite_a=True, check_finite=False)


def fields(H, k):
    N, v = H.fields(k)
    return np.asarray(N, float), np.asarray(v, float)


def new_arrays(nr, nc):
    A = {}
    for nm in NMS:
        A["b16_" + nm] = np.zeros((nr, nc)); A["b24_" + nm] = np.zeros((nr, nc))
    for cut in CUTS:
        A["beyond_Rt_" + cut] = np.full((nr, nc), np.nan); A["Rt_" + cut] = np.full(nr, np.nan); A["Ri_" + cut] = np.full(nr, np.nan)
        A["moving_cut_step_" + cut] = np.full(nr, np.nan)
    for c in CURRENTS:
        A["current_" + c] = np.full(nr, np.nan)
    A["n_zeros_applied"] = np.full(nr, np.nan); A["norm"] = np.full((nr, nc), np.nan)
    return A


def ckpt_paths(prefix, key):
    return prefix + "_ckpt_%s.npz" % key, prefix + "_ckpt_%s.json" % key


def save_ckpt(prefix, key, j_done, S, A, meta):
    pz, pj = ckpt_paths(prefix, key)
    tmp = pz[:-4] + ".tmp.npz"
    np.savez(tmp, S_all=S["all"], S_in_dense=S["in_dense"], S_in_applied=S["in_applied"], **A)
    os.replace(tmp, pz)
    meta = dict(meta); meta["j_done"] = int(j_done)
    with open(pj + ".tmp", "w") as fh:
        json.dump(meta, fh, default=_jdefault)
    os.replace(pj + ".tmp", pj)


def store_arrays(arrays, key, A, tt, bins0, binsE, f):
    for k_, v in A.items():
        arrays["%s_%s" % (key, k_)] = v
    arrays["%s_T" % key] = tt; arrays["%s_bins_launch" % key] = bins0; arrays["%s_bins_end" % key] = binsE; arrays["%s_f" % key] = f
    # version 1b's names for the dense-root columns, so a version 1b reader runs unchanged
    for nm in NMS:
        for c in ("b16", "b24"):
            arrays["%s_%s_%s" % (key, nm, c)] = A["%s_%s" % (c, nm)]
    arrays["%s_beyond_Rt" % key] = A["beyond_Rt_dense"]; arrays["%s_Rt" % key] = A["Rt_dense"]; arrays["%s_Ri" % key] = A["Ri_dense"]
    arrays["%s_current_outer" % key] = A["current_outer_dense"]; arrays["%s_current_inner" % key] = A["current_inner_dense"]
    arrays["%s_moving_cut_step" % key] = A["moving_cut_step_dense"]


def v1b_array_pairs(key):
    pairs = []
    for nm in NMS:
        for c in ("b16", "b24"):
            pairs.append(("%s_%s_%s" % (key, nm, c), "%s_%s" % (c, nm)))
    pairs += [("%s_beyond_Rt" % key, "beyond_Rt_dense"), ("%s_Rt" % key, "Rt_dense"), ("%s_Ri" % key, "Ri_dense"), ("%s_current_outer" % key, "current_outer_dense"),
              ("%s_current_inner" % key, "current_inner_dense"), ("%s_current_r16" % key, "current_r16"), ("%s_moving_cut_step" % key, "moving_cut_step_dense")]
    return pairs


def compare_with_v1b(L2, L1, A, V1Z, key, tt, bins0, binsE, f, tol):
    """every quantity version 1b printed: v2, v1b, the difference; the per-record arrays against version 1b's npz."""
    res = {"available": L1 is not None, "scalars": [], "lists": [], "arrays": [], "jumps": [], "tol": tol}
    if L1 is None:
        return res

    def add(label, x, y):
        d = num_diff(x, y)
        res["scalars"].append({"quantity": label, "v2": x, "v1b": y, "v2_minus_v1b": d, "within_tol": (abs(d) <= tol) if d is not None else None})
    for k in ("dense_outer", "dense_inner", "applied_outer", "applied_inner"):
        add("root at launch, " + k, L2["roots_launch"][k], g(L1, "roots_launch." + k))
    add("first closing after launch (dense, version 1b's rule)", L2["first_closing_after_launch_over_rs"]["dense"], L1.get("first_closing_after_launch_over_rs"))
    add("first closing after launch (applied) against version 1b's", L2["first_closing_after_launch_over_rs"]["applied"], L1.get("first_closing_after_launch_over_rs"))
    for cut in CUTS:
        add("inside the %s root at launch" % cut, L2["inside_root_at_launch"][cut], g(L1, "inside_root_at_launch." + cut))
    add("occupation sum", L2["occupation_sum"], L1.get("occupation_sum"))
    for nm in NMS:
        for c in ("b16", "b24"):
            add("at the end, %s, beyond %s r_s" % (nm, c[1:]), L2["end"][nm][c], g(L1, "end.%s.%s" % (nm, c)))
    for cut in CUTS:
        add("cross term beyond 16 r_s at the end, %s cut" % cut, L2["cross_term_b16_end"][cut], g(L1, "cross_term_b16_end." + cut))
    add("Chebyshev K max", L2["cheb_K_max"], L1.get("cheb_K_max"))
    for k in ("killing_launch_per_column", "killing_end_per_column", "mean_killing_launch", "mean_killing_end"):
        d = arr_diff(L2[k], L1.get(k)) if isinstance(L1.get(k), list) else {"max_abs_diff": None, "nan_mismatch": None, "shape_v2": None, "shape_v1b": None}
        res["lists"].append(dict(quantity=k, **d))
    c1 = L1.get("outward_crossing_clock_per_column")
    for cut in CUTS:
        res["lists"].append({"quantity": "outward crossing clocks (%s) against version 1b's" % cut, "equal": lists_equal(L2["outward_crossing_clock_per_column"][cut], c1),
                             "v2": L2["outward_crossing_clock_per_column"][cut], "v1b": c1})
    if V1Z is not None:
        for v1name, v2name in v1b_array_pairs(key):
            if v1name in V1Z.files:
                res["arrays"].append(dict(array=v1name, v2_array=v2name, **arr_diff(A[v2name], V1Z[v1name])))
            else:
                res["arrays"].append({"array": v1name, "v2_array": v2name, "max_abs_diff": None, "note": "not in version 1b's npz"})
        for v1name, v2arr in (("%s_T" % key, tt), ("%s_bins_launch" % key, bins0), ("%s_bins_end" % key, binsE), ("%s_f" % key, f)):
            if v1name in V1Z.files:
                res["arrays"].append(dict(array=v1name, v2_array=v1name[len(key) + 1:], **arr_diff(v2arr, V1Z[v1name])))
    rows1 = g(L1, "jump_ledger_sampled.rows") or []
    rows2 = L2["jump_ledger_sampled"]["rows"]
    for i in range(max(len(rows1), len(rows2))):
        r1 = rows1[i] if i < len(rows1) else {}; r2 = rows2[i] if i < len(rows2) else {}
        pc = arr_diff(r2.get("above_m_jump_per_column"), r1.get("above_m_jump_per_column")) if (r1 and r2) else {"max_abs_diff": None}
        bd = None
        if r1 and r2:
            d1 = arr_diff(r2.get("bins_before"), r1.get("bins_before"))["max_abs_diff"]; d2 = arr_diff(r2.get("bins_after"), r1.get("bins_after"))["max_abs_diff"]
            bd = max([x for x in (d1, d2) if x is not None], default=None)
        res["jumps"].append({"t_over_rs_v2": r2.get("t_over_rs"), "t_over_rs_v1b": r1.get("t_over_rs"), "v2_weighted": r2.get("above_m_jump_weighted"), "v1b_weighted": r1.get("above_m_jump_weighted"),
                             "diff": num_diff(r2.get("above_m_jump_weighted"), r1.get("above_m_jump_weighted")), "per_column_max_diff": pc["max_abs_diff"], "bins_max_diff": bd})
    return res


def dense_vs_applied(A, f):
    d = {}
    for which in ("Rt", "Ri"):
        x, y = A[which + "_dense"], A[which + "_applied"]
        both = np.isfinite(x) & np.isfinite(y)
        d[which] = {"records_dense": int(np.isfinite(x).sum()), "records_applied": int(np.isfinite(y).sum()), "records_both": int(both.sum()),
                    "only_dense": int((np.isfinite(x) & ~np.isfinite(y)).sum()), "only_applied": int((~np.isfinite(x) & np.isfinite(y)).sum()),
                    "max_abs_diff_both": float(np.abs(x[both] - y[both]).max()) if both.any() else None, "mean_applied_minus_dense_both": float((y[both] - x[both]).mean()) if both.any() else None}
    wb = {cut: A["beyond_Rt_" + cut] @ f for cut in CUTS}
    both = np.isfinite(wb["dense"]) & np.isfinite(wb["applied"])
    d["weighted_beyond_Rt_max_abs_diff_both"] = float(np.abs(wb["dense"][both] - wb["applied"][both]).max()) if both.any() else None
    for nm in ("moving_cut_step", "current_outer", "current_inner"):
        x, y = A[nm + "_dense"], A[nm + "_applied"]; both = np.isfinite(x) & np.isfinite(y)
        d[nm + "_max_abs_diff_both"] = float(np.abs(x[both] - y[both]).max()) if both.any() else None
    for cut in CUTS:
        c = A["current_outer_" + cut]; fin = np.isfinite(c)
        d["current_outer_%s_outward_records" % cut] = int((c[fin] > 0).sum()); d["current_outer_%s_records" % cut] = int(fin.sum())
        d["current_outer_%s_max" % cut] = float(c[fin].max()) if fin.any() else None; d["current_outer_%s_min" % cut] = float(c[fin].min()) if fin.any() else None
    return d


def build_parser():
    ap = argparse.ArgumentParser(description="DUST_root_history version 2: version 1b with the applied roots, both, in place of the dense zeros; resumable")
    ap.add_argument("--columns-npz", default="<internal>/B5v2_cols_prod.npz")
    ap.add_argument("--record", default=REC1); ap.add_argument("--fields", default=FLD1)
    ap.add_argument("--record2", default=REC2); ap.add_argument("--fields2", default=FLD2); ap.add_argument("--switch-T-rs", type=float, default=40.0)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--tags", default=None, help="comma list; default every tag in the columns npz"); ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--keys", default=None, help="comma list of keys TAG_kK; default version 1b's keys when its json is given, else every tag x kappa")
    ap.add_argument("--T-end-rs", type=float, default=56.0)
    ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--propagator", default="ops", choices=["ops", "scipy"], help="ops: the producer's Chebyshev action (PH.cheb_apply); scipy: expm_multiply, the busi lane's control")
    ap.add_argument("--eig-backend", default="scipy", choices=["scipy", "ops"]); ap.add_argument("--eig-driver", default="evr")
    ap.add_argument("--jump-every-rs", type=float, default=4.0)
    ap.add_argument("--jump-ledger", default="recompute", choices=["recompute", "copy-v1b", "none"])
    ap.add_argument("--crossing-threshold", type=float, default=0.5, help="a column's outward crossing clock: the first record with its weight beyond R_t above this")
    ap.add_argument("--checkpoint-every-rs", type=float, default=4.0)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--root-search-sites", default="ne", help="'ne' (the engine's sites, version 1b's domain) or an integer count of sites from the centre")
    ap.add_argument("--gpu", type=int, default=-1); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--engine-dir", default="<internal>/code"); ap.add_argument("--page-v6-path", default="<internal>/v6")
    ap.add_argument("--v1b-json", default="<internal>/dust_root_history_v1b.json"); ap.add_argument("--v1b-npz", default="<internal>/dust_root_history_v1b.npz")
    ap.add_argument("--tol-v1b", type=float, default=1e-6, help="the acceptance for the dense columns against version 1b (the busi lane's 1e-6)")
    ap.add_argument("--out-prefix", default="<internal>/dust_root_history_v2")
    ap.add_argument("--resume", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    outs = {ext: a.out_prefix + ext for ext in (".json", ".npz", ".md")}
    existing = [p for p in outs.values() if os.path.exists(p)]
    if existing and not a.resume:
        sys.exit("refusing to overwrite %s (pass --resume to continue)" % existing)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    torch, BV2 = load_engine(a.engine_dir, a.page_v6_path)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    BE = Backend(a, torch, BV2, dev); OPS = BE.OPS
    H = BV2.CombinedHistory(a.record, a.fields, a.nsites, a.record2 or None, a.fields2 or None, a.switch_T_rs if a.record2 else None)
    rs = float(H.rs_nom); m = a.m; b = math.sqrt(m * m + 1.0); R = np.asarray(H.R, float); n = len(R)
    ne = int(getattr(H, "ne", n)) if a.root_search_sites == "ne" else int(a.root_search_sites)
    TT = np.asarray(H.T, float) / rs
    kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
    O = np.load(a.columns_npz)
    tags_all = sorted({k_.split("_")[2] for k_ in O.files if k_.startswith("cols_dust_")})
    tags = [t for t in a.tags.split(",") if t] if a.tags else tags_all
    kappas = [int(x) for x in a.kappas.split(",") if x]
    V1 = None; V1Z = None
    if a.v1b_json and os.path.exists(a.v1b_json):
        with open(a.v1b_json) as fh:
            V1 = json.load(fh)
    if a.v1b_npz and os.path.exists(a.v1b_npz):
        V1Z = np.load(a.v1b_npz)
    v1_launch = {L["key"]: L for L in (V1.get("launches", []) if V1 else []) if isinstance(L, dict) and "key" in L}
    wanted = [k for k in a.keys.split(",") if k] if a.keys else (list(v1_launch.keys()) if v1_launch else None)
    out = None; arrays = {}
    if a.resume and os.path.exists(outs[".json"]):
        with open(outs[".json"]) as fh:
            out = json.load(fh)
        if os.path.exists(outs[".npz"]):
            Zp = np.load(outs[".npz"]); arrays = {k: Zp[k] for k in Zp.files}
        log("resuming: %d launches done (%s)" % (len(out.get("launches", [])), ", ".join(L_["key"] for L_ in out.get("launches", []))))
    if out is None:
        out = {"script": "dust_root_history_v2.py", "script_md5": md5(os.path.abspath(__file__)), "args": vars(a), "columns_npz": a.columns_npz, "columns_npz_md5": md5(a.columns_npz),
               "files": getattr(H, "files", None), "engine_module": BV2.__file__, "engine_module_md5": md5(BV2.__file__),
               "ops_module": getattr(OPS, "__file__", None), "ops_module_md5": md5(OPS.__file__) if getattr(OPS, "__file__", None) else None,
               "v1b_json": a.v1b_json if V1 else None, "v1b_json_md5": md5(a.v1b_json) if V1 else None, "v1b_npz": a.v1b_npz if V1Z is not None else None, "v1b_npz_md5": md5(a.v1b_npz) if V1Z is not None else None,
               "m": m, "b": b, "rs_nominal": rs, "n_sites": n, "ne": ne, "T_end_over_rs": float(TT[kend]), "k_end": kend, "bins": BIN_LABELS,
               "rules": {"applied_outer_root": "the largest rising zero of N + v on the applied site fields on the first %d sites, linear between sites (version 1b's finder)" % ne,
                         "applied_inner_root": "the largest falling zero of N + v below the applied outer root (new in version 2; version 1b printed the record's inner trapping zero, the dense inner root)",
                         "dense_roots": "the record's trapping zeros per record (version 1b's per-record roots); at the launch record B v2's dense roots on the record's sources beside",
                         "root_dependent_quantities": "the launch tags, the weight beyond R_t(t), the moving-cut reassignment, the currents at the outer and inner roots, the first closing clock, the outward crossing clocks: each in a dense and an applied version",
                         "jump_ledger": a.jump_ledger, "propagator": a.propagator, "eig_backend": a.eig_backend, "crossing_threshold": a.crossing_threshold},
               "not_in_v2": ["the tagging on HORIZON version 2's columns 1 and 7", "the jump ledger at every record (sampled every %g r_s)" % a.jump_every_rs, "mutual information (none anywhere)"],
               "launches": []}
    done = {L_["key"] for L_ in out["launches"]}
    jstep = max(1, int(round(a.jump_every_rs * rs / float(H.tau))))
    ckstep = max(1, int(round(a.checkpoint_every_rs * rs / float(H.tau))))
    for tg in tags:
        T_rs = float(O["T_%s" % tg]); k0 = int(np.argmin(np.abs(TT - T_rs)))
        keys_here = ["%s_k%d" % (tg, kap) for kap in kappas if "cols_dust_%s_k%d" % (tg, kap) in O.files and (wanted is None or "%s_k%d" % (tg, kap) in wanted)]
        if all(k_ in done for k_ in keys_here) or not keys_here:
            continue
        # the record (dense) and applied roots at launch
        rd = BV2.roots_at(H, k0, a.fields_module)
        Ro_d, Ri_d, _ = BV2.outer_inner_of(rd, H.rec[k0])
        N0, v0 = fields(H, k0)
        Ro_a, Ri_a, za0 = applied_outer_inner(R[:ne], (N0 + v0)[:ne])
        # first closing after launch: dense by version 1b's rule, applied by the field zero
        kc_d = next((k for k in range(k0 + 1, len(TT)) if not (H.rec[k].get("trapping_zeros") or H.rec[k].get("R_t"))), None)
        kc_a = None
        for k in range(k0 + 1, len(TT)):
            Nk, vk = fields(H, k)
            if applied_outer_inner(R[:ne], (Nk + vk)[:ne])[0] is None:
                kc_a = k; break
        log("%s: launch T/r_s %.3f (record %d); dense roots %s / %s; applied roots %s / %s (all field zeros %s); first closing dense %s applied %s" % (
            tg, TT[k0], k0, fmt(Ro_d, 9), fmt(Ri_d, 9), fmt(Ro_a, 9), fmt(Ri_a, 9), za0, fmt(TT[kc_d] if kc_d is not None else None), fmt(TT[kc_a] if kc_a is not None else None)))
        for kap in kappas:
            key = "%s_k%d" % (tg, kap); kd = "cols_dust_%s_k%d" % (tg, kap)
            if key in done or key not in keys_here:
                continue
            t_c = time.time()
            A_ = np.asarray(O[kd], np.complex128); f = np.asarray(O["f_dust_%s_k%d" % (tg, kap)], float); nc = A_.shape[1]
            W = np.zeros((2 * n, nc), np.complex128); W[:A_.shape[0]] = A_
            mask_d = np.repeat(R < (Ro_d or 0.0), 2); mask_a = np.repeat(R < (Ro_a or 0.0), 2)
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            nr = kend - k0 + 1
            A = new_arrays(nr, nc)
            S = None; j_start = 0; jumps = []; Kmax = 0; bins0 = mean0 = None; inside0 = None
            pz, pj = ckpt_paths(a.out_prefix, key)
            if a.resume and os.path.exists(pz) and os.path.exists(pj):
                with open(pj) as fh:
                    meta = json.load(fh)
                Zc = np.load(pz)
                S = {"all": np.asarray(Zc["S_all"]), "in_dense": np.asarray(Zc["S_in_dense"]), "in_applied": np.asarray(Zc["S_in_applied"])}
                for k_ in A:
                    if k_ in Zc.files:
                        A[k_] = np.asarray(Zc[k_])
                j_start = int(meta["j_done"]) + 1; jumps = meta["jumps"]; Kmax = int(meta["cheb_K_max"])
                bins0 = np.asarray(meta["bins_launch"]); mean0 = np.asarray(meta["mean_launch"]); inside0 = meta["inside0"]
                log("%s: resumed at record index %d of %d" % (key, j_start, nr))
            if S is None:
                S = {"all": W, "in_dense": W * mask_d[:, None], "in_applied": W * mask_a[:, None]}
                inside0 = {cut: float(np.dot(f, site_w(S["in_" + cut], n).sum(axis=0))) for cut in CUTS}
                e, U = BE.eigh(OPS.clocked_csr(h0, Q, N0, v0)); bins0, mean0 = killing_bins(e, U, S["all"], m, b); del e, U
                log("%s: launch eigendecomposition done; weighted bins %s" % (key, np.array2string(bins0 @ f, precision=6)))
            meta = {"key": key, "cheb_K_max": Kmax, "jumps": jumps, "bins_launch": bins0, "mean_launch": mean0, "inside0": inside0, "nr": nr, "nc": nc}
            h_prev = None
            for j in range(j_start, nr):
                kk = k0 + j
                if j > 0:
                    if h_prev is None:
                        h_prev = OPS.clocked_csr(h0, Q, *fields(H, kk - 1))
                    tau = float(H.T[kk] - H.T[kk - 1])
                    for nm in S:
                        S[nm], K = BE.propagate(h_prev, S[nm], tau); Kmax = max(Kmax, K)
                Nk, vk = fields(H, kk)
                hn = OPS.clocked_csr(h0, Q, Nk, vk)
                if a.jump_ledger == "recompute" and j > 0 and kk < kend and (j % jstep == 0):
                    ea, Ua = BE.eigh(h_prev); ba, _ = killing_bins(ea, Ua, S["all"], m, b); del ea, Ua
                    eb, Ub = BE.eigh(hn); bb, _ = killing_bins(eb, Ub, S["all"], m, b); del eb, Ub
                    jumps.append({"t_over_rs": float(TT[kk]), "record": kk, "above_m_jump_weighted": float(np.dot(f, (bb[3] + bb[4]) - (ba[3] + ba[4]))),
                                  "above_m_jump_per_column": ((bb[3] + bb[4]) - (ba[3] + ba[4])).tolist(), "bins_before": ba.T.tolist(), "bins_after": bb.T.tolist()})
                    log("%s: jump ledger at T/r_s %.3f: %+.4e (weighted)" % (key, TT[kk], jumps[-1]["above_m_jump_weighted"]))
                Wall = site_w(S["all"], n); A["norm"][j] = Wall.sum(axis=0)
                for cut in CUTS:
                    Win = site_w(S["in_" + cut], n); Wout = site_w(S["all"] - S["in_" + cut], n)
                    A["b16_in_" + cut][j] = Win[R > 16 * rs].sum(axis=0); A["b24_in_" + cut][j] = Win[R > 24 * rs].sum(axis=0)
                    A["b16_out_" + cut][j] = Wout[R > 16 * rs].sum(axis=0); A["b24_out_" + cut][j] = Wout[R > 24 * rs].sum(axis=0)
                A["b16_all"][j] = Wall[R > 16 * rs].sum(axis=0); A["b24_all"][j] = Wall[R > 24 * rs].sum(axis=0)
                rt_d, ri_d = record_roots(H.rec[kk])
                rt_a, ri_a, zk = applied_outer_inner(R[:ne], (Nk + vk)[:ne])
                A["Rt_dense"][j] = rt_d if rt_d else np.nan; A["Ri_dense"][j] = ri_d if ri_d else np.nan
                A["Rt_applied"][j] = rt_a if rt_a else np.nan; A["Ri_applied"][j] = ri_a if ri_a else np.nan; A["n_zeros_applied"][j] = len(zk)
                for cut, rt in (("dense", rt_d), ("applied", rt_a)):
                    if rt:
                        A["beyond_Rt_" + cut][j] = Wall[R > rt].sum(axis=0)
                        if j > 0 and np.isfinite(A["Rt_" + cut][j - 1]):
                            A["moving_cut_step_" + cut][j] = float(np.dot(f, Wall[R > rt].sum(axis=0) - Wall[R > A["Rt_" + cut][j - 1]].sum(axis=0)))
                for cname, cutR in (("outer_dense", rt_d), ("inner_dense", ri_d), ("outer_applied", rt_a), ("inner_applied", ri_a), ("r16", 16 * rs)):
                    if cutR:
                        JX = BE.apply(OPS.number_current_csr(hn, R, cutR), S["all"])
                        A["current_" + cname][j] = float(np.dot(f, (S["all"].conj() * JX).sum(axis=0).real))
                h_prev = hn
                if (j % ckstep == 0 and j > 0) or j == nr - 1:
                    meta["cheb_K_max"] = Kmax; meta["jumps"] = jumps
                    save_ckpt(a.out_prefix, key, j, S, A, meta)
                    log("%s: record %d/%d (T/r_s %.3f) checkpointed; dense R_t %s, applied R_t %s; beyond 16 r_s (all, weighted) %.4e" % (key, j + 1, nr, TT[kk], fmt(rt_d, 6), fmt(rt_a, 6), float(np.dot(f, A["b16_all"][j]))))
            Nn, vn = fields(H, kend)
            e, U = BE.eigh(OPS.clocked_csr(h0, Q, Nn, vn)); binsE, meanE = killing_bins(e, U, S["all"], m, b); del e, U
            tt = TT[k0: kend + 1]
            cross = {cut: [next((float(tt[i]) for i in range(nr) if A["beyond_Rt_" + cut][i, c] > a.crossing_threshold), None) for c in range(nc)] for cut in CUTS}
            if a.jump_ledger == "copy-v1b":
                jumps = list(g(v1_launch.get(key, {}), "jump_ledger_sampled.rows") or [])
                jump_label = "version 1b's rows copied (root-independent, not recomputed), sampled every %g r_s, not summed over every jump" % a.jump_every_rs
            elif a.jump_ledger == "none":
                jump_label = "not computed in this run"
            else:
                jump_label = "recomputed; sampled every %g r_s, not summed over every jump" % a.jump_every_rs
            fw = lambda v: float(np.dot(f, v))
            L_ = {"key": key, "tag": tg, "kappa": kap, "T_launch_over_rs": float(TT[k0]), "k0": k0, "k_end": kend, "T_end_over_rs": float(TT[kend]), "n_records": nr, "n_columns": nc,
                  "f": f.tolist(), "occupation_sum": float(f.sum()),
                  "roots_launch": {"dense_outer": Ro_d, "dense_inner": Ri_d, "applied_outer": Ro_a, "applied_inner": Ri_a, "applied_zeros_all": [{"R": r_, "slope_sign": s_} for r_, s_ in za0],
                                   "record_zeros": list(H.rec[k0].get("trapping_zeros") or []), "sites_inside_dense": int((R < (Ro_d or 0.0)).sum()), "sites_inside_applied": int((R < (Ro_a or 0.0)).sum())},
                  "first_closing_after_launch_over_rs": {"dense": float(TT[kc_d]) if kc_d is not None else None, "applied": float(TT[kc_a]) if kc_a is not None else None},
                  "inside_root_at_launch": inside0,
                  "end": {nm: {"b16": fw(A["b16_" + nm][-1]), "b24": fw(A["b24_" + nm][-1])} for nm in NMS},
                  "cross_term_b16_end": {cut: fw(A["b16_all"][-1] - A["b16_in_" + cut][-1] - A["b16_out_" + cut][-1]) for cut in CUTS},
                  "cross_term_b24_end": {cut: fw(A["b24_all"][-1] - A["b24_in_" + cut][-1] - A["b24_out_" + cut][-1]) for cut in CUTS},
                  "killing_launch_per_column": bins0.T.tolist(), "killing_end_per_column": binsE.T.tolist(), "killing_launch_weighted": (bins0 @ f).tolist(), "killing_end_weighted": (binsE @ f).tolist(),
                  "mean_killing_launch": mean0.tolist(), "mean_killing_end": meanE.tolist(),
                  "outward_crossing_clock_per_column": cross,
                  "jump_ledger_sampled": {"mode": a.jump_ledger, "every_rs": a.jump_every_rs, "label": jump_label, "rows": jumps},
                  "dense_vs_applied": dense_vs_applied(A, f), "norm_drift_max": float(np.nanmax(np.abs(A["norm"] - A["norm"][0]))) if np.isfinite(A["norm"]).any() else None,
                  "cheb_K_max": Kmax, "seconds": time.time() - t_c}
            L_["v1b_comparison"] = compare_with_v1b(L_, v1_launch.get(key), A, V1Z, key, tt, bins0, binsE, f, a.tol_v1b)
            out["launches"].append(L_); done.add(key)
            store_arrays(arrays, key, A, tt, bins0, binsE, f)
            with open(outs[".json"] + ".tmp", "w") as fh:
                json.dump(out, fh, default=_jdefault)
            os.replace(outs[".json"] + ".tmp", outs[".json"])
            np.savez(outs[".npz"] + ".tmp.npz", **arrays); os.replace(outs[".npz"] + ".tmp.npz", outs[".npz"])
            with open(outs[".md"], "w") as fh:
                fh.write(compose_md(out, a))
            if not a.keep_checkpoints:
                for p in ckpt_paths(a.out_prefix, key):
                    if os.path.exists(p):
                        os.remove(p)
            log("%s done in %.0f s: beyond 16 r_s at %.3f: all %.4e; from inside the dense root %.4e, the applied root %.4e; crossing clocks dense %s applied %s; first closing dense %s applied %s" % (
                key, L_["seconds"], TT[kend], L_["end"]["all"]["b16"], L_["end"]["in_dense"]["b16"], L_["end"]["in_applied"]["b16"], cross["dense"], cross["applied"],
                L_["first_closing_after_launch_over_rs"]["dense"], L_["first_closing_after_launch_over_rs"]["applied"]))
    out["seconds_total_this_run"] = time.time() - T0
    with open(outs[".json"] + ".tmp", "w") as fh:
        json.dump(out, fh, default=_jdefault)
    os.replace(outs[".json"] + ".tmp", outs[".json"])
    with open(outs[".md"], "w") as fh:
        fh.write(compose_md(out, a))
    log("wrote %s, %s (%d arrays), %s" % (outs[".json"], outs[".npz"], len(arrays), outs[".md"]))


def compose_md(out, a):
    L = ["# DUST_root_history_v2: the shed dust against the applied roots (version 1b's computation with the zeros of N + v on the applied site fields in place of the dense zeros; the dense-root columns beside; every quantity version 1b printed with its difference)", ""]
    L.append("Label: the dust = the stored dust-tagged occupied columns with their occupations f (B.5 version 2's selection) on the production's solved history (the records then B's continuation), per radial channel, "
             "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75; the massive channel ma 0.2 only; counts in fermions, currents per unit lattice time, outward positive; "
             "weight above m is not escape, weight above b has no propagating partner at infinity and is never read as unbound; the applied roots are the applied site fields' zeros of N + v, the dense roots the record's trapping zeros; no mutual information.")
    L.append("")
    L.append("Produced by dust_root_history_v2.py (md5 %s) with engine_single_object_B_v2 (md5 %s) and the ops module (md5 %s), the propagator %s (Chebyshev tolerance %s), the eigendecompositions by %s; the columns %s (md5 %s); "
             "the history %s; version 1b's json %s (md5 %s) and npz %s (md5 %s); m = %s, b = %s, the records to T/r_s %s; the acceptance for the dense columns against version 1b %s; the jump ledger %s." % (
                 out.get("script_md5"), out.get("engine_module_md5"), out.get("ops_module_md5"), g(out, "rules.propagator"), out["args"].get("cheb_tol"), g(out, "rules.eig_backend"), out["columns_npz"], out["columns_npz_md5"],
                 out.get("files"), out.get("v1b_json"), out.get("v1b_json_md5"), out.get("v1b_npz"), out.get("v1b_npz_md5"), fmt(out["m"]), fmt(out["b"], 10), fmt(out["T_end_over_rs"]), fmt(out["args"].get("tol_v1b")), g(out, "rules.jump_ledger")))
    L.append("")
    L.append("Rules: " + "; ".join("%s: %s" % (k, v) for k, v in out["rules"].items()) + ". Not in version 2: " + "; ".join(out["not_in_v2"]) + ".")
    for Lk in out["launches"]:
        key = Lk["key"]; cmp_ = Lk.get("v1b_comparison", {})
        L.append("")
        L.append("## %s: launch T/r_s %s (record %s), end %s (record %s), %d records, %d columns, f %s, sum %s, %.0f s" % (
            key, fmt(Lk["T_launch_over_rs"]), Lk["k0"], fmt(Lk["T_end_over_rs"]), Lk["k_end"], Lk["n_records"], Lk["n_columns"], Lk["f"], fmt(Lk["occupation_sum"]), Lk["seconds"]))
        rl = Lk["roots_launch"]
        L.append("")
        L.append("Roots at launch (a): dense outer %s, dense inner %s (the record's zeros %s; B v2's dense roots on the sources); applied outer %s, applied inner %s (all field zeros %s); sites inside the dense root %d, the applied root %d. "
                 "First closing after launch (T/r_s): dense (version 1b's rule, no record zero) %s; applied (no field zero) %s." % (
                     fmt(rl["dense_outer"], 10), fmt(rl["dense_inner"], 10), rl["record_zeros"], fmt(rl["applied_outer"], 10), fmt(rl["applied_inner"], 10),
                     ", ".join("%s (%s)" % (fmt(z["R"], 8), "rising" if z["slope_sign"] > 0 else "falling") for z in rl["applied_zeros_all"]), rl["sites_inside_dense"], rl["sites_inside_applied"],
                     fmt(Lk["first_closing_after_launch_over_rs"]["dense"]), fmt(Lk["first_closing_after_launch_over_rs"]["applied"])))
        if cmp_.get("available"):
            L.append("")
            L.append("| quantity | version 2 | version 1b | v2 - v1b | within %s |" % fmt(cmp_["tol"]))
            L.append("|---|---|---|---|---|")
            for r_ in cmp_["scalars"]:
                L.append("| %s | %s | %s | %s | %s |" % (r_["quantity"], fmt(r_["v2"], 10), fmt(r_["v1b"], 10), fmt(r_["v2_minus_v1b"], 3), fmt(r_["within_tol"])))
            L.append("")
            L.append("| per-column lists | max abs diff | nan mismatches | shapes v2 / v1b | equal (clocks) |")
            L.append("|---|---|---|---|---|")
            for r_ in cmp_["lists"]:
                L.append("| %s | %s | %s | %s / %s | %s |" % (r_["quantity"], fmt(r_.get("max_abs_diff"), 3), fmt(r_.get("nan_mismatch")), r_.get("shape_v2"), r_.get("shape_v1b"), fmt(r_.get("equal"))))
        else:
            L.append("")
            L.append("Version 1b's launch %s not available for comparison (no json given or the key absent)." % key)
            L.append("")
            L.append("| quantity | version 2 |")
            L.append("|---|---|")
            for cut in CUTS:
                L.append("| inside the %s root at launch | %s |" % (cut, fmt(Lk["inside_root_at_launch"][cut], 10)))
            for nm in NMS:
                for c in ("b16", "b24"):
                    L.append("| at the end, %s, beyond %s r_s | %s |" % (nm, c[1:], fmt(Lk["end"][nm][c], 10)))
        L.append("")
        L.append("Cross terms at the end (remainder all - in - out): beyond 16 r_s dense %s, applied %s; beyond 24 r_s dense %s, applied %s. Outward crossing clocks (T/r_s; first record with the column's weight beyond R_t above %s): dense %s, applied %s. "
                 "Killing bins weighted, launch %s, end %s; mean Killing energy per column launch %s, end %s. Norm drift max %s; Chebyshev K max %s." % (
                     fmt(Lk["cross_term_b16_end"]["dense"], 6), fmt(Lk["cross_term_b16_end"]["applied"], 6), fmt(Lk["cross_term_b24_end"]["dense"], 6), fmt(Lk["cross_term_b24_end"]["applied"], 6), fmt(a.crossing_threshold),
                     Lk["outward_crossing_clock_per_column"]["dense"], Lk["outward_crossing_clock_per_column"]["applied"], [fmt(x, 6) for x in Lk["killing_launch_weighted"]], [fmt(x, 6) for x in Lk["killing_end_weighted"]],
                     [fmt(x, 6) for x in Lk["mean_killing_launch"]], [fmt(x, 6) for x in Lk["mean_killing_end"]], fmt(Lk["norm_drift_max"], 2), Lk["cheb_K_max"]))
        dva = Lk["dense_vs_applied"]
        L.append("")
        L.append("Dense against applied within version 2, per record: outer root present at %d dense / %d applied records (%d both, %d only dense, %d only applied), max |applied - dense| %s a over both, mean %s a; inner root at %d / %d (%d both, %d only dense, %d only applied), max |diff| %s a; "
                 "the weighted weight beyond R_t differs by at most %s; the moving-cut step by %s; the outer current by %s, the inner current by %s; the outer current outward at %d of %d records (dense; max %s, min %s) and %d of %d (applied; max %s, min %s)." % (
                     dva["Rt"]["records_dense"], dva["Rt"]["records_applied"], dva["Rt"]["records_both"], dva["Rt"]["only_dense"], dva["Rt"]["only_applied"], fmt(dva["Rt"]["max_abs_diff_both"], 4), fmt(dva["Rt"]["mean_applied_minus_dense_both"], 4),
                     dva["Ri"]["records_dense"], dva["Ri"]["records_applied"], dva["Ri"]["records_both"], dva["Ri"]["only_dense"], dva["Ri"]["only_applied"], fmt(dva["Ri"]["max_abs_diff_both"], 4),
                     fmt(dva["weighted_beyond_Rt_max_abs_diff_both"], 3), fmt(dva["moving_cut_step_max_abs_diff_both"], 3), fmt(dva["current_outer_max_abs_diff_both"], 3), fmt(dva["current_inner_max_abs_diff_both"], 3),
                     dva["current_outer_dense_outward_records"], dva["current_outer_dense_records"], fmt(dva["current_outer_dense_max"], 3), fmt(dva["current_outer_dense_min"], 3),
                     dva["current_outer_applied_outward_records"], dva["current_outer_applied_records"], fmt(dva["current_outer_applied_max"], 3), fmt(dva["current_outer_applied_min"], 3)))
        jl = Lk["jump_ledger_sampled"]
        if jl["rows"]:
            L.append("")
            L.append("Jump ledger (%s): Delta p above m = Tr[(1_{(m,inf)}(h_k) - 1_{(m,inf)}(h_{k-1})) C_d(t_k)], weighted; version 1b's beside where recomputed." % jl["label"])
            L.append("| T/r_s | record | Delta p above m, weighted | per column | 5 bins before (weighted) | 5 bins after (weighted) | version 1b weighted | diff | per column max diff | bins max diff |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            f = np.asarray(Lk["f"], float)
            for i, r_ in enumerate(jl["rows"]):
                cj = cmp_.get("jumps", [])[i] if i < len(cmp_.get("jumps", [])) else {}
                bb = np.asarray(r_.get("bins_before"), float); ba = np.asarray(r_.get("bins_after"), float)
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (fmt(r_.get("t_over_rs")), fmt(r_.get("record")), fmt(r_.get("above_m_jump_weighted"), 6), [fmt(x, 3) for x in (r_.get("above_m_jump_per_column") or [])],
                                                                          [fmt(x, 6) for x in (f @ bb if bb.ndim == 2 else [])], [fmt(x, 6) for x in (f @ ba if ba.ndim == 2 else [])],
                                                                          fmt(cj.get("v1b_weighted"), 6), fmt(cj.get("diff"), 2), fmt(cj.get("per_column_max_diff"), 2), fmt(cj.get("bins_max_diff"), 2)))
        if cmp_.get("arrays"):
            L.append("")
            L.append("Per-record arrays against version 1b's npz (the dense columns under version 1b's names):")
            L.append("| array (version 1b's name) | version 2 array | shapes v2 / v1b | both finite | max abs diff | nan mismatches |")
            L.append("|---|---|---|---|---|---|")
            for r_ in cmp_["arrays"]:
                L.append("| %s | %s | %s / %s | %s | %s | %s |" % (r_["array"], r_["v2_array"], r_.get("shape_v2"), r_.get("shape_v1b"), fmt(r_.get("both_finite")), fmt(r_.get("max_abs_diff"), 3), fmt(r_.get("nan_mismatch"))))
    L.append("")
    L.append("Arrays (npz): per key the dense and applied per-record arrays <key>_{b16,b24}_{all,in_dense,out_dense,in_applied,out_applied} (records x columns), <key>_beyond_Rt_{dense,applied}, <key>_{Rt,Ri}_{dense,applied}, "
             "<key>_n_zeros_applied, <key>_current_{outer_dense,inner_dense,outer_applied,inner_applied,r16}, <key>_moving_cut_step_{dense,applied}, <key>_norm, <key>_T, <key>_bins_{launch,end}, <key>_f; the dense columns also under version 1b's names.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
