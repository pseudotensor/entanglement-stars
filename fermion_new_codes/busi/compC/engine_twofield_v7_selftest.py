#!/usr/bin/env python
"""engine_twofield_v7_selftest.py (the busi lane, Computation C): each stage --perf-v7 changes, against version 3's path on small inputs, on
the device C5.backend(--gpu) gives (the box CPU without --gpu; a pod's card with --gpu 0, which also exercises the device-side reorder), plus
2 tools for the gate. Subcommands:
  all | light | solves | writer | marks     the tests (default all; marks needs --v3 and --v7, the 2 source files)
  compare-runs A/TAG B/TAG [--after T] [--floor F]   2 runs' records and fields rows after the clock T (T/r_s), number by number: BITWISE,
                                            WITHIN-FLOOR (every relative difference at most F) or FAIL per quantity, the largest difference printed
Each test prints PASS or FAIL with the largest difference it found; the exit code is the number of failures. The engine modules are found as
the engine finds them (BUSI_ENGINE_DIR, BUSI_PAGE_DIR, default <internal>/engine and .../page) and engine_twofield_v7.py beside
this file. What the tests check, stage by stage:
  light:  light_continue_v36 against the 3 phases (A on the main thread, B on a thread, C on the main thread) per light channel on a 128-site
          chain, for a 1-piece walk, a many-piece walk and a forced fallback: the mask, the reference's columns, the rotated eigenvectors and every
          entry of the continuation's record (the seconds apart) bitwise; then light_eval_v36 against light_eval_begin_v7/light_eval_end_v7
          (2 eigendecompositions of the same matrix: their equality is the device's determinism the lane's bitwise gates rest on) and the counters;
  solves: engine_collapse_v36.solve on a plus and a minus source set here, against perf_solve_pair_v7 with a spawned worker (the kernel cache
          installed when importable): N, v, W and the evaluation counts bitwise, twice (a cold and a warm worker cache);
  writer: a run-file group (json with numpy scalars, 4 npz with object arrays) and a slice written directly and by the forked child: byte-equal json,
          equal npz keys and arrays; a child made to fail: the foreground redo and the failure count;
  marks:  every changed or added hunk of the v7 source against v3 carries the mark v7; no deletions without a replacement."""
import argparse
import difflib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
for _d in (os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"), os.environ.get("BUSI_PAGE_DIR", "<internal>/page")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.append(_d)

import numpy as np

SKIP_KEYS = {"seconds", "_sigma", "kernel_cache", "perf_v7"}


def _is_tensor(x):
    return hasattr(x, "device") and hasattr(x, "cpu") and hasattr(x, "numel")


def deep_diff(x, y, path, out, skip=SKIP_KEYS):
    """every difference between 2 nested structures: numbers exactly (NaN equals NaN), arrays and tensors bitwise; (path, kind, size) appended."""
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            out.append((path, "keys", sorted(set(x) ^ set(y))))
        for k in x:
            if k in y and k not in skip:
                deep_diff(x[k], y[k], path + "/" + str(k), out, skip)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append((path, "length", (len(x), len(y))))
            return
        for i, (p, q) in enumerate(zip(x, y)):
            deep_diff(p, q, path + "[%d]" % i, out, skip)
        return
    if _is_tensor(x) and _is_tensor(y):
        if tuple(x.shape) != tuple(y.shape):
            out.append((path, "shape", (tuple(x.shape), tuple(y.shape))))
        elif x.numel() and not bool((x == y).all()):
            out.append((path, "tensor", float((x - y).abs().max())))
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        xa = np.asarray(x); ya = np.asarray(y)
        if xa.shape != ya.shape:
            out.append((path, "shape", (xa.shape, ya.shape)))
        elif xa.dtype.kind == "O" or ya.dtype.kind == "O":
            for i, (p, q) in enumerate(zip(xa.ravel(), ya.ravel())):
                deep_diff(p, q, path + "[%d]" % i, out, skip)
        elif not np.array_equal(xa, ya, equal_nan=True) if xa.dtype.kind in "fc" else not np.array_equal(xa, ya):
            out.append((path, "array", float(np.max(np.abs(xa.astype(complex) - ya.astype(complex))))))
        return
    if isinstance(x, (bool, int, float, np.integer, np.floating)) and isinstance(y, (bool, int, float, np.integer, np.floating)):
        fx, fy = float(x), float(y)
        if fx != fy and not (fx != fx and fy != fy):
            out.append((path, "number", abs(fx - fy) / max(abs(fx), abs(fy), 1e-300)))
        return
    if x != y:
        out.append((path, "value", (repr(x)[:60], repr(y)[:60])))


def report(name, diffs, extra=""):
    if diffs:
        worst = max((d_[2] if isinstance(d_[2], float) else 0.0) for d_ in diffs)
        print("FAIL %s: %d differences, the largest %.3e%s; the first 5: %s" % (name, len(diffs), worst, extra, diffs[:5]))
        return 1
    print("PASS %s: bitwise%s" % (name, extra))
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------- light
def test_light(V7, gpu):
    import torch
    V7.C5.backend(gpu)
    n = 128
    a = SimpleNamespace(nsites=n, half_offset=0, light_kappas="1,2", light_guard_kappas="3", light_tol_block=1e-8, light_sigma_tol=1e-6, light_transport="banded",
                        light_d_min=1e-7, light_max_pieces=4096, tau=2.0, c_LZ=1.0, weyl_refine_max=10)
    chl = V7.light_channels_v36(a)
    sgn_dev, S_host = V7.light_S_v36(n)
    R = chl[0].R
    x = R / n
    N0 = 1.0 + 0.05 * np.exp(-((x - 0.4) / 0.12) ** 2)
    v0 = -0.25 * x * np.exp(-((x - 0.5) / 0.3) ** 2)
    try:
        stL, headL = V7.light_init_v36(a, chl, N0, v0, sgn_dev, S_host)
    except SystemExit as ex:
        print("FAIL light: the light seas could not be prepared at %d sites (%s)" % (n, ex))
        return 1
    fails = 0
    cases = [("one piece", 1e-4, 1e-6), ("many pieces", 2e-2, 1e-6), ("forced fallback (sector tolerance -1)", 1e-4, -1.0)]
    for label, scale, sig_tol in cases:
        a.light_sigma_tol = sig_tol
        N1 = N0 + scale * np.exp(-((x - 0.5) / 0.2) ** 2); v1 = v0 - scale * x
        for chL in chl:
            e, U = V7.eigT(V7.clocked_gpu(chL, N1, v1))
            Ua = U.clone(); Ub = U.clone()
            s_ = stL[chL.key]; window = s_["window"]; n_sea = s_["n_sea"]
            ref_a, info_a, mask_a = V7.light_continue_v36(a, chL, window, (N1, v1), e, Ua, n_sea, sgn_dev, S_host)
            A_ = V7.light_continue_phaseA_v7(a, chL, window, (N1, v1), e, Ub, n_sea, sgn_dev)
            box = {"B": None, "err": None}
            th = threading.Thread(target=V7._light_phaseB_thread_v7, args=(a, chL, A_, (N1, v1), n_sea, S_host, box))
            th.start(); th.join()
            if box["err"] is not None:
                print("FAIL light %s kappa %d: phase B raised %r" % (label, chL.kappa, box["err"][0])); fails += 1
                continue
            ref_b, info_b, mask_b = V7.light_continue_phaseC_v7(a, chL, window, (N1, v1), e, Ub, n_sea, A_, box["B"])
            diffs = []
            deep_diff(np.asarray(mask_a), np.asarray(mask_b), "mask", diffs)
            deep_diff(ref_a, ref_b, "ref", diffs)
            deep_diff(Ua, Ub, "U_rotated", diffs)
            deep_diff(info_a, info_b, "info", diffs)
            deep_diff(info_a["_sigma"], info_b["_sigma"], "sigma", diffs)
            cert = info_a.get("certified")
            extra = " (%s, kappa %d: %s, pieces %s, Cholesky %s, fallback %s)" % (label, chL.kappa, info_a.get("mode", "")[:40], (cert or {}).get("pieces") if isinstance(cert, dict) else None,
                                                                             (cert or {}).get("cholesky_calls") if isinstance(cert, dict) else None, info_a.get("fallback"))
            fails += report("light phases", diffs, extra)
    # the whole evaluation: light_eval_v36 against begin/end (2 eigendecompositions of the same matrix, the device's determinism included)
    a.light_sigma_tol = 1e-6
    N1 = N0 + 2e-2 * np.exp(-((x - 0.5) / 0.2) ** 2); v1 = v0 - 2e-2 * x
    V7.tml_reset_v36()
    cache_a = {}
    V7.light_eval_v36(a, chl, stL, cache_a, N1, v1, sgn_dev, S_host)
    tml_a = dict(V7.TML)
    V7.tml_reset_v36()
    V7.PERF["on"] = True; V7.PERF["stages"] = {"light"}; V7.perf_reset_v7()
    try:
        h = V7.light_eval_begin_v7(a, chl, stL, N1, v1, sgn_dev, S_host)
        cache_b = {}
        V7.light_eval_end_v7(a, chl, stL, cache_b, h)
    finally:
        V7.PERF["on"] = False; V7.PERF["stages"] = set()
    tml_b = dict(V7.TML)
    diffs = []
    for chL in chl:
        _h, e_a, U_a, ref_a, ci_a = cache_a[chL.key]; _h, e_b, U_b, ref_b, ci_b = cache_b[chL.key]
        deep_diff(e_a, e_b, "e[%d]" % chL.kappa, diffs); deep_diff(U_a, U_b, "U[%d]" % chL.kappa, diffs); deep_diff(ref_a, ref_b, "ref[%d]" % chL.kappa, diffs)
        deep_diff(ci_a, ci_b, "cinfo[%d]" % chL.kappa, diffs)
        deep_diff(np.asarray(ci_a["_mask"]), np.asarray(ci_b["_mask"]), "mask[%d]" % chL.kappa, diffs)
    for k_ in tml_a:
        if not k_.endswith("_s") and tml_a[k_] != tml_b[k_]:
            diffs.append(("TML/" + k_, "counter", (tml_a[k_], tml_b[k_])))
    fails += report("light_eval_v36 against light_eval_begin_v7/end_v7", diffs, " (device %s; host walks %d, %.3f s on threads, %.3f s waited)" % (
        V7.BK["dev"], V7.PERF["light_walks"], V7.PERF["light_host_s"], V7.PERF["light_wait_s"]))
    return fails


# --------------------------------------------------------------------------------------------------------------------------------- solves
def test_solves(V7, gpu):
    n = 128; g = 0.25; ell = 4
    R = np.arange(n, dtype=float) + 1.0
    eps = 0.4 * np.exp(-((R - 40.0) / 6.0) ** 2)
    J = -0.1 * eps
    t = 1e-4
    deps = 0.05 * eps * np.cos(R / 7.0); dJ = -0.02 * eps
    V7.OPT["collar_policy"] = "flag"; V7.OPT["support_rel"] = 1e-14; V7.OPT["site_fields"] = "averaged"
    V7.SOLVE_TOL["rtol"] = 1e-12; V7.SOLVE_TOL["atol"] = 1e-16
    fails = 0
    try:
        import engine_fields_kcache_v1 as KC  # noqa: F401
        kc_options = [False, True]
    except ImportError:
        kc_options = [False]
        print("note: engine_fields_kcache_v1 not importable here; the solves test runs without the kernel cache only")
    for kc in kc_options:
        if kc:
            import engine_fields_kcache_v1 as KC
            import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
            KC.install([_F4, _F12, _F14, _F16, V7.FLD]); V7.V35.V36["kcache"] = KC.STATS
        a = SimpleNamespace(perf_v7_stages="solves", fields_v17=False, field_kernel_cache=kc, v36_fixes=True, collar_policy="flag", support_rel=1e-14,
                            site_fields="averaged", field_rtol=1e-12, field_atol=1e-16, g=g, ell=ell)

        def do_solve(srcs_):
            t_s = time.time()
            out = V7.solve(srcs_[0], srcs_[1], g, R, ell, None)
            V7.TM["solve_s"] += time.time() - t_s; V7.TM["solve_n"] += 1
            if out[3].get("ivp"):
                V7.TM["solve_nfev"] += int(out[3]["ivp"]["nfev"])
            return out
        sp = (eps + t * deps, J + t * dJ); sm = (eps - t * deps, J - t * dJ)
        V7.tm_reset(); t0 = time.time()
        out_p = do_solve(sp); out_m = do_solve(sm)
        t_serial = time.time() - t0; tm_serial = dict(V7.TM)
        if out_p[3].get("failed") or out_m[3].get("failed"):
            print("FAIL solves (kernel cache %s): the serial solves failed (%s, %s)" % (kc, out_p[3].get("failed"), out_m[3].get("failed"))); fails += 1
            continue
        V7.perf_start_v7(a)
        try:
            if V7.PERF["pool"] is None:
                print("FAIL solves (kernel cache %s): the worker did not start" % kc); fails += 1
                continue
            V7.PERF["R"] = R
            for round_ in ("cold worker cache", "warm worker cache"):
                V7.tm_reset(); t0 = time.time()
                p2, m2 = V7.perf_solve_pair_v7(do_solve, sp, sm, a)
                t_pair = time.time() - t0; tm_pair = dict(V7.TM)
                diffs = []
                for i_, nm in ((0, "N"), (1, "v"), (2, "W_nodes")):
                    deep_diff(np.asarray(out_p[i_]), np.asarray(p2[i_]), "plus/" + nm, diffs)
                    deep_diff(np.asarray(out_m[i_]), np.asarray(m2[i_]), "minus(worker)/" + nm, diffs)
                if m2[3].get("failed") is not None:
                    diffs.append(("minus/failed", "value", (None, m2[3].get("failed"))))
                for k_ in ("solve_n", "solve_nfev"):
                    if tm_serial[k_] != tm_pair[k_]:
                        diffs.append(("TM/" + k_, "counter", (tm_serial[k_], tm_pair[k_])))
                fails += report("solve pair (%s, kernel cache %s)" % (round_, kc), diffs, " (serial %.2f s, pair %.2f s wall; worker pid %s)" % (
                    t_serial, t_pair, (V7.PERF.get("pool_ping") or {}).get("pid")))
        finally:
            V7.perf_finish_v7()
            V7.PERF["on"] = False; V7.PERF["stages"] = set()
    return fails


# --------------------------------------------------------------------------------------------------------------------------------- writer
def test_writer(V7, gpu):
    base = tempfile.mkdtemp(prefix="v7_writer_")
    fails = 0
    try:
        rng = np.random.default_rng(158)
        recs = [{"T": 2.0 * k, "T_over_rs": k / 16.0, "E": np.float64(rng.normal()), "flags": [True, False, None], "nested": {"x": np.float64(1e-300), "y": np.int64(7), "z": float("nan")},
                 "list": [np.float64(q) for q in rng.normal(size=5)]} for k in range(12)]
        head = {"E0": np.float64(31.4159), "label": "a test", "n": 12}
        fields = {k_: [rng.normal(size=16) for _ in range(12)] for k_ in ("N", "v")}
        nodes = [np.arange(3.0 + (k % 2)) for k in range(12)]
        spectra = {"T": [2.0 * k for k in range(12)], "e_1": [rng.normal(size=32) for _ in range(12)], "mask_1": [np.packbits(rng.random(32) > 0.5) for _ in range(12)]}
        arrs = {"T": 24.0, "sea_1": (rng.normal(size=(16, 8)) + 1j * rng.normal(size=(16, 8))).astype(np.complex128), "mask_1": np.packbits(np.ones(16, bool)), "dtype": "complex128"}

        def write_group(a_):
            V7.json_dump_v7(os.path.join(a_.out, "collapse_t.json"), {"script": "x", "head": head, "records": recs})
            np.savez_compressed(V7.run_file_v7(a_, "collapse_t_fields.npz"), N=np.array(fields["N"]), v=np.array(fields["v"]), r_nodes=np.array(nodes, dtype=object), label="l")
            np.savez(V7.run_file_v7(a_, "collapse_t_spectra.npz"), **{k_: np.array(v_) for k_, v_ in spectra.items()})
            V7.commit_run_files_v7()
        A = os.path.join(base, "a"); B = os.path.join(base, "b"); os.makedirs(A); os.makedirs(B)
        V7.PERF["on"] = False; V7.PERF["stages"] = set(); V7.PERF["writer_disabled"] = False; V7.PERF["writer_failures"] = 0
        write_group(SimpleNamespace(out=A))
        np.savez(os.path.join(A, "orbitals_t.npz"), **arrs)
        V7.PERF["on"] = True; V7.PERF["stages"] = {"writer"}; V7.perf_reset_v7()
        t0 = time.time()
        V7.persist_v7("test group", lambda: write_group(SimpleNamespace(out=B)))
        V7.savez_v7(os.path.join(B, "orbitals_t.npz"), arrs)
        t_parent = time.time() - t0
        V7.perf_writer_wait_v7(block=True)
        diffs = []
        if open(os.path.join(A, "collapse_t.json"), "rb").read() != open(os.path.join(B, "collapse_t.json"), "rb").read():
            diffs.append(("collapse_t.json", "bytes", 1.0))
        for nm in ("collapse_t_fields.npz", "collapse_t_spectra.npz", "orbitals_t.npz"):
            za = np.load(os.path.join(A, nm), allow_pickle=True); zb = np.load(os.path.join(B, nm), allow_pickle=True)
            if sorted(za.files) != sorted(zb.files):
                diffs.append((nm, "keys", sorted(set(za.files) ^ set(zb.files))))
            for k_ in za.files:
                if k_ in zb.files:
                    deep_diff(za[k_], zb[k_], nm + "/" + k_, diffs)
        left = [f_ for f_ in os.listdir(os.path.join(B, ".perf_v7_tmp"))] if os.path.isdir(os.path.join(B, ".perf_v7_tmp")) else []
        if left:
            diffs.append((".perf_v7_tmp", "leftover", left))
        fails += report("writer (forked child against the direct write)", diffs, " (the parent spent %.3f s on %d forks; %d writes, failures %d)" % (
            t_parent, V7.PERF["forks"], V7.PERF["writer_runs"], V7.PERF["writer_failures"]))
        # the failure path: a write that fails in the child only
        parent = os.getpid()
        C = os.path.join(base, "c"); os.makedirs(C)

        def flaky():
            if os.getpid() != parent:
                raise RuntimeError("the child fails on purpose")
            V7.json_dump_v7(os.path.join(C, "collapse_t.json"), {"records": recs})
        V7.persist_v7("flaky", flaky)
        V7.perf_writer_wait_v7(block=True)
        diffs = []
        if V7.PERF["writer_failures"] != 1:
            diffs.append(("writer_failures", "counter", (1, V7.PERF["writer_failures"])))
        if not os.path.exists(os.path.join(C, "collapse_t.json")):
            diffs.append(("collapse_t.json", "missing", 1.0))
        elif json.load(open(os.path.join(C, "collapse_t.json")))["records"][3]["T"] != recs[3]["T"]:
            diffs.append(("collapse_t.json", "content", 1.0))
        if V7.PERF["writer_disabled"]:
            diffs.append(("writer_disabled", "value", (False, True)))
        fails += report("writer failure path (the child exits 1, the parent redoes the write)", diffs)
    finally:
        V7.PERF["on"] = False; V7.PERF["stages"] = set(); V7.PERF["writer_disabled"] = False; V7.PERF["writer_failures"] = 0
        shutil.rmtree(base, ignore_errors=True)
    return fails


# ---------------------------------------------------------------------------------------------------------------------------------- marks
def test_marks(v3_path, v7_path):
    a_ = open(v3_path, encoding="utf-8").read().splitlines(); b_ = open(v7_path, encoding="utf-8").read().splitlines()
    doc_end = next(i for i, l_ in enumerate(b_) if i > 0 and l_.rstrip().endswith('"""'))
    sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
    bad = []; hunks = 0; added = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1; added += j2 - j1
        if j2 <= doc_end + 1:
            continue
        if tag == "delete":
            bad.append(("deleted without replacement", a_[i1:i2][:2]))
        elif not any("v7" in l_ for l_ in b_[j1:j2]):
            bad.append(("hunk without the mark v7", b_[j1:j2][:2]))
    if bad:
        print("FAIL marks: %d hunks, %d lines added, %d hunks without the mark: %s" % (hunks, added, len(bad), bad[:3]))
        return 1
    print("PASS marks: %d changed hunks, %d lines added, every hunk outside the docstring carries the mark v7, nothing deleted without a replacement" % (hunks, added))
    return 0


# --------------------------------------------------------------------------------------------------------------------------- compare-runs
def compare_runs(A, B, after, floor):
    """2 runs' json records (E not None, T/rs beyond `after`) and the fields npz rows at those clocks; the keys SKIP_KEYS and every seconds-like
    key skipped (wall clock), everything else number by number."""
    def load(run):
        d_, tag = os.path.split(run)
        J = json.load(open(os.path.join(d_, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(d_, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        return J, Z
    JA, ZA = load(A); JB, ZB = load(B)
    rs = float(JA["args"]["rs"])
    def recs(J):
        return {round(r["T"] / rs, 9): r for r in J["records"] if r.get("E") is not None and (after is None or r["T"] / rs > after + 1e-9)}
    RA, RB = recs(JA), recs(JB)
    common = sorted(set(RA) & set(RB))
    status = {}
    if set(RA) != set(RB):
        status["records present"] = ("FAIL", "clocks only in A %s, only in B %s" % (sorted(set(RA) - set(RB)), sorted(set(RB) - set(RA))))
    skip = set(SKIP_KEYS) | {"timing"}
    per_key = {}
    for t_ in common:
        ra, rb = RA[t_], RB[t_]
        for k_ in set(ra) | set(rb):
            if k_ in skip or k_.endswith("_s"):
                continue
            diffs = []
            deep_diff(ra.get(k_), rb.get(k_), "%s@%.4f" % (k_, t_), diffs, skip | {k for k in SKIP_KEYS})
            worst = per_key.get(k_, 0.0)
            for d_ in diffs:
                worst = max(worst, d_[2] if isinstance(d_[2], float) else float("inf"))
            per_key[k_] = worst
        ta, tb = ra.get("timing") or {}, rb.get("timing") or {}
        for k_ in set(ta) | set(tb):
            if k_.endswith("_s"):
                continue
            if ta.get(k_) != tb.get(k_):
                per_key["timing/" + k_] = float("inf")
    for k_, w_ in sorted(per_key.items()):
        status["record/" + k_] = ("BITWISE", "") if w_ == 0.0 else (("WITHIN-FLOOR", "max rel %.3e" % w_) if w_ <= floor else ("FAIL", "max rel %.3e" % w_))
    TA = np.asarray(ZA["T"], float) / rs; TB = np.asarray(ZB["T"], float) / rs
    for k_ in ZA.files:
        arr = ZA[k_]
        if k_ not in ZB.files or not (hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] == len(TA)) or k_ == "T":
            continue
        brr = ZB[k_]
        worst = 0.0
        for t_ in common:
            ia = int(np.argmin(np.abs(TA - t_))); ib = int(np.argmin(np.abs(TB - t_)))
            x = np.asarray(arr[ia], float) if arr.dtype.kind != "O" else np.asarray(arr[ia], float); y = np.asarray(brr[ib], float) if brr.dtype.kind != "O" else np.asarray(brr[ib], float)
            if x.shape != y.shape:
                worst = float("inf")
            elif not np.array_equal(x, y):
                worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-300))))
        status["fields/" + k_] = ("BITWISE", "") if worst == 0.0 else (("WITHIN-FLOOR", "max rel %.3e" % worst) if worst <= floor else ("FAIL", "max rel %.3e" % worst))
    order = {"BITWISE": 0, "WITHIN-FLOOR": 1, "FAIL": 2}
    overall = max((v_[0] for v_ in status.values()), key=lambda s: order[s]) if status else "FAIL"
    for k_, (v_, w_) in sorted(status.items()):
        print("%-14s %s %s" % (v_, k_, w_))
    print("compare-runs %s against %s over %d common records after T/rs %s: %s (floor %.1e; wall-clock seconds, timing seconds, kernel_cache and perf_v7 not compared)" % (
        A, B, len(common), after, overall, floor))
    return 0 if overall != "FAIL" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "light", "solves", "writer", "marks", "compare-runs"])
    ap.add_argument("runs", nargs="*", help="compare-runs: A/TAG B/TAG")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--v3", default=None); ap.add_argument("--v7", default=None)
    ap.add_argument("--after", type=float, default=None, help="compare-runs: compare the records after this T/rs (the slice's clock)")
    ap.add_argument("--floor", type=float, default=0.0, help="compare-runs: the WITHIN-FLOOR band on the relative difference (0: bitwise only)")
    a = ap.parse_args()
    if a.what == "compare-runs":
        if len(a.runs) != 2:
            ap.error("compare-runs needs A/TAG B/TAG")
        sys.exit(compare_runs(a.runs[0], a.runs[1], a.after, a.floor))
    import engine_twofield_v7 as V7
    fails = 0
    if a.what in ("all", "light"):
        fails += test_light(V7, a.gpu)
    if a.what in ("all", "solves"):
        fails += test_solves(V7, a.gpu)
    if a.what in ("all", "writer"):
        fails += test_writer(V7, a.gpu)
    if a.what == "marks" or (a.what == "all" and a.v3 and a.v7):
        if not (a.v3 and a.v7):
            ap.error("marks needs --v3 and --v7")
        fails += test_marks(a.v3, a.v7)
    print("selftest: %d failure(s)" % fails)
    sys.exit(fails)


if __name__ == "__main__":
    main()
