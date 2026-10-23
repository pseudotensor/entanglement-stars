#!/usr/bin/env python
"""rd13_fixture_gate_v1a.py (busi, analyst, 21 Oct): rd13_fixture_gate_v1.py under erratum v1a of the tolerance ruling (ReviewA_rd1gate_v1.md
sec. 3 (i)): the "no rd13 line in the defaults' log" check skips the command line (line 1 of each log, written by run(), which names the
engine file). Everything else as v1. Added --reread: no engine is run; the checks are recomputed from the run dirs already under --out.
v1 docstring follows.
rd13_fixture_gate_v1.py (busi, analyst; PERF RD-1 + RD-3 self-test part (a) on the fixture, plus an engine-level smoke of the options):
engine_collapse_v36nq_rd13.py against its host engine_collapse_v36nq.py on the v36n self-test's fixture (engine_collapse_v36n_selftest.py:
m 0.2, r_s 16, 512 sites, kappa_sea 10, tau 2, T/r_s 0 to 0.25; its base_cmd and compare_runs; the T = 0 complex128 slice), in the form of
h4_slice_async_gate_v1.py.
CPU pairs (host H against the build at its defaults A; bitwise = compare_runs: the records with the walls removed, the head, the fields,
spectra and handoff npz, the T = 0 slice, byte for byte; the args differ by eigh and jvp only):
  D   --iter-cache drop
  L   --iter-cache drop --prep-memory lean
  UD  --mix damped --mix-alpha 3 --maxit 3 --stop-fields 1e-14 --iter-cache drop
  N   the ISO_128 line's options (--dust-set isotropic --iter-cache drop --prep-memory lean, --mix newton with the NK flags, --nk-set resolve,
      --nk-plain-fallback on)
GPU part (--gpu G; the engine's --gpu 0 under CUDA_VISIBLE_DEVICES as the caller sets it): N on the GPU with the defaults (GN) and with
  --eigh banded --jvp sternheimer (GB), the smoke of the options through a whole fresh run (the preparation's non-particle-hole well
  hamiltonian included): exit codes, the rd13 lines, and the differences GB - GN read (records' E/E0, min(N - |v|), the fields, the masks of
  the spectra npz, the set decisions); READ lines, not gates (the gates of the options are parts (b) and (c)).
    python rd13_fixture_gate_v1.py --code DIR --out DIR [--threads 8] [--only D,L,UD,N,G]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

NEWTON = ["--dust-set", "isotropic", "--iter-cache", "drop", "--prep-memory", "lean", "--mix", "newton", "--nk-krylov", "40", "--nk-halvings", "6",
          "--nk-inner", "15", "--nk-jvp", "exact", "--nk-line-search", "halving", "--nk-set", "resolve", "--nk-maxiter", "30", "--nk-restarts", "2",
          "--nk-plain-fallback", "on"]
PAIRS = {"D": ["--iter-cache", "drop"], "L": ["--iter-cache", "drop", "--prep-memory", "lean"],
         "UD": ["--mix", "damped", "--mix-alpha", "3", "--maxit", "3", "--stop-fields", "1e-14", "--iter-cache", "drop"], "N": NEWTON}


def load_selftest(code):
    spec = importlib.util.spec_from_file_location("st36n", os.path.join(code, "engine_collapse_v36n_selftest.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(st, a, eng, name, extra, gpu=None):
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
    tag = "fx_" + name; sdir = os.path.join(out, "states")
    sa = argparse.Namespace(no_kcache=False, gpu=gpu)
    cmd = st.base_cmd(sa, os.path.join(a.code, eng), out, tag, sdir) + extra
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = os.path.join(out, "log.txt"); t0 = time.time()
    if a.reread:
        rc = 0 if os.path.exists(os.path.join(out, "collapse_%s.json" % tag)) else 1
    else:
        with open(log, "w") as fh:
            fh.write(" ".join(cmd) + "\n"); fh.flush()
            rc = subprocess.call(cmd, cwd=a.code, stdout=fh, stderr=subprocess.STDOUT, env=env)
    r = {"name": name, "rc": rc, "seconds": time.time() - t0, "log": log, "json": os.path.join(out, "collapse_%s.json" % tag),
         "fields": os.path.join(out, "collapse_%s_fields.npz" % tag), "spectra": os.path.join(out, "collapse_%s_spectra.npz" % tag),
         "handoff": os.path.join(out, "collapse_%s_handoff.npz" % tag), "slice": os.path.join(sdir, "orbitals_%s_T0000.00.npz" % tag)}
    print("  %s: exit %d in %.0f s" % (name, rc, r["seconds"]), flush=True)
    return r


def compare(st, H, A, label, checks):
    d = st.compare_runs(H, A)
    checks.append((label + " compare_runs (records but the walls, head, fields, spectra, handoff, T = 0 slice): byte for byte", not d, "; ".join(d) or "identical"))
    ja, jh = json.load(open(A["json"]))["args"], json.load(open(H["json"]))["args"]
    dk = sorted(k for k in set(ja) | set(jh) if ja.get(k) != jh.get(k) and k not in ("out", "tag", "slice_dir"))
    checks.append((label + " args differ by eigh and jvp only", dk == ["eigh", "jvp"], str(dk)))
    # v1a: line 1 is the command line run() writes (it names engine_collapse_v36nq_rd13.py); it is skipped
    n_rd = sum(1 for i_, l_ in enumerate(open(A["log"], errors="replace")) if i_ > 0 and "rd13" in l_)
    checks.append((label + " no rd13 line in the defaults' log (command line skipped)", n_rd == 0, "%d lines" % n_rd))


def read_diff(GN, GB, checks):
    JA, JB = json.load(open(GN["json"])), json.load(open(GB["json"]))
    ra, rb = JA["records"], JB["records"]
    lines = ["records %d / %d" % (len(ra), len(rb))]
    for p, q in zip(ra, rb):
        if p.get("T") is None:
            continue
        de = abs((p.get("E_over_E0") or 0) - (q.get("E_over_E0") or 0)); dm = abs((p.get("min_N_minus_absv") or 0) - (q.get("min_N_minus_absv") or 0))
        sets = []
        for k_ in ("n_pocket", "n_refill_R0", "inversions_total", "mask_equals_negative_set", "converged", "iterations", "eigendecompositions"):
            if p.get(k_) != q.get(k_):
                sets.append(k_)
        cp, cq = p.get("continuation") or {}, q.get("continuation") or {}
        for kk in sorted(set(cp) | set(cq)):
            a_, b_ = cp.get(kk) or {}, cq.get(kk) or {}
            for f_ in ("mode", "nsub", "verified", "inversions", "n_intermediate", "n_pocket", "mask_equals_negative_set"):
                if a_.get(f_) != b_.get(f_):
                    sets.append("cont[%s].%s" % (kk, f_))
        lines.append("T/rs %.4f dE/E0 %.1e dmin(N-|v|) %.1e differing labels %s; rd13 %s" % (p["T_over_rs"], de, dm, sets or "none",
                     {k_: q.get("rd13", {}).get(k_) for k_ in ("calls", "banded", "banded_all", "dense_fallback", "fallback_why", "st_channels", "st_fallback", "st_fallback_why")}))
    ZA, ZB = np.load(GN["fields"]), np.load(GB["fields"])
    for k_ in ("N", "v", "N_applied", "v_applied"):
        if k_ in ZA.files and k_ in ZB.files and ZA[k_].shape == ZB[k_].shape:
            lines.append("fields %s max |d| %.1e" % (k_, float(np.max(np.abs(ZA[k_] - ZB[k_])))))
    SA, SB = np.load(GN["spectra"]), np.load(GB["spectra"])
    mk = [k_ for k_ in SA.files if "mask" in k_ or k_.startswith("occ")]
    neq = [k_ for k_ in mk if k_ in SB.files and not np.array_equal(SA[k_], SB[k_])]
    lines.append("spectra npz mask-like arrays %d, differing %s" % (len(mk), neq or "none"))
    lines += [l_.rstrip() for i_, l_ in enumerate(open(GB["log"], errors="replace")) if i_ > 0 and "rd13" in l_][:12]
    checks.append(("G READ GB (banded + sternheimer) against GN (defaults), both on the GPU", None, " || ".join(lines)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--only", default="D,L,UD,N,G")
    ap.add_argument("--reread", action="store_true", help="v1a: run nothing, recompute the checks from the run dirs under --out")
    a = ap.parse_args(); a.code = os.path.abspath(a.code); os.makedirs(a.out, exist_ok=True)
    want = a.only.split(",")
    st = load_selftest(a.code)
    checks = []
    for nm in ("D", "L", "UD", "N"):
        if nm not in want:
            continue
        print("pair %s (CPU)" % nm, flush=True)
        H = run(st, a, "engine_collapse_v36nq.py", "H" + nm, PAIRS[nm])
        A0 = run(st, a, "engine_collapse_v36nq_rd13.py", "A" + nm, PAIRS[nm])
        checks.append(("%s runs exit 0" % nm, H["rc"] == 0 and A0["rc"] == 0, ""))
        if H["rc"] == 0 and A0["rc"] == 0:
            compare(st, H, A0, "%s defaults:" % nm, checks)
    if "G" in want:
        print("GPU smoke (N options)", flush=True)
        GH = run(st, a, "engine_collapse_v36nq.py", "GH", NEWTON, gpu=0)
        GN = run(st, a, "engine_collapse_v36nq_rd13.py", "GN", NEWTON, gpu=0)
        GB = run(st, a, "engine_collapse_v36nq_rd13.py", "GB", NEWTON + ["--eigh", "banded", "--jvp", "sternheimer"], gpu=0)
        checks.append(("G runs exit 0", GH["rc"] == 0 and GN["rc"] == 0 and GB["rc"] == 0, "%d %d %d" % (GH["rc"], GN["rc"], GB["rc"])))
        if GH["rc"] == 0 and GN["rc"] == 0:
            compare(st, GH, GN, "G (GPU) defaults:", checks)
        if GN["rc"] == 0 and GB["rc"] == 0:
            read_diff(GN, GB, checks)
    ok = all(c[1] is not False for c in checks)
    for n_, o_, t_ in checks:
        print("%-5s %s%s" % ("READ" if o_ is None else ("PASS" if o_ else "FAIL"), n_, (": " + t_) if t_ else ""), flush=True)
    print("rd13_fixture_gate_v1a: %s (%d checks)" % ("PASS" if ok else "FAIL", len(checks)), flush=True)
    json.dump({"pass": ok, "checks": checks}, open(os.path.join(a.out, "gate_rd13_fixture_v1a.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
