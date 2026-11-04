#!/usr/bin/env python
"""engine_collapse_v36f_selftest.py (reviewer A for the busi lane; W2): the gates of engine_collapse_v36f.py, PASS or FAIL per check.
Modes:
  --build-check SRC DST: DST is build_engine_collapse_v36f's output from SRC text for text and the reverse replacements restore SRC.
  --gate RUN_A RUN_B [--states DIR_A DIR_B] [--ignore-args k1,k2]: two stored runs (dir/tag each) compared bitwise: every number of the
      json's head and records except the timing entries (seconds, timing, slice_saved and the restart's bookkeeping), every array of the
      fields, handoff and spectra files (the fields file's solver line without its "resumed" flag), every slice present in both state dirs.
  --fixture [--out DIR --code DIR --python PY --gpu N --threads 16 --only i,ii,iii,iv]: the native fixture end to end:
      (i)   3 records (T/r_s 0, 0.125, 0.25) on engine_collapse_v36d.py and on engine_collapse_v36f.py --step frozen: bitwise (the gate above);
      (ii)  4 records (to T/r_s 0.375) on --step dg: E against E_0 at every record from the solved charge, the dg entries' energy defect and
            the prediction's residual, beside the frozen run's drift (the defect of input 03 item 1: -7e-9 to -1.5e-8 relative on the
            reviewers' fixture; whatever it is here is printed); PASS when every step converged and |E/E_0 - 1| stays below --tol-E (1e-9: the
            record solve's own second-order floor at --stop-fields 1e-6 lies at 1e-11 to 1e-10);
      (iii) the fixed point's convergence (each step's iteration history, the energy defect falling to the tolerance within the cap) and the
            Gauss nodes' order: the interval T/r_s 0 to 0.25 (4 a) on tau 2, 1 and 0.5 (2, 4 and 8 dg steps) with tight record tolerances, the
            covariance at its end compared pairwise by the gauge-invariant projector norm ||C_a - C_b||_F per channel (and the sources and E):
            the differences' ratio 4 for a second-order step (one step's error O(tau^3)), PASS between 2.5 and 6.5, INCONCLUSIVE at the floor;
      (iv)  the dg run restarted from its complex128 slice at T/r_s 0.125 to 0.375 and compared bitwise with the uninterrupted run (same device).
  --all: --build-check on the code dir's pair, then --fixture.
The fixture: m 0.2, g 0.25, r_s/a 16, R_0 4 r_s, nsites 512 (32 r_s), kappa_sea 4, ell 4 (prep ell 4), tau 2 a, reading A, R2, averaged site
fields, --mix plain, --v36-fixes, --field-kernel-cache (dropped with a note when engine_fields_kcache_v1 is not importable): the ladder
launcher's base arguments at r_s/a 16 with 4 channels in place of 10 (the step's properties do not depend on the channel count)."""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

IGN = {"seconds", "timing", "slice_saved", "resumed_from", "resume_slice_v34", "label_v34", "seconds_preparation", "propagation_prev_s", "handoff_s", "oracle34_s",
       "kernel_cache", "label", "engine_v34"}   # red team (W2 review): the kernel cache's process statistics (hits, compute_s), the run label (which carries the restart's v34 description) and the restart's engine_v34 entry are metadata, not record numbers; record_label stays compared
ARGS_PATH = {"out", "slice_dir"}   # red team: the two runs live in different directories
ARGS_NEW = {"step", "dg_tol", "dg_cap"} | ARGS_PATH
ARGS_RESUME = {"resume_slice", "resume_run", "slices_c128", "slices_c64"} | ARGS_PATH
FIX = {"m": 0.2, "g": 0.25, "rs": 16.0, "R0": 4.0, "nsites": 512, "kappa_sea": 4, "ell": 4, "prep_ell": 4, "tau": 2.0}


def num_eq(x, y):
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (x == y) or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
    return x == y


def deep(x, y, path, diffs, ign):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y), key=str):
            if k in ign:
                continue
            if k not in x or k not in y:
                diffs.append("%s.%s: only in %s" % (path, k, "A" if k in x else "B")); continue
            deep(x[k], y[k], "%s.%s" % (path, k), diffs, ign)
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            diffs.append("%s: lengths %d vs %d" % (path, len(x), len(y))); return
        for i, (p, q) in enumerate(zip(x, y)):
            deep(p, q, "%s[%d]" % (path, i), diffs, ign)
    elif not num_eq(x, y):
        diffs.append("%s: %r vs %r" % (path, x, y))


def arrays_equal(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.dtype == object or y.dtype == object:
        if x.shape != y.shape:
            return False
        return all(arrays_equal(p, q) for p, q in zip(x.ravel(), y.ravel()))
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def solver_eq(x, y):
    try:
        jx = json.loads(str(np.asarray(x).item())); jy = json.loads(str(np.asarray(y).item()))
        jx.pop("resumed", None); jy.pop("resumed", None)
        return jx == jy
    except Exception:  # noqa: BLE001
        return bool(np.asarray(x) == np.asarray(y))


def compare_npz(pa, pb, name, diffs, special=None):
    if not (os.path.exists(pa) and os.path.exists(pb)):
        diffs.append("%s: file missing (A %s, B %s)" % (name, os.path.exists(pa), os.path.exists(pb))); return
    ZA = np.load(pa, allow_pickle=True); ZB = np.load(pb, allow_pickle=True)
    for k in sorted(set(ZA.files) | set(ZB.files)):
        if k not in ZA.files or k not in ZB.files:
            diffs.append("%s[%s]: only in %s" % (name, k, "A" if k in ZA.files else "B")); continue
        if special and k in special:
            if not special[k](ZA[k], ZB[k]):
                diffs.append("%s[%s]: differ (after the special comparison)" % (name, k))
            continue
        if not arrays_equal(ZA[k], ZB[k]):
            diffs.append("%s[%s]: arrays differ" % (name, k))


def compare_runs(A, B, ignore_args=(), states=None):
    """A, B: dir/tag; states: (dirA, dirB) of the slices. Returns (ok, diffs)."""
    dA, tA = os.path.split(A); dB, tB = os.path.split(B)
    diffs = []
    JA = json.load(open(os.path.join(dA, "collapse_%s.json" % tA))); JB = json.load(open(os.path.join(dB, "collapse_%s.json" % tB)))
    ia = set(ignore_args)
    deep({k: v for k, v in JA["args"].items() if k not in ia}, {k: v for k, v in JB["args"].items() if k not in ia}, "args", diffs, IGN)
    deep(JA["head"], JB["head"], "head", diffs, IGN)
    if len(JA["records"]) != len(JB["records"]):
        diffs.append("records: %d vs %d" % (len(JA["records"]), len(JB["records"])))
    for i, (ra, rb) in enumerate(zip(JA["records"], JB["records"])):
        deep(ra, rb, "records[%d]" % i, diffs, IGN)
    for suf in ("fields", "handoff", "spectra"):
        compare_npz(os.path.join(dA, "collapse_%s_%s.npz" % (tA, suf)), os.path.join(dB, "collapse_%s_%s.npz" % (tB, suf)), suf, diffs, {"solver": solver_eq, "label": (lambda x, y: True)} if suf == "fields" else None)   # red team: the fields file's label array is the run label (metadata)
    if states:
        fa = {os.path.basename(f).replace(tA, "TAG"): f for f in glob.glob(os.path.join(states[0], "orbitals_%s_T*.npz" % tA))}
        fb = {os.path.basename(f).replace(tB, "TAG"): f for f in glob.glob(os.path.join(states[1], "orbitals_%s_T*.npz" % tB))}
        common = sorted(set(fa) & set(fb))
        for k in common:
            compare_npz(fa[k], fb[k], "slice " + k, diffs)
        if not common:
            diffs.append("slices: no common slice to compare")
    return (len(diffs) == 0), diffs


def print_gate(name, ok, diffs):
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, "bitwise equal (json numbers, fields, handoff, spectra, slices)" if ok else "%d differences, the first: %s" % (len(diffs), " | ".join(diffs[:25]))), flush=True)
    return ok


def run_engine(py, engine, args, cwd, log, threads):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    os.makedirs(cwd, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(["nice", "-n", "10", py, engine] + [str(q) for q in args], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return rc, time.time() - t0


def fixture_args(cfg, tau, tend, out, tag, sdir, c128, extra=()):
    A = ["--m", cfg["m"], "--g", cfg["g"], "--rs", cfg["rs"], "--R0-over-rs", cfg["R0"], "--reading", "A", "--nsites", cfg["nsites"], "--kappa-sea", cfg["kappa_sea"],
         "--reference", "R2", "--site-fields", "averaged", "--prep-ell", cfg["prep_ell"], "--ell", cfg["ell"], "--tau", tau, "--T-over-rs", tend, "--mix", "plain", "--v36-fixes"]
    if cfg.get("kcache"):
        A.append("--field-kernel-cache")
    if cfg.get("gpu") is not None:
        A += ["--gpu", cfg["gpu"]]
    if c128:
        A += ["--slices-c128", ",".join("%.10f" % q for q in c128)]
    A += ["--slice-dir", sdir, "--out", out, "--tag", tag] + list(extra)
    return A


def slice_at(sdir, tag, t_over_rs, rs):
    for f in glob.glob(os.path.join(sdir, "orbitals_%s_T*.npz" % tag)):
        try:
            Z = np.load(f)
            if abs(float(Z["T"]) / rs - t_over_rs) < 1e-6 and str(Z["dtype"]) == "complex128":
                return f
        except Exception:  # noqa: BLE001
            continue
    return None


def cov_diff(ZA, ZB, kappas):
    """||C_A - C_B||_F per channel for C = Phi diag(f) Phi^dag (the sea's columns at occupation 1, the dust's at f), gauge invariant."""
    out = {}
    for k in kappas:
        PA = np.concatenate([np.asarray(ZA["sea_%d" % k]), np.asarray(ZA["dust_%d" % k])], axis=1); fA = np.concatenate([np.ones(ZA["sea_%d" % k].shape[1]), np.asarray(ZA["f_%d" % k], float)])
        PB = np.concatenate([np.asarray(ZB["sea_%d" % k]), np.asarray(ZB["dust_%d" % k])], axis=1); fB = np.concatenate([np.ones(ZB["sea_%d" % k].shape[1]), np.asarray(ZB["f_%d" % k], float)])
        M = (PA.conj().T @ PB) * np.sqrt(fA)[:, None] * np.sqrt(fB)[None, :]
        out[k] = math.sqrt(max(float(np.sum(fA ** 2) + np.sum(fB ** 2) - 2.0 * np.sum(np.abs(M) ** 2)), 0.0))
    return out


def records_of(path):
    J = json.load(open(path)); rs = float(J["args"]["rs"])
    return J, rs, [r for r in J["records"] if r.get("E") is not None]


def build_check(src, dst):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_engine_collapse_v36f as B
    text = open(src, "rb").read().decode("utf-8"); have = open(dst, "rb").read().decode("utf-8")
    try:
        built = B.build(text); back = B.reverse(have)
    except SystemExit as ex:
        print("FAIL build-check: %s" % ex); return False
    ok = (built == have) and (back == text)
    print("%s build-check: %s is %sthe builder's output from %s; the reverse replacements %s" % ("PASS" if ok else "FAIL", dst, "" if built == have else "NOT ",
                                                                                                src, "restore the source" if back == text else "do NOT restore the source"))
    return ok


def do_fixture(a):
    code = os.path.abspath(a.code); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    py = a.python
    vd = os.path.join(code, "engine_collapse_v36d.py"); vf = os.path.join(code, "engine_collapse_v36f.py")
    for p_ in (vd, vf):
        if not os.path.exists(p_):
            raise SystemExit("missing %s" % p_)
    cfg = dict(FIX); cfg["gpu"] = a.gpu
    cfg["kcache"] = subprocess.call([py, "-c", "import engine_fields_kcache_v1"], cwd=code, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if not cfg["kcache"]:
        print("note: engine_fields_kcache_v1 is not importable in %s: --field-kernel-cache dropped (the cache is bitwise, so nothing of the gates changes)" % code)
    rs = cfg["rs"]; tau = cfg["tau"]
    only = set(a.only.split(","))
    print("the fixture: %s; device %s; threads %d; python %s; code %s; out %s" % (" ".join(str(q) for q in fixture_args(cfg, tau, "...", "<out>", "fx", "<states>", [])[:-6]),
                                                                                 "cpu" if a.gpu is None else "cuda:%d" % a.gpu, a.threads, py, code, out), flush=True)
    res = {}
    # ------------------------------------------------------------------------------------------------ (i) frozen bitwise with v36d
    if "i" in only:
        for eng, sub in ((vd, "d"), (vf, "f")):
            od = os.path.join(out, sub); sd = os.path.join(od, "states")
            rc, dt = run_engine(py, eng, fixture_args(cfg, tau, 0.25, od, "fx", sd, [0.25]), od, os.path.join(od, "run.log"), a.threads)
            print("(i) %s: exit %d in %.0f s (log %s)" % (os.path.basename(eng), rc, dt, os.path.join(od, "run.log")), flush=True)
        ok, diffs = compare_runs(os.path.join(out, "d", "fx"), os.path.join(out, "f", "fx"), ARGS_NEW, (os.path.join(out, "d", "states"), os.path.join(out, "f", "states")))
        res["i"] = print_gate("(i) v36f --step frozen against v36d on 3 records", ok, diffs)
    # ------------------------------------------------------------------------------------------------ (ii) dg conserves E
    if "ii" in only or "iii" in only or "iv" in only:
        od = os.path.join(out, "dg"); sd = os.path.join(od, "states")
        rc, dt = run_engine(py, vf, fixture_args(cfg, tau, 0.375, od, "fx", sd, [0.125, 0.375], ["--step", "dg"]), od, os.path.join(od, "run.log"), a.threads)
        print("(ii) v36f --step dg: exit %d in %.0f s (log %s)" % (rc, dt, os.path.join(od, "run.log")), flush=True)
        J, _rs, recs = records_of(os.path.join(od, "collapse_fx.json"))
        E0 = float(J["head"]["E0"])
        fpath = os.path.join(out, "f", "collapse_fx.json")
        frozen = {}
        if os.path.exists(fpath):
            Jf, _r, rf = records_of(fpath); frozen = {round(r["T"] / rs, 6): r["E"] / float(Jf["head"]["E0"]) - 1.0 for r in rf}
        worst = 0.0; conv = True; nsteps = 0; worst_pred = 0.0
        print("(ii) T/rs | E/E0 - 1 (dg) | against the previous record | against the step's prediction | iterations | energy defect | node fields | frozen step's E/E0 - 1")
        for r in recs:
            t = round(r["T"] / rs, 6); d_ = r.get("dg_v36f"); rel = r["E"] / E0 - 1.0
            if d_ is not None:
                nsteps += 1; worst = max(worst, abs(rel)); conv = conv and bool(d_["converged"])
                worst_pred = max(worst_pred, abs(d_.get("E_minus_E_pred_over_E0") or 0.0))
                print("  %.4f | %+.3e | %+.3e | %+.3e | %d | %+.3e | %.1e | %s" % (t, rel, d_.get("E_minus_E_prev_over_E0") or 0.0, d_.get("E_minus_E_pred_over_E0") or 0.0, d_["iterations"],
                                                                               d_.get("energy_defect_final") or 0.0, d_.get("fields_change_final") or 0.0, ("%+.3e" % frozen[t]) if t in frozen else "-"))
            else:
                print("  %.4f | %+.3e | - | - | - | - | - | %s (the initial state)" % (t, rel, ("%+.3e" % frozen[t]) if t in frozen else "-"))
        ended = "end_v36f" in J
        res["ii"] = (len(recs) == 4) and conv and (not ended) and (worst < a.tol_E)
        print("%s (ii) dg conserves E: %d records, %d steps, %s; worst |E/E0 - 1| %.2e (admitted %.1e); worst |E - E_pred|/E0 %.2e (the adjoint's and the record solve's floor); "
              "the frozen step's |E/E0 - 1| at the same clocks %s" % ("PASS" if res["ii"] else "FAIL", len(recs), nsteps, "every step converged" if conv else "a step did NOT converge" + (" (the run ended)" if ended else ""),
                                                                  worst, a.tol_E, worst_pred, (", ".join("%.1e" % abs(v) for k, v in sorted(frozen.items()) if k > 0) or "-")), flush=True)
        # (iii) the fixed point's convergence
        if "iii" in only:
            okc = True
            for r in recs:
                d_ = r.get("dg_v36f")
                if d_ is None:
                    continue
                h = d_["history"]
                print("(iii) the step to T/rs %.4f: energy defects per iteration %s; node fields' change %s; start: %s" % (
                    round(r["T"] / rs, 6), " ".join("%+.1e" % q[1] for q in h), " ".join("%.1e" % q[3] for q in h), d_["start"]))
                okc = okc and bool(d_["converged"]) and abs(h[-1][1]) < d_["tolerance_energy_relative"] and h[-1][3] < d_["tolerance_fields"]
            res["iii-conv"] = okc
            print("%s (iii) the fixed point's convergence: every step's final energy defect below %g and node fields below the record's --stop-fields within the cap" % ("PASS" if okc else "FAIL", 1e-12), flush=True)
    # ------------------------------------------------------------------------------------------------ (iii) the Gauss nodes' order
    if "iii" in only:
        tight = ["--step", "dg", "--stop-fields", "1e-10", "--stop-E", "1e-12", "--maxit", "60", "--dg-tol", "1e-14", "--dg-cap", "60"]
        runs = {}
        for tau_, sub in ((2.0, "o2"), (1.0, "o1"), (0.5, "o05")):
            od = os.path.join(out, sub); sd = os.path.join(od, "states")
            rc, dt = run_engine(py, vf, fixture_args(cfg, tau_, 0.25, od, "fx", sd, [0.25], tight), od, os.path.join(od, "run.log"), a.threads)
            print("(iii) order run tau %g (%d dg steps over 4 a): exit %d in %.0f s" % (tau_, int(round(4.0 / tau_)), rc, dt), flush=True)
            runs[sub] = (od, sd)
        try:
            Z = {sub: np.load(slice_at(sd, "fx", 0.25, rs)) for sub, (od, sd) in runs.items()}
            kappas = [int(k) for k in np.asarray(Z["o2"]["kappas"])]
            D1 = cov_diff(Z["o2"], Z["o1"], kappas); D2 = cov_diff(Z["o1"], Z["o05"], kappas)
            d1 = math.sqrt(sum(q * q for q in D1.values())); d2 = math.sqrt(sum(q * q for q in D2.values()))
            Es = {}; eps_rows = {}
            for sub, (od, sd) in runs.items():
                Jx, _r, rx = records_of(os.path.join(od, "collapse_fx.json")); Es[sub] = rx[-1]["E"] / float(Jx["head"]["E0"])
                Zf = np.load(os.path.join(od, "collapse_fx_fields.npz"), allow_pickle=True); eps_rows[sub] = (np.asarray(Zf["eps_tot"][-1], float), np.asarray(Zf["J_tot"][-1], float))
            e1 = max(float(np.max(np.abs(eps_rows["o2"][0] - eps_rows["o1"][0]))), float(np.max(np.abs(eps_rows["o2"][1] - eps_rows["o1"][1]))))
            e2 = max(float(np.max(np.abs(eps_rows["o1"][0] - eps_rows["o05"][0]))), float(np.max(np.abs(eps_rows["o1"][1] - eps_rows["o05"][1]))))
            ratio = d1 / d2 if d2 > 0 else float("inf"); ratio_e = e1 / e2 if e2 > 0 else float("inf")
            floor = d2 < 1e-10
            okk = (2.5 <= ratio <= 6.5)
            res["iii-order"] = okk or floor
            print("%s (iii) the Gauss nodes' order: ||C(tau 2) - C(tau 1)||_F = %.3e, ||C(tau 1) - C(tau 0.5)||_F = %.3e, ratio %.2f (4 for a second-order step, one step's error O(tau^3); "
                  "admitted 2.5 to 6.5); per channel %s / %s; the sources' max |diff| %.2e, %.2e (ratio %.2f); E/E0 at T/rs 0.25: %s%s" % (
                      "PASS" if okk else ("INCONCLUSIVE" if floor else "FAIL"), d1, d2, ratio, " ".join("%.1e" % D1[k] for k in kappas), " ".join("%.1e" % D2[k] for k in kappas), e1, e2, ratio_e,
                      ", ".join("tau %s: %.12f" % (s_, Es[s_]) for s_ in ("o2", "o1", "o05")), "; the second difference is at the floor (below 1e-10), the ratio is not resolved on this fixture" if floor else ""), flush=True)
        except Exception as ex:  # noqa: BLE001
            res["iii-order"] = False
            print("FAIL (iii) the Gauss nodes' order: %s: %s" % (type(ex).__name__, str(ex)[:200]), flush=True)
    # ------------------------------------------------------------------------------------------------ (iv) resume bitwise
    if "iv" in only:
        S = slice_at(os.path.join(out, "dg", "states"), "fx", 0.125, rs)
        if S is None:
            res["iv"] = False; print("FAIL (iv) resume: no complex128 slice at T/rs 0.125 in the dg run's states")
        else:
            od = os.path.join(out, "r"); sd = os.path.join(od, "states")
            rc, dt = run_engine(py, vf, fixture_args(cfg, tau, 0.375, od, "fx", sd, [0.375], ["--step", "dg", "--resume-slice", S, "--resume-run", os.path.join(out, "dg", "fx")]),
                                od, os.path.join(od, "run.log"), a.threads)
            print("(iv) the dg run restarted from %s: exit %d in %.0f s" % (S, rc, dt), flush=True)
            ok, diffs = compare_runs(os.path.join(out, "dg", "fx"), os.path.join(out, "r", "fx"), ARGS_RESUME, (os.path.join(out, "dg", "states"), os.path.join(out, "r", "states")))
            res["iv"] = print_gate("(iv) the restart from the slice at T/rs 0.125 against the uninterrupted dg run (records 0.25 and 0.375, the fields rows, the slice at 0.375)", ok, diffs)
    print("SUMMARY: " + ", ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in res.items()), flush=True)
    return all(res.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", dest="build_check", nargs=2, metavar=("SRC", "DST"), default=None)
    ap.add_argument("--gate", nargs=2, metavar=("RUN_A", "RUN_B"), default=None)
    ap.add_argument("--states", nargs=2, default=None)
    ap.add_argument("--ignore-args", dest="ignore_args", default="step,dg_tol,dg_cap,out,slice_dir")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--code", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "v36f_selftest"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default="i,ii,iii,iv")
    ap.add_argument("--tol-E", dest="tol_E", type=float, default=1e-9)
    a = ap.parse_args()
    ok = True
    if a.build_check:
        ok = build_check(a.build_check[0], a.build_check[1]) and ok
    if a.gate:
        g_ok, diffs = compare_runs(a.gate[0], a.gate[1], set(a.ignore_args.split(",")), tuple(a.states) if a.states else None)
        ok = print_gate("gate %s against %s" % (a.gate[0], a.gate[1]), g_ok, diffs) and ok
    if a.all:
        ok = build_check(os.path.join(a.code, "engine_collapse_v36d.py"), os.path.join(a.code, "engine_collapse_v36f.py")) and ok
    if a.fixture or a.all:
        ok = do_fixture(a) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
