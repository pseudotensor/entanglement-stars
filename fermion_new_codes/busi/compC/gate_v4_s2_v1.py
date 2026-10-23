#!/usr/bin/env python
"""gate_v4_s2_v1.py (the busi lane, Computation C): the gates of engine_twofield_v4's --light-start-continuation (C_s2_start_judgment_v1.md, section 2).
G1 bitwise off: engine_twofield_v3 against engine_twofield_v4 with the option off, the pathfinder's arguments at sigma/a 4 (--ell 4, coupling on), the first
   --records records; every numeric leaf of the record json (timing, seconds, kernel-cache counters, args and string labels excluded) and every array of the
   fields, spectra, handoff and s11 npz bitwise.
G2 the uncoupled limit: v4 --light-coupling off --light-start-continuation coupling against v3 --light-coupling off at sigma/a 2 (--ell 2): bitwise; that run's
   T = 0 verdict is printed (the uncoupled quench at the pathfinder's chain).
G3 on: v4 --light-coupling on --light-start-continuation coupling at sigma/a 2: every record meets the stored stop rule; the stages, stage C's path, the
   displacements, the solver's residuals (beside --prod-json's where given) and E at T = 0 beside the production's record 0 are printed.
Exit 0 on PASS. Usage (the pod): cd <internal>/compC && python gate_v4_s2_v1.py --engine-dir <internal>/compC --out <internal>/gateV4 --gpu 0
[--records 3] [--nsites 2048] [--prod-json <internal>/member2_snapshot_2100.json] [--only g1,g2,g3] [--reuse] [--no-v36-opts]."""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

SKIP = {"seconds", "timing", "kernel_cache", "args", "script"}
NPZ = (("json", "collapse_%s.json"), ("fields", "collapse_%s_fields.npz"), ("spectra", "collapse_%s_spectra.npz"), ("handoff", "collapse_%s_handoff.npz"), ("s11", "collapse_%s_s11.npz"))


def leaves(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP or k.endswith("_s"):
                continue
            leaves(v, path + (k,), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            leaves(v, path + (str(i),), out)
    elif isinstance(o, str):
        return
    else:
        out[".".join(path)] = o


def same(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (math.isnan(x) and math.isnan(y))
    return type(x) is type(y) and x == y


def cmp_json(pa, pb):
    la, lb = {}, {}
    leaves(json.load(open(pa)), (), la); leaves(json.load(open(pb)), (), lb)
    for k in sorted(set(la) | set(lb)):
        if k not in la or k not in lb:
            return False, "json leaf %s present in %s only" % (k, "the first" if k in la else "the second"), len(la)
        if not same(la[k], lb[k]):
            return False, "json leaf %s: %r against %r" % (k, la[k], lb[k]), len(la)
    return True, None, len(la)


def arr_equal(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def cmp_npz(pa, pb):
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    if sorted(A.files) != sorted(B.files):
        return False, "npz keys differ: %s" % sorted(set(A.files) ^ set(B.files)), 0
    n = 0
    for k in sorted(A.files):
        x, y = A[k], B[k]
        if x.dtype.kind in "US" or y.dtype.kind in "US":
            continue
        if x.dtype == object or y.dtype == object:
            if x.shape != y.shape:
                return False, "array %s: shape %s against %s" % (k, x.shape, y.shape), n
            for i in range(x.size):
                xi, yi = x.flat[i], y.flat[i]
                if np.asarray(xi).dtype.kind in "US":
                    continue
                if not arr_equal(xi, yi):
                    return False, "array %s[%d] differs" % (k, i), n
        elif not arr_equal(x, y):
            return False, "array %s differs" % k, n
        n += 1
    return True, None, n


def bitwise(a, tag_a, tag_b, name):
    ok_all = True; msgs = []
    for kind, pat in NPZ:
        pa = os.path.join(a.out, pat % tag_a); pb = os.path.join(a.out, pat % tag_b)
        if not os.path.exists(pa) and not os.path.exists(pb):
            continue
        if not (os.path.exists(pa) and os.path.exists(pb)):
            ok_all = False; msgs.append("%s: present in 1 run only" % kind); continue
        ok, why, n = (cmp_json if kind == "json" else cmp_npz)(pa, pb)
        msgs.append("%s %s (%d %s)%s" % (kind, "bitwise" if ok else "DIFFERS", n, "leaves" if kind == "json" else "arrays", "" if ok else ": " + why))
        ok_all = ok_all and ok
    print("%s: %s; %s" % (name, "PASS bitwise" if ok_all else "FAIL", "; ".join(msgs)), flush=True)
    return ok_all


def engine_args(a):
    T = (a.records - 1) * a.tau / a.rs
    args = ["--m", "0.2", "--g", "0.25", "--rs", "%g" % a.rs, "--R0-over-rs", "4", "--reading", "A", "--nsites", str(a.nsites), "--prep-ell", "4", "--kappa-sea", "10",
            "--tau", "%g" % a.tau, "--T-over-rs", "%.12g" % T, "--reference", "R2", "--half-offset", "0", "--collar-policy", "flag", "--support-rel", "1e-14",
            "--site-fields", "averaged", "--stop-E", "1e-7", "--stop-fields", "1e-6", "--maxit", "20", "--mix", "newton", "--nk-maxiter", "30", "--nk-krylov", "40",
            "--nk-restarts", "0", "--nk-line-search", "halving", "--nk-halvings", "6", "--source-continuation", "two-stage", "--end-above", "1e-3", "--nk-jvp", "exact",
            "--delta-rtol-every", "10", "--commutator-every", "1", "--streams", "0", "--m1-simpson-steps", "", "--c-LZ", "1.0", "--window-max", "4",
            "--light-kappas", "1,2", "--light-guard-kappas", "3", "--light-transport", "banded", "--light-d-min", "1e-7", "--s11-collars", "8,16,24", "--s11-stop", "1",
            "--slices-c128", "", "--slices-light-c128", ""]
    if a.v36_opts:
        args += ["--field-kernel-cache", "--v36-fixes"]
    return args


def run_engine(a, engine, ell, coupling, tag, extra):
    out_json = os.path.join(a.out, "collapse_%s.json" % tag)
    if a.reuse and os.path.exists(out_json):
        print("  reusing %s" % out_json, flush=True)
        return out_json
    cmd = [sys.executable, engine] + engine_args(a) + ["--ell", str(ell), "--light-coupling", coupling, "--out", a.out, "--tag", tag] + extra
    if a.gpu is not None:
        cmd += ["--gpu", str(a.gpu)]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "16"); env.setdefault("MKL_NUM_THREADS", "16"); env.setdefault("OPENBLAS_NUM_THREADS", "16")
    env.setdefault("BUSI_ENGINE_DIR", a.engine_dir); env.setdefault("BUSI_PAGE_DIR", a.engine_dir)
    log = os.path.join(a.out, "%s.log" % tag)
    t0 = time.time()
    print("running %s > %s" % (" ".join(cmd), log), flush=True)
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=a.engine_dir, env=env)
    print("  exit %d in %.0f s" % (rc, time.time() - t0), flush=True)
    if rc != 0 or not os.path.exists(out_json):
        raise SystemExit("gate: the engine run %s failed (exit %d); see %s" % (tag, rc, log))
    return out_json


def ensure_built(a, v4):
    if "def run(a):" in open(v4).read():
        return v4
    print("building the engine from the replacement list %s" % v4, flush=True)
    subprocess.check_call([sys.executable, v4, "--src", os.path.join(a.engine_dir, a.v3), "--out", v4], cwd=a.engine_dir)
    return v4


def rows(path):
    J = json.load(open(path)); out = []
    for r in J["records"]:
        s32 = r.get("solve_v32") or {}; sol = r.get("solver") or {}; L = r.get("light_v36") or {}
        out.append({"T": r.get("T_over_rs"), "E": r.get("E"), "E_over_E0": r.get("E_over_E0"), "converged": r.get("converged"), "meets": s32.get("meets_stop_rule"),
                    "kept_from": s32.get("kept_from"), "kept": s32.get("kept_residual"), "confirming": s32.get("confirming_residual"), "dE_c": s32.get("confirming_dE"),
                    "continuation": s32.get("continuation"), "solves": s32.get("solves"), "v4": s32.get("light_start_continuation_v4"), "W": sol.get("residual_W"),
                    "invN": sol.get("residual_invN"), "Einv": L.get("E_inv_light_over_E"), "failed": r.get("failed"), "ends": s32.get("ends_member")})
    return J, out


def f(x, fm="%.3e"):
    return (fm % x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def print_rows(name, rs, prod=None):
    for k, r in enumerate(rs):
        p = prod[k] if (prod is not None and k < len(prod)) else None
        print("  %s T/rs %s: converged %s meets stop rule %s; kept from %s residual %s confirming %s dE/E %s; E %s (E/E0 %s); solver residuals W %s 1/N %s; light E_inv/E %s; %s%s" % (
            name, f(r["T"], "%.4f"), r["converged"], r["meets"], r["kept_from"], f(r["kept"], "%.2e"), f(r["confirming"], "%.2e"), f(r["dE_c"], "%.1e"), f(r["E"], "%.9f"),
            f(r["E_over_E0"], "%.7f"), f(r["W"], "%.1e"), f(r["invN"], "%.1e"), f(r["Einv"]), r["continuation"],
            ("; production: E %s converged %s kept %s W %s 1/N %s" % (f(p["E"], "%.9f"), p["converged"], f(p["kept"], "%.2e"), f(p["W"], "%.1e"), f(p["invN"], "%.1e"))) if p else ""), flush=True)
        if r["failed"]:
            print("    FAILED: %s" % r["failed"], flush=True)
        for q in (r["solves"] or []):
            print("    stage: %s" % json.dumps(q, default=str)[:700], flush=True)
        if r["v4"]:
            print("    v4: %s" % json.dumps(r["v4"], default=str)[:1400], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--records", type=int, default=3); ap.add_argument("--nsites", type=int, default=2048); ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--v3", default="engine_twofield_v3.py"); ap.add_argument("--v4", default="engine_twofield_v4.py")
    ap.add_argument("--no-v36-opts", dest="v36_opts", action="store_false", help="run without --field-kernel-cache --v36-fixes")
    ap.add_argument("--prod-json", default=None, help="the production's member2 json (records' E and solver residuals printed beside; the chains differ unless --nsites 1024)")
    ap.add_argument("--only", default="g1,g2,g3"); ap.add_argument("--reuse", action="store_true", help="reuse engine outputs already in --out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v3 = os.path.join(a.engine_dir, a.v3); v4 = ensure_built(a, os.path.join(a.engine_dir, a.v4))
    only = set(x.strip() for x in a.only.split(",") if x.strip())
    prod = rows(a.prod_json)[1] if a.prod_json else None
    print("gate v4 (sigma/a 2 start): engines %s and %s; %d sites, tau %g, rs %g, the first %d records (T/rs to %.4f); v36 options %s; device %s" % (
        v3, v4, a.nsites, a.tau, a.rs, a.records, (a.records - 1) * a.tau / a.rs, "on" if a.v36_opts else "off", "gpu %d" % a.gpu if a.gpu is not None else "cpu"), flush=True)
    verdict = {}
    if "g1" in only:
        run_engine(a, v3, 4, "on", "g1_s4_v3", [])
        run_engine(a, v4, 4, "on", "g1_s4_v4_off", [])
        verdict["G1 bitwise off (sigma/a 4, coupling on)"] = bitwise(a, "g1_s4_v3", "g1_s4_v4_off", "G1")
    if "g2" in only:
        j3 = run_engine(a, v3, 2, "off", "g2_s2_v3_off", [])
        j4 = run_engine(a, v4, 2, "off", "g2_s2_v4_off_option", ["--light-start-continuation", "coupling"])
        verdict["G2 the uncoupled limit (sigma/a 2, coupling off, option on)"] = bitwise(a, "g2_s2_v3_off", "g2_s2_v4_off_option", "G2")
        r4 = rows(j4)[1]
        print("G2 the uncoupled quench at %d sites (the chain-length question): T = 0 %s" % (a.nsites, "CONVERGED" if (r4 and r4[0]["meets"]) else "NOT CONVERGED"), flush=True)
        print_rows("G2 v4 off", r4, prod)
    if "g3" in only:
        j = run_engine(a, v4, 2, "on", "g3_s2_v4_on_option", ["--light-start-continuation", "coupling"])
        r = rows(j)[1]
        ok = len(r) == a.records and all(q["meets"] is True and not q["failed"] for q in r)
        verdict["G3 on (sigma/a 2, coupling on, option on): every record meets the stop rule"] = ok
        print("G3: %s (%d of %d records meet the stored stop rule)" % ("PASS" if ok else "FAIL", sum(1 for q in r if q["meets"] is True), a.records), flush=True)
        print_rows("G3 v4 on", r, prod)
        if prod and r:
            d = abs(r[0]["E"] - prod[0]["E"]) / 59.788988
            print("G3 E at T = 0 against the production's record 0: |dE/E0| %.2e (the floor 3.0e-7; informational, the chains differ unless --nsites 1024)" % d, flush=True)
        if r and r[0]["v4"]:
            v = r[0]["v4"]; sB = v.get("stage_B") or {}
            print("G3 reading: stage A %s (kept %s), stage B %s (kept %s), plain closure %s" % (
                "converged" if (v.get("stage_A") or {}).get("converged") else "NOT converged", f((v.get("stage_A") or {}).get("kept_residual"), "%.2e"),
                "converged" if sB.get("converged") else "NOT converged", f(sB.get("kept_residual"), "%.2e"), json.dumps(sB.get("plain_closure"), default=str)), flush=True)
    for k_, v_ in verdict.items():
        print("%s: %s" % (k_, "PASS" if v_ else "FAIL"), flush=True)
    allok = all(verdict.values()) if verdict else False
    print("GATE V4: %s" % ("PASS" if allok else "FAIL"), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
