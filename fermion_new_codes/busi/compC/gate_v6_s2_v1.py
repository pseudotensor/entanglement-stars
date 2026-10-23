#!/usr/bin/env python
"""gate_v6_s2_v1.py (the busi lane, Computation C): the gates of engine_twofield_v6 (C_s2_start_judgment_v3.md, section 4), in gate_v5_s2_v1's form.
G1 bitwise off: engine_twofield_v4 (its option off) against engine_twofield_v6 (the ramp off) at sigma/a 4 (--ell 4, coupling on), the first --records records;
   every numeric leaf of the record json (timing, seconds, kernel-cache counters, args and string labels excluded) and every array of the fields, spectra,
   handoff and s11 npz bitwise.
G2 the uncoupled limit: v6 --light-coupling off with the ramp, gain and floor options on against v4 --light-coupling off at sigma/a 2: bitwise; T = 0's verdict printed.
G3 on: v6 --light-coupling on --light-coupling-ramp RAMP --light-ramp-bisect B --light-ramp-gain-vectors K --light-ramp-floor-factor F at sigma/a 2: as gate v5's
   (every record after T = 0 meets the stored stop rule; T = 0 meets it or is written below --end-above, the production's practice; no record ends the member).
   Printed per record: the stages, the homotopy's entry (the family, per stage the parameter, the advance rule, kept and confirming residuals, the soft gain g
   with its site, the displacement, the kept fields' change, the central cell, the device memory), the direct stage's soft gain, the fold readout where the walk
   stalls (g and the cell along the walked stages), the solver's residuals beside --prod-json's, E at T = 0 beside the production's record 0.
Every run's peak device memory (the engine's "v6 memory" lines) is printed; an engine run that exits nonzero is reported with its log's tail, not raised.
Exit 0 on PASS. Usage (the pod): cd <internal>/compC && python gate_v6_s2_v1.py --engine-dir <internal>/compC --out <internal>/gateV6 --gpu 0
[--records 3] [--nsites 2048] [--ramp 0.25,0.5,0.75,1] [--bisect 2] [--gain 24] [--floor 2] [--base-above X] [--prod-json ...] [--only g3] [--reuse] [--no-v36-opts].
About 2.5 hours at 2048 sites (G3 1.5 to 2 of them); --only g3 first for the reading."""
import argparse
import json
import math
import os
import re
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


def tail(log, n=25):
    try:
        with open(log, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    return lines[-n:]


def peak_memory(log):
    """the run's peak device memory (GiB) from the engine's 'v6 memory' lines; None without them (a version 4 engine, or a crash before the first record)."""
    pk = None
    if not os.path.exists(log):
        return None
    with open(log, errors="replace") as fh:
        for ln in fh:
            if "v6 memory" not in ln:
                continue
            for m in re.finditer(r"peak(?: device memory allocated)? ([0-9.]+) GiB", ln):
                v = float(m.group(1)); pk = v if pk is None else max(pk, v)
    return pk


def report_memory(tag, log, mem):
    pk = peak_memory(log)
    mem[tag] = pk
    print("  peak device memory (%s): %s" % (tag, ("%.2f GiB" % pk) if pk is not None else "not measured (no v6 memory lines: a version 4 engine, or no record completed)"), flush=True)


def run_engine(a, engine, ell, coupling, tag, extra, mem):
    """runs the engine; returns (the record json's path or None, the exit code). A failure is reported with the log's tail and its memory lines, not raised."""
    out_json = os.path.join(a.out, "collapse_%s.json" % tag)
    log = os.path.join(a.out, "%s.log" % tag)
    if a.reuse and os.path.exists(out_json):
        print("  reusing %s" % out_json, flush=True)
        report_memory(tag, log, mem)
        return out_json, 0
    cmd = [sys.executable, engine] + engine_args(a) + ["--ell", str(ell), "--light-coupling", coupling, "--out", a.out, "--tag", tag] + extra
    if a.gpu is not None:
        cmd += ["--gpu", str(a.gpu)]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "16"); env.setdefault("MKL_NUM_THREADS", "16"); env.setdefault("OPENBLAS_NUM_THREADS", "16")
    env.setdefault("BUSI_ENGINE_DIR", a.engine_dir); env.setdefault("BUSI_PAGE_DIR", a.engine_dir)
    t0 = time.time()
    print("running %s > %s" % (" ".join(cmd), log), flush=True)
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=a.engine_dir, env=env)
    print("  exit %d in %.0f s" % (rc, time.time() - t0), flush=True)
    report_memory(tag, log, mem)
    if rc != 0 or not os.path.exists(out_json):
        print("  the engine run %s FAILED (exit %d)%s; the log's tail:" % (tag, rc, "" if os.path.exists(out_json) else ", no record json"), flush=True)
        for ln in tail(log, 25):
            print("    | %s" % ln.rstrip(), flush=True)
    return (out_json if os.path.exists(out_json) else None), rc


def ensure_built(a, path, src):
    """a replacement list (no engine body) is built over itself from src by its own builder; a built engine is returned as is."""
    if "def run(a):" in open(path).read():
        return path
    print("building %s from its replacement list over %s" % (path, src), flush=True)
    subprocess.check_call([sys.executable, path, "--src", src, "--out", path], cwd=a.engine_dir)
    if "def run(a):" not in open(path).read():
        raise SystemExit("gate: %s did not build" % path)
    return path


def rows(path):
    J = json.load(open(path)); out = []
    for r in J["records"]:
        s32 = r.get("solve_v32") or {}; sol = r.get("solver") or {}; L = r.get("light_v36") or {}
        out.append({"T": r.get("T_over_rs"), "E": r.get("E"), "E_over_E0": r.get("E_over_E0"), "converged": r.get("converged"), "meets": s32.get("meets_stop_rule"),
                    "kept_from": s32.get("kept_from"), "kept": s32.get("kept_residual"), "confirming": s32.get("confirming_residual"), "dE_c": s32.get("confirming_dE"),
                    "continuation": s32.get("continuation"), "solves": s32.get("solves"), "v4": s32.get("light_start_continuation_v4"),
                    "ramp": s32.get("homotopy_ramp_v6") or s32.get("light_coupling_ramp_v5"), "gain_direct": s32.get("soft_gain_direct_v6"),
                    "rule": s32.get("first_record_rule_v5"), "W": sol.get("residual_W"), "invN": sol.get("residual_invN"), "Einv": L.get("E_inv_light_over_E"),
                    "failed": r.get("failed"), "ends": s32.get("ends_member"), "unconverged": r.get("unconverged")})
    return J, out


def f(x, fm="%.3e"):
    return (fm % x) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def print_ramp(v):
    if not v:
        return
    b = v.get("base") or {}; pn = v.get("parameter") or "s"
    print("    v6 homotopy (%s): targets %s bisect %s; base rule %s, floor factor %s, gain vectors %s; base %s = 0 %s kept %s; walked %s; %d stage(s), %d bisection(s); reached %s = 1 %s; stalled %s; closing %s; final kept %s converged %s advanced by %s (%s s)" % (
        v.get("family"), v.get("targets"), v.get("bisect_max"), f(v.get("base_above"), "%g"), f(v.get("floor_factor"), "%g"), v.get("gain_vectors"), pn,
        "converged" if b.get("converged") else "NOT converged", f(b.get("kept_residual"), "%.2e"), v.get("walked"), len(v.get("stages") or []), v.get("bisections"), pn, v.get("reached_1"),
        json.dumps(v.get("stalled")), json.dumps(v.get("closing_stage"), default=str), f(v.get("final_kept_residual"), "%.2e"), v.get("final_converged"), v.get("final_advanced_by"),
        f(v.get("seconds"), "%.0f")), flush=True)
    entries = ([b] if b else []) + list(v.get("stages") or [])
    for st_ in entries:
        cc = st_.get("central_cell") or {}; ch = st_.get("change_from_start") or {}; dp = st_.get("displacement") or {}; sg = st_.get("soft_gain") or {}; mm = st_.get("memory_gib") or {}
        adv = st_.get("advanced_by")
        verdict = "converged" if st_.get("converged") else (("advanced (%s)" % adv) if adv else "NOT advanced")
        print("      %s %s from %s%s: %s kept %s confirming %s (%s evaluations); soft gain g %s (1 - g %s; |Ritz| max %s; in %s at site %s; Arnoldi residual %s, %s products); moved %s (%s site %s); "
              "largest trial displacement %s (%s site %s); N site 0 %s; N_dense(0) %s W/2g(1) %s W/2g(2) %s W/2g(ell) %s rho_ell(0) %s J/v(1) %s; memory %s GiB (peak %s)" % (
                  pn, f(st_.get("s"), "%g"), f(st_.get("s_from"), "%g"), " (closing)" if st_.get("closing") else "", verdict, f(st_.get("kept_residual"), "%.2e"),
                  f(st_.get("confirming_residual"), "%.2e"), st_.get("evaluations"), f(sg.get("gain"), "%.6f"), f(sg.get("one_minus_gain"), "%.3e"), f(sg.get("largest_modulus"), "%.4f"),
                  sg.get("field"), sg.get("site"), f(sg.get("residual"), "%.1e"), sg.get("products"), f(ch.get("max_abs"), "%.3e"), ch.get("field"), ch.get("site"),
                  f(dp.get("max_abs_displacement"), "%.3e"), dp.get("field"), dp.get("site"), f(ch.get("N_site0"), "%.7f"), f(cc.get("N_dense_0"), "%.7f"),
                  f(cc.get("W_over_2g_at_1"), "%.4e"), f(cc.get("W_over_2g_at_2"), "%.4e"), f(cc.get("W_over_2g_at_ell"), "%.4e"), f(cc.get("rho_ell_0"), "%.3e"),
                  f(cc.get("J_over_v_at_1"), "%.4e"), f(mm.get("allocated"), "%.1f"), f(mm.get("peak"), "%.1f")), flush=True)
    walked = [q for q in entries if q.get("advanced_by")]
    gains = "; ".join("%s %s: g %s, N_dense(0) %s, W/2g(1) %s, J/v(1) %s" % (
        pn, f(q.get("s"), "%g"), f((q.get("soft_gain") or {}).get("gain"), "%.6f"), f((q.get("central_cell") or {}).get("N_dense_0"), "%.6f"),
        f((q.get("central_cell") or {}).get("W_over_2g_at_1"), "%.3e"), f((q.get("central_cell") or {}).get("J_over_v_at_1"), "%.3e")) for q in walked)
    if v.get("stalled"):
        s_ = v["stalled"]
        print("    FOLD READOUT (%s): the walk stalled between %s = %s and %s = %s after %s bisection(s)%s; along the walked stages: %s. g rising to 1 at the stall with the cell turning is the fold; g below 1 there is the solver's soft direction" % (
            v.get("family"), pn, f(s_.get("s_from"), "%g"), pn, f(s_.get("s_to"), "%g"), s_.get("bisections"), " (stage cap reached)" if s_.get("cap_reached") else "", gains or "-"), flush=True)
    elif v.get("reached_1"):
        print("    the walk reached %s = 1; along the walked stages: %s. g above 1 at a reached %s = 1 says the floor rule let the walk leave the branch" % (pn, gains or "-", pn), flush=True)


def print_rows(name, rs, prod=None):
    for k, r in enumerate(rs):
        p = prod[k] if (prod is not None and k < len(prod)) else None
        print("  %s T/rs %s: converged %s meets stop rule %s; kept from %s residual %s confirming %s dE/E %s; E %s (E/E0 %s); solver residuals W %s 1/N %s; light E_inv/E %s; %s%s" % (
            name, f(r["T"], "%.4f"), r["converged"], r["meets"], r["kept_from"], f(r["kept"], "%.2e"), f(r["confirming"], "%.2e"), f(r["dE_c"], "%.1e"), f(r["E"], "%.9f"),
            f(r["E_over_E0"], "%.7f"), f(r["W"], "%.1e"), f(r["invN"], "%.1e"), f(r["Einv"]), r["continuation"],
            ("; production: E %s converged %s kept %s W %s 1/N %s" % (f(p["E"], "%.9f"), p["converged"], f(p["kept"] if p["kept"] is not None else (p["unconverged"] or {}).get("residual"), "%.2e"),
                                                                        f(p["W"], "%.1e"), f(p["invN"], "%.1e"))) if p else ""), flush=True)
        if r["failed"]:
            print("    FAILED: %s" % r["failed"], flush=True)
        if r["ends"]:
            print("    the member ENDS at this record (kept residual %s above the end rule)" % f(r["kept"], "%.2e"), flush=True)
        if r["rule"]:
            print("    rule: %s" % r["rule"], flush=True)
        for q in (r["solves"] or []):
            print("    stage: %s" % json.dumps(q, default=str)[:700], flush=True)
        if r["gain_direct"]:
            g_ = r["gain_direct"]
            print("    direct stage's soft gain at its kept iterate: g %s (1 - g %s; in %s at site %s; Arnoldi residual %s after %s products)" % (
                f(g_.get("gain"), "%.6f"), f(g_.get("one_minus_gain"), "%.3e"), g_.get("field"), g_.get("site"), f(g_.get("residual"), "%.1e"), g_.get("products")), flush=True)
        if r["v4"]:
            print("    v4: %s" % json.dumps(r["v4"], default=str)[:1400], flush=True)
        print_ramp(r["ramp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--records", type=int, default=3); ap.add_argument("--nsites", type=int, default=2048); ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--v3", default="engine_twofield_v3.py"); ap.add_argument("--v4", default="engine_twofield_v4.py"); ap.add_argument("--v5", default="engine_twofield_v5.py")
    ap.add_argument("--v6", default="engine_twofield_v6.py")
    ap.add_argument("--ramp", default="0.25,0.5,0.75,1", help="the homotopy's targets (--light-coupling-ramp)"); ap.add_argument("--bisect", type=int, default=2, help="--light-ramp-bisect")
    ap.add_argument("--gain", type=int, default=24, help="--light-ramp-gain-vectors (the soft gain's Arnoldi products per stage; 0 off)")
    ap.add_argument("--floor", type=float, default=2.0, help="--light-ramp-floor-factor"); ap.add_argument("--base-above", type=float, default=None, help="--light-ramp-base-above (default the engine's: --end-above)")
    ap.add_argument("--no-v36-opts", dest="v36_opts", action="store_false", help="run without --field-kernel-cache --v36-fixes")
    ap.add_argument("--prod-json", default=None, help="the production's member2 json (records' E and solver residuals printed beside; the chains differ unless --nsites 1024)")
    ap.add_argument("--only", default="g1,g2,g3"); ap.add_argument("--reuse", action="store_true", help="reuse engine outputs already in --out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v3 = os.path.join(a.engine_dir, a.v3)
    v4 = ensure_built(a, os.path.join(a.engine_dir, a.v4), v3)
    v5 = ensure_built(a, os.path.join(a.engine_dir, a.v5), v4)
    v6 = ensure_built(a, os.path.join(a.engine_dir, a.v6), v5)
    only = set(x.strip() for x in a.only.split(",") if x.strip())
    prod = rows(a.prod_json)[1] if a.prod_json else None
    ramp_args = ["--light-coupling-ramp", a.ramp, "--light-ramp-bisect", str(a.bisect), "--light-ramp-gain-vectors", str(a.gain), "--light-ramp-floor-factor", "%g" % a.floor]
    if a.base_above is not None:
        ramp_args += ["--light-ramp-base-above", "%g" % a.base_above]
    mem = {}
    print("gate v6 (sigma/a 2 start, the homotopy walk): engines %s and %s; %d sites, tau %g, rs %g, the first %d records (T/rs to %.4f); ramp %s bisect %d gain vectors %d floor factor %g base rule %s; v36 options %s; device %s" % (
        v4, v6, a.nsites, a.tau, a.rs, a.records, (a.records - 1) * a.tau / a.rs, a.ramp, a.bisect, a.gain, a.floor, ("%g" % a.base_above) if a.base_above is not None else "the end rule",
        "on" if a.v36_opts else "off", "gpu %d" % a.gpu if a.gpu is not None else "cpu"), flush=True)
    verdict = {}
    if "g1" in only:
        j4, rc4 = run_engine(a, v4, 4, "on", "g1_s4_v4_off", [], mem)
        j6, rc6 = run_engine(a, v6, 4, "on", "g1_s4_v6_off", [], mem)
        key = "G1 bitwise off (sigma/a 4, coupling on)"
        if j4 and j6 and rc4 == 0 and rc6 == 0:
            verdict[key] = bitwise(a, "g1_s4_v4_off", "g1_s4_v6_off", "G1")
        else:
            verdict[key] = False
            print("G1: FAIL (an engine run failed: v4 exit %d, v6 exit %d)" % (rc4, rc6), flush=True)
    if "g2" in only:
        j4, rc4 = run_engine(a, v4, 2, "off", "g2_s2_v4_off", [], mem)
        j6, rc6 = run_engine(a, v6, 2, "off", "g2_s2_v6_off_ramp", ramp_args, mem)
        key = "G2 the uncoupled limit (sigma/a 2, coupling off, the ramp, gain and floor options on)"
        if j4 and j6 and rc4 == 0 and rc6 == 0:
            verdict[key] = bitwise(a, "g2_s2_v4_off", "g2_s2_v6_off_ramp", "G2")
        else:
            verdict[key] = False
            print("G2: FAIL (an engine run failed: v4 exit %d, v6 exit %d)" % (rc4, rc6), flush=True)
        if j6:
            r6 = rows(j6)[1]
            print("G2 the uncoupled quench at %d sites: T = 0 %s (the production's practice: written unconverged below the end rule)" % (
                a.nsites, "CONVERGED" if (r6 and r6[0]["meets"]) else "NOT CONVERGED"), flush=True)
            print_rows("G2 v6 off", r6, prod)
    if "g3" in only:
        key = "G3 on (sigma/a 2, coupling on, the homotopy): 0.0625 and 0.125 meet the stop rule, T = 0 below the end rule, no end, exit 0"
        j, rc = run_engine(a, v6, 2, "on", "g3_s2_v6_on_ramp", ramp_args, mem)
        if j is None:
            verdict[key] = False
            print("G3: FAIL (the engine run failed before any record was written; exit %d)" % rc, flush=True)
        else:
            J, r = rows(j)
            ea = J["args"].get("end_above"); end_above = float(ea) if ea is not None else float("inf")
            ok_len = len(r) == a.records
            ok0 = bool(r) and not r[0]["failed"] and (r[0]["meets"] is True or (r[0]["kept"] is not None and r[0]["kept"] < end_above))
            ok_rest = all(q["meets"] is True and not q["failed"] for q in r[1:])
            ok_end = not any(q["ends"] for q in r)
            ok = ok_len and ok0 and ok_rest and ok_end and rc == 0
            verdict[key] = ok
            print("G3: %s (exit %d; %d of %d records written; T = 0 %s; later records meeting the stored stop rule %d of %d; member ended %s)" % (
                "PASS" if ok else "FAIL", rc, len(r), a.records,
                ("meets the stop rule" if (r and r[0]["meets"] is True) else ("unconverged below the end rule %g (kept %s)" % (end_above, f(r[0]["kept"], "%.2e")) if ok0 else
                                                                                ("ABOVE the end rule or failed (kept %s)" % f(r[0]["kept"] if r else None, "%.2e")))),
                sum(1 for q in r[1:] if q["meets"] is True), max(len(r) - 1, 0), any(q["ends"] for q in r)), flush=True)
            print_rows("G3 v6 on", r, prod)
            if prod and r:
                d = abs(r[0]["E"] - prod[0]["E"]) / 59.788988
                print("G3 E at T = 0 against the production's record 0: |dE/E0| %.2e (the floor 3.0e-7; informational, the chains differ unless --nsites 1024)" % d, flush=True)
            for q in r:
                v = q["ramp"] or {}
                pn = v.get("parameter") or "s"
                if v.get("stalled"):
                    s_ = v["stalled"]
                    print("G3 reading at T/rs %s: FOLD READOUT (%s homotopy; see above): the walk stalled between %s = %s and %s = %s; kept from %s" % (
                        f(q["T"], "%.4f"), v.get("family"), pn, f(s_.get("s_from"), "%g"), pn, f(s_.get("s_to"), "%g"), q["kept_from"]), flush=True)
                elif v:
                    print("G3 reading at T/rs %s: the %s homotopy %s (final stage %s); kept from %s" % (
                        f(q["T"], "%.4f"), v.get("family"), ("reached %s = 1" % pn) if v.get("reached_1") else ("did not reach %s = 1" % pn),
                        ("met the stop rule" if v.get("final_converged") else ("advanced by the %s" % v.get("final_advanced_by") if v.get("final_advanced_by") else "above the floor rule")), q["kept_from"]), flush=True)
                else:
                    print("G3 reading at T/rs %s: the direct solve %s; kept from %s" % (f(q["T"], "%.4f"), "met the stop rule" if q["meets"] else "missed the stop rule without a homotopy", q["kept_from"]), flush=True)
    print("peak device memory per run: %s" % ("; ".join("%s %s" % (k_, ("%.2f GiB" % v_) if v_ is not None else "not measured") for k_, v_ in mem.items()) or "no run"), flush=True)
    for k_, v_ in verdict.items():
        print("%s: %s" % (k_, "PASS" if v_ else "FAIL"), flush=True)
    allok = all(verdict.values()) if verdict else False
    print("GATE V6: %s" % ("PASS" if allok else "FAIL"), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
