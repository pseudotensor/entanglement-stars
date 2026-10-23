#!/usr/bin/env python
"""engine_collapse_v36e_selftest.py (reviewer A for the solve lane's builder; LADDERS_spec_v1 section 3): the self-test of engine_collapse_v36e and
the bitwise gate tool. Needs numpy only (no torch, no GPU): the ray's functions are taken out of the built text and run on a Painleve-Gullstrand
vacuum, where the closed forms are known.
Usage:
  python engine_collapse_v36e_selftest.py [--v36d PATH]                    the build in memory, the reverse mapping, the compile, the unit tests; prints SELFTEST PASS
  python engine_collapse_v36e_selftest.py --build-check V36D V36E          the built file equals a fresh build of V36D and reverses to it; prints build-check PASS
  python engine_collapse_v36e_selftest.py --gate DIR_D/TAG DIR_E/TAG       two runs (v36d, and v36e with its option off) compared: every json number of head and
                                                                           records except the wall-clock seconds and timing, the fields and spectra arrays bitwise,
                                                                           and equally named slices bitwise; prints GATE PASS or GATE FAIL (exit 0 or 1)"""
import glob
import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_engine_collapse_v36e as B  # noqa: E402


def extract(text):
    """the FR state line and the first_return_* functions from the built text, run in a numpy-only namespace."""
    i0 = text.index("FR = {"); i1 = text.index("\n", i0)
    j0 = text.index("def first_return_desc_v36e("); j1 = text.index("\ndef run(a):")
    ns = {"np": np, "math": math}
    exec(compile(text[i0:i1] + "\n" + text[j0:j1], "v36e_ray_functions", "exec"), ns)
    return ns


def pg_info(rs, nsites):
    return {"r_collar": float(nsites), "N_dense": (lambda r: np.ones_like(np.asarray(r, float))), "v_dense": (lambda r: -np.sqrt(rs / np.maximum(np.asarray(r, float), 1e-9)))}


def closed_forms(rs, nsites, coll, x_launch, T_launch):
    """the vacuum's outgoing clock r_s [u^2 + 2u + 2 ln(u - 1)] and ingoing clock r_s [u^2 - 2u + 2 ln(u + 1)] (LADDERS_spec_v1 section 5)."""
    F = lambda u: u * u + 2.0 * u + 2.0 * math.log(u - 1.0)
    G = lambda u: u * u - 2.0 * u + 2.0 * math.log(u + 1.0)
    xe = nsites / rs
    t_end = T_launch + rs * (F(math.sqrt(xe)) - F(math.sqrt(x_launch)))
    t_ret = t_end + rs * (G(math.sqrt(xe)) - G(math.sqrt(coll)))
    return t_end, t_ret


def selftest(v36d_path):
    text = open(v36d_path, "rb").read().decode("utf-8")
    built = B.build(text)
    assert B.reverse(built) == text, "the reverse replacements do not restore v36d"
    compile(built, "engine_collapse_v36e.py", "exec")
    print("build: %d replacements, each OLD once in v36d; the reverse mapping restores v36d byte for byte; the built text compiles" % len(B.REPLACEMENTS))
    ns = extract(built)
    FR = ns["FR"]
    rs, nsites, R0 = 326.0, 3912, 4.0
    a = SimpleNamespace(first_return_collar_over_rs=8.0, first_return_launch_over_rs=1.9, first_return_end_rule="end", T_over_rs=30.0, tau=8.0, rs=rs, ell=4, nsites=nsites)
    info = pg_info(rs, nsites); R = np.arange(1.0, nsites + 1.0)
    T_ff = 0.5 * math.pi * R0 ** 1.5 * rs
    # 1. no launch without a zero
    ns["first_return_init_v36e"](a, [])
    ent, ret = ns["first_return_record_v36e"](a, info, R, rs, 4000.0, 1300.0, [])
    assert not ent["launched"] and not ret and not FR["launched"], "launched without a zero of N - |v|"
    # 2. the launch at the formation record from the option's radius; the projections against the closed forms
    ent, ret = ns["first_return_record_v36e"](a, info, R, rs, T_ff, 250.0, [230.0])
    assert ent["launched"] and abs(ent["launch_r"] - 1.9 * rs) < 1e-9 and ent["phase"] == "out" and not ret
    t_end, t_ret = closed_forms(rs, nsites, 8.0, 1.9, T_ff)
    de = abs(ent["end_arrival_T_over_rs"] * rs - t_end) / t_end; dr = abs(ent["return_T_over_rs"] * rs - t_ret) / t_ret
    print("vacuum r_s 326, 3912 sites, launch 1.9 r_s at the free-fall clock %.3f r_s: projected end arrival %.4f r_s (closed form %.4f, relative %.1e); return into 8 r_s %.4f (closed form %.4f, relative %.1e)" % (
        T_ff / rs, ent["end_arrival_T_over_rs"], t_end / rs, de, ent["return_T_over_rs"], t_ret / rs, dr))
    assert de < 1e-3 and dr < 1e-3, "the projection departs from the closed forms"
    proj_ret = ent["return_T_over_rs"]; proj_end = ent["end_arrival_T_over_rs"]
    # 3. the record-by-record advance: the return at the first record at or past the projected return; the reflection's clock equal to the projection's
    T = T_ff; k = 0; ent_first = ent
    while True:
        T += a.tau; k += 1
        ent, ret = ns["first_return_record_v36e"](a, info, R, rs, T, None, [230.0])
        if ret:
            break
        assert k < 2000, "no return within 2000 records"
    assert proj_ret <= T / rs < proj_ret + a.tau / rs + 1e-9, "the return record is not the first at or past the projected return (%.4f against %.4f)" % (T / rs, proj_ret)
    assert ent["reflected_T"] is not None and abs(ent["reflected_T"] / rs - proj_end) < 1e-6, "the actual reflection's clock differs from the projection's"
    assert not ent["return_is_projected"] and abs(ent["return_T_over_rs"] - proj_ret) < 1e-6, "the actual return's clock differs from the projection's"
    print("record by record (tau 8): returned at the record T/rs %.4f after %d records; the reflection at %.4f and the return at %.4f agree with the projections" % (T / rs, k, ent["reflected_T"] / rs, ent["return_T_over_rs"]))
    # 4. the state restored from a stored entry
    ns["first_return_init_v36e"](a, [{"T": T_ff, "first_return_v36e": ent_first}])
    assert FR["launched"] and abs(FR["r"] - ent_first["r_next"]) < 1e-12 and FR["phase"] == ent_first["phase_next"], "the restored state differs from the stored entry"
    # 5. the default launch rule: R_sh, or the outer zero plus the averaging scale when R_sh lies inside it
    a.first_return_launch_over_rs = None
    ns["first_return_init_v36e"](a, []); ent, _ = ns["first_return_record_v36e"](a, info, R, rs, T_ff, 250.0, [230.0])
    assert abs(ent["launch_r"] - 250.0) < 1e-12 and ent["launch_rule"] == "the formation record's R_sh"
    ns["first_return_init_v36e"](a, []); ent, _ = ns["first_return_record_v36e"](a, info, R, rs, T_ff, 100.0, [230.0])
    assert abs(ent["launch_r"] - 234.0) < 1e-12 and ent["launch_rule"].startswith("the formation record's outer zero plus the averaging scale")
    # 6. a ray inside the trapped region is reported trapped and never returns within the cap
    ns["first_return_init_v36e"](a, []); ent, ret = ns["first_return_record_v36e"](a, info, R, rs, T_ff, 100.0, [100.0])
    assert ent["trapped"] and ent["return_T_over_rs"] is None and not ret, "a trapped launch is not reported trapped"
    print("the launch rule, the restored state and the trapped case behave as the docstring says")
    print("SELFTEST PASS")


def build_check(v36d, v36e):
    text = open(v36d, "rb").read().decode("utf-8"); built = open(v36e, "rb").read().decode("utf-8")
    ok = (B.build(text) == built) and (B.reverse(built) == text)
    print("build-check %s: %s against a fresh build of %s (md5 %s)" % ("PASS" if ok else "FAIL", v36e, v36d, hashlib.md5(built.encode("utf-8")).hexdigest()))
    return 0 if ok else 1


SKIP = {"seconds", "timing", "slice_saved", "seconds_preparation", "compute_s"}   # v1b (the builder): "compute_s", the field kernel cache's per-process wall-clock seconds (v36.kernel_cache), a timing like those above; the gate on R1 failed on it alone
ARGS_MAY_DIFFER = {"out", "tag", "slice_dir", "gpu", "first_return_collar_over_rs", "first_return_launch_over_rs", "first_return_end_rule"}


def walk(x, y, path, diffs):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k in SKIP:
                continue
            if k not in x or k not in y:
                diffs.append("%s.%s present on one side only" % (path, k)); continue
            walk(x[k], y[k], "%s.%s" % (path, k), diffs)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append("%s: lengths %d and %d" % (path, len(x), len(y))); return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, "%s[%d]" % (path, i), diffs)
    elif isinstance(x, float) or isinstance(y, float):
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            diffs.append("%s: %r against %r" % (path, x, y)); return
        if not (fx == fy or (math.isnan(fx) and math.isnan(fy))):
            diffs.append("%s: %r against %r" % (path, x, y))
    elif x != y:
        diffs.append("%s: %r against %r" % (path, x, y))


def gate(run_d, run_e):
    dd, td = os.path.split(run_d); de_, te = os.path.split(run_e)
    JD = json.load(open(os.path.join(dd, "collapse_%s.json" % td))); JE = json.load(open(os.path.join(de_, "collapse_%s.json" % te)))
    diffs = []
    walk(JD["head"], JE["head"], "head", diffs); walk(JD["records"], JE["records"], "records", diffs)
    for k in ("end_v32", "end_s11", "end_v34", "end_v36e"):
        if (k in JD) != (k in JE):
            diffs.append("%s present on one side only" % k)
    ad = {k: v for k, v in JD["args"].items() if k not in ARGS_MAY_DIFFER}; ae = {k: v for k, v in JE["args"].items() if k not in ARGS_MAY_DIFFER}
    walk(ad, ae, "args", diffs)
    extra = sorted(set(JE["args"]) - set(JD["args"]))
    print("args: the v36e run's extra keys %s (allowed: %s)" % (extra, sorted(ARGS_MAY_DIFFER)))
    for suf in ("fields", "spectra", "handoff"):
        pd_, pe_ = os.path.join(dd, "collapse_%s_%s.npz" % (td, suf)), os.path.join(de_, "collapse_%s_%s.npz" % (te, suf))
        if not (os.path.exists(pd_) and os.path.exists(pe_)):
            print("%s npz: %s" % (suf, "absent on both sides" if not (os.path.exists(pd_) or os.path.exists(pe_)) else "present on one side only")); 
            if os.path.exists(pd_) != os.path.exists(pe_):
                diffs.append("%s npz present on one side only" % suf)
            continue
        ZD = np.load(pd_, allow_pickle=True); ZE = np.load(pe_, allow_pickle=True)
        for k in sorted(set(ZD.files) | set(ZE.files)):
            if k in ("label", "solver"):
                continue
            if k not in ZD.files or k not in ZE.files:
                diffs.append("%s npz key %s on one side only" % (suf, k)); continue
            x, y = ZD[k], ZE[k]
            same = (x.dtype == object and len(x) == len(y) and all(np.array_equal(np.asarray(p, float), np.asarray(q, float)) for p, q in zip(x, y))) or (x.dtype != object and x.shape == y.shape and np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else (x.dtype != object and x.shape == y.shape and np.array_equal(x, y)))
            if not same:
                diffs.append("%s npz key %s differs" % (suf, k))
        print("%s npz: %d keys compared bitwise" % (suf, len(ZD.files)))
    sd = sorted(os.path.basename(f) for f in glob.glob(os.path.join(dd, "states", "orbitals_*.npz"))); se = sorted(os.path.basename(f) for f in glob.glob(os.path.join(de_, "states", "orbitals_*.npz")))
    for n in sorted(set(sd) & set(se)):
        A = np.load(os.path.join(dd, "states", n)); Bz = np.load(os.path.join(de_, "states", n))
        bad = [k for k in A.files if not (k in Bz.files and np.array_equal(A[k], Bz[k]))]
        if bad:
            diffs.append("slice %s differs in %s" % (n, bad))
        print("slice %s: %d keys compared bitwise%s" % (n, len(A.files), "" if not bad else " (differences)"))
    for d_ in diffs[:40]:
        print("DIFF " + d_)
    print("%d records compared; GATE %s" % (len(JD["records"]), "PASS" if not diffs else "FAIL (%d differences)" % len(diffs)))
    return 0 if not diffs else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--build-check"]:
        sys.exit(build_check(argv[1], argv[2]))
    if argv[:1] == ["--gate"]:
        sys.exit(gate(argv[1], argv[2]))
    path = argv[argv.index("--v36d") + 1] if "--v36d" in argv else os.path.join(HERE, "engine_collapse_v36d.py")
    selftest(path)
