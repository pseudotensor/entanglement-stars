#!/usr/bin/env python
"""rd2b_whole_v1.py (busi, analyst; PERF RD-2b self-test (a), engine level): each rd2 engine (v36nq_rd2, v36r_rd2, v37a_rd2, v37n_rd2;
unchanged files) at its defaults in a code dir whose engine_fields_v16.py is engine_fields_v16e's content (CODE16E), against the same engine
at its defaults on v16d (RD-2's stored runs, RD2_build/whole/rd2/<TAG>, CODE16D), on the selftest_srf_v1b 256-site Newton fixture
(engine_collapse_v36z_selftest.py's BASE + NEWTON; v36r without --nk-host-release, as rd2_whole_v1), box CPU, nice 10, 2 runs side by side at
THREADS each (RD-2's setting). Compared by the self-test's bitwise() with no key exception (json head, records and end entries but walls;
fields, spectra, handoff and the 0.5 slice byte for byte; args equal) and the log body (walls and code dir masked; the first line skipped).
Then the option runs: each engine with --fields collocation on CODE16E (tag <X>C) against RD-2's stored v16d option runs
(RD2_build/whole/rd2opt/<X>C): the fields, spectra, handoff npz and the 0.5 slice byte for byte (the fixture has no negative centre, so v16e's
collocation is v16d's there), and the record lines' counters.
usage: python rd2b_whole_v1.py --code16e DIR --code16d DIR --ref DIR --out DIR --selftest .../engine_collapse_v36z_selftest.py [--threads 4]"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
for k in ("--code16e", "--code16d", "--ref", "--out", "--selftest"):
    ap.add_argument(k, required=True)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("st36z", a.selftest)
ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
os.makedirs(a.out, exist_ok=True)
NEWTON_N = [x for i, x in enumerate(ST.NEWTON) if x != "--nk-host-release" and not (i > 0 and ST.NEWTON[i - 1] == "--nk-host-release")]
ENG = [("NQN", "v36nq", ST.NEWTON), ("RN", "v36r", NEWTON_N), ("AN", "v37a", ST.NEWTON), ("NN", "v37n", ST.NEWTON)]


def run(label, code, tag, engfile, extra):
    d = os.path.join(a.out, label, tag)
    if os.path.exists(os.path.join(d, "collapse_%s.json" % tag)):
        raise SystemExit("%s exists (nothing overwritten)" % d)
    os.makedirs(d, exist_ok=True)
    cmd = ["nice", "-n", "10", sys.executable, "-B", os.path.join(code, engfile)] + ST.BASE + extra + [
        "--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
    env = dict(os.environ)
    for k in ("H2_PW_ONE", "H2_LOCAL_SPLINE", "H2_RESIDUAL_PAIR", "PYTHONPATH", "QI2_FIELDS_ROUTE", "QI2_FIELDS_WARM", "QI2_FIELDS_FLAT"):
        env.pop(k, None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    log = os.path.join(d, tag + ".log")
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n"); fh.flush()
        rc = subprocess.call(cmd, cwd=code, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag))) if rc == 0 else None
    return {"tag": tag, "rc": rc, "dir": d, "json": J, "log": open(log, errors="replace").read()}


def stored(label, tag):
    d = os.path.join(a.ref, label, tag)
    return {"tag": tag, "rc": 0, "dir": d, "json": json.load(open(os.path.join(d, "collapse_%s.json" % tag))), "log": open(os.path.join(d, tag + ".log"), errors="replace").read()}


def mask_log(s):
    s = re.sub(r"[0-9.]+ s\b|in [0-9.]+ s|\([0-9.]+ s;", "", s)
    return s.replace(a.code16d, "CODE").replace(a.code16e, "CODE")


fails = []
jobs = [(t, h, x, "rd2") for t, h, x in ENG] + [(t + "C", h, x + ["--fields", "collocation"], "rd2opt") for t, h, x in ENG]
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [(t, h, lab, ex.submit(run, lab, a.code16e, t, "engine_collapse_%s_rd2.py" % h, x)) for t, h, x, lab in jobs]
    res = {(t, lab): (h, f.result()) for t, h, lab, f in futs}
for (t, lab), (h, r) in res.items():
    print("%s %s (%s_rd2 on v16e): exit %d (%s)" % (t, lab, h, r["rc"], r["dir"]), flush=True)
    if r["rc"] != 0:
        fails.append("%s %s exit %d" % (t, lab, r["rc"]))
        continue
    ref = stored(lab, t)
    if lab == "rd2":
        ST.bitwise(r, ref, set(), fails, "%s default (%s_rd2 on v16e) against %s_rd2 on v16d (RD-2's stored run)" % (t, h, h))
        b1 = mask_log(r["log"].split("\n", 1)[1]); b0 = mask_log(ref["log"].split("\n", 1)[1])
        print("%s log body (walls and code dir masked) equals v16d's: %s" % (t, b1 == b0), flush=True)
        if b1 != b0:
            fails.append("%s log body differs" % t)
    else:
        diffs = []
        for f in ("fields", "spectra", "handoff"):
            b = ST.same_npz(os.path.join(r["dir"], "collapse_%s_%s.npz" % (t, f)), os.path.join(ref["dir"], "collapse_%s_%s.npz" % (t, f)))
            if b:
                diffs.append("%s %s" % (f, b))
        b = ST.same_npz(os.path.join(r["dir"], "states", "orbitals_%s_T0000.50.npz" % t), os.path.join(ref["dir"], "states", "orbitals_%s_T0000.50.npz" % t))
        if b:
            diffs.append("slice 0.5 %s" % b)
        cnt = lambda s: (s.count("v16d RD2 fallback"), s.count("v16d RD2 ivp by tolerance"), s.count("  rd2 fields: route"))
        c1, c0 = cnt(r["log"]), cnt(ref["log"])
        nc = [((q.get("fields_rd2") or {}).get("since_previous_record") or {}).get("negative_centre") for q in r["json"]["records"]]
        print("%s --fields collocation on v16e against v16d's stored run: %s; (fallback, ivp-by-tolerance, record) lines %s vs %s; negative_centre per record %s"
              % (t, "; ".join(diffs) or "fields, spectra, handoff npz and the 0.5 slice byte for byte", c1, c0, nc), flush=True)
        if diffs:
            fails.append("%s option run: %s" % (t, "; ".join(diffs)))
print("WHOLE (a) %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)), flush=True)
for f in fails:
    print("  " + f)
