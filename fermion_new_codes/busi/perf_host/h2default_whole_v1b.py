#!/usr/bin/env python
"""h2default_whole_v1b.py (busi, analyst; inbox #343): h2default_whole_v1.py with the v36r engine run on NEWTON
without "--nk-host-release on" (v36r has no v36m offload option; v1 exited 2 on it; v36nq has it and passed in v1). Otherwise v1: every Newton engine lineage on the selftest_srf_v1b 256-site fixture
(engine_collapse_v36z_selftest.py's BASE + NEWTON, box CPU, nice 10), from a code dir whose engine_fields_v16.py is v16 (1ffcf5e8) and from
its copy whose engine_fields_v16.py is engine_fields_v16c's content (the kit route; options at their defaults = A and B on), compared by the
selftest's bitwise() (json head, records and end entries but walls; fields, spectra, handoff and the 0.5 slice byte for byte). For v36z also
PLAIN, the Newton fixture with H2_PW_ONE=0 H2_LOCAL_SPLINE=0 (off = v16), and the Newton fixture with --fields-v17 (v17 over v16c against
v17 over v16). Runs go in pairs (reference and v16c side by side, THREADS each).
usage: python h2default_whole_v1b.py --code16 DIR --code16c DIR --out DIR --selftest .../engine_collapse_v36z_selftest.py [--threads 4]
       [--only TAG,...]"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
for k in ("--code16", "--code16c", "--out", "--selftest"):
    ap.add_argument(k, required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--only", default=None)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("st36z", a.selftest)
ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
os.makedirs(a.out, exist_ok=True)
OFF = {"H2_PW_ONE": "0", "H2_LOCAL_SPLINE": "0"}
ENG = [("Q", "v36q_srf"), ("R", "v36r"), ("NQ", "v36nq"), ("Y", "v36y"), ("Z", "v36z"), ("A", "v37a"), ("N", "v37n")]
# (tag, engine, extra args, env for the v16c arm)
NEWTON_N = [x for i, x in enumerate(ST.NEWTON) if x != "--nk-host-release" and not (i > 0 and ST.NEWTON[i - 1] == "--nk-host-release")]
CASES = [(t + "N", e, NEWTON_N if e == "v36r" else ST.NEWTON, {}) for t, e in ENG]
CASES += [("ZP", "v36z", ST.PLAIN, {}), ("ZN17", "v36z", ST.NEWTON + ["--fields-v17"], {})]
if a.only:
    keep = set(a.only.split(","))
    CASES = [c for c in CASES if c[0] in keep]


def run(label, code, tag, eng, extra, opts):
    d = os.path.join(a.out, label, tag)
    if os.path.exists(os.path.join(d, "collapse_%s.json" % tag)):
        raise SystemExit("%s exists (nothing overwritten)" % d)
    os.makedirs(d, exist_ok=True)
    cmd = ["nice", "-n", "10", sys.executable, "-B", os.path.join(code, "engine_collapse_%s.py" % eng)] + ST.BASE + extra + [
        "--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
    env = dict(os.environ)
    for k in ("H2_PW_ONE", "H2_LOCAL_SPLINE", "H2_RESIDUAL_PAIR", "PYTHONPATH"):
        env.pop(k, None)
    env.update(opts)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    log = os.path.join(d, tag + ".log")
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "  # env %s\n" % opts); fh.flush()
        rc = subprocess.call(cmd, cwd=code, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag))) if rc == 0 else None
    return {"tag": tag, "rc": rc, "dir": d, "json": J, "log": open(log, errors="replace").read()}


strip_w = lambda s: re.sub(r"[0-9.]+ s\b|in [0-9.]+ s|\([0-9.]+ s;", "", s).replace(a.code16c, "CODE").replace(a.code16, "CODE")
fails = []
for tag, eng, extra, opts in CASES:
    arms = [("ref16", a.code16, {}), ("v16c", a.code16c, opts)]
    if tag == "ZN":
        arms.append(("v16c_off", a.code16c, OFF))
    with ThreadPoolExecutor(max_workers=2) as ex:
        R = {}
        for i in range(0, len(arms), 2):
            fs = {lab: ex.submit(run, lab, code, tag, eng, extra, o) for lab, code, o in arms[i:i + 2]}
            R.update({lab: f.result() for lab, f in fs.items()})
    for lab, r in R.items():
        print("%s %s (%s): exit %d (%s)" % (tag, lab, eng, r["rc"], r["dir"]), flush=True)
        if r["rc"] != 0:
            fails.append("%s %s exit %d" % (tag, lab, r["rc"]))
    if any(r["rc"] != 0 for r in R.values()):
        continue
    for lab in [l for l in R if l != "ref16"]:
        ST.bitwise(R[lab], R["ref16"], set(), fails, "%s %s (%s) against v16" % (tag, lab, eng))
        b1 = strip_w(R[lab]["log"].split("\n", 1)[1]); b0 = strip_w(R["ref16"]["log"].split("\n", 1)[1])
        print("%s %s log body (walls and code dir masked) equals v16's: %s" % (tag, lab, b1 == b0), flush=True)
print("WHOLE %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)))
for f in fails:
    print("  " + f)
