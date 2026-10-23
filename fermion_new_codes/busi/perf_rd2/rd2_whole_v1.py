#!/usr/bin/env python
"""rd2_whole_v1.py (busi, analyst; PERF RD-2/RD-5 self-test (a), engine level): each rd2 engine at its defaults, in a code dir whose
engine_fields_v16.py is engine_fields_v16d's content (CODE16D), against its host in the H2 kit's code dir (engine_fields_v16.py = v16c,
CODE16C), on the selftest_srf_v1b 256-site Newton fixture (engine_collapse_v36z_selftest.py's BASE + NEWTON; v36r without --nk-host-release,
as h2default_whole_v1b), box CPU, nice 10, the two arms side by side at THREADS each. Compared by the self-test's bitwise() (json head,
records and end entries but walls; fields, spectra, handoff and the 0.5 slice byte for byte; the json's args differ by the two new keys
only) and the log body (walls and code dir masked; the first line, the command, skipped). Then, as smoke runs of the options (not part of
(a)): each rd2 engine with --fields collocation (tag <X>C) and v37a_rd2 also with --fields-warm-start on (ANW), outputs kept for the
comparator (cmp_rd2_replay_v1.py --fixture).
usage: python rd2_whole_v1.py --code16c DIR --code16d DIR --out DIR --selftest .../engine_collapse_v36z_selftest.py [--threads 4] [--only TAG,...]"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
for k in ("--code16c", "--code16d", "--out", "--selftest"):
    ap.add_argument(k, required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--only", default=None)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("st36z", a.selftest)
ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
os.makedirs(a.out, exist_ok=True)
NEWTON_N = [x for i, x in enumerate(ST.NEWTON) if x != "--nk-host-release" and not (i > 0 and ST.NEWTON[i - 1] == "--nk-host-release")]
ENG = [("NQN", "v36nq", ST.NEWTON), ("RN", "v36r", NEWTON_N), ("AN", "v37a", ST.NEWTON), ("NN", "v37n", ST.NEWTON)]
if a.only:
    keep = set(a.only.split(","))
    ENG = [c for c in ENG if c[0] in keep]


def run(label, code, tag, engfile, extra):
    d = os.path.join(a.out, label, tag)
    if os.path.exists(os.path.join(d, "collapse_%s.json" % tag)):
        raise SystemExit("%s exists (nothing overwritten)" % d)
    os.makedirs(d, exist_ok=True)
    cmd = ["nice", "-n", "10", sys.executable, "-B", os.path.join(code, engfile)] + ST.BASE + extra + [
        "--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
    env = dict(os.environ)
    for k in ("H2_PW_ONE", "H2_LOCAL_SPLINE", "H2_RESIDUAL_PAIR", "PYTHONPATH", "QI2_FIELDS_ROUTE", "QI2_FIELDS_WARM"):
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


def mask_log(s):
    s = re.sub(r"[0-9.]+ s\b|in [0-9.]+ s|\([0-9.]+ s;", "", s)
    return s.replace(a.code16d, "CODE").replace(a.code16c, "CODE")


fails = []
for tag, h, extra in ENG:
    with ThreadPoolExecutor(max_workers=2) as ex:
        fr = ex.submit(run, "host", a.code16c, tag, "engine_collapse_%s.py" % h, extra)
        fn = ex.submit(run, "rd2", a.code16d, tag, "engine_collapse_%s_rd2.py" % h, extra)
        R = {"host": fr.result(), "rd2": fn.result()}
    for lab, r in R.items():
        print("%s %s (%s): exit %d (%s)" % (tag, lab, h, r["rc"], r["dir"]), flush=True)
        if r["rc"] != 0:
            fails.append("%s %s exit %d" % (tag, lab, r["rc"]))
    if any(r["rc"] != 0 for r in R.values()):
        continue
    ST.bitwise(R["rd2"], R["host"], {"fields", "fields_warm_start"}, fails, "%s rd2 default (%s_rd2 on v16d) against the host (%s on v16c)" % (tag, h, h))
    b1 = mask_log(R["rd2"]["log"].split("\n", 1)[1]); b0 = mask_log(R["host"]["log"].split("\n", 1)[1])
    print("%s rd2 log body (walls and code dir masked) equals the host's: %s" % (tag, b1 == b0), flush=True)
    if b1 != b0:
        fails.append("%s log body differs" % tag)
print("WHOLE (a) %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)), flush=True)
for f in fails:
    print("  " + f)
# the smoke runs of the options (not part of (a))
smoke = [(t + "C", h, x + ["--fields", "collocation"]) for t, h, x in ENG] + [("ANW", "v37a", ST.NEWTON + ["--fields", "collocation", "--fields-warm-start", "on"])]
smoke = [s for s in smoke if any(s[1] == h for _, h, _ in ENG)]
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = {t: ex.submit(run, "rd2opt", a.code16d, t, "engine_collapse_%s_rd2.py" % h, x) for t, h, x in smoke}
    for t, f in futs.items():
        r = f.result()
        n_fb = r["log"].count("v16d RD2 fallback"); n_tol = r["log"].count("v16d RD2 ivp by tolerance"); n_rec = r["log"].count("  rd2 fields: route")
        print("smoke %s: exit %d; %d record lines 'rd2 fields', %d fallback lines, %d ivp-by-tolerance lines (%s)" % (t, r["rc"], n_rec, n_fb, n_tol, r["dir"]), flush=True)
