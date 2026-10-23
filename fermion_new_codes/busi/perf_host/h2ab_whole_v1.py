#!/usr/bin/env python
"""h2ab_whole_v1.py (busi, analyst; PERF H2-A/H2-B item 3): the whole Newton engine v36z (924eefed) on the selftest_srf_v1b 256-site
fixture (engine_collapse_v36z_selftest.py's BASE, NEWTON and PLAIN, 8 threads, box CPU), with engine_fields_v16b swapped in as the code dir's
engine_fields_v16.py (the deploy route: the engine imports the module by name), options off and A+B, against fresh v16 references in the
same environment and against the tourn331 selftest's whole_b QN-era references ZN, ZP; the comparison is the selftest's bitwise() (json
head, records and end entries but walls; fields, spectra, handoff and the 0.5 slice byte for byte).
usage: python h2ab_whole_v1.py --code16 DIR --code16b DIR --out DIR --selftest .../engine_collapse_v36z_selftest.py --ref-dir .../whole_b"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
for k in ("--code16", "--code16b", "--out", "--selftest", "--ref-dir"):
    ap.add_argument(k, required=True)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("st36z", a.selftest)
ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
os.makedirs(a.out, exist_ok=True)


def run(label, code, tag, extra, opts):
    d = os.path.join(a.out, label, tag)
    if os.path.exists(os.path.join(d, "collapse_%s.json" % tag)):
        raise SystemExit("%s exists (nothing overwritten)" % d)
    os.makedirs(d, exist_ok=True)
    eng = os.path.join(code, "engine_collapse_v36z.py")
    cmd = ["nice", "-n", "10", sys.executable, eng] + ST.BASE + extra + ["--out", d, "--tag", tag, "--slice-dir", os.path.join(d, "states")]
    env = dict(os.environ)
    for k in ("H2_PW_ONE", "H2_LOCAL_SPLINE", "H2_RESIDUAL_PAIR"):
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


AB = {"H2_PW_ONE": "1", "H2_LOCAL_SPLINE": "1"}
jobs = [("ref16", a.code16, "ZN", ST.NEWTON, {}), ("ref16", a.code16, "ZP", ST.PLAIN, {}),
        ("v16b_off", a.code16b, "ZN", ST.NEWTON, {}), ("v16b_off", a.code16b, "ZP", ST.PLAIN, {}),
        ("v16b_AB", a.code16b, "ZN", ST.NEWTON, AB), ("v16b_AB", a.code16b, "ZP", ST.PLAIN, AB)]
with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
    futs = {(j[0], j[2]): ex.submit(run, *j) for j in jobs}
R = {k: f.result() for k, f in futs.items()}
fails = []
for k, r in sorted(R.items()):
    print("%s %s: exit %d (%s)" % (k[0], k[1], r["rc"], r["dir"]))
    if r["rc"] != 0:
        fails.append("%s %s exit %d" % (k[0], k[1], r["rc"]))
if not fails:
    old = {t: {"tag": t, "dir": os.path.join(a.ref_dir, t), "json": json.load(open(os.path.join(a.ref_dir, t, "collapse_%s.json" % t)))} for t in ("ZN", "ZP")}
    for t in ("ZN", "ZP"):
        ST.bitwise(R[("ref16", t)], old[t], set(), fails, "fresh v16 %s against tourn331 whole_b %s (environment check)" % (t, t))
        ST.bitwise(R[("v16b_off", t)], R[("ref16", t)], set(), fails, "v16b off %s against v16 %s" % (t, t))
        ST.bitwise(R[("v16b_AB", t)], R[("ref16", t)], set(), fails, "v16b A+B %s against v16 %s" % (t, t))
        for lab in ("v16b_off", "v16b_AB"):
            b1 = R[(lab, t)]["log"].split("\n", 1)[1]; b0 = R[("ref16", t)]["log"].split("\n", 1)[1]
            import re
            strip_w = lambda s: re.sub(r"[0-9.]+ s\b|in [0-9.]+ s|\([0-9.]+ s;", "", s)
            print("%s %s log body (walls masked) equals v16's: %s" % (lab, t, strip_w(b1) == strip_w(b0)))
print("WHOLE %s (%d failures)" % ("PASS" if not fails else "FAIL", len(fails)))
for f in fails:
    print("  " + f)
