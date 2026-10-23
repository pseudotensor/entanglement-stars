#!/usr/bin/env python
"""gate_w256_rd1d_v1.py (busi, inbox #464): the gate of W256_s4's RD-1d resume from T0001.50 on qi2_h100_1, ON THE CARD, read only until it acts.
Every PERIOD s it reads:
  (1) MEMORY: nvidia-smi memory.used of the card. Above --mem-cap-mib (default 76800 = 75 GiB) -> FAIL (memory).
  (2) BITWISE: once the new run's json holds a record past --t0 (1.5), every record with t0 < T <= min(T_new_max, T_ref_max) is compared
      with the stored reference run's record at the same clock, by cmp_rd13m_v1's own rule (diff/skip: every json key but the walls) and,
      for the per-record rows of the fields npz (every array of shape (n_records, ...)), np.array_equal row by row. All equal -> PASS
      (written once; the memory watch continues to the end); any difference -> FAIL (bitwise). RD-1d's check: the fix is bitwise against
      the parent's path (busi/notes/PERF_RD1d_build_v1.md, arms U1/U3 and C2), so the tolerance here is exact equality.
  On FAIL: GATE_FAIL.txt is written with the reason, then the engine python (pid from RUN/pid) and its children get TERM by PID (KILL after
  120 s). The launcher reads GATE_FAIL.txt after the engine exits and starts the cache-on fallback.
  gate_w256_rd1d_v1.py MEMBER RUN REF_RUN TAG ENGINE_PID [--t0 1.5] [--period 60] [--mem-cap-mib 76800]"""
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cmp_rd13m_v1 as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("member"); ap.add_argument("run"); ap.add_argument("ref"); ap.add_argument("tag"); ap.add_argument("pid", type=int)
ap.add_argument("--t0", type=float, default=1.5); ap.add_argument("--period", type=float, default=60.0)
ap.add_argument("--mem-cap-mib", dest="cap", type=float, default=76800.0)
a = ap.parse_args()
M = a.member


def say(s):
    line = "%s %s" % (dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), s)
    open(os.path.join(M, "GATE_RD1D.txt"), "a").write(line + "\n"); print(line, flush=True)


def alive(p):
    try:
        os.kill(p, 0); return True
    except OSError:
        return False


def fail(why):
    open(os.path.join(M, "GATE_FAIL.txt"), "w").write(why + "\n"); say("GATE FAIL: " + why)
    kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(a.pid)], capture_output=True, text=True).stdout.split()
    for p in [a.pid] + [int(k) for k in kids]:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(24):
        if not alive(a.pid):
            break
        time.sleep(5)
    if alive(a.pid):
        os.kill(a.pid, signal.SIGKILL); say("KILL sent")
    sys.exit(1)


def load(run):
    j = json.load(open(os.path.join(run, "collapse_%s.json" % a.tag)))
    z = np.load(os.path.join(run, "collapse_%s_fields.npz" % a.tag), allow_pickle=True)
    return j, z


def compare():
    jn, zn = load(a.run); jr, zr = load(a.ref)
    rn = {round(r["T_over_rs"], 9): r for r in jn["records"]}; rr = {round(r["T_over_rs"], 9): r for r in jr["records"]}
    tmax = min(max(rn), max(rr))
    clocks = sorted(t for t in rn if a.t0 + 1e-9 < t <= tmax + 1e-12)
    if not clocks:
        return None, []
    out = []
    for t in clocks:
        if t not in rr:
            out.append("T %.4f: not in the reference" % t); continue
        C.diff(rn[t], rr[t], "records[T=%.4f]" % t, out)
    Tn = np.round(zn["T"].astype(float), 9); Tr = np.round(zr["T"].astype(float), 9)
    nrows = 0
    for k in zn.files:
        x = zn[k]
        if k == "T" or k not in zr.files or x.ndim < 1 or x.shape[0] != len(Tn):
            continue
        y = zr[k]
        for t in [c for c in clocks if c in rr]:   # the fields npz keys its rows by the record's T (lattice units), the json clocks by T/r_s
            i = np.where(Tn == round(float(rn[t]["T"]), 9))[0]; j = np.where(Tr == round(float(rr[t]["T"]), 9))[0]
            if len(i) != 1 or len(j) != 1:
                out.append("fields[%s] T %.4f: row missing" % (k, t)); continue
            xr, yr = x[i[0]], y[j[0]]   # NaN equal for float rows only (cmp_rd13m_v1's rule)
            if not (np.array_equal(xr, yr, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(xr, yr)):
                out.append("fields[%s] T %.4f: row differs" % (k, t))
            nrows += 1
    return (clocks, nrows), out


say("watching %s (engine pid %d) against the reference %s from T > %.4f; memory cap %.0f MiB" % (a.run, a.pid, a.ref, a.t0, a.cap))
passed = False
while alive(a.pid):
    try:
        used = max(float(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                    capture_output=True, text=True).stdout.split())
    except Exception:
        used = 0.0
    if used > a.cap:
        fail("memory: nvidia-smi memory.used %.0f MiB > %.0f MiB" % (used, a.cap))
    if not passed and os.path.exists(os.path.join(a.run, "collapse_%s.json" % a.tag)):
        try:
            res, out = compare()
        except Exception as ex:   # the json is being rewritten: read again next period
            res, out = None, []; say("read deferred: %s" % str(ex)[:120])
        if res is not None:
            clocks, nrows = res
            if out:
                fail("bitwise: %d differences over T %s (first: %s)" % (len(out), ", ".join("%.4f" % t for t in clocks), out[0]))
            passed = True
            open(os.path.join(M, "GATE_PASS.txt"), "w").write("BITWISE on T %s (json records and %d fields rows)\n" % (
                ", ".join("%.4f" % t for t in clocks), nrows))
            say("GATE PASS: BITWISE on T %s (json records and %d fields rows); the memory watch continues" % (
                ", ".join("%.4f" % t for t in clocks), nrows))
    time.sleep(a.period)
say("engine %d ended; gate %s" % (a.pid, "PASS" if passed else "not reached"))
