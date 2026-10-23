#!/usr/bin/env python
"""Strand 3 batch driver for A.3 (sphere_mass.py, one process per mass, --workers 2) and A.4 (the paper's balls.py
measure, one process per run, 4 BLAS threads each), as subprocesses in parallel within 16 threads.  Builder analyst."""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run_all(cmds, env):
    procs = [(name, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)) for name, cmd, log in cmds]
    codes = {name: p.wait() for name, p in procs}
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--what", choices=["sphere", "balls"], required=True)
    a = ap.parse_args()
    M = json.load(open(a.masses))
    t0 = time.time()
    if a.what == "sphere":
        d = os.path.join(a.out, "sphere")
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        cmds = []
        for q in M["sphere_masses"]:
            tag = "%.12g" % q
            cmds.append((tag, ["nice", "-n", "10", PY, os.path.join(HERE, "sphere_mass.py"), "--mesh", "2", "--cache",
                               os.path.join(d, "cache_q" + tag), "--depth", "16", "--nk", "12", "--workers", "2",
                               "--mass", repr(q), "--output", os.path.join(d, "sphere_q%s.json" % tag)],
                         os.path.join(d, "sphere_q%s.log" % tag)))
    else:
        d = os.path.join(a.out, "balls")
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        cmds = []
        for q, c in M["ball_jobs"]:
            tag = "ball_q%.12g_c%s" % (q, "_".join("%g" % v for v in c))
            cmds.append((tag, ["nice", "-n", "10", PY, os.path.join(HERE, "orient", "balls.py"), "measure", "--radius", "8.5",
                               "--box", "64", "--method", "slq", "--probes", "32", "--steps", "96", "--seed", "781",
                               "--mass", repr(q), "--center"] + ["%g" % v for v in c] + ["--output", os.path.join(d, tag + ".json")],
                         os.path.join(d, tag + ".log")))
    codes = run_all(cmds, env)
    print("%s batch: %s in %.0f s" % (a.what, codes, time.time() - t0), flush=True)
    sys.exit(max(codes.values()) if codes else 0)


if __name__ == "__main__":
    main()
