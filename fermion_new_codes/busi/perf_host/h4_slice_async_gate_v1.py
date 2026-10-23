#!/usr/bin/env python
"""h4_slice_async_gate_v1.py (busi, analyst; PERF H4(a), #348): the gates of engine_collapse_v36nq_aw.py (--slice-async) against its host
engine_collapse_v36nq.py on the box CPU, on the v36n self-test's fixture (engine_collapse_v36n_selftest.py: m 0.2, r_s 16, 512 sites,
kappa_sea 10, tau 2, T/r_s 0 to 0.25; its base_cmd and compare_runs), each run writing 2 complex128 slices (T/r_s 0 and 0.125).
Pairs (host H against the build A):
  D   --iter-cache drop                                    A with the option off: bitwise with H
  L   --iter-cache drop --prep-memory lean                 A off: bitwise with H
  UD  --mix damped --mix-alpha 3 --maxit 3 --stop-fields 1e-14 --iter-cache drop     A off: bitwise with H
  N   the pod6 line's options on the fixture (--dust-set isotropic --iter-cache drop --prep-memory lean, --mix newton and the NK flags,
      --nk-plain-fallback on): A off and A on (--slice-async on), each bitwise with H
"bitwise" is compare_runs (the records with the walls removed, the head, the fields, spectra and handoff npz, the T = 0 slice) plus, for
both slices, (i) every zip member's bytes (the .npy streams) md5-equal and (ii) the whole file's md5 with the zip entries' DOS date/time
fields zeroed (np.savez stamps each member with the wall clock, so two runs' raw md5s differ by those 4 bytes per entry only); the args
differ by slice_async only; no .partial_ file left; the v36aw log lines counted.
Part K (the kill): a child process imports the build, writes slice 1 through slice_submit_v36aw/slice_join_v36aw, starts slice 2 (a large
array) and is SIGKILLed while slice 2's .partial_ file is being written: slice 1 must load and equal its reference, slice 2's name must not
exist. Part E (the error): a submit into a missing directory raises RuntimeError at the join.
    python h4_slice_async_gate_v1.py --code DIR --out DIR [--threads 8] [--only D,L,UD,N,K,E]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np

SL = "0,0.125"
NEWTON = ["--dust-set", "isotropic", "--iter-cache", "drop", "--prep-memory", "lean", "--mix", "newton", "--nk-krylov", "40", "--nk-halvings", "6",
          "--nk-inner", "15", "--nk-jvp", "exact", "--nk-line-search", "halving", "--nk-set", "resolve", "--nk-maxiter", "30", "--nk-restarts", "2",
          "--nk-plain-fallback", "on"]
PAIRS = {"D": ["--iter-cache", "drop"], "L": ["--iter-cache", "drop", "--prep-memory", "lean"],
         "UD": ["--mix", "damped", "--mix-alpha", "3", "--maxit", "3", "--stop-fields", "1e-14", "--iter-cache", "drop"], "N": NEWTON}


def load_selftest(code):
    spec = importlib.util.spec_from_file_location("st36n", os.path.join(code, "engine_collapse_v36n_selftest.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def zip_norm_md5(path):
    """the file's md5 with every local and central zip entry's DOS time/date (4 bytes) zeroed."""
    b = bytearray(open(path, "rb").read())
    with zipfile.ZipFile(path) as z:
        infos = z.infolist(); start = z.start_dir
    for zi in infos:
        o = zi.header_offset
        assert b[o:o + 4] == b"PK\x03\x04"
        b[o + 10:o + 14] = b"\0\0\0\0"
    o = start
    for _ in infos:
        assert b[o:o + 4] == b"PK\x01\x02"
        b[o + 12:o + 16] = b"\0\0\0\0"
        n, e, c = struct.unpack("<HHH", bytes(b[o + 28:o + 34]))
        o += 46 + n + e + c
    return hashlib.md5(bytes(b)).hexdigest()


def member_md5(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for zi in z.infolist():
            h = hashlib.md5()
            with z.open(zi) as fh:
                for ch in iter(lambda: fh.read(1 << 24), b""):
                    h.update(ch)
            out[zi.filename] = h.hexdigest()
    return out


def slices_of(r):
    sd = os.path.join(os.path.dirname(r["json"]), "states")
    return sorted(os.path.join(sd, f) for f in os.listdir(sd)) if os.path.isdir(sd) else []


def run(st, a, eng, name, extra):
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
    tag = "fx_" + name; sdir = os.path.join(out, "states")
    sa = argparse.Namespace(no_kcache=False, gpu=None)
    cmd = st.base_cmd(sa, os.path.join(a.code, eng), out, tag, sdir) + extra
    cmd[cmd.index("--slices-c128") + 1] = SL
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(a.threads)
    log = os.path.join(out, "log.txt"); t0 = time.time()
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n"); fh.flush()
        rc = subprocess.call(cmd, cwd=a.code, stdout=fh, stderr=subprocess.STDOUT, env=env)
    r = {"name": name, "rc": rc, "seconds": time.time() - t0, "log": log, "json": os.path.join(out, "collapse_%s.json" % tag),
         "fields": os.path.join(out, "collapse_%s_fields.npz" % tag), "spectra": os.path.join(out, "collapse_%s_spectra.npz" % tag),
         "handoff": os.path.join(out, "collapse_%s_handoff.npz" % tag), "slice": os.path.join(sdir, "orbitals_%s_T0000.00.npz" % tag)}
    print("  %s: exit %d in %.0f s" % (name, rc, r["seconds"]), flush=True)
    return r


def compare(st, H, A, label, checks):
    d = st.compare_runs(H, A)
    checks.append((label + " compare_runs (records but the walls, head, fields, spectra, handoff, T = 0 slice)", not d, "; ".join(d) or "identical"))
    sh, sa = slices_of(H), slices_of(A)
    nh = [os.path.basename(p).replace("fx_" + H["name"], "X") for p in sh]; na = [os.path.basename(p).replace("fx_" + A["name"], "X") for p in sa]
    checks.append((label + " slice files: the same names, 2 each, no .partial_", nh == na and len(nh) == 2 and not any(".partial_" in p for p in na + nh), "%s | %s" % (nh, na)))
    for p, q in zip(sh, sa):
        mh, mq = member_md5(p), member_md5(q)
        zh, zq = zip_norm_md5(p), zip_norm_md5(q)
        raw = (hashlib.md5(open(p, "rb").read()).hexdigest(), hashlib.md5(open(q, "rb").read()).hexdigest())
        checks.append((label + " %s: every member's bytes md5-equal (%d members)" % (os.path.basename(q)[-14:], len(mh)), mh == mq, ""))
        checks.append((label + " %s: the file md5 with the zip date/time fields zeroed equal, the sizes equal" % os.path.basename(q)[-14:],
                       zh == zq and os.path.getsize(p) == os.path.getsize(q), "%s %s (%d B); raw md5 %s %s" % (zh[:8], zq[:8], os.path.getsize(q), raw[0][:8], raw[1][:8])))
    ja, jh = json.load(open(A["json"]))["args"], json.load(open(H["json"]))["args"]
    dk = sorted(k for k in set(ja) | set(jh) if ja.get(k) != jh.get(k) and k not in ("out", "tag", "slice_dir"))
    checks.append((label + " args differ by slice_async only", dk == ["slice_async"], str(dk)))
    lines = [l_.rstrip() for l_ in open(A["log"], errors="replace") if "v36aw slice" in l_]
    checks.append((label + " READ the v36aw log lines", None, " || ".join(lines)))


KCHILD = r'''
import sys, time, os
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("aw", os.path.join(sys.argv[1], "engine_collapse_v36nq_aw.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
import numpy as np
d = sys.argv[2]
a1 = {"T": 1.0, "sea_1": (np.arange(2 ** 22, dtype=np.float64) * (1 + 1j)).reshape(2048, 2048), "f_1": np.linspace(0, 1, 7)}
E.slice_submit_v36aw(os.path.join(d, "orbitals_k_T0001.00.npz"), a1, 1.0, 0.0); E.slice_join_v36aw()
print("SLICE1_DONE", flush=True)
a2 = {"T": 2.0, "sea_1": np.ones((int(sys.argv[3]), 2 ** 17), dtype=np.complex128)}
E.slice_join_v36aw(); E.slice_submit_v36aw(os.path.join(d, "orbitals_k_T0002.00.npz"), a2, 2.0, 0.0)
print("SLICE2_SUBMITTED", flush=True)
time.sleep(3600)
'''


def part_k(a, checks):
    d = os.path.join(a.out, "K"); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    rows = 1536                                                         # 1536 x 2^17 x 16 B = 3 GiB: seconds to write, long enough to catch
    p = subprocess.Popen([sys.executable, "-I", "-c", KCHILD, a.code, d, str(rows)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
    seen = []; killed_at = None; part = os.path.join(d, ".partial_orbitals_k_T0002.00.npz")
    for ln in p.stdout:
        seen.append(ln.rstrip())
        if ln.startswith("SLICE2_SUBMITTED"):
            break
    t0 = time.time()
    while time.time() - t0 < 120:
        if os.path.exists(part) and os.path.getsize(part) > (1 << 28):
            killed_at = os.path.getsize(part); p.send_signal(signal.SIGKILL); break
        if os.path.exists(os.path.join(d, "orbitals_k_T0002.00.npz")):
            break
        time.sleep(0.01)
    p.wait()
    f1 = os.path.join(d, "orbitals_k_T0001.00.npz"); f2 = os.path.join(d, "orbitals_k_T0002.00.npz")
    ok1 = False
    if os.path.exists(f1):
        Z = np.load(f1)
        ref = (np.arange(2 ** 22, dtype=np.float64) * (1 + 1j)).reshape(2048, 2048)
        ok1 = np.array_equal(Z["sea_1"], ref) and float(Z["T"]) == 1.0 and np.array_equal(Z["f_1"], np.linspace(0, 1, 7))
    checks.append(("K the child was SIGKILLed during slice 2's write (the .partial_ file at %s B of 3 GiB)" % killed_at, killed_at is not None, " | ".join(seen)))
    checks.append(("K slice 1 intact after the kill (loads, arrays equal to the reference)", ok1, f1))
    checks.append(("K slice 2's name does not exist (only the .partial_ file, overwritten by the next run's write of that slice)", not os.path.exists(f2), "partial exists: %s" % os.path.exists(part)))


def part_e(a, checks):
    sys.path.insert(0, a.code)
    spec = importlib.util.spec_from_file_location("aw", os.path.join(a.code, "engine_collapse_v36nq_aw.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    E.slice_submit_v36aw(os.path.join(a.out, "no_such_dir", "orbitals_e_T0001.00.npz"), {"T": 1.0, "sea_1": np.zeros(4)}, 1.0, 0.0)
    try:
        E.slice_join_v36aw(); ok, txt = False, "no error raised"
    except RuntimeError as e_:
        ok, txt = True, str(e_)
    checks.append(("E a failed background write raises RuntimeError at the join", ok, txt))
    small = {"T": 1.0, "sea_1": np.arange(6.0), "f_1": np.arange(3.0)}
    keep = small["f_1"]
    E.slice_submit_v36aw(os.path.join(a.out, "orbitals_e_T0002.00.npz"), small, 2.0, 0.0); keep[:] = -1.0; E.slice_join_v36aw()
    Z = np.load(os.path.join(a.out, "orbitals_e_T0002.00.npz"))
    checks.append(("E the small arrays are copied at submit (an in-place change after the submit does not reach the file)", np.array_equal(Z["f_1"], np.arange(3.0)), str(Z["f_1"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--only", default="D,L,UD,N,K,E")
    a = ap.parse_args(); a.code = os.path.abspath(a.code); os.makedirs(a.out, exist_ok=True)
    want = a.only.split(",")
    st = load_selftest(a.code)
    checks = []
    for nm in ("D", "L", "UD", "N"):
        if nm not in want:
            continue
        print("pair %s" % nm, flush=True)
        H = run(st, a, "engine_collapse_v36nq.py", "H" + nm, PAIRS[nm])
        A0 = run(st, a, "engine_collapse_v36nq_aw.py", "A" + nm + "off", PAIRS[nm])
        checks.append(("%s runs exit 0" % nm, H["rc"] == 0 and A0["rc"] == 0, ""))
        if H["rc"] == 0 and A0["rc"] == 0:
            compare(st, H, A0, "%s off:" % nm, checks)
        if nm == "N":
            A1 = run(st, a, "engine_collapse_v36nq_aw.py", "ANon", PAIRS[nm] + ["--slice-async", "on"])
            checks.append(("N on exit 0", A1["rc"] == 0, ""))
            if H["rc"] == 0 and A1["rc"] == 0:
                compare(st, H, A1, "N on:", checks)
    if "K" in want:
        part_k(a, checks)
    if "E" in want:
        part_e(a, checks)
    ok = all(c[1] is not False for c in checks)
    for n_, o_, t_ in checks:
        print("%-5s %s%s" % ("READ" if o_ is None else ("PASS" if o_ else "FAIL"), n_, (": " + t_) if t_ else ""), flush=True)
    print("h4_slice_async_gate_v1: %s (%d checks)" % ("PASS" if ok else "FAIL", len(checks)), flush=True)
    json.dump({"pass": ok, "checks": checks}, open(os.path.join(a.out, "gate_h4.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
